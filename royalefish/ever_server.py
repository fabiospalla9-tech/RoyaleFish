"""Server di gioco ultra-rapido per un modello Everlight (V4) con Pinned Memory, unpack sparse e CUDA Graph fp16.
Accetta sia righe sparse (ib, vb) sia righe dense bytes per retrocompatibilita'.
Restituisce tuple leggere (act10_bytes, logp_float, val_float, version_int)."""
import argparse
import pickle
import queue
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
from royalefish.net import send_msg, recv_exact  # noqa: E402
from royalefish.teacher_pack import views, unpack_sparse_into  # noqa: E402
from native_runner.training.v4.policy_session import load_policy_v4  # noqa: E402


def recv_msg(sock):
    (n,) = struct.unpack('>I', recv_exact(sock, 4))
    return pickle.loads(recv_exact(sock, n))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cap', type=int, default=256)
    p.add_argument('--wait-ms', type=float, default=2.5)
    p.add_argument('--pool', type=int, default=4096)
    p.add_argument('--weights-dir', default=None)
    p.add_argument('--name', default='modello')
    p.add_argument('--half', action='store_true')
    a = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device(a.device)
    CAP = a.cap
    CAPS = [c for c in (16, 32, 64, 128, 256, 512) if c < CAP] + [CAP]

    loaded = load_policy_v4(a.checkpoint, device=dev)
    model = loaded.model.eval()
    if hasattr(model, 'lstm') and hasattr(model.lstm, 'flatten_parameters'):
        model.lstm.flatten_parameters()

    version = [0, 0.]
    states = {}
    layouts = {}
    pinned_in = {}
    G = {}

    st0 = model.initial_state(1, device=dev)
    POOL = a.pool
    pool_h = torch.zeros((POOL, *st0.hidden.shape[1:]), dtype=st0.hidden.dtype, device=dev)
    pool_c = torch.zeros((POOL, *st0.cell.shape[1:]), dtype=st0.cell.dtype, device=dev)
    free = list(range(POOL))
    last_use = {}

    pin_idx = torch.empty(CAP, dtype=torch.long, pin_memory=True)
    pin_sm = torch.empty(CAP, dtype=torch.bool, pin_memory=True)
    pin_act_out = torch.empty((CAP, 10), dtype=torch.int64, pin_memory=True)
    pin_lv_out = torch.empty((CAP, 2), dtype=torch.float32, pin_memory=True)
    dev_idx = torch.empty(CAP, dtype=torch.long, device=dev)

    def row_for(key):
        r = states.get(key)
        if r is not None:
            return r, False
        if not free:
            for k in sorted(last_use, key=last_use.get)[:max(256, POOL // 8)]:
                free.append(states.pop(k))
                last_use.pop(k, None)
        r = free.pop()
        states[key] = r
        return r, True

    def release(key):
        r = states.pop(key, None)
        last_use.pop(key, None)
        if r is not None:
            free.append(r)

    jobs = queue.Queue()
    stats = dict(rows=0, groups=0, prep=0., fwd=0.)

    def maybe_reload():
        if not a.weights_dir:
            return
        f = Path(a.weights_dir) / 'latest.pt'
        try:
            mt = f.stat().st_mtime
        except OSError:
            return
        if mt <= version[1]:
            return
        try:
            payload = torch.load(f, map_location=dev, weights_only=False)
            sd = payload['model'] if isinstance(payload, dict) and 'model' in payload else payload
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if k in sd:
                        v.copy_(sd[k])
            if hasattr(model, 'lstm') and hasattr(model.lstm, 'flatten_parameters'):
                model.lstm.flatten_parameters()
            version[0] = int(payload.get('version', version[0] + 1)) if isinstance(payload, dict) else version[0] + 1
            version[1] = mt
            print(time.strftime('%H:%M:%S'), a.name, 'pesi aggiornati alla versione', version[0], flush=True)
        except Exception as e:
            print('ricarica pesi fallita', repr(e)[:150], flush=True)

    def handle(conn):
        lock = threading.Lock()
        try:
            while True:
                msg = recv_msg(conn)
                if msg.get('layout'):
                    sig, tmpl, items, total = msg['layout']
                    if sig not in layouts:
                        layouts[sig] = (tmpl, items, total)
                    send_msg(conn, 'ok', lock)
                    continue
                done = threading.Event()
                box = {}
                jobs.put((msg, done, box))
                done.wait()
                send_msg(conn, box.get('out'), lock)
        except (ConnectionError, EOFError, OSError):
            pass
        finally:
            conn.close()

    def setup(sig, cap):
        template, items, total = layouts[sig]
        if sig not in pinned_in:
            pb = torch.empty((CAP, total), dtype=torch.uint8, pin_memory=True)
            pinned_in[sig] = (pb, pb.numpy())
        buf = torch.zeros((cap, total), dtype=torch.uint8, device=dev)
        raw = views(template, items, buf)
        st = model.initial_state(cap, device=dev)
        ep = torch.ones(cap, dtype=torch.bool, device=dev)
        ws = torch.cuda.Stream(device=dev)
        ws.wait_stream(torch.cuda.current_stream(dev))
        amp = torch.autocast('cuda', dtype=torch.float16, enabled=a.half)
        with torch.cuda.stream(ws), torch.no_grad(), amp:
            for _ in range(3):
                o_w = model.sample_for_ppo_rollout(raw.to_model_input(dev), st, episode_start=ep, validate=False)
                _ = torch.cat([
                    o_w.actions.gate[:, None], o_w.actions.micro_action_count[:, None],
                    o_w.actions.candidate_index, o_w.actions.candidate_uid,
                    o_w.actions.target_cell, o_w.actions.delay_offset_bin,
                ], dim=1)
                _ = torch.stack([o_w.log_prob.float(), o_w.value.float()], dim=1)
        torch.cuda.current_stream(dev).wait_stream(ws)
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=a.half):
            out = model.sample_for_ppo_rollout(raw.to_model_input(dev), st, episode_start=ep, validate=False)
            act_cat = torch.cat([
                out.actions.gate[:, None], out.actions.micro_action_count[:, None],
                out.actions.candidate_index, out.actions.candidate_uid,
                out.actions.target_cell, out.actions.delay_offset_bin,
            ], dim=1)
            lv_cat = torch.stack([out.log_prob.float(), out.value.float()], dim=1)
        G[(sig, cap)] = dict(buf=buf, st=st, ep=ep, graph=graph, out=out, act_cat=act_cat, lv_cat=lv_cat, total=total)
        print(a.name, 'registrazione CUDA pronta', sig, 'righe', cap, flush=True)

    def worker():
        last_log = time.monotonic()
        last_reload = 0.
        pin_idx_np = pin_idx.numpy()
        pin_sm_np = pin_sm.numpy()
        pin_act_np = pin_act_out.numpy()
        pin_lv_np = pin_lv_out.numpy()

        while True:
            first = jobs.get()
            batch = [first]
            queued = len(first[0]['items'])
            deadline = time.monotonic() + a.wait_ms / 1000.
            while queued < CAP:
                rem = deadline - time.monotonic()
                if rem <= 0.:
                    break
                try:
                    nxt_job = jobs.get(timeout=rem)
                    batch.append(nxt_job)
                    queued += len(nxt_job[0]['items'])
                except queue.Empty:
                    break

            now_m = time.monotonic()
            if now_m - last_reload > 4.:
                last_reload = now_m
                maybe_reload()

            for job in batch:
                for key in job[0].get('end', ()):
                    release(key)

            reqs = [(job, item) for job in batch for item in job[0]['items']]
            results = [None] * len(reqs)
            try:
                by_sig = {}
                for i, (_, it) in enumerate(reqs):
                    by_sig.setdefault(it[2], []).append(i)
                for sig, idxs in by_sig.items():
                    if sig not in layouts:
                        continue
                    for c0 in range(0, len(idxs), CAP):
                        part = idxs[c0:c0 + CAP]
                        n = len(part)
                        t0 = time.monotonic()
                        cap = next(c for c in CAPS if c >= n)
                        if (sig, cap) not in G:
                            setup(sig, cap)
                        g = G[(sig, cap)]
                        pb_tensor, pb_np = pinned_in[sig]

                        first_payload = reqs[part[0]][1][3]
                        if isinstance(first_payload, tuple):
                            unpack_sparse_into([reqs[i][1][3] for i in part], pb_np)
                        else:
                            pb_np[:n] = np.frombuffer(b''.join(reqs[i][1][3] for i in part), np.uint8).reshape(n, g['total'])

                        g['buf'][:n].copy_(pb_tensor[:n], non_blocking=True)

                        now = time.monotonic()
                        for j, i in enumerate(part):
                            key, start = reqs[i][1][0], reqs[i][1][1]
                            r, new = row_for(key)
                            pin_idx_np[j] = r
                            pin_sm_np[j] = bool(start or new)
                            last_use[key] = now

                        idx_gpu = dev_idx[:n]
                        idx_gpu.copy_(pin_idx[:n], non_blocking=True)
                        g['ep'][:n].copy_(pin_sm[:n], non_blocking=True)
                        sm_gpu = g['ep'][:n]
                        g['st'].hidden[:n].copy_(torch.where(sm_gpu[:, None], 0, pool_h.index_select(0, idx_gpu)))
                        g['st'].cell[:n].copy_(torch.where(sm_gpu[:, None], 0, pool_c.index_select(0, idx_gpu)))

                        t1 = time.monotonic()
                        stats['prep'] += t1 - t0

                        g['graph'].replay()
                        out = g['out']
                        pool_h.index_copy_(0, idx_gpu, out.next_state.hidden[:n])
                        pool_c.index_copy_(0, idx_gpu, out.next_state.cell[:n])
                        pin_act_out[:n].copy_(g['act_cat'][:n], non_blocking=True)
                        pin_lv_out[:n].copy_(g['lv_cat'][:n], non_blocking=True)
                        torch.cuda.current_stream(dev).synchronize()

                        stats['fwd'] += time.monotonic() - t1

                        ver = version[0]
                        for j, i in enumerate(part):
                            results[i] = (pin_act_np[j].tobytes(), float(pin_lv_np[j, 0]), float(pin_lv_np[j, 1]), ver)
                        stats['rows'] += n
                        stats['groups'] += 1
            except Exception:
                import traceback
                traceback.print_exc()

            k = 0
            for job in batch:
                m = len(job[0]['items'])
                job[2]['out'] = results[k:k + m]
                k += m
                job[1].set()

            if time.monotonic() - last_log > 30.:
                dt_log = time.monotonic() - last_log
                last_log = time.monotonic()
                gr = max(stats['groups'], 1)
                print(time.strftime('%H:%M:%S'), a.name,
                      'mosse/s', round(stats['rows'] / dt_log),
                      'media_lotto', round(stats['rows'] / gr, 1),
                      'gruppi', stats['groups'],
                      '| ms/gruppo prep', round(1000 * stats['prep'] / gr, 2),
                      'modello', round(1000 * stats['fwd'] / gr, 2),
                      '| partite_mem', len(states),
                      '| ver', version[0], flush=True)
                stats.update(rows=0, groups=0, prep=0., fwd=0.)

    maybe_reload()
    threading.Thread(target=worker, daemon=True).start()
    srv = socket.create_server((a.host, a.port), backlog=128)
    print(a.name, 'pronto', loaded.checkpoint_id, 'porta', a.port, flush=True)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == '__main__':
    main()
