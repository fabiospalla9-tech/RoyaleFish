"""Allenamento PPO di Everlight 2.6 (Modello A) sulla Tesla V100 in fp16 Tensor Core:
1. Fase 0 automatica (Critic Warmup): per i primi --warmup-updates aggiornamenti congela l'attore/LSTM
   e calibra solo i 14 tensori della Value Head contro la lega avversaria (zero deriva iniziale dei pesi!).
2. Fase 1+ (PPO Chirurgico): sblocca tutti i pesi con learning rate basso, ancora KL verso Everlight-RL-page134,
   lotti grandi da 24 partite complete (~22.000 passi reali per update).
3. Aggiorna continuamente runs/ever/pfsp_decks.json leggendo i risultati reali dei raccoglitori,
   cosi' A affronta fino a 4x piu' spesso i mazzi counter dove il winrate e' sotto il 50%."""
from collections import deque
import argparse
import fcntl
import gc
import json
import threading
import queue
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
from native_runner.training.v4.policy_session import load_policy_v4  # noqa: E402
from native_runner.training.v4.ppo import StoredRolloutV4, PPOConfigV4  # noqa: E402
from native_runner.training.v4.distributed_ppo import distributed_ppo_update_v4  # noqa: E402
from native_runner.training.v4.ppo_anchor import value_only_parameter_names  # noqa: E402
from native_runner.training.v4.tensors import ActionSequenceV4  # noqa: E402
from royalefish.teacher_pack import views, unpack_episode_sparse  # noqa: E402
from royalefish.ever_packed import build_row_packed  # noqa: E402


class _HoldOptimizer:
    """Tiene i gradienti: distributed_ppo non azzera e non fa il passo; lo facciamo noi dopo --accum gruppi."""
    def __init__(self, opt): self.opt = opt
    def zero_grad(self, set_to_none=True): pass
    def step(self, *a, **k): pass
    def __getattr__(self, k): return getattr(self.opt, k)


class PFSPTracker:
    """Legge in modo incrementale i risultati da collectors/e*.jsonl e aggiorna i pesi PFSP dei mazzi."""
    def __init__(self, run_dir: Path, window_per_deck: int = 80):
        self.run_dir = run_dir
        self.offsets = {}
        self.by_deck = {}
        self.window = window_per_deck
        self.recent_games = deque(maxlen=2000)

    def update(self):
        cdir = self.run_dir / 'collectors'
        if not cdir.exists():
            return
        now = time.time()
        for p in cdir.glob('e*.jsonl'):
            off = self.offsets.get(p.name, 0)
            try:
                sz = p.stat().st_size
                if sz <= off:
                    continue
                with open(p, 'rb') as f:
                    f.seek(off)
                    chunk = f.read()
                    self.offsets[p.name] = f.tell()
            except OSError:
                continue
            for raw_line in chunk.splitlines():
                if not raw_line:
                    continue
                try:
                    d = json.loads(raw_line)
                except Exception:
                    continue
                if d.get('event') != 'game':
                    continue
                oa = 0 if d['models'].get('0') == 'A' else 1
                res = float(d['result'][oa])
                score = 1.0 if res > 0 else (0.5 if res == 0 else 0.0)
                self.recent_games.append((float(d.get('utc', now)), score, bool(d.get('mirror', False)), d['deck_b']))
                if not d.get('mirror', False):
                    dk = d['deck_b']
                    dq = self.by_deck.get(dk)
                    if dq is None:
                        dq = deque(maxlen=self.window)
                        self.by_deck[dk] = dq
                    dq.append(score)

        # Calcolo pesi PFSP: (1 - winrate_smussato)^2, limitato in [0.25, 4.0]
        weights = {}
        deck_wr = {}
        for dk, dq in self.by_deck.items():
            n = len(dq)
            s = sum(dq)
            wr = (s + 2.0) / (n + 4.0)  # Laplace smoothing centrato a 0.50
            w = max(0.25, min(4.0, ((1.0 - wr) / 0.5) ** 2.0))
            weights[dk] = round(w, 4)
            deck_wr[dk] = (n, round(wr, 4))

        if weights:
            tmp = self.run_dir / 'pfsp_decks.json.tmp'
            tmp.write_text(json.dumps({'utc': now, 'weights': weights, 'stats': deck_wr}))
            os.replace(tmp, self.run_dir / 'pfsp_decks.json')

        # Calcolo ritmo partite/ora reale sugli ultimi 10 minuti
        recent_10m = [g for g in self.recent_games if now - g[0] <= 600.]
        rate_h = 0.0
        if len(recent_10m) >= 20:
            span = max(30.0, now - recent_10m[0][0])
            rate_h = 3600.0 * len(recent_10m) / span
        wr_all = float(np.mean([g[1] for g in self.recent_games])) if self.recent_games else 0.5
        worst = sorted(((wr, n, dk) for dk, (n, wr) in deck_wr.items() if n >= 5))[:8]
        summary_tmp = self.run_dir / 'summary.json.tmp'
        summary_tmp.write_text(json.dumps({
            'utc': now,
            'games_per_hour_10m': round(rate_h, 0),
            'recent_winrate_2000': round(wr_all, 4),
            'tracked_decks': len(deck_wr),
            'hardest_decks': [{'deck': dk[:12], 'n': n, 'winrate': wr} for wr, n, dk in worst],
        }, indent=2))
        os.replace(summary_tmp, self.run_dir / 'summary.json')


def configure_trainable(model, warmup: bool, lr: float, warmup_lr: float):
    vnames = set(value_only_parameter_names(model))
    for name, param in model.named_parameters():
        param.requires_grad_(name in vnames if warmup else True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=(warmup_lr if warmup else lr)), ('warmup' if warmup else 'ppo')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--lanes', type=int, default=24)
    p.add_argument('--minibatch', type=int, default=8)
    p.add_argument('--micro', type=int, default=4)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1.5e-5)
    p.add_argument('--warmup-lr', type=float, default=5e-5)
    p.add_argument('--warmup-updates', type=int, default=8)
    p.add_argument('--anchor', type=float, default=0.04)
    p.add_argument('--entropy', type=float, default=0.0025)
    p.add_argument('--chunk', type=int, default=128)
    p.add_argument('--max-stale', type=int, default=4)
    p.add_argument('--lock', default='/run/ever_learner.lock')
    p.add_argument('--amp', default='fp16', choices=['none', 'fp16', 'bf16'])
    p.add_argument('--gpu-budget', type=float, default=0.0)
    p.add_argument('--history-every', type=int, default=25)
    p.add_argument('--accum', type=int, default=3)   # gruppi accumulati in UN passo (come l'originale: ~60k mosse per aggiornamento)
    a = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run = a.run
    inbox = run / 'inbox'
    wdir = run / 'weights'
    hdir = wdir / 'history'
    inbox.mkdir(parents=True, exist_ok=True)
    hdir.mkdir(parents=True, exist_ok=True)

    dev = torch.device('cuda:0')
    model = load_policy_v4(a.checkpoint, device=dev).model
    ref = load_policy_v4(a.checkpoint, device=dev).model.eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    if hasattr(model, 'lstm') and hasattr(model.lstm, 'flatten_parameters'):
        model.lstm.flatten_parameters()
        ref.lstm.flatten_parameters()

    version = 0
    state_f = run / 'learner-state.pt'
    resumed_opt_state = None
    if state_f.exists():
        st = torch.load(state_f, map_location=dev, weights_only=False)
        model.load_state_dict(st['model'])
        version = int(st['version'])
        resumed_opt_state = st.get('opt')
        print('ripreso dalla versione', version, flush=True)

    in_warmup = version < a.warmup_updates
    opt, phase = configure_trainable(model, in_warmup, a.lr, a.warmup_lr)
    if resumed_opt_state is not None and st.get('phase') == phase:
        try:
            opt.load_state_dict(resumed_opt_state)
        except Exception:
            pass

    cfg = PPOConfigV4(
        gamma=0.9993, gae_lambda=0.95, clip_range=0.2, value_clip_range=0.2,
        value_coefficient=0.5, entropy_coefficient=a.entropy, max_grad_norm=1.0,
    )
    cfg_acc = PPOConfigV4(gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range, value_clip_range=cfg.value_clip_range,
                          value_coefficient=cfg.value_coefficient, entropy_coefficient=cfg.entropy_coefficient, max_grad_norm=None)
    acc_i = 0; acc_frames = 0; acc_lanes = 0; acc_wins = []; acc_load = 0.; acc_upd = 0.
    temps = (float(model.ppo_gate_temperature), float(model.ppo_action_temperature), float(model.ppo_continue_temperature))
    metrics = open(run / 'metrics.jsonl', 'a', buffering=1)
    pfsp = PFSPTracker(run.parent)

    def publish():
        tmp = wdir / 'latest.pt.tmp'
        torch.save(dict(model=model.state_dict(), version=version), tmp)
        os.replace(tmp, wdir / 'latest.pt')
        if version > 0 and version % a.history_every == 0:
            torch.save(dict(model=model.state_dict(), version=version), hdir / f'v{version:04d}.pt')

    publish()
    print(f'allenamento pronto: versione={version}, fase={phase}, lanes={a.lanes}, amp={a.amp}, temp={temps}', flush=True)

    picked = set()
    loaded_q = queue.Queue(maxsize=1)

    def _load():
        files = []
        now_wall = time.time()
        for f in inbox.glob('*.npz'):
            try:
                mt = f.stat().st_mtime
            except FileNotFoundError:
                continue
            if f in picked:
                continue
            if now_wall - mt > 900:
                f.unlink(missing_ok=True)
            else:
                files.append((mt, f))
        files = [f for _, f in sorted(files, reverse=True)]
        if len(files) < a.lanes:
            return None

        t0 = time.time()
        eps = []
        stale = 0
        sig0 = None
        for f in files:
            try:
                z = np.load(f, allow_pickle=False)
                meta = json.loads(bytes(z['meta']).decode())
                if version - int(meta['version']) > a.max_stale:
                    stale += 1
                    f.unlink(missing_ok=True)
                    continue
                if sig0 is None:
                    sig0 = meta['sig']
                elif meta['sig'] != sig0:
                    continue
                tmpl = pickle.loads(bytes(z['tmpl']))
                if meta.get('sparse'):
                    data = unpack_episode_sparse(z['offsets'], z['idx_u16'], z['val_u64'], int(meta['total']))
                else:
                    data = z['data']
                eps.append(dict(
                    data=data, act=z['act'], lp=z['lp'], val=z['val'], rew=z['rew'],
                    meta=meta, tmpl=tmpl, file=f,
                ))
            except Exception as e:
                print('file scartato', f.name, repr(e)[:100], flush=True)
                f.unlink(missing_ok=True)
            if len(eps) >= a.lanes:
                break

        if len(eps) < a.lanes:
            return None

        B = len(eps)
        T = max(len(e['rew']) for e in eps)
        total = eps[0]['meta']['total']
        items = [tuple(x[:2]) + (tuple(x[2]),) + tuple(x[3:]) for x in eps[0]['meta']['items']]

        raw = np.zeros((T, B, total), np.uint8)
        act = np.zeros((T, B, 10), np.int64)
        act[:, :, 2:] = -1
        lp = np.zeros((T, B), np.float32)
        val = np.zeros((T, B), np.float32)
        rew = np.zeros((T, B), np.float32)
        valid = np.zeros((T, B), bool)
        term = np.zeros((T, B), bool)
        nxt = np.zeros((T, B), np.float32)

        for j, e in enumerate(eps):
            L = len(e['rew'])
            raw[:L, j] = e['data'][:L]
            raw[L:, j] = e['data'][L - 1]
            e['data'] = None
            act[:L, j] = e['act'][:L]
            lp[:L, j] = e['lp'][:L]
            val[:L, j] = e['val'][:L]
            rew[:L, j] = e['rew'][:L]
            valid[:L, j] = True
            term[L - 1, j] = True
            nxt[:L - 1, j] = e['val'][1:L]

        rawt = torch.from_numpy(raw)
        packed = build_row_packed(eps[0]['tmpl'], items, rawt.reshape(T * B, total))   # via veloce dell'originale
        del rawt, raw; gc.collect(); rawt = None; raw = None
        obs = tuple(packed.narrow_rows(t * B, B).record for t in range(T))
        acts = tuple(
            ActionSequenceV4(
                gate=torch.from_numpy(act[t, :, 0].copy()),
                micro_action_count=torch.from_numpy(act[t, :, 1].copy()),
                candidate_index=torch.from_numpy(act[t, :, 2:4].copy()),
                candidate_uid=torch.from_numpy(act[t, :, 4:6].copy()),
                target_cell=torch.from_numpy(act[t, :, 6:8].copy()),
                delay_offset_bin=torch.from_numpy(act[t, :, 8:10].copy()),
            )
            for t in range(T)
        )
        es = torch.zeros((T, B), dtype=torch.bool)
        es[0] = True
        rollout = StoredRolloutV4(
            observations=obs, actions=acts, episode_start=es,
            valid_mask=torch.from_numpy(valid),
            behavior_log_prob=torch.from_numpy(lp),
            behavior_value=torch.from_numpy(val),
            next_value=torch.from_numpy(nxt),
            rewards=torch.from_numpy(rew),
            terminated=torch.from_numpy(term),
            truncated=torch.zeros((T, B), dtype=torch.bool),
            initial_state=model.initial_state(B, device='cpu'),
            gate_temperature=temps[0], action_temperature=temps[1], continue_temperature=temps[2],
            policy_state_id=f'ever-v{version}', packed_observation_rows=packed,
        )
        gc.collect()
        for e in eps:
            picked.add(e['file'])
        return dict(rollout=rollout, eps=eps, valid=valid, stale=stale, B=B, load=time.time() - t0)

    def _loader():
        while True:
            try:
                r = _load()
            except Exception as e:
                print('caricamento fallito', repr(e)[:150], flush=True); r = None
            if r is None:
                time.sleep(1.0); continue
            loaded_q.put(r)

    import torch.multiprocessing as tmp
    from royalefish.ever_loader import run_loader
    from native_runner.training.v4.remote_worker_transport import ReceivedRowPackedTensorRecord
    ctx = tmp.get_context('spawn')
    ver_shared = ctx.Value('i', version)
    loaded_q = ctx.Queue(maxsize=1)
    ctx.Process(target=run_loader, args=(str(inbox), a.lanes, a.lanes * a.accum, a.max_stale, ver_shared, loaded_q), daemon=True).start()

    while not (run / 'STOP').exists() and not (run.parent / 'STOP').exists():
        pfsp.update()

        # Passaggio automatico da Fase 0 (Warmup Critic) a Fase 1 (PPO completo)
        want_warmup = version < a.warmup_updates
        if want_warmup != in_warmup:
            in_warmup = want_warmup
            opt, phase = configure_trainable(model, in_warmup, a.lr, a.warmup_lr)
            print(f'=== PASSAGGIO ALLA FASE {phase.upper()} ALLA VERSIONE {version} ===', flush=True)

        try:
            g = loaded_q.get(timeout=5)
        except queue.Empty:
            continue
        T = g['T']; B = g['B']; stale = 0
        packed = ReceivedRowPackedTensorRecord(g['schema'], T * B, tuple(g['buffers']), {d: d for d in g['buffers']}, g['buffers'])
        obs = tuple(packed.narrow_rows(t * B, B).record for t in range(T))
        actg = g['act']
        acts = tuple(ActionSequenceV4(gate=actg[t, :, 0].clone(), micro_action_count=actg[t, :, 1].clone(),
                                      candidate_index=actg[t, :, 2:4].clone(), candidate_uid=actg[t, :, 4:6].clone(),
                                      target_cell=actg[t, :, 6:8].clone(), delay_offset_bin=actg[t, :, 8:10].clone()) for t in range(T))
        es = torch.zeros((T, B), dtype=torch.bool); es[0] = True
        valid = g['valid'].numpy()
        rollout = StoredRolloutV4(observations=obs, actions=acts, episode_start=es, valid_mask=g['valid'],
                                  behavior_log_prob=g['lp'], behavior_value=g['val'], next_value=g['nxt'], rewards=g['rew'],
                                  terminated=g['term'], truncated=torch.zeros((T, B), dtype=torch.bool),
                                  initial_state=model.initial_state(B, device='cpu'),
                                  gate_temperature=temps[0], action_temperature=temps[1], continue_temperature=temps[2],
                                  policy_state_id=f'ever-v{version}', packed_observation_rows=packed)
        eps = [dict(file=Path(f), meta=dict(result=r)) for f, r in zip(g['files'], g['results'])]
        t0 = time.time() - g['load']
        t1 = time.time()
        use_anchor = 0.0 if (in_warmup or acc_i % 2 == 1) else a.anchor   # ancora su meta' dei gruppi: meta' calcolo
        with open(a.lock, 'a') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                model.train()
                if acc_i == 0:
                    opt.zero_grad(set_to_none=True)
                stats = distributed_ppo_update_v4(
                    model, _HoldOptimizer(opt), rollout,
                    epochs=(1 if in_warmup else a.epochs),
                    lane_minibatch_size=a.minibatch,
                    lane_microbatch_size=a.micro,
                    sequence_chunk_steps=a.chunk,
                    config=cfg_acc,
                    autocast_dtype={'none': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[a.amp],
                    device_rollout_budget_gib=a.gpu_budget,
                    reference_model=(None if use_anchor <= 0.0 else ref),
                    reference_initial_state=(None if use_anchor <= 0.0 else ref.initial_state(B, device='cpu')),
                    anchor_kl_coefficient=use_anchor,
                )
                model.eval()
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

        del rollout, g, packed, obs, acts
        gc.collect()
        for e in eps:
            e['file'].unlink(missing_ok=True)
        acc_i += 1; acc_frames += int(valid.sum()); acc_lanes += B; acc_wins += [e['meta']['result'] > 0 for e in eps]
        acc_load += t1 - t0; acc_upd += time.time() - t1
        if acc_i < a.accum:
            continue
        # UN solo passo con i gradienti di tutti i gruppi (media), come l'aggiornamento originale di Everlight
        params = [q for q in model.parameters() if q.grad is not None]
        for q in params:
            q.grad.div_(acc_i)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0)) if params else 0.
        if math.isfinite(gnorm):
            opt.step()
        opt.zero_grad(set_to_none=True)
        version += 1
        ver_shared.value = version
        publish()
        last = stats[-1] if stats else {}
        wins = float(np.mean(acc_wins)); B = acc_lanes
        rec_out = dict(
            version=version, phase=phase, utc=time.time(), lanes=acc_lanes, frames=acc_frames, stale=stale, grad_norm=round(gnorm, 4),
            win=round(wins, 3), load_s=round(acc_load, 2), update_s=round(acc_upd, 2),
            **{k: round(float(v), 5) for k, v in last.items() if isinstance(v, (int, float))},
        )
        metrics.write(json.dumps(rec_out) + '\n')
        print(time.strftime('%H:%M:%S'), f'[{phase}] v{version}', 'partite', acc_lanes, 'mosse', acc_frames,
              'win', round(wins, 3), 'load', round(acc_load, 2), 's upd', round(acc_upd, 2), 's',
              'ev', round(float(last.get('value_explained_variance', 0.)), 3),
              'kl', round(float(last.get('anchor_kl', 0.)), 4), flush=True)
        acc_i = 0; acc_frames = 0; acc_lanes = 0; acc_wins = []; acc_load = 0.; acc_upd = 0.

        if version % 4 == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), version=version, phase=phase), state_f)

    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), version=version, phase=phase), state_f)
    print('fermato e salvato alla versione', version, flush=True)


if __name__ == '__main__':
    main()
