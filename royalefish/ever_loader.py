"""Processo separato che prepara i gruppi di partite per l'allenamento (cosi' la GPU non aspetta e Python non rallenta il
calcolo). Raggruppa le partite per lunghezza simile: meno riempimento inutile. Passa i dati in memoria condivisa."""
import json, os, pickle, time
from pathlib import Path

import numpy as np
import torch

from royalefish.teacher_pack import unpack_episode_sparse
from royalefish.ever_packed import build_row_packed


def run_loader(inbox, lanes, pool, max_stale, ver, q, max_age=900):
    inbox = Path(inbox); picked = set(); pending = []
    torch.set_num_threads(2)
    while True:
        try:
            if not pending:
                files = []; now = time.time()
                for f in inbox.glob('*.npz'):
                    if f in picked:
                        continue
                    try:
                        mt = f.stat().st_mtime
                    except FileNotFoundError:
                        continue
                    if now - mt > max_age:
                        f.unlink(missing_ok=True)
                    else:
                        files.append((mt, f))
                picked = {f for f in picked if f.exists()}
                files = [f for _, f in sorted(files, reverse=True)]
                if len(files) < lanes:
                    time.sleep(1.0); continue
                cand = []
                for f in files[:pool]:
                    try:
                        z = np.load(f, allow_pickle=False); meta = json.loads(bytes(z['meta']).decode())
                        if ver.value - int(meta['version']) > max_stale:
                            f.unlink(missing_ok=True); continue
                        cand.append((len(z['rew']), meta['sig'], f))
                    except Exception:
                        f.unlink(missing_ok=True)
                cand.sort(key=lambda x: (x[1], x[0]))          # stessa impronta, poi per lunghezza
                groups = []
                i = 0
                while i + lanes <= len(cand):
                    g = cand[i:i + lanes]
                    if len({x[1] for x in g}) == 1:
                        groups.append([x[2] for x in g]); i += lanes
                    else:
                        i += 1
                if not groups:
                    time.sleep(1.0); continue
                pending = groups
            group = pending.pop(0)
            for f in group:
                picked.add(f)
            t0 = time.time(); eps = []
            for f in group:
                try:
                    z = np.load(f, allow_pickle=False); meta = json.loads(bytes(z['meta']).decode())
                    data = unpack_episode_sparse(z['offsets'], z['idx_u16'], z['val_u64'], int(meta['total'])) if meta.get('sparse') else z['data']
                    eps.append(dict(data=data, act=z['act'], lp=z['lp'], val=z['val'], rew=z['rew'], meta=meta,
                                    tmpl=pickle.loads(bytes(z['tmpl'])), file=str(f)))
                except Exception as e:
                    print('file scartato', f.name, repr(e)[:100], flush=True)
            if len(eps) < max(2, lanes // 2):
                continue
            B = len(eps); T = max(len(e['rew']) for e in eps); total = eps[0]['meta']['total']
            items = [tuple(x[:2]) + (tuple(x[2]),) + tuple(x[3:]) for x in eps[0]['meta']['items']]
            raw = np.zeros((T, B, total), np.uint8)
            act = np.zeros((T, B, 10), np.int64); act[:, :, 2:] = -1
            lp = np.zeros((T, B), np.float32); val = np.zeros((T, B), np.float32); rew = np.zeros((T, B), np.float32)
            valid = np.zeros((T, B), bool); term = np.zeros((T, B), bool); nxt = np.zeros((T, B), np.float32)
            for j, e in enumerate(eps):
                L = min(len(e['rew']), len(e['data']), len(e['act']))
                raw[:L, j] = e['data'][:L]; raw[L:, j] = e['data'][L - 1]; e['data'] = None
                act[:L, j] = e['act'][:L]; lp[:L, j] = e['lp'][:L]; val[:L, j] = e['val'][:L]; rew[:L, j] = e['rew'][:L]
                valid[:L, j] = True; term[L - 1, j] = True; nxt[:L - 1, j] = e['val'][1:L]
            packed = build_row_packed(eps[0]['tmpl'], items, torch.from_numpy(raw).reshape(T * B, total))
            del raw
            out = dict(schema=packed.schema, buffers=packed.buffers, T=T, B=B,
                       act=torch.from_numpy(act), lp=torch.from_numpy(lp), val=torch.from_numpy(val), rew=torch.from_numpy(rew),
                       valid=torch.from_numpy(valid), term=torch.from_numpy(term), nxt=torch.from_numpy(nxt),
                       files=[e['file'] for e in eps], results=[float(e['meta']['result']) for e in eps],
                       load=time.time() - t0)
            q.put(out)
        except Exception as e:
            print('caricatore: errore', repr(e)[:200], flush=True); time.sleep(2)
