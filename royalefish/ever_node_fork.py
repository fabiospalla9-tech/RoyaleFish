"""Lanciatore fork per tutti i raccoglitori Everlight:
carica una sola volta PyTorch, NumPy e i cataloghi V4 nel padre (condivisi in RAM sul nodo NUMA 0),
poi crea i figli legandoli ciascuno al proprio core fisico con OOMScoreAdjust=800 e riavvio automatico."""
import os
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  PYTHONUNBUFFERED='1', CR_ALLOW_UNKNOWN_SEMANTICS='1')

import argparse
import gc
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import torch
torch.set_num_threads(1)

import royalefish.ever_collector as EC  # noqa: E402


def warm():
    try:
        from native_runner.training.v4.ppo_runtime import ppo_environment_config_v4
        from native_runner.training.v4.deck_pool import DeckPoolManifestV4
        from royalefish.config import POOL
        ppo_environment_config_v4()
        DeckPoolManifestV4.load(POOL)
    except Exception as e:
        print('warm parziale:', repr(e), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True, help='JSON list of collector configs')
    args = p.parse_args()
    run = args.run
    specs = json.loads(args.spec.read_text())
    logs = run / 'logs'
    logs.mkdir(parents=True, exist_ok=True)

    warm()
    gc.collect()
    gc.freeze()

    children = {}
    restarts = {c['index']: 0 for c in specs}
    by_idx = {c['index']: c for c in specs}
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))

    def spawn(c):
        idx = c['index']
        pid = os.fork()
        if pid == 0:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            try:
                os.sched_setaffinity(0, {int(c['cpu'])})
            except Exception:
                pass
            try:
                Path('/proc/self/oom_score_adj').write_text('800\n')
            except Exception:
                pass
            fd = os.open(str(logs / f'col-{idx:02d}.log'), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            sys.argv = [
                'ever_collector.py',
                '--run', str(run),
                '--index', str(idx),
                '--port', str(c['port']),
                '--slots', str(c.get('slots', 16)),
                '--a', str(c['a']),
                '--b', str(c['b']),
                '--record', str(c.get('record', 0.30)),
                '--mirror-frac', str(c.get('mirror_frac', 0.10)),
            ]
            code = 0
            try:
                EC.main()
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else 1
            except BaseException:
                import traceback
                traceback.print_exc()
                code = 1
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
        children[idx] = pid

    for c in specs:
        spawn(c)
        time.sleep(0.08)  # avvio scaglionato per evitare picchi simultanei sui server

    print(f'ever_node_fork attivo con {len(specs)} raccoglitori', flush=True)
    while not stopping[0] and not (run / 'STOP').exists():
        for idx, pid in list(children.items()):
            done, status = os.waitpid(pid, os.WNOHANG)
            if done and not (run / 'STOP').exists() and not stopping[0]:
                restarts[idx] += 1
                print(f'collector {idx} uscito (status={status}), riavvio #{restarts[idx]}', flush=True)
                if restarts[idx] <= 500:
                    time.sleep(1.5)
                    spawn(by_idx[idx])
                else:
                    children.pop(idx, None)
        time.sleep(2.0)

    for pid in children.values():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 30.
    for pid in children.values():
        while time.time() < deadline:
            if os.waitpid(pid, os.WNOHANG)[0]:
                break
            time.sleep(0.2)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


if __name__ == '__main__':
    main()
