"""Creates the league folder from two published checkpoints.
Usage: python training/setup_pool.py --pool pool --a models/H-minion-giant-28-v342.pt --b models/B-general-12m-424k.pt"""
import argparse, json, os, shutil, sys
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ap = argparse.ArgumentParser(); ap.add_argument('--pool', required=True); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True); x = ap.parse_args()
os.makedirs(x.pool, exist_ok=True); shutil.copy(os.path.join(ROOT, 'royalefish12m', 'meta.json'), os.path.join(x.pool, 'meta.json')); taus = {}
for i, (name, path) in enumerate((('A', x.a), ('B', x.b))):
    c = torch.load(path, map_location='cpu', weights_only=True); taus[str(i)] = float(c['tau'])
    torch.save({k[4:]: v for k, v in c.items() if k.startswith('net.')}, os.path.join(x.pool, f'{name}_init.pt'))
json.dump(taus, open(os.path.join(x.pool, 'taus.json'), 'w')); json.dump({'a': [], 'b': []}, open(os.path.join(x.pool, 'manifest.json'), 'w')); print('pool ready:', x.pool)
