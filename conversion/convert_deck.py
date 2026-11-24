"""Converts the WON games of one exact deck from a replay dataset into RoyaleSim states.

Example (Minion Giant 2.8, Fireball or Lightning):
  RF_TABLE=client160635007_20261103 NATIVE_RUNTIME=/path/to/runtime python conversion/convert_deck.py \
    --dataset /data/IL_Replay_1115/replays --out /data/converted \
    --cards knight,ice-wizard-hero,cannon-ev1,bats-ev1,minion-giant,barbarian-barrel,skeletons \
    --last fireball:2.0,lightning:1.5 --check IceWizard,MinionGiant
"""
import argparse, collections as c, glob, gzip, json, os, pickle, sys, time, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow.parquet as Q
import adapt_1115 as A
import convert_replay as C

TABLE = {'cards.CARD_TABLE': json.dumps(os.environ['RF_TABLE']), 'placement.SPELL_AS_DEPLOY_NO_DEPLOY_CELLS': json.dumps('client16635_can_place_on_buildings')} if os.environ.get('RF_TABLE') else {}


def init():
    C.init_worker(); original = C.royalesim.Battle
    C.royalesim.Battle = lambda *a, **k: original(*a, **{**k, 'calibration_overrides': {**TABLE, **(k.get('calibration_overrides') or {})}})


def scan(job):
    path, base, last = job; found = []; at = 0
    for batch in Q.ParquetFile(path).iter_batches(batch_size=256, columns=['payload_json']):
        for text in batch.column(0).to_pylist():
            b = json.loads(text)['battle']
            for side, me, foe in ((0, 'team', 'opponent'), (1, 'opponent', 'team')):
                try:
                    deck = {x['card_key'] for x in b[me]['players'][0]['deck']}
                except (IndexError, KeyError):
                    continue
                rest = deck - base
                if b[me]['crowns'] > b[foe]['crowns'] and base < deck and len(deck) == 8 and len(rest) == 1 and next(iter(rest)) in last:
                    found.append((path, at, side, last[next(iter(rest))]))
            at += 1
    return found


def work(job):
    path, chunk, rows, out, check = job; name = f"{out}/{os.path.basename(path).split('.')[0]}_c{chunk:03d}"; st = c.Counter()
    if os.path.exists(name + '.done'):
        return json.load(open(name + '.done'))
    want = {r for r, _, _ in rows}; texts = {}; at = 0
    for batch in Q.ParquetFile(path).iter_batches(batch_size=128, columns=['payload_json']):
        col = batch.column(0)
        for i in range(len(col)):
            if at + i in want:
                texts[at + i] = col[i].as_py()
        at += len(col)
    tmp = name + '.tmp'; output = gzip.open(tmp, 'wb', compresslevel=3)
    for row, side, weight in rows:
        st['seen'] += 1
        st['abilities_in_replays'] += sum(e.get('kind') == 'activate_ability' and e.get('side') == ('team' if side == 0 else 'opponent') for e in json.loads(texts[row])['events'])
        try:
            x = C.convert((A.adapt(texts[row]), side))
        except Exception:
            st['crash'] += 1; continue
        if 'error' in x:
            st['error'] += 1; continue
        if not set(check) <= set(x['decks'][0]):
            st['wrong_side'] += 1; continue
        cut = 10 ** 9
        for team in (0, 1):      # stop where the simulated hand no longer matches the replay
            for i, h in enumerate([h for h in x['history'] if h[1] == team]):
                planned = x['planned'][team]
                if i >= len(planned) or h[2] != planned[i]:
                    cut = min(cut, h[0]); st['cut'] += 1; break
        x['samples'] = [s for s in x['samples'] if s['tick'] < cut]; x['history'] = [h for h in x['history'] if h[0] < cut]
        if not any(s['action'] for s in x['samples']):
            st['empty'] += 1; continue
        st['abilities_kept'] += sum(1 for s in x['samples'] if s['action'] and s['action'][0] == 'ability')
        x['tier'] = 0; x['weight'] = weight; x.pop('planned', None); x.pop('dt_card', None); pickle.dump(x, output, protocol=4); st['kept'] += 1; st['samples'] += len(x['samples'])
    output.close(); os.replace(tmp, name + '.pkl.gz') if st['kept'] else os.remove(tmp)
    json.dump(dict(st), open(name + '.t', 'w')); os.replace(name + '.t', name + '.done'); return dict(st)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); ap.add_argument('--out', required=True); ap.add_argument('--cards', required=True, help='the 7 fixed card keys'); ap.add_argument('--last', required=True, help='allowed eighth cards with weights, e.g. fireball:2.0,lightning:1.5'); ap.add_argument('--check', required=True, help='simulator card names that must be on the converted side'); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); base = set(a.cards.split(',')); last = {k: float(v) for k, v in (x.split(':') for x in a.last.split(','))}; files = sorted(glob.glob(a.dataset + '/**/*.parquet', recursive=True)); check = a.check.split(',')
    with mp.Pool(a.workers) as pool:
        rows = sorted(y for found in pool.imap_unordered(scan, [(f, base, last) for f in files]) for y in found)
    print('selected', dict(c.Counter(r[3] for r in rows)), flush=True); rows = rows[:a.limit] if a.limit else rows; groups = c.defaultdict(list)
    for r in rows:
        groups[r[0]].append((r[1], r[2], r[3]))
    jobs = [(f, k, g[i:i + 40], a.out, check) for f, g in sorted(groups.items()) for k, i in enumerate(range(0, len(g), 40))]; total = c.Counter(); t0 = time.time()
    with mp.Pool(a.workers, initializer=init) as pool:
        for d in pool.imap_unordered(work, jobs):
            total.update(d)
    print('done in', round((time.time() - t0) / 60, 1), 'min', json.dumps(dict(total)), flush=True)
