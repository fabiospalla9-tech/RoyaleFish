"""Converted replays -> compact arrays on disk for model B.
Variable-length storage: entities and the match history are kept once, a sample only points to them.
Usage: prepare_dataset.py CONV_DIR OUT_DIR [workers]
"""
import glob, gzip, json, math, os, pickle, random, sys, time, zlib
import multiprocessing as mp
import numpy as np
import royalesim

PARTS = os.environ.get('PARTS', '/tmp/royalefish_parts')   # temporary pieces live in RAM, not on the nearly full disk
MAXE = 64
WAIT_KEEP = 0.25
DELAY = 20
ENT_W = 20       # card, unit type, form, mine, x, y, hp, maxhp, flying, deploying, attacking, tower, stun, knockback, shield, charged, buffs, ability windup, ability active, spell
G_W = 25
_b = royalesim.Battle(card_names=None, slot_of_k=[[0, 1, 2]] * 2)
VOCAB = ['<pad>', '<king>', '<princess>', '<unknown>'] + [r[0] for r in json.loads(_b.catalogue_json())]
VID = {n: i for i, n in enumerate(VOCAB)}
UNITS = json.loads(_b.unit_types_json())
UNIT_REMAP = None
if os.environ.get('RF_TABLE'):        # keep every unit on the embedding row it was trained with; units new in 16.635 go after them
    _n = json.loads(royalesim.Battle(card_names=None, slot_of_k=[[0, 1, 2]] * 2, calibration_overrides={'cards.CARD_TABLE': json.dumps(os.environ['RF_TABLE'])}).unit_types_json())
    _extra = [u for u in _n if u not in UNITS]; UNITS_ALL = UNITS + _extra; UNIT_REMAP = [UNITS_ALL.index(u) for u in _n]
BITS = {n: 1 << i for i, n in enumerate(royalesim.STATUS_BITS)}


def ent_form(flags):
    return 2 if flags & BITS['hero'] else 1 if flags & BITS['evolved'] else 0


def encode(s, names):
    rows = []
    def vid(i):
        return VID.get(names[i], VID['<unknown>']) if 0 <= i < len(names) else VID['<unknown>']
    for e in s['ents']:
        uid, team, kind, cid, x, y, hp, mhp, fly, dep, stun, kb, ph, fl, ut, sh, ch, nb = e
        nid = VID['<king>'] if kind == 2 else VID['<princess>'] if kind == 3 else vid(cid)
        rows.append([nid, ((UNIT_REMAP[ut] if UNIT_REMAP else ut) + 1) if ut is not None and ut >= 0 else 0, ent_form(fl), 1.0 if team == 0 else 0.0, x / 18000, y / 32000, hp / max(1, mhp), math.log1p(max(mhp, 0)) / 9,
                     1.0 if fly else 0.0, 1.0 if dep and dep > 0 else 0.0, 1.0 if ph and ph > 0 else 0.0, 1.0 if kind in (2, 3) else 0.0, 1.0 if stun and stun > 0 else 0.0,
                     1.0 if kb and kb > 0 else 0.0, 1.0 if sh and sh > 0 else 0.0, 1.0 if ch else 0.0, min(nb, 4) / 4, 1.0 if fl & BITS['ability_windup'] else 0.0,
                     1.0 if fl & BITS['ability_active'] else 0.0, 0.0])
    for sp in s['spells']:
        rows.append([vid(sp[1]), 0, 0, 1.0 if sp[0] == 0 else 0.0, sp[2] / 18000, sp[3] / 32000, 1.0, 0.0, 0, 1.0, 0, 0, 0, 0, 0, 0, min(max(sp[4], 0), 60) / 60, 0, 0, 1.0])
    rows.sort(key=lambda r: (r[11] == 0.0, r[19], abs(r[5] * 32000 - 16000)))
    E = np.array(rows[:MAXE], np.float16).reshape(-1, ENT_W)
    me, op = s['players']
    th = list(me[3]) + list(op[3]); maxk, maxp = max(th[0], th[3], 1), max(th[1], th[2], th[4], th[5], 1)
    g = [me[0] / 10000, s['tick'] / 7200, 1.0 if s['overtime'] else 0.0, th[0] / maxk, th[1] / maxp, th[2] / maxp, th[3] / maxk, th[4] / maxp, th[5] / maxp, (s.get('rate') or 1) / 3]
    buttons = [[0, 0]] * 3; ready = 0
    for k in range(3):
        r = me[5][k] if k < len(me[5]) else None
        ok = bool(r and r[0] and r[2] * 1000 <= me[0])
        ready |= (1 << k) if ok else 0
        g += [1.0 if ok else 0.0, 1.0 if r and r[1] else 0.0, (r[2] / 10) if r else 0.0]
        if r:
            buttons[k] = [vid(r[3]), 2]
    for k in range(3):
        r = op[5][k] if k < len(op[5]) else None
        g += [1.0 if r and r[0] else 0.0, 1.0 if r and r[1] else 0.0]
    evo = {r[0]: r for r in me[4]}
    hand = []
    for c in list(me[1])[:4] + [me[2]]:
        if c is None or c < 0:
            hand.append([0, 0, 0]); continue
        r = evo.get(c)
        hand.append([vid(c), (1 if r[2] else 0) if r else None, int(100 * min(r[1] / max(r[3], 1), 1)) if r else 0])
    return E, np.array(g, np.float32), hand, buttons, ready


def one(f):
    out = []
    with gzip.open(f, 'rb') as z:
        while True:
            try:
                r = pickle.load(z)
            except EOFError:
                break
            names = r['names']; deck = r['decks'][0]; forms = r['forms'][0]; fm = dict(zip(deck, forms))
            split = 'val' if zlib.crc32(r['tag'].encode()) % 10 == 0 else 'train'
            rng = random.Random(zlib.crc32(r['tag'].encode()))
            hist = np.zeros((len(r['history']), 6), np.float32)
            for i, (t, team, nm, fo, x, y) in enumerate(r['history']):
                hist[i] = [t, team, VID.get(nm, VID['<unknown>']), fo, x / 18000, y / 32000]
            E_, G_, H_, I_ = [], [], [], []
            ticks = hist[:, 0] if len(hist) else np.zeros(0)
            for s in r['samples']:
                a = s['action']
                if a is None and rng.random() > WAIT_KEEP:
                    continue
                E, G, hand, buttons, ready = encode(s, names)
                H = np.zeros((8, 3), np.int16)
                for k, h in enumerate(hand):
                    nm = VOCAB[h[0]] if h[0] else None
                    H[k] = [h[0], (h[1] if h[1] is not None else (2 if fm.get(nm) == 2 else 0)), h[2]]
                for k, b in enumerate(buttons):
                    H[5 + k] = [b[0], b[1], 0]
                if a is None:
                    A, X, Y = 0, -1, -1
                elif a[0] == 'card':
                    A, X, Y = 1 + int(a[1]), min(17, max(0, int(a[2]))), min(31, max(0, int(a[3])))
                else:
                    A, X, Y = 5 + int(a[1]), -1, -1
                E_.append(E); G_.append(G); H_.append(H)
                I_.append((len(E), int(np.searchsorted(ticks, s['tick'], 'left')), s['tick'], A, X, Y, ready))
            if I_:
                out.append((split, r['weight'], r['tier'], hist, np.concatenate(E_).astype(np.float16), np.stack(G_), np.stack(H_), np.array(I_, np.int32)))
    tmp = f + '.prep.tmp'
    return out


def worker(a):
    f, d = a
    parts = {'train': [], 'val': []}
    for x in one(f):
        parts[x[0]].append(x)
    st = {}
    for sp, L in parts.items():
        if not L:
            continue
        p = f"{PARTS}/{sp}_{os.path.basename(f).replace('.pkl.gz', '')}.npz"
        np.savez(p, hist=np.concatenate([x[3] for x in L]), hn=np.array([len(x[3]) for x in L], np.int32), E=np.concatenate([x[4] for x in L]), G=np.concatenate([x[5] for x in L]),
                 H=np.concatenate([x[6] for x in L]), I=np.concatenate([x[7] for x in L]), sn=np.array([len(x[7]) for x in L], np.int32), w=np.array([x[1] for x in L], np.float32), tier=np.array([x[2] for x in L], np.int8))
        st[sp] = (len(L), int(sum(len(x[7]) for x in L)), int(sum(len(x[4]) for x in L)), int(sum(len(x[3]) for x in L)))
    return st


def main():
    src, out = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    os.makedirs(PARTS, exist_ok=True); os.makedirs(out, exist_ok=True)
    fs = sorted(glob.glob(f'{src}/*.pkl.gz')); t0 = time.time(); tot = {'train': [0, 0, 0, 0], 'val': [0, 0, 0, 0]}
    with mp.Pool(n) as P:
        for st in P.imap_unordered(worker, [(f, out) for f in fs]):
            for sp, v in st.items():
                tot[sp] = [a + b for a, b in zip(tot[sp], v)]
    for sp in ('train', 'val'):
        rn, sn, en, hn = tot[sp]
        if not sn:
            continue
        Em = np.lib.format.open_memmap(f'{out}/{sp}_E.npy', 'w+', np.float16, (en, ENT_W)); Gm = np.lib.format.open_memmap(f'{out}/{sp}_G.npy', 'w+', np.float32, (sn, G_W))
        Hm = np.lib.format.open_memmap(f'{out}/{sp}_H.npy', 'w+', np.int16, (sn, 8, 3)); Mm = np.lib.format.open_memmap(f'{out}/{sp}_hist.npy', 'w+', np.float32, (hn, 6))
        Sm = np.lib.format.open_memmap(f'{out}/{sp}_S.npy', 'w+', np.int64, (sn, 10))   # e0, en, h0, hn, tick, A, X, Y, ready, weight x100
        e = s = h = 0
        for p in sorted(glob.glob(f'{PARTS}/{sp}_*.npz')):
            z = np.load(p); I = z['I']
            Em[e:e + len(z['E'])] = z['E']; Gm[s:s + len(I)] = z['G']; Hm[s:s + len(I)] = z['H']; Mm[h:h + len(z['hist'])] = z['hist']
            ee = e + np.concatenate([[0], np.cumsum(I[:, 0])[:-1]]); wt = np.repeat(z['w'], z['sn'])
            rh = np.repeat(np.concatenate([[0], np.cumsum(z['hn'])[:-1]]) + h, z['sn'])
            Sm[s:s + len(I)] = np.column_stack([ee, I[:, 0], rh, I[:, 1], I[:, 2], I[:, 3], I[:, 4], I[:, 5], I[:, 6], (wt * 100).astype(np.int64)])
            e += len(z['E']); s += len(I); h += len(z['hist'])
        for m in (Em, Gm, Hm, Mm, Sm):
            m.flush()
        print(sp, 'partite', rn, 'campioni', sn, 'entita', en, 'storia', hn, flush=True)
    json.dump({'vocab': VOCAB, 'units': len(UNITS) + 1, 'ent_w': ENT_W, 'g_w': G_W}, open(f'{out}/meta.json', 'w'))
    shutil_rm = [os.remove(p) for p in glob.glob(f'{PARTS}/*.npz')]
    print('fatto in', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
