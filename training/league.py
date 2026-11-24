"""League: new 2.6 (id 0) and generalist B (id 1) learn at the same time, against each other, their past versions and the frozen old 2.6.
Every decision kept for learning goes through the optimizer once. Usage: league.py OUT [workers] [battles]"""
import json, math, os, pickle, queue, random, socket, struct, sys, threading, time
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, '..', 'conversion'), os.path.join(HERE, '..')]
import numpy as np, torch, torch.nn.functional as F

GAMES = int(os.environ.get('GAMES', '256')); GAMMA, LAMBDA, CLIP, VF = 0.999, 0.95, 0.2, 0.5; LR = float(os.environ.get('LR', '3.3e-6')); BS = 512; TOKENS = int(os.environ.get('TOKENS', '42000'))
WAIT_KEEP = 0.1; WARMUP = int(os.environ.get('WARMUP', '2')); SNAP_EVERY, SNAP_KEEP = 10, 6; KL_TARGET = 0.10; BETA_MIN, BETA_MAX = 0.03, 3.0; REMOTE = os.environ.get('REMOTE', '127.0.0.1:7792')
A_INIT = os.environ['A_INIT']; B_INIT = os.environ['B_INIT']; A_DECK = json.loads(os.environ.get('A_DECK', 'null')); DECKS = os.path.join(HERE, 'league_decks_web.json')
DRILL_P = float(os.environ.get('DRILL_P', '0.25')); DRILL_B = {'Cannon', 'Tesla', 'GoblinCage', 'InfernoTower', 'Tombstone', 'GoblinHut', 'BombTower', 'BarbarianHut'}   # predict drill: games vs a past version, which defends with a building on real pro cells
BARREL_P = float(os.environ.get('BARREL_P', '0.25'))   # share of opponent decks that hold a Goblin Barrel: the Log timing is Fabio's priority
HARD_P = float(os.environ.get('HARD_P', '0.40')); HARD_MIN = 30    # share of opponent decks taken from the families this model struggles against; games needed before trusting a family
MIX = tuple(map(tuple, json.loads(os.environ.get('MIX', '[[0.60,"ab"],[0.22,"a_past"],[0.18,"b_past"]]'))))


# ------------------------------------------------------------------ worker
def worker(wid, n_battles, done_q, seed):
    import rl_env, convert_replay as C, prepare_dataset as P
    from league_common import OLD, POOL
    C.init_worker(); SUB = C.SUB; EF = C.EF; LAND = rl_env.COMMAND_DELAY
    cat = {r[0]: r for r in json.loads(rl_env.royalesim.Battle(card_names=None, slot_of_k=[[0, 1, 2]] * 2).catalogue_json())}
    from deck_shares import family
    decks = json.load(open(DECKS)); W = [d['w'] for d in decks]; rng = random.Random(seed); FAM = {}; hard_cache = [{}, 0.0]
    for d in decks:
        FAM.setdefault(family(d), []).append(d)
    PRO = json.load(open(os.path.join(HERE, 'pro_building_cells.json'))); BAR = [d for d in decks if 'GoblinBarrel' in d['d']]; BW = [d['w'] for d in BAR]

    def pick_hard(hero):
        if time.time() - hard_cache[1] > 20:
            hard_cache[1] = time.time()
            try:
                hard_cache[0] = json.load(open(POOL + '/hard.json'))
            except (OSError, ValueError):
                pass
        items = [(k, v) for k, v in hard_cache[0].get(str(hero), {}).items() if v['n'] >= HARD_MIN and k in FAM]
        if len(items) < 5:
            return None
        k = rng.choices([x[0] for x in items], [(1 - x[1]['wr']) ** 2 + 1e-3 for x in items])[0]
        return rng.choices(FAM[k], [d['w'] for d in FAM[k]])[0]
    host, port = REMOTE.split(',')[wid % len(REMOTE.split(','))].split(':'); socks = [socket.create_connection((host, int(port))) for _ in range(2)]
    for s in socks:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def send(sock, x):
        o = pickle.dumps(x, protocol=5); sock.sendall(struct.pack('!Q', len(o)) + o)

    def recv(sock):
        def rd(n):
            b = bytearray()
            while len(b) < n:
                c = sock.recv(n - len(b))
                if not c:
                    raise ConnectionError
                b += c
            return bytes(b)
        return pickle.loads(rd(struct.unpack('!Q', rd(8))[0]))

    def past(kind):
        try:
            ids = json.load(open(POOL + '/manifest.json'))[kind]
        except (OSError, ValueError, KeyError):
            ids = []
        if len(ids) > 2:      # two past versions per model at a time, rotating every 5 minutes: fewer model switches on the GPU, all versions still get played
            ids = random.Random(f'{kind}{int(time.time() // 300)}').sample(ids, 2)
        return rng.choice(ids) if ids else (0 if kind == 'a' else 1)

    def new_game():
        r = rng.random(); k = MIX[-1][1]; q = 0
        for p, name in MIX:
            q += p
            if r < q:
                k = name; break
        if k == 'ab': who = [0, 1]
        elif k == 'a_old': who = [0, OLD]
        elif k == 'b_old': who = [1, OLD]
        elif k == 'a_past': who = [0, past('a' if rng.random() < .5 else 'b')]
        else: who = [1, past('b' if rng.random() < .5 else 'a')]
        hero = who[0]; ht = 0; pushed = False
        if rng.random() < .5:
            who = who[::-1]; ht = 1
        meta = lambda m: m == 1 or m >= 2000

        def deck(t):
            nonlocal pushed
            if not meta(who[t]):
                return (A_DECK['d'], A_DECK['f'], os.environ.get('A_NAME', 'Fornace Cimitero')) if A_DECK and who[t] != OLD else (rl_env.HOG, rl_env.FORMS, None)
            x = None
            if t != ht and rng.random() < BARREL_P:
                x = rng.choices(BAR, BW)[0]
            elif t != ht and rng.random() < HARD_P:
                x = pick_hard(hero); pushed = pushed or x is not None
            x = x or rng.choices(decks, W)[0]
            return x['d'], x['f'], family(x)
        d0, f0, a0 = deck(0); d1, f1, a1 = deck(1)
        b = rl_env.Battle(rng.randrange(1 << 30), deck0=d0, deck1=d1, forms0=f0, forms1=f1); b.fams = [a0, a1]; b.no_predict = [who[1] == OLD, who[0] == OLD]; b.predict_cap = [2 if meta(who[1]) else 1, 2 if meta(who[0]) else 1]; b.pushed = pushed; b.elig = meta(who[1 - ht])
        opp = 1 - ht; b.drill = who[opp] >= 1000 and bool(set(d1 if opp == 1 else d0) & DRILL_B) and rng.random() < DRILL_P; b.drill_team = opp; b.drill_last = -999; b.drill_n = 0
        return b, who, k

    def encode(b, team):
        st = b.state; ents = []
        for e in st['entities']:
            x, y = e[EF['x']] // SUB, e[EF['y']] // SUB; tm = e[EF['team']]
            if team == 1:
                x, y, tm = 18000 - x, 32000 - y, 1 - tm
            ents.append((e[EF['uid']], tm, e[EF['kind']], e[EF['card_id']], x, y, e[EF['hp']], e[EF['max_hp']], e[EF['flying']], e[EF['deploy_ticks']], e[EF['stun_ticks']], e[EF['knockback_ticks']], e[EF['attack_phase']], e[EF['status_flags']] or 0, e[EF['unit_type']], e[EF['shield']] or 0, e[EF['charge']] or 0, len(e[EF['buffs']] or ())))
        spells = []
        for s in st['spells']:
            ax, ay, tm = s[5] // SUB, s[6] // SUB, s[0]
            if team == 1:
                ax, ay, tm = 18000 - ax, 32000 - ay, 1 - tm
            spells.append((tm, s[1], ax, ay, s[7]))
        pl = [(p['elixir_milli'], tuple(p['hand']), p['next_card'], tuple(p['tower_hp']), tuple(map(tuple, p['evo'])), tuple(map(tuple, p['abilities']))) for p in st['players']]
        if team == 1:
            pl = pl[::-1]
        now = st['tick'] + LAND; E, G, hand, buttons, ready = P.encode({'tick': now, 'ents': ents, 'spells': spells, 'players': pl, 'overtime': st['overtime'], 'rate': st['elixir_rate']}, b.names)
        fm = dict(zip(b.decks[team], b.forms[team])); H = np.zeros((8, 3), np.int16)
        for k, h in enumerate(hand):
            nm = P.VOCAB[h[0]] if h[0] else None; H[k] = [h[0], h[1] if h[1] is not None else (2 if fm.get(nm) == 2 else 0), h[2]]
        for k, x in enumerate(buttons):
            H[5 + k] = [x[0], x[1], 0]
        rows = []
        for (tk, tm, nm, f, x, y) in b.history[-160:]:
            land = tk + LAND; mine = tm == team
            if not mine and land > now - LAND:
                continue
            mx, my = (18000 - x, 32000 - y) if team == 1 else (x, y); rows.append((P.VID.get(nm, 3), f, 1.0 if mine else 0.0, mx / 18000, my / 32000, (now - land) / 600, 1.0))
        M = np.array(rows, np.float16).reshape(-1, 7); me = pl[0]; idx = list(me[1])[:4]; cost = st['players'][team]['hand_costs']
        A = [(k < len(cost) and cost[k] * 1000 <= me[0]) for k in range(4)]; S = [(k < len(idx) and cat[b.names[idx[k]]][1] in (2, 4, 5)) for k in range(4)]   # spells, Goblin Barrel, Miner and Drill can land on the enemy side
        return E, G, H, M, ready, A, S

    B = [new_game() for _ in range(n_battles)]; traj = [[[], []] for _ in range(n_battles)]; lanes = [range(0, n_battles // 2), range(n_battles // 2, n_battles)]

    def drill_act(i, acts):
        b = B[i][0]; t = b.drill_team
        if not b.drill or b.done or not b.can_play(t) or b.tick - b.drill_last < 100:
            return
        if b.utypes is None:
            b.utypes = json.loads(b.b.unit_types_json())
        near = False
        for e in b.state['entities']:
            u = e[EF['unit_type']]
            if e[EF['team']] != t and e[EF['kind']] not in (2, 3) and u is not None and 0 <= u < len(b.utypes) and b.utypes[u] in rl_env.WIN_UNITS:
                y = e[EF['y']] // SUB; y = 32000 - y if t == 1 else y
                near = near or 13500 <= y <= 17000   # the win condition is at the bridge: when pros drop the building
        if not near:
            return
        p = b.state['players'][t]
        for k, c in enumerate(list(p['hand'])[:4]):
            nm = b.names[c]
            if nm in DRILL_B and k < len(p['hand_costs']) and p['hand_costs'][k] * 1000 <= p['elixir_milli']:
                cells = PRO[nm]; x, y, _ = rng.choices(cells, [q[2] for q in cells])[0]
                acts.setdefault(i, {})[t] = (k, min(17, max(0, x + rng.choice((-1, 0, 0, 1)))), min(14, max(0, y + rng.choice((-1, 0, 0, 1))))); b.drill_last = b.tick; b.drill_n += 1
                return

    def prepare(L):
        nw = []; od = []
        for i in lanes[L]:
            b, who, _ = B[i]
            for team in (0, 1):
                if b.can_play(team):
                    if who[team] == OLD:
                        od.append((i, team, b.encode(team), [b.names[h] == 'Fireball' for h in b.state['players'][team]['hand']]))
                    else:
                        nw.append((i, team, who[team], encode(b, team)))
        if not nw and not od:
            return None
        msg = {'new': None, 'old': None}
        if nw:
            ne = max(1, max(len(x[3][0]) for x in nw)); nh = max(1, max(len(x[3][3]) for x in nw)); n = len(nw); E = np.zeros((n, ne, P.ENT_W), np.float16); M = np.zeros((n, nh, 7), np.float16)
            for q, x in enumerate(nw):
                E[q, :len(x[3][0])] = x[3][0]; M[q, :len(x[3][3])] = x[3][3]
            msg['new'] = {'E': E, 'N': np.array([len(x[3][0]) for x in nw], np.int16), 'G': np.stack([x[3][1] for x in nw]), 'H': np.stack([x[3][2] for x in nw]), 'M': M, 'R': np.array([x[3][4] for x in nw], np.int8), 'A': np.array([x[3][5] for x in nw], bool), 'S': np.array([x[3][6] for x in nw], bool), 'P': np.array([x[2] for x in nw], np.int32), 'X': np.array([any(x[3][5]) and B[x[0]][0].state['players'][x[1]]['elixir_milli'] >= 10000 and not any(h[1] == x[1] for h in B[x[0]][0].history) for x in nw], bool)}
        if od:
            msg['old'] = {'E': np.stack([x[2][0] for x in od]).astype(np.float16), 'N': np.array([x[2][1] for x in od], np.int16), 'G': np.stack([x[2][2] for x in od]), 'H': np.stack([x[2][3] for x in od]).astype(np.int16), 'M': np.stack([x[2][4] for x in od]).astype(np.float16), 'S': np.array([x[3] for x in od], bool)}
        send(socks[L], msg); return nw, od, msg

    def finish(L, pend):
        nw, od, msg = pend; r = recv(socks[L]); acts = {}
        if nw:
            ch, cell, lp, v = r['new']; x = msg['new']
            for q, (i, team, mid, _) in enumerate(nw):
                b = B[i][0]; c = int(ch[q]); forced = bool(x['X'][q])
                if forced and c >= 0:
                    b.stats['forced'][team] += 1
                if c >= 0:
                    acts.setdefault(i, {})[team] = (c, int(cell[q]) % 18, int(cell[q]) // 18) if c < 4 else (c, 0, 0)
                if mid not in (0, 1):
                    b.take_reward(team); continue
                tr = traj[i][team]
                if tr:
                    tr[-1]['r'] += b.take_reward(team)
                else:
                    b.take_reward(team)
                keep = not forced and (c >= 0 or rng.random() < WAIT_KEEP); row = {'c': c + 1, 'cell': int(cell[q]), 'lp': float(lp[q]), 'v': float(v[q]), 'r': 0.0, 'keep': keep}
                if keep:
                    ne, hl = int(x['N'][q]), int(x['M'][q, :, 6].sum()); row.update(E=x['E'][q, :ne].copy(), G=x['G'][q].copy(), H=x['H'][q].copy(), M=x['M'][q, :hl].copy(), R=int(x['R'][q]), A=x['A'][q].copy(), S=x['S'][q].copy())
                tr.append(row)
        if od:
            sl, ce = r['old']
            for q, (i, team, _, _) in enumerate(od):
                B[i][0].take_reward(team)
                if sl[q] >= 0:
                    acts.setdefault(i, {})[team] = (int(sl[q]), int(ce[q]) % 18, int(ce[q]) // 18)
        for i in lanes[L]:
            drill_act(i, acts)
        return acts

    def advance(L, acts):
        for i in lanes[L]:
            b, who, kind = B[i]; b.step(acts.get(i, {}))
            if b.done:
                out = []
                for team in (0, 1):
                    tr = traj[i][team]
                    if tr:
                        tr[-1]['r'] += b.take_reward(team)
                    out.append((who[team], tr))
                done_q.put({'trajs': out, 'winner': b.winner, 'who': who, 'kind': kind, 'ticks': b.tick, 'plays': len(b.history), 'stats': b.stats, 'fams': b.fams, 'pushed': b.pushed, 'elig': b.elig, 'far': [sum(1 for h in b.history if h[1] == t and h[2] in ('GoblinBarrel', 'Miner', 'GoblinDrill') and (h[5] > 16000 if t == 0 else h[5] < 16000)) for t in (0, 1)], 'near': [sum(1 for h in b.history if h[1] == t and h[2] in ('GoblinBarrel', 'Miner', 'GoblinDrill') and not (h[5] > 16000 if t == 0 else h[5] < 16000)) for t in (0, 1)], 'drill': b.drill, 'drill_team': b.drill_team, 'drill_n': b.drill_n})
                B[i] = new_game(); traj[i] = [[], []]

    pending = [prepare(L) for L in (0, 1)]
    while True:
        for L in (0, 1):
            acts = finish(L, pending[L]) if pending[L] is not None else {}
            advance(L, acts); pending[L] = prepare(L)


# ------------------------------------------------------------------ learner
def gae(tr):
    r = np.array([t['r'] for t in tr], np.float32); v = np.array([t['v'] for t in tr], np.float32); a = np.zeros_like(r); last = 0.0
    for k in reversed(range(len(tr))):
        nv = v[k + 1] if k + 1 < len(tr) else 0.0; d = r[k] + GAMMA * nv - v[k]; last = d + GAMMA * LAMBDA * last; a[k] = last
    return a, a + v


def update(model, anchor, opt, st, trajs, dev, warm):
    from league_common import evaluate, anchor_kl
    rows, adv, ret, wts = [], [], [], []
    for tr in trajs:
        a, g = gae(tr)
        for k, t in enumerate(tr):
            if t['keep']:
                rows.append(t); adv.append(a[k]); ret.append(g[k]); wts.append(1.0 if t['c'] > 0 else 1.0 / WAIT_KEEP)
    n = len(rows)
    if not n:
        return {'raccolte': 0, 'studiate': 0}
    adv = np.array(adv, np.float32); adv = (adv - adv.mean()) / (adv.std() + 1e-8); ret = np.array(ret, np.float32); wts = np.array(wts, np.float32)
    ne = np.array([len(t['E']) for t in rows]); nh = np.array([len(t['M']) for t in rows]); perm = np.random.permutation(n); chunks = []
    for p0 in range(0, n, BS * 4):
        pool = perm[p0:p0 + BS * 4]; pool = pool[np.argsort(ne[pool] + nh[pool], kind='stable')]; cur = []; mx = 0
        for i in pool.tolist():
            c = 9 + int(ne[i]) + int(nh[i]); nm = max(mx, c)
            if cur and (len(cur) >= BS or (len(cur) + 1) * nm > TOKENS):
                chunks.append(cur); cur = []; nm = c
            cur.append(i); mx = nm
        if cur:
            chunks.append(cur)
    random.shuffle(chunks); studied = 0; S = []; K = []; beta = st['beta']; dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for j in chunks:
        L1, L2 = max(1, int(ne[j].max())), max(1, int(nh[j].max())); B = len(j); E = np.zeros((B, L1, 20), np.float32); M = np.zeros((B, L2, 7), np.float32)
        for q, i in enumerate(j):
            t = rows[i]; E[q, :len(t['E'])] = t['E']; M[q, :len(t['M'])] = t['M']
        f = lambda a, ty: torch.as_tensor(a, device=dev).to(ty)
        x = (f(E, torch.float32), f(ne[j], torch.long), f(np.stack([rows[i]['G'] for i in j]), torch.float32), f(np.stack([rows[i]['H'] for i in j]).astype(np.int64), torch.long), f(M, torch.float32), f([rows[i]['R'] for i in j], torch.long), f(np.stack([rows[i]['A'] for i in j]), torch.bool), f(np.stack([rows[i]['S'] for i in j]), torch.bool))
        ch = f([rows[i]['c'] for i in j], torch.long); ce = f([rows[i]['cell'] for i in j], torch.long); old = f([rows[i]['lp'] for i in j], torch.float32); A = f(adv[j], torch.float32); R = f(ret[j], torch.float32); W = f(wts[j], torch.float32); W = W / W.mean()
        try:
            with torch.autocast('cuda', dtype=dt):
                logp, lp, cl, v, play = evaluate(model, *x, ch, ce); lr = logp.float() - old; ratio = lr.clamp(-20, 5).exp()
                pg = -(torch.min(ratio * A, ratio.clamp(1 - CLIP, 1 + CLIP) * A) * W).mean(); vl = ((v.float() - R) ** 2 * W).mean(); kg = kc = kx = torch.zeros((), device=dev)
                if not warm:
                    q = max(16, B // 4); sub = tuple(y[:q] for y in x)
                    with torch.no_grad():
                        _, l0, c0, _, pl0 = evaluate(anchor, *sub, ch[:q], ce[:q])
                    idx = play[:q].sum().item(); c1 = cl[:idx] if cl is not None and idx else None
                    kg, kc, kx = anchor_kl(l0, lp[:q], c0 if idx else None, c1, pl0, W[:q])
                loss = (0.0 if warm else 1.0) * (pg + beta * (kg + kc + kx)) + VF * vl
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step(); studied += B
            S.append((pg.item(), vl.item())); K.append((float(kg), float(kc), float(kx)))
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); st['oom'] = st.get('oom', 0) + 1
    k = np.mean(K, 0) if K else np.zeros(3)
    if not warm:
        m = float(k.max()); st['beta'] = min(BETA_MAX, beta * 2) if m > 1.5 * KL_TARGET else max(BETA_MIN, beta / 2) if m < KL_TARGET / 1.5 else beta
    s = np.mean(S, 0) if S else np.zeros(2)
    return {'raccolte': n, 'studiate': studied, 'passi': len(S), 'pg': round(float(s[0]), 4), 'v': round(float(s[1]), 4), 'kl': [round(float(y), 4) for y in k], 'beta': beta}


def main():
    from league_common import RLB, POOL, OLD, grow
    out = sys.argv[1]; nw = int(sys.argv[2]) if len(sys.argv) > 2 else 32; nb = int(sys.argv[3]) if len(sys.argv) > 3 else 32; os.makedirs(out, exist_ok=True); dev = 'cuda'; m = json.load(open(POOL + '/meta.json')); taus = json.load(open(POOL + '/taus.json'))
    models, anchors, opts, st = [], [], [], []
    for i, p in enumerate((A_INIT, B_INIT)):
        a = RLB(len(m['vocab']), m['units'], m['g_w'], taus[str(i)]).to(dev); a.net.load_state_dict(grow(torch.load(p, map_location=dev), m['units'], m['grow'])); a.eval(); a.requires_grad_(False)
        x = RLB(len(m['vocab']), m['units'], m['g_w'], taus[str(i)]).to(dev); x.net.load_state_dict(grow(torch.load(p, map_location=dev), m['units'], m['grow'])); x.eval()
        r = os.environ.get(f'RESUME{i}')
        if r:
            x.load_state_dict(grow(torch.load(r, map_location=dev), m['units'], m['grow']))
        models.append(x); anchors.append(a); opts.append(torch.optim.AdamW(x.parameters(), lr=LR, weight_decay=0.0)); st.append({'beta': 0.03})
        tmp = f'{POOL}/m{i}.tmp'; torch.save({k: v.half() if v.is_floating_point() else v for k, v in x.state_dict().items()}, tmp); os.replace(tmp, f'{POOL}/m{i}.pt')
    man = json.load(open(POOL + '/manifest.json')) if os.environ.get('RESUME0') else {'a': [], 'b': []}; json.dump(man, open(POOL + '/manifest.tmp', 'w')); os.replace(POOL + '/manifest.tmp', POOL + '/manifest.json')
    ctx = mp.get_context('spawn'); dq = ctx.Queue(maxsize=512); P = [ctx.Process(target=worker, args=(w, nb, dq, int(os.environ.get('SEED', '41000')) + w), daemon=True) for w in range(nw)]
    for p in P:
        p.start()
    fs = {0: {}, 1: {}}
    log = open(f'{out}/train.jsonl', 'a'); t0 = time.time(); games = tot_st = 0; upd = int(os.environ.get('START_UPD', '0')); buf = []; last = t0
    while True:
        g = dq.get(); games += 1; buf.append(g)
        if len(buf) < GAMES:
            continue
        rec = {'update': upd, 'games': games}; ws = c_stats = None; tr = [[], []]
        for w in buf:
            for mid, x in w['trajs']:
                if mid in (0, 1) and x:
                    tr[mid].append(x)
        tl = time.time(); collected = studied = 0
        for i in (0, 1):
            r = update(models[i], anchors[i], opts[i], st[i], tr[i], dev, upd < WARMUP); rec['AB'[i]] = r; collected += r['raccolte']; studied += r['studiate']
            tmp = f'{POOL}/m{i}.tmp'; torch.save({k: v.half() if v.is_floating_point() else v for k, v in models[i].state_dict().items()}, tmp); os.replace(tmp, f'{POOL}/m{i}.pt')
        now = time.time(); tot_st += GAMES * (studied / max(collected, 1)); rec.update(learn_s=round(now - tl, 1), round_s=round(now - last, 1), games_h=round(GAMES * 3600 / max(now - last, 1e-3)), studied_pct=round(100 * studied / max(collected, 1), 1), studied_h=round(GAMES * (studied / max(collected, 1)) * 3600 / max(now - last, 1e-3)), avg_h=round(games * 3600 / (now - t0)), raccolte=collected, studiate=studied, oom=sum(s.get('oom', 0) for s in st)); last = now
        def score(kind, mid):
            X = [w for w in buf if w['kind'] == kind and mid in w['who']]; return [sum(w['winner'] == w['who'].index(mid) for w in X), len(X)]
        rec.update(A_vs_B=score('ab', 0), A_vs_old=score('a_old', 0), B_vs_old=score('b_old', 1), ticks=round(float(np.mean([w['ticks'] for w in buf]))), plays=round(float(np.mean([w['plays'] for w in buf])), 1))
        for i, n_ in ((0, 'A'), (1, 'B')):
            sd = [(w['stats'], t) for w in buf for t, mid in enumerate(w['who']) if mid == i]; ns = max(1, len(sd))
            rec[n_ + '_abil'] = round(sum(s['ability'][t] for s, t in sd) / ns, 2); rec[n_ + '_fb'] = round(sum(s['fireball'][t] for s, t in sd) / ns, 2); rec[n_ + '_pred'] = round(sum(s['predict'][t] for s, t in sd) / ns, 3)
            for q in ('ice_support', 'ice_gydef', 'ice_bridge', 'ice_air', 'forced', 'ice_slow', 'ice_pre', 'ice_blind', 'ice_solo', 'ice_log', 'ice_short', 'ice_long', 'ice_gy', 'open_skel', 'gy_poison', 'gy_full', 'gy_lane', 'gy_wrong', 'gy_punish', 'gy_double', 'front', 'barb_ok', 'hut_eg', 'gy_time', 'gy_early', 'gy_far', 'gy_counter', 'hut_pos', 'hut_threat', 'hut_idle', 'fur_air', 'poi_ok', 'stack', 'gy', 'gy_ok', 'gy_bad', 'poi_bad', 'splash', 'iw', 'iw_fast', 'hut_ok', 'gb_evo', 'gb_evo_right', 'gb_evo_wrong', 'gb_waste', 'gb_can', 'gb_kill_can', 'gb_seen', 'gb_log', 'gb_kill', 'gb_early', 'gb_late', 'gb_paid', 'predict_paid', 'turret_paid', 'turret', 'turret_ok', 'turret_bad', 'ice', 'ice_ok', 'ice_bad', 'zap', 'zap_ok', 'zap_bad'):
                rec[n_ + '_' + q] = round(sum(s[q][t] for s, t in sd) / ns, 3)
        for w in buf:
            for t, mid in enumerate(w['who']):
                f = w['fams'][1 - t]
                if mid in (0, 1) and f:
                    x = fs[mid].setdefault(f, [0.0, 0.0]); x[1] += 1; x[0] += w['winner'] == t
        H = {str(i): {f: {'wr': (x[0] + 1) / (x[1] + 2), 'n': x[1]} for f, x in fs[i].items()} for i in (0, 1)}
        json.dump(H, open(POOL + '/hard.tmp', 'w')); os.replace(POOL + '/hard.tmp', POOL + '/hard.json')
        for i in (0, 1):
            for x in fs[i].values():
                x[0] *= 0.98; x[1] *= 0.98
        el = sum(w['elig'] for w in buf); rec['spinte_pct'] = round(100 * sum(w['pushed'] for w in buf) / max(el, 1), 1)
        for i, n_ in ((0, 'A'), (1, 'B')):
            rec[n_ + '_dure'] = [(f, round(v['wr'], 2)) for f, v in sorted(H[str(i)].items(), key=lambda y: y[1]['wr']) if v['n'] >= HARD_MIN][:3]
        for i, n_ in ((0, 'A'), (1, 'B')):
            by = {}
            for w in buf:
                for t, mid in enumerate(w['who']):
                    if mid == i:
                        x = by.setdefault(w['kind'], [0, 0, 0, 0, 0]); x[0] += w['stats']['predict'][t]; x[1] += w['stats']['fireball'][t]; x[2] += 1; x[3] += w['stats']['turret'][t]; x[4] += w['stats']['turret_ok'][t]
            rec[n_ + '_pred_by'] = by
        rec['far'] = sum(w['far'][t] for w in buf for t, mid in enumerate(w['who']) if mid in (0, 1)); rec['near'] = sum(w['near'][t] for w in buf for t, mid in enumerate(w['who']) if mid in (0, 1))
        dg = [w for w in buf if w['drill']]
        rec['drill'] = {'partite': len(dg), 'edifici': sum(w['drill_n'] for w in dg), 'fb': sum(w['stats']['fireball'][1 - w['drill_team']] for w in dg), 'pred': sum(w['stats']['predict'][1 - w['drill_team']] for w in dg)}
        upd += 1
        if upd % SNAP_EVERY == 0:
            for i, k in enumerate('ab'):
                sid = (1000 if i == 0 else 2000) + upd; tmp = f'{POOL}/s{sid}.tmp'; torch.save({a: v.half() if v.is_floating_point() else v for a, v in models[i].state_dict().items()}, tmp); os.replace(tmp, f'{POOL}/p_{sid}.pt'); man[k].append(sid)
                if len(man[k]) > SNAP_KEEP:
                    d = man[k].pop(0)
                    try:
                        os.remove(f'{POOL}/p_{d}.pt')
                    except OSError:
                        pass
            json.dump(man, open(POOL + '/manifest.tmp', 'w')); os.replace(POOL + '/manifest.tmp', POOL + '/manifest.json')
        for i, k in enumerate('AB'):
            torch.save(models[i].state_dict(), f'{out}/{k}_latest.tmp'); os.replace(f'{out}/{k}_latest.tmp', f'{out}/{k}_latest.pt')
            if upd % 100 == 0:
                torch.save(models[i].state_dict(), f'{out}/{k}_{upd:05d}.pt')
        log.write(json.dumps(rec) + '\n'); log.flush(); print(json.dumps(rec), flush=True); buf = []


if __name__ == '__main__':
    main()
