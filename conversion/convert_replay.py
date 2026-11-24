"""Generalist model B: any replay -> RoyaleSim states + the WINNER's decisions (cards AND ability presses).

The winner is always team 0 (bottom); if it was native owner 1 the arena is mirrored. Each state keeps what the engine
really knows: unit type (a Barbarian out of a Battle Ram is not a Ram), evolved / hero flags, stun, buffs, spells in flight,
projectiles, Evo counters and ability buttons. The match history is stored once per replay, a sample only keeps its length.
"""
import gzip, json, os, pickle, sys, time, collections as c
if os.environ.get('NATIVE_RUNTIME'):
    sys.path.append(os.environ['NATIVE_RUNTIME'])   # folder holding the V4 framework's native_runner; AFTER site-packages

WAIT_EVERY = 10
MAX_TICKS = 7200
OFFSET = int(os.environ.get('OFFSET', '0'))    # the replay's tick is already the tick the card lands
DELAY = int(os.environ.get('DELAY', '20'))
ABILITY_SLOT = 4
TOWER_TROOP = {'tower-princess': None, 'dagger-duchess': 'DaggerDuchess', 'cannoneer': 'Cannoneer', 'royal-chef': 'ChefTower'}
MAX_RETRY = 60
START_ELIXIR = int(os.environ.get('START_ELIXIR', '6000'))   # the real engine starts at 6.0
REGULAR_S = os.environ.get('REGULAR_S', '')
FULL_END = os.environ.get('FULL_END', '1') == '1'   # play on until the engine ends the match


def base_key(k):
    return k.removesuffix('-ev1').removesuffix('-hero')


def form_of(k):
    return 1 if k.endswith('-ev1') else 2 if k.endswith('-hero') else 0


def init_worker():
    global royalesim, prepare, SUB, NAME_OF, KEY_TO_ID, resolve, EF
    import royalesim as _r
    from native_runner.royaleapi_replay import prepare_collected_replay, resolve_native_card
    from native_runner.card_specs import build_card_catalog
    royalesim = _r; prepare = prepare_collected_replay; resolve = resolve_native_card; SUB = _r.SUBTILE_PER_MILLITILE
    NAME_OF = {int(s['card_id']): s['name'] for s in build_card_catalog().to_dict()['specs']}
    KEY_TO_ID = {}
    EF = {n: i for i, n in enumerate(_r.ENTITY_FIELDS)}


def kid(key):
    if key not in KEY_TO_ID:
        KEY_TO_ID[key] = resolve(base_key(key)).card_id
    return KEY_TO_ID[key]


def snapshot(b):
    st = json.loads(b.state_json())
    f = EF
    ents = [(e[f['uid']], e[f['team']], e[f['kind']], e[f['card_id']], e[f['x']] // SUB, e[f['y']] // SUB, e[f['hp']], e[f['max_hp']],
             e[f['flying']], e[f['deploy_ticks']], e[f['stun_ticks']], e[f['knockback_ticks']], e[f['attack_phase']], e[f['status_flags']] or 0,
             e[f['unit_type']], e[f['shield']] or 0, e[f['charge']] or 0, len(e[f['buffs']] or ())) for e in st['entities']]
    spells = [(s[0], s[1], s[5] // SUB, s[6] // SUB, s[7]) for s in st['spells']]
    pl = [(p['elixir_milli'], tuple(p['hand']), p['next_card'], tuple(p['tower_hp']), tuple(map(tuple, p['evo'])), tuple(map(tuple, p['abilities'])))
          for p in st['players']]
    return {'tick': st['tick'], 'ents': ents, 'spells': spells, 'players': pl, 'overtime': st['overtime'], 'rate': st['elixir_rate']}


def convert(item, keep_samples=True):
    """item = (payload_json text, winner side: 0 team / 1 opponent)."""
    text, win_side = item
    p = json.loads(text)
    try:
        pr = prepare(p)
    except Exception as e:
        return {'error': f'prepare: {type(e).__name__}'}
    b_ = p['battle']
    sides = {0: b_['team']['players'][0], 1: b_['opponent']['players'][0]}
    ep = pr.replay.episode_config
    ids = {s: {kid(x['card_key']) for x in sides[s]['deck']} for s in (0, 1)}
    if set(ep.deck0) == ids[0] and set(ep.deck1) == ids[1]:
        side_of = {0: 0, 1: 1}
    elif set(ep.deck0) == ids[1] and set(ep.deck1) == ids[0]:
        side_of = {0: 1, 1: 0}
    else:
        return {'error': 'decks do not match'}
    me_owner = next(o for o, s in side_of.items() if s == win_side)
    decks, forms, levels, tlev = [], [], [], []
    for owner in (0, 1):
        pl = sides[side_of[owner]]
        order = list(pr.opening_cards[owner]) + list(pr.queue_cards[owner])
        by_id = {kid(x['card_key']): x for x in pl['deck']}
        decks.append([NAME_OF[i] for i in order]); forms.append([form_of(by_id[i]['card_key']) for i in order])
        levels.append([int(by_id[i]['level']) for i in order]); tlev.append(int((pl.get('tower_card') or {}).get('level') or levels[-1][0]))
    perm = (me_owner, 1 - me_owner)
    decks = [decks[o] for o in perm]; forms = [forms[o] for o in perm]; levels = [levels[o] for o in perm]; tlev = [tlev[o] for o in perm]
    names = list(dict.fromkeys(decks[0] + decks[1]))
    ov = {'match.REGULAR_TIME_S': REGULAR_S} if REGULAR_S else None
    b = royalesim.Battle(card_names=names, slot_of_k=[[0, 1, 2]] * 2, calibration_overrides=ov)
    tt = [TOWER_TROOP.get((sides[win_side].get('tower_card') or {}).get('card_key')), TOWER_TROOP.get((sides[1 - win_side].get('tower_card') or {}).get('card_key'))]
    b.set_tower_troops(tt[0], tt[1])
    tg = dict(ep.tags)                                   # the King Tower has its own level
    kl = [tg.get(f'king_tower{me_owner}_level'), tg.get(f'king_tower{1 - me_owner}_level')]
    pl_ = [tg.get(f'tower_troop{me_owner}_level'), tg.get(f'tower_troop{1 - me_owner}_level')]
    tlev = [int(pl_[i]) if pl_[i] is not None else tlev[i] for i in (0, 1)]
    b.set_king_levels(int(kl[0]) if kl[0] is not None else None, int(kl[1]) if kl[1] is not None else None)
    b.reset(ep.seed % (1 << 30), [[names.index(x) for x in decks[0]], [names.index(x) for x in decks[1]]], 0, 0,
            [START_ELIXIR, START_ELIXIR], None, [], forms=forms, levels=levels, tower_levels=tlev)
    b.set_command_delay_ticks(0, 0)
    plays = {}
    t = 0
    dt_card = c.Counter()
    planned = {0: [], 1: []}
    for op in pr.replay.operations:
        for a in op.actions:
            t_exec = t + a.execute_offset_ticks + OFFSET
            team = 0 if a.owner == me_owner else 1
            if a.kind.value == 'play_card':
                gx, gy = a.target_grid; ox, oy = a.subcell_offset or (0.0, 0.0)
                x = (gx + 0.5 + ox) * 1000; y = (gy + 0.5 + oy) * 1000
                if me_owner == 1:
                    x, y = 18000 - x, 32000 - y
                plays.setdefault(t_exec, []).append(('card', team, a.hand_slot, int(x), int(y), a.card_id))
                planned[team].append(NAME_OF.get(int(a.card_id)) if a.card_id is not None else None)
                sc = (a.metadata or {}).get('source_command_tick')
                if sc is not None:
                    dt_card[t + a.execute_offset_ticks - int(sc)] += 1
            elif a.kind.value == 'activate_ability':
                who = tuple((a.metadata or {}).get('ability_source_keys') or ())
                plays.setdefault(t_exec, []).append(('ability', team, who))
        t += max((a.next_decision_ticks for a in op.actions), default=0)
    stats = c.Counter(); ab = c.Counter(); samples = []; history = []
    end = MAX_TICKS if FULL_END else min(MAX_TICKS, max(plays) + 40 if plays else 0)
    last_play = max(plays) if plays else 0
    my_ticks = sorted(u for u, v in plays.items() if any(x[1] == 0 for x in v)); mt = set(my_ticks)
    decisions = {u: 'act' for u in my_ticks}
    for u in range(DELAY, min(end, last_play + 40) + 1, WAIT_EVERY):
        if not any(v in mt for v in range(u - 5, u + 6)):
            decisions.setdefault(u, 'wait')
    need = {u - DELAY: u for u in decisions if u - DELAY >= 0}
    queue = [c.deque(), c.deque()]
    snaps = {}; key_of = {n: base_key(x['card_key']) for s in (0, 1) for x in sides[s]['deck'] for n in [NAME_OF[kid(x['card_key'])]]}
    for tick in range(end + 1):
        if keep_samples and tick in need:
            snaps[need[tick]] = snapshot(b)
        for x in plays.get(tick, []):
            queue[x[1]].append((tick, x))
        cmds, meta = [], []
        st_now = json.loads(b.state_json()) if queue[0] or queue[1] else None
        for team in (0, 1):
            while queue[team] and tick - queue[team][0][0] > MAX_RETRY:      # could not be played in time: give up on it
                o_, x_ = queue[team].popleft()
                if x_[0] == 'card':
                    stats['carte_perse'] += 1
                else:
                    ab[('+'.join(sorted(x_[2])) or '?', 'non pronto')] += 1
            if not queue[team]:
                continue
            origin, x = queue[team][0]
            pl_ = st_now['players'][team]
            if x[0] == 'card':
                want = NAME_OF.get(int(x[5])) if x[5] is not None else None
                hh = [names[h] for h in pl_['hand']]
                slot = hh.index(want) if want in hh else (x[2] if want is None else None)
                if slot is None:
                    stats['attesa_carta'] += 1; continue       # the card is not back in hand yet: keep the queue order
                px, py = x[3], x[4]
                why = royalesim.DEPLOY_REASONS[b.check_deploy(team, slot, px * SUB, py * SUB)]
                if why in ('OUT_OF_TERRITORY', 'NO_DEPLOY', 'WATER', 'OCCUPIED'):     # the real tower was down, here it is not
                    for qy in ((min(py, 14500), 14000, 13500, 12500) if team == 0 else (max(py, 17500), 18000, 18500, 19500)):
                        if royalesim.DEPLOY_REASONS[b.check_deploy(team, slot, px * SUB, qy * SUB)] == 'OK':
                            py = qy; why = 'OK'; stats['posizione_corretta'] += 1; break
                if why != 'OK':
                    stats['attesa_' + why] += 1; continue
                cmds.append((team, slot, px * SUB, py * SUB)); meta.append(('card', team, slot, origin))
            else:
                rows = pl_['abilities']; want = {base_key(k) for k in x[2]}
                cand = [k for k, r in enumerate(rows) if 0 <= r[3] < len(names) and key_of.get(names[r[3]]) in want]
                ready = [k for k in cand if rows[k][0] and rows[k][2] * 1000 <= pl_['elixir_milli']]
                if not ready:
                    continue
                cmds.append((team, ABILITY_SLOT + ready[0], 0, 0)); meta.append(('ability', team, ready[0], origin, '+'.join(sorted(x[2])) or '?'))
        if keep_samples and decisions.get(tick) in ('act', 'wait') and tick in snaps:
            s = snaps.pop(tick); now = snapshot(b)
            s['players'] = [now['players'][0], s['players'][1]]; s['tick'] = now['tick']
            s['n_hist'] = sum(1 for h in history if h[1] == 0 or h[0] <= tick - DELAY)
            mine = [(m, q) for m, q in zip(meta, cmds) if m[1] == 0 and m[3] == tick]
            if mine:
                m, q = mine[0]
                s['action'] = ('card', m[2], q[2] // SUB // 1000, q[3] // SUB // 1000) if m[0] == 'card' else ('ability', m[2])
                samples.append(s)
            elif decisions[tick] == 'wait':
                s['action'] = None; samples.append(s)
        res = b.step(cmds, 1) if cmds else b.step([], 1)
        for m, q, r in zip(meta, cmds, res or []):
            ok = r[1] == 0
            if ok:
                queue[m[1]].popleft()
            if m[0] == 'card':
                stats['carte_ok' if ok else 'carte_rifiutate'] += 1
                if not ok:
                    stats['rif_' + royalesim.DEPLOY_REASONS[r[1]]] += 1
                if ok:
                    h = st_now['players'][m[1]]['hand'][m[2]]
                    fm = 0
                    for row in st_now['players'][m[1]]['evo']:
                        if row[0] == h and row[2]:
                            fm = 1
                    if forms[m[1]][decks[m[1]].index(names[h])] == 2:
                        fm = 2
                    history.append((tick, m[1], names[h], fm, q[2] // SUB, q[3] // SUB))
            else:
                if ok:
                    ab[(m[4], 'ok')] += 1
        if tick % 20 == 0 and json.loads(b.state_json())['game_over']:
            break
    stats['carte_perse'] += sum(1 for q_ in queue for _, x_ in q_ if x_[0] == 'card')
    final = json.loads(b.state_json())
    got = {k: [h[2] for h in history if h[1] == k] for k in (0, 1)}
    seq_ok = all(got[k] == planned[k][:len(got[k])] for k in (0, 1))
    seq_missing = max(len(planned[k]) - len(got[k]) for k in (0, 1))
    real = [sides[win_side], sides[1 - win_side]]
    rt = [int((x.get('final_tower_hitpoints') or {}).get('total', 0)) for x in real]
    rc = (int(b_['team'].get('crowns', 0) or 0), int(b_['opponent'].get('crowns', 0) or 0))
    check = {'sim_hp': [sum(max(0, v) for v in q['tower_hp']) for q in final['players']], 'real_hp': rt, 'sim_crowns': [q['crowns'] for q in final['players']],
             'real_crowns': [rc[win_side], rc[1 - win_side]], 'tower_troops': [(x.get('tower_card') or {}).get('card_key') for x in real], 'end_tick': final['tick']}
    return {'tag': pr.replay_tag, 'samples': samples, 'history': history, 'decks': decks, 'forms': forms, 'names': names,
            'stats': dict(stats), 'ability': dict(ab), 'check': check, 'seq_ok': seq_ok, 'seq_missing': seq_missing, 'planned': planned, 'dt_card': dict(dt_card), 'side_flip': side_of[0], 'sim_winner': final['winner'], 'sim_over': final['game_over']}
