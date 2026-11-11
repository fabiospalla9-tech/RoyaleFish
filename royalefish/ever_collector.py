"""Raccoglitore Everlight Turbo (100% partite reali sul motore nativo x86):
- A gioca sempre Hog 2.6 (Everlight RL 134 live PPO).
- B gioca 90% Everlight v15 general sui ~165 mazzi pesati via PFSP (piu' frequenti i counter dove A perde!)
  e 10% Mirror Self-Play Hog 2.6 contro A stesso.
- Registra solo le traiettorie vere di A in formato sparse lossless uint64 (zero zlib, zero doppio pack).
- Supporta fino a 16 partite simultanee per motore x86."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace as _replace
import json
import pickle
import queue
import random
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
from native_runner.cr_native_env import ResidentNativeClashEnv  # noqa: E402
from native_runner.resident_batch_vector import ResidentBatchCoordinatorV1, ResidentBatchNativeProxyV1  # noqa: E402
from native_runner.training.v4.ppo_runtime import ppo_environment_config_v4, build_ppo_environment_v4, episode_for_matchup_v4  # noqa: E402
from native_runner.training.v4.deck_pool import DeckPoolManifestV4  # noqa: E402
from native_runner.training.v4.matchmaking import ScheduledMatchupV4, POLICY_CURRENT  # noqa: E402
from native_runner.training.v4.factory import build_episode_tensorizer_v4  # noqa: E402
from native_runner.training.v4.policy_session import _fixed_cuda_graph_batch_v4  # noqa: E402
from native_runner.training.v4.decoding import decode_action_sequence_v4  # noqa: E402
from native_runner.training.v4.expert import POLICY_DECISION_TICKS  # noqa: E402
from native_runner.training.v4.tensors import ActionSequenceV4  # noqa: E402
from native_runner.contracts import ObservationTier, ActionV1  # noqa: E402
from royalefish.config import POOL, BATTLE_LEVEL, LEARNER_DECK_ID, GAMEPLAY_END_TICK, BASE_LATENCY_TICKS  # noqa: E402
from royalefish.config import RESET  # noqa: E402
from royalefish.net import send_msg, recv_msg  # noqa: E402
from royalefish.teacher_pack import FastPacker, pack_episode_sparse  # noqa: E402
from royalefish.reward import RewardAdapter  # noqa: E402

SLIM_EFFECT_CAP = 32
SLIM_RELATION_CAP = 192

_WC = ['hog-rider', 'royal-giant', 'golem', 'lava-hound', 'giant', 'balloon', 'x-bow', 'mortar', 'graveyard', 'goblin-barrel',
       'miner', 'ram-rider', 'battle-ram', 'royal-hogs', 'elixir-golem', 'electro-giant', 'goblin-giant', 'three-musketeers',
       'mega-knight', 'pekka', 'sparky', 'wall-breakers', 'skeleton-barrel', 'goblin-drill', 'giant-skeleton', 'royal-recruits']


class Link:
    """Collegamento veloce a un ever_server (riceve pacchetti gia' pronti, zero ricalcolo layout/pack!)."""
    def __init__(self, hostport):
        self.hostport = hostport
        self.sock = None
        self.sent = set()
        self._connect()

    def _connect(self):
        h, p = self.hostport.rsplit(':', 1)
        s = socket.create_connection((h, int(p)), timeout=300.)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s
        self.sent = set()

    def act(self, items, layouts_meta, end=()):
        """items: [(key, start, sig, (ib, vb))]"""
        for attempt in range(2):
            try:
                for sig, (tmpl, lay_items, total) in layouts_meta.items():
                    if sig not in self.sent:
                        send_msg(self.sock, {'layout': (sig, tmpl, lay_items, total)})
                        recv_msg(self.sock)
                        self.sent.add(sig)
                send_msg(self.sock, {'items': items, 'end': list(end)})
                return recv_msg(self.sock)
            except (ConnectionError, EOFError, OSError):
                if attempt == 1:
                    raise
                time.sleep(0.5)
                self._connect()


def _legal_play(x, full, owner):
    """La giocata va esattamente dove la vuole il modello; se casella non permessa dal regolamento -> None."""
    if getattr(x.kind, 'name', '') != 'PLAY_CARD' or x.target_grid is None:
        return x
    entry = full.action_mask.placement_masks.get(str(x.hand_slot))
    if entry is None or not full.action_mask.hand_slots[x.hand_slot]:
        return None
    g = np.asarray(entry['row_major'], dtype=bool)
    if not g.any():
        return None
    tx, ty = int(x.target_grid[0]), int(x.target_grid[1])
    if not (0 <= ty < g.shape[0] and 0 <= tx < g.shape[1] and g[ty, tx]):
        return None
    sub = entry.get('model_subcell_offset')
    if sub is not None:
        sub = (-float(sub[0]), -float(sub[1])) if owner == 1 else (float(sub[0]), float(sub[1]))
    return _replace(x, target_grid=(tx, ty), subcell_offset=sub, card_id=int(entry.get('visible_card_id', entry['card_id'])))


def _act_from_bytes(act_bytes):
    arr = np.frombuffer(act_bytes, dtype=np.int64)
    return arr, ActionSequenceV4(
        gate=torch.from_numpy(arr[0:1].copy()),
        micro_action_count=torch.from_numpy(arr[1:2].copy()),
        candidate_index=torch.from_numpy(arr[2:4].reshape(1, 2).copy()),
        candidate_uid=torch.from_numpy(arr[4:6].reshape(1, 2).copy()),
        target_cell=torch.from_numpy(arr[6:8].reshape(1, 2).copy()),
        delay_offset_bin=torch.from_numpy(arr[8:10].reshape(1, 2).copy()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--index', type=int, required=True)
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--slots', type=int, default=16)
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--hog-frac', type=float, default=0.18)
    ap.add_argument('--mirror-frac', type=float, default=0.10)
    ap.add_argument('--record', type=float, default=0.30)
    ap.add_argument('--record-b', type=float, default=1.0)   # anche B impara
    ap.add_argument('--eval-mirror', action='store_true')   # test: B gioca lo stesso mazzo Hog 2.6 di A
    args = ap.parse_args()

    run = args.run
    (run / 'collectors').mkdir(parents=True, exist_ok=True)
    log = open(run / 'collectors' / f'e{args.index:03d}.jsonl', 'a', buffering=1)
    rng = random.Random(1000 + args.index * 7919 + int(time.time() * 1000) % 1000000)

    decks = list(DeckPoolManifestV4.load(POOL).decks)
    hog_deck = next(d for d in decks if d.deck_id == LEARNER_DECK_ID)
    by_wc = {}
    for d in decks:
        keys = [c.split('-ev')[0].replace('-hero', '') for c in d.card_keys]
        by_wc.setdefault(next((w for w in _WC if w in keys), 'altro'), []).append(d)
    hog_decks = by_wc.pop('hog-rider', [])
    wc_names = sorted(by_wc)

    pfsp_state = {'mtime': 0., 'weights': {}, 'last_check': 0.}

    def refresh_pfsp():
        now = time.monotonic()
        if now - pfsp_state['last_check'] < 30.:
            return
        pfsp_state['last_check'] = now
        f = run / 'pfsp_decks.json'
        try:
            mt = f.stat().st_mtime
            if mt > pfsp_state['mtime']:
                pfsp_state['weights'] = json.loads(f.read_text()).get('weights', {})
                pfsp_state['mtime'] = mt
        except Exception:
            pass

    def pick_weighted(deck_list):
        wmap = pfsp_state['weights']
        if not wmap or len(deck_list) == 1:
            return rng.choice(deck_list)
        ws = [float(wmap.get(d.deck_id, 1.0)) for d in deck_list]
        return rng.choices(deck_list, weights=ws, k=1)[0]

    def pick_matchup():
        if args.eval_mirror:
            return hog_deck, 'B', False
        refresh_pfsp()
        if rng.random() < args.mirror_frac:
            return hog_deck, 'A', True
        if hog_decks and rng.random() < args.hog_frac:
            return pick_weighted(hog_decks), 'B', False
        # PFSP anche sulla scelta della famiglia di win-condition
        wmap = pfsp_state['weights']
        if wmap:
            fam_w = [max(0.2, float(np.mean([float(wmap.get(d.deck_id, 1.0)) for d in by_wc[w]]))) for w in wc_names]
            fam = rng.choices(wc_names, weights=fam_w, k=1)[0]
        else:
            fam = rng.choice(wc_names)
        return pick_weighted(by_wc[fam]), 'B', False

    slots = list(range(args.slots))
    natives = {i: ResidentNativeClashEnv(i, '127.0.0.1', args.port, timeout=120.) for i in slots}
    coord = ResidentBatchCoordinatorV1(tuple(natives.values()), timeout=120., training_capture=True, compressed_json=False, validate_json_capture=False)
    cfg = ppo_environment_config_v4(gamma_per_decision=0.9993, shaping_beta=0.)
    envs = {i: build_ppo_environment_v4(ResidentBatchNativeProxyV1(n, coord), cfg) for i, n in natives.items()}
    for env in envs.values():
        env.simulated_card_delay_jitter_ticks = 3
        env.prepare_runner_attestation()

    hlinks = [
        {'A': Link(args.a), 'B': Link(args.b)},
        {'A': Link(args.a), 'B': Link(args.b)},
    ]
    pending_end = [{'A': [], 'B': []}, {'A': [], 'B': []}]
    pool = ThreadPoolExecutor(2)
    writes = queue.Queue(maxsize=32)

    def writer():
        while True:
            mdl, b, result, ver, deck_b_id, is_mirror, name = writes.get()
            inbox = run / mdl / 'inbox'
            inbox.mkdir(parents=True, exist_ok=True)
            try:
                # Limite anti-intasamento disco: se ci sono gia' > 96 file freschi, salta i piu' vecchi
                existing = list(inbox.glob('*.npz'))
                if len(existing) > 96:
                    existing.sort(key=lambda p: p.stat().st_mtime)
                    for old_p in existing[:-72]:
                        old_p.unlink(missing_ok=True)
                offsets, idx_u16, val_u64 = pack_episode_sparse(b['rows'])
                tmp = inbox / (name + '.tmp')
                with open(tmp, 'wb') as f:
                    np.savez(
                        f,
                        offsets=offsets,
                        idx_u16=idx_u16,
                        val_u64=val_u64,
                        act=np.stack(b['act']),
                        lp=np.asarray(b['lp'], np.float32),
                        val=np.asarray(b['val'], np.float32),
                        rew=np.asarray(b['rew'], np.float32),
                        meta=np.frombuffer(json.dumps(dict(
                            sparse=1, version=ver, result=result, deck_b=deck_b_id, mirror=is_mirror,
                            items=b['lay'][0], total=b['lay'][1], sig=b['lay'][2],
                        )).encode(), np.uint8),
                        tmpl=np.frombuffer(pickle.dumps(b['tmpl']), np.uint8),
                    )
                os.replace(tmp, inbox / (name + '.npz'))
            except Exception as e:
                print('scrittura fallita', repr(e)[:150], flush=True)

    threading.Thread(target=writer, daemon=True).start()

    packers = {}  # (eff_cap, rel_cap) -> (FastPacker, template_fixed)
    games = {}
    serial = [0]
    stats = dict(steps=0, t_feat=0., t_model=0., t_native=0., done=0, t0=time.monotonic())

    def reset(slot):
        serial[0] += 1
        owner_a = rng.randrange(2)
        deck_b, srv_b, is_mirror = pick_matchup()
        d0, d1 = (hog_deck, deck_b) if owner_a == 0 else (deck_b, hog_deck)
        m = ScheduledMatchupV4(
            sequence=serial[0], category='focus-focus', policy_matchup=POLICY_CURRENT,
            deck0=d0, deck1=d1, current_owners=(0, 1), battle_level=BATTLE_LEVEL,
            seed=rng.randrange(1, 2**31 - 1),
        )
        ep = episode_for_matchup_v4(m, cfg)
        obs0, _ = envs[slot].reset(ep, options=RESET)
        tz = {}
        for o in (0, 1):
            full = envs[slot].observe(ObservationTier.FAIR, owner=o)
            mine = next((float(p.elixir_exact) for p in full.players if p.owner == o and p.elixir_exact is not None), 5.)
            tz[o] = build_episode_tensorizer_v4(ep, actor_owner=o)
            tz[o].start_episode(full, initial_elixir={0: mine, 1: mine})
        srv_map = {owner_a: 'A', 1 - owner_a: srv_b}
        rec_a = (rng.random() < args.record)
        rec_b = (not is_mirror) and (rng.random() < args.record_b)
        games[slot] = dict(
            tz=tz,
            owner_a=owner_a,
            models={owner_a: 'A', 1 - owner_a: ('A_mirror' if is_mirror else 'B')},
            srv=srv_map,
            mirror=is_mirror,
            start={0: True, 1: True},
            key={o: (args.index, slot, serial[0], o) for o in (0, 1)},
            deck_b=deck_b.deck_id,
            steps=0,
            plays=[0, 0],
            started=time.time(),
            ver={},
            obs=obs0,
            reward=RewardAdapter(envs[slot], obs0, {}),
            rec={owner_a: rec_a, 1 - owner_a: rec_b},
            buf={o: dict(rows=[], tmpl=None, lay=None, act=[], lp=[], val=[], rew=[]) for o in (0, 1)},
        )

    try:
        coord.close_session()
    except Exception:
        pass
    for s in slots:
        reset(s)
    coord.start_session()

    halves = [slots[:len(slots) // 2], slots[len(slots) // 2:]]

    def featurize(h):
        t0 = time.monotonic()
        req = {'A': [], 'B': []}
        lmeta = {'A': {}, 'B': {}}
        ctx = {}
        for s in halves[h]:
            g = games[s]
            for o in (0, 1):
                full = envs[s].observe(ObservationTier.FAIR, owner=o)
                hb = g['tz'][o].tensorize(full, validate=False)
                if hb.active_effects.mask.shape[1] <= SLIM_EFFECT_CAP and hb.relation_edges.mask.shape[1] <= SLIM_RELATION_CAP:
                    cap_key = (SLIM_EFFECT_CAP, SLIM_RELATION_CAP)
                    fixed = _fixed_cuda_graph_batch_v4(hb, effect_capacity=SLIM_EFFECT_CAP, relation_capacity=SLIM_RELATION_CAP)
                else:
                    cap_key = (512, 2048)
                    fixed = _fixed_cuda_graph_batch_v4(hb)
                pk_entry = packers.get(cap_key)
                if pk_entry is None:
                    pk_entry = (FastPacker(fixed), fixed)
                    packers[cap_key] = pk_entry
                packer, tmpl_fixed = pk_entry
                sig, ib, vb = packer.pack_sparse(fixed)
                srv = g['srv'][o]
                req[srv].append((g['key'][o], g['start'][o], sig, (ib, vb)))
                if sig not in lmeta[srv]:
                    lmeta[srv][sig] = (tmpl_fixed, packer.items, packer.total)
                if g['rec'][o]:
                    b = g['buf'][o]
                    if b['lay'] is None:
                        b['lay'] = packer.lay
                        b['tmpl'] = tmpl_fixed
                    if sig != b['lay'][2]:
                        g['rec'][o] = False
                    else:
                        b['rows'].append((ib, vb))
                ctx[(s, o)] = (full, hb)
                g['start'][o] = False
        stats['t_feat'] += time.monotonic() - t0
        futs = {}
        for srv in ('A', 'B'):
            ends = pending_end[h][srv]
            if req[srv] or ends:
                end_copy = list(ends)
                ends.clear()
                futs[srv] = pool.submit(hlinks[h][srv].act, req[srv], lmeta[srv], end_copy)
        return futs, ctx

    def act_and_step(h, futs, ctx):
        t1 = time.monotonic()
        res = {k: f.result() for k, f in futs.items()}
        t2 = time.monotonic()
        stats['t_model'] += t2 - t1
        commands = {}
        pos = {'A': 0, 'B': 0}
        for s in halves[h]:
            g = games[s]
            commands[s] = {}
            for o in (0, 1):
                srv = g['srv'][o]
                r = res[srv][pos[srv]]
                pos[srv] += 1
                full, hb = ctx[(s, o)]
                tz = g['tz'][o]
                wait = (ActionV1.wait(o, ticks=POLICY_DECISION_TICKS),)
                if r is None:
                    g['rec'][o] = False
                    commands[s][o] = wait
                    continue
                act_bytes, logp, val, ver = r
                g['ver'][g['models'][o]] = ver
                arr10, acts = _act_from_bytes(act_bytes)
                if g['rec'][o]:
                    b = g['buf'][o]
                    b['act'].append(arr10.copy())
                    b['lp'].append(logp)
                    b['val'].append(val)
                tz.record_action(acts, hb, row=0, validate=False)
                dec = decode_action_sequence_v4(
                    acts, hb.candidates, row=0, observation=full, catalog=tz.catalog, deck=tz.deck,
                    card_costs=tz.card_costs, ability_id_by_vocab_id=tz.ability_id_by_vocab_id,
                    horizontal_mirror=tz.perspective.horizontal_mirror,
                    hand_slot_permutation=tz.perspective.hand_slot_permutation, config=tz.config,
                    base_latency_ticks=BASE_LATENCY_TICKS, base_latency_ms=0.0, validate=False,
                )
                cmd = tuple(
                    y for y in (
                        _legal_play(
                            _replace(x, action_id=f"ever-{args.index}-{s}-{g['key'][o][2]}-{o}-{g['steps']}-{k}",
                                     next_decision_ticks=POLICY_DECISION_TICKS),
                            full, o,
                        )
                        for k, x in enumerate(dec.actions)
                    )
                    if y is not None
                )
                if any(getattr(x.kind, 'name', '') != 'WAIT' for x in cmd):
                    g['plays'][o] += 1
                commands[s][o] = cmd or wait

        step_states = {}
        expected = {}
        payload = {}
        bt = None
        for s, c in commands.items():
            env = envs[s]
            try:
                state = env._step_prepare(c, record_trace=False, policy_features_only=True)
            except Exception:
                stats['rej'] = stats.get('rej', 0) + 1
                c = {o: (ActionV1.wait(o, ticks=POLICY_DECISION_TICKS),) for o in (0, 1)}
                state = env._step_prepare(c, record_trace=False, policy_features_only=True)
            env._last_advance_event = False
            ticks = int(state[5])
            bt = ticks if bt is None else bt
            aq, eq = env.native.take_queued_step()
            step_states[s] = state
            expected[s] = eq
            payload[s] = aq

        outcomes = coord.step_direct(payload, ticks=bt)
        finished = []
        for s, state in step_states.items():
            env = envs[s]
            env.native.apply_step_result(expected[s], outcomes[s])
            after, pre = envs[s]._atomic_observation_pair()
            nxt, _, ended, trunc, _ = env._step_complete(state, after, pre)
            g = games[s]
            g['steps'] += 1
            try:
                active = int(g['obs'][0].tick) < GAMEPLAY_END_TICK
                values, replace_last = g['reward'].transition(g['obs'], nxt, active)
            except Exception:
                values, replace_last = None, False
                g['rec'] = {0: False, 1: False}
            for oo in (0, 1):
                if not g['rec'][oo]:
                    continue
                rw = g['buf'][oo]['rew']
                if values is None:
                    rw.append(0.)
                elif replace_last and rw:
                    rw[-1] = float(values[oo])
                else:
                    rw.append(float(values[oo]))
            g['obs'] = nxt
            if ended['__all__'] or trunc['__all__']:
                finished.append((s, nxt))

        if finished:
            coord.close_session()
            for s, nxt in finished:
                g = games[s]
                for o in (0, 1):
                    pending_end[h][g['srv'][o]].append(g['key'][o])
                res_o = [float(nxt[o].terminal.result_by_owner[o]) if nxt[0].terminal.ended else 0. for o in (0, 1)]
                log.write(json.dumps(dict(
                    event='game', utc=time.time(),
                    models={str(k): v for k, v in g['models'].items()},
                    result=res_o, steps=g['steps'], plays=g['plays'],
                    deck_b=g['deck_b'], mirror=g['mirror'], version=g['ver'],
                    seconds=round(time.time() - g['started'], 1),
                )) + '\n')
                for oo in (0, 1):
                    mdl = g['models'][oo]
                    b = g['buf'][oo]
                    T = len(b['rows'])
                    if mdl in ('A', 'B') and g['rec'][oo] and T > 10 and T == len(b['act']) == len(b['lp']) == len(b['rew']):
                        try:
                            writes.put_nowait((
                                mdl, b, res_o[oo], g['ver'].get(mdl, 0), g['deck_b'], g['mirror'],
                                f"{args.index}-{s}-{g['key'][oo][2]}-{oo}",
                            ))
                        except queue.Full:
                            pass
                stats['done'] += 1
                reset(s)
            coord.start_session()
        stats['t_native'] += time.monotonic() - t2
        stats['steps'] += 1

    inflight = [featurize(0), None]
    while not (run / 'STOP').exists():
        inflight[1] = featurize(1)
        act_and_step(0, *inflight[0])
        inflight[0] = featurize(0)
        act_and_step(1, *inflight[1])
        if time.monotonic() - stats['t0'] > 60.:
            dt = time.monotonic() - stats['t0']
            half_slots = len(halves[0])
            game_steps = stats['steps'] * half_slots
            log.write(json.dumps(dict(
                event='timing', utc=time.time(),
                steps=stats['steps'], game_steps=game_steps, games=stats['done'],
                feat=round(stats['t_feat'], 1), model=round(stats['t_model'], 1), native=round(stats['t_native'], 1),
                seconds=round(dt, 1), rifiutati=stats.get('rej', 0),
            )) + '\n')
            stats.update(steps=0, t_feat=0., t_model=0., t_native=0., done=0, t0=time.monotonic())


if __name__ == '__main__':
    main()
