"""Self-play environment for reinforcement learning in RoyaleSim: Hog 2.6 vs Hog 2.6, encoded exactly like the IL data.

Every battle keeps the whole-match play history; each side sees the board from its own bottom (team 1 is mirrored).
A command takes the game's own delay (COMMAND_DELAY ticks) before the card lands, like the replays the IL learned from.
"""
import json, math, os, random
import numpy as np
import royalesim

SUB = royalesim.SUBTILE_PER_MILLITILE
HOG = ['HogRider', 'Musketeer', 'Cannon', 'IceGolemite', 'IceSpirits', 'Skeletons', 'Fireball', 'Log']
FORMS = [0, 2, 1, 0, 0, 1, 0, 0]      # the classic 2.6 (4,716 of 7,652 replays): Hero Musketeer, Evo Cannon, Evo Skeletons
COMMAND_DELAY = 20
# rewards (zero-sum between the two sides, see Battle.rewards)
R_WIN = 1.0
R_TOWER = 0.1                   # a princess tower taken / lost (ClashAI tower_crown: w_crown 0.3 / 3)
R_DAMAGE_PER_HP = 0.1 / 4858    # every HP of every tower weighs the same (ClashAI w_tower 0.3 / 3 per full tower)
NO_DAMAGE_SPELLS = {'Fireball', 'Log'}
R_FB_COMBO = 0.0                # removed by Fabio: a rule valid only for the 2.6
R_FB_PREDICT = 0.50             # kept by Fabio (22 Nov): a Fireball thrown before the enemy Cannon lands on it
FB_RADIUS = 2500
R_TURRET = 0.25                 # Fabio: + only if the pressed turret really pulls a win condition onto itself; - if it ignores the turret and hits the tower
R_GB_LATE = float(os.environ.get('R_GB_LATE', '0'))   # small malus for answering a Goblin Barrel too late, at most twice per game (0 = off)
R_BARREL_LOG = 0.30             # Fabio: a Log / Barbarian Barrel that really kills every Goblin of a Barrel thrown at the tower (paid three times per game at most)
R_LOG_WASTE, R_LOG_WASTE_OT = 0.06, 0.10   # Fabio: against a Barrel that is back in hand, a Log that kills fewer than 2 units is wasted (worse in overtime)
R_GY_OK, R_GY_BAD = 0.10, 0.06    # Graveyard on a tower with a tank or Poison / alone (deck guides: never without a tank)
R_POISON_BAD = 0.05               # Poison on a tower with nothing else under it, unless it finishes the tower
R_SPLASH = 0.25                   # Fabio: one Fireball on two of Ice Wizard, Furnace, Goblin Hut
R_IW_FAST = 0.10                  # Ice Wizard dead within 5 s of appearing
R_GY_TIME, R_GY_EARLY, R_GY_FAR, R_GY_COUNTER = 0.08, 0.04, 0.04, 0.06   # pro guides: after the enemy spent, not before double elixir, far side of the tower, not into its counters
R_HUT_POS, R_HUT_THREAT, R_HUT_IDLE = 0.05, 0.05, 0.04                  # Goblin Hut central near the river, only with a threat coming
R_FUR_AIR, R_POISON_OK, R_STACK = 0.05, 0.08, 0.05                      # Furnace alive when air arrives, Poison on swarm or tower+troops, no stacking under one Fireball
R_OPEN, R_GY_POISON, R_GY_FULL, R_GY_LANE, R_GY_WRONG, R_GY_PUNISH, R_GY_DOUBLE = 0.05, 0.06, 0.05, 0.05, 0.04, 0.06, 0.04   # Deck Shop guide pasted by Fabio
R_FRONT, R_BARB, R_HUT_EG = 0.05, 0.05, 0.05
R_ICE_SLOW, R_ICE_PRE, R_ICE_BLIND, R_ICE_SOLO = 0.05, 0.03, 0.08, 0.04      # LDPlayer Hero Ice Wizard guide pasted by Fabio
R_ICE_LOG, R_ICE_SHORT, R_ICE_LONG, R_ICE_GY = 0.05, 0.05, 0.05, 0.10
R_ICE_SUPPORT, R_ICE_GYDEF, R_ICE_BRIDGE, R_ICE_AIR = 0.08, 0.10, 0.08, 0.08   # right targets from the Hero Ice Wizard guide
ICE_BRIDGE = {'Assassin', 'BossBandit', 'Ram', 'RamRider', 'BattleRam', 'BattleRam_EV1', 'Ghost', 'Ghost_EV1'}
GY_COUNTERS = {'Valkyrie', 'Witch', 'BlowdartGoblin', 'ElectroWizard'}
R_HUT = 0.10                      # Goblin Hut that really pulls a win condition
ICE_LIMIT, R_ICE_OVER = int(os.environ.get('ICE_LIMIT', '0')), float(os.environ.get('R_ICE_OVER', '0.10'))   # cubes per game before each extra one is punished, and the cube bonus stops (0 = off)
R_ICE_OK, R_ICE_BAD = 0.15, 0.10  # Hero Ice Wizard: cube that freezes a win condition or 2+ troops / cube that freezes nothing
WIN_UNITS = {'HogRider', 'RamRider', 'Ram', 'BattleRam', 'Giant', 'Giant_EV1', 'GiantHero', 'RoyalGiant', 'GoblinGiant', 'ElectroGiant', 'ElectroGiant_EV1', 'Golem', 'ElixirGolem1', 'Balloon', 'BalloonHero', 'LavaHound', 'Wallbreaker', 'RoyalHog', 'SkeletonBalloon', 'GoblinDrill', 'Miner', 'MinionGiant', 'Pekka', 'MegaKnight', 'GiantSkeleton', 'BattleRam_EV1', 'RoyalHog_EV1', 'GoblinDrill_EV1', 'GoblinGiant_EV1', 'RoyalGiant_EV1', 'SkeletonBalloon_EV1', 'Wallbreaker_EV1', 'Pekka_EV1', 'MegaKnight_EV1', 'ElixirGolem2', 'ElixirGolem4', 'Goblinstein', 'GiantBuffer'}
ABILITY_SLOT = 4                # command slot of the first ability button (HAND_SIZE + button)
DECISION_TICKS = 10             # one decision every 0.5 s, like ClashAI
MAXE, MAXH = 64, 160
_VOCAB = None
TABLE_OV = None
if os.environ.get('RF_TABLE'):        # the current client: card table and placement rule of 16.635
    TABLE_OV = {'cards.CARD_TABLE': json.dumps(os.environ['RF_TABLE']), 'placement.SPELL_AS_DEPLOY_NO_DEPLOY_CELLS': json.dumps('client16635_can_place_on_buildings')}


def vocab():
    global _VOCAB
    if _VOCAB is None:
        b = royalesim.Battle(card_names=None, slot_of_k=[[0, 1, 2]] * 2)
        v = ['<pad>', '<king>', '<princess>', '<unknown>'] + [r[0] for r in json.loads(b.catalogue_json())]
        _VOCAB = {n: i for i, n in enumerate(v)}
    return _VOCAB


class Battle:
    def __init__(self, seed, deck0=HOG, deck1=HOG, forms0=FORMS, forms1=FORMS, level=16):
        self.vid = vocab()
        self.decks = [list(deck0), list(deck1)]
        self.forms = [list(forms0), list(forms1)]
        self.names = list(dict.fromkeys(self.decks[0] + self.decks[1]))
        self.costs = {}
        self.b = royalesim.Battle(card_names=self.names, slot_of_k=[[0, 1, 2]] * 2, calibration_overrides=TABLE_OV)
        cat = {r[0]: r for r in json.loads(self.b.catalogue_json())}
        self.cost_of = {n: int(cat[n][2]) for n in self.names}
        self.no_damage = NO_DAMAGE_SPELLS | {n for n in self.names if cat[n][1] in (2, 3)}   # no spell damage in the shaping, for any deck
        self.buildings = {n for n in self.names if cat[n][1] == 1}
        rng = random.Random(seed)
        orders = []
        for d in self.decks:
            o = list(range(8)); rng.shuffle(o); orders.append(o)
        self.decks = [[d[i] for i in o] for d, o in zip(self.decks, orders)]
        self.forms = [[f[i] for i in o] for f, o in zip(self.forms, orders)]
        self.b.reset(seed, [[self.names.index(c) for c in self.decks[0]], [self.names.index(c) for c in self.decks[1]]], 0, 0,
                     [6000, 6000], None, [], forms=self.forms, levels=[[level] * 8, [level] * 8], tower_levels=[level, level])
        self.b.set_command_delay_ticks(COMMAND_DELAY, COMMAND_DELAY)
        self.history = []        # (tick, team, name, form, x, y) absolute coordinates
        self.tick = 0
        self.done = False
        self.winner = -1
        self.state = json.loads(self.b.state_json())
        self.spell_windows = []   # (from_tick, to_tick, target_team, x, y): tower damage there is not rewarded
        self.reward = [0.0, 0.0]  # accumulated since each side's last decision
        self.fireballs = []       # thrown and not landed yet
        self.stats = {k: [0, 0] for k in ('ability', 'fireball', 'combo', 'predict', 'turret', 'turret_ok', 'turret_bad', 'ice', 'ice_ok', 'ice_bad', 'zap', 'zap_ok', 'zap_bad', 'predict_paid', 'turret_paid', 'gb_seen', 'gb_log', 'gb_kill', 'gb_early', 'gb_late', 'gb_paid', 'gb_can', 'gb_kill_can', 'gb_evo', 'gb_evo_right', 'gb_evo_wrong', 'gb_waste', 'gy', 'gy_ok', 'gy_bad', 'poi_bad', 'splash', 'iw', 'iw_fast', 'hut_ok', 'p_gy', 'p_poi', 'p_splash', 'p_iw', 'p_hut', 'gy_time', 'gy_early', 'gy_far', 'gy_counter', 'hut_pos', 'hut_threat', 'hut_idle', 'fur_air', 'poi_ok', 'stack', 'p_gy_time', 'p_gy_early', 'p_gy_far', 'p_gy_counter', 'p_hut_pos', 'p_hut_threat', 'p_hut_idle', 'p_fur_air', 'p_poi_ok', 'p_stack', 'open_skel', 'gy_poison', 'gy_full', 'gy_lane', 'gy_wrong', 'gy_punish', 'gy_double', 'front', 'barb_ok', 'hut_eg', 'p_open_skel', 'p_gy_poison', 'p_gy_full', 'p_gy_lane', 'p_gy_wrong', 'p_gy_punish', 'p_gy_double', 'p_front', 'p_barb_ok', 'p_hut_eg', 'ice_slow', 'ice_pre', 'ice_blind', 'ice_solo', 'ice_log', 'ice_short', 'ice_long', 'ice_gy', 'p_ice_slow', 'p_ice_pre', 'p_ice_blind', 'p_ice_solo', 'p_ice_log', 'p_ice_short', 'p_ice_long', 'p_ice_gy', 'forced', 'ice_support', 'ice_gydef', 'ice_bridge', 'ice_air', 'p_ice_support', 'p_ice_gydef', 'p_ice_bridge', 'p_ice_air')}
        self.gys = []; self.pois = []; self.iws = {}; self.huts = set(); self.airs = set(); self.opened = set(); self.rule_on = bool({'Graveyard', 'Poison', 'IceWizard', 'FirespiritHut', 'GoblinHut'} & set(self.decks[0] + self.decks[1]))
        self.log_checks = []
        self.barrels = []; self.gb_late_paid = [0, 0]; self.track_gb = 'GoblinBarrel' in self.decks[0] + self.decks[1]
        self.predict_cap = [1, 1]; self.no_predict = [False, False]     # no predict bonus against a frozen opponent whose habits can be farmed
        self.turrets = {}; self.cubes = {}; self.zaps = {}; self.utypes = None; self.track = any(f == 2 and c in ('Musketeer', 'IceWizard', 'ElectroWizard') for d, ff in zip(self.decks, self.forms) for c, f in zip(d, ff))
        self.phi = [0.0, 0.0]     # shaping paid so far: it is taken back at the end, so only the result counts in total

    def tower_hp(self):
        return [list(p['tower_hp']) for p in self.state['players']]

    def form_of(self, team, name):
        d = self.decks[team]
        return self.forms[team][d.index(name)] if name in d else 0

    def can_play(self, team):
        """True when the side holds at least one card it can afford (the only moments worth asking the model)."""
        p = self.state['players'][team]
        if p['pending']:
            return False
        return self.ability_ready(team) or any(c <= p['elixir_milli'] // 1000 for c in p['hand_costs'])

    def ability_ready(self, team):
        p = self.state['players'][team]
        return any(r[0] and r[2] * 1000 <= p['elixir_milli'] for r in p['abilities'])

    def hand_form(self, team, name):
        """The form this card will really have if played now (an Evo card is evolved only when its cycle is full)."""
        f = self.form_of(team, name)
        if f != 1:
            return f
        for r in self.state['players'][team]['evo']:
            if 0 <= r[0] < len(self.names) and self.names[r[0]] == name:
                return 1 if r[2] else 0
        return 0

    def unit_form(self, e, nm):
        """The real form of a unit on the board. The hero's turret gets its own label (1), it is not the Musketeer."""
        flags = e[20] or 0
        if flags & 16:
            return 2
        if flags & 8:
            return 1
        return 1 if nm == 'Musketeer' and e[2] == 1 else 0

    def encode(self, team):
        """The IL network inputs for `team`, which always sees itself at the bottom."""
        st = self.state
        mir = team == 1
        def xy(x, y):
            return (18000 - x, 32000 - y) if mir else (x, y)
        rows = []
        for e in st['entities']:
            uid, tm, kind, cid = e[0], e[1], e[2], e[3]
            x, y = xy(e[5] / SUB, e[6] / SUB)
            if kind == 2:
                nid, form = self.vid['<king>'], 0
            elif kind == 3:
                nid, form = self.vid['<princess>'], 0
            else:
                nm = self.names[cid] if 0 <= cid < len(self.names) else None
                nid = self.vid.get(nm, self.vid['<unknown>']); form = self.unit_form(e, nm) if nm else 0
            mhp = max(1, e[8])
            rows.append([nid, form, 1.0 if tm == team else 0.0, x / 18000, y / 32000, e[7] / mhp, math.log1p(mhp) / 9,
                         1.0 if e[10] else 0.0, 1.0 if e[11] > 0 else 0.0, 1.0 if e[16] and e[16] > 0 else 0.0, 1.0 if kind in (2, 3) else 0.0])
        rows.sort(key=lambda r: (r[10] == 0.0, abs(r[4] * 32000 - 16000)))
        E = np.zeros((MAXE, 11), np.float32)
        n = min(MAXE, len(rows))
        if n:
            E[:n] = np.array(rows[:n], np.float32)
        me, op = st['players'][team], st['players'][1 - team]
        th = list(me['tower_hp']) + list(op['tower_hp'])
        maxk, maxp = max(th[0], th[3], 1), max(th[1], th[2], th[4], th[5], 1)
        ma, oa = me['abilities'], op['abilities']
        G = np.array([me['elixir_milli'] / 10000, st['tick'] / 7200, 1.0 if st['overtime'] else 0.0,
                      th[0] / maxk, th[1] / maxp, th[2] / maxp, th[3] / maxk, th[4] / maxp, th[5] / maxp,
                      1.0 if self.ability_ready(team) else 0.0, 1.0 if any(r[1] for r in ma) else 0.0,
                      1.0 if any(r[0] for r in oa) else 0.0, 1.0 if any(r[1] for r in oa) else 0.0], np.float32)
        hand = list(me['hand']) + [me['next_card']]
        H = np.array([[self.vid[self.names[h]], self.hand_form(team, self.names[h])] for h in hand], np.int64)
        M = np.zeros((MAXH, 7), np.float32)
        for i, (t, tm, nm, f, x, y) in enumerate(self.history[-MAXH:]):
            mx, my = xy(x, y)
            M[i] = [self.vid[nm], f, 1.0 if tm == team else 0.0, mx / 18000, my / 32000, (st['tick'] - t) / 600, 1.0]
        return E, n, G, H, M

    def step(self, actions):
        """actions: {team: (hand_slot, cell_x, cell_y) in the team's own frame}. Advances DECISION_TICKS ticks."""
        cmds = []; meta = []
        prev = {e[0]: e for e in self.state['entities']}
        for team, (slot, cx, cy) in actions.items():
            if slot >= ABILITY_SLOT:
                rows = self.state['players'][team]['abilities']; want = slot - ABILITY_SLOT
                btn = want if want < len(rows) and rows[want][0] else next((k for k, r in enumerate(rows) if r[0]), 0)
                cmds.append((team, ABILITY_SLOT + btn, 0, 0)); meta.append((team, None, None, None, None))
                continue
            if team == 1:
                cx, cy = 17 - cx, 31 - cy
            x, y = cx * 1000 + 500, cy * 1000 + 500
            cmds.append((team, slot, x * SUB, y * SUB))
            nm = self.names[self.state['players'][team]['hand'][slot]]
            meta.append((team, nm, self.hand_form(team, nm), x, y))
        res = self.b.step(cmds, 1) if cmds else self.b.step([], 1)
        accepted = []
        for (team, nm, pf, x0, y0), (_, slot, x, y), r in zip(meta, cmds, res or []):
            if r[1] == 0:
                if nm is None:
                    self.stats['ability'][team] += 1; accepted.append(team)
                    rw = self.state['players'][team]['abilities'][slot - ABILITY_SLOT]
                    if self.rule_on and 0 <= rw[3] < len(self.names) and self.names[rw[3]] == 'IceWizard':
                        self._ice_press(team)
                    if self.track and 0 <= rw[3] < len(self.names) and self.names[rw[3]] == 'ElectroWizard':
                        self.stats['zap'][team] += 1; self.zaps[(team, self.state['tick'])] = {'team': team, 'tick': self.state['tick'], 'hit': set(), 'win': False}
                    continue
                self.history.append((self.state['tick'], team, nm, pf, x / SUB, y / SUB))
                if self.rule_on:
                    self._play_rules(team, nm, x / SUB, y / SUB)
                accepted.append(team)
                if nm in ('Graveyard', 'Poison') and self.rule_on:
                    T = [(9000, 3000), (3500, 6500), (14500, 6500)]; T = T if team == 1 else [(a, 32000 - c) for a, c in T]; dd = [(a - x / SUB) ** 2 + (c - y / SUB) ** 2 for a, c in T]; kt = dd.index(min(dd))
                    if nm == 'Poison':
                        self.pois.append({'team': team, 't': self.state['tick'], 'x': x / SUB, 'y': y / SUB, 'k': kt, 'nt': dd[kt] <= 5000 ** 2})
                    elif dd[kt] <= 5000 ** 2:
                        self.gys.append({'team': team, 't': self.state['tick'], 'x': x / SUB, 'y': y / SUB, 'k': kt}); self.stats['gy'][team] += 1
                        self._gy_rules(team, kt)
                if nm == 'GoblinBarrel' and (y / SUB > 16000 if team == 0 else y / SUB < 16000):
                    T = [(9000, 3000), (3500, 6500), (14500, 6500)]; T = T if team == 1 else [(a, 32000 - c) for a, c in T]     # towers of the defender: slot 1 is always x=3500, slot 2 x=14500
                    grp = []
                    for px in ([x / SUB, 18000 - x / SUB] if pf == 1 else [x / SUB]):      # an Evo Barrel also throws a decoy into the mirrored lane
                        dist = [(a - px) ** 2 + (c - y / SUB) ** 2 for a, c in T]; kt = dist.index(min(dist))
                        if dist[kt] <= 4500 ** 2:      # only a Barrel really thrown at a tower counts
                            grp.append({'team': team, 't': self.state['tick'], 'x': px, 'y': y / SUB, 'land': None, 'uids': set(), 'log': None, 'k': kt, 'hp': None, 'group': None})
                    if grp:
                        self.stats['gb_seen'][1 - team] += 1; self.barrels += grp
                        if len(grp) == 2:
                            self.stats['gb_evo'][1 - team] += 1; grp[0]['group'] = grp[1]; grp[1]['group'] = grp[0]
                if nm in ('Log', 'BarbLog'):
                    hit = False
                    for q in self.barrels:
                        if q['team'] == 1 - team and q['log'] is None and abs(q['x'] - x / SUB) <= 6000:
                            q['log'] = self.state['tick']; hit = True; break
                    foe = [h[2] for h in self.history if h[1] == 1 - team]
                    if not hit and 'GoblinBarrel' in foe and len(foe) - 1 - max(i for i, c in enumerate(foe) if c == 'GoblinBarrel') >= 3:     # the Barrel is back in the enemy hand
                        self.log_checks.append({'team': team, 't': self.state['tick'], 'x': x / SUB, 'alive': None})
                if nm == 'Fireball':
                    self.stats['fireball'][team] += 1
                    self.fireballs.append({'team': team, 'x': x / SUB, 'y': y / SUB, 'tick': self.state['tick'], 'seen': False,
                                           'cannons': {e[0] for e in self.state['entities'] if e[1] == 1 - team and e[2] not in (2, 3) and 0 <= e[3] < len(self.names) and self.names[e[3]] in self.buildings}})
                if nm in self.no_damage:
                    t0 = self.state['tick'] + COMMAND_DELAY
                    self.spell_windows.append((t0, t0 + 60, 1 - team, x / SUB, y / SUB))
        before = self.tower_hp()
        if DECISION_TICKS > 1:
            self.b.step([], DECISION_TICKS - 1)
        self.state = json.loads(self.b.state_json())
        self.tick = self.state['tick']
        after = self.tower_hp()
        TOWER_XY = [(9000, 3000), (3500, 6500), (14500, 6500)]
        for team in (0, 1):
            for k in range(3):
                dmg = max(0, before[team][k] - after[team][k])
                if dmg == 0:
                    continue
                tx, ty = TOWER_XY[k] if team == 0 else (TOWER_XY[k][0], 32000 - TOWER_XY[k][1])     # tower slots use the same x for both teams
                spell = any(a <= self.tick <= b and tt == team and (sx - tx) ** 2 + (sy - ty) ** 2 < 4000 ** 2
                            for a, b, tt, sx, sy in self.spell_windows)
                if not spell:
                    self.reward[team] -= R_DAMAGE_PER_HP * dmg; self.phi[team] -= R_DAMAGE_PER_HP * dmg
                    self.reward[1 - team] += R_DAMAGE_PER_HP * dmg; self.phi[1 - team] += R_DAMAGE_PER_HP * dmg
                if k > 0 and before[team][k] > 0 and after[team][k] <= 0:
                    self.reward[team] -= R_TOWER; self.reward[1 - team] += R_TOWER
                    self.phi[team] -= R_TOWER; self.phi[1 - team] += R_TOWER
        self._score_fireballs(prev)
        if self.track:
            self._score_abilities(prev)
        if self.barrels or self.log_checks:
            self._score_barrels()
        if self.rule_on:
            self._score_rules()
        self.spell_windows = [w for w in self.spell_windows if w[1] >= self.tick]
        if self.state['game_over'] or self.tick >= 7200:
            self.done = True; self.winner = self.state['winner']
            for team in (0, 1):
                self.reward[team] -= self.phi[team]
            if self.winner in (0, 1):
                self.reward[self.winner] += R_WIN; self.reward[1 - self.winner] -= R_WIN
        return accepted

    def _is(self, e, name):
        return e[2] not in (2, 3) and 0 <= e[3] < len(self.names) and self.names[e[3]] == name

    def _score_fireballs(self, prev):
        """Bonus when a Fireball lands: Cannon + Musketeer together, and more if it was thrown before the Cannon was played."""
        if not self.fireballs:
            return
        now = {e[0]: e for e in self.state['entities']}
        flying = [(s[0], s[5] / SUB, s[6] / SUB) for s in self.state['spells'] if 0 <= s[1] < len(self.names) and self.names[s[1]] == 'Fireball']
        keep = []
        for f in self.fireballs:
            there = any(t == f['team'] and (ax - f['x']) ** 2 + (ay - f['y']) ** 2 < 1000 ** 2 for t, ax, ay in flying)
            if there:
                f['seen'] = True; keep.append(f); continue
            if not f['seen']:
                if self.tick - f['tick'] < 60:
                    keep.append(f)
                continue
            tags = set(); foe = 1 - f['team']; hit = {'Cannon': [], 'Musketeer': []}     # 'Cannon' = any enemy building (Tesla, Inferno, Bomb Tower...)
            for uid in set(prev) | set(now):
                e = now.get(uid) or prev[uid]
                if e[1] != foe or (e[5] / SUB - f['x']) ** 2 + (e[6] / SUB - f['y']) ** 2 > (FB_RADIUS + (e[9] or 0) / SUB) ** 2:
                    continue
                before = prev[uid][7] if uid in prev else e[8]
                after = now[uid][7] if uid in now else 0
                if after < before:
                    if self.rule_on:
                        if self.utypes is None:
                            self.utypes = json.loads(self.b.unit_types_json())
                        ut_ = e[royalesim.ENTITY_FIELDS.index('unit_type')]; nn = self.utypes[ut_] if ut_ is not None and 0 <= ut_ < len(self.utypes) else ''
                        tags |= {'iw'} if nn in ('IceWizard', 'IceWizardHero') else {'fur'} if nn in ('Furnace_rework', 'Furnace_EV1') else {'hut'} if nn == 'GoblinHut_Rework' else set()
                    if e[2] == 1 and 0 <= e[3] < len(self.names) and self.names[e[3]] in self.buildings:     # the building itself, not the troops it spawns
                        hit['Cannon'].append(uid)
                    if self._is(e, 'Musketeer') and e[2] != 1:
                        hit['Musketeer'].append(uid)
            bonus = 0.0
            if hit['Cannon'] and hit['Musketeer']:
                bonus += R_FB_COMBO; self.stats['combo'][f['team']] += 1
            later = any(h[1] == foe and h[2] in self.buildings and h[0] > f['tick'] and (h[4] - f['x']) ** 2 + (h[5] - f['y']) ** 2 <= (FB_RADIUS + 1000) ** 2 for h in self.history)   # played there, after the throw
            if later and any(u not in f['cannons'] for u in hit['Cannon']):
                self.stats['predict'][f['team']] += 1
                if not self.no_predict[f['team']] and self.stats['predict_paid'][f['team']] < self.predict_cap[f['team']]:      # 2 per game against meta decks, 1 otherwise
                    bonus += R_FB_PREDICT; self.stats['predict_paid'][f['team']] += 1
            if len(tags) >= 2:
                self.stats['splash'][f['team']] += 1
                if self.stats['p_splash'][f['team']] < 2:
                    self.stats['p_splash'][f['team']] += 1; bonus += R_SPLASH
            self.reward[f['team']] += bonus; self.reward[foe] -= bonus
        self.fireballs = keep

    def _score_abilities(self, prev):
        E = royalesim.ENTITY_FIELDS; tg, bf, ph, ut = E.index('target_uid'), E.index('buffs'), E.index('attack_phase'), E.index('unit_type')
        if self.utypes is None:
            self.utypes = json.loads(self.b.unit_types_json())
        U = self.utypes; ents = self.state['entities']; now = {e[0]: e for e in ents}
        name = lambda e: U[e[ut]] if e[ut] is not None and 0 <= e[ut] < len(U) else ''
        towers = [{e[0] for e in ents if e[1] == t and e[2] in (2, 3)} for t in (0, 1)]
        for e in ents:
            n = name(e)
            if n == 'MusketeerTurret' and e[0] not in self.turrets:
                t = e[1]; self.stats['turret'][t] += 1
                self.turrets[e[0]] = {'team': t, 'already': {x[0] for x in ents if x[1] != t and name(x) in WIN_UNITS and x[tg] in towers[t]}, 'done': False}
            elif n == 'IceWizardHero_IceCube' and e[0] not in self.cubes:
                t = e[1]; self.stats['ice'][t] += 1
                if ICE_LIMIT and self.stats['ice'][t] > ICE_LIMIT:
                    self.reward[t] -= R_ICE_OVER; self.reward[1 - t] += R_ICE_OVER
                self.cubes[e[0]] = {'team': t, 'tick': self.tick, 'hit': set(), 'win': False, 'tower': False, 'names': set(), 'big': False}
                if any(self.names[k] in ('Log', 'Arrows') for k in list(self.state['players'][1 - t]['hand'])[:4]):
                    self._pay(t, 'ice_log', 'p_ice_log', -R_ICE_LOG)
        for uid, c in list(self.turrets.items()):
            if uid not in now:
                del self.turrets[uid]; continue
            if c['done']:
                continue
            t = c['team']; foes = [x for x in ents if x[1] != t and x[2] not in (2, 3) and name(x) in WIN_UNITS]
            if any(x[tg] == uid for x in foes):
                c['done'] = True; self.stats['turret_ok'][t] += 1
                if not self.no_predict[t] and self.stats['turret_paid'][t] < 2:      # paid at most twice per game, never against the frozen old 2.6
                    self.stats['turret_paid'][t] += 1; self.reward[t] += R_TURRET; self.reward[1 - t] -= R_TURRET
            elif any(x[0] not in c['already'] and x[tg] in towers[t] and (x[ph] or 0) > 0 for x in foes):
                c['done'] = True; self.stats['turret_bad'][t] += 1; self.reward[t] -= R_TURRET; self.reward[1 - t] += R_TURRET
        sn = E.index('stun_ticks')
        for key, c in list(self.zaps.items()):      # Hero Electro Wizard: same rule as the cube, the surge must really stop a win condition or 2+ troops
            t = c['team']; h = next((x for x in ents if x[1] == t and name(x) == 'ElectroWizardHero'), None)
            if h is not None and self.tick >= c['tick'] + COMMAND_DELAY:
                for x in ents:
                    if x[1] != t and x[2] not in (2, 3) and (x[sn] or 0) > 0 and (x[5] - h[5]) ** 2 + (x[6] - h[6]) ** 2 <= (6500 * SUB) ** 2:
                        c['hit'].add(x[0]); c['win'] = c['win'] or name(x) in WIN_UNITS
            if h is None or self.tick >= c['tick'] + COMMAND_DELAY + 70:
                if c['win'] or len(c['hit']) >= 2:
                    self.stats['zap_ok'][t] += 1; self.reward[t] += R_ICE_OK; self.reward[1 - t] -= R_ICE_OK
                elif not c['hit']:
                    self.stats['zap_bad'][t] += 1; self.reward[t] -= R_ICE_BAD; self.reward[1 - t] += R_ICE_BAD
                del self.zaps[key]
        for uid, c in list(self.cubes.items()):
            t = c['team']
            for x in ents:
                if x[1] != t and x[2] not in (2, 3) and any(b[0] == 'IceWizardHero_FreezeBuff' for b in (x[bf] or ())):
                    c['hit'].add(x[0]); c['win'] = c['win'] or name(x) in WIN_UNITS; c['names'].add(name(x)); c['big'] = c['big'] or ((x[E.index('max_hp')] or 0) >= 1000 and name(x) != 'Graveyard_rework_Skeleton')
                if x[1] != t and x[2] in (2, 3) and any(b[0] == 'IceWizardHero_FreezeBuff' for b in (x[bf] or ())):
                    c['tower'] = (x[5] / SUB, x[6] / SUB)
            if self.tick - c['tick'] >= 40:
                fz = c['names']
                if any(n not in WIN_UNITS and n != 'Graveyard_rework_Skeleton' for n in fz) and any(x[1] != t and x[2] not in (2, 3) and name(x) in WIN_UNITS for x in ents):
                    self._pay(t, 'ice_support', 'p_ice_support', R_ICE_SUPPORT)
                if 'Graveyard_rework_Skeleton' in fz and c['big'] and any(h[1] != t and h[2] == 'Graveyard' and self.tick - h[0] <= 240 for h in self.history):
                    self._pay(t, 'ice_gydef', 'p_ice_gydef', R_ICE_GYDEF)
                if fz & ICE_BRIDGE:
                    self._pay(t, 'ice_bridge', 'p_ice_bridge', R_ICE_BRIDGE)
                if len(c['hit']) >= 2 and any('Balloon' in n or 'LavaHound' in n for n in fz):
                    self._pay(t, 'ice_air', 'p_ice_air', R_ICE_AIR)
                if uid not in now:
                    self._pay(t, 'ice_short', 'p_ice_short', -R_ICE_SHORT)
                elif c['hit'] or c['tower']:
                    self._pay(t, 'ice_long', 'p_ice_long', R_ICE_LONG)
                if c['tower'] and any(h[1] == t and h[2] == 'Graveyard' and c['tick'] - 200 <= h[0] <= c['tick'] + 40 and (h[4] - c['tower'][0]) ** 2 + (h[5] - c['tower'][1]) ** 2 <= 5000 ** 2 for h in self.history):   # the whole time the Graveyard is active, on the frozen tower
                    self._pay(t, 'ice_gy', 'p_ice_gy', R_ICE_GY)
                if c['win'] or len(c['hit']) >= 2:
                    self.stats['ice_ok'][t] += 1
                    if not ICE_LIMIT or self.stats['ice_ok'][t] <= ICE_LIMIT:
                        self.reward[t] += R_ICE_OK; self.reward[1 - t] -= R_ICE_OK
                elif not c['hit']:
                    self.stats['ice_bad'][t] += 1; self.reward[t] -= R_ICE_BAD; self.reward[1 - t] += R_ICE_BAD
                del self.cubes[uid]

    def _ice_press(self, team):
        E = royalesim.ENTITY_FIELDS; ut, bf = E.index('unit_type'), E.index('buffs'); fw = 1 if team == 0 else -1
        if self.utypes is None:
            self.utypes = json.loads(self.b.unit_types_json())
        U = self.utypes; ents = self.state['entities']; name = lambda e: U[e[ut]] if e[ut] is not None and 0 <= e[ut] < len(U) else ''
        hero = next((e for e in ents if e[1] == team and name(e) == 'IceWizardHero'), None)
        if hero is None:
            return
        hx, hy = hero[5] / SUB, hero[6] / SUB; d = lambda e: ((e[5] / SUB - hx) ** 2 + (e[6] / SUB - hy) ** 2) ** .5
        if any(e[1] != team and any('IceWizardHeroSlow' in b[0] for b in (e[bf] or ())) for e in ents):
            self._pay(team, 'ice_slow', 'p_ice_slow', R_ICE_SLOW)
        elif self.state['players'][team]['elixir_milli'] >= 8000 and self.tick < 2400 and (16000 - hy) * fw >= 6000 and not any(e[1] != team and e[2] not in (2, 3) and d(e) <= 6000 for e in ents):
            self._pay(team, 'ice_pre', 'p_ice_pre', R_ICE_PRE)
        else:
            self._pay(team, 'ice_blind', 'p_ice_blind', -R_ICE_BLIND)
        if (hy - 16000) * fw > 0 and not any(e[1] == team and e[0] != hero[0] and e[2] not in (2, 3) and d(e) <= 5000 for e in ents):
            self._pay(team, 'ice_solo', 'p_ice_solo', -R_ICE_SOLO)

    def _gy_rules(self, team, k):
        foe = 1 - team; pl = self.state['players'][foe]; el = pl['elixir_milli']; fw = 1 if team == 0 else -1
        if el <= 3000:
            self._pay(team, 'gy_time', 'p_gy_time', R_GY_TIME)
        elif self.state['tick'] < 1200:
            self._pay(team, 'gy_early', 'p_gy_early', -R_GY_EARLY)
        if el >= 8000:
            self._pay(team, 'gy_full', 'p_gy_full', -R_GY_FULL)
        if any(self.names[c] in GY_COUNTERS for c in list(pl['hand'])[:4]):
            self._pay(team, 'gy_counter', 'p_gy_counter', -R_GY_COUNTER)
        if k not in (1, 2):
            return
        o = 3 - k; hp = self.tower_hp()[foe]
        if any(e[1] == foe and e[2] not in (2, 3) and (e[8] or 0) >= 2500 and (e[6] / SUB - 16000) * fw >= 13000 and (1 if e[5] / SUB < 9000 else 2) == o for e in self.state['entities']):
            self._pay(team, 'gy_punish', 'p_gy_punish', R_GY_PUNISH)
        elif hp[o] > 0 and 0 < hp[k] < hp[o] - 200:
            self._pay(team, 'gy_lane', 'p_gy_lane', R_GY_LANE)
        elif 0 < hp[o] < hp[k] - 200:
            self._pay(team, 'gy_wrong', 'p_gy_wrong', -R_GY_WRONG)

    def _play_rules(self, team, nm, x, y):
        fw = 1 if team == 0 else -1
        if team not in self.opened:
            self.opened.add(team)
            if nm == 'Skeletons' and 'Graveyard' in self.decks[team] and (16000 - y) * fw >= 9000:
                self._pay(team, 'open_skel', 'p_open_skel', R_OPEN)
        if nm not in ('GoblinHut', 'FirespiritHut', 'IceWizard', 'Knight', 'BarbLog'):
            return
        E = royalesim.ENTITY_FIELDS; ut = E.index('unit_type'); fl = E.index('flying')
        if self.utypes is None:
            self.utypes = json.loads(self.b.unit_types_json())
        U = self.utypes; name = lambda e: U[e[ut]] if e[ut] is not None and 0 <= e[ut] < len(U) else ''; ents = self.state['entities']
        dist = lambda e: ((e[5] / SUB - x) ** 2 + (e[6] / SUB - y) ** 2) ** .5
        FUR = ('Furnace_rework', 'Furnace_EV1'); HUT = 'GoblinHut_Rework'; KN = ('Knight', 'Knight_EV1'); own = [e for e in ents if e[1] == team]; foes = [e for e in ents if e[1] != team]
        if nm == 'BarbLog':
            if any(e[2] not in (2, 3) and not e[fl] and name(e).startswith(('Firecracker', 'Skeleton')) and dist(e) <= 4000 for e in foes):
                self._pay(team, 'barb_ok', 'p_barb_ok', R_BARB)
            return
        eg = nm in ('GoblinHut', 'FirespiritHut') and any(name(e).startswith('ElectroGiant') for e in foes)
        if eg:
            if any((name(e) in FUR or name(e) == HUT) and dist(e) <= 4000 for e in own):
                self._pay(team, 'hut_eg', 'p_hut_eg', R_HUT_EG)
        elif nm == 'Knight':
            if any((name(e) in FUR or name(e) == HUT) and dist(e) <= 1500 for e in own):
                self._pay(team, 'stack', 'p_stack', -R_STACK)
        elif any((name(e) in ('IceWizard', 'IceWizardHero', HUT) or name(e) in FUR) and dist(e) <= 2500 or (nm != 'IceWizard' and name(e) in KN and dist(e) <= 1500) for e in own):
            self._pay(team, 'stack', 'p_stack', -R_STACK)
        if nm == 'FirespiritHut' and any(name(e) in WIN_UNITS for e in foes) and any(name(e) in KN and 1000 <= (e[6] / SUB - y) * fw <= 5000 and abs(e[5] / SUB - x) <= 3500 for e in own):
            self._pay(team, 'front', 'p_front', R_FRONT)
        if nm == 'GoblinHut':
            if abs(x - 9000) <= 3000 and 2000 <= (16000 - y) * fw <= 6000:
                self._pay(team, 'hut_pos', 'p_hut_pos', R_HUT_POS)
            if any(name(e) in WIN_UNITS for e in foes):
                self._pay(team, 'hut_threat', 'p_hut_threat', R_HUT_THREAT)
            else:
                self._pay(team, 'hut_idle', 'p_hut_idle', -R_HUT_IDLE)

    def _pay(self, team, stat, paid, r, cap=2):
        self.stats[stat][team] += 1
        if r > 0 and self.no_predict[team]:
            return
        if self.stats[paid][team] < cap:
            self.stats[paid][team] += 1; self.reward[team] += r; self.reward[1 - team] -= r

    def _score_rules(self):
        E = royalesim.ENTITY_FIELDS; tg, ut, mh = E.index('target_uid'), E.index('unit_type'), E.index('max_hp')
        if self.utypes is None:
            self.utypes = json.loads(self.b.unit_types_json())
        U = self.utypes; ents = self.state['entities']; now = {e[0] for e in ents}; name = lambda e: U[e[ut]] if e[ut] is not None and 0 <= e[ut] < len(U) else ''
        near = lambda e, q, r: (e[5] / SUB - q['x']) ** 2 + (e[6] / SUB - q['y']) ** 2 <= r ** 2
        plays = lambda team, card, q, w, r: any(h[1] == team and h[2] == card and abs(h[0] - q['t']) <= w and (h[4] - q['x']) ** 2 + (h[5] - q['y']) ** 2 <= r ** 2 for h in self.history)
        keep = []
        for q in self.gys:
            if self.tick < q['t'] + COMMAND_DELAY + 20:
                keep.append(q); continue
            tank = any(e[1] == q['team'] and e[2] == 0 and (e[mh] or 0) >= 600 and near(e, q, 5500) and name(e) not in ('Graveyard_rework_Skeleton', 'IceWizard', 'IceWizardHero') for e in ents)   # troops only: never a building, never the Ice Wizard
            if tank or plays(q['team'], 'Poison', q, 80, 4500):
                self._pay(q['team'], 'gy_ok', 'p_gy', R_GY_OK)
                if self.tick >= 2400:
                    self._pay(q['team'], 'gy_double', 'p_gy_double', R_GY_DOUBLE)
            else:
                self._pay(q['team'], 'gy_bad', 'p_gy', -R_GY_BAD)
        self.gys = keep; keep = []
        for q in self.pois:
            if self.tick < q['t'] + COMMAND_DELAY:
                keep.append(q); continue
            nf = sum(e[1] != q['team'] and e[2] not in (2, 3) and near(e, q, 3500) for e in ents); foes = nf > 0
            if plays(q['team'], 'Graveyard', q, 80, 4500):
                self._pay(q['team'], 'gy_poison', 'p_gy_poison', R_GY_POISON)
            if nf >= 3 or (q['nt'] and nf >= 1) or any(e[1] != q['team'] and name(e).startswith(('Firecracker', 'Musketeer')) and near(e, q, 3500) for e in ents):
                self._pay(q['team'], 'poi_ok', 'p_poi_ok', R_POISON_OK)
            if not q['nt']:
                continue
            if not foes and self.tower_hp()[1 - q['team']][q['k']] > 600 and not plays(q['team'], 'Graveyard', q, 80, 4500):
                self._pay(q['team'], 'poi_bad', 'p_poi', -R_POISON_BAD)
        self.pois = keep
        for e in ents:
            n = name(e)
            if n in ('IceWizard', 'IceWizardHero') and e[0] not in self.iws:
                self.iws[e[0]] = (e[1], self.tick); self.stats['iw'][e[1]] += 1
            elif n == 'GoblinHut_Rework' and e[0] not in self.huts and any(x[1] != e[1] and x[2] not in (2, 3) and x[tg] == e[0] and name(x) in WIN_UNITS for x in ents):
                self.huts.add(e[0])
                if not self.no_predict[e[1]]:
                    self._pay(e[1], 'hut_ok', 'p_hut', R_HUT)
        for e in ents:
            if e[E.index('flying')] and e[2] not in (2, 3) and e[0] not in self.airs and ('Balloon' in name(e) or name(e) == 'LavaHound'):
                self.airs.add(e[0]); own = 1 - e[1]
                if any(x[1] == own and name(x) in ('Furnace_rework', 'Furnace_EV1') and (x[5] - e[5]) ** 2 + (x[6] - e[6]) ** 2 <= (7000 * SUB) ** 2 for x in ents):
                    self._pay(own, 'fur_air', 'p_fur_air', R_FUR_AIR)
        for u, (t, t0) in list(self.iws.items()):
            if u not in now:
                del self.iws[u]
                if self.tick - t0 <= 100:
                    self._pay(t, 'iw_fast', 'p_iw', -R_IW_FAST)
            elif self.tick - t0 > 100:
                self.iws[u] = (t, -10 ** 9)

    def _gb_late(self, d):
        if R_GB_LATE and self.gb_late_paid[d] < 2:
            self.gb_late_paid[d] += 1; self.reward[d] -= R_GB_LATE; self.reward[1 - d] += R_GB_LATE

    def _score_barrels(self):
        ents = self.state['entities']; now = {e[0] for e in ents}; gid = self.names.index('GoblinBarrel'); keep = []
        for q in self.barrels:
            d = 1 - q['team']
            if q['land'] is None or self.tick - q['land'] <= 10:
                got = {e[0] for e in ents if e[1] == q['team'] and e[2] not in (2, 3) and e[3] == gid and (e[5] / SUB - q['x']) ** 2 + (e[6] / SUB - q['y']) ** 2 <= 3500 ** 2}
                if got:
                    q['uids'] |= got
                    if q['land'] is None:
                        q['land'] = self.tick; q['hp'] = self.tower_hp()[d][q['k']]; pl = self.state['players'][d]
                        q['can'] = any(self.names[c] in ('Log', 'BarbLog') and k < len(pl['hand_costs']) and pl['hand_costs'][k] * 1000 <= pl['elixir_milli'] for k, c in enumerate(list(pl['hand'])[:4])) or q['log'] is not None
                        self.stats['gb_can'][d] += bool(q['can'])
            if q['land'] is None:
                if self.tick - q['t'] <= 140:
                    keep.append(q)
                continue
            alive = q['uids'] & now; ex = None if q['log'] is None else q['log'] + COMMAND_DELAY
            if ex is not None and not alive and self.tick >= ex:
                self.stats['gb_log'][d] += 1
                hpn = self.tower_hp()[d]; o = q.get('group'); right = o is None or hpn[q['k']] <= hpn[o['k']] + 200      # Evo: the lane with less tower HP
                if o is not None:
                    self.stats['gb_evo_right' if right else 'gb_evo_wrong'][d] += 1
                if self.tick - q['land'] <= 40 and hpn[q['k']] >= q['hp'] and right:      # all dead before they hit the tower: right timing
                    self.stats['gb_kill'][d] += 1; self.stats['gb_kill_can'][d] += bool(q.get('can'))
                    if self.stats['gb_paid'][d] < 3:
                        self.stats['gb_paid'][d] += 1; self.reward[d] += R_BARREL_LOG; self.reward[q['team']] -= R_BARREL_LOG
                else:
                    self.stats['gb_late'][d] += 1
                    if right:
                        self._gb_late(d)
                continue
            if not alive or self.tick - q['land'] >= 70:
                if ex is not None:
                    self.stats['gb_log'][d] += 1; self.stats['gb_early' if ex < q['land'] else 'gb_late'][d] += 1
                    if ex >= q['land']:
                        self._gb_late(d)
                continue
            keep.append(q)
        self.barrels = keep; chk = []
        for c in self.log_checks:
            if self.tick >= c['t'] + COMMAND_DELAY and c['alive'] is None:
                c['alive'] = {e[0] for e in ents if e[1] != c['team'] and e[2] not in (2, 3) and abs(e[5] / SUB - c['x']) <= 3000}
            if c['alive'] is not None and self.tick >= c['t'] + COMMAND_DELAY + 30:
                if len(c['alive'] - now) < 2:
                    pen = R_LOG_WASTE_OT if self.state['overtime'] else R_LOG_WASTE; self.stats['gb_waste'][c['team']] += 1; self.reward[c['team']] -= pen; self.reward[1 - c['team']] += pen
                continue
            chk.append(c)
        self.log_checks = chk

    def take_reward(self, team):
        r = self.reward[team]; self.reward[team] = 0.0
        return r
