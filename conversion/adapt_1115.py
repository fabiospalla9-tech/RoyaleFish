"""Turns IL_Replay_1115 payloads (royaleapi-battle-replay.v1) into the older royaleapi-battle-actions.v1 shape read by the converter."""
import json
CHAMPS = {'archer-queen', 'golden-knight', 'skeleton-king', 'mighty-miner', 'monk', 'little-prince', 'boss-bandit', 'goblinstein'}
def able(k): return k.endswith('-hero') or k in CHAMPS
def adapt(text):
    p = json.loads(text); b = p['battle']; num = {'team': 0, 'opponent': 0}; abn = {'team': 0, 'opponent': 0}
    for s in ('team', 'opponent'):
        pl = b[s]['players'][0]; h = pl.get('source_final_tower_hp') or {}
        pl['final_tower_hitpoints'] = {**h, 'total': sum(max(0, int(v)) for v in h.values() if isinstance(v, (int, float)))}
    for e in p['events']:
        s = e['side']
        if e['kind'] == 'play_card':
            num[s] += 1; e['card_play_number'] = num[s]; e.setdefault('deck_card_key_candidates', []); e.setdefault('ability_source_candidates', []); e.setdefault('ability_source_authoritative', None)
        elif e['kind'] == 'activate_ability':
            abn[s] += 1; e['card_play_number'] = abn[s]; e.setdefault('card_key', None); e.setdefault('coordinates', None); e.setdefault('form_at_play', None); e.setdefault('deck_card_key_candidates', [])
            if not e.get('ability_source_candidates'):
                e['ability_source_candidates'] = [c['card_key'].removesuffix('-hero') for c in b[s]['players'][0]['deck'] if able(c['card_key'])]; e['ability_source_authoritative'] = False
    n = len(p['events']); a = sum(abn.values()); p['schema_version'] = 'royaleapi-battle-actions.v1'; p.setdefault('collection_schema_version', 1)
    p['validation'] = {'ability_event_count': a, 'coordinate_event_count': n - a, 'map_marker_count': n, 'normalized_event_count': n, 'timeline_event_count': n, 'unmatched_timeline_event_count': 0}
    return json.dumps(p)
