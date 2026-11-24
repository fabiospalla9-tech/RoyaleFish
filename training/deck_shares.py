"""Hand-set share (%) of each deck family in the league. Cards and forms are never touched."""
SHARES = {'Fornace Cimitero': 4, 'MK + Domatore/Domatrice': 5.6, 'MK Evo': 5.6, 'Pekka + Ariete': 5.6, 'Pekka + altra': 5.6, 'Lava + altra': 5.6, 'Mongolfiera + altra': 5.6, 'Mortaio + altra': 5.6, 'Bait doppio': 5.6, 'Altre doppie': 5.6,
          'Hog': 5, 'Golem': 5, 'Ariete': 4, 'Giganti': 4, 'Royal Hogs': 4, 'Cimitero': 4, 'Barile/Trivella': 4, 'Mongolfiera': 4, 'Assedio': 4, 'Lava/Minatore': 4, 'Altri': 3.6}
COUNTS = {k: 8 if k == 'Fornace Cimitero' else 14 if v == 5.6 else 12 if v == 5 else 9 if v == 3.6 else 10 for k, v in SHARES.items()}
FURNACE_DECK = ['FirespiritHut', 'Knight', 'IceWizard', 'BarbLog', 'GoblinHut', 'Graveyard', 'Poison', 'Skeletons']
WC = {'Golem', 'Pekka', 'MegaKnight', 'HogRider', 'RamRider', 'BattleRam', 'Balloon', 'Graveyard', 'LavaHound', 'Xbow', 'Mortar', 'Giant', 'RoyalGiant', 'GoblinGiant', 'ElixirGolem', 'RoyalHogs', 'GoblinBarrel', 'Miner', 'Wallbreakers', 'GoblinDrill', 'SkeletonBalloon'}


def family(d):
    c = set(d['d']) & WC; two = len(c) >= 2
    if len(set(d['d']) & set(FURNACE_DECK)) >= 6 and 'Graveyard' in d['d']: return 'Fornace Cimitero'
    if 'MegaKnight' in c and c & {'HogRider', 'RamRider'}: return 'MK + Domatore/Domatrice'
    if 'MegaKnight' in c and d['f'][d['d'].index('MegaKnight')] == 1: return 'MK Evo'
    if {'Pekka', 'BattleRam'} <= c: return 'Pekka + Ariete'
    if 'Pekka' in c and two: return 'Pekka + altra'
    if 'LavaHound' in c and two: return 'Lava + altra'
    if 'Balloon' in c and two: return 'Mongolfiera + altra'
    if 'Mortar' in c and two: return 'Mortaio + altra'
    if two and c & {'GoblinBarrel', 'Wallbreakers', 'GoblinDrill'}: return 'Bait doppio'
    if two: return 'Altre doppie'
    if 'HogRider' in c: return 'Hog'
    if c & {'Golem', 'ElixirGolem'}: return 'Golem'
    if c & {'BattleRam', 'RamRider'}: return 'Ariete'
    if c & {'Giant', 'RoyalGiant', 'GoblinGiant'}: return 'Giganti'
    if 'RoyalHogs' in c: return 'Royal Hogs'
    if 'Graveyard' in c: return 'Cimitero'
    if c & {'GoblinBarrel', 'GoblinDrill'}: return 'Barile/Trivella'
    if c & {'Balloon', 'SkeletonBalloon'}: return 'Mongolfiera'
    if c & {'Xbow', 'Mortar'}: return 'Assedio'
    if c & {'LavaHound', 'Miner', 'Wallbreakers'}: return 'Lava/Minatore'
    return 'Altri'
