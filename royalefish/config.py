"""Settings shared by collectors and servers. Override with environment variables."""
import os

HOG26_DECK_ID = '523b6428074d4689904add37912722f47182bc6a1679dd89c5d50a19292263d3'
POOL = os.environ.get('RF_DECK_POOL', 'runs/deck-pool.json')        # V4 deck-pool manifest for the opponents
LEARNER_DECK_ID = os.environ.get('RF_LEARNER_DECK', HOG26_DECK_ID)    # deck played by model A
BATTLE_LEVEL = int(os.environ.get('RF_BATTLE_LEVEL', '11'))
GAMEPLAY_END_TICK = 6000
BASE_LATENCY_TICKS = 1
RESET = dict(render_mode='headless', decision_ticks=5, event_driven_decisions=False, reuse_runner_attestation=True)
