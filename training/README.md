# Reinforcement learning, league and rewards

This is the self-play code used for the RoyaleFish 12.67M models. See [the training document](../docs/new-models-november-2026.md) for the full procedure and results.

**Offline research only.** Every game runs in the local RoyaleSim simulator. This code does not connect to the real game or to a device. The warning in the main README applies.

## Files

| File | What it does |
|---|---|
| `league.py` | Main loop: runs the games, chooses opponents and decks, updates both models |
| `league_server.py` | Decision server: answers for all models in large GPU batches |
| `league_common.py` | Actor-critic wrapper, sampling, evaluation and the anchor penalty |
| `rl_env.py` | Battle wrapper and **all the rewards** |
| `deck_shares.py` | Deck families and their share in the league |
| `league_decks_web.json` | The 358 meta decks of November 2026, collected from royaledeck.app |
| `pro_building_cells.json` | Real building placements, used by the optional predict drill |
| `setup_pool.py` | Creates the league folder from two published checkpoints |

## Requirements

Everything in [`conversion/README.md`](../conversion/README.md), plus PyTorch with a CUDA GPU.

`RF_TABLE=client160635007_20261103` is mandatory. Without it RoyaleSim uses an older card table and refuses Hero Ice Wizard.

## Quick start

Continue the Minion Giant 2.8 specialist against the generalist:

```bash
export RF_TABLE=client160635007_20261103
export NATIVE_RUNTIME=/path/to/runtime
export PYTHONPATH=/path/to/royalesim
export RF_POOL=$PWD/pool

python training/setup_pool.py --pool pool \
  --a models/H-minion-giant-28-v342.pt --b models/B-general-12m-424k.pt

python -u training/league_server.py 127.0.0.1 7792 &

A_INIT=pool/A_init.pt B_INIT=pool/B_init.pt \
RESUME0=models/H-minion-giant-28-v342.pt RESUME1=models/B-general-12m-424k.pt \
REMOTE=127.0.0.1:7792 GAMES=256 WARMUP=4 DRILL_P=0 \
BARREL_P=0.15 HARD_P=0.70 ICE_LIMIT=2 R_ICE_OVER=0.20 R_GB_LATE=0.05 \
A_NAME="Minion Giant" \
A_DECK='{"d":["Cannon","Bats","IceWizard","Knight","MinionGiant","BarbLog","Fireball","Skeletons"],"f":[1,1,2,0,0,0,0,0]}' \
python -u training/league.py runs/h 30 64
```

The last two numbers are CPU workers and battles per worker. Lower them on a smaller machine. Forms in `A_DECK` are `0` normal, `1` Evolution and `2` Hero.

Progress goes to `runs/h/train.jsonl`, one line per round. Checkpoints are `A_latest.pt` and `B_latest.pt`, plus one every 100 rounds.

## Main settings

| Setting | Value |
|---|---|
| Games per round | 256 |
| Learning rate | 3.3e-6 |
| Discount / GAE lambda | 0.999 / 0.95 |
| PPO clip / value weight | 0.2 / 0.5 |
| Anchor | KL penalty towards the imitation model, target 0.10 |
| Past versions | Saved every 10 rounds, the latest 6 are kept |

| Environment variable | Meaning |
|---|---|
| `A_DECK` | Fixed deck of model A. Without it A plays the Hog 2.6 |
| `MIX` | Share of each match type |
| `HARD_P` | Share of opponent decks from the families the model struggles against |
| `BARREL_P` | Share of opponent decks with a Goblin Barrel |
| `ICE_LIMIT` | Hero Ice Wizard presses per game before each extra one is punished |
| `R_ICE_OVER` | Penalty for each press over the limit |
| `R_GB_LATE` | Penalty for answering a Goblin Barrel late |
| `REMOTE` | Addresses of the decision servers |

## Difference from the original runs

The original runs also played 10% of their games against a frozen older 12.6M Hog 2.6, which is **not published**. The default `MIX` here leaves it out:

```text
Original: 55% A vs B, 5% A vs old, 5% B vs old, 20% A vs past, 15% B vs past
Default:  60% A vs B, 22% A vs past, 18% B vs past
```

Setting `a_old` or `b_old` in `MIX` stops with an error. Results with the default mix are therefore not directly comparable with the published ones.

## Rewards

Rewards are zero-sum: what one side gains, the other loses. Most are paid at most twice per game. All values are constants at the top of `rl_env.py`.

- **Match:** winning and tower damage.
- **Graveyard:** with a tank or Poison, on the weaker tower, when the opponent is low on elixir. Penalties when alone or into counters.
- **Spells:** Poison on swarms and support, Fireball predictions, and a bonus for one Fireball hitting two key buildings or troops.
- **Hero Ice Wizard:** +0.15 for freezing a win condition or two troops, -0.10 for freezing nothing, plus rules for the right targets and timing.
- **Barbarian Barrel and Log against Goblin Barrel:** +0.30 for killing every Goblin before it hits the tower, with an optional penalty for answering late.
- **Buildings and placement:** kiting win conditions, avoiding stacked cards, keeping the Ice Wizard alive.
- **Forced first card:** with no card played at 10 elixir, the model must play one, choosing card and cell itself. Forced moves are not studied.

## Tested

The published code was run from a clean copy with RoyaleSim 0.1.55 on one machine: 8 games with the Minion Giant deck and one full model update, with no errors. A complete training run has not been repeated from this cleaned version.
