# RoyaleFish

An open, community-driven Clash Royale AI. The idea is simple: something like Stockfish, but for Clash Royale. A strong engine that anyone can download, play against, study and make stronger. When someone trains a better model, it gets added here and becomes the new baseline for everyone.

Everything runs offline, in a local simulation.

## Models

| File | Deck | Card data | What it is |
|---|---|---|---|
| `models/A-hog26-v738.pt` | Hog 2.6 | **16.635** | **Latest.** Hog 2.6 specialist after 738 PPO updates. Knows the Minion Giant and the Hero Ice Wizard (see below). |
| `models/B-general-v275.pt` | many decks | **16.635** | Latest generalist, A's main sparring partner, 275 PPO updates. |
| `models/A-hog26-v98.pt` | Hog 2.6 | 15.535 | Older Hog 2.6 specialist (98 updates), kept as a reference. |
| `models/B-general-v49.pt` | many decks | 15.535 | Older generalist (49 updates). |

Both are the same network: a recurrent V4 actor-critic (transformer over cards and units, spatial encoder for the arena, LSTM memory), about 17M parameters. They load with `load_policy_v4` from the V4 framework (see Requirements).

How much did the fine-tuning help? Two tests of A at update 98 against its own starting checkpoint:

| Test | Games | A v98 | Starting checkpoint |
|---|---|---|---|
| Head to head (each side plays Hog 2.6 in turn, the other plays varied decks) | 1,056 | **54.3%** | 45.7% |
| Both play Hog 2.6 against the same opponent (the original generalist, same deck sequence) | 600 / 561 | **59.7%** win rate | 49.6% win rate |

So against the same opponent, the fine-tuned model wins about 10 percentage points more often than the checkpoint it started from.

Later versions, same test (both play Hog 2.6 against the original generalist, same deck sequence):

| Model | Games | Win rate vs original generalist |
|---|---|---|
| Starting checkpoint | 561 | 49.6% |
| A v98 | 668 | 61.2% |
| A v357 | 668 | **73.1%** |
| A v738 (latest) | — | **estimated 75–80%, not yet verified** |

A v357 more than doubled the gain over the starting checkpoint (+23.5 points, against +11.6 for v98). The v738 figure is an estimate from in-training results only; a paired measurement will be posted here soon.

## What's new in v738

**New card data (game version 16.635).** The original card catalog stopped at 122 cards, so the models saw newer content as an "unknown card". The catalog is now rebuilt from 16.635 data:

- **Minion Giant** is a real card for the model: it knows it flies, targets only buildings, costs 4 and how much HP it has. Its new embedding starts from the Balloon's.
- **Hero Ice Wizard** is supported, with its ability (`IceWizardHero_ability`). This one is **still early**: the ability is offered and used, but the model has only just started learning when to use it.
- The newer 16.635 mechanics (new effects and abilities) are in the catalog too.
- All 122 cards the model already knew keep their learned weights: they were moved to the new table row by row, nothing was retrained from scratch.

Loading these models needs the 16.635 tables in `data_16635/` and the small framework patch in `patches/native_runner_16635.diff` (wider catalogs, 26 ability slots, 453 unit archetypes). Point `CR_COMPETITIVE_DATA_ROOT` at `data_16635/`.

**Training changes:**

- **Musketeer turret.** Hero abilities were never offered to the models in the x86 engine; that is fixed. A also gets a small reward for using the Hero Musketeer turret at the right moment (a building-targeting win condition near its tower and no building of its own on the field). Using the turret at the right moment went from about **13% to 37%** of the chances, and the model now often plays two defensive structures (Cannon plus turret), which helps a lot against double win conditions.
- **Opponent spell reward.** The opponent gets a small reward when its Fireball, Arrows or Lightning kills A's Musketeer, so A learns to protect it.
- **Tower damage.** The per-hit tower damage term was raised, so small chip damage on the other lane is no longer ignored.
- **Focus decks.** About 30% of games are played against the decks A struggled with: Mortar, Evo Witch and Minion Giant decks, mostly real meta lists.

In-training win rates of A (last hour of training, not a paired evaluation):

| Opponent | Win rate |
|---|---|
| Original generalist (frozen) | about 59% |
| Older generalist B v50 (frozen) | about 55% |
| Minion Giant decks | 32% → **48%** |
| Mortar decks | 39% → **48%** |
| Evo Witch decks | 32% → **45%** |

## Known weaknesses of A v738

Found by playing against the model. We are working on all of them right now.

- **Wasting the only counter.** The biggest one. When A has a single card that answers two of the opponent's cards, it often spends it on the first, less dangerous threat, and then has nothing left for the real one. Examples: The Log on the Princess, then the Goblin Barrel hits the tower; the Cannon placed right at the bridge against Mega Knight + Witch, where it dies at once, then the Hog Rider comes in with nothing to stop it. (The original starting model had this problem even more clearly.)
- **Log Bait.** Related to the point above: it does not keep The Log for the Goblin Barrel, and against the Evo Goblin Barrel on two lanes it tends to defend the tower with more HP instead of the weaker one.
- **Ignoring damage on the other lane.** Sometimes it defends one lane perfectly and lets 1,500+ damage through on the other.
- **Splash damage.** Against Witch / Evo Witch it places the Musketeer right next to its princess tower, so the splash hits both; against Sparky it groups Musketeer and Ice Golem together, and it uses Fireball on Sparky too rarely.
- **Giant Skeleton.** It does not always stop the death bomb from reaching the tower.
- **Musketeer turret.** It already uses it much more, but still not in every spot where it would be better than playing another card.

How we are fixing them: harder opponents that punish these mistakes on purpose (a Log Bait specialist and a model trained only to beat the 2.6), focus decks, and drills rebuilt from the exact moments where the model failed, replayed many times with small variations.

## In progress

A new model is being trained **only to beat the latest Hog 2.6**. It starts from the latest generalist, learns a Minion Giant + Hero Ice Wizard deck by imitation from about 600 high-level replays, then trains by self-play against the frozen Hog 2.6 model and its older versions. Results will be posted here.

## How training works

A and B train against each other at the same time, and both keep improving. The interesting part is how much data fits through one machine. On a single PC with an RTX 5060 Ti (for playing) and a Tesla V100 (for learning), the GPU learns from **about 2,100 complete games per hour**. Each update trains on **~55,000 decisions from ~64 full games** in a single optimizer step and takes about 70–90 seconds.

That speed comes from a few changes:

1. **Native engine, no emulators.** Matches run in a resident native battle engine (16 matches per engine process, 32 processes, 512 matches at once) instead of Android emulators.
2. **Fast inference servers** (`ever_server.py`). The collectors send observations as one packed byte block per move. The server runs the model with CUDA graphs captured at several batch sizes, in fp16 (checked to choose the same actions as fp32), and keeps every match's LSTM state in a single GPU table, so it is read and written once per batch instead of once per match.
3. **Pipelined collectors** (`ever_collector.py`). Each collector splits its matches in two halves: while one half waits for the model, the other advances the engine. Moves are placed exactly where the model chose. If a cell is not legal under the game rules, the move is simply not played; it is never moved to a nearby cell.
4. **Sparse trajectories.** Finished games are stored losslessly in a sparse format (about 8 KB per move instead of ~180 KB).
5. **Row-packed learner** (`ever_learner.py` and `ever_packed.py`). Trajectories are handed to the PPO update in the packed, one-buffer-per-dtype layout the update code is optimized for. Before this, the update spent half its time rebuilding observations move by move on the CPU, and switching gave a **4x faster update**.
6. **Loader in a separate process** (`ever_loader.py`). The next group of games is decoded in another process and shared through memory while the GPU trains on the current group. Games are grouped by length, so very little compute goes to padding.
7. **One big step per update.** Four groups of 16 games accumulate gradients, then one clipped optimizer step, which matches the size of the original large-scale PPO updates.
8. **KL anchor.** A small KL term towards the starting checkpoint, computed on half of the groups, keeps the models from forgetting what they already know.
9. **Opponent choice.** B's decks are chosen more often from the families A loses to, plus 10% mirror games against A itself.

## Our machine

Everything above was trained on one home PC:

| Part | What | Used for |
|---|---|---|
| CPU | 2 × Intel Xeon E5-2695 v4 (2 × 18 cores, 72 threads) | the native battle engines and the collectors (30 engines × 16 games = 480 games at once) |
| RAM | 128 GB | engines, collectors and replay loading |
| GPU 1 | NVIDIA RTX 5060 Ti 16 GB | inference: plays the moves for all running games (fp16, CUDA graphs) |
| GPU 2 | NVIDIA Tesla V100 16 GB | training: PPO updates and imitation learning |

Most of the speed comes from the many CPU cores: every game runs on the CPU, and the GPUs answer hundreds of games in one batch.

## Requirements

- Python 3.12 with PyTorch and a CUDA GPU.
- The open-source V4 framework listed in `NOTICE`. The `royalefish/` package imports its `native_runner` modules.
- **A native battle engine, which you have to supply yourself.** This repository does not include any game binaries, resources, modified clients or engines, and never will.

The scripts are configured with command-line arguments and the `RF_*` environment variables in `royalefish/config.py`. Each script's `--help` lists its options.

## Contributing a model

If you train something stronger, on any deck, open a pull request. Please include:

- the checkpoint (or a link to it if it's too big) and the deck it plays;
- a head-to-head result against the current model for that deck: at least 400 games, with both sides swapped;
- a short note on how you trained it (starting point, number of updates, anything unusual).

If it's better, it goes in the table above and becomes the new baseline. Specialists for other decks are just as welcome. The long-term goal is an offline training mode where you can practice against the AI at different difficulty levels.

## License

Apache-2.0. See `LICENSE` and `NOTICE`. Not affiliated with, endorsed by, or representative of Supercell.
