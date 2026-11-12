# RoyaleFish

An open, community-driven Clash Royale AI. The idea is simple: something like Stockfish, but for Clash Royale. A strong engine that anyone can download, play against, study and make stronger. When someone trains a better model, it gets added here and becomes the new baseline for everyone.

Everything runs offline, in a local simulation.

## Models

| File | Deck | What it is |
|---|---|---|
| `models/A-hog26-v98.pt` | Hog 2.6 | Hog 2.6 specialist. Started from a strong pretrained 17M-parameter model, then 98 PPO updates on top. |
| `models/B-general-v49.pt` | many decks | Generalist. 49 PPO updates on top of a pretrained general model. It plays many different decks and is A's main sparring partner. |

Both are the same network: a recurrent V4 actor-critic (transformer over cards and units, spatial encoder for the arena, LSTM memory), about 17M parameters. They load with `load_policy_v4` from the V4 framework (see Requirements).

How much did the fine-tuning help? Two tests of A at update 98 against its own starting checkpoint:

| Test | Games | A v98 | Starting checkpoint |
|---|---|---|---|
| Head to head (each side plays Hog 2.6 in turn, the other plays varied decks) | 1,056 | **54.3%** | 45.7% |
| Both play Hog 2.6 against the same opponent (the original generalist, same deck sequence) | 600 / 561 | **59.7%** win rate | 49.6% win rate |

So against the same opponent, the fine-tuned model wins about 10 percentage points more often than the checkpoint it started from.

## Coming soon

A new release is coming in the next few hours. The models are being moved to the card data of game version 16.635, so they will:

- recognize the **Minion Giant**, which the old card catalog did not know (the models saw it as an "unknown card");
- know the newer 16.635 mechanics;
- be trained harder against the decks the Hog 2.6 model struggled with: Mortar, Evo Witch and Minion Giant decks.

Support for **Hero Ice Wizard** is planned next.

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
