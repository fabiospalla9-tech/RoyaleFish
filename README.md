# RoyaleFish

A Clash Royale AI built from scratch by an independent developer. The idea is simple: something like Stockfish, but for Clash Royale. A strong engine that anyone can download, play against offline and study.

Everything runs offline, in a local simulation.

## New models (24 Nov 2026)

> [!WARNING]
> **Known issue (24 Nov 2026): the published 12.67M models were trained without the intended command delay.** Because of a bug in `training/rl_env.py`, the delay was set after the battle reset, so RoyaleSim ignored it and cards landed after 1 tick instead of about 1 second. Imitation learning on replays is not affected. All win rates below were measured under the same bug. With a real delay on its side only, the Minion Giant 2.8 drops from 68.2% against the frozen generalist to 56.5% with 1 s and to 36.2% with 1.5 s (400 games each). **A fix and retraining are in progress.**

A new generation trained **from random weights**, with no earlier checkpoint: first on real high-ladder replays, then in a self-play league in RoyaleSim at about **23,000-30,000 games per hour**. From random weights to the published Furnace specialist took **about 40 hours**.

| File | Role | Replays | Self-play games | Win rate vs frozen B (400 games) |
|---|---|---|---|---|
| `models/C-furnace-graveyard-v862.pt` | Furnace Graveyard specialist | 5,500 + 4,873 | about 221,000 | **66.0% ± 4.6** |
| `models/H-minion-giant-28-v342.pt` | Minion Giant 2.8 specialist | 3,169 | about 88,000 | **68.2% ± 4.6** |
| `models/B-general-12m-424k.pt` | Meta generalist | 56,049 | 424,192 | reference |

All three have 12.67M parameters. Against a frozen older 12.6M Hog 2.6 model, the specialists won **99.7%** (Furnace, 1,266 of 1,270) and **99.1%** (Minion Giant, 1,258 of 1,269) of their games in the last 100 training rounds.

Both opponents are in the training pool, so these numbers can overstate general strength. That older 2.6 is not the 17.5M v995 below, and no large test against v995 has been run yet.

B studied an older replay dataset, about 3-4 months behind the meta. The new [November dataset](https://huggingface.co/datasets/A1A2CV3/IL_Replay_1115) (156,160 games, 4-15 Nov) has so far been used only in part, for the two specialists. A new generalist from random weights on all of it, with every ability active, is under consideration.

**Full training procedure, data and limits: [docs/new-models-november-2026.md](docs/new-models-november-2026.md).**

**Replay conversion for the offline engine: [conversion/](conversion/README.md).**

**Reinforcement learning, league and rewards: [training/](training/README.md).**

## Models (previous generation)

| File | Deck | Card data | What it is |
|---|---|---|---|
| `models/A-hog26-v995.pt` | Hog 2.6 | 16.635 | Latest model of the previous generation (offline). |
| `models/B-general-v275.pt` | many decks | 16.635 | Generalist sparring partner. |

These are recurrent V4 actor-critic models (about 17M parameters) and load with `load_policy_v4` from the V4 framework (see Requirements). Run every model on the same 16.635 card data (`data_16635/`), otherwise one side sees the wrong cards and comparisons mean nothing.

## Requirements

- Python 3.12 with PyTorch and a CUDA GPU.
- The open-source V4 framework listed in `NOTICE`. The `royalefish/` package imports its `native_runner` modules.
- **A native battle engine, which you have to supply yourself.** This repository does not include any game binaries, resources, modified clients or engines, and never will.

Source changes for a 16.635 engine (source code only, no game files): `patches/native_runner_16635.diff`, `patches/probe_x86_64_16635.diff` and the card tables in `data_16635/`.

## License

Apache-2.0. See `LICENSE` and `NOTICE`. Not affiliated with, endorsed by, or representative of Supercell.

---

## ⛔ NOT FOR RANKED, LADDER OR ANY ONLINE PLAY

> [!CAUTION]
> **RoyaleFish must never be used in the real game.** Not in ladder, not in Path of Legends, not in tournaments, not on any account, for any reason.
>
> - It is a **research project**: it plays only offline, in a local simulation, against other models or against you.
> - Using an AI to play ranked games for you is **cheating**. It ruins the game for real people who are playing fair, and it breaks Supercell's Terms of Service.
> - **I do not support it, I condemn it, and I take no responsibility** for anyone who tries.
>
> **If I find out that this AI is being used to climb the ladder (with the 2.6 or anything else), the project goes private for good.**

### Do not contact me for:

- **paying me** for models, accounts, boosting or anything else;
- **the game engine** (it is not mine to share and it will not be shared);
- **tools that read a phone's memory or play on a real device** in place of a player.

**Messages like these will be blocked and reported.**
