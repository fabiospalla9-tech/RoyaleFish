# RoyaleFish

A Clash Royale AI built from scratch by an independent developer. The idea is simple: something like Stockfish, but for Clash Royale. A strong engine that anyone can download, play against offline and study.

Everything runs offline, in a local simulation.

## New model in training (22 Nov 2026)

A new generation is training **from random weights**, with no earlier checkpoint: first on real replays, then in a self-play league. It is not published yet.

| | |
|---|---|
| Models | Hog 2.6 specialist and meta generalist, 12.67M parameters each |
| Compute so far | **about 12.3 hours** (replay conversion not included) |
| Self-play games | **about 150,000** |
| Opponents | 358 meta decks from November 2026, all 123 ladder cards, 44 Evolutions, 18 Heroes |
| Test, 1,000 games vs a frozen older model | **47.6% before self-play, 62.7% after 103,424 games** (± 3) |

That older model is also in the training pool, so 62.7% can overstate general strength. The new models have not been measured against v995 yet.

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
