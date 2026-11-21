# RoyaleFish

A Clash Royale AI built from scratch by an independent developer. The idea is simple: something like Stockfish, but for Clash Royale. A strong engine that anyone can download, play against offline and study.

Everything runs offline, in a local simulation.

## New model in training (21 Nov 2026)

**A new Hog 2.6 model is being trained from zero.** It does not start from any previous checkpoint.

- **About 12 million parameters.** A transformer that sees every troop and building on the arena, its own hand, elixir and the towers.
- **Memory of the whole match.** Every card played by both players since the start, with where and how long ago, so it knows the opponent's cycle.
- **Human reaction time.** It sees the arena with a 1 second delay, like a real player. Its own hand and elixir are always up to date.
- **Imitation learning first**, on **7,652 winning Hog 2.6 games** from real players. The replays are rebuilt in the open-source RoyaleSim simulator, which converts all of them in about 3.5 minutes.
- **Reinforcement learning next**, against a generalist sparring model and against its own older versions, with regular test games in the real offline engine.

**It will be published here once it is strong enough.**

## Models

| File | Deck | Card data | What it is |
|---|---|---|---|
| `models/A-hog26-v995.pt` | Hog 2.6 | 16.635 | Latest model of the previous generation (offline). |
| `models/A-hog26-v1881.pt` | Hog 2.6 | 16.635 | Previous release, strong defender. |
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
