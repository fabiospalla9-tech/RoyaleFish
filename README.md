# RoyaleFish

A Clash Royale AI built from scratch by an independent developer. The idea is simple: something like Stockfish, but for Clash Royale. A strong engine that anyone can download, play against offline and study.

Everything runs offline, in a local simulation.

## New model in training (22 Nov 2026)

**A new generation is being trained from random weights.** It uses no earlier RoyaleFish or FirstLight checkpoint. It is still in training and is not published yet.

There are two networks of **12.67 million parameters** each:

- **A:** a Hog 2.6 specialist.
- **B:** a generalist that plays the current meta.

### Training time so far

| Step | Compute time |
|---|---|
| Imitation learning of B, from random weights | 3.7 h |
| Imitation learning of A, starting from B | 26 min |
| Self-play league, A and B together | 7.8 h |
| **Total** | **about 12.3 h** |

This does not count converting the replays, or a 3 h 20 min power cut. Hardware: one Tesla V100 and one RTX 5060 Ti. The league has played **about 150,000 games** so far, at about 21,000 games per hour, and every collected decision goes through the optimizer.

### How it is trained

- **Imitation first**, on real replays rebuilt in the open-source [RoyaleSim](https://github.com/RoyaleGym/RoyaleSim) simulator.
- **Then a self-play league** on RoyaleSim 0.1.55 with the 16.635 card table. A and B play each other, their own past versions and a frozen older model.
- **358 meta decks from November 2026**, covering all 123 ladder cards, 44 Evolutions and 18 Heroes.
- **Matchmaking:** about 40% of opponent decks come from the deck families the model is currently losing to.
- **What the model sees:** every unit, real Evo and Hero forms, spells in flight, the Evo cycle, ability buttons and the whole match history. Opponent cards appear only after they land, and its own commands take the game's 1 second delay.

### Results so far

**Fixed test: 1,000 Hog 2.6 mirror games against a frozen older model, on the 16.635 cards, with seeds not used in training.**

| Model | Win rate | Tower damage dealt per game | Tower damage taken per game |
|---|---|---|---|
| A before self-play | 47.6% ± 3.1 | 7,269 | 7,462 |
| **A after 103,424 self-play games** | **62.7% ± 3.0** | **9,149** | 8,152 |

**Imitation learning, on replays not used for training:**

| Metric | A | B |
|---|---|---|
| Correct card | 64.9% | 64.2% |
| Exact placement cell (top 1) | 32.6% | 30.3% |
| Placement in the top 5 cells | 65.8% | 65.8% |

**Hero abilities, learned in self-play (last 16 training rounds, average per game):**

| Ability | Uses | Useful uses |
|---|---|---|
| Hero Musketeer turret (A), pulling a win condition | 3.03 | 2.07 |
| Hero Ice Wizard cube (B), freezing a win condition or 2+ troops | 0.54 | 0.40 |
| Hero Electro Wizard surge (B), stopping a win condition or 2+ troops | 0.22 | 0.18 |

### Limits

- The frozen opponent in the fixed test is also in the training pool, so **62.7% can overstate general strength**.
- The new models **have not yet been measured against v995** or in a large test against human players.
- Training runs in a simulator, which is close to the real game but not identical.
- Clash Royale is not chess: some decks hard-counter others. In training, the Hog 2.6 model wins only about 11% against Furnace + Graveyard decks. Better play can narrow a bad matchup, but it cannot remove it.

**It will be published here once it is strong enough.**

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
