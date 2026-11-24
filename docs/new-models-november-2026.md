# RoyaleFish 12.67M models: how they were trained (November 2026)

This document describes the three models published on 24 November 2026, how they were built and what was actually measured.

**These models are for offline research only.** They must never be used in the real game. See the warning in the main README.

> [!WARNING]
> **Known issue (24 Nov 2026): the published 12.67M models were trained without the intended command delay.** Because of a bug in `training/rl_env.py`, the delay was set after the battle reset, so RoyaleSim ignored it and cards landed after 1 tick instead of about 1 second. Imitation learning on replays is not affected. All win rates below were measured under the same bug. With a real delay on its side only, the Minion Giant 2.8 drops from 68.2% against the frozen generalist to 56.5% with 1 s and to 36.2% with 1.5 s (400 games each). **A fix and retraining are in progress.**

## 1. The three models

| File | Role | Deck | Replays studied | Self-play games |
|---|---|---|---|---|
| `models/B-general-12m-424k.pt` | Generalist | Many meta decks | 56,049 (older dataset) | 424,192 |
| `models/C-furnace-graveyard-v862.pt` | Specialist | Furnace Graveyard | 5,500 + 4,873 | about 221,000 |
| `models/H-minion-giant-28-v342.pt` | Specialist | Minion Giant 2.8 | 3,169 | about 88,000 |

The specialist game counts are rounds × 256 games.

**Furnace Graveyard deck:** Evo Furnace, Evo Knight, Hero Ice Wizard, Barbarian Barrel, Goblin Hut, Graveyard, Poison, Skeletons.

**Minion Giant 2.8 deck:** Evo Cannon, Evo Bats, Hero Ice Wizard, Knight, Minion Giant, Barbarian Barrel, Fireball, Skeletons.

## 2. Started from random weights

The whole family starts **from zero**. The generalist B was created with random weights, with no earlier checkpoint and no weights from older RoyaleFish or Everlight models.

The two specialists are copies of that generalist after its 424,192 self-play games. They then specialize on their own deck.

Timeline, in elapsed time:

| When | What |
|---|---|
| 22 Nov, about 05:00 | B starts imitation learning from random weights |
| 22 Nov, 11:50 | League moves to RoyaleSim 0.1.55 with the 16.635 card table |
| 23 Nov, 08:47 | B reaches 424,192 self-play games |
| 23 Nov, 09:33 | Furnace specialist forks from B |
| 23 Nov, 21:04 | Furnace specialist reaches round 862 (published version) |
| 23 Nov, 21:12 | Minion Giant specialist forks from B |
| 24 Nov, 02:29 | Minion Giant specialist reaches round 342 (published version) |

From random weights to the published Furnace model took **about 40 hours**, under two days. The Minion Giant specialist then took about 5 hours on top of B, including a power cut of about 85 minutes.

## 3. Results

All fixed tests use 400 games with RoyaleSim 0.1.55 and the 16.635 card table. The specialist plays its own deck. The opponent is the **frozen** generalist B at 424,192 games, playing decks drawn from the 358 meta decks, with sides alternated. Margins are 95% intervals.

### Furnace Graveyard

| Version | Wins | Losses | Draws | Win rate |
|---|---|---|---|---|
| Before self-play | 179 | 156 | 65 | 44.8% ± 4.9 |
| After the second imitation | 215 | 160 | 25 | 53.8% ± 4.9 |
| Round 374 | 252 | 148 | 0 | 63.0% ± 4.7 |
| Round 625 | 243 | 151 | 6 | 60.8% ± 4.8 |
| Round 802 | 273 | 124 | 3 | 68.2% ± 4.5 |
| **Round 862 (published)** | **264** | **134** | **2** | **66.0% ± 4.6** |

Round 862 is published instead of 802 because it uses the Hero Ice Wizard ability about 1.9 times per game, close to the 1.7 of real players, against 2.55 at round 802. The difference between the two win rates is inside the margin of error.

### Minion Giant 2.8

| Version | Wins | Losses | Draws | Win rate |
|---|---|---|---|---|
| Generalist B playing this deck | 104 | 219 | 77 | 26.0% ± 4.7 |
| After imitation | 225 | 173 | 2 | 56.2% ± 4.9 |
| Round 116 | 246 | 154 | 0 | 61.5% ± 4.8 |
| Round 225 | 254 | 145 | 1 | 63.5% ± 4.7 |
| **Round 342 (published)** | **273** | **125** | **2** | **68.2% ± 4.6** |

### Against a frozen older Hog 2.6

During training, both specialists regularly played a frozen older **12.6M-parameter** Hog 2.6 model:

| Specialist | All rounds | Last 100 rounds |
|---|---|---|
| Furnace Graveyard | 11,097 wins in 11,199 games (**99.1%**) | 1,266 in 1,270 (**99.7%**) |
| Minion Giant 2.8 | 4,424 wins in 4,464 games (**99.1%**) | 1,258 in 1,269 (**99.1%**) |

### Limits of these numbers

- The frozen B and the frozen older 2.6 are both part of the training pool, so the specialists have seen them. These results can overstate general strength.
- The older 12.6M model above is **not** the published 17.5M `A-hog26-v995.pt`. No large test against v995 has been run yet.
- The fixed tests do not use the forced first card described in section 6, while training does.
- 400 games leave a margin of about ±5 points. Small differences between versions are not meaningful.
- All results come from a simulator, which is close to the game but not identical to it. None of them measures strength against human players.

## 4. Datasets

**Older dataset (`IL_Replay`).** 252,238 anonymized high-ladder replays, about 3-4 months older than the current meta. B studied 56,049 of them, with 6,277 more held out for validation. They contain 854,195 ability activations, but **no Hero Ice Wizard**, so B learned that ability only in self-play.

**November dataset ([`IL_Replay_1115`](https://huggingface.co/datasets/A1A2CV3/IL_Replay_1115)).** 156,160 games played from 4 to 15 November 2026 on version 16.635, with about 11 million actions. **So far it has been used only in part**, for the two specialists:

- Furnace Graveyard: 4,873 converted games of the exact deck and close variants.
- Minion Giant 2.8: 3,533 converted winning games of the exact deck, with Fireball or the Lightning variant. Of these, 3,169 were used for training and 364 held out. The conversion kept 6,215 of 7,192 ability activations.

**Next step under consideration:** a new generalist **from random weights on all 156,160 November replays**, with every Hero and Evolution ability active, so that it learns the current meta directly. It has not been started.

## 5. Model and infrastructure

- **Network:** transformer with 384 dimensions, 6 layers, 8 heads and a 1,536-unit feed-forward block, **12.67M parameters**. It has separate outputs for waiting or playing, card choice, placement cell and hero abilities, plus a value head in the checkpoints.
- **Simulator:** RoyaleSim 0.1.55 with the 16.635 card table.
- **Speed:** about **23,000-30,000 full games per hour**, using 30 CPU workers with 64 battles each.
- **GPUs:** RTX 5060 Ti and Tesla V100, used for both inference and training.
- **Opponent decks:** 358 meta decks from November 2026, covering 123 ladder cards, 44 Evolutions and 18 Heroes.

## 6. Training procedure

### Step 1: imitation learning

Each model first studies human replays, converted to simulator states with the [conversion pipeline](../conversion/README.md). Specialists start from B with learning rate 1e-4 for 3 epochs, plus an "elastic" penalty of 0.3 that keeps them close to their starting point. Validation always uses games the model has never seen.

Minion Giant 2.8 on 364 unseen games:

| Metric | Before | After |
|---|---|---|
| Correct card | 55.8% | 63.5% |
| Exact cell | 22.7% | 31.5% |
| Correct cell in top 5 | 58.8% | 67.1% |
| Ability moments recognized | 23.4% | 25.9% |

### Step 2: self-play league

The code is in [`training/`](../training/README.md). Each round plays 256 games and updates the models with a PPO-style method. Opponents are mixed to avoid overfitting to a single rival:

| Share | Match |
|---|---|
| 55% | Specialist vs current generalist |
| 20% | Specialist vs its own or the generalist's past versions |
| 15% | Generalist vs past versions |
| 5% | Specialist vs frozen older 2.6 |
| 5% | Generalist vs frozen older 2.6 |

For the Minion Giant specialist, about **60% of opponent decks come from the families it struggles against most**, and 15% contain a Goblin Barrel.

### Step 3: rewards from strategy guides

Besides winning and tower damage, small capped rewards teach specific plays:

- **Graveyard:** placed with a tank or Poison, on the weaker tower, when the opponent is low on elixir.
- **Hero Ice Wizard:** freezing win conditions or several troops; penalties for pressing it with nothing to freeze.
- **Ability limit:** the freeze bonus is paid at most twice per game, and each press from the third onward costs 0.20. This brought usage from about 4.5 to about 2 per game, while the fixed test stayed above 60%.
- **Barbarian Barrel against Goblin Barrel:** +0.30 for killing all the Goblins before they hit the tower, and a small penalty for answering late.
- **Forced first card:** with no card played at 10 elixir, the model must play one, choosing card and cell itself. Before this rule, models often waited for each other.

## 7. Known weaknesses

- **Barbarian Barrel timing:** the Minion Giant specialist stops about 26 of 100 Goblin Barrels thrown at its tower, up from 17 before the late-answer penalty. This is the next thing to improve.
- **Hero Ice Wizard:** about one press in four still hits nothing.

## 8. Loading a model

```bash
python -m royalefish12m.load models/H-minion-giant-28-v342.pt
```

```python
from royalefish12m import load_model

net, tau = load_model("models/H-minion-giant-28-v342.pt", device="cuda")
```

This loads the network and its decision threshold. The code that turns a battle state into model input is not included yet.
