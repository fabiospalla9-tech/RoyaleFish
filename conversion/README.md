# Replay conversion for the offline engine

These scripts turn high-ladder replays into RoyaleSim states and training arrays. They are the pipeline used for the 12.67M models. See [the training document](../docs/new-models-november-2026.md).

**Offline research only.** This code reads replay datasets and runs a local simulator. It does not connect to the game, a device or any server. The warning in the main README applies.

No replays, converted data, game files or engines are included.

## Files

| File | What it does |
|---|---|
| `adapt_1115.py` | Turns `IL_Replay_1115` payloads into the replay format read by the converter |
| `convert_replay.py` | Replays a match in RoyaleSim and records the winner's decisions: cards, cells and ability presses |
| `convert_deck.py` | Selects the won games of one exact deck and converts them in parallel |
| `prepare_dataset.py` | Packs converted games into compact training and validation arrays |

## Requirements

- Python 3.12 with `numpy` and `pyarrow`.
- [RoyaleSim](https://github.com/RoyaleGym/RoyaleSim) 0.1.55 (MIT).
- The `native_runner` package from the V4 framework listed in `NOTICE`, with `patches/native_runner_16635.diff` for card data 16.635. Point `NATIVE_RUNTIME` to the folder that contains it.
- A replay dataset, such as [`IL_Replay_1115`](https://huggingface.co/datasets/A1A2CV3/IL_Replay_1115).

`RF_TABLE=client160635007_20261103` is mandatory for 16.635. RoyaleSim's default table is older and refuses Hero Ice Wizard.

## Usage

Convert the won games of the Minion Giant 2.8, with Fireball or the Lightning variant:

```bash
export RF_TABLE=client160635007_20261103
export NATIVE_RUNTIME=/path/to/runtime

python conversion/convert_deck.py \
  --dataset /data/IL_Replay_1115/replays \
  --out /data/converted \
  --cards knight,ice-wizard-hero,cannon-ev1,bats-ev1,minion-giant,barbarian-barrel,skeletons \
  --last fireball:2.0,lightning:1.5 \
  --check IceWizard,MinionGiant \
  --workers 8
```

- `--cards` lists the seven fixed cards. Form matters: `-ev1` is the Evolution, `-hero` is the Hero.
- `--last` lists the allowed eighth cards and the weight of their games.
- `--check` lists simulator names that must be on the converted side.
- `--limit 40` converts only the first 40 games, useful as a quick test.

Finished blocks are skipped, so an interrupted run can be started again.

Then build the arrays:

```bash
python conversion/prepare_dataset.py /data/converted /data/training 8
```

About 10% of the games are held out for validation, chosen from each game's tag so the split stays the same between runs.

## What the conversion keeps

- The winner is always team 0. If the winner was the other player, the arena is mirrored.
- Each state keeps unit types, Evolution and Hero flags, stuns, buffs, spells in flight, projectiles, Evolution counters and ability buttons.
- A game is cut where the simulated hand stops matching the replay, so later decisions are not learned from a wrong state.

## Measured results

On the Minion Giant 2.8 in `IL_Replay_1115`:

| | |
|---|---|
| Won games selected | 3,601 (1,800 Fireball, 1,801 Lightning) |
| Converted | 3,533 (98.1%) |
| Training samples | 477,415 from 3,169 games |
| Validation samples | 53,919 from 364 games |
| Ability presses kept | 6,215 of 7,192 (86.4%) |

The simulator is close to the game but not identical to it, which is why some games and abilities are lost.
