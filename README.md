# Squad

Four bodies, one brain. You order Wren. Ash, Bolt, and Nim walk the same weights.

A fresh brain walks the short way to the door, and that way goes through a pit. When you send Wren around it, the squad keeps that step. Spikes and cracks can inherit it. A dark mat is only a mat. It holds.

Later floors are a line of spikes, pits set further in, and two lines of cracks with mats for a gap.

Lessons stay on this computer. The factory brain shipped with the game is left alone.

## Getting started

You need **Python 3.8+**.

```bash
pip install -r requirements.txt
python game.py
```

On Windows, `start.bat` does the same.

To train the factory brain again:

```bash
python -m squad.train
```

## A round

- The gold ring is Wren. The outline on the floor is the brain's step.
- If that step is a pit, spikes, or a crack, send Wren around it. The others take the lesson before they move.
- Enter takes the brain's step as it stands. Into a danger, that falls.
- A floor is clear when everyone still standing is on the door.
- R sets the floor again. The lessons stay.

The correction is studied facing each side, and again as a pit, as spikes, and as a crack, so a shape Wren has not walked around can still share the step.

## Controls

| Action | Keys |
| --- | --- |
| Order Wren | Arrows or `WASD` |
| Wait | `Space` |
| Take the brain's step | `Enter` |
| Start, next floor, try again | `Enter` |
| Set the floor again | `R` |
| Forget lessons | On the title, **Forget lessons**, or `F2` |
| Quit | `Esc` |

## Project structure

```
squad/
├── game.py          # window, floor, and the order you give Wren
├── world.py         # floors, and what a body can see
├── brain.py         # the shared network, a lesson, load and save
├── train.py         # practice walks and the factory brain
├── settings.py      # window, colours, and the size of the network
└── weights.npz      # the factory brain, so the first step can be wrong on purpose
```
