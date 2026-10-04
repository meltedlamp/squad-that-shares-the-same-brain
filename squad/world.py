"""Floors, what a body can see, and the step a fresh brain was trained to copy."""

import math

import numpy as np

from .settings import COLS, DIRS, DIR_FEATURES, F_OCC, IN_DIM, ROWS, SQUAD, WAIT

WALL, FLOOR, PIT, SPIKE, CRACK, MAT, GOAL = range(7)
HAZARDS = {PIT, SPIKE, CRACK}
KIND = {PIT: "pit", SPIKE: "spikes", CRACK: "crack"}

_CHARS = {
    "#": WALL,
    ".": FLOOR,
    "o": PIT,
    "^": SPIKE,
    "~": CRACK,
    "=": MAT,
    "G": GOAL,
    "1": FLOOR,
    "2": FLOOR,
    "3": FLOOR,
    "4": FLOOR,
}


def _line(body):
    row = "#" + body + "#"
    if len(row) != COLS:
        raise ValueError(f"row has {len(row)} cells: {row}")
    return row


def _stamp(base, marks):
    chars = list(base)
    for index, mark in marks.items():
        chars[index] = mark
    return "".join(chars)


def _marked(fill, marks):
    return _line(_stamp(fill * 14, marks))


def _floor_rows():
    edge = "#" * COLS
    open_goal = _line("." * 13 + "G")
    lane = lambda mark: _line(mark + "o" + "." * 11 + "G")
    pits = [
        edge,
        lane("1"),
        open_goal,
        lane("2"),
        open_goal,
        lane("3"),
        open_goal,
        lane("4"),
        open_goal,
        open_goal,
        open_goal,
        edge,
    ]
    haz = _stamp("." * 14, {0: "o", 4: "^", 8: "~", 11: "o"})
    units = _stamp("." * 14, {0: "1", 4: "2", 8: "3", 11: "4"})
    shapes = [edge, _line("G" * 14), _line(haz), _line(units)]
    shapes.extend(_line("." * 14) for _ in range(7))
    shapes.append(edge)
    cracks = _stamp("~" * 14, {2: "=", 6: "=", 10: "=", 13: "="})
    spread = _stamp("." * 14, {0: "1", 4: "2", 8: "3", 11: "4"})
    mats = [edge, _line("G" * 14), _line(cracks), _line(spread)]
    mats.extend(_line("." * 14) for _ in range(7))
    mats.append(edge)
    blank = _line("." * 14)
    spike_line = [
        edge,
        _line("G" * 14),
        _marked(".", {0: "^", 4: "^", 8: "^", 11: "^"}),
        _marked(".", {0: "1", 4: "2", 8: "3", 11: "4"}),
    ]
    spike_line.extend(blank for _ in range(7))
    spike_line.append(edge)
    deeper = [edge]
    for mark in "1234":
        deeper.append(_line("." * 3 + mark + "o" + "." * 8 + "G"))
        deeper.append(_line("." * 13 + "G"))
    deeper.extend(_line("." * 13 + "G") for _ in range(2))
    deeper.append(edge)
    gate = _stamp("~" * 14, {2: "=", 6: "=", 10: "=", 13: "="})
    two_lines = [
        edge,
        _line("G" * 14),
        _line(gate),
        _line(gate),
        _marked(".", {0: "1", 4: "2", 8: "3", 11: "4"}),
    ]
    two_lines.extend(blank for _ in range(6))
    two_lines.append(edge)
    floors = (
        ("The pit", "Pits sit on the straight walk to the door. You order Wren. The others take the lesson.", pits),
        (
            "Other shapes",
            "The door is north. Spikes and a crack stand where a pit stood. The pit's lesson is what they have.",
            shapes,
        ),
        (
            "The mat",
            "Cracks make a line. The dark mats are floor. They hold.",
            mats,
        ),
        (
            "The spike line",
            "Spikes under the door. The pit's lesson is already in them.",
            spike_line,
        ),
        (
            "Deeper",
            "The pits sit further in. The door is still east.",
            deeper,
        ),
        (
            "Two lines",
            "Two lines of cracks. The mats are the gap. They hold.",
            two_lines,
        ),
    )
    for name, _blurb, rows in floors:
        if len(rows) != ROWS:
            raise ValueError(f"{name} has {len(rows)} rows")
    return floors


FLOORS = _floor_rows()


def cell_features(tile):
    """Wall, hazard, roundness, edge, door, occupied."""
    if tile == WALL:
        return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if tile == PIT:
        return (0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    if tile == SPIKE:
        return (0.0, 1.0, 0.08, 1.0, 0.0, 0.0)
    if tile == CRACK:
        return (0.0, 1.0, 0.45, 0.35, 0.0, 0.0)
    if tile == MAT:
        return (0.0, 0.0, 0.18, 0.0, 0.0, 0.0)
    if tile == GOAL:
        return (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def kind_name(tile):
    return KIND.get(tile)


class Unit:
    def __init__(self, name, color, leader, x, y):
        self.name = name
        self.color = color
        self.leader = leader
        self.x = x
        self.y = y
        self.home = (x, y)
        self.alive = True
        self.face = 1

    def reset(self):
        self.x, self.y = self.home
        self.alive = True
        self.face = 1


class Grid:
    def __init__(self, rows):
        self.tiles = np.full((ROWS, COLS), WALL, dtype=np.int64)
        self.spawns = {}
        for y, row in enumerate(rows):
            for x, char in enumerate(row):
                if char in "1234":
                    self.spawns[int(char)] = (x, y)
                self.tiles[y, x] = _CHARS[char]
        if sorted(self.spawns) != [1, 2, 3, 4]:
            raise ValueError("a floor needs spawns 1, 2, 3, and 4")
        self.goals = {
            (x, y)
            for y in range(ROWS)
            for x in range(COLS)
            if int(self.tiles[y, x]) == GOAL
        }
        if not self.goals:
            raise ValueError("a floor needs a door")

    def units(self):
        made = []
        for index, (name, color) in enumerate(SQUAD):
            x, y = self.spawns[index + 1]
            made.append(Unit(name, color, index == 0, x, y))
        return made

    def tile(self, x, y):
        return int(self.tiles[y, x])

    def observe(self, x, y, occupied):
        return observe(self.tiles, x, y, self.goals, occupied)

    def legal(self, x, y, occupied):
        return legal_mask(self.tiles, x, y, occupied)


def observe(tiles, x, y, goals, occupied=()):
    """The 4 neighbors, then the direction and distance to the nearest door."""
    occupied = set(occupied)
    height, width = tiles.shape
    values = np.empty(IN_DIM, dtype=np.float32)
    cursor = 0
    for dx, dy in DIRS:
        nx, ny = x + dx, y + dy
        if nx < 0 or ny < 0 or nx >= width or ny >= height:
            block = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            block = cell_features(int(tiles[ny, nx]))
            if (nx, ny) in occupied:
                block = block[:F_OCC] + (1.0,)
        values[cursor : cursor + DIR_FEATURES] = block
        cursor += DIR_FEATURES
    dx, dy, dist = _goal_vector(x, y, goals, width, height)
    values[cursor] = dx
    values[cursor + 1] = dy
    values[cursor + 2] = dist
    return values


def _goal_vector(x, y, goals, width, height):
    if not goals:
        return 0.0, 0.0, 0.0
    gx, gy = min(goals, key=lambda goal: (goal[0] - x) ** 2 + (goal[1] - y) ** 2)
    span = math.hypot(width, height)
    return (
        (gx - x) / max(1, width - 1),
        (gy - y) / max(1, height - 1),
        math.hypot(gx - x, gy - y) / span,
    )


def legal_mask(tiles, x, y, occupied):
    mask = np.zeros(WAIT + 1, dtype=bool)
    mask[WAIT] = True
    occupied = set(occupied)
    height, width = tiles.shape
    for action, (dx, dy) in enumerate(DIRS):
        nx, ny = x + dx, y + dy
        if nx < 0 or ny < 0 or nx >= width or ny >= height:
            continue
        if int(tiles[ny, nx]) == WALL:
            continue
        if (nx, ny) in occupied:
            continue
        mask[action] = True
    return mask


def destination(x, y, action):
    if action == WAIT or action is None:
        return x, y
    dx, dy = DIRS[action]
    return x + dx, y + dy


def naive_action(tiles, x, y, goals):
    """First step of a shortest walk that treats pits, spikes, and cracks as floor."""
    if int(tiles[y, x]) == GOAL:
        return WAIT
    dist = _distances(tiles, goals)
    here = int(dist[y, x])
    if here < 0:
        return WAIT
    best = None
    height, width = tiles.shape
    for action, (dx, dy) in enumerate(DIRS):
        nx, ny = x + dx, y + dy
        if nx < 0 or ny < 0 or nx >= width or ny >= height:
            continue
        if int(tiles[ny, nx]) == WALL:
            continue
        if int(dist[ny, nx]) == here - 1 and best is None:
            best = action
    return WAIT if best is None else best


def _distances(tiles, goals):
    height, width = tiles.shape
    dist = np.full((height, width), -1, dtype=np.int16)
    queue = []
    for gx, gy in goals:
        dist[gy, gx] = 0
        queue.append((gx, gy))
    head = 0
    while head < len(queue):
        x, y = queue[head]
        head += 1
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if int(tiles[ny, nx]) == WALL or dist[ny, nx] >= 0:
                continue
            dist[ny, nx] = dist[y, x] + 1
            queue.append((nx, ny))
    return dist


def clockwise(action):
    return (action + 1) % 4


def counter(action):
    return (action - 1) % 4


def hazard_ahead(tiles, x, y):
    """The facing a fresh brain wants, when that facing is a pit, spikes, or a crack."""
    goals = {
        (cx, cy)
        for cy in range(tiles.shape[0])
        for cx in range(tiles.shape[1])
        if int(tiles[cy, cx]) == GOAL
    }
    action = naive_action(tiles, x, y, goals)
    if action == WAIT:
        return None, WAIT
    nx, ny = destination(x, y, action)
    tile = int(tiles[ny, nx])
    if tile not in HAZARDS:
        return None, action
    return tile, action
