"""Window, squad, and the size of the shared network."""

COLS, ROWS = 16, 12
TILE = 46
BOARD_X, BOARD_Y = 24, 22
PANEL_W = 320
GAP = 22
BOTTOM = 72
WIDTH = BOARD_X + COLS * TILE + GAP + PANEL_W + 18
HEIGHT = BOARD_Y + ROWS * TILE + BOTTOM
FPS = 60
TITLE = "Squad"

# North, east, south, west. Screen y grows downward.
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
ACTION_NAMES = ("north", "east", "south", "west", "wait")
WAIT = 4
OUTPUTS = 5

# Per facing: wall, hazard, round, edge, goal, occupied. Then the way to the door.
DIR_FEATURES = 6
GOAL_FEATURES = 3
IN_DIM = 4 * DIR_FEATURES + GOAL_FEATURES
H1 = 64
H2 = 32

F_WALL, F_HAZARD, F_ROUND, F_EDGE, F_GOAL, F_OCC = range(DIR_FEATURES)

MAX_TURNS = 40
MOVE_TIME = 0.20

BG = (8, 14, 18)
TEXT = (236, 240, 242)
MUTED = (154, 170, 164)
GOLD = (255, 214, 120)
DANGER = (214, 122, 98)
FLOOR_A = (16, 34, 38)
FLOOR_B = (14, 30, 34)
WALL = (36, 54, 60)
WALL_LIT = (58, 82, 88)
PIT_RIM = (46, 58, 64)
SPIKE = (186, 196, 178)
CRACK = (132, 104, 90)
MAT = (30, 46, 52)
MAT_EDGE = (72, 96, 102)
GOAL = (214, 164, 86)
GOAL_DEEP = (120, 78, 36)
BUTTON = (14, 26, 32)
BUTTON_EDGE = (46, 68, 74)
CARD = (12, 22, 28)

SQUAD = (
    ("Wren", (255, 214, 120)),
    ("Ash", (168, 196, 214)),
    ("Bolt", (140, 196, 168)),
    ("Nim", (214, 176, 140)),
)
