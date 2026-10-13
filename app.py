# =====================================================================
#  BLOCH SPHERE QUEST
#  A qubit playground + a level-based puzzle game, built with Python.
#  Now with TWO qubits and the CNOT gate (entanglement)!
#  Run with:  streamlit run app.py
# =====================================================================
import random
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------
#  PART 1: THE QUANTUM MATHS (no Streamlit here)
# ---------------------------------------------------------------------
s = 1 / np.sqrt(2)

# Every single-qubit gate is a 2x2 matrix. Applying a gate = multiplying.
GATES = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "H": np.array([[s, s], [s, -s]], dtype=complex),
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),
    "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
}

GATE_HELP = {
    "X": "Flip: |0> <-> |1> (a NOT gate)",
    "Y": "Half-turn around the Y axis",
    "Z": "Flip the phase (half-turn around the vertical axis)",
    "H": "Hadamard: makes a superposition (pole <-> equator)",
    "S": "Quarter-turn around the vertical axis",
    "T": "Eighth-turn around the vertical axis",
}

I2 = np.eye(2, dtype=complex)

# Two qubits = a list of 4 numbers for |00>, |01>, |10>, |11>  (first letter = qubit A).
# CNOT flips the TARGET qubit only if the CONTROL qubit is 1.
CNOT_AB = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)  # A controls B
CNOT_BA = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)  # B controls A

START = np.array([1, 0], dtype=complex)           # one qubit: |0>
START2 = np.array([1, 0, 0, 0], dtype=complex)    # two qubits: |00>

ALL_MOVES_2 = [f"{g}:{q}" for q in "AB" for g in GATES] + ["CX:AB", "CX:BA"]


def start_state(nq):
    return (START if nq == 1 else START2).copy()


def move_matrix(move):
    """A move is "H" (one qubit), "H:A" (gate H on qubit A) or "CX:AB" (CNOT)."""
    if move == "CX:AB":
        return CNOT_AB
    if move == "CX:BA":
        return CNOT_BA
    if ":" in move:
        gate, qubit = move.split(":")
        return np.kron(GATES[gate], I2) if qubit == "A" else np.kron(I2, GATES[gate])
    return GATES[move]


def apply_move(state, move):
    return move_matrix(move) @ state


def run_moves(sequence, nq=1):
    """Apply a list of moves to the starting state and return the result."""
    state = start_state(nq)
    for move in sequence:
        state = apply_move(state, move)
    return state


def to_bloch(state):
    """One qubit (a, b) -> a point (x, y, z) on the sphere."""
    a, b = state[0], state[1]
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = abs(a) ** 2 - abs(b) ** 2
    return float(x), float(y), float(z)


def key_of(state):
    """Rounded (x, y, z) for ONE qubit: lets us check if two states are the same."""
    return tuple(round(v, 3) + 0.0 for v in to_bloch(state))


def state_key(state):
    """A fingerprint of a state (one or two qubits) that ignores the global phase."""
    if len(state) == 2:
        return key_of(state)
    first = int(np.argmax(np.abs(state) > 1e-6))
    amp = state / (state[first] / abs(state[first]))
    return tuple(round(v, 3) + 0.0 for v in np.concatenate([amp.real, amp.imag]))


def reduced_blochs(state):
    """For two qubits: the Bloch point of EACH qubit on its own.
    If the qubits are entangled, these arrows get SHORTER than 1."""
    M = state.reshape(2, 2)
    rho_a = M @ M.conj().T
    rho_b = M.T @ M.conj()

    def vec(r):
        return (float(2 * np.real(r[1, 0])), float(2 * np.imag(r[1, 0])), float(np.real(r[0, 0] - r[1, 1])))

    return [vec(rho_a), vec(rho_b)]


def bloch_points(state):
    return [to_bloch(state)] if len(state) == 2 else reduced_blochs(state)


def concurrence(state):
    """How entangled two qubits are: 0 = not at all, 1 = fully (a Bell state)."""
    return float(2 * abs(state[0] * state[3] - state[1] * state[2]))


def fidelity(target, state):
    """How close the state is to the target: 1 = same state."""
    return float(abs(np.vdot(target, state)) ** 2)


def find_path(start, target_key, allowed, max_depth):
    """Search (shortest first) for the fewest moves that reach the target.
    Breadth-first search: try 1 move, then 2 moves, then 3 ..."""
    if state_key(start) == target_key:
        return []
    seen = {state_key(start)}
    frontier = [(start, [])]
    for _ in range(max_depth):
        new_frontier = []
        for state, path in frontier:
            for move in allowed:
                new_state = apply_move(state, move)
                k = state_key(new_state)
                if k in seen:
                    continue
                seen.add(k)
                if k == target_key:
                    return path + [move]
                new_frontier.append((new_state, path + [move]))
        frontier = new_frontier
    return None


# Names of famous states, so the game can tell the player what the target is
KNOWN = {
    key_of(np.array([1, 0], dtype=complex)): "|0>",
    key_of(np.array([0, 1], dtype=complex)): "|1>",
    key_of(np.array([s, s], dtype=complex)): "|+>",
    key_of(np.array([s, -s], dtype=complex)): "|->",
    key_of(np.array([s, 1j * s], dtype=complex)): "|+i>",
    key_of(np.array([s, -1j * s], dtype=complex)): "|-i>",
    state_key(np.array([1, 0, 0, 0], dtype=complex)): "|00>",
    state_key(np.array([0, 1, 0, 0], dtype=complex)): "|01>",
    state_key(np.array([0, 0, 1, 0], dtype=complex)): "|10>",
    state_key(np.array([0, 0, 0, 1], dtype=complex)): "|11>",
    state_key(np.array([s, 0, 0, s], dtype=complex)): "Bell state (|00> + |11>)/√2",
    state_key(np.array([s, 0, 0, -s], dtype=complex)): "Bell state (|00> - |11>)/√2",
    state_key(np.array([0, s, s, 0], dtype=complex)): "Bell state (|01> + |10>)/√2",
    state_key(np.array([0, s, -s, 0], dtype=complex)): "Singlet (|01> - |10>)/√2",
}


# ---------------------------------------------------------------------
#  PART 2: THE LEVELS
#  "seq" is the secret list of moves that makes the target.
#  "allowed" are the moves the player may use (the others are BROKEN).
# ---------------------------------------------------------------------
WORLD_NAMES = [
    "🏝️ World 1: Pole Position",
    "🌀 World 2: Phase Town",
    "⚡ World 3: Interference Isle",
    "🔗 World 4: Entanglement Station",
]

LEVELS = [
    # ---- World 1 ----
    dict(world=0, name="Flip It", seq=["X"], allowed=["X"], depth=2,
         story="Use the **X** gate to flip the qubit from |0> to |1>."),
    dict(world=0, name="Middle Ground", seq=["H"], allowed=["X", "H"], depth=2,
         story="The **H** gate puts the qubit on the equator: a perfect 50/50 mix. Reach the target!"),
    dict(world=0, name="Mirror Image", seq=["H", "Z"], allowed=["X", "Z", "H"], depth=3,
         story="Reach |->, on the opposite side of the equator. **Z** changes the phase."),
    dict(world=0, name="Quarter Turn", seq=["H", "S"], allowed=["H", "S", "Z"], depth=3,
         story="**S** spins the arrow a quarter turn around the vertical axis."),
    # ---- World 2 ----
    dict(world=1, name="Eighth Turn", seq=["H", "T"], allowed=["H", "T"], depth=3,
         story="**T** is a tiny eighth-turn. Can you spot where it sends the arrow?"),
    dict(world=1, name="Round the Clock", seq=["H", "T", "T", "T"], allowed=["H", "S", "T"], depth=4,
         story="Three eighth-turns needed. Hint: **S** is the same as two **T** gates!"),
    dict(world=1, name="Z Is Broken!", seq=["H", "S", "Z"], allowed=["H", "S"], depth=5,
         story="🚫 Z is broken. Reach |-i>. Remember **S twice = Z**, but there may be an even shorter way!"),
    dict(world=1, name="BOSS: Tilted Qubit", seq=["H", "T", "H"], allowed=["H", "T"], depth=4, boss=True,
         story="👹 The qubit is tilted between the pole and the equator. Only **H** and **T** work!"),
    # ---- World 3 ----
    dict(world=2, name="X Is Broken!", seq=["X"], allowed=["H", "Z", "S", "T"], depth=4,
         story="🚫 X and Y are broken, but you still need |1>. Use **interference**: H, then Z, then H."),
    dict(world=2, name="Double Tilt", seq=["H", "T", "H", "S"], allowed=["H", "T", "S"], depth=4,
         story="Tilt the qubit, then spin it. Plan your gates before you click!"),
    dict(world=2, name="BOSS: Triple Threat", seq=["H", "T", "H", "T", "H"], allowed=["H", "T"], depth=5, boss=True,
         story="👹 Three H gates and two T gates hide the target. Find a shorter way if you can!"),
    dict(world=2, name="FINAL BOSS: Sphere Master", seq=["H", "T", "H", "S", "H", "T"],
         allowed=list(GATES), depth=6, boss=True,
         story="👑 All gates are back. The target is far away. Show everything you have learned!"),
    # ---- World 4: TWO QUBITS + CNOT ----
    dict(world=3, qubits=2, name="Spooky Link", seq=["H:A", "CX:AB"],
         allowed=["H:A", "H:B", "X:A", "X:B", "CX:AB"], depth=3,
         story="Two qubits, A and B, both start at 0. **CNOT** flips B only if A is 1. "
               "Make the Bell state (|00> + |11>)/√2 and watch both arrows shrink: that is **entanglement**!"),
    dict(world=3, qubits=2, name="Conditional Flip", seq=["X:A", "CX:AB"],
         allowed=["X:A", "CX:AB"], depth=3,
         story="🚫 X on B is broken. Use **CNOT** to flip B, but it only works when A is 1. Reach |11>."),
    dict(world=3, qubits=2, name="BOSS: Bell Twin", seq=["H:A", "CX:AB", "X:B"],
         allowed=["H:A", "H:B", "X:A", "X:B", "CX:AB", "CX:BA"], depth=4, boss=True,
         story="👹 Reach (|01> + |10>)/√2. Remember: either qubit can be the control!"),
    dict(world=3, qubits=2, name="FINAL BOSS: The Singlet", seq=["X:B", "H:A", "CX:AB", "Z:A"],
         allowed=["H:A", "H:B", "X:A", "X:B", "Z:A", "Z:B", "CX:AB", "CX:BA"], depth=4, boss=True,
         story="👑 Reach the singlet (|01> - |10>)/√2, the most famous entangled state of all."),
]


@lru_cache(maxsize=None)
def level_info(i):
    """Work out a level's target and its best possible score (par)."""
    lvl = LEVELS[i]
    nq = lvl.get("qubits", 1)
    target = run_moves(lvl["seq"], nq)
    target_key = state_key(target)
    path = find_path(start_state(nq), target_key, tuple(lvl["allowed"]), lvl["depth"])
    par = len(path)
    extra = 2 if lvl.get("boss") else 3
    return target, target_key, par, path, par + extra  # last = max moves


def stars_for(used, par):
    if used <= par:
        return 3
    if used <= par + 1:
        return 2
    return 1


# ===== UI =====
# ---------------------------------------------------------------------
#  PART 3: THE APP (everything you see on the screen)
# ---------------------------------------------------------------------
st.set_page_config(page_title="Bloch Sphere Quest", page_icon="🔮", layout="centered")

st.markdown(
    """
<style>
/* soft background: light lavender -> pale blue -> mint, and a deep indigo one for dark mode */
.stApp { background: linear-gradient(160deg, #ece6ff 0%, #e3eeff 50%, #dff6f1 100%) !important; }
@media (prefers-color-scheme: dark) {
    .stApp { background: linear-gradient(160deg, #1b1740 0%, #231c52 50%, #12303a 100%) !important; }
}
[data-testid="stHeader"] { background: transparent !important; }

/* soft cards */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.55); border: 1px solid rgba(123, 97, 255, 0.18);
    border-radius: 14px; padding: 0.6rem 0.9rem;
}
[data-testid="stExpander"] { border-radius: 14px; background: rgba(255, 255, 255, 0.40); }
@media (prefers-color-scheme: dark) {
    [data-testid="stMetric"] { background: rgba(255, 255, 255, 0.06); }
    [data-testid="stExpander"] { background: rgba(255, 255, 255, 0.04); }
}

.quest-title {
    font-size: 2.3rem; font-weight: 800; margin-bottom: 0;
    background: linear-gradient(90deg, #7b61ff, #1fb5a8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
div.stButton > button { border-radius: 12px; font-weight: 700; }

/* the floating "How to play" button, always in the top-left corner */
.st-key-howto { position: fixed; top: 0.7rem; left: 0.9rem; z-index: 1000000; width: auto !important; }
.st-key-howto button {
    border-radius: 999px; border: none; padding: 0.3rem 1rem;
    background: linear-gradient(90deg, #7b61ff, #1fb5a8);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}
.st-key-howto button * { color: #ffffff !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- the "How to play" window ----------
@st.dialog("📖 How to play", width="large")
def how_to_play():
    t1, t2, t3, t4, t5 = st.tabs(["🎯 Goal", "🎮 Quest", "🧪 Free play", "🔗 Two qubits", "🔑 Gates"])
    with t1:
        st.markdown(
            """
**A qubit is an arrow inside a sphere.** Gates turn the arrow. Your job is to move it to the right place.

- 🔴 **Red arrow:** your qubit
- 🟢 **Green dashed arrow:** the target you must reach
- 🟠 **Orange dots:** the path your qubit has travelled
- **|0⟩** is at the top, **|1⟩** at the bottom, and the middle ring is a 50/50 mix of both

Pick a mode at the top of the page: **Playground**, **Quantum Quest** (the game), **Random Challenge** or **Two-Qubit Lab**.
"""
        )
    with t2:
        st.markdown(
            """
1. Pick a level on the **level map**. ▶ = ready to play, 🔒 = locked, ⭐ = stars you have won.
2. Read the story and look at the green target.
3. Click gates to move the red arrow onto the target. The **Match** bar shows how close you are. **100% = level complete!**
4. Keep an eye on your **moves**. If you run out, the level is lost, but you can retry.

**Stars:** ⭐⭐⭐ = the fewest moves possible (par), ⭐⭐ = one extra move, ⭐ = any win.

**Tips**
- 🚫 Greyed-out gates are **broken** in that level. Find another way.
- 👹 Boss levels give you fewer spare moves.
- 💡 Stuck? Tick **Hint** to see your next move.
- Finish a level to unlock the next one.
"""
        )
    with t3:
        st.markdown(
            """
- **Playground:** click any gate and watch the arrow. Press **Measure** to force the qubit to pick 0 or 1, or run 1000 measurements to see the chances.
- **Random Challenge:** the game makes a secret target. Reach it with as few gates as you can.
- **↺ Reset** puts the qubit back to |0⟩ at any time.
- Open **Show the maths** to see the numbers behind the arrow.
"""
        )
    with t4:
        st.markdown(
            """
Two qubits, **A** and **B**, both start at |00⟩. Each has its own sphere and its own row of gates.

**CNOT** flips the **target** qubit only when the **control** qubit is 1. The button *A → B* means A is the control and B is the target.

**Entanglement:** try **H on A**, then **CNOT A → B**. Both arrows shrink to a dot and the Entanglement meter reaches 100%. The qubits are now linked: you can no longer describe one without the other.

Try it in the **Two-Qubit Lab**, then beat **World 4** in Quantum Quest.
"""
        )
    with t5:
        st.markdown(
            """
- **X:** flips the qubit (0 ↔ 1), like a NOT gate
- **Y:** half-turn around the Y axis
- **Z:** flips the phase (spins the arrow half-way round the vertical axis)
- **H:** makes a superposition (pole ↔ middle ring)
- **S:** quarter-turn around the vertical axis
- **T:** eighth-turn around the vertical axis
- **CNOT:** flips the target only if the control is 1
"""
        )


if st.button("❓ How to play", key="howto"):
    how_to_play()

st.markdown('<h1 class="quest-title">🔮 Bloch Sphere Quest</h1>', unsafe_allow_html=True)
st.caption("Play with qubits, then beat the levels using quantum gates!")

PLAY, GAME, RANDOM, TWO = "🧪 Playground", "🎮 Quantum Quest", "🎲 Random Challenge", "🔗 Two-Qubit Lab"


# ---------- memory: the app remembers things between clicks ----------
def init_memory():
    defaults = dict(
        nq=1, state=START.copy(), history=[], moves=[], trail=[START.copy()],
        result=None, last_stars=0, celebrated=False,
        game_level=0, stars={}, last_mode=None, seen_tip=False,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_memory()

if not st.session_state.seen_tip:  # a one-time hint for new players
    st.session_state.seen_tip = True
    st.toast("New here? Tap ❓ How to play in the top-left corner.", icon="🔮")


# ---------- actions (run when a button is clicked) ----------
def reset_qubit(nq=None):
    if nq is not None:
        st.session_state.nq = nq
    st.session_state.state = start_state(st.session_state.nq)
    st.session_state.history = []
    st.session_state.moves = []
    st.session_state.trail = [st.session_state.state.copy()]
    st.session_state.result = None
    st.session_state.celebrated = False


def apply_gate(move):
    st.session_state.state = apply_move(st.session_state.state, move)
    st.session_state.history.append(move)
    st.session_state.moves.append(move)
    st.session_state.trail.append(st.session_state.state.copy())


def do_measure():
    state = st.session_state.state
    nq = st.session_state.nq
    probs = np.abs(state) ** 2
    outcome = int(np.random.choice(len(state), p=probs / probs.sum()))
    new_state = np.zeros(len(state), dtype=complex)
    new_state[outcome] = 1
    st.session_state.state = new_state
    st.session_state.history.append(f"Measured {format(outcome, f'0{nq}b')}")
    st.session_state.trail.append(new_state.copy())


def start_level(i):
    st.session_state.game_level = i
    reset_qubit(LEVELS[i].get("qubits", 1))


def new_random():
    """Pick a random target by secretly applying 2-4 random gates."""
    while True:
        seq = [random.choice(list(GATES)) for _ in range(random.randint(2, 4))]
        target = run_moves(seq, 1)
        if key_of(target) != key_of(START):
            break
    k = state_key(target)
    st.session_state.rand_target = target
    st.session_state.rand_key = k
    st.session_state.rand_best = find_path(START, k, tuple(GATES), len(seq))
    reset_qubit(1)


# ---------- drawing the spheres ----------
def circle(kind, n=60):
    t = np.linspace(0, 2 * np.pi, n)
    c, sn, zero = np.cos(t), np.sin(t), np.zeros(n)
    return {"equator": (c, sn, zero), "xz": (c, zero, sn), "yz": (zero, c, sn)}[kind]


def draw_arrow(fig, point, color, row, col, dashed=False):
    """Arrow from the centre to a point, with a cone as the arrow head.
    If the point is (almost) at the centre, the qubit is entangled: draw a dot instead."""
    x, y, z = point
    length = (x * x + y * y + z * z) ** 0.5
    if length < 0.05:
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                                   marker=dict(size=7, color=color, symbol="circle-open"),
                                   hoverinfo="skip", showlegend=False), row=row, col=col)
        return
    fig.add_trace(go.Scatter3d(
        x=[0, 0.8 * x], y=[0, 0.8 * y], z=[0, 0.8 * z], mode="lines",
        line=dict(color=color, width=7, dash="dash" if dashed else "solid"),
        hoverinfo="skip", showlegend=False), row=row, col=col)
    fig.add_trace(go.Cone(
        x=[x], y=[y], z=[z], u=[x], v=[y], w=[z], anchor="tip",
        sizemode="absolute", sizeref=0.3, showscale=False,
        colorscale=[[0, color], [1, color]], hoverinfo="skip"), row=row, col=col)


def add_sphere(fig, k, point, target_point, trail_points):
    """Draw ONE Bloch sphere into column k+1 of the figure."""
    row, col = 1, k + 1
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    fig.add_trace(go.Surface(
        x=np.outer(np.cos(u), np.sin(v)), y=np.outer(np.sin(u), np.sin(v)),
        z=np.outer(np.ones_like(u), np.cos(v)), opacity=0.10, showscale=False,
        colorscale=[[0, "#7b61ff"], [1, "#7b61ff"]], hoverinfo="skip"), row=row, col=col)
    for kind in ("equator", "xz", "yz"):  # rings that make the sphere easy to read
        cx, cy, cz = circle(kind)
        fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode="lines", line=dict(color="#9aa0b4", width=2),
                                   hoverinfo="skip", showlegend=False), row=row, col=col)
    for xs, ys, zs in (([-1.1, 1.1], [0, 0], [0, 0]), ([0, 0], [-1.1, 1.1], [0, 0]), ([0, 0], [0, 0], [-1.1, 1.1])):
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#9aa0b4", width=3),
                                   hoverinfo="skip", showlegend=False), row=row, col=col)
    if trail_points and len(trail_points) > 1:  # the path the qubit has travelled
        tx, ty, tz = zip(*trail_points[-15:])
        fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode="lines+markers",
                                   line=dict(color="#f2a93b", width=3), marker=dict(size=4, color="#f2a93b"),
                                   hoverinfo="skip", showlegend=False), row=row, col=col)
    if target_point is not None:
        tx, ty, tz = target_point
        if (tx * tx + ty * ty + tz * tz) ** 0.5 >= 0.05:
            draw_arrow(fig, target_point, "#1db954", row, col, dashed=True)
    draw_arrow(fig, point, "#ff3b30", row, col)

    # labels are "annotations", so they always sit OUTSIDE and in front of the ball
    def label(x, y, z, text, color):
        return dict(x=x, y=y, z=z, text=text, showarrow=False, font=dict(size=15, color=color))

    axis = dict(visible=False, range=[-1.5, 1.5])
    fig.update_layout(**{("scene" if k == 0 else f"scene{k + 1}"): dict(
        xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube",
        camera=dict(eye=dict(x=1.6, y=1.3, z=0.7)),
        annotations=[
            label(0, 0, 1.3, "<b>|0⟩</b>", "#1fb5a8"),
            label(0, 0, -1.3, "<b>|1⟩</b>", "#f2a93b"),
            label(1.3, 0, 0, "|+⟩", "#8f7cf0"),
            label(-1.3, 0, 0, "|−⟩", "#8f7cf0"),
            label(0, 1.3, 0, "|+i⟩", "#8f7cf0"),
            label(0, -1.3, 0, "|−i⟩", "#8f7cf0"),
        ])})


def make_figure(state, target_state=None, trail=None):
    """One sphere for one qubit, or two spheres (A and B) for two qubits."""
    nq = 1 if len(state) == 2 else 2
    fig = make_subplots(rows=1, cols=nq, specs=[[{"type": "scene"}] * nq], horizontal_spacing=0.0)
    now = bloch_points(state)
    goal = bloch_points(target_state) if target_state is not None else [None] * nq
    history = [bloch_points(t) for t in (trail or [])]
    for k in range(nq):
        add_sphere(fig, k, now[k], goal[k], [h[k] for h in history])
    if nq == 2:
        for k, name in enumerate(("Qubit A", "Qubit B")):
            fig.add_annotation(text=f"<b>{name}</b>", xref="paper", yref="paper", x=0.25 + 0.5 * k, y=1.0,
                               showarrow=False, font=dict(size=16))
    fig.update_layout(margin=dict(l=0, r=0, t=20 if nq == 2 else 0, b=0),
                      height=380 if nq == 2 else 440, paper_bgcolor="rgba(0,0,0,0)")
    return fig


# =====================================================================
#  THE SCREEN
# =====================================================================
mode = st.radio("Mode", [PLAY, GAME, RANDOM, TWO], horizontal=True, label_visibility="collapsed")

if st.session_state.last_mode != mode:  # the player switched mode
    st.session_state.last_mode = mode
    if mode == TWO:
        reset_qubit(2)
    elif mode == GAME:
        reset_qubit(LEVELS[st.session_state.game_level].get("qubits", 1))
    else:
        reset_qubit(1)
    if mode == RANDOM:
        new_random()

in_game, in_random = mode == GAME, mode == RANDOM
state = st.session_state.state
nq = st.session_state.nq
allowed = ALL_MOVES_2 if nq == 2 else list(GATES)
target_state, target_key, par, best_path, max_moves = None, None, None, None, None

# ---------- QUANTUM QUEST: level map + HUD ----------
if in_game:
    stars = st.session_state.stars
    cur = st.session_state.game_level
    st.markdown(f"**⭐ {sum(stars.values())} / {3 * len(LEVELS)} stars**")
    for w, wname in enumerate(WORLD_NAMES):
        st.markdown(f"**{wname}**")
        idxs = [i for i, lv in enumerate(LEVELS) if lv["world"] == w]
        for col, i in zip(st.columns(len(idxs)), idxs):
            unlocked = i == 0 or (i - 1) in stars
            label = f"{i + 1} " + ("⭐" * stars[i] if i in stars else ("▶" if unlocked else "🔒"))
            col.button(label, key=f"lvl_{i}", disabled=not unlocked, on_click=start_level, args=(i,),
                       type="primary" if i == cur else "secondary", width="stretch")

    lvl = LEVELS[cur]
    target_state, target_key, par, best_path, max_moves = level_info(cur)
    allowed = lvl["allowed"]
    used = len(st.session_state.moves)
    solved = state_key(state) == target_key

    if st.session_state.result is None:  # did the player just win or lose?
        if solved:
            n = stars_for(used, par)
            st.session_state.result = "won"
            st.session_state.last_stars = n
            st.session_state.stars[cur] = max(st.session_state.stars.get(cur, 0), n)
        elif used >= max_moves:
            st.session_state.result = "lost"

    st.markdown(f"### Level {cur + 1}: {lvl['name']}")
    st.info(lvl["story"])
    st.caption(f"🎯 Target: {KNOWN.get(target_key, 'a mystery state')}")
    broken = [m for m in (ALL_MOVES_2 if nq == 2 else list(GATES)) if m not in allowed]
    if broken and nq == 1:
        st.caption("🚫 Broken gates: " + ", ".join(broken))
    h1, h2, h3 = st.columns(3)
    h1.metric("Moves used", used)
    h2.metric("Moves left", max(max_moves - used, 0))
    h3.metric("Par (best)", par)
    match = fidelity(target_state, state)
    st.progress(min(max(match, 0.0), 1.0), text=f"🎯 Match with the target: {match * 100:.0f}%")

# ---------- RANDOM CHALLENGE: HUD ----------
if in_random:
    if "rand_key" not in st.session_state:
        new_random()
    target_state, target_key = st.session_state.rand_target, st.session_state.rand_key
    best_path = st.session_state.rand_best
    if st.session_state.result is None and state_key(state) == target_key:
        st.session_state.result = "won"
    st.info("Goal: move the red arrow onto the green one in as few gates as you can.")
    match = fidelity(target_state, state)
    st.progress(min(max(match, 0.0), 1.0), text=f"🎯 Match with the target: {match * 100:.0f}%")
    st.button("🎲 New challenge", on_click=new_random, width="stretch")

if mode == TWO:
    st.info("Two qubits, A and B. **CNOT** flips the target qubit only if the control qubit is 1. "
            "Try **H on A**, then **CNOT A → B**, and watch both arrows shrink: that is entanglement!")

# ---------- GATE BUTTONS ----------
locked = st.session_state.result is not None
st.subheader("Apply a gate")
if nq == 1:
    for col, name in zip(st.columns(len(GATES)), GATES):
        col.button(name, key=f"gate_{name}", on_click=apply_gate, args=(name,),
                   disabled=(name not in allowed) or locked, help=GATE_HELP[name], width="stretch")
else:
    for q in ("A", "B"):
        st.caption(f"Qubit {q}")
        for col, name in zip(st.columns(len(GATES)), GATES):
            move = f"{name}:{q}"
            col.button(name, key=f"gate_{name}_{q}", on_click=apply_gate, args=(move,),
                       disabled=(move not in allowed) or locked, help=GATE_HELP[name], width="stretch")
    st.caption("CNOT: flips the TARGET only if the CONTROL is 1")
    n1, n2 = st.columns(2)
    n1.button("CNOT  A → B", key="gate_CXAB", on_click=apply_gate, args=("CX:AB",),
              disabled=("CX:AB" not in allowed) or locked, help="A is the control, B is the target", width="stretch")
    n2.button("CNOT  B → A", key="gate_CXBA", on_click=apply_gate, args=("CX:BA",),
              disabled=("CX:BA" not in allowed) or locked, help="B is the control, A is the target", width="stretch")

b1, b2 = st.columns(2)
b1.button("↺ Reset", on_click=reset_qubit, width="stretch")
b2.button("📏 Measure (collapse)", on_click=do_measure, disabled=mode not in (PLAY, TWO), width="stretch")

# ---------- THE SPHERE(S) ----------
st.plotly_chart(make_figure(state, target_state, st.session_state.trail), theme=None, width="stretch")

# ---------- RESULTS ----------
if in_game:
    cur = st.session_state.game_level
    if st.session_state.result == "won":
        n = st.session_state.last_stars
        st.success(f"Level complete!  {'⭐' * n}{'☆' * (3 - n)}")
        if n == 3:
            st.caption("Perfect! You matched the best possible score.")
            if not st.session_state.celebrated:
                st.balloons()
                st.session_state.celebrated = True
        else:
            st.caption(f"Par is {par} moves. Press Retry and try to match it for 3 stars!")
        c1, c2 = st.columns(2)
        c1.button("🔁 Retry", on_click=start_level, args=(cur,), width="stretch")
        if cur + 1 < len(LEVELS):
            c2.button("Next level ▶", on_click=start_level, args=(cur + 1,), type="primary", width="stretch")
        else:
            st.success(f"🏆 You finished Bloch Sphere Quest with {sum(st.session_state.stars.values())} stars!")
    elif st.session_state.result == "lost":
        st.error("Out of moves! The qubit got lost in space.")
        st.button("🔁 Retry level", on_click=start_level, args=(cur,), type="primary", width="stretch")
    elif st.checkbox("💡 Hint: show my next move", key=f"hint_{cur}"):
        left = max(max_moves - len(st.session_state.moves), 1)
        path = find_path(state, target_key, tuple(allowed), min(left, 4 if nq == 2 else 6))
        if path:
            st.write(f"Try **{path[0].replace(':', ' on ').replace('CX on AB', 'CNOT A → B').replace('CX on BA', 'CNOT B → A')}** next.")
        else:
            st.write("You are off the shortest path. Try pressing Reset.")

if in_random:
    used = len(st.session_state.moves)
    best_n = len(best_path) if best_path is not None else None
    r1, r2 = st.columns(2)
    r1.metric("Gates used", used)
    r2.metric("Best possible", best_n if best_n is not None else "?")
    if st.session_state.result == "won":
        if best_n is not None and used <= best_n:
            st.success(f"Perfect! Solved in {used} gates, the best possible.")
        else:
            st.success(f"Solved in {used} gates! Best possible is {best_n}. Press Reset to try again.")
        if not st.session_state.celebrated:
            st.balloons()
            st.session_state.celebrated = True
    if st.checkbox("💡 Show hint (shortest solution)", key="rand_hint"):
        st.write(" → ".join(best_path) if best_path else "No hint available.")

# ---------- PROBABILITIES ----------
probs = np.abs(state) ** 2
st.subheader("Probabilities")
for k, col in enumerate(st.columns(len(probs))):
    col.metric(f"Chance of {format(k, f'0{nq}b')}", f"{probs[k] * 100:.1f}%")

if nq == 2:
    c = concurrence(state)
    st.progress(min(c, 1.0), text=f"🔗 Entanglement: {c * 100:.0f}%")
    st.caption("0% = the qubits are independent. 100% = fully entangled (a Bell state). "
               "The more entangled they are, the shorter each arrow gets.")

with st.expander("📐 Show the maths"):
    def cfmt(c):
        return f"{c.real:.3f}{c.imag:+.3f}i"

    st.write("State = [" + ",  ".join(cfmt(a) for a in state) + "]")
    if nq == 1:
        x, y, z = to_bloch(state)
        st.write(f"Bloch point: x = {x:.2f}, y = {y:.2f}, z = {z:.2f}")
    if st.session_state.moves:
        last = st.session_state.moves[-1]
        m = move_matrix(last)
        st.write(f"Last move **{last}** is this matrix:")
        st.code("\n".join("   ".join(cfmt(m[r][cc]) for cc in range(m.shape[1])) for r in range(m.shape[0])))

if mode in (PLAY, TWO):
    st.subheader("Measure 1000 times")
    if st.button("Run 1000 measurements"):
        results = np.random.choice(len(probs), size=1000, p=probs / probs.sum())
        st.bar_chart({format(k, f"0{nq}b"): int(np.sum(results == k)) for k in range(len(probs))})

st.subheader("Gate history")
st.write(" → ".join(st.session_state.history) if st.session_state.history else "No gates yet")
