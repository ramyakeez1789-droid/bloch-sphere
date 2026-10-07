import random
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Bloch Sphere Playground")
st.title("Qubit Bloch Sphere Playground")

# ---------- 1. THE GATES (each gate is a 2x2 matrix) ----------
s = 1 / np.sqrt(2)
GATES = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "H": np.array([[s, s], [s, -s]], dtype=complex),
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),
    "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
}
START = np.array([1, 0], dtype=complex)  # the qubit always starts as |0>


# ---------- 2. HELPER FUNCTIONS ----------
def to_bloch(state):
    """Turn the qubit's 2 numbers (a, b) into a point (x, y, z) on the sphere."""
    a, b = state[0], state[1]
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = abs(a) ** 2 - abs(b) ** 2
    return float(x), float(y), float(z)


def key_of(state):
    """Rounded (x, y, z) so we can check if two states are the same point."""
    return tuple(round(v, 3) + 0.0 for v in to_bloch(state))


# Famous states, so the app can tell you the NAME of your target
KNOWN = {
    key_of(np.array([1, 0], dtype=complex)): "|0>",
    key_of(np.array([0, 1], dtype=complex)): "|1>",
    key_of(np.array([s, s], dtype=complex)): "|+>",
    key_of(np.array([s, -s], dtype=complex)): "|->",
    key_of(np.array([s, 1j * s], dtype=complex)): "|+i>",
    key_of(np.array([s, -1j * s], dtype=complex)): "|-i>",
}


def min_gates(target_key, max_depth=6):
    """Try every gate sequence (shortest first) to find the fewest gates needed."""
    if key_of(START) == target_key:
        return []
    seen = {key_of(START)}
    frontier = [(START, [])]
    for _ in range(max_depth):
        new_frontier = []
        for state, path in frontier:
            for name, gate in GATES.items():
                new_state = gate @ state
                k = key_of(new_state)
                if k in seen:
                    continue
                seen.add(k)
                if k == target_key:
                    return path + [name]
                new_frontier.append((new_state, path + [name]))
        frontier = new_frontier
    return None


def new_challenge():
    """Pick a random target by secretly applying 2-4 random gates."""
    while True:
        sequence = [random.choice(list(GATES)) for _ in range(random.randint(2, 4))]
        target = START.copy()
        for name in sequence:
            target = GATES[name] @ target
        if key_of(target) != key_of(START):
            break
    k = key_of(target)
    st.session_state.target_state = target
    st.session_state.target_key = k
    st.session_state.best_path = min_gates(k)
    # reset the qubit and the move counter
    st.session_state.state = START.copy()
    st.session_state.history = []
    st.session_state.moves = []
    st.session_state.celebrated = False


# ---------- 3. MEMORY (the app remembers the qubit between clicks) ----------
if "state" not in st.session_state:
    st.session_state.state = START.copy()
    st.session_state.history = []
    st.session_state.moves = []
    st.session_state.celebrated = False

# ---------- 4. CHALLENGE MODE SWITCH ----------
challenge = st.toggle("Challenge mode", value=False)

if challenge:
    if "target_key" not in st.session_state:
        new_challenge()
    target_name = KNOWN.get(st.session_state.target_key, "a mystery state")
    st.info(
        f"Goal: move the red arrow onto the green diamond ({target_name}) "
        "using as few gates as possible."
    )
    if st.button("New challenge", width="stretch"):
        new_challenge()

# ---------- 5. BUTTONS ----------
st.subheader("Apply a gate")
cols = st.columns(len(GATES))
for col, name in zip(cols, GATES):
    if col.button(name, width="stretch"):
        st.session_state.state = GATES[name] @ st.session_state.state
        st.session_state.history.append(name)
        if challenge:
            st.session_state.moves.append(name)

c1, c2 = st.columns(2)
if c1.button("Reset to |0>", width="stretch"):
    st.session_state.state = START.copy()
    st.session_state.history = []
    st.session_state.moves = []
    st.session_state.celebrated = False

state = st.session_state.state
a, b = state[0], state[1]
p0 = float(abs(a) ** 2)
p1 = float(abs(b) ** 2)

# measuring is switched off in challenge mode (it would ruin your attempt)
if c2.button("Measure (collapse)", width="stretch", disabled=challenge):
    result = np.random.choice([0, 1], p=[p0, p1])
    st.session_state.state = (
        np.array([1, 0], dtype=complex) if result == 0 else np.array([0, 1], dtype=complex)
    )
    st.session_state.history.append(f"Measured {result}")
    state = st.session_state.state
    a, b = state[0], state[1]
    p0 = float(abs(a) ** 2)
    p1 = float(abs(b) ** 2)

# ---------- 6. TURN THE STATE INTO A POINT ON THE SPHERE ----------
x, y, z = to_bloch(state)

# ---------- 7. DRAW THE SPHERE ----------
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 20)
sx = np.outer(np.cos(u), np.sin(v))
sy = np.outer(np.sin(u), np.sin(v))
sz = np.outer(np.ones_like(u), np.cos(v))

fig = go.Figure()
fig.add_trace(
    go.Surface(x=sx, y=sy, z=sz, opacity=0.15, showscale=False, colorscale="Blues")
)
# axes lines
for (ax, ay, az) in [((-1, 1), (0, 0), (0, 0)), ((0, 0), (-1, 1), (0, 0)), ((0, 0), (0, 0), (-1, 1))]:
    fig.add_trace(
        go.Scatter3d(x=ax, y=ay, z=az, mode="lines", line=dict(color="gray", width=3), showlegend=False)
    )
# the target (green) - only in challenge mode
if challenge:
    tx, ty, tz = to_bloch(st.session_state.target_state)
    fig.add_trace(
        go.Scatter3d(
            x=[0, tx], y=[0, ty], z=[0, tz],
            mode="lines+markers",
            line=dict(color="green", width=5, dash="dash"),
            marker=dict(size=6, symbol="diamond", color="green"),
            showlegend=False,
        )
    )
# the qubit arrow (red)
fig.add_trace(
    go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode="lines+markers",
        line=dict(color="red", width=8),
        marker=dict(size=[2, 8]),
        showlegend=False,
    )
)
# labels at the poles
fig.add_trace(
    go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[1.2, -1.2],
        mode="text", text=["|0>", "|1>"],
        textfont=dict(size=16), showlegend=False,
    )
)
fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    height=420,
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="cube",
    ),
)
st.plotly_chart(fig, width="stretch")

# ---------- 8. CHALLENGE SCOREBOARD ----------
if challenge:
    moves = st.session_state.moves
    best_path = st.session_state.best_path
    best_n = len(best_path) if best_path is not None else None
    solved = key_of(state) == st.session_state.target_key

    m1, m2 = st.columns(2)
    m1.metric("Gates used", len(moves))
    m2.metric("Best possible", best_n if best_n is not None else "?")

    if solved:
        if best_n is not None and len(moves) <= best_n:
            st.success(f"Perfect! You solved it in {len(moves)} gates, the best possible.")
        else:
            st.success(
                f"Solved in {len(moves)} gates! The best possible is {best_n}. "
                "Can you do better? Press Reset and try again."
            )
        if not st.session_state.celebrated:
            st.balloons()
            st.session_state.celebrated = True
    else:
        st.caption("Not there yet. Keep trying gates!")

    if st.checkbox("Show hint (the shortest solution)"):
        if best_path is not None:
            st.write(" -> ".join(best_path))
        else:
            st.write("No hint available for this one.")

# ---------- 9. SHOW THE NUMBERS ----------
st.subheader("Probabilities")
q1, q2 = st.columns(2)
q1.metric("Chance of 0", f"{p0 * 100:.1f}%")
q2.metric("Chance of 1", f"{p1 * 100:.1f}%")

st.subheader("Measure 1000 times")
if st.button("Run 1000 measurements"):
    results = np.random.choice([0, 1], size=1000, p=[p0, p1])
    counts = {"0": int(np.sum(results == 0)), "1": int(np.sum(results == 1))}
    st.bar_chart(counts)

st.subheader("Gate history")
st.write(" -> ".join(st.session_state.history) if st.session_state.history else "No gates yet")
