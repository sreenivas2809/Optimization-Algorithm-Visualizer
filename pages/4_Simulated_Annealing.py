"""
App 4 — Simulated Annealing Visualizer
Solves Exam Timetable Scheduling
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import math

st.set_page_config(page_title="Simulated Annealing", layout="wide")

st.title("Simulated Annealing — Exam Timetable Scheduling")
st.markdown("Watch **Simulated Annealing** schedule exams to eliminate clashes, step by step.")

# ── Problem data ──────────────────────────────────────────────────────────────
EXAMS = ["Mathematics","Physics","Chemistry","English","History",
         "Computer Science","Economics","Biology","Statistics","Geography"]
STUDENTS = [
    [0,1,5],[0,2,6],[1,3,7],[2,4,8],[3,5,9],
    [0,4,7],[1,6,8],[2,5,9],[3,6,0],[4,7,1],
    [5,8,2],[6,9,3],[7,0,4],[8,1,5],[9,2,6],
    [0,3,8],[1,4,9],[2,7,5],[3,8,6],[4,9,7],
    [0,5,2],[1,6,3],[2,7,4],[3,8,0],[4,9,1],
    [5,0,6],[6,1,7],[7,2,8],[8,3,9],[9,4,0],
]
NUM_EXAMS = len(EXAMS)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("SA Parameters")
    num_slots    = st.slider("Number of time slots", 3, 8, 5)
    init_temp    = st.slider("Initial temperature", 10.0, 500.0, 100.0, 10.0)
    cooling_rate = st.slider("Cooling rate", 0.80, 0.999, 0.995, 0.001,
                              format="%.3f")
    min_temp     = st.select_slider("Min temperature",
                                     options=[0.001, 0.01, 0.1, 1.0], value=0.1)
    max_iter     = st.slider("Max iterations", 500, 10000, 5000, 500)
    seed         = st.number_input("Random seed", value=42, step=1)
    run_btn      = st.button("Run SA", type="primary", use_container_width=True)

# ── SA core ───────────────────────────────────────────────────────────────────
def count_clashes(tt, n_slots):
    clashes = 0
    for se in STUDENTS:
        seen = set()
        for e in se:
            s = tt[e]
            if s in seen:
                clashes += 1
            seen.add(s)
    return clashes

def gen_neighbor(tt, n_slots):
    new = tt[:]
    e = random.randint(0, NUM_EXAMS - 1)
    cur = tt[e]
    new[e] = random.choice([s for s in range(n_slots) if s != cur])
    return new

def run_sa(n_slots, init_t, cool, min_t, max_it, seed_val):
    random.seed(seed_val)
    curr   = [random.randint(0, n_slots - 1) for _ in range(NUM_EXAMS)]
    curr_c = count_clashes(curr, n_slots)
    best   = curr[:]
    best_c = curr_c

    T = init_t
    clash_log, temp_log, accept_log = [], [], []
    accepts = 0

    for it in range(max_it):
        if T < min_t:
            break
        nbr   = gen_neighbor(curr, n_slots)
        nbr_c = count_clashes(nbr, n_slots)
        delta = nbr_c - curr_c

        accepted = False
        if delta < 0 or random.random() < math.exp(-delta / T):
            curr, curr_c = nbr, nbr_c
            accepted = True
            accepts += 1

        if curr_c < best_c:
            best, best_c = curr[:], curr_c

        clash_log.append(best_c)
        temp_log.append(T)
        accept_log.append(accepts / (it + 1))
        T *= cool

        if best_c == 0:
            break

    return best, best_c, clash_log, temp_log, accept_log

# ── Run & display ─────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running SA..."):
        best_tt, best_c, clash_log, temp_log, accept_log = run_sa(
            num_slots, init_temp, cooling_rate, min_temp, max_iter, int(seed)
        )

    # ── Top metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final clashes",    str(best_c))
    c2.metric("Iterations run",   str(len(clash_log)))
    c3.metric("Starting clashes", str(clash_log[0]))
    c4.metric("Solved?",          "Yes ✅" if best_c == 0 else "No ❌")

    # ── Convergence plots ─────────────────────────────────────────────────────
    st.subheader("Convergence")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(clash_log, color="crimson", linewidth=1.5)
    axes[0].set_title("Best Clashes over Iterations")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Best Clashes")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(temp_log, color="steelblue", linewidth=1.5)
    axes[1].set_title("Temperature Schedule")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Temperature")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(accept_log, color="darkorange", linewidth=1.5)
    axes[2].set_title("Cumulative Acceptance Rate")
    axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("Acceptance rate")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Timetable grid ────────────────────────────────────────────────────────
    st.subheader("Final Timetable")
    slot_contents = {s: [] for s in range(num_slots)}
    for exam_i, slot in enumerate(best_tt):
        slot_contents[slot].append(EXAMS[exam_i])

    slot_df = {f"Slot {s+1}": ", ".join(slot_contents[s]) or "(empty)"
               for s in range(num_slots)}

    col_left, col_right = st.columns([1, 1])
    with col_left:
        import pandas as pd
        timetable_df = pd.DataFrame(list(slot_df.items()), columns=["Time Slot", "Exams"])
        st.dataframe(timetable_df, use_container_width=True, hide_index=True)

    with col_right:
        # Heatmap: slot assignments
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        grid = np.zeros((num_slots, NUM_EXAMS))
        for exam_i, slot in enumerate(best_tt):
            grid[slot, exam_i] = 1
        im = ax2.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax2.set_xticks(range(NUM_EXAMS))
        ax2.set_xticklabels([e[:4] for e in EXAMS], rotation=45, ha="right", fontsize=8)
        ax2.set_yticks(range(num_slots))
        ax2.set_yticklabels([f"Slot {s+1}" for s in range(num_slots)])
        ax2.set_title("Timetable heatmap (orange = assigned)")
        plt.colorbar(im, ax=ax2, shrink=0.6)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ── Clash analysis ────────────────────────────────────────────────────────
    with st.expander("Clash analysis per student"):
        clash_rows = []
        for si, se in enumerate(STUDENTS):
            seen = {}
            for e in se:
                s = best_tt[e]
                seen[s] = seen.get(s, 0) + 1
            student_clashes = sum(v - 1 for v in seen.values() if v > 1)
            clash_rows.append({
                "Student": si+1,
                "Exams": ", ".join(EXAMS[e] for e in se),
                "Slots assigned": ", ".join(str(best_tt[e]+1) for e in se),
                "Clashes": student_clashes
            })
        import pandas as pd
        cdf = pd.DataFrame(clash_rows)
        st.dataframe(cdf, use_container_width=True, hide_index=True)

    # ── Parameter comparison hint ─────────────────────────────────────────────
    st.info(f"Try changing **cooling_rate** to 0.80 and re-run — notice how the convergence plot changes!")

else:
    st.info("Set parameters in the sidebar and click **Run SA**.")
    st.markdown("""
    **How SA works:**
    1. Start with a random timetable
    2. Generate a **neighbour** — move one exam to a different slot
    3. If it has fewer clashes → always accept
    4. If it has more clashes → accept with probability e^(−Δ/T)
    5. Slowly reduce temperature T → accept fewer bad moves over time

    **Key insight:** The temperature controls exploration vs exploitation.
    - **High T (early)** — accepts bad moves → escapes local minima
    - **Low T (late)** — rarely accepts bad moves → converges to solution

    **Try these cooling rates:**
    | Cooling rate | Behaviour |
    |---|---|
    | 0.80 | Very fast cooling — quick but shallow |
    | 0.95 | Medium cooling |
    | 0.995 | Slow cooling — thorough search (default) |
    """)
