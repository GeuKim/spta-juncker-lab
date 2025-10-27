import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
def clip(value, min_val, max_val):
    return np.minimum(np.maximum(value, min_val), max_val)

st.title("Online Stochastic Brownian-agent method (SBAM) simulation")

# Simulation parameters
n_sim = st.text_input("Number of analytes:", value="2") # Number of simulations
r = st.text_input("Pore radius in µm:", value="4")         # Radius in µm
L = st.text_input("Pore length in µm:", value="20")        # Length in µm
BP = st.text_input("Binding probability:", value="0.014")    # Binding probability
FS = st.text_input("Average flow speed in µm/s:", value="133")    # Flow speed in µm/s
DC = st.text_input("Diffusion coefficient in m^2/s:", value="8e-11")    # Diffusion coefficient in m²/s

if st.button("Run Simulation"):
    start_time = time.time()
    n_sim=int(n_sim)
    r=float(r)
    L=float(L)
    BP=float(BP)
    FS=float(FS)
    DC=float(DC)
    MFS = 2 * FS * 1e-6            # Max drift per µs in µm
    DL = np.sqrt(4 * DC * 1e-6) * 1e6  # Diffusion length in µm

    captured = 0
    SPTs = []  # Store all trajectories
    all_particle_no_step = np.zeros(n_sim, dtype=int)
    all_particle_no_collision = np.zeros(n_sim, dtype=int)
    all_particle_final_point = np.zeros((n_sim, 2))

    for m in range(n_sim):
        particle_trace = [np.array([np.random.uniform(-r, r), 0.0])]
        i = 0
        n = 0  # collision counter

        while True:
            current_pos = particle_trace[-1]
            i += 1

            # Drift due to parabolic flow
            drift = np.array([0, (current_pos[0] ** 2 / r**2 - 1)]) * MFS

            # Random direction
            theta = 2 * np.pi * np.random.rand()
            random_direction = np.array([np.cos(theta), np.sin(theta)])

            # Random step from normal distribution
            DLR = abs(np.random.normal(0, DL))

            # Next position
            next_pos = current_pos + DLR * random_direction + drift
            next_pos[0] = clip(next_pos[0], -r, r)

            particle_trace.append(next_pos)

            # Check boundaries
            if next_pos[1] > 0:
                # Restart from top
                particle_trace = [np.array([np.random.uniform(-r, r), 0.0])]
                i = 0
                n = 0
                continue
            elif next_pos[1] < -L:
                break

            # Side wall collision
            if next_pos[0] == -r or next_pos[0] == r:
                n += 1
                if np.random.rand() < BP:
                    break

        particle_trace_np = np.array(particle_trace)
        SPTs.append(particle_trace_np)
        all_particle_no_step[m] = i
        all_particle_no_collision[m] = n
        all_particle_final_point[m, :] = particle_trace_np[-1]

    elapsed_time = time.time() - start_time
    captured = np.sum(all_particle_final_point[:, 1] >= -L)
    st.success(f"Captured particles: {captured} out of {n_sim} ({captured / n_sim * 100:.2f}%)")
    st.success(f"Computation time: {elapsed_time:.2f} seconds")
   
    max_steps = max(len(trace) for trace in SPTs)
    frame_indices = list(range(1, max_steps, 1000))
    if frame_indices[-1] < max_steps:
        frame_indices.append(max_steps)
    plot_placeholder = st.empty()

    for step in frame_indices:
        fig, ax = plt.subplots()
        ax.set_xlim([-r, r])
        ax.set_ylim([-L, 0])
        ax.set_aspect('equal')
        for trace in SPTs:
            if len(trace) >= step:
                part = trace[:step]
            else:
                part = trace
            ax.plot(part[:, 0], part[:, 1], lw=0.3)
            if step >= len(trace):
                final_pos = trace[-1]
                if final_pos[1] < -L:
                    ax.plot(final_pos[0], final_pos[1], 'xr', markersize=8)  # Red X = exited
                else:
                    ax.plot(final_pos[0], final_pos[1], 'xk', markersize=8)  # Green X = bound
        ax.set_title("SPTs")
        ax.set_xlabel("r (µm)")
        ax.set_ylabel("y (µm)")
        plot_placeholder.pyplot(fig)
        time.sleep(0.001)

    st.success("Animation complete.")
