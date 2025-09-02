import streamlit as st
import pandas as pd
import numpy as np

st.title("Golden Boot Simulator")

# --- Inputs ---
st.sidebar.header("Simulation Controls")
N = st.sidebar.number_input("Number of simulations", 10000, 500000, 100000, step=10000)
k = st.sidebar.number_input("Dispersion parameter (k)", 1.0, 50.0, 8.0, step=0.5)

# Example default player data
default_data = [
    {"Player": "Erling Haaland", "Buy": 26.5, "Sell": 24.5, "SoFar": 2},
    {"Player": "Viktor Gyökeres", "Buy": 20.0, "Sell": 18.5, "SoFar": 2},
    {"Player": "Mohamed Salah", "Buy": 18.5, "Sell": 17.0, "SoFar": 1},
]

df = st.data_editor(pd.DataFrame(default_data), num_rows="dynamic")

# --- Simulation button ---
if st.button("Run Simulation"):

    mus = (df[["Buy","Sell"]].mean(axis=1) - df["SoFar"]).clip(lower=0).values
    so_far = df["SoFar"].values

    rng = np.random.default_rng(2025)
    gamma_scales = np.where(mus > 0, mus / k, 0.0)

    lam_draws = rng.gamma(shape=k, scale=1.0, size=(N, len(df))) * gamma_scales[None, :]
    remaining = rng.poisson(lam=lam_draws)
    final = remaining + so_far

    # Winner probabilities (ties split equally)
    max_per_sim = final.max(axis=1)
    is_winner = (final == max_per_sim[:, None])
    win_prob = (is_winner / is_winner.sum(axis=1)[:, None]).sum(axis=0) / N

    # Top 4 probabilities (ties split across 4 places)
    thresholds = np.partition(final, -4, axis=1)[:, -4]
    is_top4 = final >= thresholds[:, None]
    top4_prob = (is_top4 * (4.0 / is_top4.sum(axis=1))[:, None]).sum(axis=0) / N

    results = df.copy()
    results["Expected Final Goals"] = so_far + mus
    results["Win %"] = (win_prob * 100).round(2)
    results["Top 4 %"] = (top4_prob * 100).round(2)

    st.dataframe(results.sort_values("Win %", ascending=False))
