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
    ("Erling Haaland", 26.50, 24.50, 2.00),
    ("Viktor Gyokeres", 20.00, 18.50, 2.00),
    ("Mohamed Salah", 18.50, 17.00, 1.00),
    ("Alexander Isak", 15.00, 13.50, 1.00),
    ("Cole Palmer", 15.00, 13.50, 0.00),
    ("Ollie Watkins", 14.50, 13.00, 0.00),
    ("Chris Wood", 14.50, 13.00, 2.00),
    ("Bukayo Saka", 13.00, 11.50, 0.00),
    ("Jean-Philippe Mateta", 12.50, 11.00, 0.00),
    ("Dominic Solanke", 12.00, 11.00, 0.00),
    ("Jorgen Strand Larsen", 12.00, 10.50, 0.00),
    ("Anthony Gordon", 12.00, 10.50, 0.00),
    ("Antoine Semenyo", 11.25, 10.25, 2.00),
    ("Igor Thiago", 11.25, 10.25, 1.00),
    ("Evanilson", 11.50, 10.00, 0.00),
    ("Raul Jimenez", 11.00, 9.50, 0.00),
    ("Joao Pedro", 11.00, 9.50, 2.00),
    ("Benjamin Sesko", 11.00, 9.50, 0.00),
    ("Brennan Johnson", 10.50, 9.50, 1.00),
    ("Cody Gakpo", 10.50, 9.50, 1.00),
    ("Rodrigo Muniz", 10.25, 9.50, 1.00),
    ("Omar Marmoush", 10.00, 9.00, 0.00),
    ("Jarrod Bowen", 10.00, 9.00, 0.00),
    ("Bruno Fernandes", 10.00, 8.50, 0.00),
    ("Liam Delap", 9.50, 8.50, 0.00),
    ("Thierno Barry", 9.25, 8.25, 0.00),
    ("Morgan Rogers", 9.00, 8.00, 0.00),
    ("Matheus Cunha", 9.00, 8.00, 0.00),
    ("Kai Havertz", 8.75, 7.75, 0.00),
    ("Phil Foden", 8.75, 7.75, 1.00),
    ("Tijjani Reijnders", 8.75, 7.75, 1.00),
    ("Florian Wirtz", 8.50, 7.50, 1.00),
    ("Justin Kluivert", 8.50, 7.50, 0.00),
    ("Ismaila Sarr", 8.50, 7.50, 1.00),
    ("Iliman Ndiaye", 8.50, 7.50, 1.00),
    ("Bryan Mbeumo", 8.50, 7.50, 0.00),
    ("Anthony Elanga", 8.50, 7.50, 1.00),
    ("Eberechi Eze", 7.50, 6.50, 1.00),
    ("Harvey Barnes", 7.50, 6.50, 1.00),
    ("Eliezer Mayenda", 7.50, 6.50, 1.00),
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
