import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide") 

st.title("Premier League - Top Goalscorer")

st.markdown(
    """
    **Note:** The <span style='color:#ff4b4b'><b>Dispersion (k)</b></span> parameter controls the spread of possible outcomes for each player.  A higher value means the player's goal total is more predictable (less variance), while a lower value means more uncertainty and wider possible outcomes.  Using 8 as default was a good fit to the betfair exchange odds.
    """,
    unsafe_allow_html=True
)

# --- Inputs ---
st.sidebar.header("Simulation Controls")
N = st.sidebar.number_input("Number of simulations", 10000, 500000, 100000, step=10000)
#k = st.sidebar.number_input("Dispersion parameter (k)", 1.0, 50.0, 8.0, step=0.5)

# Example default player data
default_data = [
    (" Erling Haaland       ",26 , 24, 3),
    (" Viktor Gyokeres      ",19.5 , 18, 2),
    (" Mohamed Salah        ",16.5 , 15, 1),
    (" Alexander Isak       ",15.5 , 14, 0),
    (" Cole Palmer          ",15 , 13.5, 0),
    (" Joao Pedro           ",14 , 12.5, 2),
    (" Chris Wood           ",13.5 , 12, 2),
    (" Jean-Philippe Mateta ",13 , 11.5, 1),
    (" Bukayo Saka          ",13 , 11.5, 0),
    (" Ollie Watkins        ",13 , 11.5, 0),
    (" Igor Thiago          ",12 , 11, 2),
    (" Dominic Solanke      ",12.5 , 11, 0),
    (" Evanilson            ",12 , 10.5, 1),
    (" Jorgen Strand Larsen ",12 , 10.5, 0),
    (" Anthony Gordon       ",12 , 10.5, 0),
    (" Rodrigo Muniz        ",11 , 10, 1),
    (" Antoine Semenyo      ",11 , 10, 2),
    (" Hugo Ekitike         ",11 , 10, 2),
    (" Yoane Wissa          ",11.5 , 10, 0),
    (" Jarrod Bowen         ",10.75 , 9.75, 1),
    (" Bruno Fernandes      ",10.75 , 9.75, 1),
    (" Benjamin Sesko       ",10 , 9, 0),
    (" Nick Woltemade       ",10 , 9, 0),
    (" Omar Marmoush        ",10 , 9, 0),
    (" Liam Delap           ",9.5 , 8.5, 0),
    (" Eberechi Eze         ",9.25 , 8.25, 0),
    (" Bryan Mbeumo         ",9 , 8, 1),
    (" Matheus Cunha        ",9 , 8, 0),
    (" Iliman Ndiaye        ",9 , 8, 2),
    (" Cody Gakpo           ",9 , 8, 1),
    (" Raul Jimenez         ",9 , 8, 0),
    (" Brennan Johnson      ",9 , 8, 2),
    (" Kai Havertz          ",8.75 , 7.75, 0),
    (" Phil Foden           ",8.75 , 7.75, 0),
    (" Beto                 ",8.5 , 7.5, 1),
    (" Ismaila Sarr         ",8.5 , 7.5, 1),
    (" Tijjani Reijnders    ",8.5 , 7.5, 1),
    (" Morgan Rogers        ",8.25 , 7.25, 0),
    (" Thierno Barry        ",8.25 , 7.25, 0),
    (" Danny Welbeck        ",8.25 , 7.25, 0),
]

columns = ["Player", "Buy", "Sell", "SoFar"]
df_source = pd.DataFrame(default_data, columns=columns)

# --- Session state for data ---
if "df_mid" not in st.session_state:
    df_mid = pd.DataFrame(default_data, columns=columns)
    df_mid["Midpoint"] = df_mid[["Buy", "Sell"]].mean(axis=1)
    df_mid["k"] = 8.0
    df_mid["Win %"] = 0.0
    df_mid["Top 4 %"] = 0.0
    df_mid["Win odds"] = 0.0
    df_mid["Top 4 odds"] = 0.0
    df_mid["Distribution"] = [[] for _ in range(len(df_mid))]  # Add empty distribution column
    st.session_state.df_mid = df_mid

# Show editor with +/- controls
df_edit = st.data_editor(
    st.session_state.df_mid.drop(columns=["Buy", "Sell"]),
    num_rows="dynamic",
    column_config={
        "Midpoint": st.column_config.NumberColumn("Midpoint", step=0.25),
        "SoFar": st.column_config.NumberColumn("SoFar", step=0.25),
        "k": st.column_config.NumberColumn("Dispersion (k)", step=0.5),
        "Win %": st.column_config.NumberColumn("🏆 Win %", disabled=True),
        "Top 4 %": st.column_config.NumberColumn("🥇 Top 4 %", disabled=True),
        "Win odds": st.column_config.NumberColumn("🏆 Win odds", disabled=True),
        "Top 4 odds": st.column_config.NumberColumn("🥇 Top 4 odds", disabled=True),
        "Distribution": st.column_config.BarChartColumn("Distribution", width="small"),
    },
    key  ="grid",
    use_container_width=True
)
# --- Remember changes to Midpoint and k ---
st.session_state.df_mid["Midpoint"] = df_edit["Midpoint"]
st.session_state.df_mid["k"] = df_edit["k"]
st.session_state.df_mid["SoFar"] = df_edit["SoFar"]


# --- Simulation button ---
if st.button("Run Simulation"):

    mus = (df_edit["Midpoint"] - df_edit["SoFar"]).clip(lower=0).values
    so_far = df_edit["SoFar"].values
    ks = df_edit["k"].values

    rng = np.random.default_rng(2025)
    gamma_scales = np.where(mus > 0, mus / ks, 0.0)

    # Player-specific ks in Gamma
    lam_draws = rng.gamma(shape=ks, scale=1.0, size=(N, len(df_edit))) * gamma_scales[None, :]
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


    st.session_state.df_mid.loc[:, "Win %"] = (win_prob * 100).round(2)
    st.session_state.df_mid.loc[:, "Top 4 %"] = (top4_prob * 100).round(2)
    st.session_state.df_mid.loc[:, "Win odds"] = (1/win_prob).round(2)
    st.session_state.df_mid.loc[:, "Top 4 odds"] = (1/top4_prob).round(2)

    # Add spark bar distribution for each player
    bins = np.arange(0, 50)
    distributions = [
        np.histogram(final[:, i], bins=bins)[0].tolist()
        for i in range(final.shape[1])
    ]
    st.session_state.df_mid["Distribution"] = distributions

    st.rerun()

