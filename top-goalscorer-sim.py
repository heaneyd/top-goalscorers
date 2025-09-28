import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide") 

st.title("Premier League - Top Goalscorer")

st.markdown(
    """
    **Note:** The <span style='color:#ff4b4b'><b>Dispersion (k)</b></span> parameter controls the spread of possible outcomes for each player.  A higher value means the player's goal total is more predictable (less variance), while a lower value means more uncertainty and wider possible outcomes.  Using 20.0 as default, since it was a good fit to the previous season long goals marktes on the betfair exchange.
    <p/>
    <a href = 'https://fog2014.blogspot.com/2025/09/premier-league-top-goalscorer.html'>Additional Notes here.</a>
    """,
    unsafe_allow_html=True
)

# --- Inputs ---
st.sidebar.header("Simulation Controls")
N = st.sidebar.number_input("Number of simulations", 10000, 500000, 100000, step=10000)
#k = st.sidebar.number_input("Dispersion parameter (k)", 1.0, 50.0, 8.0, step=0.5)

# Example default player data
default_data = [    
    ("Erling Haaland",29 , 27, 6),
    ("Viktor Gyokeres",19 , 17.5, 3),
    ("Mohamed Salah",15 , 13.5, 2),
    ("Jean-Philippe Mateta",14.5 , 13, 2),
    ("Alexander Isak",14.5 , 13, 0),
    ("Cole Palmer",14 , 12.5, 1),
    ("Joao Pedro",13.5 , 12, 2),
    ("Nick Woltemade",13 , 11.5, 1),
    ("Igor Thiago",12 , 11, 2),
    ("Chris Wood",12 , 11, 2),
    ("Hugo Ekitike",11.75 , 10.75, 3),
    ("Antoine Semenyo",11.75 , 10.75, 3),
    ("Evanilson",12 , 10.5, 1),
    ("Ollie Watkins",12 , 10.5, 0),
    ("Richarlison",11.25 , 10.25, 3),
    ("Rodrigo Muniz",11 , 10, 1),
    ("Bruno Fernandes",11 , 10, 2),
    ("Jarrod Bowen",10.75 , 9.75, 2),
    ("Bukayo Saka",10.75 , 9.75, 1),
    ("Beto",10.5 , 9.5, 1),
    ("Dominic Solanke",10 , 9, 0),
    ("Omar Marmoush",9.75 , 8.75, 0),
    ("Wilson Isidor",9.5 , 8.5, 3),
    ("Liam Delap",9.5 , 8.5, 0),
    ("Benjamin Sesko",9.5 , 8.5, 0),
    ("Jorgen Strand Larsen",9.25 , 8.25, 0),
    ("Iliman Ndiaye",9.25 , 8.25, 2),
    ("Phil Foden",9 , 8, 1),
    ("Bryan Mbeumo",9 , 8, 1),
    ("Dominic Calvert-Lewin",8.75 , 7.75, 1),
    ("Kai Havertz",8.75 , 7.75, 0),
    ("Cody Gakpo",8.75 , 7.75, 1),
    ("Anthony Gordon",8.75 , 7.75, 0),
    ("Ismaila Sarr",8.5 , 7.5, 2),
    ("Kaoru Mitoma",8.25 , 7.25, 1),
    ("Danny Welbeck",8.25 , 7.25, 0),
    ("Kevin Schade",8.25 , 7.25, 1),
    ("Lucas Paqueta",8.25 , 7.25, 2),
    ("Jaidon Anthony",7.75 , 7, 3),
    ("Matheus Cunha",8 , 7, 0),
]

columns = ["Player", "Buy", "Sell", "SoFar"]
df_source = pd.DataFrame(default_data, columns=columns)

# --- Session state for data ---
if "df_mid" not in st.session_state:
    df_mid = pd.DataFrame(default_data, columns=columns)
    df_mid["Midpoint"] = df_mid[["Buy", "Sell"]].mean(axis=1)
    df_mid["k"] = 20.0
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

