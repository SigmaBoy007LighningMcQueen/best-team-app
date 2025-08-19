import streamlit as st
import pandas as pd
from typing import List, Dict
from collections import defaultdict
import requests
from io import BytesIO

# ---------------------------
# KONSTANTEN & EINSTELLUNGEN
# ---------------------------
BUDGET = 37_000_000
FORMATION = {"GOALKEEPER": 1, "DEFENDER": 3, "MIDFIELDER": 4, "FORWARD": 3}
UNIT = 100_000  # Budget-Einheiten
MAX_SPIELER_PRO_VEREIN = 1

WUNSCHSPIELER = set()
AUSGESCHLOSSEN = set()
PROGNOSE = {}
VERLETZT_HALBES_JAHR = {}

# ---------------------------
# FUNKTIONEN
# ---------------------------
def read_players_from_github(url: str) -> List[Dict]:
    resp = requests.get(url)
    resp.raise_for_status()
    file_bytes = BytesIO(resp.content)
    df = pd.read_excel(file_bytes)
    
    df["Marktwert"] = pd.to_numeric(df["MW mio."], errors="coerce") * 1_000_000
    df["Punkte"] = pd.to_numeric(df["Pkt"], errors="coerce")
    df["Verein"] = df["Team"]
    df["ID"] = df["Name"]
    df["Angezeigter Name"] = df["Name"]
    df = df.dropna(subset=["Marktwert","Punkte","Position"])
    df = df[df["Marktwert"] >= 0]
    df["cost_u"] = (df["Marktwert"] // UNIT).astype(int)
    return df.to_dict("records")

def apply_prognosen(players: List[Dict]) -> List[Dict]:
    for p in players:
        name = (p.get("Angezeigter Name") or "").strip()
        if not name:
            name = (p.get("Vorname", "") + " " + p.get("Nachname", "")).strip()
        if name in AUSGESCHLOSSEN:
            p["Punkte"] = 0
        elif name in PROGNOSE:
            p["Punkte"] = PROGNOSE[name]
        elif name in VERLETZT_HALBES_JAHR:
            p["Punkte"] = int(p["Punkte"] * VERLETZT_HALBES_JAHR[name])
    return players

def dp_position(players, n, B):
    dp = [[-10**15]*(B+1) for _ in range(n+1)]
    choose = [[False]*(B+1) for _ in range(n+1)]
    dp[0][0] = 0
    for i in range(1,n+1):
        p = players[i-1]
        c = p["cost_u"]
        v = p["Punkte"]
        for b in range(B+1):
            dp[i][b] = dp[i-1][b]
            choose[i][b] = False
            if b >= c and dp[i-1][b-c]+v > dp[i][b]:
                dp[i][b] = dp[i-1][b-c]+v
                choose[i][b] = True
    return dp, choose

def merge_blocks(g, dp, n):
    B = len(g)-1
    new_g = [-10**15]*(B+1)
    split_b = [-1]*(B+1)
    for b in range(B+1):
        for b2 in range(b+1):
            if g[b2] + dp[n][b-b2] > new_g[b]:
                new_g[b] = g[b2] + dp[n][b-b2]
                split_b[b] = b2
    return new_g, split_b

def reconstruct(order, blocks, splits, best_b):
    chosen_ids = []
    b = best_b
    for pos, split_b_list in zip(reversed(order), reversed(splits)):
        pools, need, dp, choose = blocks[pos]
        b2 = split_b_list[b]
        i = need
        while i>0 and b>=0:
            if choose[i][b-b2]:
                chosen_ids.append(pools[i-1]["ID"])
                b -= pools[i-1]["cost_u"]
            i -= 1
        b = b2
    return chosen_ids

def enforce_team_limit(team, max_pro_club):
    clubs = defaultdict(list)
    for p in team:
        clubs[p["Verein"]].append(p)
    for club, players in clubs.items():
        if len(players) > max_pro_club:
            players.sort(key=lambda x: x["Punkte"])
            for p in players[max_pro_club:]:
                team.remove(p)
    return team

def refill_team(team, players_all, formation, budget, max_pro_club):
    """
    Erzeugt ein optimales Team unter Berücksichtigung von Formation, Budget und Vereinslimits.
    Startet immer von Grund auf, auch wenn Wunschspieler vorhanden sind.
    """
    # Pool vorbereiten (ohne ausgeschlossene Spieler)
    pool = [p for p in players_all if p not in team]

    # Budget-Einheiten
    B = int(budget // UNIT)

    # Bestimme, wie viele Spieler pro Position noch fehlen
    formation_needed = formation.copy()
    for p in team:
        pos = p["Position"]
        formation_needed[pos] = max(0, formation_needed[pos]-1)

    order = [pos for pos in ["GOALKEEPER","DEFENDER","MIDFIELDER","FORWARD"] if formation_needed[pos] > 0]

    # Erstelle DP-Blöcke für jede Position
    blocks = {}
    for pos in order:
        candidates = [p for p in pool if p["Position"] == pos]
        candidates.sort(key=lambda x: (-x["Punkte"], x["cost_u"]))
        blocks[pos] = dp_position(candidates, formation_needed[pos], B) + (candidates,)

    # Merge Blöcke
    g = [0.0] + [-10**15]*B
    splits = []
    for pos in order:
        dp, choose, candidates = blocks[pos]
        blocks[pos] = (candidates, formation_needed[pos], dp, choose)
        g, split_b2 = merge_blocks(g, dp, formation_needed[pos])
        splits.append(split_b2)

    best_points, best_b = max((g[b], b) for b in range(B+1))

    chosen_ids = reconstruct(order, blocks, splits, best_b)
    id_to_row = {p["ID"]: p for p in players_all}
    team_final = [id_to_row[i] for i in chosen_ids]

    # Füge Wunschspieler hinzu (vorher sicherstellen, dass sie im Team sind)
    team_final += [p for p in team if p in players_all]

    # Durchsetze max Spieler pro Verein
    team_final = enforce_team_limit(team_final, max_pro_club)

    # Sortiere final nach Position und Punkte
    pos_rank = {"GOALKEEPER":0,"DEFENDER":1,"MIDFIELDER":2,"FORWARD":3}
    team_final.sort(key=lambda r: (pos_rank[r["Position"]], -r["Punkte"], r["Marktwert"]))

    return team_final

# ---------------------------
# STREAMLIT APP
# ---------------------------
st.title("⚽ Kicker Manager – Beste 37-Mio-Kombi Prognose")

# GitHub Excel
github_url = "https://raw.githubusercontent.com/SigmaBoy007LighningMcQueen/best-team-app/main/spieler_mit_position.xlsx"
players_all = read_players_from_github(github_url)
players_all = apply_prognosen(players_all)

# Sidebar: Wunschspieler
st.sidebar.subheader("⭐ Wunschspieler")
wunschspieler_input = st.sidebar.multiselect(
    "Wähle deine Wunschspieler aus",
    options=[p["Angezeigter Name"] for p in players_all],
    default=list(WUNSCHSPIELER)
)
WUNSCHSPIELER = set(wunschspieler_input)

# Sidebar: Ausgeschlossene Spieler
st.sidebar.subheader("⛔ Ausgeschlossene Spieler")
ausgeschlossen_input = st.sidebar.multiselect(
    "Wähle Spieler, die ausgeschlossen werden sollen",
    options=[p["Angezeigter Name"] for p in players_all],
    default=list(AUSGESCHLOSSEN)
)
AUSGESCHLOSSEN = set(ausgeschlossen_input)

# Wunschspieler in Team aufnehmen
fixed_players = []
for name in WUNSCHSPIELER:
    for p in players_all:
        pname = (p.get("Angezeigter Name") or "").strip()
        if pname == name:
            fixed_players.append(p)
            break

# Generiere das finale Team
team = refill_team(fixed_players, players_all, FORMATION, BUDGET, MAX_SPIELER_PRO_VEREIN)

# Ausgabe
team_df = pd.DataFrame(team)[["Position","Angezeigter Name","Verein","Punkte","Marktwert","ID"]]
st.subheader("📊 Beste 37-Mio-Kombination")
st.dataframe(team_df)

total_cost = sum(r["Marktwert"] for r in team)
total_points = sum(r["Punkte"] for r in team)
st.write(f"**Gesamtpunkte:** {int(total_points)}")
st.write(f"**Gesamtkosten:** {int(total_cost):,}".replace(",", "."))

# CSV Download
csv = team_df.to_csv(index=False, sep=";").encode('utf-8')
st.download_button(
    label="Team als CSV herunterladen",
    data=csv,
    file_name='kicker_manager_best_team_prognose_wunsch.csv',
    mime='text/csv',
)
