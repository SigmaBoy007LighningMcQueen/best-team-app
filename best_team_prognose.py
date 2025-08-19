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

WUNSCHSPIELER = {}
PROGNOSE = {}
AUSGESCHLOSSEN = {}
VERLETZT_HALBES_JAHR = {}

# ---------------------------
# FUNKTIONEN
# ---------------------------
def read_players_from_github(url: str) -> List[Dict]:
    resp = requests.get(url)
    resp.raise_for_status()  # Fehler falls Download fehlschlägt
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
    Füllt das Team nach Formation, Max-Pro-Club und Budget.
    Pflichtpositionen werden zuerst gefüllt, Budget für diese wird aber als "verbrannt" gerechnet.
    Danach werden optionale Spieler nur hinzugefügt, wenn Budget übrig ist.
    """
    # 1. Berechne, wie viele Spieler pro Position noch fehlen
    needed = {}
    for pos, count in formation.items():
        current = sum(1 for p in team if p["Position"] == pos)
        needed[pos] = count - current

    # 2. Pool der verfügbaren Spieler (ohne bereits gewählte)
    pool = [p for p in players_all if p not in team]
    pool.sort(key=lambda x: (-x["Punkte"], x["Marktwert"]))

    # 3. Pflichtpositionen füllen, Budget berücksichtigen **wenn möglich**, aber notfalls überschreiten
    used = sum(p["Marktwert"] for p in team)
    for pos, n in needed.items():
        candidates = [p for p in pool if p["Position"] == pos and sum(1 for t in team if t["Verein"] == p["Verein"]) < max_pro_club]
        for p in candidates[:n]:
    if used + p["Marktwert"] <= budget:
        team.append(p)
        used += p["Marktwert"]


    # 4. Optionale Spieler hinzufügen, nur wenn Budget noch passt
    for p in pool:
        if p not in team and used + p["Marktwert"] <= budget and sum(1 for t in team if t["Verein"] == p["Verein"]) < max_pro_club:
            team.append(p)
            used += p["Marktwert"]

    return team







# ---------------------------
# STREAMLIT APP
# ---------------------------
st.title("⚽ Kicker Manager – Beste 37-Mio-Kombi Prognose")

# Raw-Link von GitHub verwenden!
github_url = "https://raw.githubusercontent.com/SigmaBoy007LighningMcQueen/best-team-app/main/spieler_mit_position.xlsx"
players_all = read_players_from_github(github_url)
st.sidebar.subheader("⭐ Wunschspieler")
wunschspieler_input = st.sidebar.multiselect(
    "Wähle deine Wunschspieler aus",
    options=[p["Angezeigter Name"] for p in players_all],
    default=list(WUNSCHSPIELER)
)
WUNSCHSPIELER = set(wunschspieler_input)

st.sidebar.subheader("⛔ Ausgeschlossene Spieler")
ausgeschlossen_input = st.sidebar.multiselect(
    "Wähle Spieler, die ausgeschlossen werden sollen",
    options=[p["Angezeigter Name"] for p in players_all],
    default=list(AUSGESCHLOSSEN)
)
AUSGESCHLOSSEN = set(ausgeschlossen_input)

players_all = apply_prognosen(players_all)

# Wunschspieler
fixed_players = []
for name in WUNSCHSPIELER:
    for p in players_all:
        pname = (p.get("Angezeigter Name") or "").strip()
        if pname == name:
            fixed_players.append(p)
            break

budget_used = sum(p["Marktwert"] for p in fixed_players)
budget_left = BUDGET - budget_used

formation_left = FORMATION.copy()
for p in fixed_players:
    pos = p["Position"]
    if formation_left[pos] > 0:
        formation_left[pos] -= 1

pools = {}
for pos, need in formation_left.items():
    pool = [p for p in players_all if p["Position"] == pos and p not in fixed_players]
    pool.sort(key=lambda x: (x["Punkte"], -x["Marktwert"]), reverse=True)
    pools[pos] = pool[:200]

B = int(budget_left // UNIT)
order = [pos for pos in ["GOALKEEPER","DEFENDER","MIDFIELDER","FORWARD"] if formation_left[pos] > 0]
blocks = {}
for pos in order:
    dp, choose = dp_position(pools[pos], formation_left[pos], B)
    blocks[pos] = (pools[pos], formation_left[pos], dp, choose)

g = [0.0] + [0.0]*B
splits = []
for pos in order:
    pools_pos, need, dp, choose = blocks[pos]
    g, split_b2 = merge_blocks(g, dp, need)
    splits.append(split_b2)

best_points, best_b = -10**15, -1
for b in range(B+1):
    if g[b] > best_points:
        best_points, best_b = g[b], b

chosen_ids = reconstruct(order, blocks, splits, best_b)
id_to_row = {p["ID"]: p for p in players_all}
team = fixed_players + [id_to_row[i] for i in chosen_ids]
team = enforce_team_limit(team, MAX_SPIELER_PRO_VEREIN)
team = refill_team(team, players_all, FORMATION, BUDGET, MAX_SPIELER_PRO_VEREIN)

pos_rank = {"GOALKEEPER":0,"DEFENDER":1,"MIDFIELDER":2,"FORWARD":3}
team.sort(key=lambda r: (pos_rank[r["Position"]], -r["Punkte"], r["Marktwert"]))

team_df = pd.DataFrame(team)[["Position","Angezeigter Name","Verein","Punkte","Marktwert","ID"]]
st.subheader("📊 Beste 37-Mio-Kombination")
st.dataframe(team_df)

total_cost = sum(r["Marktwert"] for r in team)
total_points = sum(r["Punkte"] for r in team)
st.write(f"**Gesamtpunkte:** {int(total_points)}")
st.write(f"**Gesamtkosten:** {int(total_cost):,}".replace(",", "."))

csv = team_df.to_csv(index=False, sep=";").encode('utf-8')
st.download_button(
    label="Team als CSV herunterladen",
    data=csv,
    file_name='kicker_manager_best_team_prognose_wunsch.csv',
    mime='text/csv',
)











