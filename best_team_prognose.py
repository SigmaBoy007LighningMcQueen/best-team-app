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
MAX_SPIELER_PRO_VEREIN = 1

WUNSCHSPIELER = set()
PROGNOSE = {}
AUSGESCHLOSSEN = set()
VERLETZT_HALBES_JAHR = {}

# ---------------------------
# FUNKTIONEN
# ---------------------------
def read_players_from_github(url: str) -> List[Dict]:
    resp = requests.get(url)
    resp.raise_for_status()
    df = pd.read_excel(BytesIO(resp.content))
    df["Marktwert"] = pd.to_numeric(df["MW mio."], errors="coerce") * 1_000_000
    df["Punkte"] = pd.to_numeric(df["Pkt"], errors="coerce")
    df["Verein"] = df["Team"]
    df["ID"] = df["Name"]
    df["Angezeigter Name"] = df["Name"]
    df = df.dropna(subset=["Marktwert", "Punkte", "Position"])
    df = df[df["Marktwert"] >= 0]
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

def dp_position(players: List[Dict], need: int, budget_left: int):
    """
    Knapsack für eine Position: Max Punkte bei Budget.
    """
    n = len(players)
    dp = [[-10**15]*(budget_left+1) for _ in range(need+1)]
    choose = [[set() for _ in range(budget_left+1)] for _ in range(need+1)]
    dp[0][0] = 0

    for p in players:
        cost = int(p["Marktwert"])
        points = p["Punkte"]
        # rückwärts, damit jeder Spieler nur einmal benutzt wird
        for k in range(need, 0, -1):
            for b in range(budget_left, cost-1, -1):
                if dp[k-1][b-cost] + points > dp[k][b]:
                    dp[k][b] = dp[k-1][b-cost] + points
                    choose[k][b] = choose[k-1][b-cost].copy()
                    choose[k][b].add(p["ID"])
    # Wähle maximale Punkte bei genau need Spielern
    max_points, best_b = max((dp[need][b], b) for b in range(budget_left+1))
    return choose[need][best_b]

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

def build_team(players_all, formation, budget, wish_players=set(), max_pro_club=1):
    team = []

    # 1. Wunschspieler einfügen, nur wenn Budget passt
    used_budget = 0
    for p in players_all:
        if p["Angezeigter Name"] in wish_players:
            if used_budget + p["Marktwert"] <= budget:
                team.append(p)
                used_budget += p["Marktwert"]

    # 2. Positionen auffüllen via Knapsack DP
    for pos, need in formation.items():
        already = sum(1 for p in team if p["Position"] == pos)
        to_pick = need - already
        if to_pick <= 0:
            continue
        pool = [p for p in players_all if p["Position"] == pos and p not in team]
        pool.sort(key=lambda x: (-x["Punkte"], x["Marktwert"]))  # Punkt-maximierend
        # Budget in ganzen Einheiten (int)
        budget_left = budget - used_budget
        if not pool or budget_left <= 0:
            continue
        chosen_ids = dp_position(pool, to_pick, budget_left)
        for cid in chosen_ids:
            p = next(p for p in pool if p["ID"] == cid)
            team.append(p)
            used_budget += p["Marktwert"]

    # 3. Max 1 Spieler pro Verein durchsetzen
    team = enforce_team_limit(team, max_pro_club)

    # 4. Teamgröße auf 11 auffüllen, falls noch Platz, budget-strikt
    remaining_slots = 11 - len(team)
    if remaining_slots > 0:
        pool = [p for p in players_all if p not in team]
        pool.sort(key=lambda x: (-x["Punkte"], x["Marktwert"]))
        for p in pool:
            if remaining_slots <= 0:
                break
            if used_budget + p["Marktwert"] <= budget and sum(1 for t in team if t["Verein"] == p["Verein"]) < max_pro_club:
                team.append(p)
                used_budget += p["Marktwert"]
                remaining_slots -= 1

    return team

# ---------------------------
# STREAMLIT APP
# ---------------------------
st.title("⚽ Kicker Manager – Beste 37-Mio-Kombi Prognose")

# Daten laden
github_url = "https://raw.githubusercontent.com/SigmaBoy007LighningMcQueen/best-team-app/main/spieler_mit_position.xlsx"
players_all = read_players_from_github(github_url)
players_all = apply_prognosen(players_all)

# Sidebar Wunschspieler
st.sidebar.subheader("⭐ Wunschspieler")
wunschspieler_input = st.sidebar.multiselect(
    "Wähle deine Wunschspieler aus",
    options=[p["Angezeigter Name"] for p in players_all],
)
WUNSCHSPIELER = set(wunschspieler_input)

# Sidebar Ausgeschlossene Spieler
st.sidebar.subheader("⛔ Ausgeschlossene Spieler")
ausgeschlossen_input = st.sidebar.multiselect(
    "Wähle Spieler, die ausgeschlossen werden sollen",
    options=[p["Angezeigter Name"] for p in players_all],
)
AUSGESCHLOSSEN = set(ausgeschlossen_input)
players_all = [p for p in players_all if p["Angezeigter Name"] not in AUSGESCHLOSSEN]

# Team bauen
team = build_team(players_all, FORMATION, BUDGET, WUNSCHSPIELER, MAX_SPIELER_PRO_VEREIN)

# Sortieren für Anzeige
pos_rank = {"GOALKEEPER":0,"DEFENDER":1,"MIDFIELDER":2,"FORWARD":3}
team.sort(key=lambda r: (pos_rank[r["Position"]], -r["Punkte"], r["Marktwert"]))

# Tabelle anzeigen
team_df = pd.DataFrame(team)[["Position","Angezeigter Name","Verein","Punkte","Marktwert","ID"]]
st.subheader("📊 Beste 37-Mio-Kombination")
st.dataframe(team_df)

# Gesamtpunkte & Kosten
total_cost = sum(r["Marktwert"] for r in team)
total_points = sum(r["Punkte"] for r in team)
st.write(f"**Gesamtpunkte:** {int(total_points)}")
st.write(f"**Gesamtkosten:** {int(total_cost):,}".replace(",", "."))

# CSV-Download
csv = team_df.to_csv(index=False, sep=";").encode('utf-8')
st.download_button(
    label="Team als CSV herunterladen",
    data=csv,
    file_name='kicker_manager_best_team_prognose_wunsch.csv',
    mime='text/csv',
)
