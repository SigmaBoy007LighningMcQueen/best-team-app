import sys, math, csv
from typing import List, Dict, Tuple
from collections import defaultdict

BUDGET = 37_000_000
FORMATION = {"GOALKEEPER": 1, "DEFENDER": 3, "MIDFIELDER": 4, "FORWARD": 3}
UNIT = 100_000  # Budget-Einheiten
MAX_SPIELER_PRO_VEREIN = 1   # <<< HIER EINSTELLEN

# Wunschspieler – diese Namen müssen exakt wie in der Datei stehen
WUNSCHSPIELER = {
    
	"Undav",
	"Asllani",
	"Majer",
	"Doan",	
	"Grimaldo",
	"Querfeld",
	"Schwäbe",
}

# Prognosen & Anpassungen
PROGNOSE = {
    
}

AUSGESCHLOSSEN = {
    "Weiper",          # Wird verkauft
	"Ilic",          # Wird verkauft
	"Dompé",          # Wird verkauft
	"Palacios",          # Wird verkauft
	"Essende",          # Wird verkauft
	"Capaldo",          # Wird verkauft
	"Nebel",          # Wird verkauft
	"Vavro",          # Wird verkauft
	
}

VERLETZT_HALBES_JAHR = {
    
}

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

def read_players(path: str) -> List[Dict]:
    import pandas as pd
    # Excel einlesen
    df = pd.read_excel(path)

    # Marktwert (MW mio.) umrechnen auf echte Zahlen in €
    df["Marktwert"] = pd.to_numeric(df["MW mio."], errors="coerce") * 1_000_000
    df["Punkte"] = pd.to_numeric(df["Pkt"], errors="coerce")

    # Spalten für Verein/ID auffüllen
    df["Verein"] = df["Team"]
    df["ID"] = df["Name"]   # Dummy-ID
    df["Angezeigter Name"] = df["Name"]

    # Kosten-Einheiten für DP
    df = df.dropna(subset=["Marktwert","Punkte","Position"])
    df = df[df["Marktwert"] >= 0]
    df["cost_u"] = (df["Marktwert"] // UNIT).astype(int)

    return df.to_dict("records")

def dp_position(players: List[Dict], need: int, B: int):
    NEG = -10**15
    dp = [[NEG]*(B+1) for _ in range(need+1)]
    choose = [[None]*(B+1) for _ in range(need+1)]
    for b in range(B+1):
        dp[0][b] = 0.0
    for idx, p in enumerate(players):
        c = int(p["cost_u"])
        pts = float(p["Punkte"])
        for k in range(need, 0, -1):
            for b in range(B, c-1, -1):
                if dp[k-1][b-c] > NEG:
                    cand = dp[k-1][b-c] + pts
                    if cand > dp[k][b]:
                        dp[k][b] = cand
                        choose[k][b] = (idx, b-c)
    return dp, choose

def merge_blocks(g_prev: List[float], dp_pos: List[List[float]], need: int):
    NEG = -10**15
    B = len(g_prev) - 1
    g_new = [NEG]*(B+1)
    split_b2 = [-1]*(B+1)
    for b in range(B+1):
        best = NEG
        best_b2 = -1
        for b2 in range(b+1):
            prev = g_prev[b - b2]
            cur = dp_pos[need][b2]
            if prev > NEG and cur > NEG:
                val = prev + cur
                if val > best:
                    best = val
                    best_b2 = b2
        g_new[b] = best
        split_b2[b] = best_b2
    return g_new, split_b2

def reconstruct(order, blocks, splits, best_budget_u):
    chosen_ids = []
    b = best_budget_u
    for pos_idx in range(len(order)-1, -1, -1):
        pos = order[pos_idx]
        players, need, dp, choose = blocks[pos]
        b2 = splits[pos_idx][b]
        ids_pos = []
        k = need
        bb = b2
        while k > 0:
            step = choose[k][bb]
            if step is None:
                ids_pos = []
                break
            idx, prev_b = step
            ids_pos.append(players[idx]["ID"])
            bb = prev_b
            k -= 1
        chosen_ids.extend(ids_pos)
        b = b - b2
    return chosen_ids

def enforce_team_limit(team, max_pro_verein):
    """Filtert das Team, sodass max. X Spieler pro Verein bleiben."""
    verein_counter = defaultdict(list)
    for p in team:
        verein_counter[p["Verein"]].append(p)
    new_team = []
    for verein, players in verein_counter.items():
        players.sort(key=lambda x: x["Punkte"], reverse=True)
        new_team.extend(players[:max_pro_verein])
    return new_team

def refill_team(team, players_all, formation, budget, max_pro_verein):
    """Füllt nach dem Limit wieder Spieler nach, bis die Formation stimmt."""
    current_cost = sum(p["Marktwert"] for p in team)
    current_points = sum(p["Punkte"] for p in team)
    budget_left = budget - current_cost

    # Zähle vorhandene Spieler pro Position & Verein
    pos_count = defaultdict(int)
    verein_count = defaultdict(int)
    for p in team:
        pos_count[p["Position"]] += 1
        verein_count[p["Verein"]] += 1

    # Prüfen welche Positionen fehlen
    missing_positions = []
    for pos, need in formation.items():
        if pos_count[pos] < need:
            missing_positions.extend([pos] * (need - pos_count[pos]))

    # Kandidaten suchen
    pool = [p for p in players_all if p not in team]
    pool.sort(key=lambda x: (x["Punkte"] / (x["Marktwert"]+1)), reverse=True)

    for pos in missing_positions:
        for cand in pool:
            if cand["Position"] != pos:
                continue
            if cand["Marktwert"] > budget_left:
                continue
            if verein_count[cand["Verein"]] >= max_pro_verein:
                continue
            # add candidate
            team.append(cand)
            budget_left -= cand["Marktwert"]
            verein_count[cand["Verein"]] += 1
            break
    return team

def main(path):
    players_all = read_players(path)
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

    # Vereinslimit
    team = enforce_team_limit(team, MAX_SPIELER_PRO_VEREIN)

    # Fehlende Positionen nachfüllen
    team = refill_team(team, players_all, FORMATION, BUDGET, MAX_SPIELER_PRO_VEREIN)

    # Ausgabe
    pos_rank = {"GOALKEEPER":0,"DEFENDER":1,"MIDFIELDER":2,"FORWARD":3}
    team.sort(key=lambda r: (pos_rank[r["Position"]], -r["Punkte"], r["Marktwert"]))

    total_cost = sum(r["Marktwert"] for r in team)
    total_points = sum(r["Punkte"] for r in team)

    print("BESTE 37-MIO-KOMBI (3-4-3) — Prognosepunkte mit Vereinslimit")
    print("--------------------------------------------------")
    for r in team:
        name = r.get("Angezeigter Name") or (r.get("Vorname","") + " " + r.get("Nachname","")).strip()
        print(f'{r["Position"]:<11}  {name:<25}  {r["Verein"]:<15}  Punkte: {int(r["Punkte"]):>4}   Preis: {int(r["Marktwert"]):>9,d}')
    print("--------------------------------------------------")
    print(f"Gesamtpunkte: {int(total_points)}")
    print(f"Gesamtkosten: {int(total_cost):,}".replace(",", "."))

    out = "kicker_manager_best_team_prognose_wunsch.csv"
    import pandas as pd
    pd.DataFrame(team)[["Position","Angezeigter Name","Verein","Punkte","Marktwert","ID"]].to_csv(out, index=False, sep=";")
    print(f"\nTeam als CSV gespeichert: {out}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Nutzung: python best_team_prognose.py <pfad-zur-exceldatei>")
        sys.exit(1)
    main(sys.argv[1])
