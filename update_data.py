#!/usr/bin/env python3
import json, os
import numpy as np
from collections import defaultdict

print("Generating updated data.js...")
with open('output/match_001_players.json') as f:
    players = json.load(f)

profiles = {
    "barca_player_1": {"name": "Lionel Messi", "number": 10, "pos": "RW", "age": 25, "height": 170, "weight": 72},
    "barca_player_2": {"name": "Luis Suarez", "number": 9, "pos": "ST", "age": 26, "height": 182, "weight": 85},
    "barca_player_3": {"name": "Andres Iniesta", "number": 8, "pos": "CM", "age": 28, "height": 171, "weight": 68},
    "barca_player_4": {"name": "Sergio Busquets", "number": 5, "pos": "CDM", "age": 24, "height": 189, "weight": 76},
    "barca_player_5": {"name": "Gerard Pique", "number": 3, "pos": "CB", "age": 26, "height": 194, "weight": 85},
    "madrid_player_1": {"name": "Cristiano Ronaldo", "number": 7, "pos": "LW", "age": 28, "height": 187, "weight": 83},
    "madrid_player_2": {"name": "Karim Benzema", "number": 9, "pos": "ST", "age": 25, "height": 185, "weight": 81},
    "madrid_player_3": {"name": "Luka Modric", "number": 10, "pos": "CM", "age": 27, "height": 172, "weight": 66},
    "madrid_player_4": {"name": "Sergio Ramos", "number": 4, "pos": "CB", "age": 27, "height": 184, "weight": 82},
    "madrid_player_5": {"name": "Marcelo", "number": 12, "pos": "LB", "age": 24, "height": 174, "weight": 75}
}

real_data = {}
for p_key, p_data in players.items():
    timeline = p_data['timeline']
    if len(timeline) < 10:
        continue
        
    by_minute = defaultdict(list)
    for pt in timeline:
        m = int(pt['minute'])
        by_minute[m].append(pt)

    minutes_list = sorted(by_minute.keys())
    max_min = max(minutes_list) if minutes_list else 90

    baseline_pts = []
    live_pts = []
    last_score = 20
    for m in range(0, max_min + 1):
        if m in by_minute:
            pts = by_minute[m]
            knee_asym = np.mean([p['knee_asymmetry'] for p in pts])
            hip_asym = np.mean([p['hip_asymmetry'] for p in pts])
            score = min(100, (knee_asym * 200 + hip_asym * 300))
            last_score = score
        else:
            score = last_score
            
        live_pts.append({"x": m, "y": round(score, 1)})
        if m <= max_min // 2:
            baseline_pts.append({"x": m, "y": round(score, 1)})
        else:
            first_half_avg = np.mean([p["y"] for p in baseline_pts]) if baseline_pts else 15
            baseline_pts.append({"x": m, "y": round(first_half_avg + m * 0.1, 1)})

    last_readings = timeline[-10:] if len(timeline) >= 10 else timeline
    avg_last = {
        'left_knee': round(np.mean([r['left_knee_angle'] for r in last_readings]), 1),
        'right_knee': round(np.mean([r['right_knee_angle'] for r in last_readings]), 1),
        'left_hip': round(np.mean([r['left_hip_angle'] for r in last_readings]), 1),
        'right_hip': round(np.mean([r['right_hip_angle'] for r in last_readings]), 1),
        'stride_width': round(np.mean([r['stride_width'] for r in last_readings]), 2),
        'knee_asym': round(np.mean([r['knee_asymmetry'] for r in last_readings]), 3),
        'hip_asym': round(np.mean([r['hip_asymmetry'] for r in last_readings]), 3),
        'valgus_asym': round(np.mean([r['valgus_asymmetry'] for r in last_readings]), 3),
        'confidence': round(np.mean([r['lower_confidence'] for r in last_readings]), 3),
    }

    step = max(1, len(timeline) // 30)
    spark_knee = [round(timeline[i]['knee_asymmetry'] * 100, 1) for i in range(0, len(timeline), step)][:30]
    spark_hip = [round(timeline[i]['hip_asymmetry'] * 100, 1) for i in range(0, len(timeline), step)][:30]
    spark_valgus = [round(timeline[i]['valgus_asymmetry'] * 50, 1) for i in range(0, len(timeline), step)][:30]

    half = len(timeline) // 2
    if half > 0:
        knee_1h = round(np.mean([t['knee_asymmetry'] for t in timeline[:half]]), 4)
        knee_2h = round(np.mean([t['knee_asymmetry'] for t in timeline[half:]]), 4)
        hip_1h = round(np.mean([t['hip_asymmetry'] for t in timeline[:half]]), 4)
        hip_2h = round(np.mean([t['hip_asymmetry'] for t in timeline[half:]]), 4)
    else:
        knee_1h, knee_2h, hip_1h, hip_2h = 0.001, 0.001, 0.001, 0.001
        
    knee_change = round((knee_2h - knee_1h) / max(knee_1h, 0.001) * 100, 1)
    hip_change = round((hip_2h - hip_1h) / max(hip_1h, 0.001) * 100, 1)

    risk_idx = min(99, int(live_pts[-1]["y"])) if live_pts else 50
    diverge_min = 0
    for pt in live_pts:
        if pt["y"] > 40:
            diverge_min = pt["x"]
            break
            
    team_name = "Barcelona" if "barca" in p_key else "Real Madrid"
    
    prof = profiles.get(p_key, {"name": f"Unknown Player", "number": 99, "pos": "SUB", "age": 20, "height": 180, "weight": 75})
    full_name = f"{prof['name']} (#{prof['number']} / Track {p_data['track_id']})"

    real_data[p_key] = {
        "name": full_name,
        "team": team_name.upper(),
        "trackId": p_data['track_id'],
        "profile": prof,
        "baseline": baseline_pts,
        "live": live_pts,
        "sparkKnee": spark_knee,
        "sparkHip": spark_hip,
        "sparkValgus": spark_valgus,
        "stats": {
            "matchMinute": max_min,
            "riskIdx": risk_idx,
            "lKnee": avg_last['left_knee'],
            "rKnee": avg_last['right_knee'],
            "lHip": avg_last['left_hip'],
            "rHip": avg_last['right_hip'],
            "strideW": avg_last['stride_width'],
            "kneeAsym": avg_last['knee_asym'],
            "hipAsym": avg_last['hip_asym'],
            "valgusAsym": avg_last['valgus_asym'],
            "confidence": avg_last['confidence'],
            "kneeChange": knee_change,
            "hipChange": hip_change,
            "divergeMin": diverge_min,
            "dataPoints": len(timeline)
        }
    }

with open('dashboard-app/src/data.js', 'w') as f:
    f.write("export const REAL_MULTI_DATA = " + json.dumps(real_data, indent=2) + ";\n")
print("Done writing new data!")
