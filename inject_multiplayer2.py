#!/usr/bin/env python3
import json, gzip, base64, re
import numpy as np
from collections import defaultdict

print("Loading players data...")
with open('output/match_001_players.json') as f:
    players = json.load(f)

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
    player_num = p_key.split('_')[-1]
    name = f"{team_name} Player {player_num} (Track {p_data['track_id']})"
    if p_key == "barca_player_1":
        name = "Lionel Messi (Barca #10 / Track 6)"

    real_data[p_key] = {
        "name": name,
        "team": p_data['team'].upper(),
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

print("Reading dashboard HTML...")
with open('KinematicAI Dashboard _standalone_.backup.html') as f:
    html = f.read()

m_start = html.index('"__bundler/manifest">') + len('"__bundler/manifest">')
m_end = html.index('</script>', m_start)
manifest = json.loads(html[m_start:m_end].strip())

app_uuid = '0ab20de8-f43e-4df5-8241-701cd2dd0635'
entry = manifest[app_uuid]
raw = base64.b64decode(entry['data'])
if entry.get('compressed'):
    raw = gzip.decompress(raw)
app_js = raw.decode('utf-8')

# We inject a global dict, and a global proxy `REAL_STATS` will just be `window.REAL_STATS` or similar
# Wait, let's just make sure we replace the function App() block correctly.
app_js = app_js.replace("function App() {", """
const REAL_MULTI_DATA = """ + json.dumps(real_data) + """;

function App() {
  const [selectedPlayer, setSelectedPlayer] = useState('barca_player_1');
  const pData = REAL_MULTI_DATA[selectedPlayer] || REAL_MULTI_DATA[Object.keys(REAL_MULTI_DATA)[0]];
  const REAL_STATS = pData.stats; window.REAL_STATS = REAL_STATS;
""")

app_js = app_js.replace(
    "function genBaseline() {\n  const r = seeded(7);\n  const pts = [];\n  for (let m = 0; m <= 90; m += 1) {\n    const drift = m * 0.18;\n    const noise = (r() - 0.5) * 2.5;\n    const halfBump = m > 45 && m < 50 ? -3 : 0; // halftime dip\n    pts.push({ x: m, y: 18 + drift + noise + halfBump });\n  }\n  return pts;\n}",
    "function genBaseline(pData) { return pData.baseline; }"
)

app_js = app_js.replace(
    "function genLive(divergeAt = 55, currentMin = 78) {\n  const r = seeded(13);\n  const pts = [];\n  for (let m = 0; m <= 90; m += 1) {\n    if (m > currentMin) break;\n    const drift = m * 0.18;\n    const noise = (r() - 0.5) * 3.2;\n    const halfBump = m > 45 && m < 50 ? -2 : 0;\n    let extra = 0;\n    if (m > divergeAt) {\n      const t = (m - divergeAt) / (90 - divergeAt);\n      extra = Math.pow(t, 1.6) * 48;\n    }\n    pts.push({ x: m, y: 18 + drift + noise + halfBump + extra });\n  }\n  return pts;\n}",
    "function genLive(pData, currentMin) { return pData.live.filter(p => p.x <= currentMin); }"
)

app_js = app_js.replace(
    "const baseline = useMemo(() => genBaseline(), []);",
    "const baseline = useMemo(() => genBaseline(pData), [pData]);"
)
app_js = app_js.replace(
    "const live = useMemo(() => genLive(55, Math.floor(matchMinute)), [Math.floor(matchMinute)]);",
    "const live = useMemo(() => genLive(pData, Math.floor(matchMinute)), [pData, Math.floor(matchMinute)]);"
)

app_js = app_js.replace("const sparkKnee = useMemo(() => genSparkline(101, 1.2), []);", "const sparkKnee = pData.sparkKnee;")
app_js = app_js.replace("const sparkHip = useMemo(() => genSparkline(202, 0.6), []);", "const sparkHip = pData.sparkHip;")
app_js = app_js.replace("const sparkValgus = useMemo(() => genSparkline(303, 1.5), []);", "const sparkValgus = pData.sparkValgus;")

app_js = app_js.replace("const matchMinute = 78 + ((tick * 0.04) % 5);", "const matchMinute = REAL_STATS.matchMinute;")
app_js = app_js.replace("const riskIdx = 73 + Math.round(Math.sin(tick / 3) * 1);", "const riskIdx = REAL_STATS.riskIdx;")
app_js = app_js.replace("const confidence = 0.88 + Math.sin(tick / 4) * 0.02 - 0.005;", "const confidence = REAL_STATS.confidence;")
app_js = app_js.replace("const xMax = 90;", "const xMax = window.REAL_STATS ? window.REAL_STATS.matchMinute : 90;")
app_js = app_js.replace("const xTicks = [0, 15, 30, 45, 60, 75, 90];", "const xTicks = [0, 15, 30, 45, 60, 75, 90, 105];")

select_html = """<select value={selectedPlayer} onChange={e => setSelectedPlayer(e.target.value)} style={{background: '#111118', color: '#34d399', border: '1px solid #1e1e2e', padding: '4px', borderRadius: '4px', fontSize: '14px', fontWeight: 'bold', width: '100%', cursor: 'pointer', outline: 'none'}}>
                    {Object.keys(REAL_MULTI_DATA).map(k => <option key={k} value={k}>{REAL_MULTI_DATA[k].name}</option>)}
                  </select>"""
app_js = app_js.replace('<div className="player-name">Target Player A</div>', f'<div className="player-name" style={{{{"marginTop":"4px"}}}}> {select_html} </div>')
app_js = app_js.replace('SUBJECT 4711-A', '{pData.team} TEAM')

app_js = app_js.replace("148.2°", "{REAL_STATS.lKnee}°")
app_js = app_js.replace("141.6°", "{REAL_STATS.rKnee}°")
app_js = app_js.replace("164.8°", "{REAL_STATS.lHip}°")
app_js = app_js.replace("158.1°", "{REAL_STATS.rHip}°")
app_js = app_js.replace("0.31m", "{(REAL_STATS.strideW/100).toFixed(2)}m")

app_js = app_js.replace('value="0.142"', 'value={REAL_STATS.kneeAsym}')
app_js = app_js.replace('value="6.8"', 'value={Math.abs(REAL_STATS.lHip - REAL_STATS.rHip).toFixed(1)}')
app_js = app_js.replace('value="2.41"', 'value={REAL_STATS.valgusAsym}')
app_js = app_js.replace('trend={18.4}', 'trend={REAL_STATS.kneeChange}')
app_js = app_js.replace('trend={12.4}', 'trend={REAL_STATS.hipChange}')

app_js = app_js.replace("DIVERGENCE ONSET · 55'", "DIVERGENCE ONSET · {REAL_STATS.divergeMin}'")
app_js = app_js.replace("{sx(55)}", "{sx(REAL_STATS.divergeMin)}")

app_js = app_js.replace("117,432", "30,989")
app_js = app_js.replace("MOTIONBERT-L · 198M", "YOLO11m-pose · Apple MPS")
app_js = app_js.replace("FP16 · TENSORRT", "YOLO11m · MPS")
app_js = app_js.replace("FEED <b>SKY-01</b>", "DATA <b>{REAL_STATS.dataPoints} pts</b>")
app_js = app_js.replace("WINDOW · 90:00 LIVE", "WINDOW · {REAL_STATS.matchMinute}:00 ANALYZED")

app_js = app_js.replace("10.84 km", "{REAL_STATS.matchMinute} min")
app_js = app_js.replace(">DISTANCE<", ">DURATION<")
app_js = app_js.replace(">31<", ">{REAL_STATS.dataPoints}<")
app_js = app_js.replace(">HI SPRINTS<", ">DATA POINTS<")
app_js = app_js.replace("1.42", "{REAL_STATS.kneeChange}%")
app_js = app_js.replace(">ACWR (28d)<", ">KNEE Δ<")
app_js = app_js.replace("L HAM 38d", "HIP {REAL_STATS.hipChange > 0 ? '+' : ''}{REAL_STATS.hipChange}%")
app_js = app_js.replace(">PRIOR INJURY<", ">HIP Δ<")

compressed = gzip.compress(app_js.encode('utf-8'))
new_b64 = base64.b64encode(compressed).decode('ascii')
manifest[app_uuid]['data'] = new_b64

new_manifest_json = json.dumps(manifest)
new_html = html[:m_start] + "\n" + new_manifest_json + "\n" + html[m_end:]

with open('KinematicAI Dashboard _standalone_.html', 'w') as f:
    f.write(new_html)

print("✅ Multi-player Dashboard fixed!")
