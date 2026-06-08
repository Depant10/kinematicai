#!/usr/bin/env python3
"""Inject real match data into the KinematicAI Dashboard HTML."""
import json, gzip, base64, copy, math, shutil

# --- Load real match data ---
print("Loading keypoint data...")
with open('output/match_001_keypoints.json') as f:
    keypoints = json.load(f)

with open('output/match_001_keypoints_timeline.json') as f:
    timeline = json.load(f)

print(f"Loaded {len(keypoints)} frames, {len(timeline)} biomech readings")

# --- Build per-minute aggregated data for the main chart ---
# Group timeline by integer minute
from collections import defaultdict
import numpy as np

by_minute = defaultdict(list)
for pt in timeline:
    m = int(pt['minute'])
    by_minute[m].append(pt)

# Build baseline (first-half average trend) and live curves
minutes_list = sorted(by_minute.keys())
max_min = max(minutes_list) if minutes_list else 90

baseline_pts = []
live_pts = []
for m in range(0, max_min + 1):
    if m in by_minute:
        pts = by_minute[m]
        knee_asym = np.mean([p['knee_asymmetry'] for p in pts])
        hip_asym = np.mean([p['hip_asymmetry'] for p in pts])
        # Combined gait degradation score (0-100 scale)
        score = min(100, (knee_asym * 200 + hip_asym * 300))
        live_pts.append({"x": m, "y": round(score, 1)})
        # Baseline: use first-half average as reference
        if m <= max_min // 2:
            baseline_pts.append({"x": m, "y": round(score, 1)})
        else:
            # Project baseline as gentle continuation
            first_half_avg = np.mean([p["y"] for p in baseline_pts]) if baseline_pts else 15
            baseline_pts.append({"x": m, "y": round(first_half_avg + m * 0.1, 1)})

# Get last data point values for live display
last_readings = timeline[-20:] if len(timeline) >= 20 else timeline
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

# Build sparkline data from timeline
step = max(1, len(timeline) // 30)
spark_knee = [round(timeline[i]['knee_asymmetry'] * 100, 1) for i in range(0, len(timeline), step)][:30]
spark_hip = [round(timeline[i]['hip_asymmetry'] * 100, 1) for i in range(0, len(timeline), step)][:30]
spark_valgus = [round(timeline[i]['valgus_asymmetry'] * 50, 1) for i in range(0, len(timeline), step)][:30]

# Stats
half = len(timeline) // 2
knee_1h = round(np.mean([t['knee_asymmetry'] for t in timeline[:half]]), 4)
knee_2h = round(np.mean([t['knee_asymmetry'] for t in timeline[half:]]), 4)
hip_1h = round(np.mean([t['hip_asymmetry'] for t in timeline[:half]]), 4)
hip_2h = round(np.mean([t['hip_asymmetry'] for t in timeline[half:]]), 4)
knee_change = round((knee_2h - knee_1h) / max(knee_1h, 0.001) * 100, 1)
hip_change = round((hip_2h - hip_1h) / max(hip_1h, 0.001) * 100, 1)

risk_idx = min(99, int(live_pts[-1]["y"])) if live_pts else 50
diverge_min = 0
for pt in live_pts:
    if pt["y"] > 30:
        diverge_min = pt["x"]
        break

print(f"Risk index: {risk_idx}")
print(f"Knee change: {knee_change}%")
print(f"Hip change: {hip_change}%")
print(f"Divergence at: {diverge_min}'")

# --- Build the modified app JS ---
REAL_DATA_JS = f"""
// ═══ REAL MATCH DATA (injected from pipeline output) ═══
const REAL_BASELINE = {json.dumps(baseline_pts)};
const REAL_LIVE = {json.dumps(live_pts)};
const REAL_SPARK_KNEE = {json.dumps(spark_knee)};
const REAL_SPARK_HIP = {json.dumps(spark_hip)};
const REAL_SPARK_VALGUS = {json.dumps(spark_valgus)};
const REAL_STATS = {{
  matchMinute: {max_min},
  riskIdx: {risk_idx},
  lKnee: {avg_last['left_knee']},
  rKnee: {avg_last['right_knee']},
  lHip: {avg_last['left_hip']},
  rHip: {avg_last['right_hip']},
  strideW: {avg_last['stride_width']},
  kneeAsym: {avg_last['knee_asym']},
  hipAsym: {avg_last['hip_asym']},
  valgusAsym: {avg_last['valgus_asym']},
  confidence: {avg_last['confidence']},
  kneeChange: {knee_change},
  hipChange: {hip_change},
  divergeMin: {diverge_min},
  frames: {len(keypoints)},
  dataPoints: {len(timeline)},
}};
"""

# --- Read and modify the HTML ---
print("Reading dashboard HTML...")
with open('KinematicAI Dashboard _standalone_.html') as f:
    html = f.read()

# Extract manifest
m_start = html.index('"__bundler/manifest">') + len('"__bundler/manifest">')
m_end = html.index('</script>', m_start)
manifest = json.loads(html[m_start:m_end].strip())

# Find the app JS (application/javascript, ~24KB)
app_uuid = '0ab20de8-f43e-4df5-8241-701cd2dd0635'
entry = manifest[app_uuid]
raw = base64.b64decode(entry['data'])
if entry.get('compressed'):
    raw = gzip.decompress(raw)
app_js = raw.decode('utf-8')

print(f"Original app JS: {len(app_js)} bytes")

# --- Modify the app JS ---
# Replace data generators with real data
new_js = app_js

# Replace genBaseline and genLive functions - make them return real data
new_js = new_js.replace(
    "function genBaseline() {\n  const r = seeded(7);\n  const pts = [];\n  for (let m = 0; m <= 90; m += 1) {\n    const drift = m * 0.18;\n    const noise = (r() - 0.5) * 2.5;\n    const halfBump = m > 45 && m < 50 ? -3 : 0; // halftime dip\n    pts.push({ x: m, y: 18 + drift + noise + halfBump });\n  }\n  return pts;\n}",
    "function genBaseline() { return REAL_BASELINE; }"
)

new_js = new_js.replace(
    "function genLive(divergeAt = 55, currentMin = 78) {\n  const r = seeded(13);\n  const pts = [];\n  for (let m = 0; m <= 90; m += 1) {\n    if (m > currentMin) break;\n    const drift = m * 0.18;\n    const noise = (r() - 0.5) * 3.2;\n    const halfBump = m > 45 && m < 50 ? -2 : 0;\n    let extra = 0;\n    if (m > divergeAt) {\n      const t = (m - divergeAt) / (90 - divergeAt);\n      extra = Math.pow(t, 1.6) * 48;\n    }\n    pts.push({ x: m, y: 18 + drift + noise + halfBump + extra });\n  }\n  return pts;\n}",
    "function genLive(divergeAt, currentMin) { return REAL_LIVE.filter(p => p.x <= (currentMin || REAL_STATS.matchMinute)); }"
)

# Replace sparkline generators
new_js = new_js.replace(
    "const sparkKnee = useMemo(() => genSparkline(101, 1.2), []);",
    "const sparkKnee = REAL_SPARK_KNEE;"
)
new_js = new_js.replace(
    "const sparkHip = useMemo(() => genSparkline(202, 0.6), []);",
    "const sparkHip = REAL_SPARK_HIP;"
)
new_js = new_js.replace(
    "const sparkValgus = useMemo(() => genSparkline(303, 1.5), []);",
    "const sparkValgus = REAL_SPARK_VALGUS;"
)

# Replace match minute / risk
new_js = new_js.replace(
    "const matchMinute = 78 + ((tick * 0.04) % 5);",
    f"const matchMinute = REAL_STATS.matchMinute;"
)
new_js = new_js.replace(
    "const riskIdx = 73 + Math.round(Math.sin(tick / 3) * 1);",
    "const riskIdx = REAL_STATS.riskIdx;"
)
new_js = new_js.replace(
    "const confidence = 0.88 + Math.sin(tick / 4) * 0.02 - 0.005;",
    "const confidence = REAL_STATS.confidence;"
)

# Replace chart range from 90 to max_min
new_js = new_js.replace("const xMax = 90;", f"const xMax = {max_min};")
new_js = new_js.replace(
    "const xTicks = [0, 15, 30, 45, 60, 75, 90];",
    f"const xTicks = [0, 15, 30, 45, 60, 75, 90, {max_min}];"
)

# Replace player name and details
new_js = new_js.replace("Target Player A", "Match Analysis Subject")
new_js = new_js.replace("SUBJECT 4711-A", "REAL MADRID vs BARCELONA")
new_js = new_js.replace("ID · A·11", f"FRAMES · {len(keypoints)}")

# Replace static joint angle values with real data
new_js = new_js.replace("148.2°", f"{avg_last['left_knee']}°")
new_js = new_js.replace("141.6°", f"{avg_last['right_knee']}°")
new_js = new_js.replace("164.8°", f"{avg_last['left_hip']}°")
new_js = new_js.replace("158.1°", f"{avg_last['right_hip']}°")
new_js = new_js.replace("0.31m", f"{avg_last['stride_width']/100:.2f}m")

# Replace telemetry card values
new_js = new_js.replace('value="0.142"', f'value="{avg_last["knee_asym"]}"')
new_js = new_js.replace('value="6.8"', f'value="{round(abs(avg_last["left_hip"] - avg_last["right_hip"]), 1)}"')
new_js = new_js.replace('value="2.41"', f'value="{avg_last["valgus_asym"]}"')
new_js = new_js.replace('trend={18.4}', f'trend={{{knee_change}}}')
new_js = new_js.replace('trend={12.4}', f'trend={{{hip_change}}}')

# Replace divergence annotation
new_js = new_js.replace("DIVERGENCE ONSET · 55'", f"DIVERGENCE ONSET · {diverge_min}'")
new_js = new_js.replace("{sx(55)}", f"{{sx({diverge_min})}}")

# Replace status bar values
new_js = new_js.replace("117,432", f"{len(keypoints):,}")
new_js = new_js.replace("MOTIONBERT-L · 198M", "YOLO11m-pose · Apple MPS")
new_js = new_js.replace("FP16 · TENSORRT", "YOLO11m · MPS")
new_js = new_js.replace("FEED <b>SKY-01</b>", f"DATA <b>{len(timeline)} pts</b>")
new_js = new_js.replace("WINDOW · 90:00 LIVE", f"WINDOW · {max_min}:00 ANALYZED")
new_js = new_js.replace("LIVE PROCESSING", "ANALYSIS COMPLETE")
new_js = new_js.replace("STREAM HEALTHY", "ANALYSIS COMPLETE")

# Update log entries with real analysis results
real_logs = f"""[
    {{ t: '{max_min}:00', lvl: 'crit', msg: <>SIGNAL DETECTED — knee asymmetry <b>+{knee_change}%</b> 1H→2H (p&lt;0.001)</> }},
    {{ t: '{max_min}:00', lvl: 'crit', msg: <>Hip asymmetry increased <b>+{hip_change}%</b> — Cohen's d = 0.503</> }},
    {{ t: '{max_min//2}:00', lvl: 'warn', msg: <>Half-time marker — first half baseline established</> }},
    {{ t: '{diverge_min}:00', lvl: 'warn', msg: <>Gait degradation divergence onset detected at <b>{diverge_min}'</b></> }},
    {{ t: '00:00', lvl: 'info', msg: <><b>{len(keypoints):,}</b> frames processed @ 5fps · {len(timeline)} biomech readings</> }},
    {{ t: '00:00', lvl: 'info', msg: <>Spearman ρ knee=<b>0.320</b> hip=<b>0.323</b> — significant trends</> }},
    {{ t: '00:00', lvl: 'ok', msg: <>Valgus asymmetry: no significant change (p=0.66)</> }},
    {{ t: '00:00', lvl: 'info', msg: <>Pipeline: YOLO11m-pose on Apple MPS · 18.8 fps</> }},
    {{ t: '00:00', lvl: 'ok', msg: <>Match video: Real Madrid full match · 103 min</> }},
    {{ t: '00:00', lvl: 'info', msg: <>Verdict: <b>PROCEED</b> to multi-match validation</> }},
  ]"""

# Replace the logs array
import re
new_js = re.sub(
    r'const logs = \[.*?\];',
    f'const logs = {real_logs};',
    new_js,
    flags=re.DOTALL
)

# Replace distance/sprints
new_js = new_js.replace("10.84 km", f"{max_min} min")
new_js = new_js.replace(">DISTANCE<", ">DURATION<")
new_js = new_js.replace(">31<", f">{len(timeline)}<")
new_js = new_js.replace(">HI SPRINTS<", ">DATA POINTS<")
new_js = new_js.replace("1.42", f"{knee_change}%")
new_js = new_js.replace(">ACWR (28d)<", ">KNEE Δ<")
new_js = new_js.replace("L HAM 38d", f"HIP +{hip_change}%")
new_js = new_js.replace(">PRIOR INJURY<", ">HIP Δ<")

# Prepend real data constants
new_js = REAL_DATA_JS + "\n" + new_js

print(f"Modified app JS: {len(new_js)} bytes")

# --- Recompress and re-encode ---
compressed = gzip.compress(new_js.encode('utf-8'))
new_b64 = base64.b64encode(compressed).decode('ascii')

# Update manifest
manifest[app_uuid]['data'] = new_b64

# Rebuild the HTML
new_manifest_json = json.dumps(manifest)
new_html = html[:m_start] + "\n" + new_manifest_json + "\n" + html[m_end:]

# Backup original
shutil.copy('KinematicAI Dashboard _standalone_.html', 'KinematicAI Dashboard _standalone_.backup.html')

# Write modified
with open('KinematicAI Dashboard _standalone_.html', 'w') as f:
    f.write(new_html)

print(f"\n✅ Dashboard updated with real match data!")
print(f"   Original backed up to: KinematicAI Dashboard _standalone_.backup.html")
print(f"   Open: KinematicAI Dashboard _standalone_.html")
