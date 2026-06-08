import os
import json

out_js = "dashboard-app/src/soccernet_data.js"
matches_dir = "soccernet_processed/analysis"

matches = {}
if os.path.exists(matches_dir):
    for match_id in os.listdir(matches_dir):
        json_path = os.path.join(matches_dir, match_id, "timeline_analysis.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                matches[match_id] = json.load(f)

js_content = f"export const SOCCERNET_DATA = {json.dumps(matches, indent=2)};"

with open(out_js, 'w') as f:
    f.write(js_content)

print(f"Injected SoccerNet data to {out_js}")
