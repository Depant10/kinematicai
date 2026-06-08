import re
import os

with open('dashboard-app/src/App.jsx.bak', 'r') as f:
    app_code = f.read()

# Add AGGREGATE_DATA import
app_code = app_code.replace("import { REAL_MULTI_DATA } from './data';", "import { REAL_MULTI_DATA, AGGREGATE_DATA } from './data';")

# Add the BucketDistributionChart component
bucket_chart = """
function BucketDistributionChart({ aggregateData }) {
  const [metric, setMetric] = useState('knee_asymmetry');
  const metrics = aggregateData.metrics;
  const buckets = aggregateData.buckets;
  const mData = metrics[metric];

  const W = 1000, H = 460;
  const pad = { l: 56, r: 24, t: 56, b: 40 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;

  let yMax = 0;
  buckets.forEach(b => {
    if (mData[b]) {
        const mx = mData[b].p75 + mData[b].std * 1.5;
        if (mx > yMax) yMax = mx;
    }
  });
  if (yMax === 0) yMax = 1;
  const yMin = 0;

  const sx = i => pad.l + (i + 0.5) * (innerW / buckets.length);
  const sy = y => pad.t + (1 - (Math.min(yMax, Math.max(yMin, y)) - yMin) / (yMax - yMin)) * innerH;

  const yTicks = [0, yMax * 0.25, yMax * 0.5, yMax * 0.75, yMax];
  
  return (
    <div className="chart-wrap">
      <div className="chart-legend" style={{display: 'flex', gap: '10px', alignItems: 'center'}}>
        <span className="legend-item"><span className="legend-swatch swatch-live"/>MEDIAN & IQR (BOX)</span>
        <span className="legend-item"><span style={{width:10, height:0, borderTop:'1px solid #22d3ee'}}/>WHISKERS (±1 STD)</span>
        <span className="legend-item" style={{marginLeft: 'auto'}}>METRIC:</span>
        <select value={metric} onChange={e => setMetric(e.target.value)} style={{background: '#111118', color: '#34d399', border: '1px solid #1e1e2e', padding: '2px 4px'}}>
          <option value="knee_asymmetry">Knee Asymmetry</option>
          <option value="hip_asymmetry">Hip Asymmetry</option>
          <option value="hip_drop">Hip Drop</option>
          <option value="valgus_asymmetry">Valgus Asymmetry</option>
        </select>
      </div>
      <svg className="chart-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#161d27" strokeWidth="0.5"/>
          </pattern>
        </defs>
        <rect x={pad.l} y={pad.t} width={innerW} height={innerH} fill="url(#grid)" opacity="0.6"/>

        <line x1={pad.l} x2={pad.l+innerW} y1={pad.t+innerH} y2={pad.t+innerH} stroke="#243042"/>
        <line x1={pad.l} x2={pad.l} y1={pad.t} y2={pad.t+innerH} stroke="#243042"/>

        {yTicks.map((t, i) => (
          <g key={i}>
            <line x1={pad.l-4} x2={pad.l} y1={sy(t)} y2={sy(t)} stroke="#243042"/>
            <text x={pad.l-8} y={sy(t)+3} textAnchor="end" fill="#5a6675" fontSize="10" fontFamily="IBM Plex Mono">{t.toFixed(2)}</text>
          </g>
        ))}

        {buckets.map((b, i) => {
          const stats = mData[b];
          if (!stats || stats.n === 0) return (
             <text key={b} x={sx(i)} y={pad.t+innerH+18} textAnchor="middle" fill="#5a6675" fontSize="10" fontFamily="IBM Plex Mono">{b}'</text>
          );
          
          const boxW = 30;
          return (
            <g key={b}>
              <line x1={sx(i)} x2={sx(i)} y1={sy(Math.max(0, stats.mean - stats.std))} y2={sy(stats.mean + stats.std)} stroke="#22d3ee" strokeWidth="1" opacity="0.6"/>
              <line x1={sx(i)-boxW/4} x2={sx(i)+boxW/4} y1={sy(Math.max(0, stats.mean - stats.std))} y2={sy(Math.max(0, stats.mean - stats.std))} stroke="#22d3ee" strokeWidth="1" opacity="0.6"/>
              <line x1={sx(i)-boxW/4} x2={sx(i)+boxW/4} y1={sy(stats.mean + stats.std)} y2={sy(stats.mean + stats.std)} stroke="#22d3ee" strokeWidth="1" opacity="0.6"/>
              
              <rect x={sx(i)-boxW/2} y={sy(stats.p75)} width={boxW} height={Math.max(1, sy(stats.p25) - sy(stats.p75))} fill="#10b981" fillOpacity="0.2" stroke="#10b981" strokeWidth="1"/>
              
              <line x1={sx(i)-boxW/2} x2={sx(i)+boxW/2} y1={sy(stats.median)} y2={sy(stats.median)} stroke="#ef4444" strokeWidth="2"/>
              
              <text x={sx(i)} y={pad.t+innerH+18} textAnchor="middle" fill="#5a6675" fontSize="10" fontFamily="IBM Plex Mono">{b}'</text>
              <text x={sx(i)} y={pad.t+innerH+32} textAnchor="middle" fill="#3a4452" fontSize="9" fontFamily="IBM Plex Mono">n={stats.n}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
"""

# Insert it before function App()
app_code = app_code.replace("function App() {", bucket_chart + "\nfunction App() {")

# Update App() to handle viewMode
app_code = app_code.replace(
    "const [selectedPlayer, setSelectedPlayer] = useState('barca_player_1');",
    "const [viewMode, setViewMode] = useState('summary');\n  const [selectedMatch, setSelectedMatch] = useState('match_001');\n  const [selectedPlayer, setSelectedPlayer] = useState('barca_player_1');"
)

# Update Player Card replacement logic inside App return
player_card_orig = """        {/* PLAYER CARD */}
        <div className="cell-player panel">
          <div className="panel-h">
            <h3>SUBJECT · TARGET PLAYER</h3>
            <span className="h-meta">ID · {pData.trackId}</span>
          </div>
          <div className="panel-b">
            <div className="player-card">
              <div className="player-head">
                <div className="headshot">
                  <span className="headshot-corner"/>
                  <span className="headshot-corner br"/>
                </div>
                <div className="player-meta">
                  <div className="player-id">{pData.team} TEAM</div>
                  <div className="player-name" style={{"marginTop":"4px"}}> <select value={selectedPlayer} onChange={e => setSelectedPlayer(e.target.value)} style={{background: '#111118', color: '#34d399', border: '1px solid #1e1e2e', padding: '4px', borderRadius: '4px', fontSize: '14px', fontWeight: 'bold', width: '100%', cursor: 'pointer', outline: 'none'}}>
                    {Object.keys(REAL_MULTI_DATA).map(k => <option key={k} value={k}>{REAL_MULTI_DATA[k].name}</option>)}
                  </select> </div>
                  <div className="player-tags">
                    <span className="tag solid">{pData.profile.pos}</span>
                    <span className="tag">{pData.profile.age} yrs</span>
                    <span className="tag">{pData.profile.height} cm</span>
                    <span className="tag">{pData.profile.weight} kg</span>
                  </div>
                </div>
              </div>

              <div className="risk-block">
                <div className="risk-label">
                  <span>CURRENT RISK INDEX</span>
                  <span className="delta">▲ +14.6 vs T-15:00</span>
                </div>
                <div className="risk-num">
                  {riskIdx}<span className="of">/ 100</span>
                </div>
                <div className="risk-bar" style={{ '--w': `${riskIdx}%` }}/>
                <div className="risk-zones">
                  <span>0 BASELINE</span>
                  <span>40 ELEV</span>
                  <span>70 THRESH</span>
                  <span>100</span>
                </div>
              </div>

              <div className="player-stats">
                <div className="kv"><span className="kv-k">DURATION</span><span className="kv-v">{REAL_STATS.matchMinute} min</span></div>
                <div className="kv"><span className="kv-k">DATA POINTS</span><span className="kv-v">{REAL_STATS.dataPoints}</span></div>
                <div className="kv"><span className="kv-k">KNEE Δ</span><span className="kv-v warn">{REAL_STATS.kneeChange}%</span></div>
                <div className="kv"><span className="kv-k">HIP Δ</span><span className="kv-v crit">HIP {REAL_STATS.hipChange > 0 ? '+' : ''}{REAL_STATS.hipChange}%</span></div>
              </div>
            </div>
          </div>
        </div>"""

player_card_new = """        {/* SELECTOR / PLAYER CARD */}
        <div className="cell-player panel">
          <div className="panel-h">
            <h3>ANALYSIS CONTEXT</h3>
            <span className="h-meta">{viewMode === 'summary' ? 'AGGREGATE MODE' : 'SINGLE CLIP'}</span>
          </div>
          <div className="panel-b" style={{display: 'flex', flexDirection: 'column', gap: '16px'}}>
            
            <div style={{display: 'flex', gap: '8px', marginBottom: '8px'}}>
              <button 
                onClick={() => setViewMode('summary')}
                style={{flex: 1, padding: '8px', background: viewMode === 'summary' ? '#10b981' : '#111118', color: viewMode === 'summary' ? '#fff' : '#5a6675', border: '1px solid #1e1e2e', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold'}}
              >
                MATCH SUMMARY
              </button>
              <button 
                onClick={() => setViewMode('inspector')}
                style={{flex: 1, padding: '8px', background: viewMode === 'inspector' ? '#10b981' : '#111118', color: viewMode === 'inspector' ? '#fff' : '#5a6675', border: '1px solid #1e1e2e', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold'}}
              >
                CLIP INSPECTOR
              </button>
            </div>

            <div style={{display: 'flex', flexDirection: 'column', gap: '4px'}}>
              <label style={{fontSize: '10px', color: '#5a6675', letterSpacing: '0.1em'}}>MATCH SELECTION</label>
              <select value={selectedMatch} onChange={e => setSelectedMatch(e.target.value)} style={{background: '#111118', color: '#34d399', border: '1px solid #1e1e2e', padding: '6px', borderRadius: '4px', fontSize: '14px', width: '100%', outline: 'none'}}>
                <option value="match_001">2017 El Clásico (Match 001)</option>
              </select>
            </div>

            {viewMode === 'inspector' ? (
              <div className="player-card" style={{marginTop: '0px'}}>
                <div className="player-head">
                  <div className="player-meta" style={{width: '100%'}}>
                    <label style={{fontSize: '10px', color: '#5a6675', letterSpacing: '0.1em'}}>CLIP / TRACK ID</label>
                    <select value={selectedPlayer} onChange={e => setSelectedPlayer(e.target.value)} style={{background: '#111118', color: '#34d399', border: '1px solid #1e1e2e', padding: '6px', borderRadius: '4px', fontSize: '14px', fontWeight: 'bold', width: '100%', outline: 'none', marginTop: '4px'}}>
                      {Object.keys(REAL_MULTI_DATA).map(k => <option key={k} value={k}>{REAL_MULTI_DATA[k].name}</option>)}
                    </select>
                    <div className="player-tags" style={{marginTop: '12px'}}>
                      <span className="tag solid">{pData.profile.pos}</span>
                      <span className="tag">{pData.profile.age} yrs</span>
                      <span className="tag">{pData.profile.height} cm</span>
                      <span className="tag">{pData.profile.weight} kg</span>
                    </div>
                  </div>
                </div>

                <div className="risk-block">
                  <div className="risk-label">
                    <span>CLIP RISK INDEX</span>
                  </div>
                  <div className="risk-num">
                    {riskIdx}<span className="of">/ 100</span>
                  </div>
                  <div className="risk-bar" style={{ '--w': `${riskIdx}%` }}/>
                </div>

                <div className="player-stats">
                  <div className="kv"><span className="kv-k">DURATION</span><span className="kv-v">{REAL_STATS.matchMinute} min</span></div>
                  <div className="kv"><span className="kv-k">FRAMES</span><span className="kv-v">{REAL_STATS.dataPoints}</span></div>
                </div>
              </div>
            ) : (
              <div className="player-card" style={{marginTop: '0px'}}>
                 <div className="player-stats" style={{gridTemplateColumns: '1fr', gap: '16px'}}>
                    <div className="kv"><span className="kv-k">TOTAL MATCHES</span><span className="kv-v" style={{fontSize: '24px'}}>1</span></div>
                    <div className="kv"><span className="kv-k">TOTAL CLIPS</span><span className="kv-v" style={{fontSize: '24px'}}>10</span></div>
                    <div className="kv"><span className="kv-k">BUCKET SIZE</span><span className="kv-v" style={{fontSize: '24px'}}>15 MIN</span></div>
                 </div>
                 <div style={{marginTop: '20px', fontSize: '12px', color: '#98a4b3', lineHeight: '1.5'}}>
                   <b>HYPOTHESIS:</b> Do gait asymmetry distributions in late-match buckets differ significantly from early-match buckets when aggregated across clips?
                 </div>
              </div>
            )}
          </div>
        </div>"""

app_code = app_code.replace(player_card_orig, player_card_new)

# Update Main Chart
main_chart_orig = """        {/* MAIN CHART */}
        <div className="cell-chart panel">
          <div className="panel-h">
            <h3>TEMPORAL RISK CURVE · GAIT DEGRADATION vs MATCH MINUTE</h3>
            <span className="h-meta">MOTIONBERT · TCN→XFMR · ANOMALY HEAD v3</span>
          </div>
          <div className="panel-b flush">
            <TemporalRiskChart matchMinute={matchMinute} baseline={baseline} live={live} realStats={REAL_STATS}/>
          </div>
        </div>"""

main_chart_new = """        {/* MAIN CHART */}
        <div className="cell-chart panel">
          <div className="panel-h">
            <h3>{viewMode === 'summary' ? 'GAIT METRIC DISTRIBUTION BY MATCH-MINUTE BUCKET' : 'TEMPORAL RISK CURVE · GAIT DEGRADATION vs MATCH MINUTE'}</h3>
            <span className="h-meta">{viewMode === 'summary' ? 'AGGREGATE STATISTICS' : 'MOTIONBERT · TCN→XFMR'}</span>
          </div>
          <div className="panel-b flush">
            {viewMode === 'summary' ? (
                <BucketDistributionChart aggregateData={AGGREGATE_DATA}/>
            ) : (
                <TemporalRiskChart matchMinute={matchMinute} baseline={baseline} live={live} realStats={REAL_STATS}/>
            )}
          </div>
        </div>"""

app_code = app_code.replace(main_chart_orig, main_chart_new)

# Telemetry update
tele_orig = """        {/* TELEMETRY */}
        <div className="cell-tele panel">
          <div className="panel-h">
            <h3>BIOMECHANICAL TELEMETRY · 15:00 ROLLING</h3>
            <span className="h-meta">ALL METRICS · PERSONAL BASELINE NORMALIZED</span>
          </div>
          <div className="panel-b flush">
            <div className="tele-grid">
              <TeleCard
                name="KNEE ASYMMETRY"
                sub="Δ |L–R| / mean · norm"
                value={REAL_STATS.kneeAsym}
                unit="rad"
                status="crit"
                trend={REAL_STATS.kneeChange}
                spark={sparkKnee}
                baseline="0.094"
                threshold={70}
              />
              <TeleCard
                name="HIP DROP ANGLE"
                sub="contralateral pelvic dip"
                value={Math.abs(REAL_STATS.lHip - REAL_STATS.rHip).toFixed(1)}
                unit="°"
                status="warn"
                trend={REAL_STATS.hipChange}
                spark={sparkHip}
                baseline="4.1°"
              />
              <TeleCard
                name="VALGUS LOAD PROXY"
                sub="lateral knee deviation · proxy GRF"
                value={REAL_STATS.valgusAsym}
                unit="N·m/kg"
                status="crit"
                trend={22.1}
                spark={sparkValgus}
                baseline="1.78"
              />
            </div>
          </div>
        </div>"""

tele_new = """        {/* TELEMETRY */}
        <div className="cell-tele panel" style={{ opacity: viewMode === 'summary' ? 0.3 : 1, pointerEvents: viewMode === 'summary' ? 'none' : 'auto' }}>
          <div className="panel-h">
            <h3>BIOMECHANICAL TELEMETRY · {viewMode === 'summary' ? 'DISABLED IN SUMMARY' : 'CLIP ROLLING'}</h3>
            <span className="h-meta">ALL METRICS · PERSONAL BASELINE NORMALIZED</span>
          </div>
          <div className="panel-b flush">
            <div className="tele-grid">
              <TeleCard
                name="KNEE ASYMMETRY"
                sub="Δ |L–R| / mean · norm"
                value={REAL_STATS.kneeAsym}
                unit="rad"
                status="crit"
                trend={REAL_STATS.kneeChange}
                spark={sparkKnee}
                baseline="0.094"
                threshold={70}
              />
              <TeleCard
                name="HIP DROP ANGLE"
                sub="contralateral pelvic dip"
                value={Math.abs(REAL_STATS.lHip - REAL_STATS.rHip).toFixed(1)}
                unit="°"
                status="warn"
                trend={REAL_STATS.hipChange}
                spark={sparkHip}
                baseline="4.1°"
              />
              <TeleCard
                name="VALGUS LOAD PROXY"
                sub="lateral knee deviation · proxy GRF"
                value={REAL_STATS.valgusAsym}
                unit="N·m/kg"
                status="crit"
                trend={22.1}
                spark={sparkValgus}
                baseline="1.78"
              />
            </div>
          </div>
        </div>"""

app_code = app_code.replace(tele_orig, tele_new)

# Side panel updates
side_orig = """        {/* SIDE PANEL */}
        <div className="cell-side panel">
          <div className="panel-h">
            <h3>POSE ESTIMATION · LIVE</h3>
            <span className="h-meta">17 KP · COCO</span>
          </div>"""

side_new = """        {/* SIDE PANEL */}
        <div className="cell-side panel" style={{ opacity: viewMode === 'summary' ? 0.3 : 1, pointerEvents: viewMode === 'summary' ? 'none' : 'auto' }}>
          <div className="panel-h">
            <h3>POSE ESTIMATION · {viewMode === 'summary' ? 'DISABLED IN SUMMARY' : 'LIVE'}</h3>
            <span className="h-meta">17 KP · COCO</span>
          </div>"""

app_code = app_code.replace(side_orig, side_new)

# Status bar text update
app_code = app_code.replace("<span>TRACK ID <b>#{pData.trackId}</b></span>", "<span>{viewMode === 'summary' ? 'TRACK ID MULTIPLE (10)' : <>TRACK ID <b>#{pData.trackId}</b></>}</span>")

with open('dashboard-app/src/App.jsx', 'w') as f:
    f.write(app_code)
