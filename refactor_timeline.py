import os

with open('dashboard-app/src/App.jsx', 'r') as f:
    code = f.read()

# Import SOCCERNET_DATA
code = code.replace(
    "import { REAL_MULTI_DATA, AGGREGATE_DATA } from './data';",
    "import { REAL_MULTI_DATA, AGGREGATE_DATA } from './data';\nimport { SOCCERNET_DATA } from './soccernet_data';"
)

# Define SoccerNetTimelineChart
timeline_chart = """
function SoccerNetTimelineChart({ playerData }) {
  const [metric, setMetric] = useState('knee_asymmetry');
  
  const mData = playerData?.metrics?.[metric]?.buckets || {};
  const buckets = Object.keys(mData);
  
  const W = 1000, H = 460;
  const pad = { l: 56, r: 24, t: 56, b: 40 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;

  let yMax = 0;
  buckets.forEach(b => {
    if (mData[b] && mData[b].n > 0) {
        const mx = mData[b].mean + mData[b].std * 1.5;
        if (mx > yMax) yMax = mx;
    }
  });
  if (yMax === 0) yMax = 1;
  const yMin = 0;

  const sx = i => pad.l + (i + 0.5) * (innerW / buckets.length);
  const sy = y => pad.t + (1 - (Math.min(yMax, Math.max(yMin, y)) - yMin) / (yMax - yMin)) * innerH;

  const yTicks = [0, yMax * 0.25, yMax * 0.5, yMax * 0.75, yMax];
  
  // Create line path
  const validIndices = buckets.map((b, i) => mData[b] && mData[b].n > 0 ? i : -1).filter(i => i !== -1);
  let dPath = "";
  validIndices.forEach((vi, idx) => {
    const b = buckets[vi];
    const x = sx(vi);
    const y = sy(mData[b].mean);
    dPath += idx === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`;
  });

  return (
    <div className="chart-wrap">
      <div className="chart-legend" style={{display: 'flex', gap: '10px', alignItems: 'center'}}>
        <span className="legend-item"><span className="legend-swatch swatch-live"/>BUCKET MEAN & ±1 STD</span>
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
        
        <path d={dPath} fill="none" stroke="#34d399" strokeWidth="2" strokeDasharray="4 4" />

        {buckets.map((b, i) => {
          const stats = mData[b];
          if (!stats || stats.n === 0) return (
             <text key={b} x={sx(i)} y={pad.t+innerH+18} textAnchor="middle" fill="#5a6675" fontSize="10" fontFamily="IBM Plex Mono">{b}'</text>
          );
          
          return (
            <g key={b}>
              <line x1={sx(i)} x2={sx(i)} y1={sy(Math.max(0, stats.mean - stats.std))} y2={sy(stats.mean + stats.std)} stroke="#22d3ee" strokeWidth="1"/>
              <line x1={sx(i)-10} x2={sx(i)+10} y1={sy(Math.max(0, stats.mean - stats.std))} y2={sy(Math.max(0, stats.mean - stats.std))} stroke="#22d3ee" strokeWidth="1"/>
              <line x1={sx(i)-10} x2={sx(i)+10} y1={sy(stats.mean + stats.std)} y2={sy(stats.mean + stats.std)} stroke="#22d3ee" strokeWidth="1"/>
              
              <circle cx={sx(i)} cy={sy(stats.mean)} r={4} fill="#34d399"/>
              
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

code = code.replace("function App() {", timeline_chart + "\nfunction App() {")

# Update States
code = code.replace(
    "const [selectedMatch, setSelectedMatch] = useState('match_001');",
    "const [selectedMatch, setSelectedMatch] = useState('match_001');\n  const [soccerNetMatch, setSoccerNetMatch] = useState(Object.keys(SOCCERNET_DATA)[0] || '');\n  const [soccerNetTrack, setSoccerNetTrack] = useState(SOCCERNET_DATA[Object.keys(SOCCERNET_DATA)[0]] ? Object.keys(SOCCERNET_DATA[Object.keys(SOCCERNET_DATA)[0]])[0] : '');"
)

# Update the toggle buttons and Selectors
old_toggle = """            <div style={{display: 'flex', gap: '8px', marginBottom: '8px'}}>
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
            </div>"""

new_toggle = """            <div style={{display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '8px'}}>
              <div style={{display: 'flex', gap: '8px'}}>
                <button 
                  onClick={() => setViewMode('summary')}
                  style={{flex: 1, padding: '6px', background: viewMode === 'summary' ? '#10b981' : '#111118', color: viewMode === 'summary' ? '#fff' : '#5a6675', border: '1px solid #1e1e2e', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', fontSize: '11px'}}
                >
                  MATCH SUMMARY
                </button>
                <button 
                  onClick={() => setViewMode('inspector')}
                  style={{flex: 1, padding: '6px', background: viewMode === 'inspector' ? '#10b981' : '#111118', color: viewMode === 'inspector' ? '#fff' : '#5a6675', border: '1px solid #1e1e2e', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', fontSize: '11px'}}
                >
                  CLIP INSPECTOR
                </button>
              </div>
              <button 
                onClick={() => setViewMode('timeline')}
                style={{width: '100%', padding: '6px', background: viewMode === 'timeline' ? '#10b981' : '#111118', color: viewMode === 'timeline' ? '#fff' : '#5a6675', border: '1px solid #1e1e2e', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', fontSize: '11px'}}
              >
                PLAYER TIMELINE (SOCCERNET)
              </button>
            </div>

            <div style={{display: 'flex', flexDirection: 'column', gap: '4px'}}>
              <label style={{fontSize: '10px', color: '#5a6675', letterSpacing: '0.1em'}}>{viewMode === 'timeline' ? 'SOCCERNET MATCH' : 'BROADCAST MATCH'}</label>
              {viewMode === 'timeline' ? (
                <select value={soccerNetMatch} onChange={e => { setSoccerNetMatch(e.target.value); setSoccerNetTrack(Object.keys(SOCCERNET_DATA[e.target.value])[0]); }} style={{background: '#111118', color: '#34d399', border: '1px solid #1e1e2e', padding: '6px', borderRadius: '4px', fontSize: '14px', width: '100%', outline: 'none'}}>
                  {Object.keys(SOCCERNET_DATA).map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              ) : (
                <select value={selectedMatch} onChange={e => setSelectedMatch(e.target.value)} style={{background: '#111118', color: '#34d399', border: '1px solid #1e1e2e', padding: '6px', borderRadius: '4px', fontSize: '14px', width: '100%', outline: 'none'}}>
                  <option value="match_001">2017 El Clásico (Match 001)</option>
                </select>
              )}
            </div>"""

code = code.replace(old_toggle, new_toggle)

# Update Player Card for Timeline
old_inspector = """{viewMode === 'inspector' ? ("""
new_inspector = """{viewMode === 'timeline' && SOCCERNET_DATA[soccerNetMatch] ? (
              <div className="player-card" style={{marginTop: '0px'}}>
                <div className="player-head">
                  <div className="player-meta" style={{width: '100%'}}>
                    <label style={{fontSize: '10px', color: '#5a6675', letterSpacing: '0.1em'}}>PLAYER TRACK</label>
                    <select value={soccerNetTrack} onChange={e => setSoccerNetTrack(e.target.value)} style={{background: '#111118', color: '#34d399', border: '1px solid #1e1e2e', padding: '6px', borderRadius: '4px', fontSize: '14px', fontWeight: 'bold', width: '100%', outline: 'none', marginTop: '4px'}}>
                      {Object.entries(SOCCERNET_DATA[soccerNetMatch]).map(([t, data]) => <option key={t} value={t}>Track {data.track_id} ({data.duration_mins} mins)</option>)}
                    </select>
                  </div>
                </div>
                <div style={{marginTop: '20px', fontSize: '12px', color: '#94a3b8', lineHeight: '1.5', padding: '8px', border: '1px solid #eab30833', background: '#eab30811', borderRadius: '4px'}}>
                   <span style={{color: '#eab308', fontWeight: 'bold'}}>DATA: SoccerNet (research-only)</span><br/>
                   This view leverages ground-truth MOT tracking data for full match temporal analysis.
                </div>
              </div>
            ) : viewMode === 'inspector' ? ("""
code = code.replace(old_inspector, new_inspector)


# Update Main Chart
main_chart_orig = """        {/* MAIN CHART */}
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

main_chart_new = """        {/* MAIN CHART */}
        <div className="cell-chart panel">
          <div className="panel-h">
            <h3>{viewMode === 'timeline' ? 'PLAYER TIMELINE ANALYSIS (SOCCERNET)' : viewMode === 'summary' ? 'GAIT METRIC DISTRIBUTION BY MATCH-MINUTE BUCKET' : 'TEMPORAL RISK CURVE · GAIT DEGRADATION vs MATCH MINUTE'}</h3>
            <span className="h-meta">{viewMode === 'summary' ? 'AGGREGATE STATISTICS' : 'MOTIONBERT · TCN→XFMR'}</span>
          </div>
          <div className="panel-b flush">
            {viewMode === 'timeline' ? (
                <SoccerNetTimelineChart playerData={SOCCERNET_DATA[soccerNetMatch]?.[soccerNetTrack]}/>
            ) : viewMode === 'summary' ? (
                <BucketDistributionChart aggregateData={AGGREGATE_DATA}/>
            ) : (
                <TemporalRiskChart matchMinute={matchMinute} baseline={baseline} live={live} realStats={REAL_STATS}/>
            )}
          </div>
        </div>"""

code = code.replace(main_chart_orig, main_chart_new)

# Update Right Panel (Pose Estimation -> Stats Test Results)
side_orig = """        {/* SIDE PANEL */}
        <div className="cell-side panel" style={{ opacity: viewMode === 'summary' ? 0.3 : 1, pointerEvents: viewMode === 'summary' ? 'none' : 'auto' }}>
          <div className="panel-h">
            <h3>POSE ESTIMATION · {viewMode === 'summary' ? 'DISABLED IN SUMMARY' : 'LIVE'}</h3>
            <span className="h-meta">17 KP · COCO</span>
          </div>
          <div className="panel-b">
            <div className="pose-stats">
              <span className="label">POSE ESTIMATION CONFIDENCE</span>
              <div className="val">0.832<span className="of"> / 1.000</span></div>
              <span className="sub">LOW-CONFIDENCE FRAMES FILTERED · σ &lt; 0.65 DROPPED</span>
              <div className="conf-bar">
                {Array.from({length:30}).map((_,i) => <span key={i} className={`c-tick ${i>24?'fade':''}`}/>)}
              </div>
            </div>

            <div className="skeleton">
              <div className="kp-svg">
                <svg viewBox="0 0 100 150">
                  <path d="M50 20 L50 60 L30 100 L25 140 M50 60 L70 100 L75 140 M50 30 L20 70 M50 30 L80 70" fill="none" stroke="#607d8b" strokeWidth="2"/>
                  <circle cx="50" cy="15" r="4" fill="#607d8b"/>
                  <circle cx="50" cy="30" r="2" fill="#22d3ee"/>
                  <circle cx="50" cy="60" r="2" fill="#22d3ee"/>
                  <circle cx="30" cy="100" r="2" fill="#22d3ee"/>
                  <circle cx="70" cy="100" r="2" fill="#22d3ee"/>
                  <circle cx="20" cy="70" r="2" fill="#22d3ee"/>
                  <circle cx="80" cy="70" r="2" fill="#22d3ee"/>
                  <circle cx="75" cy="140" r="2" fill="#f44336" className="pulse-crit"/>
                </svg>
              </div>
              <div className="kp-meta">
                <div className="k-row"><span className="k-l">L KNEE</span><span className="k-v">{REAL_STATS.lKnee}</span></div>
                <div className="k-row"><span className="k-l">R KNEE</span><span className="k-v">{REAL_STATS.rKnee}</span></div>
                <div className="k-row"><span className="k-l">L HIP</span><span className="k-v">{REAL_STATS.lHip}</span></div>
                <div className="k-row"><span className="k-l">R HIP</span><span className="k-v">{REAL_STATS.rHip}</span></div>
                <div className="k-row"><span className="k-l">STRIDE W</span><span className="k-v">{REAL_STATS.strideWidth}</span></div>
                <div className="k-row"><span className="k-l">TORSO</span><span className="k-v">+4.2°</span></div>
              </div>
            </div>


          </div>
        </div>"""

side_new = """        {/* SIDE PANEL */}
        <div className="cell-side panel" style={{ opacity: viewMode === 'summary' ? 0.3 : 1, pointerEvents: viewMode === 'summary' ? 'none' : 'auto' }}>
          {viewMode === 'timeline' ? (
            <>
              <div className="panel-h">
                <h3>STATISTICAL SIGNIFICANCE</h3>
                <span className="h-meta">MANN-WHITNEY U TEST</span>
              </div>
              <div className="panel-b" style={{display: 'flex', flexDirection: 'column', gap: '16px'}}>
                {SOCCERNET_DATA[soccerNetMatch]?.[soccerNetTrack]?.metrics && Object.entries(SOCCERNET_DATA[soccerNetMatch][soccerNetTrack].metrics).map(([m, data]) => {
                  if (!data.test) return null;
                  const sig = data.test.p_value < 0.05;
                  return (
                    <div key={m} style={{padding: '12px', background: '#111118', border: `1px solid ${sig ? '#f43f5e' : '#1e1e2e'}`, borderRadius: '4px'}}>
                      <div style={{fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '8px', fontWeight: 'bold'}}>{m.replace('_', ' ')}</div>
                      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '4px'}}>
                         <span style={{fontSize: '10px', color: '#64748b'}}>p-value</span>
                         <span style={{fontSize: '12px', color: sig ? '#f43f5e' : '#f8fafc', fontWeight: sig ? 'bold' : 'normal'}}>{data.test.p_value.toFixed(4)} {sig ? '*' : ''}</span>
                      </div>
                      <div style={{display: 'flex', justifyContent: 'space-between'}}>
                         <span style={{fontSize: '10px', color: '#64748b'}}>Cohen's d</span>
                         <span style={{fontSize: '12px', color: '#f8fafc'}}>{data.test.cohens_d.toFixed(2)}</span>
                      </div>
                      <div style={{marginTop: '8px', fontSize: '10px', color: '#475569', borderTop: '1px solid #1e1e2e', paddingTop: '8px'}}>
                         Compare: {data.test.first_bucket} vs {data.test.last_bucket}
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <>
              <div className="panel-h">
                <h3>POSE ESTIMATION · {viewMode === 'summary' ? 'DISABLED IN SUMMARY' : 'LIVE'}</h3>
                <span className="h-meta">17 KP · COCO</span>
              </div>
              <div className="panel-b">
                <div className="pose-stats">
                  <span className="label">POSE ESTIMATION CONFIDENCE</span>
                  <div className="val">0.832<span className="of"> / 1.000</span></div>
                  <span className="sub">LOW-CONFIDENCE FRAMES FILTERED · σ &lt; 0.65 DROPPED</span>
                  <div className="conf-bar">
                    {Array.from({length:30}).map((_,i) => <span key={i} className={`c-tick ${i>24?'fade':''}`}/>)}
                  </div>
                </div>

                <div className="skeleton">
                  <div className="kp-svg">
                    <svg viewBox="0 0 100 150">
                      <path d="M50 20 L50 60 L30 100 L25 140 M50 60 L70 100 L75 140 M50 30 L20 70 M50 30 L80 70" fill="none" stroke="#607d8b" strokeWidth="2"/>
                      <circle cx="50" cy="15" r="4" fill="#607d8b"/>
                      <circle cx="50" cy="30" r="2" fill="#22d3ee"/>
                      <circle cx="50" cy="60" r="2" fill="#22d3ee"/>
                      <circle cx="30" cy="100" r="2" fill="#22d3ee"/>
                      <circle cx="70" cy="100" r="2" fill="#22d3ee"/>
                      <circle cx="20" cy="70" r="2" fill="#22d3ee"/>
                      <circle cx="80" cy="70" r="2" fill="#22d3ee"/>
                      <circle cx="75" cy="140" r="2" fill="#f44336" className="pulse-crit"/>
                    </svg>
                  </div>
                  <div className="kp-meta">
                    <div className="k-row"><span className="k-l">L KNEE</span><span className="k-v">{REAL_STATS.lKnee}</span></div>
                    <div className="k-row"><span className="k-l">R KNEE</span><span className="k-v">{REAL_STATS.rKnee}</span></div>
                    <div className="k-row"><span className="k-l">L HIP</span><span className="k-v">{REAL_STATS.lHip}</span></div>
                    <div className="k-row"><span className="k-l">R HIP</span><span className="k-v">{REAL_STATS.rHip}</span></div>
                    <div className="k-row"><span className="k-l">STRIDE W</span><span className="k-v">{REAL_STATS.strideWidth}</span></div>
                    <div className="k-row"><span className="k-l">TORSO</span><span className="k-v">+4.2°</span></div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>"""

code = code.replace(side_orig, side_new)

with open('dashboard-app/src/App.jsx', 'w') as f:
    f.write(code)

print("Updated App.jsx successfully!")
