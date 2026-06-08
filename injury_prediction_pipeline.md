# Multimodal Injury Prediction Pipeline: Deep Technical Analysis & Market Assessment

## 1. The CV Bottleneck: Broadcast-to-Biomechanics

### The Core Mathematical Problem

The fundamental bottleneck is **monocular 3D pose estimation under projective ambiguity**. A single broadcast camera collapses 3D joint positions onto a 2D image plane via a perspective projection:

```
x_2d = K[R|t] · X_3d
```

where `K` is the intrinsic camera matrix (unknown and *changing* with zoom/pan), `[R|t]` is the extrinsic pose, and `X_3d` is what you need. This is an ill-posed inverse problem — infinite 3D configurations project to the same 2D skeleton. The depth ambiguity is not a software bug; it's a mathematical reality of projective geometry.

**But it gets worse for this use case.** You don't just need pose — you need **joint torques and gait asymmetry**, which require:

1. **Accurate 3D joint angles** (not just positions) — small angular errors compound through the kinematic chain
2. **Temporal derivatives** — velocity and acceleration of joint angles to estimate loading rates
3. **Inverse dynamics** — which requires ground reaction force (GRF) estimation *without a force plate*

The error propagation is brutal. A 5deg error in hip abduction angle at frame `t`, when numerically differentiated twice to get angular acceleration, produces torque estimates that can be off by 40-60%. This is the **primary research risk**.

### State-of-the-Art Research

**Tier 1 — 3D Pose Lifting (the foundation):**

| Architecture | Key Paper | Why It Matters |
|---|---|---|
| **MotionBERT** (Zhu et al., 2023) | *Learning Human Motion Representations: A Unified Perspective* | Dual-stream transformer that pre-trains on both pose and motion tasks. Currently SOTA on Human3.6M for monocular 3D. Handles temporal coherence natively. |
| **D3DP** (Shan et al., 2023) | *Diffusion-Based 3D Human Pose Estimation with Multi-Hypothesis Aggregation* | Uses diffusion models to generate *distributions* over 3D poses rather than point estimates. Critical — uncertainty quantification is needed, not just a single prediction. |
| **4DHumans / HMR 2.0** (Goel et al., 2023) | Uses a ViT backbone with a transformer decoder to regress SMPL body model parameters directly. Gives mesh-level detail, not just skeleton. |

**Tier 2 — Handling broadcast degradation:**

- **SportsPose** (Ingwersen et al., 2023): Multi-view sports dataset with ground-truth 3D. Train domain adaptation from lab to broadcast.
- **ByteTrack** or **BoT-SORT** for multi-object tracking through occlusion. Robust Re-ID needed to maintain player identity when bodies overlap.
- **Camera calibration from pitch lines**: Use the known pitch geometry (105m x 68m) as a calibration target. Sha et al.'s work on sports field registration recovers `K` and `[R|t]` per frame, even through broadcast cuts.

**Tier 3 — The torque problem (hardest):**

- **ContactOpt / HUMOR** (Rempe et al., 2021): Physics-based motion estimation that enforces ground contact constraints and estimates GRFs from video. Closest to markerless inverse dynamics.
- **Physics-informed neural networks (PINNs)**: Embed Euler-Lagrange equations of motion as a physics loss term. The network doesn't just *predict* joint torques — it's constrained to produce physically plausible ones:

```
L_physics = || tau - M(q)q_ddot - C(q,q_dot)q_dot - G(q) ||^2
```

where `M` is the mass-inertia matrix, `C` is Coriolis/centrifugal, `G` is gravity. This regularizer is what makes broadcast-derived torques usable rather than noise.

### Practical Architecture for the CV Head

```
Broadcast Frame
    |
    |---> Field Registration CNN ---> Camera K, [R|t] per frame
    |
    |---> Player Detection (YOLOv8/RT-DETR) ---> Bounding Boxes
    |         |
    |         +---> BoT-SORT Tracker ---> Player ID trajectories
    |
    +---> Crop per player
              |
              v
         HMR 2.0 / MotionBERT
              |
              |---> 3D Joint Positions (per frame)
              |---> SMPL Mesh Parameters
              +---> Uncertainty sigma per joint
                        |
                        v
              Temporal Smoothing (Savitzky-Golay or learned)
                        |
                        v
              Physics-Constrained Inverse Dynamics Head
                        |
                        |---> Joint Torques (tau)
                        |---> Gait Asymmetry Index
                        +---> Confidence Score
```

**Critical design decision**: Output uncertainty bounds on every kinematic feature. Downstream models must know *when the CV is unreliable* (post-occlusion, extreme zoom, broadcast replay cuts) and weight those windows accordingly.

---

## 2. The Temporal Framework: Anomaly Detection Over Match Timelines

### Why This Is Not a Standard Time-Series Problem

The signal has three properties that break most off-the-shelf approaches:

1. **Multi-rate, irregular sampling**: Broadcast gives 25-50 fps, but with cuts, replays, and occlusion gaps. There is *missing data* that is **not random** — it's systematically biased toward high-action moments (exactly when injury risk peaks).
2. **Non-stationarity driven by tactical context**: A player's kinematic baseline shifts when the team changes formation. A sustained 4-3-3 high press vs. a 5-4-1 low block produces fundamentally different movement distributions for the same player.
3. **Extremely rare positive labels**: Non-contact soft-tissue injuries occur ~0.1-0.3 times per player per season. This cannot be framed as supervised classification without severe class imbalance artifacts.

### Recommended Architecture: Hierarchical Temporal Fusion

Don't choose between Transformers and LSTMs — **layer them for different temporal scales**:

```
Layer 1: Stride-Level Encoder (per-stride, ~0.5-1s windows)
    Architecture: 1D Temporal Convolutional Network (TCN)
    Input: Raw joint kinematics per stride cycle
    Output: Stride embedding (gait quality vector)
    Why TCN: Causal convolutions handle variable-length strides
             efficiently; no vanishing gradient over short windows

Layer 2: Segment-Level Contextualizer (~5-15 min tactical segments)
    Architecture: Transformer Encoder with rotary positional embeddings
    Input: Sequence of stride embeddings + tactical context tokens
    Output: Contextualized movement state per segment
    Why Transformer: Self-attention captures non-local dependencies
                     (e.g., "this sprint pattern resembles one from
                     30 minutes ago that preceded a gait change")

Layer 3: Match-Level Anomaly Detector (full 90 min)
    Architecture: Variational Autoencoder (VAE) or
                  Neural Controlled Differential Equation (Neural CDE)
    Input: Sequence of segment embeddings
    Output: Reconstruction error -> anomaly score
    Why generative: With rare labels, learn the manifold of
                    "normal fatigue progression" and flag deviations
```

### The Tactical Conditioning Mechanism

Inject tactical context as **conditional input**, not just another feature:

```python
# Pseudocode for tactical conditioning
class TacticalConditionedEncoder(nn.Module):
    def __init__(self, d_model, n_formations):
        self.formation_embed = nn.Embedding(n_formations, d_model)
        self.press_intensity = nn.Linear(1, d_model)  # PPDA or similar
        self.film_gamma = nn.Linear(d_model, d_model)  # FiLM conditioning
        self.film_beta = nn.Linear(d_model, d_model)

    def condition(self, kinematic_features, tactical_context):
        # FiLM: Feature-wise Linear Modulation
        # Shifts the kinematic representation based on tactical state
        ctx = self.formation_embed(tactical_context.formation)
        ctx = ctx + self.press_intensity(tactical_context.ppda)
        gamma = self.film_gamma(ctx)
        beta = self.film_beta(ctx)
        return gamma * kinematic_features + beta
```

**FiLM conditioning** (Perez et al., 2018) is the right modulation approach — it lets the tactical context *rescale and shift* what counts as "normal" kinematics, rather than just concatenating features and hoping the model figures it out.

### Key Research

| Paper/Method | Relevance |
|---|---|
| **Neural CDEs** (Kidger et al., 2020) | Continuous-time sequence models that natively handle irregular sampling. Superior to LSTMs for this data. |
| **TFC** (Zhang et al., 2022) — *Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency* | Pre-train temporal encoder without injury labels using contrastive learning across time and frequency domains. |
| **FITS** (Xu et al., 2024) | Lightweight frequency-domain interpolation for time series — useful for the variable-framerate problem. |
| **Anomaly Transformer** (Xu et al., 2022) | Association-discrepancy based anomaly detection specifically designed for time-series. Compute both prior and series associations to detect distributional shifts. |

### Label Strategy

Since true injury labels are too rare for supervised learning:

1. **Pre-train** with self-supervised contrastive loss (normal vs. fatigued movement, which *can* be labeled via match minute / distance covered)
2. **Fine-tune the anomaly threshold** using a small set of known injury events + expert physiotherapist annotations of "concerning movement patterns"
3. **Output a continuous risk score**, not a binary prediction. Frame it as: *"How far is this player's current movement signature from their personal baseline, conditioned on the tactical load they've been under?"*

---

## 3. AWS Architecture: Enterprise-Grade Inference Pipeline

```
+---------------------------------------------------------------------+
|                        INGESTION LAYER                               |
|                                                                      |
|  S3 Bucket (Raw)          <-- MediaConvert (normalize to H.264,     |
|  s3://injury-pred/raw/        constant 25fps, consistent codec)      |
|       |                   <-- Kinesis Video Streams (live ingest)    |
|       |                   <-- S3 Transfer Acceleration (upload)      |
+-------|------------------------------------------------------------- +
        |
        v
+---------------------------------------------------------------------+
|                     PREPROCESSING LAYER                              |
|                                                                      |
|  Step Functions Orchestrator                                         |
|       |                                                              |
|       |---> Lambda: Scene-cut detection, replay removal             |
|       |    (lightweight — ffprobe + simple CNN classifier)            |
|       |                                                              |
|       |---> Lambda: Pitch registration & camera param extraction    |
|       |    (per-shot segments, cached in DynamoDB)                   |
|       |                                                              |
|       +---> SQS Queue ---> Frame chunks ready for GPU inference     |
|                                                                      |
+-------|------------------------------------------------------------- +
        |
        v
+---------------------------------------------------------------------+
|                    GPU INFERENCE LAYER                                |
|                                                                      |
|  SageMaker Endpoints (or EKS + Karpenter with GPU node pools)       |
|                                                                      |
|  +-----------------+  +------------------+  +------------------+    |
|  | Detection +     |  | 3D Pose Lifting  |  | Inverse Dynamics |    |
|  | Tracking        |  | (MotionBERT)     |  | + Gait Analysis  |    |
|  | (RT-DETR+BoT)   |  |                  |  |                  |    |
|  |                 |  | g5.2xlarge (A10G) |  | g5.xlarge        |    |
|  | g5.xlarge       |  | or p4d (A100)    |  |                  |    |
|  +---------+-------+  +---------+--------+  +---------+--------+    |
|            |                    |                      |             |
|            +--------------------+----------------------+             |
|                                 |                                    |
|  Instance selection:                                                 |
|    Batch (post-match): g5.2xlarge (A10G, cost-effective)            |
|    Near-real-time:     p4d.24xlarge (8xA100, <2min latency)        |
|    Use Spot for batch, On-Demand/RI for live                        |
|                                                                      |
|  Model optimization: TensorRT or torch.compile, FP16 inference,    |
|  batch frames aggressively (32-64 crops per forward pass)           |
|                                                                      |
+-------|------------------------------------------------------------- +
        |
        v
+---------------------------------------------------------------------+
|                   FEATURE STORE + TEMPORAL LAYER                     |
|                                                                      |
|  Kinematic features ---> SageMaker Feature Store (online + offline) |
|                          or Apache Iceberg on S3 (cheaper at scale)  |
|                                                                      |
|  Tactical context   ---> DynamoDB (formation, PPDA, press triggers) |
|  (from separate           ingested via API Gateway + Lambda          |
|   data provider)                                                     |
|                                                                      |
|  Temporal Model     ---> SageMaker Real-time Endpoint               |
|  (anomaly scorer)        Input: player_id + match window            |
|                          Output: risk_score + confidence + features  |
|                                                                      |
+-------|------------------------------------------------------------- +
        |
        v
+---------------------------------------------------------------------+
|                    SERVING + MONITORING                               |
|                                                                      |
|  API Gateway ---> Lambda ---> Risk Score API                        |
|                               (authenticated, per-player, per-match) |
|                                                                      |
|  CloudWatch Metrics:                                                 |
|    - Pose confidence drift (model degradation signal)                |
|    - Inference latency P99                                           |
|    - Anomaly score distribution shift (concept drift)                |
|                                                                      |
|  SageMaker Model Monitor:                                            |
|    - Data quality checks on kinematic features                       |
|    - Bias/drift detection on risk scores per player demographic      |
|                                                                      |
|  S3 (Processed) ---> Athena/QuickSight for retrospective analysis   |
|                                                                      |
+---------------------------------------------------------------------+
```

### Key AWS Design Decisions

**Why SageMaker Endpoints over Lambda for GPU inference**: Lambda has no GPU support. For the CV models (MotionBERT at ~200M params), persistent GPU instances are needed. SageMaker async inference endpoints are ideal for batch — they auto-scale to zero when idle and queue requests via SNS.

**Cost optimization strategy**:
- **Batch processing** (post-match analysis): Use SageMaker Async Endpoints on `g5.2xlarge` Spot instances. A full 90-min match at 25fps = ~135K frames. With batched inference (64 crops/forward pass), a single A10G processes this in ~15-20 minutes. Cost: ~$1.50/match.
- **Near-real-time** (half-time reports): Use provisioned `g5.12xlarge` (4x A10G) with model parallelism. Target latency: <2 min from half-time whistle to risk scores.
- **Storage**: S3 Intelligent-Tiering. Raw video is hot for 7 days (reprocessing), then transitions to Infrequent Access. Extracted features stay hot permanently — they're small (~50MB/match vs. ~10GB raw video).

**Data pipeline orchestration**: Step Functions, not Airflow. For this workload (event-driven, variable duration, needs branching logic for error handling), Step Functions integrates natively with SageMaker, Lambda, and SQS without managing an Airflow cluster.

---

## 4. Market Assessment: Feasibility & Defensibility

### Why the Gap Exists

The current injury prediction market is split into two camps, and neither occupies the proposed position:

**Camp 1 — Wearable-first vendors (Catapult, STATSports, Playermaker)**
They own internal load data (GPS, accelerometer, heart rate). Their injury models are essentially **cumulative load thresholds** — acute:chronic workload ratio, high-speed running distance, etc. These are correlational, not causal. They miss *how* a player is moving, only *how much*. A hamstring doesn't tear because a player ran 11.2 km; it tears because their gait mechanics degraded under specific tactical demands at minute 78.

**Camp 2 — Video analytics vendors (Second Spectrum, SkillCorner, Stats Perform)**
They extract tactical and positional data from broadcast video. They track *where* players are, not *how their bodies are moving*. They can tell you a player made 47 high-intensity sprints. They cannot tell you his right knee valgus angle increased 3 degrees over the second half.

**The proposed position sits in the gap between these two camps** — extracting biomechanical signal from broadcast video and contextualizing it against tactical load. Nobody is doing this because it's genuinely hard.

### What IS Defensible

**1. The compound data asset (the real moat)**
Every match processed generates player-specific kinematic longitudinal profiles. After two seasons: ~100 matches per player x 25fps x 17 joint keypoints = dense movement fingerprints that *no one else has*. This data doesn't exist in any public dataset. Wearable companies can't build it (wrong sensor modality). Video analytics companies won't build it (requires biomechanics expertise they don't have). This compounds over time — models get better, baselines get more personalized, and switching costs increase.

**2. The tactical conditioning layer**
Contextualizing biomechanical anomalies against tactical schemes is a novel framing that requires genuine football domain expertise baked into the model architecture. It's not something a competitor can replicate by just hiring more ML engineers — they need the sports science insight too.

**3. Regulatory/trust barrier**
Medical-adjacent predictions in elite sport carry reputational risk. Clubs will not switch providers lightly once they trust a system. First-mover advantage in trust is real.

### What is NOT Defensible

**1. The CV pipeline itself**
MotionBERT, HMR 2.0, and the underlying pose estimation models are open-source. Any well-funded competitor can replicate the inference stack within 6 months. The *architecture* is not a moat — the *data flywheel* it feeds is.

**2. Broadcast video access**
The footage is not owned. League broadcast rights are controlled by media companies. The entire input pipeline depends on licensing agreements with no negotiating leverage as a startup. Second Spectrum already has exclusive deals with the Premier League. Stats Perform has Opta + broadcast feeds locked up across multiple leagues. **This is the biggest structural risk.**

**3. The academic research**
Everything in the technical architecture is built on published papers. A lab at MIT, ETH Zurich, or a well-funded sports science department could replicate the research. The advantage must come from *engineering execution and data scale*, not algorithmic novelty.

### Feasibility Risks (Ranked by Severity)

**Risk 1: The Physics Gap — CRITICAL**

Extracting clinically meaningful joint torques from monocular broadcast video may not be feasible at the accuracy required. The error propagation from 2D to 3D lifting through numerical differentiation to inverse dynamics is severe.

The hard question: Is a noisy biomechanical signal from broadcast video actually *more predictive* than the clean-but-shallow signal from a $50 IMU on the player's shin? The bet is on **information richness over signal quality**. That bet may not pay off.

Mitigation: Don't target clinical-grade torque estimates. Instead, target **relative anomaly detection** — the absolute value of knee valgus torque is not needed, only that it's *12% higher than this player's rolling baseline*. Relative metrics are more robust to systematic bias in the CV pipeline.

**Risk 2: Broadcast Access — HIGH**

If reliable, low-latency access to broadcast feeds cannot be secured, nothing else matters. The big video analytics companies have multi-year exclusive deals with leagues.

Mitigation options:
- Start with leagues where broadcast rights are less locked down (MLS, Eredivisie, Portuguese Liga, Championship) rather than the Premier League
- Partner with clubs directly — they often have their own tactical camera systems (fixed, high-angle) which are actually *better* for this use case than broadcast feeds (consistent angle, no cuts)
- Position the product to work with club-owned tactical cameras as the primary input, broadcast as secondary

**Risk 3: Ground Truth Scarcity — HIGH**

Non-contact soft-tissue injuries are rare events. Across an entire Premier League season (~380 matches), there are roughly 200-300 muscle injuries total, and only a subset are non-contact. Multi-season, multi-league data is needed to train anything meaningful.

Mitigation: Frame the model as fatigue/anomaly detection, not injury prediction. Predict "movement degradation" (which happens every match and is continuously measurable) rather than "injury" (which is binary and rare). This gives dense training signal and is arguably more useful to practitioners anyway.

**Risk 4: Validation Credibility — MEDIUM**

Clubs will ask: *"Prove this works."* Prospective validation is needed — showing that the system flagged a player *before* they got injured — across a statistically meaningful sample. This takes 1-2 full seasons minimum. Selling will be on promise, not proof, for a long time.

### Market Sizing

| Segment | Size | Willingness to Pay |
|---|---|---|
| Top 5 European leagues (98 clubs) | ~$50-150K/club/year for proven injury analytics | High, but long sales cycles (6-18 months) |
| Next tier (MLS, Eredivisie, etc., ~200 clubs) | ~$20-50K/club/year | Medium, more price-sensitive |
| National federations (FIFA 211 members) | ~$100-500K/federation/year for tournament prep | Lumpy, grant-driven |
| Insurance/reinsurance (player injury policies) | Potentially largest — Lloyd's underwrites multi-billion in player injury coverage | Very high if actuarial value can be demonstrated |

**The insurance angle is underexplored and potentially the most lucrative market.** Insurers pay out massive sums on player injury claims and would pay significantly for better risk models. They don't need real-time — batch analysis is fine. They don't need clinical precision — they need *better-than-baseline* prediction. And they have the budget.

### Strategic Verdict

| Dimension | Rating | Notes |
|---|---|---|
| Market gap reality | **Strong** | Genuine whitespace between wearables and tactical analytics |
| Technical feasibility | **Uncertain** | Hinges entirely on the broadcast-to-biomechanics accuracy threshold |
| Defensibility at scale | **Strong** | Longitudinal kinematic data asset compounds and creates switching costs |
| Defensibility at launch | **Weak** | No data moat yet, no broadcast access locked up, replicable tech stack |
| Time to revenue | **Long** | 12-18 months minimum before a credible product demo, 2+ years to validation |
| Capital intensity | **High** | GPU compute, data licensing, sports science hires, long validation cycles |

### Recommended Next Steps

1. **Run the validation experiment first** (broadcast CV vs. wearable ground truth). If the signal isn't there, pivot before burning capital.
2. **Start with club-owned tactical cameras**, not broadcast. Better signal, easier access, solves two risks at once.
3. **Sell to insurers first**, not clubs. They have budget, tolerance for batch processing, and lower accuracy thresholds. Use insurance revenue to fund the longer club sales cycle.
4. **Accumulate data maniacally.** Every match processed, even for free, builds the moat. The models are open science. The longitudinal player movement database is not.

---

## 5. Refined Action Plan (Starting from Day 1 — Idea Stage)

> **Core Principle**: Do not build anything until the fundamental signal is validated. This plan is designed to kill bad assumptions fast and cheaply.

### Phase 1: Signal Validation (Week 1-2) — "Does the physics even work?"

**Goal**: Answer one question — *Can I extract a gait asymmetry signal from broadcast video that is stable enough to distinguish between players and changes detectably within a match?*

| Day | Action | Output |
|-----|--------|--------|
| 1-2 | Download 2-3 full match broadcasts (YouTube full replays, any league). Set up MMPose or MotionBERT on a local GPU or Colab. | Working pose estimation pipeline on broadcast frames |
| 3-4 | Pick one player per match. Extract hip and knee joint angles across the full 90 minutes. | Raw kinematic time-series CSV per player |
| 5-7 | Plot joint angle distributions in 5-min rolling windows. Compare minute 0-15 vs. minute 75-90. Compute gait asymmetry index (left vs. right stride metrics). | Visualization showing whether fatigue drift is visible above the noise floor |
| 8-10 | Quantify: What is the signal-to-noise ratio? Compute inter-player variance vs. intra-player variance. Can you distinguish Player A from Player B by gait signature alone? | Statistical summary: "signal exists / signal is buried in noise / inconclusive" |
| 11-14 | If signal exists: test on 3 more matches with different broadcast conditions (different stadiums, camera angles, weather). Does it generalize or was the first result a fluke? | Go/no-go decision for Phase 2 |

**Kill criteria**: If intra-player variance across a match is smaller than inter-frame noise, the broadcast resolution is fundamentally insufficient. Pivot to club tactical cameras or abandon the approach.

**Tools needed**: Python, MMPose/MotionBERT, single GPU (local or Colab Pro), matplotlib. Total cost: ~$0-20.

### Phase 2: Problem-Market Fit Discovery (Week 2-3, overlapping with Phase 1)

**Goal**: Find out if the problem you're solving is the problem clubs are actually paying to solve.

**Send 20 LinkedIn messages to these people (expect 3-5 replies):**

| Target | Role | Key Questions |
|--------|------|---------------|
| Club sports scientists / Heads of Performance (x8) | The internal user | "What injury data do you look at weekly? What signals do you wish you had? Would you trust a video-derived biomechanical metric?" |
| Club physios / medical staff (x4) | The clinical validator | "What does a player look like 2-3 days before a soft tissue injury? Can you see it in training? What do you wish you could measure remotely?" |
| Ex-Catapult / STATSports employees (x3) | The industry insider | "What do clubs actually complain about with wearables? What features get ignored? Where are the data gaps?" |
| Sports insurance brokers — Lloyd's, Howden (x3) | The alternative buyer | "How do you currently model player injury risk for premium pricing? What data would materially change your models?" |
| Ex-Second Spectrum / SkillCorner engineers (x2) | The competitor intel | "Has your team ever explored pose estimation on broadcast? If not, why? If yes, what killed it?" |

**Message template**:
> "Hi [Name], I'm an AI engineer researching whether broadcast video can extract useful biomechanical signals for injury risk in elite football. I have no product — I'm testing whether this problem is worth solving. 15 minutes of your time would be genuinely invaluable. Happy to share my findings afterward."

**What you're listening for**:
- Do they light up at "injury prediction" or do they say "we already handle that"?
- Do they describe a *different* pain point you hadn't considered?
- What would they actually pay for vs. what sounds cool but isn't actionable?

### Phase 3: Wedge Selection (Week 3-4)

**Goal**: Based on Phase 1 (signal quality) and Phase 2 (market pull), pick ONE entry point.

| Option | When to Pick It | Signal Requirement | Buyer | First Revenue Path |
|--------|----------------|-------------------|-------|-------------------|
| **A: Transfer Due Diligence** | Conversations reveal sporting directors want biomechanical risk assessment before big signings | Comparative ranking (Player A vs. B), not absolute accuracy | Sporting Director / DoF | One-time analysis fee per target player ($5-25K per report) |
| **B: In-Match Fatigue Monitoring** | Conversations reveal managers/analysts want real-time substitution support | Relative drift detection within a single match | Head Coach / Match Analyst | Subscription, per-season |
| **C: Return-to-Play Validation** | Conversations reveal medical staff lack objective movement benchmarks post-rehab | Before/after comparison for same player (cleanest ground truth) | Head of Medical / Club Doctor | Per-assessment fee or subscription |
| **D: Insurance Risk Modeling** | Brokers confirm they'd pay for better actuarial data on player bodies | Population-level comparative metrics, batch processing | Insurance underwriter | Consulting/data licensing deal |

**Recommended default if conversations are inconclusive: Option A (Transfer Due Diligence).** Reasons:
- Lowest infrastructure requirement (batch, not real-time)
- Clearest budget holder (sporting directors spend $10-100M+ on transfers)
- Accuracy bar is relative, not absolute ("Player X is higher risk than Player Y")
- Natural expansion to insurance (same analysis, different buyer)
- Each report generates data that feeds the longitudinal moat

### Phase 4: Ugly MVP (Week 4-8)

**Goal**: Deliver one real analysis to one real person. No product, no UI, no infrastructure.

| Week | Action |
|------|--------|
| 4-5 | Build a repeatable Jupyter notebook pipeline: broadcast video in → per-player kinematic profile out. Hardcode everything. No abstractions. |
| 5-6 | Produce a **1-page PDF report** for a real transfer target (pick a recent high-profile signing). Include: gait asymmetry index, fatigue curve shape, comparison to 3-5 peers at the same position, confidence intervals. Make it look like something a sporting director would read. |
| 6-7 | Send the report (cold or warm) to 3-5 contacts from Phase 2 conversations. Ask: "Is this useful? Would you have wanted this before signing [Player X]? What's missing?" |
| 7-8 | Iterate based on feedback. If 2+ people say "I would pay for this" — you have a business. If 0 say that — pivot or kill. |

**Do NOT build during this phase**:
- AWS infrastructure (use Colab or a single rented GPU)
- A web app or dashboard
- Real-time processing
- Multi-league support
- Any temporal anomaly detection model (the simple kinematic profile is enough for the MVP)

### Phase 5: Design Partner + Validation (Month 3-6)

**Only enter this phase if Phase 4 produced paying interest.**

| Action | Purpose |
|--------|---------|
| Sign 1-2 design partner clubs (free or heavily discounted) | Get access to their internal wearable data as ground truth to validate your video-derived metrics |
| Run the correlation study: your broadcast-derived gait asymmetry vs. their IMU/Catapult ground truth | This is the publishable validation that makes or breaks the company |
| If correlation > 0.7: write it up, publish/pre-print, use as sales collateral | Credibility unlock — moves you from "interesting idea" to "proven approach" |
| If correlation < 0.5: pivot to club tactical cameras as primary input (fixed angle, higher resolution, no broadcast cuts) and re-run | The thesis may still work with better video input |
| Begin scoping the temporal model (anomaly detection) only after the CV layer is validated | Do not stack research risk |

### Phase 6: Productize (Month 6-12)

**Only enter this phase with validated signal + at least 1 paying customer.**

- Build the AWS pipeline (Section 3 of this document)
- Implement the temporal framework (Section 2 of this document)
- Hire: 1 sports scientist (domain), 1 ML engineer (pipeline), 1 commercial lead (sales)
- Target: 3-5 paying clubs or 1 insurance deal by end of Month 12

### Key Mindset Shifts

| Stop Thinking | Start Thinking |
|---------------|----------------|
| "I'm building an injury prediction platform" | "I'm testing whether broadcast video contains usable biomechanical signal" |
| "I reject wearables" | "Video-first, sensor-augmented when available" |
| "I need AWS infrastructure" | "I need a Jupyter notebook and 5 conversations" |
| "What's the best model architecture?" | "Is there even a signal to model?" |
| "How do I make this defensible?" | "How do I make this *real* for one person?" |
| "My competitor is Catapult" | "My competitor is a club's sports scientist saying 'I already do this'" |

### Decision Tree Summary

```
Week 2: Can I see gait signal in broadcast video?
    |
    +-- NO --> Pivot to club tactical cameras. Retest.
    |              |
    |              +-- Still NO --> Kill the idea. Signal doesn't exist
    |                               at this resolution.
    |              +-- YES --> Continue (but narrower market — need
    |                          club cooperation)
    |
    +-- YES --> Week 3: Do 3+ people confirm this solves a real problem?
                   |
                   +-- NO --> The problem is real but the buyer isn't
                   |          who you think. Reframe based on what
                   |          they actually said they need.
                   |
                   +-- YES --> Week 6: Does the ugly MVP report
                                       get "I'd pay for this"?
                                  |
                                  +-- NO --> Wrong wedge. Try another
                                  |          option from Phase 3.
                                  |
                                  +-- YES --> Phase 5. You have
                                              a business. Go.
```

---

## 6. Competitive Landscape

### Direct Competitors: NONE

Nobody is currently extracting biomechanical injury risk signals from broadcast video and contextualizing them against tactical load. The gap is real.

### Tier 1: Existential Threats (Could replicate in 6-12 months)

| Company | What They Have | What They're Missing | Threat |
|---------|---------------|---------------------|--------|
| **Second Spectrum** (Genius Sports, ~$1.5B) | Exclusive optical tracking: Premier League, La Liga, MLS. Process every match already. | Pose estimation / biomechanics. Track position centroids, not body kinematics. | CRITICAL |
| **SkillCorner** (owned by Stats Perform) | Broadcast-based tracking across 50+ leagues. Already processing same video input. | Same gap — positional only, no body-level kinematics. But they've built the hardest part of this pipeline already. | CRITICAL |
| **Stats Perform** (parent of Opta + SkillCorner) | Largest sports data company. Opta event data + SkillCorner tracking. $1B+ revenue. | Biomechanics outside current product vision. Focused on performance analytics and media. | HIGH |
| **Catapult Sports** (ASX: CAT, ~$200M mcap) | 3,000+ team clients. Wearable data monopoly. Acquired PlayerMaker (gait IMU). | No video-based biomechanics. Moat is hardware, not software. | HIGH |

### Tier 2: Adjacent Competitors (Own a piece, different focus)

| Company | Overlap | Why They Haven't Done It |
|---------|---------|------------------------|
| **Movella/Xsens** | Core tech (3D pose from video) is literally the CV layer. Demonstrated markerless mocap at 2022 World Cup. | Focused on entertainment/broadcast graphics, not injury prediction. |
| **Sportlight Technology** (Catapult ex-CTO) | LiDAR-based biomechanical load in stadiums. Closest to the biomechanical angle. | Requires hardware installation. Not scalable from broadcast. |
| **Zone7** (~$30M raised) | AI injury prediction. Claims 70%+ accuracy. 100+ team partners. | Uses wearable data only. No video capability. **This is the company clubs will compare you to.** |
| **Kitman Labs** (acquired by Catapult) | Injury analytics platform. | Locked into wearable-first via Catapult ecosystem. |

### Tier 3: Research Groups

- **Needham Lab** (KU Leuven) — Markerless mocap validation for sports biomechanics
- **Claudino/Ekstrand group** (UEFA Medical Committee) — Largest football injury epidemiology dataset
- **Kanko et al.** (U of Ottawa) — Markerless vs gold-standard motion capture validation
- **Goel/Pavlakos/Malik labs** (Berkeley/UT Austin) — Building HMR 2.0, 4DHumans foundations

### Tier 4: Future Entrants (Watch List)

- **Apple** — Pose estimation (ARKit), health monitoring, MLS broadcast deal
- **Google DeepMind** — Sports analytics publications, TPU infrastructure
- **Hawk-Eye (Sony)** — Multi-camera in every PL stadium, trivial to add pose
- **FIFA/UEFA** — Could mandate biomechanical monitoring for player welfare
- **Big clubs internally** — Man City, Liverpool, Bayern have 10-15 person data science teams

### Competitive Map

```
                    BIOMECHANICAL DEPTH
                         ^
                         |
           Sportlight    |    *** YOUR POSITION ***
           (LiDAR)       |    (broadcast video + tactical)
                         |
           PlayerMaker   |
           (foot IMU)    |
                         |
           Catapult      |         Zone7
           (wearable)    |    (wearable + AI)
                         |
    INTERNAL DATA -------+---------- EXTERNAL/SCALABLE
    (requires device)    |           (no device needed)
                         |
           STATSports    |    SkillCorner
                         |    Second Spectrum
                         |    Opta / Stats Perform
                         v
                    POSITIONAL ONLY
```

Top-right quadrant is empty. That's either opportunity or a signal the market decided it's not viable.

### Why The Gap Exists (Three Possible Answers)

1. **Technically impossible at broadcast resolution** — Signal-to-noise too low. Phase 1 experiment answers this.
2. **Market doesn't want it enough** — Clubs satisfied with wearables. Phase 2 conversations answer this.
3. **Nobody has combined the right skills yet** — CV + biomechanics + football domain expertise is rare. **This is the bet.**

---

## 7. The One Unique Advantage

**You can analyze a player's body without ever touching them.**

Every wearable company needs the player's club to cooperate. Every tracking company needs a league deal. You just need footage that's already public.

This single asymmetry enables:
- **Transfer due diligence**: Analyze a target player at another club (impossible with wearables — selling club won't share data)
- **Insurance underwriting**: Assess player risk from broadcast without club cooperation
- **Retrospective analysis**: Analyze historical footage that wearable data was never captured for
- **Cross-club comparison**: Compare players across different teams from the same video source

### What IS Defensible

1. **Compound data asset** — Longitudinal kinematic profiles per player. After 2 seasons: ~100 matches x 25fps x 17 keypoints per player. Doesn't exist anywhere else. Compounds over time.
2. **Tactical conditioning insight** — Contextualizing biomechanics against tactical schemes requires genuine football domain knowledge baked into the model.
3. **Trust/regulatory barrier** — Medical-adjacent predictions carry reputational risk. Clubs won't switch once they trust a system.

### What is NOT Defensible

1. **The CV pipeline** — Open-source models (YOLO, MotionBERT, HMR 2.0). Replicable in 6 months.
2. **Broadcast video access** — You don't own the footage. Second Spectrum has exclusive league deals.
3. **Academic research** — Built on published papers. Any lab could replicate.
4. **The idea itself** — Zero IP protection. Defensibility comes from execution and data accumulation.

### Defensibility Timeline

| Stage | Moat Level | Why |
|-------|-----------|-----|
| Now (zero) | None | No data, no code, no relationships |
| After 5 reports (Month 3-4) | Weak | Working pipeline + beginning of player data |
| After 1 season (Month 12) | Moderate | 200-500 player profiles that don't exist elsewhere |
| After 2 seasons (Month 24) | Strong | Longitudinal data impossible to replicate without time travel |
| After validation published (Month 18-30) | Very strong | Peer-reviewed credibility + 2-year data head start |

---

## 8. Dual-Market Strategy: Clubs + Insurers

### Why Insurance Changes Everything

Player injury insurance is enormous and shockingly unsophisticated:
- Premiums: 5-10% of player's annual salary
- $10M/year player costs $500K-1M/year to insure
- Squad of 25 at avg $5M salary = $6-12M/year in premiums
- Underwriters at Lloyd's currently price risk using: age, injury history, position, league
- They have ZERO biomechanical data

### Club Sale vs Insurance Sale

| Dimension | Clubs | Insurers |
|-----------|-------|----------|
| Accuracy bar | High — sports scientists scrutinize | Lower — just need better than age + injury history |
| Sales cycle | 6-18 months | 2-6 months |
| Trust needed | Enormous | Moderate |
| Real-time needed | Often yes | Never — batch is fine |
| Data access politics | Complicated | Irrelevant — analyzing public footage |
| Deal size | $5-25K per report | $200K-2M/year data licensing |

### Insurance Revenue Model

| Tier | Product | Price |
|------|---------|-------|
| Per-player assessment | One-off risk report for underwriting | $2-5K per player |
| Portfolio monitoring | Quarterly updates on all insured players at a club | $100-300K/club/year |
| Data licensing | Raw risk scores via API into actuarial models | $500K-2M/year/insurer |
| Claims validation | Post-injury analysis for fraud/claims assessment | $5-10K per claim |

### Dual-Market Flywheel

```
Club buys transfer report → process 20 matches of target player
  → data feeds insurer's risk model → insurer validates your metrics
  → credibility unlocks next club sale → more data processed → cycle repeats
```

### Revised Market Sizing

| Segment | TAM | Realistic Year 3 Share |
|---------|-----|----------------------|
| Club transfer due diligence | $20-50M | $1-3M |
| Club in-season monitoring | $50-100M | $500K-2M |
| Insurance underwriting data | $100-300M | $1-5M |
| Agent-side assessments | $5-15M | $100-500K |
| **Total Year 3** | | **$2.5-10M** |

### Go-To-Market: Insurance First

```
Month 1-2:  Validate signal (Phase 1 experiment)
Month 2-3:  Talk to insurance brokers (Howden, Lloyd's, Miller, Lockton)
Month 3-4:  Deliver 3-5 free portfolio risk reports to underwriters
Month 4-6:  First paid insurance engagement
Month 6-12: Use insurance revenue + credibility to approach clubs
Month 12-18: Dual revenue: insurance data licensing + club reports
```

---

## 9. Probability Assessment

### Overall Success Probability

```
Signal exists in broadcast video? -------- 35-40% YES
  |
  +-> Signal correlates with outcomes? ---- 60% YES (given signal exists)
       |
       +-> Can close a paying customer? --- ~80% (given validated signal)

Joint probability of real business: ~20-25%
```

This is decent for a deep-tech startup at day zero. Most ideas are sub-5%.

### Failure Modes (Ranked)

1. **Physics gap** (60-65% chance) — Signal too noisy at broadcast resolution. CRITICAL. Phase 1 answers this.
2. **Broadcast access** — Big analytics companies have exclusive league deals. Mitigated by starting with club tactical cameras.
3. **Ground truth scarcity** — Injuries are rare events. Mitigated by framing as fatigue/anomaly detection.
4. **Validation timeline** — 1-2 seasons for prospective proof. Long runway needed.

### If It Works: Realistic Trajectory

| Year | Revenue | Team | Key Milestone |
|------|---------|------|--------------|
| 1 | $75-300K | 1-2 | First paying club or insurer |
| 2 | $500K-1.5M | 3-5 | Validated signal, 500+ player profiles |
| 3 | $2.5-10M | 8-12 | Dual revenue, acquisition interest |
| Exit | $15-60M | — | Acquired by Genius Sports/Catapult/Stats Perform for the data asset |

---

## 10. Technical Pipeline (Installed & Ready)

### Environment

- **GPU**: RTX 5070 (Blackwell, sm_120)
- **PyTorch**: 2.12.0.dev+cu128 (nightly, Blackwell support)
- **Pose Model**: YOLO11m-pose (Ultralytics)
- **Python**: 3.12.10

### Project Location

```
C:\Users\Admin\injury-pred\
  00_test_setup.py           <- Verify installation (ALL PASSED)
  01_process_match.py        <- Video -> keypoints JSON (YOLO11 + RTX 5070)
  02_analyze_gait.py         <- Keypoints -> gait asymmetry + fatigue plots
  03_batch_process.py        <- Batch process multiple matches
  04_compare_matches.py      <- Cross-match comparison
  05_injury_validation.py    <- Track specific player, compare pre-injury vs baseline
  find_player.py             <- Annotate first frame to identify player positions
  yolo11m-pose.pt            <- Pose model (medium, accurate)
  yolo11n-pose.pt            <- Pose model (nano, fast fallback)
  videos/                    <- Match video files
  output/                    <- Results
```

### How To Run (Manual)

```bash
cd C:/Users/Admin/injury-pred

# Download a match
yt-dlp -f "bestvideo[height>=720]+bestaudio" --merge-output-format mp4 \
  -o "videos/match_001.mp4" "URL"

# Find target player position in first frame
python find_player.py videos/match_001.mp4

# Extract keypoints (~15-25 min on RTX 5070)
python 01_process_match.py videos/match_001.mp4 output/match_001_keypoints.json --fps 5

# General gait analysis (auto-selects most visible player)
python 02_analyze_gait.py output/match_001_keypoints.json

# Injury-specific validation (e.g., Rodrygo)
python 05_injury_validation.py output/match_001_keypoints.json \
  --player-pos 520,380 \
  --injury-minute 58 \
  --player-name "Rodrygo" \
  --pre-window 15

# Batch + cross-match comparison
python 03_batch_process.py
python 04_compare_matches.py output/*_timeline.json
```

### Data Extracted Per Player Per Frame

**17 COCO keypoints** (x, y, confidence):
nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles

**6 computed biomechanical metrics**:
- Left/right knee angle (flexion/extension)
- Left/right hip angle
- Knee valgus proxy (lateral displacement from hip-ankle line)
- Hip drop (pelvic instability)

**3 asymmetry indices**:
- Knee asymmetry: |L_knee - R_knee| / mean
- Hip asymmetry: |L_hip - R_hip| / mean
- Valgus asymmetry: |L_valgus - R_valgus| / mean

**Additional metrics**: stride width, stride length, torso lean

**Statistical tests (automatic)**:
- t-test: first half vs second half distribution difference
- Cohen's d: effect size
- Spearman correlation: monotonic trend over match time
- Rolling z-score anomaly detection
- Pre-injury vs baseline comparison (injury validation mode)

---

## 11. Immediate Next Step

**Run the experiment. Download a match. Process it. Look at the plots.**

Everything above is a hypothesis until the data says otherwise. The RTX 5070 is ready. The pipeline is installed. One match takes 15-25 minutes to process.

The answer to "is this feasible, defensible, and unique" lives in the output of:

```bash
python 02_analyze_gait.py output/match_001_keypoints.json
```

Not in more analysis documents.
