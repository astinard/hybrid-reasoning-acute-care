# Reinforcement Learning for Clinical Decision Support: A Comprehensive Research Review

## Executive Summary

This document surveys reinforcement learning (RL) approaches for clinical decision support systems, with emphasis on treatment optimization in acute care settings. The review covers 60+ recent papers focusing on sepsis management, mechanical ventilation, medication dosing, and offline/batch RL methodologies critical for safe healthcare deployment.

**Key Findings:**
- Conservative Q-learning and offline RL methods show promise for safe clinical deployment
- State representation from EHR data remains a critical challenge requiring domain expertise
- Off-policy evaluation is essential but faces unique challenges in healthcare settings
- Clinician acceptance depends heavily on interpretability and alignment with clinical protocols
- Real-world deployment requires robust handling of distribution shift and uncertainty

---

## Table of Contents

1. [Off-Policy Evaluation Challenges in Healthcare](#1-off-policy-evaluation-challenges-in-healthcare)
2. [Conservative Q-Learning for Safety](#2-conservative-q-learning-for-safety)
3. [Counterfactual Reasoning in Treatment Decisions](#3-counterfactual-reasoning-in-treatment-decisions)
4. [State Representation from EHR Data](#4-state-representation-from-ehr-data)
5. [Reward Function Design for Clinical Outcomes](#5-reward-function-design-for-clinical-outcomes)
6. [Clinician Acceptance and Interpretability](#6-clinician-acceptance-and-interpretability)
7. [Real-World Deployment Considerations](#7-real-world-deployment-considerations)
8. [Case Studies: Sepsis and Ventilation](#8-case-studies-sepsis-and-ventilation)
9. [Future Directions](#9-future-directions)

---

## 1. Off-Policy Evaluation Challenges in Healthcare

### 1.1 The Critical Importance of OPE in Clinical Settings

Off-policy evaluation (OPE) is paramount in healthcare because online policy evaluation through direct patient interaction is often unsafe, unethical, or prohibitively expensive. Unlike simulation environments where policies can be tested freely, clinical decision support systems must be validated exclusively on historical data before deployment.

**Key Challenge:** Standard OPE methods developed for general RL settings often fail in healthcare due to:
- High-stakes consequences of estimation errors
- Partial observability of patient states
- Unobserved confounders affecting both treatments and outcomes
- Sparse and delayed rewards (e.g., 90-day mortality)
- Non-stationary behavior policies in logged data

### 1.2 Fundamental OPE Methods and Their Healthcare Limitations

**Importance Sampling (IS):**
```
V^π(ρ) = E_τ~ρ [∏(t=0 to T) π(a_t|s_t)/μ(a_t|s_t) · R(τ)]
```

Where:
- π is the target policy
- μ is the behavior (logging) policy
- ρ is the data distribution
- R(τ) is the cumulative reward

**Limitations in Healthcare:**
1. **High Variance:** Product of importance weights explodes with trajectory length, common in ICU settings with multi-day stays
2. **Behavior Policy Estimation:** Clinician policies are often unknown or poorly documented
3. **Support Mismatch:** Target policies may recommend actions rarely seen in historical data

**Weighted Importance Sampling (WIS):**
```
V^π_WIS = (Σᵢ wᵢRᵢ) / (Σᵢ wᵢ)
where wᵢ = ∏ᵗ π(aₜ|sₜ)/μ(aₜ|sₜ)
```

Reduces variance but introduces bias, particularly problematic when evaluating novel treatment strategies.

### 1.3 Advanced OPE for Healthcare: The HOPE Framework

Gao et al. (2023) developed HOPE (Human-centric Off-Policy Evaluation) specifically for healthcare challenges:

**Key Innovations:**
1. **Aggregate Reward Reconstruction:** Addresses partially observable states and aggregate rewards (e.g., final test scores, hospital discharge status)
2. **Immediate Reward Estimation:** Decomposes aggregate rewards into immediate rewards considering partial observability
3. **Human-Centric Validation:** Incorporates domain expert feedback in evaluation

**Theoretical Guarantee:**
Under mild assumptions on reward aggregation and state observability, HOPE provides consistent estimates:
```
|V̂^π_HOPE - V^π| ≤ ε with probability ≥ 1-δ
```

where ε decreases as O(1/√n) with sample size n.

**Empirical Results:**
- Sepsis treatment evaluation: HOPE reduced estimation error by 34% vs. standard IS
- Intelligent tutoring systems: Correctly ranked 89% of policy pairs vs. 67% for baseline methods

### 1.4 Model-Based OPE and Fitted Q-Evaluation

**Fitted Q-Evaluation (FQE)** learns the Q-function for the target policy:
```
Q̂^π(s,a) = r(s,a) + γ Σₛ' P(s'|s,a) V^π(s')
where V^π(s') = Σₐ' π(a'|s') Q^π(s',a')
```

**Advantages in Clinical Settings:**
1. Lower variance than IS-based methods
2. Can leverage function approximation for generalization
3. Naturally handles continuous state/action spaces

**Challenges:**
1. **Model Misspecification:** Errors in dynamics model P(s'|s,a) propagate
2. **Distributional Shift:** Q-values for OOD actions are unreliable
3. **Computational Cost:** Requires training separate value networks

**Healthcare-Specific Adaptations:**

Killian et al. (2020) demonstrated that FQE combined with learned state representations achieves superior performance in sepsis policy evaluation:
- Mean squared error reduced by 41% vs. direct Q-learning
- Policy ranking correlation with ground truth: 0.87 (vs. 0.64 for IS)

### 1.5 Doubly-Robust (DR) Estimation

DR methods combine model-based and IS approaches to reduce both bias and variance:

```
V^π_DR = (1/n) Σᵢ [w(τᵢ)(R(τᵢ) - Q̂(τᵢ)) + Σₜ V̂(sₜ^i)]
```

**Key Properties:**
1. Consistent if either the Q-function OR importance weights are correct
2. Variance reduction when both are partially correct
3. More robust to model misspecification

**Challenges in Clinical Data:**
- Requires accurate behavior policy estimation (often unavailable)
- High-dimensional medical records make Q-function learning difficult
- Treatment assignment may depend on unobserved factors

### 1.6 OPE Under Partial Observability

**POMDP Formulation:**
In healthcare, true patient state s is often unobserved; we only observe o:
```
o_t ~ Ω(·|s_t)
μ(a|h_t) where h_t = (o_1, a_1, ..., o_t)
```

**Theoretical Result (Tennenholtz et al., 2019):**
Under partial observability without additional assumptions, OPE can have arbitrarily large bias:
```
|V̂^π - V^π| can grow exponentially with horizon T
```

**Mitigation Strategies:**
1. **Decoupled POMDPs:** Separate observed and unobserved dynamics
2. **Pessimistic Bounds:** Compute worst-case performance under observation uncertainty
3. **Auxiliary Information:** Leverage clinical notes, lab results as additional observations

### 1.7 Evaluating OPE Methods: DataCOPE Framework

Sun et al. (2023) introduced DataCOPE to evaluate when OPE is reliable:

**Three Key Questions:**
1. Can we evaluate the target policy with available data?
2. Which sub-groups in data yield accurate OPE?
3. How does data collection strategy affect OPE quality?

**Methodology:**
- Predictive models for OPE error without environment access
- Identifies high-uncertainty regions in state-action space
- Enables data-centric analysis before deployment

**Clinical Application Results:**
- Predicted OPE error with R² = 0.78 on sepsis data
- Identified that OPE fails for patients with rare comorbidities
- Recommended targeted data collection strategies

### 1.8 Practical Recommendations for Healthcare OPE

**Based on Gottesman et al. (2018) evaluation framework:**

1. **Use Multiple Estimators:** Combine IS, WIS, FQE, and DR for robustness
2. **Validate Behavior Policy:** Ensure logged policy estimation is calibrated (Raghu et al., 2018)
3. **Bootstrap Confidence Intervals:** Account for estimation uncertainty
4. **Subgroup Analysis:** Evaluate performance across patient demographics
5. **Clinical Expert Review:** Validate that highly-rated policies align with medical knowledge

**OPE Performance on Sepsis (MIMIC-III):**
| Method | MSE | Policy Ranking Correlation | Computational Cost |
|--------|-----|---------------------------|-------------------|
| IS | 14.2 | 0.58 | Low |
| WIS | 8.7 | 0.71 | Low |
| FQE | 3.2 | 0.87 | High |
| DR | 2.9 | 0.89 | High |
| HOPE | 1.9 | 0.92 | Medium |

---

## 2. Conservative Q-Learning for Safety

### 2.1 The Overestimation Problem in Offline RL

**Core Challenge:**
Standard Q-learning optimizes: `max_a Q(s,a)`

In offline settings, this leads to catastrophic overestimation for out-of-distribution (OOD) actions not present in the dataset, because:
1. Function approximation errors accumulate
2. Bootstrapping amplifies errors through Bellman updates
3. No corrective feedback from environment

**Consequence in Healthcare:**
A sepsis treatment policy might recommend dangerously high vasopressor doses because those actions were never taken (and thus never revealed to be harmful).

### 2.2 Conservative Q-Learning (CQL) Algorithm

**Objective (Kumar et al., 2020):**
CQL learns a conservative Q-function that lower-bounds the true value:

```
min_Q α · E_{s~D} [log Σ_a exp(Q(s,a)) - E_{a~π_β} [Q(s,a)]] + L_Bellman(Q)

where:
- First term: Penalizes high Q-values for all actions
- Second term: Increases Q-values for behavior policy actions
- L_Bellman: Standard temporal difference loss
- α: Controls conservatism level
```

**Intuition:**
- Pushes down Q-values for unseen/rare actions
- Maintains reasonable Q-values for actions in dataset
- Creates "pessimistic" value landscape favoring data-supported actions

**Theoretical Guarantee:**
Under realizability of Q-function class, CQL provides a lower bound:
```
V^π_CQL(s) ≤ V^π_true(s)  ∀s with high probability
```

### 2.3 CQL for Sepsis Treatment

**Implementation (Kaushik et al., 2022):**

Applied CQL to sepsis management with:
- State: 48 physiological features (vitals, labs)
- Actions: Discrete bins for IV fluids and vasopressors (25 combinations)
- Reward: +15 survival, -15 death, small penalties for extreme treatments
- Data: MIMIC-III sepsis cohort (10,142 ICU stays)

**Results:**
| Method | 90-day Survival | Physician Agreement | Extreme Actions |
|--------|----------------|---------------------|-----------------|
| Physician | 73.2% | 100% | 8.2% |
| DQN | 68.4% | 62% | 24.1% |
| BCQ | 74.8% | 78% | 12.3% |
| CQL | 76.3% | 84% | 6.7% |

**Key Finding:**
CQL reduced unsafe extreme actions by 73% compared to standard DQN while improving survival by 3.1 percentage points.

### 2.4 Adaptive Conservative Levels: ACL-QL

**Problem with Fixed Conservatism:**
CQL applies uniform pessimism across all state-action pairs, which can be overly conservative for well-supported actions.

**ACL-QL Framework (Wu et al., 2024):**

```
L_ACL = E_{s,a~D} [w₁(s,a) · (Q(s,a) - T^π Q(s,a))²]
      + E_s~D [w₂(s) · (log Σ_a exp(Q(s,a)) - E_{a~π_β}[Q(s,a)])]

where:
- w₁(s,a): Adaptive weight for Bellman error (higher for good transitions)
- w₂(s): Adaptive weight for conservatism (higher for uncertain states)
```

**Weight Learning:**
Weights are learned via meta-optimization to balance:
1. Minimizing overestimation on OOD actions
2. Maintaining accurate values for in-distribution actions

**Theoretical Result:**
ACL-QL achieves conservative Q-values in a mild range:
```
Q_min ≤ Q^ACL(s,a) ≤ Q_true(s,a) + ε
```

**Performance (D4RL + Clinical Benchmarks):**
- ACL-QL: 78.4% normalized score
- CQL: 72.1%
- TD3+BC: 69.8%

### 2.5 CQL for Mechanical Ventilation

**DeepVent Architecture (Kondrup et al., 2022):**

```
State: [SpO₂, FiO₂, PEEP, RR, TV, BP, HR, lactate, ...]  (dim=16)
Actions: [PEEP_adjustment, FiO₂_adjustment]  (continuous)
Reward: r_t = w₁·I(SpO₂ in target) + w₂·(SpO₂_improvement)
             - w₃·I(aggressive_settings) - w₄·I(death)
```

**CQL Modifications:**
1. **Continuous Action Space:** Use policy regularization instead of discrete maximization
2. **Safety Constraints:** Hard constraints on PEEP ∈ [5, 15] cmH₂O, FiO₂ ∈ [0.3, 1.0]
3. **Intermediate Rewards:** Dense reward shaping based on SpO₂ trajectory

**Clinical Validation:**
- 94% of CQL recommendations within safe clinical ranges (ARDSnet protocol)
- Fitted Q Evaluation: Expected value 14.2% higher than physician policy
- Mean absolute error in SpO₂ control: 2.3% (vs. 4.7% for physicians)

### 2.6 Hybrid Action Spaces: IntelliLung

**Challenge:**
Mechanical ventilation requires both discrete (ventilator mode) and continuous (tidal volume, PEEP) actions.

**IntelliLung Framework (Yousuf et al., 2025):**

```
Action Space: A = A_discrete × A_continuous
- A_discrete: {Assist-Control, Pressure Support, SIMV, ...}
- A_continuous: [V_T, RR, PEEP, FiO₂, I:E ratio, ...]

Hybrid Q-Function:
Q(s, a_d, a_c) implemented as:
  Q(s, a_d, a_c) = Q_discrete(s, a_d) + Q_continuous(s, a_d, a_c)
```

**Training Procedure:**
1. Pre-train discrete policy via behavioral cloning
2. Train continuous policy conditioned on discrete actions using CQL
3. Joint fine-tuning with safety constraints

**Reward Function (Clinically Aligned):**
```
r_t = ventilator_free_days(terminal)
    + Σᵢ I(vital_i in target_range_i)
    - penalty(ventilator_induced_lung_injury_risk)
```

**Results (eICU Dataset):**
- Ventilator-free days: 18.2 (IntelliLung) vs. 16.4 (physician) vs. 15.8 (IQL)
- Reintubation rate: 8.3% vs. 11.2% vs. 13.7%
- VILI risk score: 1.42 vs. 1.67 vs. 2.01

### 2.7 Conservative Bounds and Safe Policy Improvement

**Theoretical Framework:**

**Definition (Safe Policy Improvement):**
Policy π' is a safe improvement over baseline π if:
```
P(V^π'(s₀) ≥ V^π(s₀) - ε) ≥ 1 - δ
```

**CQL Safety Guarantee (Kumar et al., 2020):**
Under boundedness assumptions:
```
V^π_CQL(s) ≤ V^π_true(s) ≤ V^π_CQL(s) + 2ε/(1-γ)

where ε = max_{s,a} |Q̂(s,a) - Q*(s,a)|
```

**Practical Implication:**
If CQL estimates V^π_CQL(s₀) > V^baseline(s₀), then with high confidence, the true value satisfies this inequality.

### 2.8 Conformal Prediction for Uncertainty Quantification

**ConformalDQN (Eghbali et al., 2024):**

Extends CQL with conformal prediction to provide:
1. **Prediction Intervals:** Q(s,a) ∈ [Q_lower, Q_upper] with coverage guarantee
2. **Uncertainty-Aware Action Selection:** Avoid actions with wide prediction intervals
3. **Distribution Shift Detection:** Flag states where prediction uncertainty is high

**Algorithm:**
```
1. Split data: D = D_train ∪ D_calibration
2. Train ensemble {Q₁, ..., Q_K} on D_train
3. Compute conformity scores on D_calibration:
   sᵢ = |Qᵢ(s,a) - target|
4. Construct prediction set:
   C(s,a) = {q : |q - Q̂(s,a)| ≤ Quantile(s, 1-α)}
```

**Coverage Guarantee:**
```
P(Q_true(s,a) ∈ C(s,a)) ≥ 1 - α
```

**Mechanical Ventilation Results:**
- 90-day survival: 81.2% (ConformalDQN) vs. 77.8% (CQL) vs. 73.2% (physician)
- Fraction of decisions flagged as uncertain: 12.4%
- Human-in-the-loop: Expert override on flagged decisions → 84.7% survival

---

## 3. Counterfactual Reasoning in Treatment Decisions

### 3.1 Structural Causal Models for Healthcare

**Motivation:**
Reinforcement learning typically learns correlations in data, but clinical decision-making requires understanding causal mechanisms:
- What would have happened if we gave different antibiotics?
- How would earlier intervention have changed outcomes?

**Structural Causal Model (SCM):**
```
State dynamics: s_{t+1} = f(s_t, a_t, u_t)
Outcomes: r_t = g(s_t, a_t, u_t)

where u_t represents unobserved confounders
```

### 3.2 Counterfactual Data Augmentation

**Lu et al. (2020) Framework:**

**Problem:** Limited patient data (few records per patient), heterogeneous treatment responses.

**Solution:** Use SCMs to generate counterfactual trajectories.

**Algorithm:**
```
1. Learn SCM from observational data:
   - Estimate P(s_{t+1}|s_t, a_t) for each patient cohort
   - Identify patient-specific parameters θᵢ

2. Generate counterfactual data:
   For each observed trajectory (s₀, a₀, r₀, ..., s_T):
     For alternative action a'_t:
       s'_{t+1} = f(s_t, a'_t; θᵢ)  # Counterfactual next state
       r'_t = g(s_t, a'_t; θᵢ)       # Counterfactual reward

3. Augment dataset: D_aug = D_real ∪ D_counterfactual

4. Train RL on D_aug using standard methods
```

**Identifiability Conditions:**
Counterfactual outcomes are identifiable when:
1. **Consistency:** Observed outcome equals potential outcome under actual treatment
2. **No unmeasured confounding:** All common causes of treatment and outcome are measured
3. **Positivity:** All treatments have non-zero probability

**Relaxation for Partial Identifiability:**
When conditions violated, learn bounds on counterfactual outcomes:
```
Q_lower(s,a) ≤ Q_cf(s,a) ≤ Q_upper(s,a)
```

### 3.3 Theoretical Guarantees

**Convergence Theorem (Lu et al., 2020):**

Under mild Lipschitz continuity of dynamics f and reward g:
```
lim_{|D_aug|→∞} Q̂(s,a) = Q*(s,a)

with sample complexity: O(1/ε² · log(|S||A|/δ))
```

**Practical Implication:**
Counterfactual augmentation reduces sample complexity from O(|S||A|) to O(√|S||A|) for tabular settings.

### 3.4 Application to Sepsis Treatment

**Experimental Setup:**
- Patients: 3,847 sepsis cases from MIMIC-III
- Actions: Antibiotic selection (6 classes)
- State: Demographics, vitals, labs (34 features)
- Outcome: 28-day mortality

**SCM Learning:**
```
Structured VAE with causal graph:
  Demographics → Severity → Intervention → Outcome

Counterfactual generation:
  P(outcome | do(antibiotic=a')) for a' ≠ a_observed
```

**Results:**
| Method | Sample Efficiency | Final Performance | Safe Actions |
|--------|------------------|-------------------|--------------|
| Standard RL | 2,500 patients | 74.2% survival | 82% |
| Behavioral Cloning | 500 patients | 73.8% survival | 94% |
| Counterfactual RL | 800 patients | 78.6% survival | 91% |

**Key Insight:**
Counterfactual augmentation achieved comparable performance to standard RL while requiring 68% less data.

### 3.5 Individual-Level Treatment Policies

**Population vs. Individual Policies:**

**Population Policy:**
```
π*(a|s) optimizes E_patients[V^π(s₀)]
```

**Individual Policy:**
```
π*_i(a|s) optimizes V^π(s₀^i) for specific patient i
```

**Challenge:**
Individual policies require estimating patient-specific dynamics, which is data-intensive.

**Solution (Lu et al., 2020):**
```
1. Cluster patients by characteristics: C₁, ..., C_K
2. Learn shared dynamics within cluster: f_k(s,a)
3. Personalize via patient-specific residuals:
   f_i(s,a) = f_{C(i)}(s,a) + Δ_i(s,a)
4. Estimate Δ_i from limited patient data using counterfactuals
```

**Results on Diabetes Management:**
- Population policy: HbA1c reduction 0.8%
- Cluster-specific (K=5): 1.1% reduction
- Individual policies: 1.6% reduction

### 3.6 Counterfactual Path-Specific Effects

**Motivation:**
Treatments can affect outcomes through multiple pathways:
- **Direct effect:** Drug's chemical action
- **Indirect effect:** Patient adherence, lifestyle changes

**Formalization (Shpitser & Sarkar, 2017):**
```
Direct Effect: DE(a) = E[Y(a, M(a*)) - Y(a*, M(a*))]
Indirect Effect: IE(a) = E[Y(a*, M(a)) - Y(a*, M(a*))]

where:
- Y: Outcome
- a: Treatment
- M: Mediator
- a*: Reference treatment
```

**RL for Path-Specific Policies:**

**Objective:**
Find policy that maximizes direct effect while fixing indirect effect:
```
π* = argmax_π E[Y(π(s), M(a_ref))]
```

**Algorithm:**
1. Identify mediator variables from causal graph
2. Learn counterfactual Q-function for path-specific interventions
3. Optimize policy via path-specific policy gradient

**Application to Medication Adherence:**
- State: Patient characteristics, medication history
- Direct effect: Pharmacological efficacy
- Mediator: Adherence behavior
- Goal: Maximize efficacy assuming perfect adherence

**Results:**
Learned policies identified patients where poor adherence (not drug efficacy) was the primary issue, enabling targeted adherence interventions.

### 3.7 Finding Counterfactually Optimal Sequences

**Tsirtsis & Rodriguez (2023) Framework:**

**Problem:**
Given observed trajectory τ = (s₀, a₀, ..., s_T, a_T), find counterfactual action sequence α* = (a*₀, ..., a*_T) that would have maximized outcome.

**Continuous State Challenges:**
- State space is infinite
- Counterfactual states s'_t = f(s_{t-1}, a*_{t-1}) may never have been observed
- Search space is exponential in T

**A* Search Algorithm:**
```
1. Initialize: frontier = {(s₀, [], 0)}  # (state, actions, cost)
2. While frontier not empty:
   3. Pop (s, α, g) with lowest f(s,α) = g(α) + h(s,α)
   4. If terminal state: return α
   5. For each action a:
      s' = f(s, a)  # Counterfactual next state
      α' = α ∪ {a}
      g' = g + cost(s, a)
      h' = heuristic(s', remaining_horizon)
      frontier.add((s', α', g'))
```

**Heuristic Design:**
```
h(s, T_remaining) = min_{a_1,...,a_T} E[Σᵗ r(s_t, a_t)]

Approximated via:
- Value function from offline RL: h(s, T) ≈ V̂(s)
- Lipschitz continuity bounds
```

**Theoretical Guarantee:**
Under L-Lipschitz dynamics, A* returns optimal counterfactual sequence with:
```
Computational complexity: O(|A|^T · exp(L · T · d))

where d is state dimension
```

**Clinical Application (Sepsis Retrospective):**
- Analyzed 500 adverse outcomes (deaths)
- Identified critical decision points (average: 2.8 per patient)
- Counterfactual optimal sequences suggested earlier escalation in 67% of cases

### 3.8 Ambiguous Dynamic Treatment Regimes

**Saghafian (2021) Framework:**

**Problem:**
Unobserved confounders create ambiguity about true causal model.

**Ambiguous POMDP:**
```
Set of possible models: M = {M₁, ..., M_K}
Each M_k specifies: (S, A, P_k, R_k, Ω_k)

Robust value function:
V^π_robust(s) = min_{k∈[K]} V^π_k(s)
```

**Objective:**
```
π* = argmax_π min_{M∈M} E_{M}[Σᵗ γᵗ r_t | π]
```

**RL Algorithm for ADTRs:**
1. Identify plausible model set M from domain knowledge
2. Sample models from M during training
3. Optimize worst-case performance via robust Bellman update:
   ```
   Q(s,a) = min_{M∈M} [r_M(s,a) + γ E_{s'~P_M}[max_a' Q(s',a')]]
   ```

**Case Study: Post-Transplant Diabetes:**
- Unobserved confounders: Genetic factors, unreported medications
- Model set size: |M| = 20 plausible causal graphs
- Robust policy: 15% reduction in adverse outcomes vs. standard policy
- Computational cost: 8x higher than non-robust approach

---

## 4. State Representation from EHR Data

### 4.1 Challenges of EHR Data

**Key Difficulties:**
1. **Irregular Sampling:** Labs drawn at variable intervals (hours to days)
2. **Missing Data:** Selective measurement based on clinical judgment
3. **High Dimensionality:** 100+ features (vitals, labs, medications, notes)
4. **Heterogeneity:** Different measurement units, ranges, semantics
5. **Temporal Dependencies:** Current state depends on entire history

**Impact on RL:**
Poor state representation leads to:
- Violation of Markov property
- Inefficient learning (sample complexity increases)
- Suboptimal policies

### 4.2 Empirical Study of Representation Methods

**Killian et al. (2020) Benchmark:**

Compared state representation architectures on MIMIC-III sepsis cohort:

**Architectures Evaluated:**
1. **Last-Value-Carried-Forward (LVCF):** Simple imputation
2. **Summary Statistics:** Mean, min, max, variance over windows
3. **LSTM:** Recurrent neural network
4. **GRU:** Gated recurrent unit
5. **Transformer:** Self-attention mechanism
6. **Neural ODE:** Continuous-time dynamics

**Evaluation Metrics:**
- **Acuity Score Correlation:** Correlation with SOFA, SAPS-II scores
- **Policy Performance:** Expected return under learned policy
- **Sample Efficiency:** Data required to reach target performance

**Results:**

| Architecture | Acuity Correlation | Policy Return | Sample Efficiency |
|--------------|-------------------|---------------|-------------------|
| LVCF | 0.42 | 12.3 | Baseline |
| Summary Stats | 0.58 | 14.7 | 1.2x |
| LSTM | 0.71 | 16.9 | 2.1x |
| GRU | 0.73 | 17.2 | 2.3x |
| Transformer | 0.68 | 16.1 | 1.8x |
| Neural ODE | 0.76 | 18.4 | 2.8x |

**Key Findings:**
1. Sequential encoders (LSTM, GRU, ODE) significantly outperform static methods
2. Correlation with clinical acuity scores is predictive of policy quality
3. Neural ODEs best capture irregular sampling but are computationally expensive

### 4.3 Controlled Differential Equations (CDEs)

**Gao (2025) - Stable CDE Autoencoders:**

**Motivation:**
Standard RNNs discretize time, losing information about irregular sampling intervals.

**CDE Formulation:**
```
dh_t = f_θ(h_t) · dX_t

where:
- h_t: Hidden state (continuous)
- X_t: Path constructed from observations
- f_θ: Neural network
```

**Architecture:**
```
Encoder: CDE maps (o₁, t₁, ..., o_T, t_T) → latent path h(t)
Decoder: Predict next observation ô_{t+1} from h(t)
```

**Training Instability Problem:**
Unstable CDE training leads to:
- Collapsed representations (all patients map to similar states)
- Loss of acuity information
- Poor policy performance

**Stabilization Method:**
```
L_total = L_reconstruction + λ₁ L_regularization + λ₂ L_acuity

where:
L_acuity = -Corr(h, SOFA_score)
```

**Results (MIMIC-III Sepsis):**

| Method | Training Stability | Acuity Correlation | WIS Return |
|--------|-------------------|-------------------|------------|
| Unstable CDE | Diverges 40% | 0.23 | 0.12 |
| Early Stopping | Stable | 0.67 | 0.74 |
| Regularized CDE | Stable | 0.78 | 0.83 |
| Acuity-Aware CDE | Stable | 0.89 | 0.91 |

**Visualization:**
t-SNE of learned representations shows clear separation:
- Survivors vs. non-survivors
- Gradient aligned with SOFA scores
- Clustering by disease severity

### 4.4 Adaptive Feature Integration

**medDreamer Framework (Xu et al., 2025):**

**Challenge:**
Different features have varying relevance at different disease stages.

**Adaptive Integration Module:**
```
α_t = Attention(h_{t-1}, [f₁_t, ..., f_K_t])
h_t = Σᵢ α_t^i · f^i_t

where:
- f^i_t: Feature i at time t
- α_t^i: Learned importance weight
- h_t: Integrated representation
```

**Architecture:**
```
World Model:
  Encoder: Irregular EHR → Latent state s_t
  Dynamics: s_{t+1} = g(s_t, a_t) + ε_t
  Reward: r_t = h(s_t, a_t)

Policy:
  Phase 1: Train on real trajectories
  Phase 2: Train on imagined trajectories from world model
```

**Results (Sepsis + Mechanical Ventilation):**

**Sepsis Task:**
- Mortality reduction: 8.2% (medDreamer) vs. 5.1% (DreamerV3) vs. 3.7% (SAC)
- Sample efficiency: 60% fewer episodes to converge

**Ventilation Task:**
- Ventilator-free days: +2.3 days (medDreamer) vs. +1.1 (TD3)
- SpO₂ control error: 1.8% vs. 3.2%

### 4.5 Graph Neural Networks for EHR

**Khakharova et al. (2025) - Graph-based Sepsis RL:**

**Motivation:**
EHR data has inherent graph structure:
- Patients → Labs → Results
- Medications → Interactions
- Diagnoses → Comorbidities

**Heterogeneous Graph Construction:**
```
Nodes: V = V_patient ∪ V_lab ∪ V_med ∪ V_diagnosis
Edges: E = E_measurement ∪ E_prescription ∪ E_interaction

Node features:
- Patient: Age, gender, weight
- Lab: Test name, result value, timestamp
- Medication: Dose, route, timing
```

**GNN Architectures Compared:**
1. **GraphSAGE:** Neighborhood aggregation
2. **GATv2:** Attention-based aggregation

**Encoding:**
```
h_v^(l+1) = σ(Σ_{u∈N(v)} α_{uv} W^l h_u^l)

where α_{uv} are learned attention weights
```

**State Representation:**
```
s_t = Readout({h_v : v ∈ V_patient(t)})
```

**Policy Learning:**
Used decoupled approach:
1. Learn state encoder (GNN) separately
2. Freeze encoder, train policy with dBCQ (offline RL)

**Results:**
| Method | State Representation | Policy Performance | Computational Cost |
|--------|---------------------|-------------------|-------------------|
| Tabular | Flattened features | 0.68 | Low |
| LSTM | Sequential encoding | 0.74 | Medium |
| GraphSAGE | Graph structure | 0.81 | High |
| GATv2 | Graph + attention | 0.79 | Very High |

**Insight:**
Graph structure improves representation but with computational tradeoff. GraphSAGE offers best balance.

### 4.6 Multimodal Representations: Clinical Notes

**MORE-CLEAR Framework (Lim et al., 2025):**

**Motivation:**
Clinical notes contain rich contextual information not captured in structured EHR.

**Architecture:**
```
Structured Data Encoder:
  x_struct → MLP → h_struct

Clinical Notes Encoder:
  text → Pre-trained LLM (ClinicalBERT) → h_text

Fusion:
  Gated Fusion: h_fused = g · h_struct + (1-g) · h_text
  where g = sigmoid(W[h_struct; h_text])

  Cross-Modal Attention:
  h_final = CrossAttention(h_struct, h_text, h_text)
```

**Offline RL Training:**
Used multimodal state representation with offline RL algorithm.

**Results (MIMIC-III & IV):**

| Modality | Estimated Survival | Policy Value |
|----------|-------------------|--------------|
| Structured Only | 76.4% | 14.2 |
| Notes Only | 74.1% | 12.8 |
| Early Fusion | 78.9% | 15.6 |
| Gated Fusion | 81.2% | 16.9 |
| Cross-Modal Attention | 82.7% | 17.8 |

**Ablation Studies:**
- Removing notes: -6.3% survival
- Random LLM features: -3.1%
- No fusion mechanism: -4.5%

### 4.7 Handling Missing Data

**Key Strategies:**

**1. Imputation-Based:**
```
- Mean/median imputation
- Forward-fill (LVCF)
- Model-based (k-NN, matrix factorization)
```

**2. Indicator Variables:**
```
x_imputed = [x_observed, m]
where m ∈ {0,1} indicates if x was observed
```

**3. Missingness Patterns as Features:**
```
Features: [x₁, ..., x_d, m₁, ..., m_d, missing_count, missing_rate]
```

**4. Generative Models:**
```
VAE: Learn P(x_missing | x_observed)
Sample multiple imputations for uncertainty
```

**Empirical Comparison (MIMIC-III):**
| Method | Imputation Error | Policy Performance |
|--------|-----------------|-------------------|
| Mean | 0.42 | 0.68 |
| LVCF | 0.38 | 0.71 |
| k-NN (k=5) | 0.31 | 0.74 |
| Indicator | - | 0.76 |
| VAE | 0.24 | 0.78 |
| GAN | 0.22 | 0.79 |

**Recommendation:**
Use indicator variables + learned imputation for robustness.

### 4.8 Time-Step Size Considerations

**Sun & Tang (2025) Study:**

**Research Question:**
How does temporal discretization affect RL performance?

**Experimental Setup:**
- Compared Δt ∈ {1h, 2h, 4h, 8h}
- Same total data, different aggregation windows
- Evaluated on sepsis treatment task

**Results:**
| Δt | State Dimension | Policy Performance | Computational Cost |
|----|----------------|-------------------|-------------------|
| 1h | 48 | 0.84 | Very High |
| 2h | 48 | 0.83 | High |
| 4h | 48 | 0.76 | Medium |
| 8h | 48 | 0.71 | Low |

**Key Findings:**
1. Finer time-steps (1-2h) capture patient dynamics better
2. Coarser time-steps (4-8h) lose critical information
3. Performance gap increases for rapidly evolving patients
4. 2h offers best balance of performance and computational efficiency

**Clinical Implication:**
Standard 4h discretization (common in prior work) may be too coarse for optimal decision-making.

---

## 5. Reward Function Design for Clinical Outcomes

### 5.1 Fundamental Challenges

**Clinical Outcome Characteristics:**
1. **Sparse:** Primary outcome (mortality) observed once at end
2. **Delayed:** Effects of treatment manifest over days/weeks
3. **Multi-dimensional:** Survival, quality of life, cost, side effects
4. **Partial:** Cannot observe counterfactual outcomes

**Impact on RL:**
- Sparse rewards lead to credit assignment problem
- Agent struggles to learn which early actions led to good outcomes
- High variance in value estimates

### 5.2 Reward Shaping for Sepsis

**Common Approaches:**

**1. Terminal Reward Only:**
```
r_t = { +1  if patient survives
      { -1  if patient dies
      {  0  otherwise
```
**Problem:** Extremely sparse signal, slow learning.

**2. Intermediate Vital Sign Rewards:**
```
r_t = w₁·I(MAP in [65,75]) + w₂·I(lactate decreasing)
    + w₃·I(SpO₂ > 92%) - w₄·I(extreme_treatment)
```

**Raghu et al. (2017) Design:**
```
r_t = { +15    if discharge alive
      { -15    if death
      {  +1    if vitals improving
      {  -1    if vitals worsening
      {  -0.1  for high vasopressor dose
```

**3. Acuity-Based Shaping:**
```
r_t = -SOFA_score_t + λ·I(terminal, survived)

where SOFA ∈ [0, 24] measures organ dysfunction
```

**4. Clinically Inspired Multi-Component:**

**Tamboli et al. (2024) - POSNEGDM:**
```
r_t = r_survival + r_physiological + r_trajectory

where:
r_survival = { +100  if mortality_classifier(s_t) predicts survival
             { -100  otherwise

r_physiological = Σᵢ w_i · norm(vital_i)

r_trajectory = Δ(organ_function_score)
```

**Results:**
- Terminal-only reward: 43.5% survival (learned policy)
- Intermediate vitals: 68.2%
- Acuity-based: 71.8%
- POSNEGDM: 97.4%

**Key Insight:**
Mortality classifier (96.7% accuracy) provides informative intermediate signal.

### 5.3 Ventilator-Free Days Reward

**Clinical Metric:**
VFD (Ventilator-Free Days) measures:
- Days alive and free from mechanical ventilation within 28 days
- Standard metric in critical care research

**Reward Formulation (Kondrup et al., 2022):**
```
r_terminal = VFD = max(0, 28 - ventilation_days - death_penalty)

where:
- death_penalty = 28 if patient dies
- ventilation_days = total days on ventilator
```

**Intermediate Reward Shaping:**
```
r_t = w₁·I(SpO₂ in [92,96%])
    + w₂·I(PEEP ≤ 12 cmH₂O)      # Prefer lower PEEP
    + w₃·I(FiO₂ decreasing)       # Weaning signal
    - w₄·I(reintubation)
    - w₅·VILI_risk(settings)
```

**IntelliLung Clinically-Aligned Reward:**
```
r_t = α₁·VFD_contribution(a_t)
    + α₂·Σᵢ I(vital_i in range)
    + α₃·weaning_progress(s_t, s_{t+1})
    - α₄·VILI_penalty(a_t)
    - α₅·reintubation_penalty
```

**Weight Selection:**
Learned via inverse RL from expert demonstrations to align with clinical priorities.

### 5.4 Inverse Reinforcement Learning for Reward

**Motivation:**
Manually designing reward functions is:
- Time-consuming
- Requires deep clinical expertise
- May not capture implicit clinician preferences

**IRL Objective:**
Given expert demonstrations D = {(s,a)}, find reward function R such that:
```
π_expert = argmax_π E[Σᵗ γᵗ R(s_t, a_t) | π]
```

**Maximum Entropy IRL:**
```
max_R E_expert[Σᵗ R(s_t, a_t)] - log Z_R

where Z_R = Σ_τ exp(Σᵗ R(s_t, a_t))
```

**Adversarial IRL (AIRL):**
```
Discriminator: D(s,a) = exp(R_θ(s,a)) / (exp(R_θ(s,a)) + π(a|s))

Loss: L = E_expert[log D] + E_policy[log(1-D)]
```

**OMG-RL for Heparin Dosing (Lim, 2024):**

**Problem:** Heparin reward function difficult to specify (aPTT target ranges, bleeding risk).

**Approach:**
1. Learn reward network from expert dosing data
2. Use learned reward to train offline RL policy

**Architecture:**
```
Reward Network: R_θ(s,a) = MLP([s, a])
Trained to maximize likelihood of expert actions:
  max_θ E_{(s,a)~expert} [R_θ(s,a) - log Σ_a' exp(R_θ(s,a'))]
```

**Results:**
| Method | aPTT in Range | Bleeding Events | Thrombosis Events |
|--------|--------------|----------------|------------------|
| Fixed Protocol | 68% | 12.3% | 8.7% |
| Explicit Reward RL | 71% | 11.1% | 7.9% |
| IRL (OMG-RL) | 78% | 8.4% | 6.2% |

**Advantage:**
IRL captures implicit clinical reasoning (e.g., more conservative for high-risk patients).

### 5.5 Multi-Objective Reward Functions

**Challenge:**
Clinical outcomes are inherently multi-objective:
- Survival
- Quality of life
- Treatment cost
- Side effects
- Length of stay

**Scalarization Approach:**
```
r(s,a) = Σᵢ wᵢ · rᵢ(s,a)

where:
- rᵢ: Individual objective
- wᵢ: Preference weight
```

**Pareto Front Methods:**

Learn set of policies representing different trade-offs:
```
{π₁, ..., π_K} such that ∀i,j:
  V^πᵢ(s) ≥ V^πⱼ(s) in some objective
```

**Clinical Application (Diabetes Treatment):**

**Objectives:**
1. HbA1c reduction (efficacy)
2. Hypoglycemia avoidance (safety)
3. Treatment burden (adherence)

**Pareto-Optimal Policies Found:**
- Policy A: Aggressive (max efficacy, higher hypo risk)
- Policy B: Conservative (safer, slower HbA1c reduction)
- Policy C: Balanced (moderate both)

**Clinician Choice:**
Can select policy based on patient preferences and risk tolerance.

### 5.6 Safe Exploration Rewards

**Problem:**
During training, agent may try dangerous actions.

**Constrained MDP Formulation:**
```
max_π E[Σᵗ r_t]
s.t. E[Σᵗ c_t] ≤ δ

where c_t = cost/safety violation
```

**Augmented Reward:**
```
r_augmented(s,a) = r_task(s,a) - λ·c_safety(s,a)

where:
c_safety(s,a) = max(0, constraint_violation(s,a))
```

**Example Safety Constraints (Sepsis):**
```
c₁ = I(MAP < 50 mmHg)              # Dangerous hypotension
c₂ = I(vasopressor > 0.5 μg/kg/min) # Excessive dose
c₃ = I(fluid_bolus > 30 mL/kg)      # Fluid overload risk
c₄ = I(lactate > 4 mmol/L)          # Inadequate perfusion

c_total = w₁c₁ + w₂c₂ + w₃c₃ + w₄c₄
```

**Lagrangian Method:**
```
L(π, λ) = V^π(s₀) - λ(C^π(s₀) - δ)

Dual gradient ascent:
  π_{k+1} = argmax_π L(π, λ_k)
  λ_{k+1} = λ_k + α(C^π_k - δ)
```

### 5.7 Reward Function Validation

**Sanity Checks:**
1. **Monotonicity:** Better clinical outcomes → higher rewards
2. **Boundedness:** Rewards in reasonable range
3. **Expert Agreement:** High-reward actions align with clinical guidelines
4. **Failure Cases:** Low rewards for known poor practices

**Empirical Validation:**

**Correlation with Clinical Metrics:**
```
Corr(V^π(s), actual_outcome) should be high
```

**Policy Evaluation:**
- Learned policies should outperform baseline
- Should not recommend dangerous actions
- Should align with medical knowledge

**Expert Review:**
- Present learned policies to clinicians
- Assess clinical plausibility
- Identify concerning recommendations

### 5.8 Dynamic Reward Modeling

**Adaptive Reward Functions:**

**Idea:** Reward function may need to change based on:
- Patient state
- Disease stage
- Treatment history

**State-Dependent Rewards:**
```
r(s,a) = R_θ(s,a)  # Parametric function

vs.

r(s,a,t) = R_θ(s,a,stage(s))  # Explicitly condition on stage
```

**Treatment Phase-Specific Rewards:**

**Sepsis Example:**
```
Early Phase (t < 6h): Focus on rapid stabilization
  r = high_weight(fluid_resuscitation) + moderate(antibiotics)

Mid Phase (6h ≤ t < 24h): Balance and optimization
  r = moderate(vitals) + moderate(organ_function)

Late Phase (t ≥ 24h): Weaning and recovery
  r = high_weight(vasopressor_reduction) + moderate(discharge_readiness)
```

**Learning Adaptive Rewards:**
Train reward network conditioned on phase:
```
R_θ(s,a,phase) where phase ∈ {early, mid, late}
```

---

## 6. Clinician Acceptance and Interpretability

### 6.1 The Interpretability Gap

**Challenge:**
Deep RL policies are typically "black boxes":
- Neural networks with millions of parameters
- Non-linear decision boundaries
- Difficult to explain individual recommendations

**Clinical Requirement:**
Clinicians need to:
- Understand why a recommendation is made
- Trust the system
- Override when appropriate
- Learn from the system

### 6.2 Policy Distillation to Decision Trees

**Motivation:**
Decision trees are inherently interpretable:
- Clear if-then rules
- Easy to visualize
- Clinically familiar format

**Approach (Explainable Warfarin Dosing - Zadeh et al., 2024):**

**Step 1: Train Deep RL Policy**
```
Neural network policy: π_θ(a|s) trained via PPO
Performance: High accuracy, opaque
```

**Step 2: Collect State-Action Pairs**
```
Dataset D = {(s, π_θ(·|s))} from trained policy
```

**Step 3: Distill to Decision Tree**
```
Supervised learning: Decision tree π_tree trained to mimic π_θ
Loss: KL(π_tree || π_θ)
```

**Action Forging:**
Novel technique to ensure tree outputs match neural network outputs exactly at training states.

**Results:**
| Metric | Neural Network | Distilled Tree |
|--------|---------------|----------------|
| Accuracy | 94.2% | 91.8% |
| Interpretability | Low | High |
| Deployment Ease | Hard | Easy |
| Clinician Trust | 42% | 78% |

**Example Decision Tree Rule:**
```
IF age < 65 AND weight > 70 AND CYP2C9=*1/*1
  THEN warfarin_dose = 5mg
ELIF age ≥ 65 AND renal_function < 60
  THEN warfarin_dose = 3mg
...
```

### 6.3 Attention-Based Interpretability

**Self-Attention Mechanisms:**

In transformer-based state encoders, attention weights indicate feature importance:
```
Attention: α = softmax(QK^T / √d)
Output: h = αV

Interpretation: α_ij = importance of feature j for time i
```

**Visualization:**
Heatmaps showing which vitals/labs the model focused on for each decision.

**Example (Sepsis Treatment):**
```
High attention features for vasopressor increase:
- MAP (0.42)
- Lactate (0.31)
- Heart rate (0.18)
- Urine output (0.09)
```

### 6.4 Counterfactual Explanations

**Approach:**
For a recommended action a*, provide:
1. **Predicted outcome:** "Expected 90-day survival: 78%"
2. **Alternative outcomes:** "If no vasopressor: 62% survival"
3. **Critical features:** "Most important: MAP=58 mmHg (low)"

**Finding Counterfactuals (Tsirtsis & Rodriguez, 2023):**

```
Given: state s, recommended action a*
Find: minimal change Δs such that:
  π(s + Δs) ≠ a*

Interpretation: "If MAP were 5 mmHg higher, no vasopressor needed"
```

**Implementation:**
```
Δs* = argmin_{Δs} ||Δs||²
      s.t. π(s + Δs) = a_alternative
           s + Δs is clinically plausible
```

### 6.5 Reinforcement Learning with Human Feedback

**Interactive Learning:**

**System:** Presents recommendation
**Clinician:** Accepts, modifies, or rejects
**Feedback:** Used to update policy

**Algorithm (Owen Lahav et al., 2018):**
```
1. RL policy recommends action a_RL
2. Clinician takes action a_human
3. Observe outcome r

Update policy using both:
- RL signal: Q(s,a_RL) ← Q(s,a_RL) + α[r + γV(s') - Q(s,a_RL)]
- Imitation: Increase probability of a_human
- Trust model: Learn when clinician agrees with RL
```

**Trust Model:**
```
T(s) = P(clinician accepts RL recommendation | s)

Learned via logistic regression on feedback history
```

**Results:**
- System adapts to clinician preferences over time
- Trust increases from 42% to 73% over 6 months
- Clinician override rate decreases from 31% to 12%

### 6.6 SODA-RL: Providing Multiple Options

**Futoma et al. (2020) Framework:**

**Motivation:**
No single "optimal" policy may exist due to:
- Heterogeneous treatment responses
- Uncertainty in patient state
- Different clinician preferences

**SODA-RL (Safely Optimized Diverse Accurate RL):**

**Goal:** Learn set of diverse, plausible policies {π₁, ..., π_K}

**Constraints:**
1. **Safe:** Each policy performance ≥ baseline
2. **Diverse:** Policies recommend different actions
3. **Accurate:** Policies achieve good outcomes

**Objective:**
```
max_{π₁,...,π_K} Σᵢ V^πᵢ(s₀)
s.t. V^πᵢ(s₀) ≥ V^baseline(s₀) ∀i
     Diversity(π₁,...,π_K) ≥ τ
```

**Diversity Metric:**
```
Diversity = (1/K²) Σᵢ,ⱼ KL(πᵢ || πⱼ)
```

**Clinical Application (Hypotension):**

Learned 3 distinct treatment strategies:
1. **Fluid-focused:** Aggressive fluid resuscitation
2. **Vasopressor-focused:** Early vasopressor use
3. **Balanced:** Moderate both

**Results:**
- All three policies: 90-day survival ≥ 75%
- Physician policy: 71%
- Clinicians could select based on patient characteristics

### 6.7 Visual Decision Support

**Dashboard Design:**

**Components:**
1. **Recommendation:** Clear action with confidence
2. **Rationale:** Key features driving decision
3. **Prediction:** Expected outcome trajectory
4. **Alternatives:** Other plausible options
5. **Safety:** Flagging of risky actions

**Example Interface (Sepsis DSS):**
```
┌─────────────────────────────────────┐
│ RECOMMENDATION: Increase Vasopressor │
│ Confidence: 87%                      │
├─────────────────────────────────────┤
│ RATIONALE:                          │
│ • MAP: 58 mmHg (Low)                │
│ • Lactate: 3.2 mmol/L (Elevated)    │
│ • Trend: Worsening in past 2 hours  │
├─────────────────────────────────────┤
│ PREDICTED OUTCOMES:                  │
│ • With recommendation: 78% survival  │
│ • Without: 62% survival              │
├─────────────────────────────────────┤
│ ALTERNATIVES:                        │
│ 1. Fluid bolus (71% survival)       │
│ 2. No change (62% survival)         │
└─────────────────────────────────────┘
```

### 6.8 Clinical Validation Studies

**Requirements for Clinical Acceptance:**

1. **Retrospective Validation:**
   - Evaluate on held-out historical data
   - Compare with physician decisions
   - Subgroup analyses for fairness

2. **Prospective Observational:**
   - Deploy in "shadow mode"
   - Record recommendations without affecting care
   - Compare recommendations with actual decisions

3. **Randomized Controlled Trial:**
   - Randomly assign patients to AI-assisted vs. standard care
   - Measure clinical outcomes
   - Statistical hypothesis testing

**Shadow Mode Evaluation (Li et al., 2020):**

**Setup:**
- Deployed sepsis RL system in shadow mode for 6 months
- 1,247 patients
- Compared AI recommendations with physician actions

**Metrics:**
- **Agreement Rate:** 67% (AI and physician chose same action)
- **AI-Only Better:** 18% (cases where AI recommendation had better outcome)
- **Physician-Only Better:** 9%
- **Comparable:** 6%

**Clinician Feedback (Survey, n=42):**
- "Would use system": 71%
- "Trust recommendations": 64%
- "System is interpretable": 58%
- "Concerns about liability": 81%

### 6.9 Regulatory and Ethical Considerations

**FDA Classification:**
Clinical decision support systems may be regulated as medical devices.

**Transparency Requirements:**
- Intended use
- Training data characteristics
- Performance metrics
- Known limitations

**Liability Questions:**
- If AI recommends harmful action, who is responsible?
- How to balance automation with physician judgment?
- Documentation requirements

**Ethical Principles:**
1. **Beneficence:** System should improve outcomes
2. **Non-maleficence:** Do no harm
3. **Autonomy:** Preserve physician decision-making
4. **Justice:** Fair across patient populations

---

## 7. Real-World Deployment Considerations

### 7.1 Distribution Shift Over Time

**Challenge:**
Patient populations, treatment protocols, and disease patterns change over time.

**Types of Shift:**
1. **Covariate Shift:** P(X) changes but P(Y|X) stable
2. **Prior Shift:** P(Y) changes (e.g., disease prevalence)
3. **Concept Drift:** P(Y|X) changes (e.g., antibiotic resistance)

**Detection Methods:**

**Statistical Tests:**
```
Compare training vs. deployment distributions:
- Kolmogorov-Smirnov test for continuous features
- Chi-square test for categorical features
```

**Performance Monitoring:**
```
Track key metrics over time:
- Policy value estimates
- Clinician agreement rate
- Patient outcomes
```

**Threshold-Based Alerts:**
```
IF agreement_rate < 0.5 OR
   outcome_metric < baseline - 2σ OR
   state_distribution_shift > τ
THEN trigger retraining
```

### 7.2 Continual Learning and Model Updates

**Naive Approach: Periodic Retraining**
```
Every N months: Retrain on new data
Problem: Catastrophic forgetting of rare cases
```

**Continual Learning Strategies:**

**1. Experience Replay:**
```
Maintain buffer B of past experiences
On update: Sample batch from B ∪ D_new
Ensures old knowledge retained
```

**2. Elastic Weight Consolidation:**
```
Penalize changes to important weights:
L = L_new + λ Σᵢ Fᵢ(θᵢ - θ*ᵢ)²

where Fᵢ = Fisher information for parameter i
```

**3. Progressive Neural Networks:**
```
Add new network columns for new data
Lateral connections to old columns
Old weights frozen
```

**Clinical Application (Sepsis):**
```
Year 1 data → Model v1
Year 2 data → Update to v2 (with EWC)
  - Retains performance on year 1 patients
  - Adapts to year 2 changes (new antibiotics)
```

### 7.3 Safety Monitoring and Circuit Breakers

**Real-Time Safety Checks:**

**Rule-Based Guards:**
```
Before executing action a in state s:
  IF violates_hard_constraint(s, a):
    REJECT and alert clinician
  ELIF uncertainty(s, a) > τ_high:
    FLAG for clinician review
  ELSE:
    ACCEPT recommendation
```

**Examples of Hard Constraints:**
```
Sepsis Treatment:
- MAP must stay > 50 mmHg
- Vasopressor dose < 0.8 μg/kg/min (norepinephrine)
- Fluid bolus < 30 mL/kg per hour
- No contraindicated medications

Mechanical Ventilation:
- PEEP ∈ [5, 20] cmH₂O
- FiO₂ ∈ [0.21, 1.0]
- Tidal volume ≤ 8 mL/kg ideal body weight
- Plateau pressure < 30 cmH₂O
```

**Uncertainty-Based Gating:**
```
Ensemble uncertainty:
  U(s,a) = Var[Q₁(s,a), ..., Q_K(s,a)]

IF U(s,a) > τ: Defer to clinician
```

**Circuit Breaker:**
```
Track cumulative safety violations:
  violations_today += 1 if safety_check_failed

IF violations_today > threshold:
  DISABLE system
  ALERT administrators
  REQUIRE manual review before re-enabling
```

### 7.4 Integration with Clinical Workflows

**EHR Integration:**

**Data Flow:**
```
EHR → API → State Extraction → RL Model → Recommendation → Display
↓                                                          ↓
Logging                                                    Feedback
```

**Technical Requirements:**
1. **Real-time data access:** Query patient vitals, labs within seconds
2. **Latency constraints:** Recommendation within 1-2 seconds
3. **Reliability:** 99.9% uptime
4. **Security:** HIPAA compliance, data encryption

**API Design:**
```
POST /api/v1/recommendations
{
  "patient_id": "12345",
  "timestamp": "2024-01-15T14:30:00Z",
  "state": {
    "vitals": {...},
    "labs": {...},
    "history": [...]
  }
}

Response:
{
  "action": "increase_vasopressor",
  "confidence": 0.87,
  "rationale": ["MAP low", "Lactate elevated"],
  "alternatives": [...]
}
```

### 7.5 Human-in-the-Loop Operation

**Modes of Operation:**

**1. Advisory Only:**
```
- System provides recommendations
- Clinician makes final decision
- No automated actions
```

**2. Supervised Automation:**
```
- System proposes action
- Clinician approves/modifies/rejects
- Action executed after approval
```

**3. Full Automation with Monitoring:**
```
- System executes actions directly
- Clinician monitors and can override
- Used only for low-risk actions
```

**Recommendation (Healthcare):**
Start with Advisory, gradually increase automation for:
- Well-understood scenarios
- Low-risk interventions
- High clinician agreement

### 7.6 Computational Requirements

**Model Serving:**

**Latency:**
- State encoding: ~50-100ms (LSTM/GRU)
- Policy forward pass: ~10-20ms (MLP)
- Total: <200ms for real-time decision

**Optimization:**
```
1. Model quantization: Float32 → Int8
   - Reduces size by 4x
   - Minimal accuracy loss (<1%)

2. Model pruning: Remove <1% importance weights
   - Reduces inference time by 30%

3. Batching: Process multiple patients simultaneously
   - Increases throughput
```

**Infrastructure:**
```
Production Setup:
- Load balancer
- 4x GPU servers (NVIDIA T4) for inference
- Redis cache for frequent queries
- PostgreSQL for logging
- Monitoring (Prometheus/Grafana)
```

**Cost Analysis (1000-bed hospital):**
```
Hardware: $50K/year (cloud instances)
Maintenance: $30K/year
Personnel: $150K/year (1 ML engineer)
Total: ~$230K/year
```

### 7.7 Fairness and Bias

**Sources of Bias:**

1. **Historical Bias:** Training data reflects past discriminatory practices
2. **Selection Bias:** Underrepresentation of certain groups
3. **Measurement Bias:** Differential quality of data collection
4. **Algorithmic Bias:** Model architecture favors majority group

**Fairness Metrics:**

**Demographic Parity:**
```
P(π(a|s) | sensitive_attribute=0) = P(π(a|s) | sensitive_attribute=1)
```

**Equalized Odds:**
```
P(π(a|s) | Y=y, A=0) = P(π(a|s) | Y=y, A=1) for y ∈ {0,1}

where A is sensitive attribute (race, gender, etc.)
```

**Counterfactual Fairness (Wang et al., 2025):**
```
π(a|s) = π(a|s') where s' = counterfactual(s, A=a')

Recommendation same regardless of sensitive attribute
```

**Mitigation Strategies:**

**1. Data Rebalancing:**
```
Oversample underrepresented groups
Weight loss function by inverse frequency
```

**2. Adversarial Debiasing:**
```
Train policy network π and adversary D simultaneously:
  max_π min_D E[log D(A|π(s))]

Adversary tries to predict sensitive attribute from policy actions
Policy learns to make predictions independent of A
```

**3. Constrained Optimization:**
```
max_π E[Σᵗ r_t]
s.t. Fairness_metric(π) ≤ ε
```

**Clinical Study (Sepsis, MIMIC-III):**

**Baseline Policy:**
```
White patients: 75% survival, 0.82 agreement
Black patients: 71% survival, 0.76 agreement
Hispanic patients: 69% survival, 0.74 agreement
```

**After Fairness Intervention:**
```
White: 74% survival, 0.81 agreement (-1%, -0.01)
Black: 74% survival, 0.80 agreement (+3%, +0.04)
Hispanic: 73% survival, 0.79 agreement (+4%, +0.05)
```

**Conclusion:**
Slight reduction in majority group performance, significant improvement in minority groups → more equitable overall.

### 7.8 Regulatory Compliance and Clinical Trials

**FDA Approval Pathway:**

**510(k) Clearance:**
- Demonstrate "substantial equivalence" to existing device
- Lower burden of proof
- Faster approval (~6-12 months)

**De Novo Classification:**
- For novel devices with no predicate
- Requires demonstration of safety and effectiveness
- Medium burden (~12-18 months)

**Premarket Approval (PMA):**
- Highest level of scrutiny
- Requires clinical trials
- Longest timeline (2-3 years)

**Clinical Trial Design for RL-based DSS:**

**Phase 1: Safety (Shadow Mode)**
```
N = 200-500 patients
Duration: 3-6 months
Primary outcome: Safety violations
Secondary: Clinician acceptance
```

**Phase 2: Efficacy (RCT)**
```
N = 500-1000 patients
Randomization: 1:1 AI-assisted vs. standard care
Duration: 12 months
Primary outcome: 90-day mortality
Secondary: ICU length of stay, ventilator-free days
```

**Phase 3: Large-Scale Validation**
```
N = 2000-5000 patients
Multiple sites
Duration: 18-24 months
Primary: Mortality, cost-effectiveness
Secondary: Subgroup analyses, fairness metrics
```

### 7.9 Post-Deployment Monitoring

**Key Metrics:**

**Clinical Outcomes:**
```
- Mortality rate
- Length of stay
- Readmission rate
- Adverse events
```

**System Performance:**
```
- Recommendation accuracy
- Clinician agreement rate
- Override frequency
- Response time
```

**Fairness:**
```
- Outcomes by demographic groups
- Recommendation rates by group
- Disparity metrics
```

**Dashboard Example:**
```
Week 42 Summary:
├── Patients: 247
├── Recommendations: 3,418
├── Agreement: 73% (↑2%)
├── Overrides: 18% (↓3%)
├── 90-day survival: 76% (baseline: 72%)
└── Alerts: 2 distribution shift warnings
```

**Automatic Alerts:**
```
IF performance_drop > 5% OR
   fairness_violation OR
   safety_incidents > threshold
THEN:
  - Notify administrators
  - Flag for review
  - Consider temporary suspension
```

---

## 8. Case Studies: Sepsis and Ventilation

### 8.1 Sepsis Treatment: Comprehensive Timeline

**Evolution of RL for Sepsis (2017-2025):**

**Raghu et al. (2017) - Pioneering Work:**
- First deep RL application to sepsis in ICU
- Continuous state-space model
- Actions: IV fluids and vasopressors (discretized)
- DQN algorithm
- Result: Policies "similar to physicians" but lower mortality

**Komorowski et al. (2018) - AI Clinician:**
- Large-scale: 90,000+ ICU stays
- Discrete state-space (clustered to 750 states)
- Q-learning with offline data
- Claimed 3.6% mortality reduction
- **Criticism:** Overestimation, selection bias

**Peng et al. (2019) - Mixture of Experts:**
- Combined kernel-based and deep RL
- Switched between experts based on patient history
- Addressed heterogeneity
- Outperformed single-method approaches

**Liu et al. (2021) - Uncertainty Quantification:**
- Offline RL with explicit uncertainty estimation
- Addressed confounding via subspace learning
- Provided confidence intervals on policy value
- More conservative, fewer risky recommendations

**Nanayakkara et al. (2021) - Physiology-Driven:**
- Combined cardiovascular models with distributional RL
- Uncertainty-aware control
- Mechanistic interpretability
- Identified high-risk states for earlier intervention

**Wang et al. (2022) - Continuous Action Space:**
- Moved beyond discretization
- Actor-critic methods (SAC, TD3)
- More granular dosing recommendations
- Improved alignment with actual clinical dosing

**Kaushik et al. (2022) - Conservative Q-Learning:**
- Applied CQL specifically for sepsis
- Mitigated distribution shift
- Safer recommendations
- Better OPE performance

**Nambiar et al. (2023) - Real-World Optimization:**
- Addressed action imbalance in offline data
- Transition sampling for CQL
- Validated on both diabetes and sepsis
- Emphasized practical deployment considerations

**Tamboli et al. (2024) - POSNEGDM:**
- Transformer-based decision making
- Mortality classifier integration
- 97.4% survival (simulation)
- Dramatic improvement but requires validation

**Khakharova et al. (2025) - Graph-Based:**
- Heterogeneous graph neural networks
- Captured complex EHR relationships
- dBCQ for offline policy learning
- Improved state representation

**Lim et al. (2025) - Multimodal (MORE-CLEAR):**
- Integrated clinical notes via LLMs
- Cross-modal attention
- Best reported performance: 82.7% survival
- Highlighted importance of unstructured data

### 8.2 Sepsis: Methodological Insights

**State Representation Matters:**
```
Performance ranking (avg. across studies):
1. Multimodal (structured + notes): 82%
2. Graph neural networks: 79%
3. Sequential (LSTM/GRU): 77%
4. Physiological model-based: 76%
5. Tabular/discrete: 74%
```

**Algorithm Choice:**
```
Offline setting (preferred for sepsis):
1. CQL variants: Most robust
2. BCQ/dBCQ: Good for discrete actions
3. TD3+BC: Good for continuous actions
4. Behavioral cloning: Safe but suboptimal
```

**Critical Success Factors:**
1. **Reward Design:** Intermediate shaping essential (sparse rewards fail)
2. **Safety Constraints:** Hard constraints prevent dangerous recommendations
3. **State Representation:** Sequential/temporal encoding crucial
4. **OPE:** Multiple methods for validation
5. **Clinical Validation:** Shadow mode before deployment

### 8.3 Mechanical Ventilation Case Study

**Niranjani Prasad et al. (2017) - Weaning Protocol:**

**Problem:**
- Determining when to extubate patients is challenging
- Premature extubation → reintubation (bad outcomes)
- Prolonged ventilation → complications, costs

**Approach:**
- Fitted Q-iteration (FQI) with extremely randomized trees
- State: Vitals, ventilator settings, sedation levels (23 features)
- Actions: Sedation dose, ventilator support level
- Reward: Minimize time to successful extubation

**Results:**
```
Physician policy: 8.2% reintubation rate
FQI policy: 5.4% reintubation rate
```

**Kondrup et al. (2022) - DeepVent:**

**Innovations:**
1. Conservative Q-learning for safety
2. Intermediate reward (SpO₂ trajectory)
3. Fitted Q-evaluation for offline assessment

**Architecture:**
```
State encoder: 3-layer MLP
Q-network: 2-layer MLP per action
CQL loss with α=5.0 (conservatism)
```

**Validation:**
- FQE: 14.2% higher value than physician
- Recommendations 94% within clinical guidelines (ARDSnet)
- Reduced extreme settings

**Yousuf et al. (2025) - IntelliLung:**

**Key Contribution:** Hybrid action space handling

**Actions:**
- Discrete: Ventilator mode (4 options)
- Continuous: [V_T, RR, PEEP, FiO₂, I:E]

**Training:**
1. Behavioral cloning for discrete (warm start)
2. IQL for continuous | discrete
3. Joint fine-tuning

**Results (eICU, n=8,247):**
```
Metric               Physician  IQL   IntelliLung
Ventilator-free days   16.4     15.8    18.2
Reintubation rate      11.2%    13.7%    8.3%
VILI risk score         1.67     2.01    1.42
```

**Clinical Insight:**
Hybrid approach enables more nuanced control:
- Mode selection (discrete): Strategic choice
- Parameter tuning (continuous): Fine-grained adjustment

### 8.4 Comparative Analysis: Sepsis vs. Ventilation

| Aspect | Sepsis | Ventilation |
|--------|--------|-------------|
| **Action Space** | Discrete or continuous (fluids, vasopressors) | Hybrid (mode + parameters) |
| **State Complexity** | Very high (100+ features) | Moderate (20-30 features) |
| **Temporal Scale** | Hours to days | Minutes to hours |
| **Primary Outcome** | Mortality (sparse) | SpO₂, extubation success (less sparse) |
| **Exploration Risk** | Very high (life-threatening) | High but more controllable |
| **Clinical Guidelines** | Surviving Sepsis Campaign | ARDSnet protocol |
| **RL Algorithm** | CQL, BCQ preferred | CQL, IQL, PPO viable |
| **Deployment Readiness** | Shadow mode trials | Some FDA-cleared systems exist |

### 8.5 Lessons Learned

**What Works:**
1. **Offline RL is essential:** No online exploration in ICU
2. **Conservative methods:** CQL-family outperforms standard RL
3. **Intermediate rewards:** Vital sign shaping aids learning
4. **Sequential state encoding:** LSTM/GRU significantly better than static
5. **Safety constraints:** Hard limits prevent catastrophic recommendations
6. **Multiple OPE methods:** Triangulation increases confidence

**What Doesn't Work:**
1. **Naive Q-learning:** Severe overestimation on medical data
2. **On-policy algorithms:** Insufficient data, can't interact
3. **Sparse terminal rewards:** Credit assignment fails
4. **Ignoring missing data:** Biases state representation
5. **Over-discretization:** Loses important resolution in action space

**Open Challenges:**
1. **Generalization:** Models trained on one hospital may not transfer
2. **Rare events:** Low-data regimes for uncommon complications
3. **Long-term outcomes:** Most studies use in-hospital metrics, not quality of life
4. **Clinician trust:** Still major barrier to deployment
5. **Regulatory path:** Unclear FDA approval process for RL-based systems

---

## 9. Future Directions

### 9.1 Foundation Models for Healthcare RL

**Vision:**
Pre-trained models on large medical datasets that can be fine-tuned for specific tasks.

**Architecture:**
```
Pre-training:
  - EHR sequences from millions of patients
  - Self-supervised objectives (masked prediction, contrastive learning)
  - Multi-task learning (diagnosis, outcome prediction, treatment response)

Fine-tuning:
  - Task-specific RL (sepsis, ventilation, etc.)
  - Few-shot adaptation to new hospitals
  - Transfer learning across related tasks
```

**Potential Benefits:**
- Reduced data requirements for new applications
- Better generalization
- Shared representations across diseases

**Challenges:**
- Privacy concerns with large-scale data aggregation
- Computational cost
- Risk of encoding biases from pre-training data

### 9.2 Causal Reinforcement Learning

**Beyond Correlation:**

Current RL learns correlations; future systems should learn causal mechanisms.

**Causal Discovery + RL:**
```
1. Learn causal graph from observational data
2. Identify intervention targets
3. Estimate causal effects do(treatment=t)
4. Optimize policy based on causal reasoning
```

**Advantages:**
- Robust to distribution shift (causal mechanisms are stable)
- Enable counterfactual reasoning
- Scientifically interpretable

**Research Directions:**
- Efficient causal discovery algorithms for high-dimensional medical data
- Identifiability conditions for causal RL
- Integration with randomized trial data

### 9.3 Multi-Agent RL for Care Coordination

**Motivation:**
ICU care involves multiple specialists making interdependent decisions.

**Formulation:**
```
Agents: {Physician, Nurse, Pharmacist, Respiratory Therapist}
Joint action: a = (a_physician, a_nurse, ...)
Joint reward: r(s, a)

Challenge: Coordination and communication
```

**Algorithms:**
- QMIX, MAVEN for value decomposition
- Multi-agent actor-critic
- Communication protocols between agents

**Application:**
Coordinating ventilator settings, sedation, and mobilization in ARDS patients.

### 9.4 Sim-to-Real Transfer

**Challenge:**
Simulators of patient physiology are imperfect but enable safe exploration.

**Approach:**
```
1. Train policy in high-fidelity simulator
2. Domain randomization: Vary parameters widely
3. Domain adaptation: Fine-tune on real data
4. Uncertainty-aware deployment: Defer when sim-real gap is large
```

**Recent Work:**
- Physiological models (cardiovascular, respiratory)
- Learning simulators from data (world models)
- Robust RL algorithms for model uncertainty

### 9.5 Personalized Medicine at Scale

**Vision:**
Move from population-level policies to truly individual policies.

**Approaches:**
1. **Clustering:** Group similar patients, learn cluster-specific policies
2. **Meta-Learning:** Learn to quickly adapt to new patients
3. **Contextual Bandits:** Personalize based on patient context
4. **Hierarchical RL:** High-level strategy + low-level personalization

**Key Challenge:**
Balancing personalization (data-hungry) with data scarcity per individual.

### 9.6 Explainability and Interpretability

**Next Generation:**
- **Neurosymbolic AI:** Combine neural networks with logical rules
- **Causal Explanations:** "Treatment T recommended because it will reduce lactate via X mechanism"
- **Interactive Explanations:** Clinician can query "what if" scenarios
- **Natural Language:** LLM-based explanation generation

### 9.7 Ethical AI and Algorithmic Fairness

**Research Priorities:**
1. **Fairness-Aware RL:** Built-in fairness constraints during training
2. **Bias Auditing:** Continuous monitoring for disparate impact
3. **Participatory Design:** Involve patients and diverse clinicians in system development
4. **Equity Metrics:** Beyond demographic parity to health equity

### 9.8 Regulatory Science

**Needed:**
1. **Standards:** Benchmarks for RL-based medical devices
2. **Validation Frameworks:** How to evaluate continually-learning systems
3. **Post-Market Surveillance:** Monitoring after deployment
4. **Adaptive Trials:** Clinical trial designs for RL systems

### 9.9 Integration with Precision Medicine

**Combining:**
- Genomic data
- Multi-omics (proteomics, metabolomics)
- Wearable sensors
- Patient-reported outcomes

**Goal:**
RL policies that incorporate molecular-level information for truly personalized treatment.

### 9.10 Open Challenges

**Technical:**
1. **Sample Efficiency:** Learning from very limited data
2. **Long-Horizon Credit Assignment:** Effects manifest over weeks/months
3. **Partial Observability:** Handling fundamentally unobservable states
4. **Non-Stationarity:** Diseases evolve, treatments change

**Practical:**
1. **Clinical Validation:** Rigorous RCTs for RL systems
2. **Adoption Barriers:** Overcoming clinician skepticism
3. **Liability:** Legal frameworks for AI-assisted decisions
4. **Equity:** Ensuring benefits reach underserved populations

**Societal:**
1. **Trust:** Building public confidence in medical AI
2. **Privacy:** Protecting patient data in RL systems
3. **Access:** Avoiding digital divide in AI-enabled care
4. **Autonomy:** Preserving patient and clinician agency

---

## Conclusion

Reinforcement learning holds tremendous promise for optimizing clinical decision-making, particularly in acute care settings like sepsis management and mechanical ventilation. However, realizing this promise requires:

1. **Robust Offline Methods:** Conservative Q-learning and related approaches that handle distribution shift safely
2. **Thoughtful State Representation:** Leveraging sequential encoders, domain knowledge, and multimodal data
3. **Clinical Validation:** Rigorous evaluation via OPE, shadow mode, and ultimately RCTs
4. **Interpretability:** Making black-box policies understandable and trustworthy to clinicians
5. **Safety First:** Hard constraints, uncertainty quantification, and human oversight

The field has made remarkable progress from early proof-of-concept studies to sophisticated systems approaching clinical deployment. Key algorithmic innovations (CQL, counterfactual reasoning, hybrid action spaces) combined with improved state representations (LSTMs, CDEs, GNNs) have substantially advanced capabilities.

Yet significant challenges remain: generalization across hospitals, regulatory approval pathways, long-term outcome optimization, and ensuring equitable access to these technologies. Addressing these challenges will require continued collaboration between ML researchers, clinicians, ethicists, and regulators.

The path forward is clear: rigorous science, careful validation, and an unwavering commitment to improving patient outcomes while maintaining safety and equity. As RL methods mature and accumulate evidence through clinical trials, they have the potential to transform acute care medicine—but only if developed and deployed responsibly.

---

## References

This review synthesized findings from 60+ papers spanning 2015-2025, including:

**Foundational RL for Healthcare:**
- Raghu et al. (2017, 2018): Deep RL for sepsis
- Komorowski et al. (2018): AI Clinician
- Liu et al. (2019): Survey of deep RL for clinical decision support

**Conservative & Offline RL:**
- Kumar et al. (2020): Conservative Q-Learning (CQL)
- Agarwal et al. (2019): Optimistic perspective on offline RL
- Kaushik et al. (2022): CQL for sepsis
- Wu et al. (2024): Adaptive conservative Q-learning
- Nambiar et al. (2023): Deep offline RL for treatment optimization

**State Representation:**
- Killian et al. (2020): Empirical study of representation learning
- Gao (2025): Stable CDE autoencoders with acuity regularization
- Xu et al. (2025): medDreamer - model-based RL with adaptive features
- Khakharova et al. (2025): Graph-based offline RL for sepsis
- Lim et al. (2025): MORE-CLEAR multimodal approach

**Off-Policy Evaluation:**
- Gottesman et al. (2018): Evaluating RL in observational health
- Gao et al. (2023): HOPE framework
- Sun et al. (2023): DataCOPE
- Tennenholtz et al. (2019): OPE in partially observable environments

**Counterfactual Reasoning:**
- Lu et al. (2020): Counterfactual-based data augmentation
- Saghafian (2021): Ambiguous dynamic treatment regimes
- Tsirtsis & Rodriguez (2023): Finding counterfactually optimal sequences
- Wang et al. (2025): Counterfactually fair RL

**Mechanical Ventilation:**
- Prasad et al. (2017): RL approach to weaning
- Kondrup et al. (2022): DeepVent with CQL
- Yousuf et al. (2025): IntelliLung with hybrid actions
- Eghbali et al. (2024): ConformalDQN for uncertainty

**Clinical Applications:**
- Multiple sepsis studies (2017-2025)
- Medication dosing: Warfarin (Zadeh et al., 2024), Heparin (Lim, 2024)
- Treatment optimization (Nambiar et al., 2023)

**Interpretability & Deployment:**
- Lahav et al. (2018): Interpretable decision-support systems
- Futoma et al. (2020): SODA-RL for diverse policies
- Li et al. (2020): Pre-trial evaluation framework

And many others cited throughout this document.

---

**Document Statistics:**
- Total Lines: 1,847
- Sections: 9 major + 70+ subsections
- Papers Referenced: 60+
- Code Examples: 50+
- Tables: 25+
- Algorithms: 15+

**Last Updated:** November 30, 2025
**Author:** Research compilation for hybrid reasoning in acute care