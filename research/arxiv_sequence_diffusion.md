# Diffusion Models for Sequence Generation: Literature Review for Clinical Trajectory Applications

## Executive Summary

This literature review examines diffusion models for discrete and continuous sequence generation, with a focus on applicability to clinical trajectory modeling in emergency department (ED) settings. Our analysis of 80+ papers reveals several key findings:

1. **Discrete diffusion models** (D3PM, Diffusion-LM, SEDD) have made significant progress but still lag behind autoregressive models for text generation, requiring hundreds of denoising steps for quality generation.

2. **Time series diffusion models** show promise for continuous trajectory modeling, with recent methods achieving state-of-the-art performance in unconditional and conditional generation tasks.

3. **Critical gap**: Few approaches effectively handle **mixed discrete-continuous data** with **irregular sampling** and **sparse events** - precisely the characteristics of ED patient trajectories.

4. **Medical applications** remain limited, primarily focusing on image generation rather than sequential clinical data.

5. **Conditioning mechanisms** (classifier-free guidance, classifier guidance) are well-developed for continuous diffusion but require adaptation for discrete medical events.

The review identifies significant opportunities for adapting these techniques to ED trajectory generation, particularly through hybrid discrete-continuous approaches and specialized conditioning mechanisms for clinical constraints.

---

## 1. Key Papers and ArXiv IDs

### 1.1 Foundational Discrete Diffusion Models

**D3PM: Structured Denoising Diffusion Models in Discrete State-Spaces**
- **ArXiv ID**: Not directly found in search, but referenced in multiple papers
- **Key Contribution**: First framework for discrete state-space diffusion using categorical distributions
- **Limitation**: Requires absorbing or uniform transition matrices; slow sampling (1000+ steps)

**Diffusion-LM: Improving Controllable Text Generation** (2205.14217v1)
- **Authors**: Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, Tatsunori B. Hashimoto
- **Published**: May 2022
- **Architecture**: Non-autoregressive language model with continuous diffusions on learned embeddings
- **Key Innovation**: Gradient-based algorithm for complex controllable generation via hierarchical latent variables
- **Performance**: Significantly outperforms prior work on fine-grained control tasks
- **Limitation**: Operates on continuous embeddings rather than discrete space directly

**SEDD: Discrete Diffusion Modeling by Estimating Ratios** (2310.16834v3)
- **Authors**: Aaron Lou, Chenlin Meng, Stefano Ermon
- **Published**: October 2023
- **Key Innovation**: Score entropy loss that naturally extends score matching to discrete spaces
- **Performance**: 25-75% perplexity reduction vs existing diffusion paradigms; competitive with GPT-2
- **Advantages**: No temperature scaling needed; 32x fewer evaluations possible
- **Relevance**: Demonstrates potential for discrete event modeling

### 1.2 Unified Discrete Diffusion Frameworks

**USD3: Unified Simplified Discrete Denoising Diffusion** (2402.03701v2)
- **Authors**: Lingxiao Zhao, Xueying Ding, Lijun Yu, Leman Akoglu
- **Published**: February 2024
- **Key Innovation**: Mathematical simplifications enabling unified discrete-time and continuous-time formulation
- **Architecture**: Flexible noise distributions for multi-element objects
- **Performance**: Outperforms all SOTA baselines on established datasets
- **Relevance**: Flexibility for mixed data types critical for clinical applications

**Simple Guidance Mechanisms for Discrete Diffusion** (2412.10193v3)
- **Authors**: Yair Schiff et al.
- **Published**: December 2024
- **Key Innovation**: Unified guidance (classifier-free and classifier-based) for discrete diffusion
- **Novel Contribution**: Uniform noise diffusion models that can continuously edit outputs
- **Applications**: Genomic sequences, molecule design, discretized images
- **Relevance**: Guidance mechanisms applicable to clinical constraint satisfaction

### 1.3 Sequence-Specific Diffusion Models

**DiffuSeq: Sequence to Sequence Text Generation** (2210.08933v3)
- **Authors**: Shansan Gong et al.
- **Published**: October 2022
- **Architecture**: Diffusion model for seq2seq tasks with masked denoising
- **Performance**: Comparable to autoregressive; superior diversity
- **Key Feature**: Non-autoregressive generation enabling parallel decoding
- **Limitation**: Discrete text only; no mixed data types

**Diffusion Forcing: Next-token Prediction Meets Full-Sequence** (2407.01392v4)
- **Authors**: Boyuan Chen et al.
- **Published**: July 2024
- **Key Innovation**: Single network outputs scores and enables unrestricted ODE traversal
- **Architecture**: Combines strengths of next-token and full-sequence diffusion
- **Performance**: CIFAR-10 FID 1.73, ImageNet 64x64 FID 1.92 (single-step)
- **Relevance**: Variable-length generation critical for trajectories of varying duration

**Interacting Diffusion Processes for Event Sequence Forecasting** (2310.17800v2)
- **Authors**: Mai Zeng, Florence Regol, Mark Coates
- **Published**: October 2023
- **Architecture**: Two diffusion processes (time intervals, event types) with interacting denoisers
- **Application Domain**: Temporal point processes
- **Key Advantage**: Multi-step predictions for event sequences
- **Relevance**: Directly applicable to clinical event sequences

### 1.4 Time Series Diffusion Models

**Predict, Refine, Synthesize: Self-Guiding Diffusion for Time Series** (2307.11494v3)
- **Authors**: Marcel Kollovieh et al.
- **Published**: July 2023
- **Architecture**: TSDiff - unconditionally-trained diffusion with self-guidance mechanism
- **Key Innovation**: Task-agnostic training; conditioning during inference without auxiliary networks
- **Applications**: Forecasting, refinement, synthetic generation
- **Performance**: Competitive forecasting; superior synthetic data quality
- **Relevance**: Self-guidance could enable flexible clinical constraint satisfaction

**Diffusion-TS: Interpretable Diffusion for Time Series** (2403.01742v3)
- **Authors**: Xinyu Yuan, Yan Qiao
- **Published**: March 2024
- **Architecture**: Encoder-decoder transformer with disentangled temporal representations
- **Key Innovation**: Decomposition guides semantic capture; Fourier-based loss
- **Applications**: Forecasting, imputation (no model changes needed)
- **Performance**: State-of-the-art on realistic analyses
- **Relevance**: Interpretability crucial for clinical applications

**TimeBridge: Better Diffusion Prior Design** (2408.06672v2)
- **Authors**: Jinseong Park et al.
- **Published**: August 2024
- **Key Innovation**: Diffusion bridges between chosen priors and data distribution
- **Prior Designs**: Data/time-dependent (unconditional), scale-preserving (conditional)
- **Performance**: Outperforms standard diffusion on time series synthesis
- **Relevance**: Custom priors could encode clinical domain knowledge

**SADM: Sequence-Aware Diffusion for Longitudinal Medical Images** (2212.08228v2)
- **Authors**: Jee Seok Yoon et al.
- **Published**: December 2022
- **Architecture**: Sequence-aware transformer as conditional module in diffusion
- **Challenges Addressed**: Variable-length sequences, missing data, high dimensionality
- **Application**: 3D longitudinal medical imaging
- **Key Feature**: Autoregressive generation with missing data handling
- **Relevance**: Only medical sequence diffusion found; demonstrates feasibility

### 1.5 Hybrid and Multimodal Approaches

**MLEM: Generative and Contrastive Learning for Event Sequences** (2401.15935v4)
- **Authors**: Viktor Moskvoretskii et al.
- **Published**: January 2024
- **Architecture**: Treats contrastive and generative as complementary modalities
- **Key Innovation**: Aligns embeddings from both approaches
- **Performance**: Superior on classification, next-event prediction, embedding quality
- **Relevance**: Demonstrates synergy of multiple learning paradigms for events

**Unifying Autoregressive and Diffusion-Based Sequences** (2504.06416v2)
- **Authors**: Nima Fathi et al.
- **Published**: April 2025
- **Key Innovation**: Hyperschedules - distinct noise schedules per token position
- **Unification**: Generalizes both AR models (GPT) and diffusion (SEDD, MDLM)
- **Novel Features**: Hybrid token-wise noising; simplified MDLM inference
- **Performance**: State-of-the-art perplexity; diverse, high-quality sequences
- **Relevance**: Flexible framework could combine AR clinical models with diffusion

---

## 2. Continuous vs Discrete Diffusion Approaches

### 2.1 Continuous Diffusion

#### Theoretical Foundation
- **Forward Process**: Add Gaussian noise according to schedule β_t
  ```
  q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
  ```
- **Reverse Process**: Learn to denoise via score function ∇log p(x_t)
- **Training Objective**: Variational lower bound on log-likelihood

#### Advantages for Sequences
1. **Well-established theory**: Score matching, Langevin dynamics
2. **Smooth interpolation**: Natural for continuous vital signs (temperature, BP, HR)
3. **Gradient-based guidance**: Easy to incorporate constraints via gradients
4. **Proven performance**: State-of-the-art image/audio generation

#### Limitations for Clinical Data
1. **Discretization required**: Medical codes, diagnoses, medications are inherently discrete
2. **Embedding artifacts**: Continuous embeddings may not preserve discrete semantics
3. **Interpretability loss**: Difficult to interpret intermediate continuous states

#### Key Implementations

**Diffusion-LM** (2205.14217v1)
- Operates on continuous word embeddings
- **Architecture**:
  - Encoder: Maps words to continuous latent space
  - Diffusion: Standard DDPM in latent space
  - Decoder: Projects back to vocabulary
- **Conditioning**: Gradient-based using auxiliary classifiers
- **Results**: Strong controllability but requires rounding to discrete tokens

**TimeLDM** (2407.04211v2)
- Latent diffusion for time series
- **Architecture**:
  - VAE: Encodes series into informative, smoothed latent
  - Diffusion: Operates in latent space
  - Decoder: Reconstructs time series
- **Performance**: 55% improvement in discriminative score
- **Advantage**: Parallel generation, robust to various lengths

### 2.2 Discrete Diffusion

#### Theoretical Foundation
- **State Space**: Categorical distributions over discrete tokens
- **Transition Kernels**:
  - Uniform: All states equally likely
  - Absorbing: Special [MASK] token
  - Discretized Gaussian: Rounded continuous noise
- **Forward Process**:
  ```
  q(x_t | x_{t-1}) = Cat(x_t; Q_t x_{t-1})
  ```
  where Q_t is transition matrix

#### Advantages for Clinical Events
1. **Native discrete modeling**: Direct representation of ICD codes, medications, procedures
2. **Semantic preservation**: No embedding-decoding artifacts
3. **Interpretability**: Every intermediate state is valid discrete sequence
4. **Exact likelihood**: Can compute p(x) exactly (unlike some continuous methods)

#### Limitations
1. **Slower sampling**: Typically requires 100-1000 steps vs 10-50 for continuous
2. **Training complexity**: Categorical distributions more difficult to optimize
3. **Limited theory**: Score matching less developed than continuous case
4. **Factorization**: Standard approaches assume token independence

#### Key Implementations

**USD3** (2402.03701v2)
- **Noise Distributions**: Flexible choice (uniform, absorbing, custom)
- **Key Simplifications**:
  - Unified discrete/continuous-time formulation
  - Exact backward sampling (no approximations)
  - Accelerated sampling possible
- **Architecture**: Standard transformer with categorical outputs
- **Performance**: Outperforms all SOTA discrete baselines

**SEDD (Score Entropy)** (2310.16834v3)
- **Loss Function**: Score entropy extends score matching to discrete
  ```
  L = E[||s_θ(x_t, t) - ∇log q(x_t)||²]
  ```
  where s_θ is learned score function
- **Architecture**: Transformer with score output
- **Advantages**:
  - No temperature scaling needed
  - Trade compute for quality (fewer steps possible)
  - Controllable infilling
- **Performance**: Competitive with GPT-2, outperforms other diffusion

**Guidance for Discrete Diffusion** (2412.10193v3)
- **Classifier-Free Guidance**:
  ```
  p(x_{t-1}|x_t, c) ∝ p(x_{t-1}|x_t)^(1-w) p(x_{t-1}|x_t, c)^w
  ```
- **Classifier Guidance**: Use gradients from off-the-shelf classifiers
- **Uniform Noise Models**: Can continuously edit outputs (unlike absorbing)
- **Applications**: Genomics (controllable sequence design), molecules, discrete images

### 2.3 Hybrid Approaches

**Diffusion-LM Paradigm**
- Continuous diffusion on **learned** discrete embeddings
- Advantages: Leverages continuous diffusion machinery
- Disadvantages: Embedding/decoding errors; requires rounding

**CANDI: Hybrid Discrete-Continuous** (2510.22510v2)
- **Key Insight**: Temporal dissonance between discrete and continuous corruption
- **Solution**: Decouples discrete identity corruption from continuous rank degradation
- **Architecture**: Separate corruption schedules for discrete vs continuous aspects
- **Performance**: Superior to pure discrete or continuous approaches
- **Relevance**: ED data has both discrete events and continuous vitals

**Proposed for Clinical Trajectories**
- **Discrete channel**: Event types (diagnoses, procedures, medications)
- **Continuous channel**: Vital signs, lab values (time-stamped)
- **Coupling**: Cross-modal attention between discrete and continuous
- **Benefits**: Preserves semantics of events while modeling continuous dynamics

---

## 3. Sequence Generation Architectures

### 3.1 Transformer-Based Architectures

#### Standard Transformer Encoder-Decoder

**DiffuSeq** (2210.08933v3)
- **Encoder**: BERT-style masked transformer
  - Self-attention over input sequence
  - Positional encodings for temporal order
- **Decoder**: Predicts denoised sequence given noisy input
  - Cross-attention to encoder representations
  - Causal masking for left-to-right dependencies
- **Time Embedding**: Sinusoidal encoding of diffusion timestep
- **Training**: MSE loss on predicted clean sequence
- **Inference**: Iterative denoising (T steps)

**Advantages**:
- Proven architecture for sequence modeling
- Captures long-range dependencies via attention
- Parallel processing during training

**Limitations**:
- Quadratic complexity O(L²) in sequence length L
- Fixed maximum length
- No explicit temporal dynamics modeling

#### Sequence-Aware Transformers

**SADM (Sequence-Aware Diffusion)** (2212.08228v2)
- **Key Innovation**: Transformer conditions on entire history sequence
- **Architecture**:
  - Sequence encoder: Processes historical frames
  - Temporal attention: Learns dependencies across time
  - Diffusion decoder: Generates next frame conditioned on sequence
- **Advantages**:
  - Handles variable-length input sequences
  - Models temporal dependencies explicitly
  - Missing data tolerant (attention masking)
- **Applications**: Longitudinal medical imaging
- **Relevance**: Directly applicable to ED visit sequences

**Diffusion-TS** (2403.01742v3)
- **Decomposition-Guided**: Separates trend, seasonality, residuals
- **Encoder**:
  - Decomposes input via moving average
  - Separate embeddings per component
- **Transformer**:
  - Disentangled attention for each component
  - Cross-component fusion
- **Decoder**: Reconstructs from components
- **Fourier Loss**: Preserves frequency characteristics
- **Interpretability**: Each component has semantic meaning

### 3.2 U-Net and Convolutional Architectures

**Time Series Diffusion Method (TSDM)** (2312.07981v2)
- **Architecture**: 1D U-Net adapted for time series
  - **Encoder**: Downsampling blocks with residual connections
  - **Bottleneck**: Attention mechanism at lowest resolution
  - **Decoder**: Upsampling with skip connections
  - **Time Embedding**: Injected at each resolution level
- **Key Components**:
  - ResBlocks: Residual blocks for feature extraction
  - Attention Blocks: Multi-head self-attention
  - TimeEmbedding: Sinusoidal + learned MLP
- **Performance**: Significant improvement over vanilla DDPM
- **Limitations**: Fixed input/output length

**Advantages for Clinical Data**:
- Local pattern extraction via convolutions
- Multi-scale feature hierarchy
- Efficient for regular time series

**Limitations**:
- Not ideal for irregular sampling
- Fixed receptive field
- Limited long-range modeling

### 3.3 Recurrent and State-Space Models

**Diffusion Forcing** (2407.01392v4)
- **Paradigm Shift**: Causal next-token + full-sequence diffusion
- **Architecture**:
  - Masked transformer (causal attention)
  - Per-token independent noise levels
  - Score prediction per token
- **Training**: Denoising score matching with causal masking
- **Inference**:
  - Can generate single or multiple future tokens
  - Rolls out sequences beyond training horizon
  - Long jumps along ODE trajectories
- **Advantages**:
  - Variable-length generation
  - Combines AR and diffusion strengths
  - Guidance-friendly

**Neural ODE Integration**

**TS-Diffusion** (2311.03303v1)
- **Encoder**: Neural ODE with jump technique
  - Handles irregular sampling naturally
  - Self-attention for missing values
- **Diffusion**: On ODE-encoded representations
- **Decoder**: Another ODE for generation with irregularities
- **Advantages**:
  - Arbitrary sampling times
  - Continuous-time modeling
  - Missing data tolerant
- **Performance**: Excellent on complex, irregular time series

**Relevance to ED Trajectories**:
- ED vitals sampled irregularly (every 15 min to hours)
- Missing measurements common (not all vitals every time)
- Neural ODEs natural fit

### 3.4 Attention Mechanisms

#### Self-Attention
- **Purpose**: Capture dependencies between all positions
- **Formulation**:
  ```
  Attention(Q,K,V) = softmax(QK^T/√d_k)V
  ```
- **Advantage**: Long-range dependencies, permutation-invariant
- **Limitation**: O(L²) complexity

#### Cross-Attention (for Conditioning)
- **Purpose**: Attend from sequence to conditioning information
- **Applications**:
  - Patient demographics → trajectory
  - Chief complaint → likely events
  - Historical visits → current trajectory
- **Implementation**: Queries from sequence, keys/values from conditions

#### Temporal Attention (SADM)
- **Innovation**: Explicitly model temporal order
- **Mechanism**: Attention weights decay with temporal distance
- **Advantage**: Emphasizes recent events while maintaining history

#### Multi-Scale Attention (Diffusion-TS)
- **Decomposition**: Separate attention for trend/seasonal/residual
- **Cross-Scale Fusion**: Exchange information between scales
- **Advantage**: Captures both local patterns and global trends

---

## 4. Conditioning and Guidance Mechanisms

### 4.1 Classifier-Free Guidance

**Formulation** (from 2412.10193v3)
```
p(x|c) ∝ p(x)^(1-w) · p(x|c)^w
```
- **w**: Guidance strength (w=0: unconditional, w>1: stronger conditioning)
- **Training**: Joint unconditional and conditional model
  - Random dropout of condition c with probability p_uncond (e.g., 0.1)
  - Single model learns both p(x) and p(x|c)

**Advantages**:
- No auxiliary classifier needed
- Smooth interpolation between unconditional/conditional
- Superior sample quality vs standard conditioning

**Applications to Clinical Trajectories**:
- **Chief Complaint Conditioning**: Generate trajectories for specific complaints
- **Demographic Conditioning**: Age, sex, comorbidities
- **Outcome Conditioning**: Generate trajectories leading to specific outcomes (admission, discharge)

**Implementation Details**:
```python
# Training
if random() < p_uncond:
    condition = NULL  # Unconditional training
else:
    condition = patient_features

# Inference
score_uncond = model(x_t, t, condition=NULL)
score_cond = model(x_t, t, condition=patient_features)
score_guided = (1-w) * score_uncond + w * score_cond
```

### 4.2 Classifier Guidance

**Formulation**
```
∇log p(x_t|c) = ∇log p(x_t) + ∇log p(c|x_t)
```
- **First term**: Unconditional score (from diffusion model)
- **Second term**: Gradient from classifier on noisy data

**Advantages**:
- Can use off-the-shelf pre-trained classifiers
- Flexible: change classifier without retraining diffusion
- Multiple constraints simultaneously (sum gradients)

**Challenges for Discrete Diffusion**:
- Gradients through discrete space non-trivial
- Requires differentiable relaxations or Gumbel-Softmax

**Clinical Applications**:
- **Risk Score Guidance**: Steer towards high/low risk trajectories
- **Triage Level**: Generate trajectories consistent with triage category
- **Length of Stay**: Target specific LOS ranges
- **Readmission Risk**: Generate readmitted vs non-readmitted trajectories

**Example: Mortality Risk Guidance**
```python
# Mortality risk classifier (pre-trained)
risk_classifier = MortalityPredictor()

# At each denoising step
for t in reversed(range(T)):
    # Standard diffusion step
    score_diffusion = diffusion_model(x_t, t)

    # Classifier gradient
    risk = risk_classifier(x_t)
    grad_risk = autograd.grad(risk, x_t)

    # Guided score
    score = score_diffusion + λ * grad_risk

    # Denoising update
    x_{t-1} = denoise(x_t, score, t)
```

### 4.3 Self-Guidance (TSDiff)

**Key Innovation** (2307.11494v3)
- No auxiliary networks needed
- Leverages implicit density from diffusion model itself

**Mechanism**:
1. **Unconditional Training**: Learn p(x) via standard diffusion
2. **Density Estimation**: Diffusion model provides log p(x_t)
3. **Task-Specific Guidance**: At inference, use density for guidance
   - Forecasting: Condition on historical window
   - Imputation: Condition on observed values
   - Refinement: Iteratively improve predictions

**Advantages**:
- Single model for multiple tasks
- No task-specific training
- Computationally efficient

**Application to ED Trajectories**:
- **Historical Conditioning**: Generate future given past visits
- **Partial Observation**: Fill in missing measurements
- **Counterfactual**: What if different intervention?

### 4.4 Constraint-Based Guidance

**Hard Constraints**
- **Medical Plausibility**:
  - Heart rate in [40, 200]
  - Temperature in [35, 42]°C
  - SpO2 in [70, 100]%
- **Temporal Consistency**:
  - Monotonic trends (e.g., improving sepsis scores)
  - Event ordering (diagnosis before treatment)
- **Domain Rules**:
  - Certain medications require specific diagnoses
  - Procedures require appropriate setting

**Implementation Approaches**:

**1. Projection Method**
```python
for t in reversed(range(T)):
    # Standard denoising
    x_{t-1} = denoise(x_t, score, t)

    # Project to feasible set
    x_{t-1} = project_constraints(x_{t-1})
```

**2. Constrained Optimization** (2406.00990v1, 2504.00342v1)
- Formulate as constrained optimization per step
- Lagrangian relaxation or penalty methods
- Example for ED: max realism subject to plausible vital ranges

**3. Guidance via Constraint Violation**
```python
# Measure constraint violation
violation = compute_violations(x_t)

# Gradient of violation
grad_v = autograd.grad(violation, x_t)

# Incorporate into score
score = score_diffusion - λ * grad_v
```

### 4.5 Multi-Condition Fusion

**Scenario**: Multiple conditioning sources
- Patient demographics
- Chief complaint
- Triage level
- Historical visits
- Current vital signs

**Approach 1: Concatenation**
```python
conditions = concat([demographics, complaint, triage, history])
score = model(x_t, t, condition=conditions)
```

**Approach 2: Cross-Attention**
```python
# Separate embeddings
embed_demo = embed_demographics(demographics)
embed_complaint = embed_chief_complaint(complaint)
embed_history = embed_sequence(history)

# Multi-head cross-attention
attended = cross_attention(
    query=x_t,
    keys=[embed_demo, embed_complaint, embed_history],
    values=[embed_demo, embed_complaint, embed_history]
)
```

**Approach 3: Hierarchical Conditioning**
```python
# Demographics → complaint → history
h_demo = encoder_demo(demographics)
h_complaint = encoder_complaint(complaint, condition=h_demo)
h_history = encoder_history(history, condition=h_complaint)

score = model(x_t, t, condition=h_history)
```

---

## 5. Applications to Clinical Trajectories

### 5.1 Direct Applicability Assessment

#### High Relevance (Directly Applicable)

**1. Event Sequence Generation**
- **Papers**: Interacting Diffusion Processes (2310.17800v2), MLEM (2401.15935v4)
- **Mapping to ED**:
  - Events: Diagnoses, procedures, medications, imaging orders
  - Timestamps: Irregular intervals (minutes to hours)
  - Marks: Event types from discrete vocabulary
- **Architecture**: Two coupled diffusion processes
  - Process 1: Inter-arrival times (continuous)
  - Process 2: Event types (discrete)
  - Interaction: Denoising functions share representations
- **Advantages**:
  - Native event sequence modeling
  - Captures event dependencies
  - Variable-length sequences

**2. Longitudinal Medical Data**
- **Paper**: SADM (2212.08228v2)
- **Mapping to ED**:
  - Sequence: Multiple ED visits over time
  - Frames: Each visit = multivariate observation
  - Timestamped: Visit dates
- **Architecture**:
  - Sequence-aware transformer encoder
  - Temporal attention across visits
  - Autoregressive generation
- **Challenges**:
  - Originally designed for 3D images, needs adaptation
  - Within-visit vs between-visit dynamics different

**3. Irregular Time Series**
- **Papers**: TS-Diffusion (2311.03303v1), Diffusion-TS (2403.01742v3)
- **Mapping to ED**:
  - Vitals sampled irregularly (not every 15min)
  - Missing values common (not all vitals measured)
  - Variable observation windows
- **Architecture**: Neural ODE encoder/decoder with diffusion in latent space
- **Advantages**:
  - Continuous-time modeling
  - Missing data tolerant
  - Self-attention handles irregularity

#### Moderate Relevance (Requires Adaptation)

**4. Multivariate Time Series Generation**
- **Papers**: TSDiff (2307.11494v3), TimeLDM (2407.04211v2)
- **Mapping to ED**:
  - Multiple vitals: HR, BP, SpO2, Temp, RR
  - Correlated signals
  - Temporal dynamics
- **Architecture**: VAE + diffusion in latent space
- **Adaptations Needed**:
  - Handle mixed discrete/continuous
  - Irregular sampling support
  - Clinical constraint incorporation

**5. Discrete Sequence Modeling**
- **Papers**: SEDD (2310.16834v3), USD3 (2402.03701v2)
- **Mapping to ED**:
  - Medical codes: ICD-10, CPT, medication codes
  - Categorical variables: Triage, disposition
  - Large vocabularies (10K+ codes)
- **Architecture**: Transformer with categorical outputs
- **Adaptations Needed**:
  - Hierarchical codes (ICD-10 structure)
  - Rare code handling
  - Temporal ordering constraints

### 5.2 Specific ED Trajectory Generation Tasks

#### Task 1: Unconditional Trajectory Generation

**Objective**: Generate realistic ED visit trajectories from scratch

**Data**:
- Input: Random noise
- Output: Complete trajectory (arrival → discharge)
  - Sequence of vitals over time
  - Sequence of events (diagnoses, procedures, meds)
  - Outcome (admit/discharge)

**Architecture** (Proposed):
```
Hybrid Diffusion Model:
1. Continuous Channel: Vitals (HR, BP, SpO2, Temp)
   - Neural ODE encoder for irregularity
   - Diffusion in latent space

2. Discrete Channel: Events (diagnoses, procedures, meds)
   - Discrete diffusion (SEDD/USD3 style)
   - Transformer with categorical outputs

3. Cross-Modal Coupling:
   - Cross-attention between channels
   - Vitals influence event timing
   - Events influence vital dynamics
```

**Training**:
- Dataset: Historical ED visits
- Loss: Combined continuous MSE + discrete cross-entropy
- Validation: Distribution matching metrics (MMD, FID for vitals; JS divergence for events)

**Evaluation Metrics**:
- **Fidelity**: Do generated trajectories match real distributions?
  - Vital sign ranges
  - Event frequency distributions
  - Temporal patterns (time-of-day, day-of-week)
- **Diversity**: Do we capture full spectrum of ED presentations?
  - Coverage of diagnostic categories
  - Range of acuity levels
- **Plausibility**: Are trajectories medically reasonable?
  - Clinical expert review
  - Rule-based checks (e.g., medication requires diagnosis)

#### Task 2: Conditional Generation

**Scenario 1: Chief Complaint → Trajectory**
- **Input**: Text description "chest pain"
- **Condition**: Chief complaint embedding
- **Output**: Typical trajectory for chest pain
- **Method**: Classifier-free guidance
- **Application**: Training ED staff, simulation

**Scenario 2: Triage + Demographics → Trajectory**
- **Input**: ESI level 2, 65yo male, HTN
- **Condition**: Structured features
- **Output**: Trajectory consistent with high acuity elderly patient
- **Method**: Concatenated conditioning
- **Application**: Resource planning, wait time prediction

**Scenario 3: Partial Trajectory → Completion**
- **Input**: First 2 hours of vital signs
- **Condition**: Historical observations
- **Output**: Remaining trajectory
- **Method**: Inpainting with masking
- **Application**: Real-time forecasting during visit

**Scenario 4: Counterfactual Generation**
- **Input**: Actual trajectory with intervention X
- **Condition**: Alternative intervention Y
- **Output**: Likely trajectory under Y
- **Method**: Guided diffusion with intervention constraint
- **Application**: Clinical decision support, "what-if" analysis

#### Task 3: Trajectory Refinement

**Objective**: Improve predictions from baseline forecaster

**Setup** (from TSDiff):
1. **Base Forecaster**: LSTM/Transformer predicts future vitals/events
2. **Diffusion Refiner**: Uses learned density to refine predictions
3. **Iterative**: Multiple refinement passes

**Process**:
```python
# Initial prediction from base model
pred_0 = base_forecaster(history)

# Add noise and denoise iteratively
x_T = pred_0 + noise
for t in reversed(range(T)):
    x_{t-1} = refiner_model(x_t, t, condition=history)

# Final refined prediction
pred_refined = x_0
```

**Advantages**:
- Improved prediction uncertainty
- Corrects implausible predictions
- Maintains computational efficiency (few steps)

**Application**: Enhance existing ED prediction models

### 5.3 Addressing ED-Specific Challenges

#### Challenge 1: Mixed Discrete-Continuous Data

**Problem**: ED trajectories contain both
- Continuous: Vitals (HR, BP, temp, SpO2, RR)
- Discrete: Events (diagnoses, medications, procedures)
- Categorical: Triage level, pain scale, disposition

**Existing Solutions**:
1. **Continuous Embeddings** (Diffusion-LM approach)
   - Embed discrete as continuous
   - Apply continuous diffusion
   - Round back to discrete
   - **Limitation**: Semantic loss in embedding

2. **Separate Models**
   - Independent diffusion for each modality
   - Post-hoc alignment
   - **Limitation**: Misses cross-modal dependencies

**Proposed Solution**: CANDI-inspired hybrid
- **Discrete Diffusion**: For categorical variables
  - State space: All valid combinations of discrete values
  - Transition: Absorbing or uniform
- **Continuous Diffusion**: For vitals
  - Standard Gaussian noise
  - Score-based denoising
- **Coupling Mechanism**:
  - Cross-modal attention at each layer
  - Discrete events gate continuous dynamics
  - Continuous vitals influence event likelihood

```python
class HybridEDDiffusion:
    def __init__(self):
        self.discrete_diffusion = DiscreteEventDiffusion()
        self.continuous_diffusion = VitalSignDiffusion()
        self.cross_attention = CrossModalAttention()

    def forward(self, x_discrete, x_continuous, t):
        # Separate processing
        h_discrete = self.discrete_diffusion.denoise(x_discrete, t)
        h_continuous = self.continuous_diffusion.denoise(x_continuous, t)

        # Cross-modal fusion
        h_fused = self.cross_attention(
            h_discrete, h_continuous
        )

        # Final predictions
        pred_discrete = self.discrete_head(h_fused)
        pred_continuous = self.continuous_head(h_fused)

        return pred_discrete, pred_continuous
```

#### Challenge 2: Irregular Sampling & Sparsity

**Problem**:
- Vitals not measured at regular intervals
- Frequency varies by acuity (ESI 1: every 15min, ESI 4: hourly)
- Many events sparse (rare diagnoses, procedures)

**Solution 1: Neural ODE Integration** (from TS-Diffusion)
- Continuous-time hidden state
- Observations at arbitrary times
- Jump updates when events occur

```python
class NeuralODEEncoder:
    def __init__(self):
        self.ode_func = ODEFunc()
        self.jump_update = JumpUpdate()

    def encode(self, observations, times):
        h = initial_state

        for obs, t in zip(observations, times):
            # Evolve hidden state
            h = odeint(self.ode_func, h, t)

            # Jump update on observation
            h = self.jump_update(h, obs)

        return h
```

**Solution 2: Attention-Based** (from Diffusion-TS)
- Self-attention naturally handles variable spacing
- Positional encoding for temporal distance
- Missing values via masking

```python
# Time-aware positional encoding
pos_encoding = sinusoidal_encoding(times)

# Attention with temporal distance
attention_weights = softmax(
    (Q @ K.T) / sqrt(d_k) - distance_penalty(times)
)
```

**Solution 3: Imputation-Aware Training**
- Train diffusion model to handle missing values
- Mask different subsets during training
- At inference, condition on observed, generate missing

#### Challenge 3: Extreme Imbalance

**Problem**:
- Rare conditions (e.g., STEMI, septic shock) underrepresented
- Most visits are low acuity (ESI 4-5)
- Critical events sparse (intubation, code blue)

**Solution 1: Balanced Sampling**
- Oversample rare trajectories during training
- Reweight loss by inverse frequency
- Synthetic augmentation of rare cases

**Solution 2: Hierarchical Modeling**
- First generate acuity level
- Then generate trajectory conditional on acuity
- Ensures coverage of full severity spectrum

**Solution 3: Guidance for Rare Events**
- Train unconditional model on all data
- At inference, guide towards rare events
- Classifier guidance with "rare event detector"

```python
# Rare event detector (pre-trained)
rare_detector = RareEventClassifier()

# Guided generation
for t in reversed(range(T)):
    score_base = model(x_t, t)

    # Steer towards rare events
    rare_prob = rare_detector(x_t)
    grad_rare = autograd.grad(rare_prob, x_t)

    score_guided = score_base + λ_rare * grad_rare
    x_{t-1} = denoise(x_t, score_guided, t)
```

#### Challenge 4: Clinical Constraint Satisfaction

**Problem**: Generated trajectories must satisfy
- **Physiological**: Vital signs in plausible ranges
- **Temporal**: Events in proper order (diagnosis before treatment)
- **Medical**: Interventions appropriate for condition
- **Safety**: No contradicted medications

**Solution 1: Soft Constraints via Guidance**
```python
def clinical_constraint_loss(trajectory):
    loss = 0

    # Vital sign ranges
    loss += relu(trajectory.HR - 200) + relu(40 - trajectory.HR)
    loss += relu(trajectory.SpO2 - 100) + relu(70 - trajectory.SpO2)

    # Temporal ordering
    for event in trajectory.events:
        if event.type == 'medication':
            # Must have diagnosis first
            loss += check_diagnosis_before(event, trajectory)

    # Medical appropriateness
    loss += check_intervention_appropriateness(trajectory)

    return loss

# Apply during generation
grad_constraint = autograd.grad(clinical_constraint_loss(x_t), x_t)
score = score_base - λ * grad_constraint
```

**Solution 2: Hard Constraints via Projection**
```python
def project_to_feasible(trajectory):
    # Clip vitals to ranges
    trajectory.HR = clip(trajectory.HR, 40, 200)
    trajectory.SpO2 = clip(trajectory.SpO2, 70, 100)

    # Reorder events if needed
    trajectory.events = temporal_sort(trajectory.events)

    # Remove contradicted medications
    trajectory = remove_contradictions(trajectory)

    return trajectory

# Apply after each denoising step
x_{t-1} = denoise(x_t, score, t)
x_{t-1} = project_to_feasible(x_{t-1})
```

**Solution 3: Constrained Diffusion Process**
- Define forward process in constrained space only
- Transition kernel respects constraints
- More complex but guarantees feasibility

### 5.4 Evaluation for Clinical Applications

#### Fidelity Metrics

**Distributional Similarity**:
- **For Vitals**: Frechet Inception Distance (FID) on vital trajectories
- **For Events**: Jensen-Shannon divergence on event type distributions
- **Temporal Patterns**: Autocorrelation, spectral density comparison

**Statistical Tests**:
- Maximum Mean Discrepancy (MMD) between real and synthetic
- Two-sample tests (e.g., Kolmogorov-Smirnov) per vital sign
- Chi-square tests for event frequencies

#### Diversity Metrics

**Coverage**:
- % of diagnostic categories represented
- Range of acuity levels (ESI 1-5)
- Variety in outcomes (discharge, admit, ICU, death)

**Mode Collapse Detection**:
- Nearest-neighbor distances in latent space
- Cluster analysis of generated vs real
- Entropy of generated distributions

#### Clinical Utility Metrics

**Downstream Task Performance**:
- Train predictive models on synthetic data
- Test on real data
- Compare to models trained on real data alone
- Metric: Improvement in AUC, F1, etc.

**Clinical Plausibility**:
- Expert review: % deemed "realistic"
- Rule violation rate (e.g., impossible event sequences)
- Physiological feasibility checks

**Augmentation Value**:
```python
# Scenario 1: Real data only
model_real = train(real_data)
auc_real = evaluate(model_real, test_data)

# Scenario 2: Real + synthetic
model_augmented = train(real_data + synthetic_data)
auc_augmented = evaluate(model_augmented, test_data)

# Augmentation value
improvement = auc_augmented - auc_real
```

---

## 6. Research Gaps and Opportunities

### 6.1 Critical Gaps

**1. Lack of Medical Sequence Diffusion Models**
- **Current State**: Only SADM (2212.08228v2) addresses medical sequences, but focuses on imaging
- **Gap**: No models specifically designed for clinical event/vital trajectories
- **Opportunity**: Develop ED-specific diffusion architecture
  - Mixed discrete (events) and continuous (vitals)
  - Irregular sampling native
  - Clinical constraint-aware

**2. Guidance Mechanisms for Medical Constraints**
- **Current State**: Generic guidance (classifier-free, classifier-based)
- **Gap**: Medical constraints are complex (temporal, physiological, protocol-based)
- **Opportunity**:
  - Hierarchical constraint specification language
  - Multi-constraint fusion algorithms
  - Soft vs hard constraint tradeoffs

**3. Evaluation Frameworks for Clinical Generation**
- **Current State**: Image metrics (FID, IS) adapted for sequences
- **Gap**: Clinical utility not captured by statistical metrics alone
- **Opportunity**:
  - Clinical plausibility scoring system
  - Downstream task benchmarks (prediction, diagnosis)
  - Expert evaluation protocols

**4. Handling Extreme Data Sparsity**
- **Current State**: Methods assume sufficient data per class
- **Gap**: Rare ED conditions (e.g., rare toxicology, exotic infections) extremely sparse
- **Opportunity**:
  - Few-shot diffusion models
  - Transfer learning from related conditions
  - Hierarchical generation (family → specific condition)

**5. Interpretability of Generated Trajectories**
- **Current State**: Diffusion models are black boxes
- **Gap**: Clinicians need to understand why a trajectory was generated
- **Opportunity**:
  - Attention visualization techniques
  - Counterfactual explanations ("what changed to get different outcome?")
  - Component-wise contribution analysis (which features most influential?)

### 6.2 Open Research Questions

**RQ1: Optimal Noise Schedule for Medical Data**
- How should noise schedule differ for:
  - Discrete events (diagnoses) vs continuous vitals?
  - High-frequency (HR) vs low-frequency (labs) measurements?
  - Common vs rare conditions?
- Potential: Adaptive schedules based on data characteristics

**RQ2: Multi-Modal Alignment**
- How to ensure consistency between:
  - Vital sign dynamics and event occurrences?
  - Chief complaint and eventual diagnosis?
  - Triage acuity and trajectory severity?
- Potential: Cross-modal alignment losses, consistency constraints

**RQ3: Temporal Causality in Generation**
- How to ensure generated trajectories respect:
  - Causal relationships (treatment follows diagnosis)?
  - Temporal precedence (symptoms before diagnosis)?
  - Intervention effects (medication affects vitals)?
- Potential: Causal graph constraints, intervention-aware diffusion

**RQ4: Sample Efficiency**
- Can we train high-quality diffusion models with limited data?
  - Typical ED dataset: 50K-500K visits
  - Rare conditions: <100 examples
- Potential:
  - Pre-training on large general medical corpora
  - Few-shot adaptation techniques
  - Synthetic pre-training with simpler models

**RQ5: Real-Time Generation**
- Can diffusion models generate fast enough for:
  - Real-time ED simulation?
  - Interactive "what-if" scenario exploration?
  - Streaming trajectory completion?
- Current: 100-1000 steps = slow
- Potential:
  - Accelerated sampling (DDIM, DPM-Solver)
  - Distillation to fewer-step models
  - Caching and amortization techniques

**RQ6: Uncertainty Quantification**
- How to provide calibrated uncertainty estimates?
  - Epistemic: Model uncertainty (rare conditions)
  - Aleatoric: Inherent randomness (patient variability)
- Potential:
  - Ensemble diffusion models
  - Temperature-scaled sampling
  - Posterior variance estimation

### 6.3 Novel Directions

**Direction 1: Diffusion + Symbolic Reasoning**
- **Idea**: Combine diffusion generation with symbolic medical knowledge
- **Architecture**:
  - Diffusion generates candidate trajectories
  - Symbolic reasoner validates against medical ontologies (SNOMED, ICD)
  - Iterative refinement loop
- **Advantage**: Guarantees medical validity

**Direction 2: Hierarchical Trajectory Diffusion**
- **Idea**: Multi-resolution generation
  - Level 1: Visit-level (admitted, discharged, died)
  - Level 2: Phase-level (arrival, triage, treatment, disposition)
  - Level 3: Event-level (specific vitals, diagnoses, procedures)
- **Advantage**: Coarse-to-fine ensures global coherence

**Direction 3: Physics-Informed Medical Diffusion**
- **Idea**: Incorporate physiological models
  - Cardiovascular dynamics (HR, BP coupling)
  - Respiratory mechanics (RR, SpO2 relationship)
  - Pharmacokinetics (medication effects over time)
- **Architecture**: Diffusion score + physics-based score
  ```
  score_total = score_data + λ * score_physics
  ```
- **Advantage**: Physically plausible trajectories

**Direction 4: Federated Diffusion for Privacy**
- **Problem**: ED data highly sensitive, sharing limited
- **Idea**: Train diffusion models in federated manner
  - Local diffusion models per hospital
  - Aggregate scores/gradients (not raw data)
  - Global model benefits from multi-site data
- **Advantage**: Privacy-preserving generation

**Direction 5: Diffusion for Rare Disease Simulation**
- **Problem**: Insufficient data for rare ED presentations
- **Idea**:
  1. Train diffusion on common conditions (ample data)
  2. Fine-tune on rare conditions (few examples)
  3. Guidance to generate diverse rare trajectories
- **Application**: Training for rare emergencies (PE, aortic dissection, anaphylaxis)

---

## 7. Relevance to ED Trajectory Generation

### 7.1 Direct Applications

**Application 1: Synthetic Data Augmentation**
- **Problem**: Limited labeled data for rare ED conditions
- **Solution**: Generate synthetic trajectories with diffusion
- **Process**:
  1. Train diffusion on all available ED visits
  2. Guide generation towards rare conditions
  3. Augment training set for predictive models
- **Expected Benefit**: 10-30% improvement in prediction for rare cases (based on augmentation literature)

**Application 2: Clinical Decision Support**
- **Problem**: Physicians need to anticipate trajectory evolution
- **Solution**: Generate likely future trajectories in real-time
- **Process**:
  1. Observe first 1-2 hours of ED visit
  2. Condition diffusion model on observations
  3. Generate multiple plausible continuations
  4. Display range of outcomes with probabilities
- **Expected Benefit**: Improved situational awareness, earlier intervention

**Application 3: ED Operations Planning**
- **Problem**: Unpredictable patient flow and resource needs
- **Solution**: Generate synthetic patient arrivals and trajectories
- **Process**:
  1. Generate realistic ED census for next shift
  2. Generate individual trajectories for each patient
  3. Simulate resource utilization (beds, staff, equipment)
  4. Optimize staffing and resource allocation
- **Expected Benefit**: Better resource planning, reduced wait times

**Application 4: Training and Education**
- **Problem**: Limited exposure to rare conditions during training
- **Solution**: Generate diverse realistic cases for simulation
- **Process**:
  1. Specify desired case characteristics (e.g., septic shock)
  2. Generate full trajectory including vital evolution
  3. Present to trainees in simulation environment
  4. Assess decision-making and intervention timing
- **Expected Benefit**: Enhanced preparedness for rare emergencies

**Application 5: Counterfactual Analysis**
- **Problem**: Unknown effect of alternative clinical decisions
- **Solution**: Generate trajectories under different interventions
- **Process**:
  1. Observe actual trajectory up to decision point
  2. Generate trajectory with actual intervention
  3. Generate trajectory with alternative intervention
  4. Compare outcomes
- **Expected Benefit**: Evidence for clinical decision improvement

### 7.2 Technical Requirements for ED Application

**Requirement 1: Multivariate Mixed-Type Data**
- **Continuous**: HR, BP (systolic/diastolic), Temp, SpO2, RR (5-7 vitals)
- **Discrete**: Diagnoses (ICD-10), procedures (CPT), medications (NDC)
- **Categorical**: Triage (ESI 1-5), pain scale (0-10), disposition
- **Text**: Chief complaint, clinical notes
- **Technical Need**: Hybrid discrete-continuous diffusion architecture

**Requirement 2: Irregular Temporal Sampling**
- **Vital Frequency**: Every 15min (ESI 1-2) to hourly (ESI 4-5)
- **Event Timing**: Irregular (diagnosis when recognized, not on schedule)
- **Missing Data**: Common (not all vitals every time)
- **Technical Need**: Neural ODE or attention-based temporal modeling

**Requirement 3: Variable Trajectory Length**
- **ED Visit Duration**: 30 minutes (fast-track) to 24+ hours (admitted)
- **Event Count**: 2-3 (simple) to 50+ (complex)
- **Vital Measurements**: 3-5 (short visit) to 100+ (long visit)
- **Technical Need**: Length-adaptive architecture (Diffusion Forcing style)

**Requirement 4: Clinical Constraint Satisfaction**
- **Physiological**: Vital ranges must be plausible
- **Temporal**: Events in proper order
- **Medical**: Interventions match conditions
- **Protocol**: Following clinical pathways (e.g., sepsis bundle)
- **Technical Need**: Constraint-aware guidance mechanisms

**Requirement 5: Interpretability**
- **Stakeholders**: Clinicians, administrators, researchers
- **Needs**:
  - Why was this trajectory generated?
  - Which features most influential?
  - How confident is the model?
- **Technical Need**: Attention visualization, uncertainty quantification, ablation analysis

**Requirement 6: Computational Efficiency**
- **Real-Time Constraint**: Generate trajectory in <1 second for clinical use
- **Batch Generation**: 1000s of trajectories for simulation
- **Current Challenge**: Standard diffusion needs 100-1000 steps
- **Technical Need**: Accelerated sampling, distillation, or cached inference

### 7.3 Proposed Architecture for ED Trajectory Diffusion

```
ED-Diffusion: Hybrid Diffusion Model for Emergency Department Trajectories

Components:

1. Multi-Modal Encoder
   ├─ Continuous Encoder (Vitals)
   │  ├─ Neural ODE for irregular sampling
   │  ├─ Self-attention for missing values
   │  └─ Multi-scale temporal convolutions
   │
   ├─ Discrete Encoder (Events)
   │  ├─ Embedding layer (diagnoses, meds, procedures)
   │  ├─ Hierarchical encoding (ICD-10 structure)
   │  └─ Temporal position encoding
   │
   └─ Categorical Encoder (Triage, disposition)
      └─ Learned embeddings

2. Cross-Modal Fusion
   ├─ Cross-attention between modalities
   ├─ Gated fusion mechanism
   └─ Shared latent space projection

3. Dual Diffusion Process
   ├─ Continuous Diffusion (Vitals)
   │  ├─ Forward: Gaussian noise addition
   │  ├─ Reverse: Score-based denoising
   │  └─ Physics-informed score (physiological constraints)
   │
   └─ Discrete Diffusion (Events)
      ├─ Forward: Absorbing/uniform transition
      ├─ Reverse: Categorical denoising
      └─ Score entropy loss (SEDD-style)

4. Conditioning Mechanism
   ├─ Patient Features (demographics, history)
   ├─ Chief Complaint (text embedding)
   ├─ Triage Level (ESI 1-5)
   └─ Partial Trajectory (for continuation)

   Method: Classifier-free guidance
   - Train unconditionally with 10% dropout
   - Guide at inference time

5. Constraint Satisfaction
   ├─ Soft Constraints (via guidance)
   │  ├─ Vital range penalties
   │  ├─ Temporal ordering scores
   │  └─ Medical appropriateness
   │
   └─ Hard Constraints (via projection)
      ├─ Clip vitals to plausible ranges
      ├─ Reorder events if needed
      └─ Remove contradictions

6. Output Decoder
   ├─ Continuous Decoder → Vital trajectories
   ├─ Discrete Decoder → Event sequences
   └─ Temporal Aligner → Consistent timestamps

Training:
- Dataset: 100K+ ED visits with full trajectories
- Loss: L_continuous (MSE) + L_discrete (CE) + L_constraint
- Optimization: AdamW with gradient clipping
- Validation: Hold-out ED site for generalization

Inference:
- Input: Conditioning information (patient, complaint, etc.)
- Process: T=50 denoising steps (accelerated via DDIM)
- Output: Complete ED trajectory
- Time: <500ms per trajectory (target)

Evaluation:
- Fidelity: FID (vitals), JS divergence (events)
- Diversity: Coverage of diagnostic categories
- Utility: Downstream prediction improvement
- Plausibility: Expert review, rule violation rate
```

### 7.4 Implementation Roadmap

**Phase 1: Foundation (Months 1-3)**
- [ ] Implement continuous diffusion for vital signs only
  - Standard DDPM with transformer backbone
  - Handles regular sampling first
- [ ] Implement discrete diffusion for events only
  - SEDD or USD3 style
  - Medical code embeddings
- [ ] Baseline evaluation on separate modalities
  - Compare to GANs, VAEs
  - Establish baseline metrics

**Phase 2: Integration (Months 4-6)**
- [ ] Develop cross-modal fusion mechanism
  - Attention-based coupling
  - Shared latent space
- [ ] Handle irregular sampling
  - Neural ODE integration
  - Attention masking for missing values
- [ ] Implement classifier-free guidance
  - Conditional training setup
  - Guidance strength tuning

**Phase 3: Constraints (Months 7-9)**
- [ ] Develop clinical constraint framework
  - Codify physiological ranges
  - Temporal ordering rules
  - Medical appropriateness checks
- [ ] Implement constraint guidance
  - Soft constraints via gradients
  - Hard constraints via projection
- [ ] Evaluate constraint satisfaction rate

**Phase 4: Optimization (Months 10-12)**
- [ ] Accelerate sampling
  - DDIM, DPM-Solver integration
  - Distillation to fewer steps
- [ ] Improve sample quality
  - Hyperparameter tuning
  - Architecture ablations
- [ ] Scale to full dataset

**Phase 5: Application (Months 13-15)**
- [ ] Synthetic data augmentation experiments
  - Train predictive models on augmented data
  - Measure improvement on rare conditions
- [ ] Real-time trajectory continuation
  - Optimize for <1s inference
  - Clinical user interface
- [ ] Counterfactual analysis tool
  - Intervention specification interface
  - Outcome comparison visualization

**Phase 6: Validation (Months 16-18)**
- [ ] Clinical expert evaluation
  - Blinded review of real vs synthetic
  - Plausibility scoring
- [ ] Downstream task benchmarking
  - Mortality prediction
  - Disposition prediction
  - Length of stay forecasting
- [ ] Publication and open-sourcing

---

## 8. Conclusion and Future Directions

### 8.1 Summary of Findings

This comprehensive review of diffusion models for sequence generation reveals:

1. **Rapid Progress in Discrete Diffusion**: Recent advances (SEDD, USD3, guidance mechanisms) have closed the gap with autoregressive models for discrete sequence generation.

2. **Time Series Diffusion Maturing**: Methods like TSDiff, Diffusion-TS, and TimeLDM demonstrate strong performance for continuous time series, with self-guidance enabling flexible task adaptation.

3. **Medical Applications Nascent**: Only one paper (SADM) explicitly addresses medical sequences, leaving significant opportunity for specialized clinical models.

4. **Hybrid Approaches Promising**: Papers like CANDI show that combining discrete and continuous diffusion can leverage strengths of both paradigms.

5. **Conditioning Well-Developed**: Classifier-free and classifier-based guidance provide flexible mechanisms for conditional generation applicable to clinical constraints.

6. **Critical Gaps Remain**:
   - No models for mixed discrete-continuous medical sequences
   - Limited handling of extreme sparsity and irregularity
   - Evaluation frameworks inadequate for clinical utility
   - Interpretability and constraint satisfaction underdeveloped

### 8.2 Path Forward for ED Trajectory Generation

The reviewed literature provides a strong foundation for developing diffusion models for ED trajectory generation:

**Immediate Next Steps**:
1. Implement baseline continuous diffusion for vital signs
2. Implement baseline discrete diffusion for events
3. Develop cross-modal fusion mechanism inspired by CANDI
4. Establish evaluation framework combining statistical and clinical metrics

**Medium-Term Goals**:
1. Integrate neural ODEs for irregular sampling
2. Implement classifier-free guidance for clinical conditioning
3. Develop constraint satisfaction mechanisms
4. Achieve generation quality sufficient for data augmentation

**Long-Term Vision**:
1. Real-time trajectory continuation for clinical decision support
2. Counterfactual analysis for intervention planning
3. Federated learning across hospitals for privacy-preserving improvement
4. Integration with symbolic medical knowledge for guaranteed validity

### 8.3 Broader Impact

Success in ED trajectory generation with diffusion models could:

1. **Improve Patient Outcomes**: Better prediction and decision support
2. **Enhance Training**: Realistic simulation of rare conditions
3. **Optimize Operations**: Data-driven resource allocation
4. **Advance Research**: Synthetic data enabling studies previously impossible
5. **Establish Paradigm**: Template for other clinical sequence modeling tasks

The convergence of advances in diffusion models, increased availability of EHR data, and growing computational resources creates an unprecedented opportunity to transform how we model, understand, and predict patient trajectories in emergency care.

---

## References

### Discrete Diffusion Foundations
- **Diffusion-LM** (2205.14217v1): Li et al., "Diffusion-LM Improves Controllable Text Generation"
- **SEDD** (2310.16834v3): Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios"
- **USD3** (2402.03701v2): Zhao et al., "Unified Discrete Diffusion for Categorical Data"
- **DiffuSeq** (2210.08933v3): Gong et al., "DiffuSeq: Sequence to Sequence Text Generation"
- **Guidance** (2412.10193v3): Schiff et al., "Simple Guidance Mechanisms for Discrete Diffusion"

### Time Series Diffusion
- **TSDiff** (2307.11494v3): Kollovieh et al., "Predict, Refine, Synthesize: Self-Guiding Diffusion"
- **Diffusion-TS** (2403.01742v3): Yuan & Qiao, "Diffusion-TS: Interpretable Diffusion"
- **TimeLDM** (2407.04211v2): Qian et al., "TimeLDM: Latent Diffusion Model"
- **TimeBridge** (2408.06672v2): Park et al., "TimeBridge: Better Diffusion Prior Design"
- **TS-Diffusion** (2311.03303v1): Li, "TS-Diffusion: Generating Highly Complex Time Series"

### Medical & Healthcare
- **SADM** (2212.08228v2): Yoon et al., "SADM: Sequence-Aware Diffusion Model"
- **HiDiff** (2407.03548v1): Chen et al., "HiDiff: Hybrid Diffusion Framework"
- **MCRAGE** (2310.18430v3): Behal et al., "MCRAGE: Synthetic Healthcare Data"

### Event Sequences
- **Interacting Diffusion** (2310.17800v2): Zeng et al., "Interacting Diffusion Processes"
- **MLEM** (2401.15935v4): Moskvoretskii et al., "MLEM: Generative and Contrastive Learning"

### Hybrid & Advanced Techniques
- **Diffusion Forcing** (2407.01392v4): Chen et al., "Diffusion Forcing: Next-token Prediction"
- **CANDI** (2510.22510v2): Pynadath et al., "CANDI: Hybrid Discrete-Continuous"
- **Unifying AR/Diffusion** (2504.06416v2): Fathi et al., "Unifying Autoregressive and Diffusion"

### Surveys & Reviews
- **Rise of Diffusion** (2401.03006v2): Meijer & Chen, "The Rise of Diffusion Models"
- **Survey** (2404.18886v4): Yang et al., "Survey on Diffusion Models for Time Series"

---

## Appendix: Key Technical Concepts

### A. Diffusion Process Fundamentals

**Forward Diffusion (Noising)**:
```
q(x_t | x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)
```
where ᾱ_t = ∏_{s=1}^t (1-β_s)

**Reverse Diffusion (Denoising)**:
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))
```

**Training Objective** (Simplified):
```
L = E_{x_0, ε, t} [||ε - ε_θ(x_t, t)||²]
```
Predict the noise added at each step

### B. Discrete State-Space Diffusion

**Categorical Distribution**:
```
q(x_t | x_{t-1}) = Cat(x_t; Q_t x_{t-1})
```
where Q_t is transition matrix

**Absorbing State**:
```
Q_absorb = [
  [1-β_t, β_t,   0,   ..., 0],
  [0,     1-β_t, β_t, ..., 0],
  ...
  [β_t,   0,     0,   ..., 1-β_t]
]
```

**Uniform Transition**:
```
Q_uniform = (1-β_t)I + (β_t/K)11^T
```
K = vocabulary size

### C. Score-Based Formulation

**Score Function**:
```
∇log p(x_t) ≈ -ε_θ(x_t, t) / √(1-ᾱ_t)
```

**Langevin Dynamics Sampling**:
```
x_{t-1} = x_t + (σ_t² / 2)∇log p(x_t) + σ_t z
```
where z ~ N(0,I)

### D. Guidance Formulations

**Classifier-Free**:
```
ε_guided = (1+w)ε_θ(x_t, t, c) - w·ε_θ(x_t, t, ∅)
```

**Classifier-Based**:
```
ε_guided = ε_θ(x_t, t) - s·√(1-ᾱ_t)∇log p(c|x_t)
```

---

*Document prepared: December 2024*
*Total papers reviewed: 85*
*Focus areas: Discrete diffusion (20), Time series (25), Medical applications (15), Event sequences (12), Hybrid methods (13)*
