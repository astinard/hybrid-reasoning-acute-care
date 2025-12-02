# Clinical Missing Data Imputation: A Comprehensive ArXiv Research Synthesis

**Research Date:** December 1, 2025
**Domain:** Clinical AI, Healthcare Machine Learning, Missing Data Imputation
**Total Papers Analyzed:** 80+ unique papers from ArXiv

---

## Executive Summary

Missing data represents a fundamental challenge in clinical machine learning, particularly in Electronic Health Records (EHRs) and time-series medical data. This synthesis analyzes state-of-the-art imputation methods for clinical applications, with emphasis on handling irregular temporal patterns, high missing rates, and complex missingness mechanisms.

### Key Findings:

1. **Deep learning methods** outperform traditional statistical approaches for complex clinical time series, particularly when handling multivariate temporal dependencies
2. **Missing data mechanisms** (MCAR, MAR, MNAR) significantly impact imputation strategy selection and model performance
3. **Downstream task performance** does not always correlate with imputation accuracy, highlighting the need for task-aware imputation
4. **Temporal dynamics** and irregular sampling in clinical data require specialized architectures beyond standard imputation methods
5. **No single method** dominates across all clinical scenarios; performance depends on data characteristics, missing patterns, and downstream objectives

### Critical Gap for ED Applications:

Current methods focus primarily on ICU settings with relatively structured data collection. Emergency Department (ED) data presents unique challenges:
- Extremely high missing rates (often >80% for some variables)
- Highly irregular observation patterns
- Time-critical decision requirements
- Mixed missing mechanisms within the same dataset

---

## 1. Key Papers with ArXiv IDs

### 1.1 Systematic Reviews and Benchmarks

**2210.08258v1** - "Handling missing values in healthcare data: A systematic review of deep learning-based imputation techniques"
- **Authors:** Liu et al. (2022)
- **Key Contribution:** Comprehensive review of 64 DL-based imputation studies
- **Key Finding:** "Integrated" strategy (concurrent imputation with downstream tasks) popular for temporal (50%) and multi-modal data (71.4%)
- **Data Types Covered:** Tabular static (26.6%), temporal (37.5%), multi-modal (7%)
- **Conclusion:** DL methods customizable by data type; portability and fairness remain open questions

**2302.10902v2** - "Deep Imputation of Missing Values in Time Series Health Data: A Review with Benchmarking"
- **Authors:** Kazijevs & Samad (2023)
- **Key Contribution:** Six data-centric experiments on five health datasets
- **Key Finding:** No single method outperforms across all datasets; performance depends on data types, statistics, missing rates, and types
- **Important Insight:** Methods jointly performing cross-sectional and longitudinal imputation yield better results
- **Datasets:** MIMIC-III, PhysioNet, eICU

**2405.17508v3** - "Beyond Random Missingness: Clinically Rethinking for Healthcare Time Series Imputation"
- **ArXiv ID:** 2405.17508v3
- **Authors:** Qian et al. (2024)
- **Key Contribution:** Challenges random masking evaluation paradigm
- **Critical Finding:** Masking strategy significantly influences model performance; imputation accuracy ≠ clinical prediction capability
- **Dataset:** PhysioNet Challenge 2012
- **Tasks:** Mortality prediction
- **Insight:** Current evaluation frameworks may need reconsideration for clinical deployment

### 1.2 Novel Deep Learning Architectures

**2312.16713v5** - "CSAI: Conditional Self-Attention Imputation for Healthcare Time-series"
- **Authors:** Qian et al. (2023)
- **Method:** Conditional Self-Attention with attention-based hidden state initialization
- **Key Features:**
  - Domain-informed temporal decay mimicking clinical recording patterns
  - Non-uniform masking for non-random missingness
  - Calibrated weights based on temporal and cross-sectional characteristics
- **Performance:** State-of-the-art on four EHR benchmarks
- **Integration:** Available in PyPOTS open-source toolbox

**2401.16796v2** - "Learnable Prompt as Pseudo-Imputation (PAI)"
- **Authors:** Liao et al. (2024)
- **Innovation:** Challenges necessity of traditional imputation
- **Method:** Learnable prompt modeling implicit preferences for missing values
- **Key Advantage:** No imputed data injection, reducing bias risk
- **Performance:** Significant improvement over state-of-the-art EHR models
- **Robustness:** Higher robustness with data insufficiency and high missing rates
- **Applicability:** Plug-and-play protocol for any EHR model

**1911.07572v2** - "Bayesian Recurrent Framework for Missing Data Imputation and Prediction"
- **Authors:** Guo et al. (2019)
- **Method:** Bayesian recurrent network with uncertainty quantification
- **Key Feature:** Provides reliability assessment of imputations and predictions
- **Performance:** Strong gains over SOTA on MIMIC-III and PhysioNet
- **Advantage:** Probability distributions for assessing reliability

**2103.02349v1** - "Hamiltonian Monte Carlo Model for Imputation and Augmentation"
- **Authors:** Pourshahrokhi et al. (2021)
- **Method:** Folded Hamiltonian Monte Carlo (F-HMC) with Bayesian inference
- **Application:** Cancer symptom assessment dataset
- **Key Features:**
  - Processes cross-dimensional relations
  - Random walk and Hamiltonian dynamics
  - Privacy-preserving approach
- **Performance:** Improved precision, accuracy, recall, F1 score, and propensity metrics

### 1.3 Temporal and Irregular Data Handling

**2304.07821v1** - "Time-dependent Iterative Imputation (TDI)"
- **Authors:** Noy & Shamir (2023)
- **Method:** Integrates forward-filling with Iterative Imputer
- **Key Innovation:** Patient, variable, observation-specific dynamic weighting
- **Dataset:** MIMIC-III (500,000+ observations), COVID-19 inpatient data
- **Performance:** RMSE 0.63 vs. 0.85 for SoftImpute (25/30 variables)
- **Application:** Demonstrated improved risk prediction

**2412.11164v1** - "Missing data imputation for noisy time-series data and applications in healthcare"
- **Authors:** Le et al. (2024)
- **Key Finding:** MICE-RF outperforms deep learning (SAITS, BRITS, Transformer) for noisy data
- **Missing Rates Tested:** 10%-80%
- **Important Insight:** Imputation can have denoising effects
- **Implication:** Simpler methods may be preferable for noisy clinical data

**2211.06045v2** - "Integrated Convolutional and Recurrent Neural Networks for Health Risk Prediction"
- **Authors:** Liu et al. (2022)
- **Method:** End-to-end without imputation data generation
- **Key Advantage:** Avoids clinical meaning distortion from imputed data
- **Architecture:** Captures both long- and short-term temporal patterns
- **Performance:** Superior to imputation-based methods on two real-world datasets

**2107.14293v2** - "Self-Supervised Transformer for Sparse and Irregularly Sampled Multivariate Clinical Time-Series (STraTS)"
- **Authors:** Tipirneni & Reddy (2021)
- **Method:** Continuous Value Embedding without discretization
- **Key Features:**
  - Treats time-series as observation triplets
  - Self-supervision via time-series forecasting
  - Multi-head attention for contextual embeddings
- **Performance:** Better than SOTA, especially with limited labeled data
- **Dataset:** UK Biobank
- **Code:** Available on GitHub

### 1.4 Missing Data Mechanisms

**2206.12295v1** - "Imputation and Missing Indicators in Clinical Prediction Models"
- **Authors:** Sisk et al. (2022)
- **Key Finding:** Commonly taught MI principles may not apply to clinical prediction models
- **Critical Insight:** Performance depends on whether missing data allowed at deployment
- **Recommendation:** Regression imputation (RI) can outperform MI when imputation needed at deployment
- **Missing Indicators:** Helpful in specific cases but harmful when missingness depends on outcome

**2304.11749v1** - "Missing Values and Imputation: Can Interpretable Machine Learning Help?"
- **Authors:** Chen et al. (2023)
- **Method:** Explainable Boosting Machines (EBMs) for understanding missingness
- **Applications:**
  1. Gain insights on missingness mechanisms
  2. Detect/alleviate risks from imputation algorithms
- **Dataset:** Real-world medical datasets
- **Contribution:** Helps users understand causes of missingness and imputation risks

**2002.12359v1** - "A Kernel to Exploit Informative Missingness in Multivariate Time Series from EHRs"
- **Authors:** Mikalsen et al. (2020)
- **Method:** TCK_IM kernel using ensemble of mixed-mode Bayesian mixture models
- **Key Innovation:** Exploits information in missing patterns without imputation
- **Advantage:** Robust to hyperparameters with limited labels
- **Performance:** Effective on three real-world clinical datasets

### 1.5 Specialized Clinical Applications

**2107.11882v1** - "Lung Cancer Risk Estimation with Incomplete Data"
- **Authors:** Gao et al. (2021)
- **Method:** Conditional PBiGAN (C-PBiGAN)
- **Application:** Multi-modal missing imputation (image + non-image)
- **Key Innovation:** Conditional latent space + class regularization
- **Dataset:** NLST + external validation cohort
- **Performance:** AUC increase of +2.9% (NLST), +4.3% (in-house)

**2005.06935v1** - "Simultaneous imputation and disease classification via Multigraph Geometric Matrix Completion (MGMC)"
- **Authors:** Vivar et al. (2020)
- **Method:** Multiple recurrent GCNs with multigraph signal fusion
- **Key Feature:** Each graph = independent population model (age, sex, cognitive function)
- **Advantage:** End-to-end learning without separate imputation
- **Performance:** Superior classification and imputation vs. SOTA

**2106.11878v1** - "Multiple Organ Failure Prediction with Classifier-Guided GANs"
- **Authors:** Zhang et al. (2021)
- **Method:** Classifier-GAIN with label-aware imputation
- **Application:** ICU mortality and organ failure prediction
- **Key Innovation:** Joint training of imputer and classifier
- **Performance:** Outperforms classical and SOTA baselines across missing scenarios

---

## 2. Imputation Methods Taxonomy

### 2.1 Deep Learning Methods

#### A. Recurrent Neural Networks

**GRU-D (Gated Recurrent Unit with Decay)**
- **Papers:** Multiple studies (2312.16713v5, others)
- **Mechanism:** Temporal decay to handle irregular intervals
- **Strength:** Effective for sequential dependencies
- **Limitation:** May struggle with very long sequences

**BRITS (Bidirectional Recurrent Imputation for Time Series)**
- **Comparison Studies:** 2412.11164v1, 2302.10902v2
- **Mechanism:** Bidirectional RNN with cross-variable correlations
- **Performance:** Mixed results; sometimes outperformed by simpler methods

**Bayesian RNN**
- **Paper:** 1911.07572v2
- **Mechanism:** Uncertainty-aware imputation
- **Advantage:** Reliability estimates
- **Dataset:** MIMIC-III, PhysioNet

#### B. Attention-Based Methods

**SAITS (Self-Attention Imputation for Time Series)**
- **Comparison Studies:** 2412.11164v1, 2302.10902v2
- **Mechanism:** Self-attention with temporal encoding
- **Performance:** Variable across datasets

**Transformer-Based**
- **Papers:** 2412.11164v1, 2107.14293v2 (STraTS)
- **Mechanism:** Multi-head attention for dependencies
- **Strength:** Captures long-range dependencies
- **Challenge:** Computational complexity for long sequences

**CSAI (Conditional Self-Attention Imputation)**
- **Paper:** 2312.16713v5
- **Mechanism:** Attention-based hidden state initialization
- **Features:** Domain-informed decay, non-uniform masking
- **Performance:** SOTA on four EHR benchmarks

#### C. Variational Autoencoders

**MIWAE (Multiple Imputation with VAE)**
- **Papers:** 2304.08054v1 (Fed-MIWAE), others
- **Mechanism:** Deep latent variable model
- **Advantage:** Handles MAR data
- **Applications:** Federated learning scenarios

**GP-VAE (Gaussian Process VAE)**
- **Paper:** 1907.04155v5
- **Mechanism:** GP-based smooth temporal evolution
- **Strength:** Uncertainty quantification
- **Application:** Computer vision, healthcare

**Variational-Recurrent Networks**
- **Paper:** 2003.00662v2
- **Mechanism:** Combines VAE with RNN
- **Features:** Uncertainty-aware, temporal dynamics
- **Dataset:** PhysioNet 2012, MIMIC-III

#### D. Generative Adversarial Networks

**Classifier-GAIN**
- **Paper:** 2106.11878v1
- **Mechanism:** GAN with classifier guidance
- **Innovation:** Label-aware imputation
- **Application:** Organ failure prediction

**C-PBiGAN**
- **Paper:** 2107.11882v1
- **Mechanism:** Conditional latent space
- **Application:** Multi-modal medical data
- **Dataset:** NLST lung cancer screening

#### E. Neural ODEs

**Neural ODE-based methods**
- **Papers:** 2005.10693v1, 2509.25381v1 (FCRN)
- **Mechanism:** Continuous-time dynamics
- **Advantage:** Natural for irregular sampling
- **Application:** ICU time series

**FCRN (Functional Competing Risk Net)**
- **Paper:** 2509.25381v1
- **Mechanism:** Gradient-based imputation module
- **Application:** ICU survival analysis
- **Dataset:** MIMIC-IV, Cleveland Clinic

### 2.2 Statistical Methods

#### A. Multiple Imputation

**MICE (Multiple Imputation by Chained Equations)**
- **Status:** Widely used baseline
- **Variants:** MICE-RF (Random Forest variant)
- **Performance:** 2412.11164v1 shows competitive results for noisy data
- **Limitation:** Assumes MAR

**MI with Random Forest (MICE-RF)**
- **Paper:** 2412.11164v1
- **Finding:** Outperforms deep learning for noisy time-series (10%-80% missing)
- **Advantage:** Denoising effects
- **Application:** Healthcare monitoring data

#### B. Matrix Completion

**MGMC (Multigraph Geometric Matrix Completion)**
- **Paper:** 2005.06935v1
- **Mechanism:** Graph-based population modeling
- **Features:** Recurrent GCNs, self-attention fusion
- **Performance:** Superior on medical datasets

**SoftImpute**
- **Comparison:** Outperformed by TDI (2304.07821v1)
- **RMSE:** 0.85 vs. 0.63 for TDI

#### C. Gaussian Processes

**Multi-task GP**
- **Paper:** 2505.12076v1
- **Mechanism:** Deep GP emulation with stochastic imputation
- **Features:** Uncertainty estimation, longitudinal + cross-sectional info
- **Performance:** Better than MICE, last-known value, individual GPs

**Mixture-based GP**
- **Paper:** 1908.04209v3
- **Mechanism:** GP + mixture models with individualized weights
- **Strength:** Captures cross-sectional and temporal correlations
- **Dataset:** Real-world and synthetic clinical data

#### D. Iterative Methods

**TDI (Time-Dependent Iterative Imputation)**
- **Paper:** 2304.07821v1
- **Mechanism:** Forward-filling + Iterative Imputer with dynamic weighting
- **Innovation:** Patient, variable, observation-specific weights
- **Performance:** RMSE 0.63 (MIMIC-III, 25/30 variables best)

**MedImpute**
- **Paper:** 1812.00418v1
- **Mechanism:** Extension of OptImpute for panel data
- **Features:** K-NN model with scalable first-order methods
- **Application:** Framingham Heart Study

### 2.3 Hybrid and Novel Approaches

**PAI (Learnable Prompt as Pseudo-Imputation)**
- **Paper:** 2401.16796v2
- **Innovation:** Avoids traditional imputation entirely
- **Mechanism:** Learnable prompt for implicit missing value preferences
- **Performance:** Significant improvement on four datasets
- **Robustness:** High performance with data insufficiency

**No Imputation Approaches**
- **Paper:** 2211.06045v2
- **Philosophy:** Process missing data directly without generation
- **Advantage:** Preserves clinical meaning
- **Performance:** Superior to imputation-based methods

**Determinantal Point Processes (DPP)**
- **Paper:** 2303.17893v2
- **Enhancement:** For MICE and MissForest
- **Advantages:** Improved quality, deterministic, variance removal
- **Includes:** Quantum hardware experiments

---

## 3. Missing Data Mechanisms

### 3.1 MCAR (Missing Completely At Random)

**Definition:** Missingness independent of observed and unobserved data

**Implications:**
- Least restrictive assumption
- Complete case analysis unbiased
- Loss of power main concern

**Detection Methods:**
- Little's MCAR test (widely used benchmark)
- PKLM test (2109.10150v2) - non-parametric, handles discrete/continuous data
- U-statistics-based test (2501.05596v1) - utilizes partially observed variables

**Clinical Reality:**
- Rare in practice
- Most clinical missingness has informative patterns

### 3.2 MAR (Missing At Random)

**Definition:** Missingness depends on observed data only

**Implications:**
- Most common assumption for clinical imputation
- Likelihood-based methods valid
- Multiple imputation appropriate

**Papers:**
- **2206.12295v1:** MI may not apply to clinical prediction when missing data at deployment
- **2407.13904v1:** MAR preferred over latent ignorability for outcome missingness
- **2411.14542v1:** Deterministic imputation better for clinical risk prediction under MAR

**Challenges:**
- Difficult to verify empirically
- Requires rich observed data
- May not hold for clinical measurements

### 3.3 MNAR (Missing Not At Random)

**Definition:** Missingness depends on unobserved values

**Implications:**
- Most challenging mechanism
- Requires sensitivity analysis
- Model-based approaches needed

**Approaches:**
- **No Self-Censoring (NSC) model** (2302.12894v1)
- **Selection models**
- **Pattern mixture models**
- **Informative missingness kernels** (2002.12359v1)

**Clinical Examples:**
- Lab tests not ordered due to patient condition
- Missing measurements indicating stability
- Selective reporting of abnormal values

### 3.4 Informative Missingness

**Key Papers:**
- **2002.12359v1:** TCK_IM kernel exploits informative patterns
- **2405.17508v3:** Emphasizes need for clinically-informed masking
- **2304.11749v1:** EBMs to understand missingness causes

**Characteristics:**
- Missingness pattern carries clinical information
- Common in EHR data
- Reflects clinical decision-making

**Strategies:**
- Missing indicators as features (2206.12295v1)
- Missingness-aware attention (2312.16713v5)
- Pattern-based kernels (2002.12359v1)

---

## 4. Downstream Task Impact

### 4.1 Critical Finding: Imputation ≠ Prediction Performance

**Key Papers:**
- **2405.17508v3:** Imputation accuracy doesn't translate to optimal clinical prediction
- **2401.16796v2:** PAI improves prediction without traditional imputation
- **2211.06045v2:** Direct processing outperforms imputation for prediction

**Implications:**
- Task-specific imputation strategies needed
- End-to-end learning may be preferable
- Evaluation metrics should include downstream tasks

### 4.2 Clinical Prediction Tasks

#### Mortality Prediction

**Papers:**
- **1911.07572v2:** Bayesian framework on MIMIC-III, PhysioNet
- **2405.17508v3:** PhysioNet Challenge 2012 (11 methods)
- **2107.14293v2:** STraTS mortality prediction
- **2106.11878v1:** Multiple organ failure and mortality

**Key Findings:**
- Recurrent architectures show consistent performance
- Uncertainty quantification improves reliability
- Missing patterns often informative for mortality

#### Disease Progression

**Papers:**
- **2304.07821v1:** TDI for risk prediction
- **2107.11882v1:** Lung cancer risk estimation
- **2005.06935v1:** MGMC for disease classification

**Challenges:**
- Long-term missing patterns
- Evolving clinical status
- Treatment effects on missingness

#### Organ Failure and Sepsis

**Papers:**
- **2106.11878v1:** Multiple organ failure with Classifier-GAIN
- **2010.13952v1:** Septic shock early prediction
- **1708.05894v1:** Real-time sepsis detection with GP-RNN

**Critical Requirements:**
- Early prediction capability
- Real-time imputation
- Uncertainty handling

#### Risk Stratification

**Papers:**
- **2411.14542v1:** Clinical risk prediction with bootstrap + deterministic imputation
- **2203.11391v1:** IPF survival analysis
- **2509.25381v1:** FCRN for competing risks

**Key Considerations:**
- Calibration important
- Missing data at deployment common
- Need for transportability

### 4.3 Performance Metrics

**Imputation Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Reconstruction accuracy

**Prediction Metrics:**
- AUROC (Area Under ROC)
- AUPRC (Area Under Precision-Recall)
- Calibration (slope, intercept)
- Brier Score

**Important Insight (2405.17508v3):**
"Imputation accuracy doesn't necessarily translate to optimal clinical prediction capabilities"

---

## 5. Clinical Applications by Domain

### 5.1 Intensive Care Unit (ICU)

**Primary Datasets:**
- **MIMIC-III:** Most widely used (20+ papers)
- **MIMIC-IV:** Newer version (2509.25381v1, 2505.19785v2)
- **eICU:** Multi-center validation (2410.09199v1)
- **PhysioNet Challenge 2012:** Benchmark (2405.17508v3, 1911.07572v2)

**Common Tasks:**
- Mortality prediction
- Length of stay
- Organ failure prediction
- Sepsis detection
- Mechanical ventilation prediction

**Data Characteristics:**
- High frequency measurements
- Structured missingness patterns
- Multivariate time series
- Missing rates: 10-80%

**Key Papers:**
- **2509.25381v1:** FCRN for ICU competing risks (MIMIC-IV)
- **2106.11878v1:** Organ failure prediction
- **2010.13952v1:** Sepsis early prediction across EHR systems
- **2102.01147v1:** COVID-19 mechanical ventilation prediction

### 5.2 Emergency Department

**Identified Gap:** Limited research specific to ED settings

**ED-Specific Challenges:**
- Extremely irregular observation patterns
- Very high missing rates (>80% common)
- Time-critical decisions
- Heterogeneous patient presentations
- Limited historical data

**Relevant Methods:**
- Real-time imputation (2012.01099v1)
- Fast inference models (2401.16796v2 - PAI)
- Uncertainty-aware methods (1911.07572v2)

**Opportunities:**
- Adaptation of ICU methods
- Specialized architectures for ED workflows
- Integration with clinical decision rules

### 5.3 Oncology

**Papers:**
- **2107.11882v1:** Lung cancer screening (NLST)
- **2103.02349v1:** Cancer symptom assessment
- **2203.11391v1:** Idiopathic Pulmonary Fibrosis

**Challenges:**
- Long-term follow-up
- Treatment-related missingness
- Multi-modal data (imaging + clinical)

### 5.4 Cardiovascular Disease

**Papers:**
- **1812.00418v1:** Framingham Heart Study
- **2010.01052v3:** Heart-brain interactions (UK Biobank)

**Characteristics:**
- Longitudinal measurements
- Lifestyle factors
- Multi-modal assessments

### 5.5 Neurology

**Papers:**
- **2206.08019v2:** Alzheimer's disease (longitudinal, multimodal)
- **2511.20154v1:** AD progression with irregular sampling

**Challenges:**
- Cognitive assessments
- Imaging + clinical data
- Long disease trajectories

---

## 6. Research Gaps and Future Directions

### 6.1 Identified Research Gaps

#### 1. Emergency Department Applications
- **Gap:** Most research focuses on ICU; limited ED-specific methods
- **Challenges:** Higher missing rates, more irregular patterns, time-critical decisions
- **Opportunity:** Develop ED-optimized imputation strategies

#### 2. Real-Time Clinical Deployment
- **Gap:** Most methods evaluated offline; limited real-time validation
- **Papers Addressing:** 2012.01099v1 (real-time imputation), 1708.05894v1 (sepsis)
- **Needs:** Computational efficiency, online learning, deployment frameworks

#### 3. Fairness and Bias
- **Gap:** Limited evaluation of fairness across demographics
- **Mention:** 2210.08258v1 identifies fairness as open question
- **Needs:** Systematic bias evaluation, mitigation strategies

#### 4. Interpretability
- **Gap:** Many deep methods are black boxes
- **Papers Addressing:** 2304.11749v1 (EBMs), 2107.14293v2 (attention visualization)
- **Needs:** Clinically meaningful explanations of imputations

#### 5. Multi-Center Generalization
- **Gap:** Limited cross-institution validation
- **Papers Addressing:** 2010.13952v1 (domain adaptation), 2410.09199v1 (external validation)
- **Needs:** Robust methods across different EHR systems

#### 6. Missing Mechanism Detection
- **Gap:** Limited tools for identifying missingness type
- **Papers:** 2304.11749v1, 2002.12359v1
- **Needs:** Automated mechanism classification

### 6.2 Methodological Improvements Needed

#### 1. Handling Extreme Missing Rates
- **Current:** Most methods tested at 10-80%
- **ED Reality:** Often >80% for many variables
- **Direction:** Develop methods robust to extreme sparsity

#### 2. Mixed Missing Mechanisms
- **Current:** Single mechanism assumed
- **Reality:** Different variables have different mechanisms
- **Direction:** Variable-specific mechanism modeling

#### 3. Temporal Irregularity
- **Current:** Methods handle irregular sampling
- **Gap:** Extreme irregularity (ED) less studied
- **Direction:** Adaptive temporal models

#### 4. Computational Efficiency
- **Current:** Many deep methods computationally expensive
- **Need:** Real-time clinical decision support
- **Direction:** Efficient architectures, model compression

### 6.3 Clinical Integration Challenges

#### 1. Trust and Adoption
- **Challenge:** Clinicians reluctant to trust black-box imputations
- **Papers:** 2411.09591v2 (clinician survey on IML)
- **Finding:** Clinicians prefer methods aligning with medical intuition
- **Direction:** Interpretable imputation, uncertainty communication

#### 2. Regulatory Approval
- **Challenge:** Limited FDA-approved AI with imputation
- **Needs:** Validation frameworks, safety demonstrations
- **Direction:** Clinical trial integration, regulatory science

#### 3. EHR Integration
- **Challenge:** Integration with existing clinical workflows
- **Needs:** Standardized interfaces, minimal disruption
- **Direction:** FHIR-compatible implementations

### 6.4 Future Research Opportunities

#### 1. Foundation Models for Clinical Time Series
- **Papers:** 2506.07584v6 (MIRA), 2509.24118v1 (HyMaTE)
- **Direction:** Pre-trained models for clinical imputation
- **Potential:** Transfer learning across institutions

#### 2. Federated Learning
- **Papers:** 2304.08054v1 (Fed-MIWAE), 2509.20867v1 (FMI)
- **Direction:** Privacy-preserving multi-center imputation
- **Benefit:** Larger effective sample sizes

#### 3. Causal Imputation
- **Gap:** Limited causal perspective on imputation
- **Direction:** Causal graphs for missingness mechanisms
- **Benefit:** Better handling of MNAR

#### 4. Multi-Modal Integration
- **Papers:** 2107.11882v1 (imaging + clinical), 2206.08019v2
- **Direction:** Joint imputation across modalities
- **Challenge:** Heterogeneous missing patterns

#### 5. Active Learning for Measurement Selection
- **Direction:** Guide which measurements to obtain
- **Benefit:** Optimize information gain vs. cost
- **Application:** Resource-constrained settings

#### 6. Uncertainty Quantification
- **Papers:** 1911.07572v2, 1907.04155v5 (GP-VAE)
- **Direction:** Better uncertainty estimates
- **Application:** Risk-aware clinical decisions

---

## 7. Relevance to ED Incomplete Data

### 7.1 ED-Specific Challenges

**1. Extremely High Missing Rates**
- **ED Reality:** 80-95% missing for many lab values
- **Literature Gap:** Most methods tested at 10-80%
- **Applicable Methods:**
  - PAI (2401.16796v2): Robust to high missing rates
  - CSAI (2312.16713v5): Non-uniform masking
  - No-imputation approaches (2211.06045v2)

**2. Highly Irregular Temporal Patterns**
- **ED Reality:** Measurements driven by clinical suspicion, not schedule
- **Literature Gap:** ICU has more regular patterns
- **Applicable Methods:**
  - STraTS (2107.14293v2): Handles irregular sampling
  - Neural ODE methods (2005.10693v1, 2509.25381v1)
  - Continuous-time models (2506.07584v6 - MIRA)

**3. Time-Critical Decision Making**
- **ED Reality:** Need rapid predictions (minutes, not hours)
- **Literature Gap:** Most methods not evaluated for speed
- **Applicable Methods:**
  - PAI (2401.16796v2): Single forward pass
  - Lightweight methods: MICE-RF (2412.11164v1)
  - Fast inference transformers

**4. Heterogeneous Patient Populations**
- **ED Reality:** Wide range of acuity and presentations
- **Applicable Methods:**
  - MGMC (2005.06935v1): Population-specific graphs
  - Domain adaptation (2010.13952v1)
  - Multi-view integration (2101.09986v2)

**5. Limited Historical Data**
- **ED Reality:** Often first encounter, no prior visits
- **Literature Focus:** Longitudinal data with history
- **Adaptation Needed:**
  - Cross-sectional imputation emphasis
  - Population-level priors
  - Similar patient matching

### 7.2 Recommended Approaches for ED Settings

#### Tier 1: Immediately Applicable

**1. PAI (Learnable Prompt as Pseudo-Imputation)**
- **Paper:** 2401.16796v2
- **Reasons:**
  - Robust to high missing rates
  - Fast inference (single pass)
  - Plug-and-play integration
  - No bias from imputed data
- **Adaptation:** Train on ED-specific data

**2. MICE-RF for Baseline**
- **Paper:** 2412.11164v1
- **Reasons:**
  - Handles noisy data well
  - Computationally efficient
  - Interpretable
  - Denoising effects
- **Limitation:** Assumes MAR

**3. Missing Indicators + Simple Imputation**
- **Paper:** 2206.12295v1
- **Reasons:**
  - Captures informative missingness
  - Simple to implement
  - Fast inference
- **Caution:** Can harm if missingness depends on outcome

#### Tier 2: Requires Adaptation

**4. CSAI (Conditional Self-Attention)**
- **Paper:** 2312.16713v5
- **Strengths:**
  - Domain-informed temporal decay
  - Non-uniform masking
  - State-of-the-art performance
- **Adaptation:** Configure for ED temporal patterns

**5. TDI (Time-Dependent Iterative)**
- **Paper:** 2304.07821v1
- **Strengths:**
  - Patient-specific weighting
  - Proven on MIMIC-III
  - Improved risk prediction
- **Adaptation:** Tune for ED missing patterns

**6. Direct Missing Data Processing**
- **Paper:** 2211.06045v2
- **Strengths:**
  - No imputation bias
  - Preserves clinical meaning
  - Good for high missingness
- **Adaptation:** Architecture for ED features

#### Tier 3: Research Development

**7. Real-Time Bayesian Imputation**
- **Based on:** 1911.07572v2, 2012.01099v1
- **Development:**
  - Optimize for ED speed requirements
  - Uncertainty for clinical decisions
  - Online updates as data arrives

**8. ED-Specific Foundation Model**
- **Based on:** 2506.07584v6 (MIRA)
- **Development:**
  - Pre-train on large ED corpus
  - Fine-tune for local patterns
  - Transfer across institutions

**9. Informative Missingness Exploitation**
- **Based on:** 2002.12359v1, 2304.11749v1
- **Development:**
  - Learn ED-specific missing patterns
  - Encode clinical decision-making
  - Missingness as feature

### 7.3 Implementation Recommendations

**Phase 1: Baseline Establishment (Months 1-3)**
1. Implement MICE-RF for benchmark
2. Add missing indicators
3. Evaluate on ED retrospective data
4. Establish performance metrics

**Phase 2: Advanced Methods (Months 4-6)**
1. Implement PAI
2. Adapt CSAI or TDI
3. Compare performance
4. Clinical validation

**Phase 3: Deployment (Months 7-12)**
1. Real-time implementation
2. Prospective evaluation
3. Clinician feedback
4. Iterative refinement

**Phase 4: Research Extension (Year 2)**
1. Foundation model development
2. Multi-center validation
3. Regulatory pathway
4. Publication and dissemination

### 7.4 Evaluation Framework for ED Imputation

**Imputation Metrics:**
- RMSE/MAE at varying missing rates (80-95%)
- Stratified by variable type
- Stratified by acuity level

**Clinical Metrics:**
- Downstream prediction accuracy (AUROC, AUPRC)
- Calibration across risk strata
- Decision curve analysis
- Clinical utility metrics

**Operational Metrics:**
- Inference time (target: <1 second)
- Memory footprint
- Integration complexity
- Clinician acceptability

**Robustness Metrics:**
- Performance across missing patterns
- Performance across patient subgroups
- Temporal stability
- Cross-site generalization

---

## 8. Conclusions and Recommendations

### 8.1 Key Takeaways

**1. No Universal Solution**
- Performance depends on data characteristics, missing mechanisms, and downstream tasks
- Need for task-specific and data-specific method selection
- Benchmark across multiple approaches

**2. Deep Learning Shows Promise**
- Outperforms classical methods for complex temporal dependencies
- Especially effective with adequate data and computation
- BUT: Simple methods (MICE-RF) competitive for noisy data

**3. Imputation ≠ Prediction**
- High imputation accuracy doesn't guarantee good downstream performance
- End-to-end or task-aware methods often preferable
- Evaluation should include downstream tasks

**4. Missing Mechanisms Matter**
- MCAR rare in clinical practice
- MAR common assumption but often violated
- MNAR requires sensitivity analysis
- Informative missingness can be exploited

**5. Clinical Integration Critical**
- Interpretability enhances adoption
- Uncertainty quantification builds trust
- Real-time capability required for ED
- Alignment with clinical intuition important

### 8.2 Recommendations for Practitioners

**For Researchers:**
1. Report missing data characteristics (rates, patterns, mechanisms)
2. Evaluate multiple imputation methods
3. Include downstream task performance
4. Provide uncertainty estimates
5. Test across missing rate ranges
6. Conduct sensitivity analyses
7. Share code and data where possible

**For Clinicians:**
1. Understand missing data patterns in your setting
2. Question imputation assumptions
3. Request uncertainty quantification
4. Prefer interpretable methods when possible
5. Validate on local data before deployment
6. Monitor performance over time

**For Healthcare Organizations:**
1. Invest in data quality improvement
2. Document missing data mechanisms
3. Support multi-center collaborations
4. Enable infrastructure for real-time imputation
5. Establish validation frameworks
6. Plan regulatory pathways

### 8.3 Critical Success Factors for ED Implementation

**1. Data Requirements:**
- Large ED dataset with ground truth
- Diverse patient populations
- Multiple missing patterns
- Sufficient outcome events

**2. Technical Requirements:**
- Real-time inference capability (<1 second)
- EHR integration
- Model monitoring and updating
- Version control and reproducibility

**3. Clinical Requirements:**
- Clinical validation studies
- Clinician training and education
- Feedback mechanisms
- Override capabilities

**4. Organizational Requirements:**
- Leadership support
- IT infrastructure
- Regulatory compliance
- Change management

### 8.4 Future Research Priorities

**Highest Priority:**
1. ED-specific imputation methods development
2. Real-time deployment frameworks
3. Fairness and bias evaluation
4. Clinical validation studies
5. Interpretability enhancements

**High Priority:**
6. Multi-center generalization
7. Foundation model development
8. Causal imputation methods
9. Active measurement selection
10. Federated learning approaches

**Medium Priority:**
11. Multi-modal integration
12. Uncertainty quantification improvements
13. Missing mechanism detection
14. Computational efficiency
15. Long-term impact studies

---

## References Summary

### Core Review Papers
- **2210.08258v1:** Liu et al. - Systematic review of DL imputation (64 papers)
- **2302.10902v2:** Kazijevs & Samad - Benchmark on 5 health datasets
- **2405.17508v3:** Qian et al. - Rethinking masking strategies
- **2010.12493v2:** Sun et al. - Review of irregularly sampled methods

### Key Methodology Papers
- **2401.16796v2:** Liao et al. - PAI (Learnable Prompt)
- **2312.16713v5:** Qian et al. - CSAI (Conditional Self-Attention)
- **2304.07821v1:** Noy & Shamir - TDI (Time-Dependent Iterative)
- **1911.07572v2:** Guo et al. - Bayesian Recurrent Framework
- **2107.14293v2:** Tipirneni & Reddy - STraTS (Self-Supervised Transformer)

### Clinical Application Papers
- **2509.25381v1:** Gao et al. - FCRN for ICU competing risks
- **2106.11878v1:** Zhang et al. - Organ failure prediction
- **2107.11882v1:** Gao et al. - Lung cancer with incomplete data
- **2005.06935v1:** Vivar et al. - MGMC for disease classification

### Missing Mechanism Papers
- **2206.12295v1:** Sisk et al. - Imputation in prediction models
- **2304.11749v1:** Chen et al. - Interpretable ML for missingness
- **2002.12359v1:** Mikalsen et al. - Informative missingness kernel

### Total Papers Analyzed: 80+
### Datasets Referenced: MIMIC-III, MIMIC-IV, PhysioNet, eICU, NLST, UK Biobank, and many others
### Methods Covered: 25+ distinct imputation approaches
### Clinical Domains: ICU, ED, Oncology, Cardiology, Neurology, General Medicine

---

**Document Prepared:** December 1, 2025
**Prepared by:** Claude (Anthropic)
**Purpose:** Research synthesis for hybrid reasoning acute care project
**Next Steps:** Apply findings to ED incomplete data challenge

---

## Appendix: ArXiv ID Quick Reference

### Highly Relevant for ED Applications
- 2401.16796v2 - PAI (Learnable Prompt)
- 2412.11164v1 - MICE-RF for noisy data
- 2312.16713v5 - CSAI
- 2304.07821v1 - TDI
- 2211.06045v2 - No imputation approach
- 2206.12295v1 - Missing indicators
- 1911.07572v2 - Bayesian uncertainty
- 2012.01099v1 - Real-time imputation

### Foundation for Understanding
- 2210.08258v1 - Systematic review
- 2302.10902v2 - Benchmarking
- 2405.17508v3 - Rethinking evaluation
- 2010.12493v2 - Irregularly sampled review

### Advanced Methods for Future Work
- 2506.07584v6 - MIRA foundation model
- 2509.25381v1 - FCRN with Neural ODE
- 2304.08054v1 - Federated imputation
- 2107.14293v2 - STraTS transformer
- 2005.06935v1 - MGMC graph-based

### Clinical Validation Examples
- 2106.11878v1 - Organ failure
- 2107.11882v1 - Lung cancer
- 2010.13952v1 - Sepsis across systems
- 2203.11391v1 - IPF survival