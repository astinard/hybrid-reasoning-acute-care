# Causal Inference from Electronic Health Record Data: A Research Synthesis

## Executive Summary

This synthesis reviews recent advances in causal inference methods specifically designed for electronic health record (EHR) data. The focus is on three critical areas: treatment effect estimation, confounding adjustment, and methods for handling the unique challenges of EHR data including missing data, temporal dependencies, and unstructured text information.

**Key Papers Reviewed:**
1. Dynamic Survival Transformers for Causal Inference with EHRs (Chatha et al., 2022)
2. Targeted-BEHRT: Deep Learning for Observational Causal Inference on Longitudinal EHRs (Rao et al., 2022)
3. Leveraging Text Data for Causal Inference Using EHRs (Mozer et al., 2023)
4. Structure Maintained Representation Learning for Causal Inference (Sun et al., 2025)

---

## 1. Overview of Causal Inference Challenges in EHR Data

### 1.1 Fundamental Challenges

**Selection Bias and Confounding by Indication:**
- Sicker patients are more likely to receive treatment (confounding by indication)
- This creates systematic differences between treatment and control groups
- Standard supervised learning approaches fail due to missing counterfactual outcomes

**Data Quality Issues:**
- Missingness rates of 10-30% common in EHR covariates
- Missing not at random (MNAR) mechanisms prevalent
- Discrepancies across different EHR systems

**Temporal Complexity:**
- Time-varying confounders and treatments
- Irregular observation times
- Right-censored survival outcomes

**High Dimensionality:**
- Structured data: demographics, lab values, vital signs, medications
- Unstructured data: clinical notes, imaging, free text
- Need to balance information richness with model complexity

### 1.2 Identifying Assumptions

All methods rely on standard causal inference assumptions:

1. **Strong Ignorability (Unconfoundedness):** Treatment assignment is independent of potential outcomes given observed covariates
2. **Positivity (Overlap):** Each subject has non-zero probability of receiving either treatment
3. **SUTVA (Stable Unit Treatment Value Assumption):** No interference between units
4. **Consistency:** Observed outcomes equal potential outcomes under assigned treatment

---

## 2. Treatment Effect Estimation Methods

### 2.1 Deep Learning Approaches for Individual Treatment Effects (ITE)

#### Representation Learning Framework

**Core Concept:**
Modern deep learning methods use representation learning to map high-dimensional EHR covariates to a learned representation space where:
- Covariate distributions are balanced between treatment groups
- Predictive information for outcomes is preserved

**Key Architectures:**

1. **TARNET (Treatment Agnostic Representation Network):**
   - Shared representation layers for all units
   - Separate hypothesis networks for treatment and control outcomes
   - No explicit balancing constraint

2. **CFRNet (Counterfactual Regression Network):**
   - Extends TARNET with Integral Probability Metrics (IPM) to enforce balance
   - Uses Wasserstein distance or Maximum Mean Discrepancy (MMD)
   - Trades off between balance and predictive accuracy

3. **Structure Maintained Representation Learning (SMRL):**
   - Introduces "structure keeper" to maintain correlation between original covariates and representations
   - Uses adversarial discriminator instead of fixed IPM
   - Addresses the balance-information tradeoff more effectively
   - **Theoretical Contribution:** Proves that PEHE (Precision in Estimation of Heterogeneous Effects) is bounded by:
     ```
     PEHE ≤ 2[L_F|z=0 + L_F|z=1 + d_D(Φ)·ℓ^max - 2σ²_Y - λ·L_RSK]
     ```
     where d_D is H-divergence and L_RSK is the structure keeper loss

**Performance Results (SMRL on MIMIC-III Sepsis):**
- ε_PEHE: 0.56 ± 0.07 (vs. 0.94 ± 0.12 for GANITE)
- ε_ATE: 0.04 ± 0.01 (vs. 0.11 ± 0.05 for GANITE)
- Substantial improvements over CF, BART, and other deep learning methods

#### Transformer-Based Models for Time-Varying Data

**Dynamic Survival Transformer (DynST):**

**Architecture:**
- Leverages transformer self-attention mechanism for temporal EHR sequences
- Handles both static and time-varying covariates
- Autoregressive masking ensures predictions only use past information
- Predicts time-varying hazard functions: h(t|Z, V^(t))

**Key Features:**
1. **Input Embeddings:** Linear transformations for both time-varying and static features
2. **Positional Encodings:** Sinusoidal functions encode temporal positions
3. **Masked Self-Attention:** 8 attention heads with autoregressive constraints
4. **Dual Loss Function:**
   - Cross-entropy adapted for censored data
   - MSE for predicted survival times

**Causal Estimation:**
- Combined with doubly robust estimators (AIPW - Augmented Inverse Probability Weighting)
- AIPW provides unbiased estimates if either outcome model OR propensity model is correct
- On MIMIC-III semi-synthetic data:
  - AIPW (DynST + Logistic): Bias -0.039 ± 0.013 (τ=8)
  - Outperforms Cox regression and standard IPW

**Targeted-BEHRT:**

**Architecture:**
- Extends BERT (Bidirectional Encoder Representations from Transformers) for EHRs
- Pre-training on 6.7M patients via Masked EHR Modeling (MEM)
- Handles multiple data modalities: diagnoses, medications, BP, smoking status, sex

**Key Innovations:**
1. **Two-Part Masked EHR Modeling:**
   - Temporal variables: MLM-style masking of encounters
   - Static variables: VAE-based reconstruction

2. **Propensity Score Modeling:**
   - Integrated 1-hidden layer MLP for propensity prediction
   - Enables CV-TMLE (Cross-Validated Targeted Maximum Likelihood Estimation)

3. **Doubly Robust Estimation:**
   - CV-TMLE corrects for finite-sample bias
   - More stable than standard TMLE with cross-validation
   - Robust to violations of overlap assumption

**Performance on Semi-Synthetic Data:**
- Persistent confounding (Sex, β=1,5,10): Sum Absolute Error = 0.1 ± 0.08
- Transient confounding (Cardiometabolic, β=25,50,75): SAE = 0.556 ± 0.05
- Consistently outperforms LR, BART, TARNET across all confounding intensities

**Clinical Application (Antihypertensives & Cancer):**
- Validated null association (RR ≈ 1.0) across 4 drug class comparisons
- 95% CI covers null hypothesis despite observational confounding
- Empirical RR biased (ranging 1.28-1.55), corrected to near-null

### 2.2 Survival Analysis Methods

**Restricted Mean Survival Time (RMST):**

Definition: Expected survival time up to cutoff τ
```
Y_τ = E[min{T,τ}] = Σ(t=1 to τ) S(t|X)
```

**Advantages over Hazard Ratios:**
- No proportional hazards assumption required
- More interpretable for clinical decision-making
- Captures both mortality and timing information

**DynST Performance:**
- Mean Absolute Error: 11.19 ± 0.24 (vs. 16.04 for Cox Oracle)
- Better handling of time-varying features and complex interactions
- AIPW estimation of ATE on RMST shows minimal bias

---

## 3. Confounding Adjustment Strategies

### 3.1 Representation Balancing

**H-Divergence Based Balancing (SMRL):**

Instead of pre-specified metrics (Wasserstein, MMD), use adversarial discriminator:

```
d_D(Φ) = max_{D∈D} |1/N₀ Σ D(Φ(x_i)) - 1/N₁ Σ D(Φ(x_j))|
```

**Training Procedure:**
- Discriminator D tries to classify treatment vs. control from representations
- Representation layers Φ trained to fool discriminator (adversarial)
- Uses LSGAN objectives for stable training:
  - L_D: (D(Φ(x))+1)² for treated, (D(Φ(x))-1)² for control
  - L_Φ: (D(Φ(x)))² to minimize discriminability

**Advantages:**
- Automatically adapts to data distribution
- No need to specify distance metric
- Better gradient flow than standard GAN

### 3.2 Propensity Score Methods Enhanced by Deep Learning

**Traditional Challenges:**
- Misspecification leads to high bias
- Poor overlap yields high variance
- First-moment balance insufficient for heterogeneous effects

**Deep Learning Solutions:**

1. **Dragonnet Architecture:**
   - Joint modeling of propensity and outcomes
   - Shared representation layers
   - Targeted regularization toward AIPW objective

2. **CV-TMLE with Neural Networks:**
   - Uses deep models for both e(X) and μ_z(X)
   - Cross-validation prevents overfitting
   - Correction step ensures √n-consistency
   - Advantages over CV-AIPTW: more robust to overlap violations

**Results (Targeted-BEHRT):**
- Finite sample experiments (2.5% to 100% of data)
- CV-TMLE maintains stable SAE across all sample sizes
- Without CV-TMLE, TARNET performance degrades severely in small samples

### 3.3 Leveraging Unstructured Text Data

**Motivation:**
- Clinical notes contain information not in structured data
- Doctors mention only notable findings
- Text captures severity, prognosis, clinical reasoning

**Text Representations:**

1. **Bag-of-Words (BoW):**
   - Unigrams, bigrams, trigrams
   - Most interpretable
   - Best performance in heterogeneity detection

2. **ClinicalBERT Embeddings:**
   - 768-dimensional dense vectors
   - Pre-trained on MIMIC-III clinical notes
   - Captures semantic relationships

3. **Multinomial Inverse Regression (MNIR):**
   - Low-dimensional text projections
   - Preserves information about structured covariates
   - Used for missing data imputation

**Three Applications:**

**A. Missing Data Imputation:**

Algorithm:
1. Fit MNIR on complete cases: regress DTM on observed X
2. Generate SR (Sufficient Reduction) scores for all units
3. Augment X with SR scores: X* = (X, Ŝ)
4. Apply MICE to X*

Results (15 variables with missingness):
- RMSE reduction: 5.7% to 17.6%
- R² improvement: 3.7% to 154%
- Largest gains: Weight (16.9%), Temperature (16.8%), Hemoglobin (16.4%)

**B. Matching Quality:**

Procedure:
1. Calculate numerical propensity scores
2. Define propensity score calipers
3. Within calipers, match on cosine distance over DTM

Covariate Balance Results:
- 15/46 structured covariates imbalanced at baseline
- 16/30 text covariates imbalanced at baseline
- PSM alone: balances structured, fails on text
- Text matching: balances both structured AND text covariates
- Standard error reduction: 9.2% → 7.5% treatment effect

**C. Treatment Effect Heterogeneity:**

Method:
- For each n-gram, create indicator variable
- Fit interaction: Y ~ T + I(n-gram present) + T×I(n-gram)
- FDR correction for multiple testing
- Shrinkage estimation for point estimates

Top Positive Heterogeneity (TTE benefits most):
- "sinus tach" (sinus tachycardia): τ = 0.344
- "ogt placed" (mechanical ventilation): τ = 0.301
- "unclear if": τ = 0.298
- "discussed with": τ = 0.287
- "worsening": τ = 0.283

Top Negative Heterogeneity (TTE harmful):
- "renal function": τ = -0.311
- "obese extremities": τ = -0.298
- "solumedrol": τ = -0.287
- "oxycodone": τ = -0.276
- "organomegaly": τ = -0.264

**Clinical Interpretation:**
- Positive effects: cardiac-related conditions (target population)
- Negative effects: non-cardiac conditions (treatment diverts from appropriate care)
- Text heterogeneity effects ~10× larger than structured covariate effects

---

## 4. Handling EHR-Specific Challenges

### 4.1 Missing Data

**Types of Missingness:**

1. **MCAR (Missing Completely at Random):** Rare in EHR
2. **MAR (Missing at Random):** Missingness depends on observed data
3. **MNAR (Missing Not at Random):** Missingness depends on unobserved data

**Text-Augmented MICE:**

Why it works:
- Clinical notes more frequently recorded than lab values
- Contains proxy information about missing values
- MNIR identifies which text features predict each covariate

Assumptions strengthened:
- MAR becomes more plausible with richer observed data
- But does NOT guarantee MAR holds

Best practices:
- Do NOT include outcome in imputation model
- Use all available covariates (structured + text features)
- Multiple imputation (M=5 typically sufficient)
- Combine results using Rubin's rules

### 4.2 Time-Varying Confounding

**Challenge:**
- Past treatment affects future confounders
- Future confounders affect future treatment
- Standard methods fail

**Solutions:**

1. **G-methods:**
   - G-computation
   - Inverse probability of treatment weighting (IPTW)
   - G-estimation

2. **Sequential Modeling:**
   - RNNs/LSTMs for temporal dependencies
   - Transformers with autoregressive masking (DynST)
   - Marginal structural models with time-varying weights

**DynST Approach:**
- At each time t, predict h(t|Z, V^(≤t))
- Use only information available up to time t
- No "future leakage" via autoregressive masking
- Cumulative survival: S(t) = ∏(1 - h(τ))

### 4.3 Informative Censoring

**Problem:**
- Patients lost to follow-up non-randomly
- Death competes with outcome of interest
- Discharge timing related to health status

**Conditional Independence Assumption:**
T ⊥ C | X (survival time independent of censoring given covariates)

**DynST Solution:**
- Models both event and censoring processes
- Shared representation layers
- Loss function explicitly handles censoring indicator δ
- For censored (δ=0): maximize S(t) over observation period
- For events (δ=1): maximize S(t) up to event, minimize after

### 4.4 High-Dimensional Data

**Curse of Dimensionality:**
- P >> N common in EHR studies
- Sparse data in high-dimensional covariate space
- Overfitting risk

**Dimension Reduction Strategies:**

1. **Feature Selection:**
   - Clinical domain knowledge
   - LASSO/Elastic Net regularization
   - Random forests variable importance

2. **Representation Learning:**
   - Neural network bottlenecks (d << P)
   - Autoencoders for unsupervised pre-training
   - Masked language modeling (BERT-style)

3. **Structured Regularization:**
   - SMRL structure keeper: preserves X-Φ(X) correlation
   - Group LASSO for related features
   - Graph-based regularization for related diagnoses

**SMRL High-Dimensional Results:**
- P=50: ε_PEHE = 1.56 (vs. 2.12 for CF)
- P=800: ε_PEHE = 4.58 (vs. 8.20 for TARNET)
- Consistent superiority as dimensionality increases

---

## 5. Causal Discovery Methods

### 5.1 Learning Causal Structure

**Goal:** Infer directed acyclic graph (DAG) of causal relationships

**Challenges in EHR:**
- Unmeasured confounding common
- Feedback loops (treatment → outcome → future treatment)
- Selection bias in data collection

**Approaches:**

1. **Constraint-Based (PC Algorithm):**
   - Tests conditional independence
   - Assumes causal sufficiency (no unmeasured confounders)
   - Often violated in EHR

2. **Score-Based (GES, NOTEARS):**
   - Optimize score function (BIC, AIC)
   - Search over DAG space
   - Computationally intensive for high dimensions

3. **Functional Causal Models:**
   - Assume structural equations
   - Exploit non-Gaussianity (LiNGAM)
   - Or non-linearity (ANM, PNL)

### 5.2 Causal Variable Selection

**Backdoor Criterion:**
Sufficient to adjust for all variables that:
- Affect both treatment and outcome (confounders)
- Do NOT lie on causal path from treatment to outcome

**Methods:**

1. **Deconfounder (Zhang et al.):**
   - Infer latent confounder from observed variables
   - Requires multiple causes
   - Applicable when unmeasured confounding suspected

2. **Sufficient Adjustment Sets:**
   - Identify minimal sets satisfying backdoor criterion
   - Use causal discovery as pre-processing
   - Reduces dimension and variance

**Targeted-BEHRT Insight:**
- Pre-trained representations may capture latent confounders
- MEM forces model to learn structure of EHR data
- Shared representations across tasks

---

## 6. Methodological Comparisons

### 6.1 Performance Summary Across Studies

**IHDP Benchmark (Semi-Synthetic):**

| Method | ε_PEHE | ε_ATE |
|--------|---------|-------|
| SMRLNN | **0.74 ± 0.01** | **0.19 ± 0.01** |
| CFRNet | 0.76 ± 0.02 | 0.27 ± 0.01 |
| TARNET | 0.95 ± 0.02 | 0.35 ± 0.02 |
| CF | 3.8 ± 0.2 | 0.40 ± 0.03 |
| BART | 2.3 ± 0.1 | 0.34 ± 0.02 |

**MIMIC-III Sepsis (Mechanical Ventilation):**

| Method | ε_PEHE | ε_ATE |
|--------|---------|-------|
| SMRLNN | **0.56 ± 0.07** | **0.04 ± 0.01** |
| CFRNet | 0.71 ± 0.10 | 0.08 ± 0.02 |
| TARNET | 0.63 ± 0.08 | 0.07 ± 0.01 |
| GANITE | 0.94 ± 0.12 | 0.11 ± 0.05 |

**Jobs Dataset (Policy Risk):**

| Method | R_pol | ε_ATT |
|--------|-------|-------|
| SMRLNN | **0.18 ± 0.01** | **0.05 ± 0.01** |
| Dragonnet | 0.20 ± 0.02 | 0.08 ± 0.02 |
| CFRNet | 0.21 ± 0.01 | 0.08 ± 0.03 |
| BART | 0.25 ± 0.02 | 0.08 ± 0.03 |

### 6.2 Trade-offs and Recommendations

**Representation Learning vs. Traditional Methods:**

Advantages of Deep Learning:
- Handle high-dimensional, multimodal data
- Flexible non-linear relationships
- Automatic feature learning
- Better balance without explicit propensity scores

Advantages of Traditional Methods:
- Interpretability (especially tree-based)
- Well-understood statistical properties
- Lower computational cost
- Better with small samples (<500)

**When to Use Which:**

1. **Linear relationships, P<50, N>1000:** Logistic regression, BART
2. **Non-linear, high-dimensional, N>5000:** Deep learning (SMRL, Targeted-BEHRT)
3. **Time-varying confounding:** DynST, marginal structural models
4. **Survival outcomes:** DynST, Cox with time-varying covariates
5. **Rich text data:** Text-augmented matching, Targeted-BEHRT
6. **Small samples with text:** Text imputation → BART/CF

---

## 7. Implementation Considerations

### 7.1 Software and Code Availability

**Targeted-BEHRT:**
- PyTorch implementation
- Requires MIMIC-III access (credentialed)
- Pre-trained weights available for BEHRT base

**DynST:**
- TensorFlow/PyTorch
- Adam optimizer with exponential decay
- Hyperparameters: d_model ∈ {32, 48, 64}, layers ∈ {2, 3, 4}

**SMRL:**
- GitHub: https://github.com/SMRLNN/SMRLNN
- Complete pipeline including discriminator training
- Ablation components available (SMRLNN-v0, v1, v2)

**Text Matching:**
- R implementation
- GitHub: https://github.com/reaganmozer/textmatch
- Functions for MNIR, cosine matching, heterogeneity detection

### 7.2 Computational Requirements

**Pre-training (Targeted-BEHRT):**
- 6.7M patients, 5 epochs
- Multiple GPUs (V100 recommended)
- ~48 hours wall time

**Inference:**
- DynST: ~0.1s per patient (CPU)
- SMRL: ~0.05s per patient (GPU)
- Text matching: ~10s for 2000 patients (CPU)

**Memory:**
- Full DTM: ~10GB for 2000 patients, 5000 vocabulary
- BERT embeddings: ~2GB for 2000 patients
- Representation learning: ~4GB GPU memory

### 7.3 Validation Strategies

**Semi-Synthetic Data:**
1. Use real covariates X
2. Simulate outcomes Y(0), Y(1) from known functions
3. True ITE = Y(1) - Y(0) available
4. Compute ε_PEHE, ε_ATE

**Cross-Study Validation:**
- Train on observational data
- Validate on RCT subset (if available)
- Jobs dataset approach

**Sensitivity Analysis:**
1. **Rosenbaum Bounds:** How strong would unmeasured confounder need to be?
2. **E-values:** Minimum strength of association to explain away effect
3. **Tipping Point Analysis:** How many unmeasured confounders?

**Falsification Tests:**
1. Negative control outcomes (known null effects)
2. Placebo treatments
3. Pre-treatment outcome trends

---

## 8. Clinical Applications and Case Studies

### 8.1 Transthoracic Echocardiography (TTE) in Sepsis

**Study:** Text-augmented matching (Mozer et al.)

**Population:** 2,625 sepsis patients in ICU
- Treatment: 1,333 received TTE
- Control: 1,292 did not receive TTE
- Outcome: 28-day mortality

**Text Features Used:**
- Clinical progress notes (physicians, nurses)
- Specialist evaluations
- ~7,000-8,000 characters per patient

**Results:**
- Overall effect: 7.5% ± 2.7% reduction in mortality
- Heterogeneity by text features:
  - Cardiac conditions (sinus tach, etc.): Large positive effects
  - Non-cardiac conditions (renal, obesity): Negative effects
- Interpretation: TTE most beneficial for cardiac-related sepsis

**Clinical Impact:**
- Refined patient selection criteria
- Avoid TTE for non-cardiac critical illness
- Focus resources on likely responders

### 8.2 Antihypertensive Medications and Cancer Risk

**Study:** Targeted-BEHRT (Rao et al.)

**Population:** 516,365 patients across 5 drug classes
- ACEIs: 186,709
- Beta-blockers: 150,098
- CCBs: 128,597
- Diuretics: 28,991
- ARBs: 21,970

**RCT Evidence:** Null association (multiple trials)

**Observational Challenge:**
- Empirical RR ranges 1.28-1.55 (appears harmful)
- Severe confounding by indication
- Different disease profiles across drug classes

**Results:**
- All 4 comparisons: 95% CI includes RR = 1.0
- Validated null association despite confounding
- Demonstrates model's ability to adjust for complex confounding

**Clinical Significance:**
- Reassurance about medication safety
- Illustrates pitfall of naive observational analysis
- Importance of proper confounding adjustment

### 8.3 Mechanical Ventilation in Sepsis

**Study:** SMRL on MIMIC-III (Sun et al.)

**Population:** 20,225 sepsis-3 patients
- Treatment: 4,210 (20.8%) received mechanical ventilation
- Mortality: 1,208 (28.7%) in-hospital deaths
- 47 baseline covariates

**Severe Imbalance:**
- SOFA score, Elixhauser score much higher in treated
- Confounding by indication extreme

**Results:**
- ε_PEHE: 0.56 ± 0.07
- Substantial heterogeneity by patient characteristics
- Model identifies subgroups with positive vs. negative effects

**Clinical Application:**
- Risk stratification for mechanical ventilation
- Personalized decision support
- Optimal timing of intervention

---

## 9. Future Directions

### 9.1 Methodological Advances

**1. Handling Unmeasured Confounding:**
- Deconfounder approaches for latent variables
- Sensitivity analysis automation
- Multiple robustness (beyond doubly robust)

**2. Continuous and Multi-Valued Treatments:**
- Dose-response functions
- Optimal treatment regimes (dynamic treatment rules)
- Generalized propensity scores

**3. Interference and Spillover:**
- Clustered/hierarchical data (hospital effects)
- Contagion in ICU settings
- Network causal inference

**4. Integration with Causal Discovery:**
- Use discovered DAG to guide adjustment sets
- Joint modeling of structure and effects
- Active learning for causal relationships

### 9.2 Clinical Translation

**Implementation Challenges:**
1. **Model Interpretability:**
   - Black-box deep learning vs. clinical acceptance
   - SHAP values, attention visualization
   - Rule extraction from neural networks

2. **Generalizability:**
   - External validation across hospitals
   - Demographic shifts (distribution shift)
   - Transfer learning approaches

3. **Real-Time Deployment:**
   - Integration with EHR systems
   - Computational efficiency
   - Model updating with new data

4. **Regulatory and Ethical:**
   - FDA approval for decision support
   - Fairness and bias (race, gender, SES)
   - Transparency and accountability

### 9.3 Emerging Data Sources

**1. Multi-Modal Data:**
- Medical imaging + EHR
- Genomics + clinical data
- Wearable sensors + EHR

**2. Real-World Evidence:**
- Insurance claims
- Patient-reported outcomes
- Social determinants of health

**3. Federated Learning:**
- Multi-site collaboration without data sharing
- Privacy-preserving causal inference
- Differential privacy

---

## 10. Practical Guidelines

### 10.1 Study Design Checklist

**Before Analysis:**
- [ ] Define treatment, outcome, population clearly
- [ ] Specify time zero (index date)
- [ ] List all potential confounders based on clinical knowledge
- [ ] Draw causal DAG
- [ ] Consider unmeasured confounding sources
- [ ] Plan sensitivity analyses

**Data Preparation:**
- [ ] Handle missing data (mechanism assessment)
- [ ] Define pre-treatment period (avoid immortal time bias)
- [ ] Check for measurement error, outliers
- [ ] Assess positivity (overlap) graphically
- [ ] Split into train/validation/test

**Model Selection:**
- [ ] Choose method based on data structure (Table in Section 6.2)
- [ ] Pre-specify hyperparameter search space
- [ ] Use appropriate validation metric (ε_PEHE for ITE, ε_ATE for ATE)

**Validation:**
- [ ] Check covariate balance (standardized differences)
- [ ] Assess overlap in representation space
- [ ] Perform sensitivity analysis
- [ ] Compare to benchmark methods
- [ ] Clinical plausibility check

### 10.2 Reporting Standards

Follow STROBE-CIM (Causal Inference Extension):

1. **Introduction:**
   - Causal estimand clearly stated
   - Assumptions explicitly listed

2. **Methods:**
   - DAG or conceptual model
   - Handling of time-varying confounding
   - Missing data approach
   - Sensitivity analyses planned

3. **Results:**
   - Covariate balance before/after adjustment
   - ε_PEHE and ε_ATE with confidence intervals
   - Heterogeneity analyses
   - Sensitivity analysis results

4. **Discussion:**
   - Assumptions plausibility
   - Limitations (unmeasured confounding, etc.)
   - Clinical interpretation
   - Generalizability

---

## 11. Key Takeaways

### For Methodologists:

1. **Representation learning** offers powerful framework for high-dimensional EHR data
2. **Balance-information tradeoff** is critical; SMRL's structure keeper addresses this
3. **Text data** contains valuable confounding information beyond structured data
4. **Transformers** handle temporal complexity better than traditional time-series methods
5. **Doubly robust estimators** (CV-TMLE, AIPW) essential for finite-sample performance

### For Clinicians:

1. **Observational EHR studies** can yield valid causal estimates with proper methods
2. **Text in clinical notes** captures nuances not in structured data
3. **Heterogeneous effects** are the norm; average effects insufficient
4. **Selection bias** severe in EHR; sophisticated adjustment necessary
5. **Validation** against RCTs critical where available

### For Data Scientists:

1. **Deep learning** outperforms traditional methods for N>5,000, P>50
2. **Pre-training** on large EHR databases improves downstream performance
3. **Hyperparameter tuning** essential; use validation set with causal metric
4. **Code availability** improving; use existing implementations when possible
5. **Computational cost** manageable with modern hardware

---

## References

### Primary Papers

1. Chatha, P., Wang, Y., Wu, Z., & Regier, J. (2022). Dynamic Survival Transformers for Causal Inference with Electronic Health Records. *NeurIPS Workshop on Learning from Time Series for Health*.

2. Rao, S., Mamouei, M., Salimi-Khorshidi, G., Li, Y., Ramakrishnan, R., Hassaine, A., Canoy, D., & Rahimi, K. (2022). Targeted-BEHRT: Deep learning for observational causal inference on longitudinal electronic health records. *arXiv:2202.03487*.

3. Mozer, R., Kaufman, A. R., Celi, L. A., & Miratrix, L. (2023). Leveraging text data for causal inference using electronic health records. *arXiv:2307.03687*.

4. Sun, Y., Lu, W., & Zhou, Y. H. (2025). Structure Maintained Representation Learning Neural Network for Causal Inference. *Journal of Machine Learning Research*, 1, 1-XX. *arXiv:2508.01865*.

### Additional Key References

5. Shalit, U., Johansson, F. D., & Sontag, D. (2017). Estimating individual treatment effect: generalization bounds and algorithms. *ICML*.

6. Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference. *Journal of Computational and Graphical Statistics*, 20(1), 217-240.

7. van der Laan, M. J., & Rose, S. (2011). Targeted learning: Causal inference for observational and experimental data. *Springer*.

8. Hernán, M. A., & Robins, J. M. (2020). Causal inference: What if. *Chapman & Hall/CRC*.

---

## Appendix: Notation and Definitions

**Core Notation:**
- N: Number of subjects
- Z_i: Treatment indicator (1=treated, 0=control)
- X_i: Pre-treatment covariates (P-dimensional)
- Y_i(1), Y_i(0): Potential outcomes under treatment/control
- Y_i: Observed outcome
- T_i: Survival time
- C_i: Censoring time
- δ_i: Event indicator

**Estimands:**
- τ(x) = E[Y(1) - Y(0) | X=x]: Conditional ATE (CATE) or ITE
- τ_ATE = E[Y(1) - Y(0)]: Average Treatment Effect
- τ_ATT = E[Y(1) - Y(0) | Z=1]: Average Treatment Effect on Treated

**Performance Metrics:**
- ε_PEHE = sqrt(E[(τ̂(X) - τ(X))²]): Precision in Estimation of Heterogeneous Effect
- ε_ATE = |τ̂_ATE - τ_ATE|: Absolute bias in ATE
- R_pol: Policy risk (expected loss under estimated treatment rule)

**Representation Learning:**
- Φ: X → R^d: Representation function (neural network)
- d: Dimension of representation space (d << P typically)
- H: R^d × {0,1} → Y: Hypothesis function (outcome prediction)
- D: R^d → [0,1]: Discriminator (for balancing)

**Text Representations:**
- T_i: Text document for subject i
- C_i: Document-term matrix (BoW representation)
- DTM: N × d matrix of document-term counts
- SR: Sufficient reduction scores (from MNIR)

---

*Document created: January 2025*
*Based on arXiv papers: 2210.15417, 2202.03487, 2307.03687, 2508.01865*
