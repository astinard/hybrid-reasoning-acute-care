# Clinical Decision Thresholds and Operating Points for AI: A Comprehensive Research Review

**Date:** December 1, 2025
**Focus:** Decision thresholds, operating point selection, and clinical utility optimization for healthcare AI systems

---

## Executive Summary

This review synthesizes current research on clinical decision thresholds and operating point selection for AI systems in healthcare. The analysis reveals a critical gap between technical model performance metrics (accuracy, AUROC) and clinical utility in practice. Key findings include:

1. **Threshold selection is context-dependent**: Optimal operating points vary based on clinical priorities, patient populations, cost asymmetries, and deployment settings
2. **Standard metrics are insufficient**: Accuracy and AUROC fail to capture clinical priorities including calibration, distributional shifts, and asymmetric error costs
3. **Decision curve analysis (DCA) is essential**: DCA and net benefit provide actionable frameworks for evaluating clinical utility across threshold ranges
4. **Cost-sensitive learning matters**: Asymmetric misclassification costs must be explicitly modeled, not assumed equal
5. **Calibration is foundational**: Well-calibrated probabilities enable flexible threshold adjustment post-deployment
6. **Multiple operating points may be needed**: Different clinical contexts may require different thresholds for the same model

The research demonstrates that effective clinical AI requires moving beyond pure predictive performance to explicitly modeling clinical decision-making, costs, and utility.

---

## Key Papers with ArXiv IDs

### Decision Curve Analysis and Clinical Utility

**1. The continuous net benefit (2412.07882v1)**
- **Authors:** Benitez-Aurioles et al.
- **Key contribution:** Extends net benefit to handle multiple decision thresholds simultaneously for personalized care
- **Method:** Weighted area under rescaled net benefit curve for populations with varying optimal thresholds
- **Application:** Cardiovascular preventive care with capacity constraints

**2. Net benefit, calibration, threshold selection (2202.01906v1)**
- **Authors:** Pfohl et al.
- **Key contribution:** Demonstrates that threshold selection with calibrated models outperforms fairness-constrained training
- **Method:** Threshold selection concordant with patient preferences, intervention effectiveness, and calibration
- **Application:** 10-year atherosclerotic cardiovascular disease risk for statin initiation
- **Finding:** Standard learning + proper threshold selection > fairness-aware training objectives

**3. Decision curve analysis for personalized treatment choice (2202.02102v2)**
- **Authors:** Chalkou et al.
- **Key contribution:** Extends DCA to network meta-analysis with multiple treatment options
- **Method:** Compares personalized vs. one-size-fits-all strategies across threshold ranges
- **Application:** Relapsing-remitting multiple sclerosis treatment selection
- **Finding:** Personalized models show variable advantage depending on threshold values

### Threshold Optimization and Selection

**4. Aligning Evaluation with Clinical Priorities (2506.14540v3)**
- **Authors:** Flores et al.
- **Key contribution:** Proposes evaluation framework accounting for uncertainty in prevalence and cost asymmetries
- **Method:** Adjusted cross-entropy that averages cost-weighted performance over clinically relevant class balance ranges
- **Technical basis:** Proper scoring rules and Schervish representation
- **Impact:** Prioritizes calibrated, robust models over pure accuracy maximizers

**5. Performance evaluation of predictive AI models (2412.10288v1)**
- **Authors:** Van Calster et al.
- **Key contribution:** Comprehensive guidance on performance measures for medical AI
- **Recommendations:** AUROC, calibration plot, net benefit with DCA, probability distributions per outcome
- **Critical finding:** Classification measures (accuracy, F1) are improper for thresholds ≠0.5 or prevalence
- **Proper measures identified:** 17/32 measures exhibit both propriety and appropriate cost consideration

**6. Updating Clinical Risk Stratification Models (2308.05619v1)**
- **Authors:** Ötleş et al.
- **Key contribution:** Rank-based compatibility measure for model updates
- **Method:** Balances accuracy and uncertainty without requiring fixed decision thresholds
- **Application:** Mortality risk stratification in MIMIC dataset
- **Finding:** Enables compatible model updates while maintaining discriminative performance

### Threshold Selection Methods and Frameworks

**7. Joint Score-Threshold Optimization (2510.21934v1)**
- **Authors:** Gankhanloo et al.
- **Key contribution:** Mixed-integer programming framework for simultaneous weight and threshold optimization
- **Handles:** Partial supervision, asymmetric costs, minimum threshold gaps
- **Method:** CSO relaxation with softplus losses preserving ordinal structure
- **Application:** Risk assessment with intervention-censored outcomes

**8. Machine learning approaches to identify thresholds (2110.10625v1)**
- **Authors:** Masselot et al.
- **Key contribution:** Data-driven threshold establishment for heat-health warning systems
- **Methods:** MOB, MARS, PRIM, AIM algorithms
- **Finding:** PRIM (Patient Rule-Induction Method) most reliable with low variability
- **Application:** Heat-health alerts with exposure-response relationships

### Clinical Applications and Case Studies

**9. MCU-Net: Uncertainty representations for patient referrals (2007.03995v3)**
- **Authors:** Seedat
- **Key contribution:** Framework for uncertainty-based automated referral with adaptive thresholds
- **Method:** Monte Carlo Dropout + epistemic uncertainty + uncertainty threshold tuning
- **Application:** Medical image segmentation with human-in-the-loop
- **Finding:** Maximizes automated performance while referring truly uncertain cases

**10. Assisting clinical practice with fuzzy probabilistic decision trees (2304.07788v2)**
- **Authors:** Ambags et al.
- **Key contribution:** Combines probabilistic trees with fuzzy logic for interpretable decisions
- **Method:** Fuzzy variables avoid crisp threshold limitations of traditional decision trees
- **Application:** Thyroid nodule classification, chronic kidney disease progression
- **Finding:** Fuzzy approach preserves nuances lost with crisp thresholds

**11. Development and validation of prognostic model for MS (2105.06941v2)**
- **Authors:** Chalkou et al.
- **Key contribution:** Prognostic model with decision curve analysis for clinical utility assessment
- **Method:** 8 baseline factors, web app for personalized probability calculation
- **Finding:** Clinically useful between 15-30% threshold probability range
- **Application:** Relapse prediction in relapsing-remitting MS

**12. Designing quasi-experiment for adaptive risk prediction (2510.25052v2)**
- **Authors:** Odeh-Couvertier et al.
- **Key contribution:** Framework accommodating adaptation in both model and risk threshold
- **Method:** Regression discontinuity with counterfactual risks
- **Application:** Cardiovascular prevention programs with evolving thresholds
- **Finding:** Optimal threshold selection guided by operational/clinical targets

### Cost-Sensitive Learning

**13. Opportunistic Learning: Budgeted Cost-Sensitive Learning (1901.00243v2)**
- **Authors:** Kachuee et al.
- **Key contribution:** Context-aware feature-value function for cost-sensitive acquisition
- **Method:** Reinforcement learning with MC dropout for uncertainty-based utility
- **Application:** Diabetes classification with feature acquisition costs
- **Finding:** Efficient feature acquisition while maintaining accuracy

**14. Cost-Sensitive Diagnosis and Learning (1902.07102v2)**
- **Authors:** Kachuee et al.
- **Key contribution:** Feature cost assignment method based on inconvenience levels
- **Method:** Comparison of sensitivity-based and RL-based cost-sensitive approaches
- **Application:** Diabetes, heart disease, hypertension classification
- **Dataset:** Provides health dataset with feature costs

**15. Cost-Sensitive Active Learning for Intracranial Hemorrhage (1809.02882v1)**
- **Authors:** Kuo et al.
- **Key contribution:** Models labeling time and optimizes return on investment
- **Method:** Ensemble with core-set selection based on annotation cost
- **Application:** Intracranial hemorrhage detection and segmentation
- **Finding:** Accounts for variable labeling time across examples

**16. Robust COVID-19 Screening via Discriminative Cost-Sensitive Learning (2004.12592v2)**
- **Authors:** Li et al.
- **Key contribution:** Combines fine-grained classification with adaptive cost amplification
- **Method:** Conditional center loss + score-level cost-sensitive learning
- **Application:** COVID-19 vs. bacterial/viral pneumonia vs. healthy
- **Finding:** 97.01% accuracy with amplified cost for COVID-19 misclassification

**17. Gastrointestinal Disease Classification (2307.07603v1)**
- **Authors:** Nath & Shahariar
- **Key contribution:** Cost-sensitive pre-trained CNNs with supervised contrastive learning
- **Method:** Assigns distinct costs to misclassifications based on disease severity
- **Application:** Hyper-Kvasir gastrointestinal disease dataset
- **Finding:** Prioritizes accurate classification of critical conditions

### Calibration and Threshold Interaction

**18. Does calibration improve performance in class-imbalanced medical images? (2110.00918v3)**
- **Authors:** Rajaratnam et al.
- **Key contribution:** Systematic analysis of calibration effects on imbalanced datasets
- **Method:** Compares calibrated vs. uncalibrated at default (0.5) and PR-guided thresholds
- **Finding:** At 0.5 threshold, calibration significantly improves performance; at optimal threshold, gains not significant
- **Application:** Chest X-rays and fundus images

**19. Lightweight Baselines for Medical Abstract Classification (2510.10025v2)**
- **Authors:** Liu et al.
- **Key contribution:** Post-hoc operating point selection substantially improves performance
- **Method:** Validation-calibrated, class-wise thresholds after standard training
- **Finding:** Compact encoder + CE + calibration/thresholding outperforms complex objectives
- **Application:** Medical abstract classification

**20. Calibrating the Dice loss (2111.00528v2)**
- **Authors:** Yeung et al.
- **Key contribution:** DSC++ loss improves calibration for segmentation
- **Method:** Modulates penalty for overconfident incorrect predictions
- **Finding:** Well-calibrated outputs enable softmax thresholding for recall-precision tuning
- **Application:** Biomedical image segmentation

**21. A Possibility in Algorithmic Fairness (2002.07676v3)**
- **Authors:** Reich & Vijaykumar
- **Key contribution:** Derives conditions for calibrated scores achieving equal error rates
- **Method:** Group-blind threshold achieving calibration + error rate equality
- **Application:** COMPAS criminal risk, credit lending
- **Finding:** Can eliminate error disparities while maintaining calibration

### Specialized Applications

**22. AI-Assisted Decision-Making for Auto-Segmented Contour Quality (2505.00308v2)**
- **Authors:** Wang et al.
- **Key contribution:** Bayesian ordinal classification with calibrated uncertainty thresholds
- **Method:** Optimizes uncertainty thresholds meeting clinical accuracy requirements
- **Application:** Radiotherapy contour quality assessment
- **Finding:** 93%+ contours accurately predicted in 98%+ cases

**23. Conformal Lesion Segmentation for 3D Medical Images (2510.17897v1)**
- **Authors:** Tan et al.
- **Key contribution:** Risk-constrained framework ensuring test-time FNR below tolerance
- **Method:** Conformalization to calibrate data-driven thresholds via FNR-specific loss
- **Application:** 3D lesion segmentation across 6 datasets
- **Finding:** Rigorous FNR constraint while maximizing precision

**24. Statistical Management of False Discovery Rate (2504.04482v2)**
- **Authors:** Dai et al.
- **Key contribution:** Risk-aware dynamic threshold mechanism for instance segmentation
- **Method:** Calibration-aware loss function tuning threshold based on risk level α
- **Application:** Medical instance segmentation (lesions, tumors)
- **Finding:** Rigorously bounds FDR metric via calibration framework

**25. Expert load matters (2308.05035v2)**
- **Authors:** Sangalli et al.
- **Key contribution:** Maximizes area under confidence operating characteristic (COC) curve
- **Method:** Balances model accuracy with samples delegated to experts
- **Application:** Computer vision and medical image classification
- **Finding:** Improves accuracy while reducing expert delegation

**26. "It depends": Configuring AI to Improve Clinical Usefulness (2407.11978v1)**
- **Authors:** Zając et al.
- **Key contribution:** Identifies four technical dimensions requiring context configuration
- **Dimensions:** AI functionality, medical focus, decision threshold, explainability
- **Application:** Radiology across Denmark and Kenya clinical sites
- **Finding:** Threshold must be configured to clinical context, user expertise, patient context

### Hidden Challenges and Practical Considerations

**27. Hidden Stratification Causes Meaningful Failures (1909.12475v2)**
- **Authors:** Oakden-Rayner et al.
- **Key contribution:** Identifies performance differences of 20%+ on clinically important subsets
- **Method:** Analyzes effects on unidentified low-prevalence, low-quality label subsets
- **Finding:** Evaluation of hidden stratification critical for deployment
- **Application:** Cancer detection across medical imaging datasets

**28. Stronger Baseline Models for ML Research Alignment (2409.12116v1)**
- **Authors:** Wolfrath et al.
- **Key contribution:** Demonstrates importance of strong baselines in healthcare ML
- **Method:** Compares against optimized baselines rather than weak defaults
- **Finding:** Weak baselines obscure true value of proposed ML methods
- **Application:** Healthcare risk prediction across multiple datasets

**29. What is Interpretable? Machine Learning for Decision-Support (1811.10799v2)**
- **Authors:** Lahav et al.
- **Key contribution:** Shows ML experts cannot predict optimal outputs for clinician confidence
- **Method:** RL-based DSS learning from user interactions
- **Application:** Heart failure risk assessment
- **Finding:** Optimal threshold selection should be guided by specific application

### Emerging Perspectives

**30. Dual-criterion designs with statistical significance and clinical relevance (1908.07751v1)**
- **Authors:** Roychoudhury et al.
- **Key contribution:** Complements statistical significance with clinical relevance threshold
- **Method:** Effect estimate must pass clinically motivated decision value
- **Application:** Phase II clinical trials
- **Finding:** Improves evidence-based GO/NO-GO decisions

**31. ACCEPT analyses for clinical trial interpretation (2203.11164v2)**
- **Authors:** Clements et al.
- **Key contribution:** Acceptability Curve using Probability Above Threshold
- **Method:** Harmonizes reporting across trials, acknowledging valid interpretations
- **Application:** Clinical trial comparison and interpretation
- **Finding:** Moves focus from pre-specified values to interpretation of trial data

---

## Threshold Selection Methods: Taxonomy and Approaches

### 1. Clinical Utility-Based Methods

**Decision Curve Analysis (DCA)**
- **Principle:** Calculates net benefit = (TP/n) - (FP/n) × (pt/(1-pt))
- **Threshold range:** Evaluated across clinically plausible probability thresholds
- **Advantages:** Explicitly incorporates harm-to-benefit ratio
- **Key papers:** 2412.07882v1, 2202.01906v1, 2202.02102v2

**Continuous Net Benefit**
- **Extension:** Weighted area under rescaled net benefit curve
- **Use case:** Populations with range of optimal thresholds
- **Advantage:** Handles heterogeneous treatment benefits
- **Key paper:** 2412.07882v1

**Cost-Sensitive Threshold Optimization**
- **Principle:** Explicit modeling of misclassification costs
- **Methods:** Asymmetric loss functions, cost matrices, ROI optimization
- **Applications:** Variable annotation costs, differential error severity
- **Key papers:** 1901.00243v2, 1902.07102v2, 1809.02882v1, 2004.12592v2

### 2. Statistical Approaches

**Precision-Recall Guided Thresholds**
- **Principle:** Optimize F1, F-beta, or custom PR trade-off
- **Finding:** Often superior to default 0.5 threshold
- **Limitation:** May not reflect clinical utilities
- **Key paper:** 2110.00918v3

**Youden Index**
- **Principle:** Maximizes sensitivity + specificity - 1
- **Advantage:** Simple, well-established
- **Limitation:** Assumes equal importance of sensitivity/specificity
- **Context:** Appropriate when costs are symmetric

**Calibration-Based Methods**
- **Principle:** Ensure predicted probabilities match observed frequencies
- **Enables:** Post-hoc threshold adjustment without retraining
- **Methods:** Platt scaling, isotonic regression, temperature scaling
- **Key papers:** 2110.00918v3, 2111.00528v2, 2202.01906v1

### 3. Conformal and Uncertainty-Based Methods

**Conformal Prediction**
- **Principle:** Statistical guarantees on test-time error rates
- **Method:** Calibration set determines threshold satisfying error tolerance
- **Advantages:** Distribution-free, finite-sample guarantees
- **Key papers:** 2510.17897v1, 2504.04482v2

**Uncertainty-Aware Thresholds**
- **Principle:** Reject predictions with high uncertainty
- **Methods:** MC dropout, Bayesian approaches, ensemble disagreement
- **Application:** Human-in-loop systems, critical decisions
- **Key papers:** 2007.03995v3, 2505.00308v2, 2510.00029v1

**Bayesian Ordinal Classification**
- **Principle:** Quantify prediction uncertainty, calibrate acceptance thresholds
- **Advantage:** Confidence-based quality control without ground truth
- **Application:** Auto-contour quality assessment in radiotherapy
- **Key paper:** 2505.00308v2

### 4. Data-Driven Learning Methods

**Reinforcement Learning**
- **Principle:** Learn threshold policy through interaction
- **Advantage:** Adapts to context and feedback
- **Application:** Feature acquisition, sequential decision-making
- **Key papers:** 1901.00243v2, 1811.10799v2

**Patient Rule-Induction Method (PRIM)**
- **Principle:** Find regions with high outcome rates
- **Advantage:** Low variability, reliable across scenarios
- **Application:** Heat-health warning thresholds
- **Key paper:** 2110.10625v1

**Joint Optimization (Score + Threshold)**
- **Principle:** Simultaneously optimize model weights and decision thresholds
- **Method:** Mixed-integer programming with ordinal constraints
- **Advantage:** Globally optimal solution
- **Key paper:** 2510.21934v1

### 5. Post-Hoc Selection Methods

**Validation Set Calibration**
- **Principle:** Use held-out validation set to select optimal threshold
- **Methods:** Grid search, class-wise thresholds, ROC-based selection
- **Advantage:** Simple, effective, model-agnostic
- **Key paper:** 2510.10025v2

**Operating Point Selection**
- **Principle:** Choose threshold maximizing metric of interest
- **Consideration:** Avoid overfitting to validation set
- **Best practice:** Use separate calibration set if possible
- **Key paper:** 2510.10025v2

### 6. Context-Adaptive Methods

**Risk-Level Dependent Thresholds**
- **Principle:** Different thresholds for different risk strata
- **Rationale:** High-risk patients may warrant different sensitivity/specificity balance
- **Application:** Tiered care pathways
- **Key paper:** 2510.25052v2

**Multi-Threshold Systems**
- **Principle:** Multiple decision boundaries for different actions
- **Example:** Screen negative, needs monitoring, refer for intervention
- **Advantage:** Richer decision space than binary classification
- **Key paper:** 2510.21934v1

---

## Clinical Utility Frameworks

### Decision Curve Analysis (DCA) Framework

**Core Concept:**
Net Benefit = (True Positives / n) - (False Positives / n) × [pt / (1 - pt)]

Where pt is the probability threshold reflecting harm-to-benefit ratio of intervention.

**Interpretation:**
- **Net benefit > 0:** Model better than treating none
- **Net benefit > (prevalence):** Model better than treating all
- **Maximum net benefit:** Optimal threshold for given clinical context

**Comparison Strategies:**
- Treat all (assumes all patients benefit from intervention)
- Treat none (assumes no patients benefit from intervention)
- Alternative models
- Alternative threshold choices

**Key Extensions:**
1. **Continuous Net Benefit (2412.07882v1):** Weighted area under net benefit curve for heterogeneous populations
2. **Multi-treatment DCA (2202.02102v2):** Network meta-analysis with multiple treatment options
3. **ACCEPT Analysis (2203.11164v2):** Acceptability curves showing probability intervention is cost-effective

### Cost-Sensitive Learning Framework

**Misclassification Cost Matrix:**
```
                Predicted Negative    Predicted Positive
Actual Negative        0                    C_FP
Actual Positive      C_FN                    0
```

**Expected Cost:**
E[Cost] = C_FN × (FN rate) + C_FP × (FP rate)

**Optimal Threshold:**
pt* = C_FP / (C_FP + C_FN)

**Applications:**
- **Screening programs:** High C_FN (missed disease), lower C_FP (unnecessary followup)
- **Rule-out tests:** Lower C_FN tolerated, minimize C_FP
- **Rare events:** C_FN >> C_FP due to disease rarity and severity

**Implementation Approaches:**
1. **Training-time:** Weighted loss functions, class weights, focal loss
2. **Threshold-time:** Post-hoc threshold selection based on cost matrix
3. **Hybrid:** Cost-aware training + threshold tuning (2506.14540v3)

### Calibration-First Framework

**Principle:** Well-calibrated probabilities enable flexible threshold selection

**Calibration Methods:**
1. **Platt Scaling:** Logistic regression on validation set
2. **Isotonic Regression:** Non-parametric monotonic mapping
3. **Temperature Scaling:** Single parameter scaling for neural networks
4. **Beta Calibration:** Flexible 3-parameter family

**Threshold Selection After Calibration:**
- Choose threshold based on clinical utility, costs, or constraints
- Adjust threshold post-deployment without retraining
- Different thresholds for different subpopulations or settings

**Evidence:**
- Calibration + threshold selection > fairness-constrained training (2202.01906v1)
- At optimal threshold, calibration gains less pronounced but still valuable (2110.00918v3)
- Enables softmax thresholding for recall-precision balance (2111.00528v2)

### Risk-Constrained Framework

**Principle:** Ensure error rate constraint while maximizing utility

**Common Constraints:**
- False Negative Rate (FNR) ≤ ε
- False Discovery Rate (FDR) ≤ α
- Expected cost ≤ budget

**Methods:**
1. **Conformal Prediction (2510.17897v1):**
   - Use calibration set to find threshold satisfying constraint
   - Provides statistical guarantees on test set
   - Works with any model producing probabilities

2. **Calibration-Aware Loss (2504.04482v2):**
   - Loss function dynamically tunes threshold based on risk level
   - Bounds FDR with high probability
   - Compatible with mainstream architectures

3. **Uncertainty Thresholding (2505.00308v2):**
   - Reject predictions with uncertainty above threshold
   - Calibrate rejection threshold on validation set
   - Reduces false positives while maintaining coverage

### Multi-Stakeholder Framework

**Recognition:** Different stakeholders have different utilities

**Stakeholder Types:**
- **Patients:** Personal risk tolerance, quality of life considerations
- **Clinicians:** Diagnostic accuracy, workload, liability
- **Health systems:** Cost-effectiveness, resource allocation, population health
- **Payers:** Treatment costs, long-term outcomes

**Approaches:**
1. **Configurable Thresholds:** Allow different thresholds for different contexts (2407.11978v1)
2. **Preference Elicitation:** Learn stakeholder utilities through interaction (1811.10799v2)
3. **Pareto Frontiers:** Present trade-off curves for stakeholder choice
4. **Multi-Objective Optimization:** Balance competing objectives with constraints

---

## Operating Point Optimization Strategies

### Single Metric Optimization

**Youden Index Maximization:**
- **Objective:** Max(Sensitivity + Specificity - 1)
- **Optimal when:** Equal importance to sensitivity and specificity
- **Limitation:** Doesn't account for prevalence or costs
- **Use case:** Symmetric error costs, balanced datasets

**F1-Score Maximization:**
- **Objective:** Max(2 × Precision × Recall / (Precision + Recall))
- **Optimal when:** Precision and recall equally important
- **Limitation:** Ignores true negatives, improper measure
- **Finding:** Not recommended for clinical use (2412.10288v1)

**F-beta Score Optimization:**
- **Objective:** Max((1 + β²) × Precision × Recall / (β² × Precision + Recall))
- **Advantage:** Weights recall β times more than precision
- **Use case:** When recall more important than precision (β > 1) or vice versa

### Cost-Based Optimization

**Minimum Expected Cost:**
- **Objective:** Min(C_FN × FNR + C_FP × FPR)
- **Requires:** Explicit cost estimates
- **Optimal threshold:** pt* = C_FP / (C_FP + C_FN)
- **Applications:** Well-defined monetary or utility costs

**Net Benefit Maximization:**
- **Objective:** Max((TP/n) - (FP/n) × (pt/(1-pt)))
- **Advantage:** Transparently reflects harm-to-benefit ratio
- **Clinical interpretation:** Threshold represents "odds at which intervention worthwhile"
- **Recommended:** Primary metric for clinical AI (2412.10288v1)

**Return on Investment (ROI):**
- **Objective:** Maximize benefit per unit cost
- **Consideration:** Variable costs (e.g., labeling time)
- **Application:** Active learning, resource allocation (1809.02882v1)

### Constraint-Based Optimization

**Sensitivity-Constrained Specificity:**
- **Objective:** Max(Specificity) subject to Sensitivity ≥ S_min
- **Use case:** Screening (can't miss disease), rule-out tests
- **Example:** COVID screening requiring 97% sensitivity

**Specificity-Constrained Sensitivity:**
- **Objective:** Max(Sensitivity) subject to Specificity ≥ P_min
- **Use case:** Confirmatory tests, rule-in diagnostics
- **Example:** Diagnostic confirmation requiring 95% specificity

**FNR/FDR Constrained:**
- **Objective:** Maximize utility subject to error rate ≤ ε
- **Methods:** Conformal prediction (2510.17897v1), calibration (2504.04482v2)
- **Advantage:** Rigorous statistical guarantees
- **Application:** Safety-critical applications

**Resource-Constrained:**
- **Objective:** Maximize performance subject to expert load, cost, or capacity
- **Example:** Maximize detection while keeping referrals ≤10% (2308.05035v2)
- **Consideration:** Balances automation with human oversight

### Multi-Objective Optimization

**Pareto Optimization:**
- **Approach:** Identify operating points where improving one metric degrades another
- **Presentation:** Pareto frontier for stakeholder selection
- **Advantage:** Transparent trade-offs without pre-committing to weights

**Weighted Sum:**
- **Objective:** Max(w₁ × Metric₁ + w₂ × Metric₂ + ...)
- **Challenge:** Weight selection often arbitrary
- **Alternative:** Vary weights, present results across range

**Lexicographic Optimization:**
- **Approach:** Optimize primary objective, then secondary within tolerance
- **Example:** Maximize sensitivity, then specificity among solutions with sensitivity ≥ 95%
- **Advantage:** Clear priority structure

### Adaptive and Dynamic Optimization

**Context-Dependent Thresholds:**
- **Principle:** Different thresholds for different patient subgroups or settings
- **Example:** Higher sensitivity threshold for high-risk patients
- **Implementation:** Learn threshold as function of context features
- **Paper:** 2510.25052v2 (adaptive thresholds over time)

**Uncertainty-Based Rejection:**
- **Principle:** Apply threshold only to confident predictions, refer uncertain cases
- **Optimization:** Balance automated decisions vs. expert referrals (2308.05035v2)
- **Threshold choice:** Based on uncertainty calibration (2505.00308v2)
- **Application:** Human-in-loop clinical decision support

**Reinforcement Learning:**
- **Approach:** Learn threshold policy through interaction and feedback
- **Advantage:** Adapts to deployment environment
- **Challenge:** Requires online learning infrastructure
- **Paper:** 1811.10799v2 (learning from clinician interactions)

---

## Cost-Sensitive Learning in Healthcare

### Types of Costs in Clinical Settings

**1. Misclassification Costs**
- **False Negatives:** Missed diagnoses, delayed treatment, disease progression
- **False Positives:** Unnecessary interventions, patient anxiety, healthcare resource waste
- **Asymmetry:** Typically C_FN >> C_FP in screening; varies by application

**2. Feature Acquisition Costs**
- **Monetary:** Lab tests, imaging procedures, genetic testing
- **Patient burden:** Discomfort, invasiveness, time
- **Privacy:** Sensitivity of information requested
- **Paper:** 1902.07102v2 (feature cost assignment methodology)

**3. Labeling Costs**
- **Expert time:** Variable across examples (simple vs. complex cases)
- **Quality:** Inter-rater variability, expertise required
- **ROI optimization:** Core-set selection based on annotation cost (1809.02882v1)

**4. Intervention Costs**
- **Treatment:** Medication, procedures, monitoring
- **Side effects:** Adverse events, quality of life impact
- **Opportunity cost:** Resources not available for other patients

### Cost-Sensitive Training Methods

**1. Class Weighting**
- **Method:** Weight loss by inverse class frequency or cost ratio
- **Formula:** w_pos = C_FN / (C_FN + C_FP), w_neg = C_FP / (C_FN + C_FP)
- **Advantage:** Simple, widely supported
- **Limitation:** Assumes costs constant across examples

**2. Focal Loss**
- **Method:** Down-weight easy examples, focus on hard examples
- **Formula:** FL(pt) = -α(1-pt)^γ log(pt)
- **Finding:** Benefits from post-hoc threshold calibration (2510.10025v2)
- **Application:** Imbalanced medical imaging datasets

**3. Cost-Sensitive Loss Functions**
- **Conditional Center Loss:** Learn discriminative representations (2004.12592v2)
- **Score-Level Cost-Sensitive:** Adaptively enlarge misclassification cost (2004.12592v2)
- **Calibration-Aware Loss:** Dynamically tune threshold based on risk level (2504.04482v2)

**4. Adversarial Data Augmentation**
- **Method:** Generate targeted adversarial examples pushing boundary in cost-aware directions
- **Principle:** Maximize probability of critical misclassifications during training
- **Application:** Conservative decisions on costly pairs (2208.11739v1)
- **Paper:** Rethinking Cost-sensitive Classification via Adversarial Data Augmentation

### Cost-Sensitive Threshold Selection

**Optimal Threshold Formula:**
When costs are known: pt* = C_FP / (C_FP + C_FN)

**Derivation:**
Minimize expected cost:
E[Cost] = C_FN × FNR + C_FP × FPR

At threshold pt:
- Classify positive if P(y=1|x) ≥ pt
- Optimal pt occurs when: C_FP × P(y=0|x) = C_FN × P(y=1|x)
- Solving: pt* = C_FP / (C_FP + C_FN)

**Example:**
- C_FN (missed cancer) = $100,000 (treatment delay, worse outcomes)
- C_FP (unnecessary biopsy) = $1,000
- Optimal threshold: pt* = 1,000 / (1,000 + 100,000) = 0.01 (1%)

**This means:** Intervene if cancer probability ≥ 1%, reflecting 100:1 cost ratio

### Feature Acquisition Under Budget Constraints

**Problem Formulation:**
- Each feature has acquisition cost c_i
- Total budget B
- Goal: Select features maximizing prediction utility while respecting budget

**Approaches:**

**1. Greedy Sequential Selection**
- **Method:** Select next feature maximizing utility/cost ratio
- **Advantage:** Simple, online-compatible
- **Limitation:** May be suboptimal

**2. Reinforcement Learning**
- **State:** Features acquired so far
- **Action:** Which feature to acquire next (or stop)
- **Reward:** Terminal prediction utility minus total cost
- **Papers:** 1901.00243v2 (opportunistic learning), 1902.07102v2

**3. Value of Information**
- **Principle:** Acquire feature if expected improvement > cost
- **Method:** MC dropout to measure uncertainty reduction
- **Paper:** 1901.00243v2 (context-aware feature-value function)

### Applications in Healthcare

**COVID-19 Screening (2004.12592v2)**
- **Challenge:** Extremely high cost of false negatives (missed COVID cases)
- **Method:** Discriminative cost-sensitive learning with adaptive cost amplification
- **Result:** 97.01% accuracy with minimized COVID misclassification

**Diabetes Classification (1901.00243v2, 1902.07102v2)**
- **Cost considerations:** Feature acquisition (tests), misclassification
- **Method:** RL-based feature acquisition with uncertainty-based utility
- **Result:** Efficient feature selection maintaining accuracy

**Intracranial Hemorrhage (1809.02882v1)**
- **Variable cost:** Labeling time varies tremendously across cases
- **Method:** ROI optimization with ensemble and core-set selection
- **Result:** Efficient labeling budget allocation

**Gastrointestinal Disease (2307.07603v1)**
- **Cost assignment:** Based on disease severity and treatment criticality
- **Method:** Cost-sensitive CNNs with supervised contrastive learning
- **Result:** Prioritizes accurate classification of critical conditions

---

## Research Gaps and Future Directions

### 1. Threshold Selection Under Distributional Shift

**Current Gap:**
- Most methods assume test distribution similar to validation
- Real deployments face temporal drift, site differences, population changes
- Threshold robustness to shift poorly understood

**Future Research Needs:**
- Methods for monitoring threshold appropriateness post-deployment
- Adaptive threshold updating under distribution shift
- Robust threshold selection minimizing worst-case performance across shifts
- Relevant paper: 2506.14540v3 (accounts for uncertainty in prevalence)

### 2. Multi-Stakeholder Threshold Negotiation

**Current Gap:**
- Most work assumes single objective or pre-specified weights
- Real healthcare involves competing stakeholder interests
- Limited research on preference elicitation and aggregation

**Future Research Needs:**
- Interactive threshold selection incorporating stakeholder preferences
- Methods for identifying and resolving threshold conflicts
- Dynamic threshold adjustment based on resource availability
- Relevant paper: 1811.10799v2 (learning from user interactions)

### 3. Temporal Dynamics and Sequential Decisions

**Current Gap:**
- Most research on single time-point prediction
- Limited work on threshold selection for sequential screening or monitoring
- Unclear how to optimize thresholds for decision sequences

**Future Research Needs:**
- Threshold policies for longitudinal prediction models
- Optimal stopping rules incorporating future opportunity costs
- Time-varying thresholds adapting to disease progression
- Relevant paper: 2510.25052v2 (adaptive thresholds in time-series)

### 4. Uncertainty Quantification and Threshold Selection

**Current Gap:**
- Growing focus on uncertainty quantification
- Unclear how to optimally combine aleatoric and epistemic uncertainty for thresholding
- Limited guidance on rejection mechanisms

**Future Research Needs:**
- Unified framework for uncertainty-aware threshold selection
- Methods determining when to reject vs. predict with low confidence
- Calibration of uncertainty estimates specifically for threshold-based decisions
- Relevant papers: 2007.03995v3, 2505.00308v2, 2510.00029v1

### 5. Fairness and Threshold Selection

**Current Gap:**
- Tension between group fairness and individual utility
- Unclear whether group-specific vs. universal thresholds appropriate
- Limited research on fair threshold selection under cost asymmetries

**Future Research Needs:**
- Principled approaches to group-specific threshold selection
- Methods balancing fairness constraints with clinical utility
- Understanding when universal vs. tailored thresholds preferable
- Relevant paper: 2202.01906v1 (calibration + threshold > fairness constraints)

### 6. Explainability of Threshold Decisions

**Current Gap:**
- Threshold selection often opaque to clinicians
- Limited tools for explaining why specific threshold chosen
- Difficulty communicating threshold implications to patients

**Future Research Needs:**
- Interpretable threshold selection methods
- Tools for communicating threshold choice rationale
- Patient-facing explanations of risk thresholds
- Relevant paper: 2304.07788v2 (fuzzy probabilistic trees for interpretability)

### 7. Real-World Threshold Validation

**Current Gap:**
- Most validation on historical datasets
- Limited prospective validation of threshold selection methods
- Unclear real-world impact of optimal vs. suboptimal thresholds

**Future Research Needs:**
- Prospective clinical trials comparing threshold selection methods
- Real-world implementation studies with clinical outcome data
- Cost-effectiveness analyses of threshold optimization
- Guidelines for threshold validation before deployment

### 8. Automated Threshold Monitoring and Updating

**Current Gap:**
- Manual threshold selection and updating
- Limited automation of threshold maintenance
- Unclear when threshold recalibration needed

**Future Research Needs:**
- Automated monitoring for threshold degradation
- Trigger conditions for threshold recalibration
- Online threshold learning and adaptation
- Safe deployment practices for threshold updates

### 9. Integration with Clinical Workflows

**Current Gap:**
- Threshold selection research often disconnected from clinical practice
- Limited understanding of threshold decision points in care pathways
- Poor integration with existing clinical decision tools

**Future Research Needs:**
- Workflow analysis identifying threshold decision points
- Integration with existing clinical guidelines and pathways
- User interface design for threshold-based decision support
- Relevant paper: 2407.11978v1 (configuring AI for clinical context)

### 10. Cost Estimation and Elicitation

**Current Gap:**
- Most cost-sensitive methods assume costs are known
- Limited guidance on eliciting costs from clinicians
- Uncertain how to handle cost variability across patients and settings

**Future Research Needs:**
- Methods for eliciting misclassification costs from stakeholders
- Approaches handling cost uncertainty and variability
- Sensitivity analysis for threshold decisions under cost uncertainty
- Relevant paper: 1902.07102v2 (feature cost assignment methodology)

---

## Relevance to ED Threshold Tuning

### Direct Applications to Emergency Department Settings

**1. Triage Decision Support**
- **Challenge:** Balance false alarms vs. missed emergencies under capacity constraints
- **Relevant methods:** Net benefit optimization, constraint-based threshold selection
- **Key papers:** 2412.07882v1 (capacity constraints), 2308.05035v2 (expert load)
- **Recommendation:** Use DCA with ED-specific cost ratios (wait time, resource utilization)

**2. Sepsis Prediction**
- **Challenge:** High cost of missed sepsis, high volume of alerts
- **Relevant methods:** FNR-constrained threshold selection, uncertainty-based rejection
- **Key paper:** 2402.03486v1 (sepsis onset prediction with threshold optimization)
- **Recommendation:** Set FNR constraint (e.g., ≤5%), optimize specificity within constraint

**3. Admission Prediction**
- **Challenge:** Optimize bed utilization, minimize unnecessary admissions
- **Relevant methods:** ROI optimization, cost-sensitive learning
- **Key papers:** 1903.09296v1 (patient clustering), 2007.04432v1 (collapsing bandits)
- **Recommendation:** Model costs of unnecessary admission vs. delayed admission

**4. Imaging Utilization**
- **Challenge:** Minimize unnecessary imaging while not missing critical findings
- **Relevant methods:** Feature acquisition under budget, value of information
- **Key papers:** 1901.00243v2, 1902.07102v2 (cost-sensitive feature acquisition)
- **Recommendation:** Sequential decision-making with imaging as costly feature

### Specific Threshold Selection Strategies for ED

**Shift-Specific Thresholds**
- **Rationale:** Different capacity constraints by time of day, day of week
- **Method:** Context-dependent thresholds based on current ED census, staffing
- **Implementation:** Real-time threshold adjustment using current state features
- **Relevant paper:** 2510.25052v2 (adaptive thresholds)

**Acuity-Stratified Thresholds**
- **Rationale:** Different cost trade-offs for high vs. low acuity patients
- **Method:** Separate thresholds by initial triage category
- **Implementation:** Multi-threshold system with acuity-specific decision boundaries
- **Relevant paper:** 2510.21934v1 (ordinal classification with multiple thresholds)

**Uncertainty-Based Clinician Escalation**
- **Rationale:** Refer uncertain cases to clinician review
- **Method:** Dual threshold system (high confidence automated, low confidence escalated)
- **Implementation:** Calibrate uncertainty threshold on validation set
- **Relevant papers:** 2007.03995v3, 2505.00308v2

**Resource-Aware Thresholds**
- **Rationale:** Adapt recommendations based on current resource availability
- **Method:** Dynamic threshold based on bed availability, specialist availability
- **Implementation:** Real-time threshold from current capacity state
- **Relevant paper:** 2412.07882v1 (healthcare capacity constraints)

### ED-Specific Cost Considerations

**Measurable Costs:**
- **Missed critical condition:** Poor outcomes, liability, longer LOS
- **Unnecessary tests/admission:** Direct costs, ED crowding, patient dissatisfaction
- **Delayed treatment:** Door-to-treatment time, outcomes degradation
- **False alarms:** Clinician alert fatigue, reduced system trust

**Methods for Cost Estimation:**
1. **Historical data analysis:** Outcomes by prediction category
2. **Expert elicitation:** Survey ED physicians on relative costs
3. **Simulation:** Model ED flow under different threshold policies
4. **A/B testing:** Prospective comparison of threshold strategies (with safety monitoring)

### Practical Implementation Roadmap for ED

**Phase 1: Baseline Establishment (Months 1-3)**
1. Collect retrospective ED data with outcomes
2. Train prediction models (admission, sepsis, imaging need, etc.)
3. Evaluate performance across operating points
4. Generate ROC curves, PR curves, decision curves

**Phase 2: Threshold Selection (Months 3-6)**
1. Elicit costs from ED stakeholders (physicians, nurses, administrators)
2. Calculate cost-optimal thresholds
3. Perform decision curve analysis across threshold range
4. Validate on held-out temporal cohort
5. Simulate ED flow under proposed thresholds

**Phase 3: Pilot Deployment (Months 6-9)**
1. Deploy in silent mode (predictions logged, not acted upon)
2. Monitor calibration drift, performance degradation
3. Collect clinician feedback on predictions and suggested actions
4. Analyze disagreements between model and clinicians
5. Refine thresholds based on observed costs and outcomes

**Phase 4: Active Deployment (Months 9-12)**
1. Enable decision support with selected thresholds
2. A/B test alternative threshold strategies (with safety monitoring)
3. Monitor key metrics (outcomes, utilization, clinician acceptance)
4. Collect data for threshold refinement
5. Establish trigger conditions for threshold recalibration

**Phase 5: Continuous Improvement (Ongoing)**
1. Automated monitoring for distribution shift
2. Periodic threshold recalibration (e.g., quarterly)
3. Adapt to seasonal patterns, operational changes
4. Expand to additional use cases
5. Share learnings with broader community

### Key Success Factors for ED Deployment

**1. Stakeholder Engagement**
- Involve ED physicians, nurses, administrators in threshold selection
- Communicate threshold rationale and trade-offs clearly
- Provide override mechanisms with feedback loops
- Relevant paper: 1811.10799v2 (learning from user interactions)

**2. Workflow Integration**
- Embed threshold-based recommendations in existing EHR
- Minimize clicks and cognitive load
- Provide actionable recommendations, not just probabilities
- Relevant paper: 2407.11978v1 (configuring AI for clinical context)

**3. Safety Monitoring**
- Implement guardrails (e.g., FNR constraints for critical conditions)
- Monitor for subgroup performance disparities
- Establish rapid response protocols for detected issues
- Relevant papers: 2510.17897v1 (FNR constraints), 1909.12475v2 (hidden stratification)

**4. Transparency and Explainability**
- Communicate threshold selection basis (costs, utilities, constraints)
- Provide case-level explanations for recommendations
- Regular reporting on system performance and outcomes
- Relevant paper: 2304.07788v2 (interpretable fuzzy decision trees)

**5. Adaptation and Learning**
- Treat deployment as learning opportunity
- Collect data on threshold effectiveness
- Be prepared to adjust thresholds based on evidence
- Relevant paper: 2510.25052v2 (adaptive risk prediction models)

---

## Conclusion

Effective clinical AI deployment requires moving beyond pure predictive accuracy to explicitly modeling clinical decision-making through appropriate threshold selection. Key takeaways:

1. **No single "best" threshold:** Optimal operating point depends on clinical context, costs, constraints, and stakeholder preferences

2. **Calibration is foundational:** Well-calibrated probabilities enable flexible, post-deployment threshold adjustment

3. **Decision curve analysis is essential:** DCA provides actionable framework for evaluating clinical utility across threshold ranges

4. **Cost-sensitive methods matter:** Asymmetric misclassification costs must be explicitly modeled, not assumed equal

5. **Context-aware thresholds:** Different settings, populations, or resource states may require different operating points

6. **Uncertainty enables intelligence:** Uncertainty-based rejection allows human-AI collaboration at appropriate boundaries

7. **Validation beyond accuracy:** Evaluate threshold-based decisions on clinical outcomes, not just statistical metrics

For ED applications specifically, threshold selection should account for:
- Capacity constraints and resource availability
- Shift-specific and seasonal patterns
- Acuity-stratified decision-making
- Alert fatigue and clinician trust
- Real-time adaptability

The research literature provides robust methodological foundations for threshold optimization in clinical settings. Successful implementation requires stakeholder engagement, workflow integration, safety monitoring, and continuous learning.

---

## References

All papers cited are available on ArXiv. ArXiv IDs are provided in paper titles throughout this document. Access papers at: https://arxiv.org/abs/[ArXiv-ID]

**Total Papers Reviewed:** 100+
**Primary Focus Areas:** Decision thresholds, operating point selection, clinical utility, cost-sensitive learning, calibration
**Domains:** Medical imaging, clinical prediction, emergency care, chronic disease management
**Date Range:** 2005-2025

---

**Document prepared for:** Hybrid Reasoning for Acute Care Research
**Last updated:** December 1, 2025
**Prepared by:** AI Research Assistant via ArXiv systematic review