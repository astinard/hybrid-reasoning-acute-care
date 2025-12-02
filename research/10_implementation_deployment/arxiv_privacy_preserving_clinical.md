# Privacy-Preserving Machine Learning in Healthcare: A Synthesis of Recent Research

**Document Created:** November 30, 2025
**Focus Areas:** Differential Privacy, DP-SGD, Federated Learning, Secure Computation, Privacy-Utility Tradeoffs

---

## Executive Summary

This synthesis reviews cutting-edge research on privacy-preserving machine learning in clinical settings, with particular emphasis on differential privacy (DP), DP-SGD (Differentially Private Stochastic Gradient Descent), secure multi-party computation, and federated learning approaches. The analysis reveals critical insights into privacy-utility tradeoffs and practical implementation strategies for healthcare AI systems.

**Key Findings:**
- DP-SGD with adaptive noise allocation achieves strong privacy (ε ≈ 9.0) while maintaining clinical utility (>77% recall)
- Compliance-aware federated learning can improve accuracy by 1-15% over traditional approaches
- Chebyshev polynomial approximations enable privacy preservation in deep belief networks with bounded error
- Robust aggregation methods successfully detect and discard faulty clients while maintaining model performance
- Layer-selective training reduces communication overhead by 75% while preserving accuracy

---

## 1. Differential Privacy Fundamentals and DP-SGD

### 1.1 Core Concepts

**Differential Privacy Definition:**
A randomized algorithm A fulfills ε-differential privacy if, for any two databases D and D' differing in at most one record:

```
Pr[A(D) = O] ≤ e^ε · Pr[A(D') = O]
```

where smaller values of ε enforce stronger privacy guarantees.

**DP-SGD Mechanism:**
The standard DP-SGD approach involves two critical steps:

1. **Per-sample Gradient Clipping:** Bounds the L2 norm of each gradient to maximum value C
   ```
   g̃_t(x_i) = g_t(x_i) / max(1, ||g_t(x_i)||_2 / C)
   ```

2. **Noise Addition:** Gaussian noise proportional to clipping bound and noise multiplier σ
   ```
   ḡ_t = (1/|B|) Σ g̃_t(x_i) + N(0, σ²C²I)
   ```

### 1.2 Advanced DP-SGD Implementations

**Privacy Budget Calculation:**
The privacy budget (ε) for a given configuration is computed as:
```
ε = calculate_epsilon(steps, num_samples, batch_size, noise_multiplier, δ)
```
where δ is typically set to 1/n² (n = dataset size).

**Key Implementation Parameters:**
- **Noise Multiplier (σ):** Controls the scale of Gaussian noise added to gradients
- **Clipping Norm (C):** Maximum L2 norm allowed for individual gradients
- **Privacy Budget (ε):** Overall privacy guarantee; lower values = stronger privacy
- **Delta (δ):** Probability of privacy breach; typically 10^-4 to 10^-5

---

## 2. Privacy-Utility Tradeoffs in Clinical Applications

### 2.1 Optimal Operating Regions

**Tertulino et al. (2508.10017v1)** identified an optimal privacy-utility frontier for cardiovascular risk prediction:

- **Optimal Configuration:** ε ≈ 9.0, σ = 1.0, C = 1.5
- **Clinical Performance:** 78% recall (high-risk patient detection)
- **Privacy Guarantee:** Strong (ε < 10)
- **Key Insight:** Non-linear relationship between privacy budget and model recall

**Privacy-Utility Curve Characteristics:**
```
At ε = 38:  FedProx achieves ~71% recall (strong utility, moderate privacy)
At ε = 9:   FedProx achieves ~78% recall (optimal balance)
At ε < 5:   Significant utility degradation begins
```

### 2.2 Multi-Stage Pipeline for Imbalanced Data

**Critical Finding:** Standard FL with DP on severely imbalanced clinical data can fail completely (0% recall).

**Robust Pipeline (Tertulino et al.):**

1. **Stage 1 - Address Class Imbalance:**
   - Apply SMOTETomek at client level
   - Result: Recall improved from 0% to 74%

2. **Stage 2 - Handle Non-IID Data:**
   - Implement FedProx (proximal term μ = 0.01)
   - Result: Recall improved to 77%

3. **Stage 3 - Apply DP:**
   - Use adaptive noise based on privacy budget
   - Result: Privacy-utility frontier achieved

**FedProx Objective Function:**
```
h_k(w) = F_k(w) + (μ/2)||w - w^t||²
```
where w^t are global model weights from previous round.

### 2.3 Compliance-Aware Differential Privacy

**Parampottupadam et al. (2505.22108v3)** introduced adaptive DP based on client compliance scores:

**Compliance Score Calculation:**
```
S_c = Σ(w_i · s_i) / Σ(w_i)
```
where w_i = weight for compliance factor i, s_i = selected option score

**Noise Multiplier Adaptation:**
```
N_m = (1.0 - S_c) + Min_Noise_Multiplier
```

**Performance Results:**
- **High compliance + low compliance clients:** 86.62% accuracy (PneumoniaMNIST)
- **High compliance only:** 81.28% accuracy
- **Improvement:** 5.34% gain from inclusive approach

**Key Compliance Factors:**
- Data encryption standards (AES-256)
- Anonymization practices (ISO/TS 25237:2017)
- Ethical AI policies (EU AI Act, FDA)
- Secure network infrastructure (NIST Framework)
- Patient consent management (HL7 CDA)

---

## 3. Federated Learning Architectures

### 3.1 Standard Federated Averaging (FedAvg)

**Basic Algorithm:**
```
w^(t+1) = Σ(|D_i|/|D|) · w_i^(t+1)
```

**Limitations:**
- Vulnerable to Byzantine attacks
- Sensitive to data heterogeneity
- Poor performance with imbalanced data

### 3.2 Robust Aggregation Methods

**Grama et al. (2009.08294v1)** compared three robust aggregation strategies:

**a) Coordinate-wise Median (COMED):**
- Takes median of all client parameters
- Robust to outliers
- Assumption: equal data sizes (limitation in practice)

**b) Multi-Krum (MKRUM):**
- Selects m most similar local models
- Score calculation: s_i = Σ||V_i - V_j||²
- Eliminates dissimilar updates

**c) Adaptive Federated Averaging (AFA):**
- Models client behavior with Hidden Markov Model
- Blocks systematically faulty clients
- Update rule:
  ```
  θ_g ← Σ(p_k · d_k · θ_k) / Σ(p_k · d_k)
  ```
  where p_k = probability of client k being benign

**Performance on Diabetes Dataset:**
- AFA: Best performance with adversarial clients
- Successfully detected and blocked malicious clients
- DP (ε = 0.0001): Minimal impact on convergence

### 3.3 Byzantine-Robust FL with Differential Privacy

**Key Finding (Grama et al.):** Privacy-preserving methods can be successfully applied alongside Byzantine-robust aggregation.

**Experimental Setup:**
- **Dataset:** Pima Indians Diabetes (768 samples, 10 clients)
- **Malicious Clients:** Label flipping attacks
- **Faulty Clients:** Noisy parameter injection
- **Privacy:** ε₁ = ε₃ = 1e-4, 10% parameter release

**Results:**
- AFA detected and blocked faulty clients
- Final convergence not significantly affected
- k-anonymity (k=4) improved robustness on larger datasets

---

## 4. Deep Learning with Differential Privacy

### 4.1 Convolutional Deep Belief Networks (Phan et al., 1706.08839v2)

**Innovation:** Functional mechanism with Chebyshev polynomial approximation

**Energy Function Approximation:**
```
E(D,W) ≈ Ẽ(D,W) = Σ_{l=0}^L α_l T_l((W^k * v^t)_ij + b_k / Z_ij^k)
```

**Sensitivity Bound:**
```
Δ ≤ 2 max_{t,k} [Σ|α_l|(Σv_{ij,rs}^{t,k} + 1/Z_ij^k)^l + Σ|v_ij^t|]
```

**Approximation Error Bounds:**
```
√(4 + 4log(L)/π²) × N_H²K × U_L(E) > S_L(E) ≥ U_L(E) ≥ (π/4)N_H²K|A_{L+1}|
```

**Key Advantages:**
1. Independent of training epochs for privacy budget
2. Works with large-scale datasets
3. Bounded approximation error
4. Applicable to energy-based models

**Performance (MNIST):**
- ε = 0.5: 93.08% accuracy (vs 88.59% for pSGD)
- Can train with unlimited epochs
- 2,400 epochs for convergence

### 4.2 Modified ResNet for Medical Imaging (Haj Fares et al., 2412.00687v1)

**DPResNet Architecture:**
- Based on ResNet-9
- GroupNormalization (32 groups) instead of BatchNorm
- No max-pooling layers
- Compatible with DP requirements

**Secure Aggregation Protocol (SecAgg+):**
- SMPC-based secure aggregation
- Protects individual client updates
- Handles client dropouts (reconstruction threshold: 4 shares)

**Privacy Parameters:**
- ε = 6.0, δ = 1.9 × 10^-4
- Clipping norm C = 7
- 50 global rounds, E = 3 local epochs

**Results (BloodMNIST):**
```
Configuration         | 10 Clients | 20 Clients
DP-/SecAgg-          | 98.76%     | 97.77%
DP-/SecAgg+          | 98.11%     | 97.01%
DP+/SecAgg+          | 97.78%     | 96.89%
PriMIA (baseline)    | 85.00%     | -
FEDMIC (SOTA)        | -          | 96.33%
```

---

## 5. Communication-Efficient Privacy-Preserving FL

### 5.1 Selective Attention Federated Learning (Li & Zhang, 2504.11793v3)

**Core Innovation:** Dynamic layer selection based on attention patterns

**Attention-Based Layer Selection:**
```
A_l = Σ_{h=1}^H Σ_{i=1}^N Σ_{j=1}^N α_{h,i,j}^l · I(t_j ∈ T)
```
where α = attention weights, T = task-relevant tokens

**Benefits:**
1. **Communication Efficiency:** 75% reduction in transmitted parameters
2. **Privacy Enhancement:** Fewer sensitive parameter updates
3. **Task Adaptability:** Dynamic selection based on current model state

**Performance (i2b2 Clinical Concept Extraction):**
```
Method              | F1 Score | Communication Reduction
Centralized         | 90.2%    | 0%
FedAvg             | 87.1%    | 0%
Layer-Skipping FL  | 88.7%    | 70%
SAFL (proposed)    | 89.6%    | 75%
```

**Privacy-Utility Trade-off (i2b2):**
```
Privacy Budget (ε) | FedAvg | SAFL
8.0               | 84.1%  | 88.2%
4.0               | 81.3%  | 86.5%
2.0               | 75.4%  | 83.1%
1.0               | 68.9%  | 76.8%
0.5               | 61.2%  | 70.3%
```

**Layer Selection Patterns:**
- **Concept Extraction:** Higher layers (13-18) selected 42.5% of time
- **Diagnosis Prediction:** Middle layers (7-12) selected 52.1% of time
- Pattern stability across heterogeneous clients

---

## 6. Practical Implementation Guidelines

### 6.1 DP-SGD Configuration Best Practices

**For Clinical Classification Tasks:**

1. **Privacy Budget Selection:**
   - Strong privacy: ε < 1.0 (significant utility trade-off)
   - Balanced: ε = 4.0-9.0 (optimal for most clinical tasks)
   - Moderate privacy: ε = 10-20 (minimal utility impact)

2. **Noise Multiplier Tuning:**
   - Start with σ = 1.0
   - Increase for stronger privacy (higher σ = more noise)
   - Monitor convergence; adjust if training unstable

3. **Clipping Norm Selection:**
   - C = 1.0-1.5 for stable gradients
   - C = 7.0 for larger models (DPResNet)
   - Balance: too low = slow convergence, too high = weak privacy

4. **Batch Size Considerations:**
   - Larger batches improve privacy (dilute individual contributions)
   - Typical: 32-64 for clinical datasets
   - Trade-off with computational resources

### 6.2 Federated Learning Setup

**Client-Side Configuration:**
- **Local Epochs:** 3-5 per round
- **Learning Rate:** 1e-4 to 2e-5 (with warmup)
- **Optimizer:** Adam (adaptive learning rates)
- **Dropout:** 0.5 (prevent overfitting)

**Server-Side Aggregation:**
- **Communication Rounds:** 50-100 for convergence
- **Client Selection:** 100% for small networks (<20 clients)
- **Aggregation Method:** FedProx for non-IID data (μ = 0.01)
- **Privacy Mechanism:** Client-side DP before transmission

### 6.3 Handling Imbalanced Clinical Data

**Multi-Stage Approach:**

1. **Preprocessing (Client-Side):**
   ```python
   # Apply SMOTETomek before local training
   X_resampled, y_resampled = SMOTETomek(random_state=42).fit_resample(X, y)
   ```

2. **Training Configuration:**
   - Use class weights in loss function
   - Stratified sampling for batches
   - Monitor recall (critical for rare disease detection)

3. **Evaluation Metrics:**
   - **Avoid:** Accuracy alone (misleading with imbalance)
   - **Use:** F1-score, Recall, Precision, AUC-ROC
   - **Priority:** Recall for high-risk patient identification

### 6.4 Privacy Accounting

**Composition Theorems:**
- Sequential composition: ε_total = Σε_i
- Advanced composition: ε_total ≈ √(2k·ln(1/δ))·ε + k·ε·(e^ε - 1)
- RDP Accountant (Opacus): More accurate for iterative training

**Privacy Budget Management:**
```python
from opacus.accountants import RDPAccountant

accountant = RDPAccountant()
for step in range(total_steps):
    accountant.step(noise_multiplier=sigma, sample_rate=batch_size/dataset_size)
epsilon = accountant.get_epsilon(delta)
```

---

## 7. Privacy Attack Vectors and Defenses

### 7.1 Known Vulnerabilities

**Membership Inference Attacks:**
- Adversary determines if specific record was in training set
- More effective on overfitted models
- **Defense:** DP with appropriate ε, regularization

**Model Inversion Attacks:**
- Reconstruct training data from model parameters
- Particularly dangerous for medical imaging
- **Defense:** DP-SGD, SecAgg, gradient pruning

**Gradient Leakage:**
- Extract sensitive information from shared gradients
- Feasible even with batch sizes up to 128
- **Defense:** Gradient clipping + noise, SecAgg

### 7.2 Defense Mechanisms

**1. Differential Privacy (DP):**
- Theoretical guarantees against inference
- Calibrated noise addition
- Privacy budget tracking

**2. Secure Multi-Party Computation (SMPC):**
- Protects individual updates during aggregation
- Honest-but-curious server model
- Computational overhead consideration

**3. Homomorphic Encryption:**
- Computation on encrypted data
- Strong privacy but high computational cost
- Not covered in detail in reviewed papers

**4. Client Selection and Validation:**
- Behavioral modeling (HMM in AFA)
- Anomaly detection
- Byzantine-robust aggregation

---

## 8. Domain-Specific Considerations for Clinical AI

### 8.1 Regulatory Compliance

**HIPAA/HITECH Requirements:**
- De-identification of Protected Health Information (PHI)
- Minimum necessary standard
- Business Associate Agreements for cloud deployment
- DP can provide "safe harbor" for de-identification

**GDPR Considerations:**
- Data minimization principle
- Purpose limitation
- Right to explanation (challenging with complex models)
- Privacy by design and default

### 8.2 Clinical Validation

**Model Performance Metrics:**
- **Sensitivity/Recall:** Critical for disease detection (minimize false negatives)
- **Specificity:** Important to avoid unnecessary interventions
- **Positive Predictive Value:** Clinician's perspective
- **Calibration:** Probability predictions must be reliable

**Deployment Considerations:**
- Interpretability vs. privacy trade-offs
- Model updating without data centralization
- Drift detection in federated settings
- Fairness across demographic groups

### 8.3 Data Heterogeneity in Healthcare

**Sources of Non-IID Data:**
- Different patient populations (demographics, comorbidities)
- Varying diagnostic protocols across institutions
- Equipment and measurement differences
- Documentation practices and EHR systems

**Mitigation Strategies:**
- **FedProx:** Regularization for heterogeneous data
- **Personalization:** Local fine-tuning on institution-specific data
- **Domain adaptation:** Transfer learning techniques
- **Clustering:** Group similar institutions

---

## 9. Comparative Analysis of Approaches

### 9.1 Privacy Mechanisms

| **Mechanism** | **Privacy Strength** | **Utility Impact** | **Computation** | **Best For** |
|---------------|---------------------|-------------------|-----------------|--------------|
| DP-SGD | Provable (ε-DP) | Moderate | Low | Standard training |
| Functional Mechanism | Provable (ε-DP) | Low | Moderate | Energy-based models |
| SecAgg (SMPC) | Strong | Minimal | High | Aggregation |
| Compliance-Aware DP | Adaptive | Low | Low | Multi-institutional |
| Selective Attention | Implicit | Minimal | Moderate | LLMs/Transformers |

### 9.2 Aggregation Methods

| **Method** | **Robustness** | **Complexity** | **Non-IID Performance** | **Privacy Compatible** |
|------------|---------------|---------------|------------------------|----------------------|
| FedAvg | Low | O(n) | Poor | Yes |
| FedProx | Moderate | O(n) | Good | Yes |
| COMED | High | O(n·log n) | Moderate | Yes |
| MKRUM | High | O(n²) | Moderate | Yes |
| AFA | Very High | O(n²) | Good | Yes |

### 9.3 Architecture Adaptations

| **Architecture** | **DP Compatibility** | **Communication** | **Accuracy** | **Clinical Domain** |
|------------------|---------------------|------------------|--------------|-------------------|
| Standard ResNet | Moderate (BatchNorm issue) | High | High | Imaging |
| DPResNet | High (GroupNorm) | High | High | Imaging |
| pCDBN | High | Moderate | High | Tabular/Time-series |
| SAFL | High | Low (75% reduction) | High | Clinical Text/NLP |

---

## 10. Research Gaps and Future Directions

### 10.1 Identified Gaps

**1. Cross-Domain Privacy Preservation:**
- Most research focuses on single modality (imaging OR text OR tabular)
- Multimodal clinical AI requires unified privacy framework
- Electronic Health Records combine all modalities

**2. Dynamic Privacy Budgets:**
- Current approaches use fixed ε throughout training
- Adaptive privacy budgets based on learning phase underexplored
- Potential for better privacy-utility trade-offs

**3. Fairness Under Privacy Constraints:**
- DP can exacerbate disparities (differential impact)
- Limited research on fairness-privacy-utility trilemma
- Critical for clinical deployment

**4. Long-Term Privacy Guarantees:**
- Composition over model lifecycle unclear
- Continuous learning with privacy preservation
- Privacy under model updates and retraining

### 10.2 Promising Research Directions

**1. Hybrid Privacy Mechanisms:**
- Combine DP + SMPC + trusted execution environments
- Layered defense approach
- Context-dependent privacy allocation

**2. Automated Privacy-Utility Optimization:**
- Meta-learning for hyperparameter selection
- Neural architecture search for DP-friendly models
- Automated compliance verification

**3. Decentralized Federated Learning:**
- Peer-to-peer architectures (no central server)
- Blockchain for audit trails
- Reduced single point of failure

**4. Privacy-Preserving Model Explanations:**
- SHAP/LIME under differential privacy
- Counterfactual explanations without data access
- Interpretability for clinical trust

**5. Standardized Benchmarks:**
- Privacy-utility curves for common clinical tasks
- Standardized evaluation protocols
- Reproducible implementations

---

## 11. Key Recommendations for Practitioners

### 11.1 Starting a Privacy-Preserving Clinical AI Project

**Phase 1: Requirements Analysis**
1. Identify regulatory requirements (HIPAA, GDPR, etc.)
2. Determine acceptable privacy-utility trade-offs
3. Assess computational resources available
4. Evaluate data heterogeneity across institutions

**Phase 2: Architecture Selection**
```
If imaging:          → DPResNet + SecAgg
If clinical text:    → SAFL (selective attention)
If tabular/EHR:      → pCDBN or standard NN with DP-SGD
If multimodal:       → Hybrid approach (domain-specific)
```

**Phase 3: Privacy Configuration**
1. **Start Conservative:** ε = 1.0, monitor utility
2. **Gradually Relax:** Increase ε until acceptable performance
3. **Validate Privacy:** Test against known attacks
4. **Document Everything:** Maintain privacy audit trail

**Phase 4: Federated Setup**
1. Choose aggregation method based on threat model
2. Implement Byzantine-robust aggregation if untrusted clients
3. Use FedProx for heterogeneous data
4. Monitor client behavior (detect anomalies)

### 11.2 Quick Reference: Privacy Parameter Selection

**Conservative (Strong Privacy):**
```
ε = 0.5-1.0
σ = 2.0-4.0
C = 1.0
Expected utility: 60-75% of non-private baseline
```

**Balanced (Recommended Starting Point):**
```
ε = 4.0-9.0
σ = 1.0-2.0
C = 1.0-1.5
Expected utility: 85-95% of non-private baseline
```

**Relaxed (Minimal Privacy Impact):**
```
ε = 10-20
σ = 0.5-1.0
C = 2.0-5.0
Expected utility: 95-99% of non-private baseline
```

### 11.3 Common Pitfalls to Avoid

**1. Privacy Budget Exhaustion:**
- ❌ Running too many training epochs without accounting
- ✓ Use privacy accountant, track cumulative ε

**2. Ignoring Class Imbalance:**
- ❌ Relying on accuracy alone with imbalanced data
- ✓ Apply SMOTETomek, monitor recall/F1

**3. Overlooking Non-IID Data:**
- ❌ Using vanilla FedAvg with heterogeneous institutions
- ✓ Use FedProx, consider personalization

**4. Inadequate Clipping:**
- ❌ Setting C too high (weak privacy) or too low (poor convergence)
- ✓ Tune C based on gradient magnitude distribution

**5. Premature Deployment:**
- ❌ Deploying without attack testing
- ✓ Validate against membership inference, model inversion

---

## 12. Conclusion

Privacy-preserving machine learning in healthcare has matured significantly, with multiple viable approaches for protecting sensitive clinical data while maintaining model utility. The research reviewed demonstrates that:

1. **DP-SGD is Production-Ready:** With proper tuning (ε = 4-9, adaptive noise), clinical utility can be preserved while providing formal privacy guarantees.

2. **Federated Learning Enables Collaboration:** Multi-institutional model training without data sharing is feasible, with robust aggregation methods effectively handling adversarial clients.

3. **Architectural Innovations Matter:** Purpose-built architectures (DPResNet, pCDBN, SAFL) achieve better privacy-utility trade-offs than generic models with DP bolted on.

4. **The Privacy-Utility Trade-off is Non-Linear:** Small increases in privacy budget near the optimal region can yield significant utility gains; careful tuning is essential.

5. **Compliance-Aware Approaches are Promising:** Adaptive privacy mechanisms based on institutional compliance enable more inclusive and effective federated learning.

**For healthcare AI practitioners**, the path forward involves:
- Starting with well-established frameworks (DP-SGD, FedProx)
- Carefully documenting privacy parameters and trade-offs
- Validating against known attack vectors
- Maintaining audit trails for regulatory compliance
- Continuously monitoring for model drift and privacy degradation

**For researchers**, key opportunities include:
- Developing automated privacy-utility optimization
- Exploring hybrid privacy mechanisms
- Addressing fairness under privacy constraints
- Creating standardized benchmarks and evaluation protocols
- Investigating privacy-preserving model interpretability

The convergence of regulatory requirements, technical capability, and demonstrated clinical utility positions privacy-preserving machine learning as a critical enabler for the next generation of healthcare AI systems.

---

## References

### Papers Reviewed

1. **Tertulino, R.** (2025). A Robust Pipeline for Differentially Private Federated Learning on Imbalanced Clinical Data using SMOTETomek and FedProx. arXiv:2508.10017v1.

2. **Parampottupadam, S., et al.** (2025). Inclusive, Differentially Private Federated Learning for Clinical Data. arXiv:2505.22108v3.

3. **Phan, N., Wu, X., & Dou, D.** (2017). Preserving Differential Privacy in Convolutional Deep Belief Networks. Machine Learning Journal. arXiv:1706.08839v2.

4. **Grama, M., et al.** (2020). Robust Aggregation for Adaptive Privacy Preserving Federated Learning in Healthcare. arXiv:2009.08294v1.

5. **Haj Fares, M., & Saad, A.M.S.E.** (2024). Towards Privacy-Preserving Medical Imaging: Federated Learning with Differential Privacy and Secure Aggregation Using a Modified ResNet Architecture. NeurIPS 2024. arXiv:2412.00687v1.

6. **Li, Y., & Zhang, L.** (2025). Selective Attention Federated Learning: Improving Privacy and Efficiency for Clinical Text Classification. arXiv:2504.11793v3.

### Key Citations from Papers

- **Abadi et al. (2016):** Deep Learning with Differential Privacy
- **McMahan et al. (2017):** Communication-Efficient Learning (FedAvg)
- **Dwork & Roth (2014):** The Algorithmic Foundations of Differential Privacy
- **Li et al. (2020):** Federated Optimization in Heterogeneous Networks (FedProx)
- **Blanchard et al. (2017):** Byzantine-Tolerant Gradient Descent
- **Muñoz-González et al. (2019):** Byzantine-Robust Federated Machine Learning (AFA)

---

**Document Metadata:**
- Total Papers Analyzed: 6 primary + 20+ cited references
- Focus Areas: DP-SGD, Federated Learning, Secure Computation, Clinical Applications
- Date Range: 2017-2025
- Practical Focus: Implementation guidelines and privacy-utility optimization
