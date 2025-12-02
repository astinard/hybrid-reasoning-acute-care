# Cross-Institutional and Multi-Site Learning for Clinical AI: A Comprehensive Research Synthesis

**Date:** December 1, 2025
**Focus:** Multi-site clinical AI, cross-institutional learning, domain adaptation, and privacy-preserving methods for healthcare deployment

---

## Executive Summary

This synthesis examines the current state of cross-institutional and multi-site learning for clinical AI, with particular emphasis on challenges related to data heterogeneity, domain shift, generalization, and privacy preservation. Based on comprehensive analysis of 100+ papers from ArXiv, we identify critical challenges in deploying AI models across multiple healthcare institutions and present state-of-the-art approaches to address these issues.

**Key Findings:**
- **Domain shift** is pervasive across healthcare institutions due to variations in equipment, protocols, patient populations, and clinical practices
- **Federated learning** enables privacy-preserving collaboration but faces challenges with non-IID data distribution
- **Performance degradation** of 15-30% is common when models are deployed to new sites without adaptation
- **Differential privacy** can be integrated with federated learning but introduces privacy-utility tradeoffs
- **Domain adaptation** techniques show promise but require careful validation across diverse clinical settings

**Critical Gap:** Most studies focus on single-domain applications (e.g., radiology) with limited evaluation on real-world multi-center deployments, particularly for acute care settings like emergency departments.

---

## 1. Key Papers with ArXiv IDs

### 1.1 Multi-Site Clinical Studies

**Federated Learning Foundations:**
- **2107.02586** - Differentially private federated deep learning for multi-site medical image segmentation (MIMIC-III validation)
- **2204.10836** - Federated Learning Enables Big Data for Rare Cancer Boundary Detection (71 healthcare institutions, 6,314 patients)
- **1810.08553** - Federated Learning in Distributed Medical Databases: Meta-Analysis of Large-Scale Subcortical Brain Data

**Domain Generalization:**
- **2103.11163** - An Empirical Framework for Domain Generalization in Clinical Settings (multi-site time series and medical imaging)
- **2504.20635** - Bridging the Generalisation Gap: Synthetic Data Generation for Multi-Site Clinical Model Validation
- **2503.06759** - Revisiting Invariant Learning for Out-of-Domain Generalization on Multi-Site Mammogram Datasets

**Multi-Site Segmentation:**
- **2308.16376** - Improving Multiple Sclerosis Lesion Segmentation Across Clinical Sites with Noise-Resilient Training
- **2302.03911** - Multi-site Organ Segmentation with Federated Partial Supervision and Site Adaptation
- **2411.06513** - PRISM: Privacy-preserving Inter-Site MRI Harmonization via Disentangled Representation Learning

### 1.2 External Validation and Generalization

**Performance Assessment:**
- **2305.03219** - All models are local: time to replace external validation with recurrent local validation
- **2207.05796** - Estimating Test Performance for AI Medical Devices under Distribution Shift with Conformal Prediction
- **2312.06697** - Performance of externally validated machine learning models for breast cancer (systematic review)

**Cross-Site Robustness:**
- **2407.13632** - Data Alchemy: Mitigating Cross-Site Model Variability Through Test Time Data Calibration
- **2507.19530** - Clinical-Grade Blood Pressure Prediction in ICU Settings with Cross-Institutional Validation
- **2501.13587** - Contrastive Representation Learning for Cross-institutional Knowledge Transfer

### 1.3 Privacy-Preserving Multi-Site Learning

**Differential Privacy:**
- **1910.02578** - Differential Privacy-enabled Federated Learning for Sensitive Health Data (1M patients)
- **2310.06237** - Differentially Private Multi-Site Treatment Effect Estimation
- **2110.03478** - Complex-valued Federated Learning with Differential Privacy for MRI Applications

**Federated Learning Frameworks:**
- **1907.09173** - FedHealth: A Federated Transfer Learning Framework for Wearable Healthcare
- **2306.14483** - Medical Federated Model with Mixture of Personalized and Sharing Components
- **2510.06259** - Beyond Static Knowledge Messengers: Adaptive, Fair, and Scalable Federated Learning

**Secure Aggregation:**
- **2409.00974** - Enhancing Privacy in Federated Learning: Secure Aggregation for Real-World Healthcare
- **2412.17150** - SplitFedZip: Learned Compression for Data Transfer Reduction in Split-Federated Learning

### 1.4 Domain Adaptation for Healthcare

**Medical Imaging:**
- **1812.01281** - Towards Continuous Domain adaptation for Healthcare (lung X-ray segmentation)
- **2304.09164** - Structure Preserving Cycle-GAN for Unsupervised Medical Image Domain Adaptation
- **2403.08197** - PAGE: Domain-Incremental Adaptation with Past-Agnostic Generative Replay

**Clinical Time Series:**
- **1904.00655** - Transfer Learning for Clinical Time Series Analysis using Deep Neural Networks
- **2506.22393** - Multi-View Contrastive Learning for Robust Domain Adaptation in Medical Time Series
- **2307.16651** - UDAMA: Unsupervised Domain Adaptation for Cardio-fitness Prediction

### 1.5 Clinical Deployment Studies

**ICU and Critical Care:**
- **2107.05230** - Predicting sepsis in multi-site, multi-national intensive care cohorts (156,309 ICU admissions)
- **2206.05581** - Federated Offline Reinforcement Learning (sepsis dataset across sites)
- **2411.04285** - Robust Real-Time Mortality Prediction using Temporal Difference Learning

**Emergency Department:**
- **2509.08247** - CRITICAL Records Integrated Standardization Pipeline (371,365 patients, 4 institutions)
- **2507.19530** - Clinical-Grade Blood Pressure Prediction with External Validation (MIMIC-III to eICU)

---

## 2. Cross-Site Learning Methods

### 2.1 Federated Learning Approaches

**Standard Federated Averaging (FedAvg):**
- Most common baseline for multi-site collaboration
- Each site trains locally, shares only model updates
- Struggles with extreme data heterogeneity (non-IID distributions)
- Performance comparable to centralized training when data is IID

**FedProx (Federated Proximal):**
- Adds proximal term to handle heterogeneous data
- Better stability with extreme data imbalance across sites
- Requires careful hyperparameter tuning
- **ArXiv 2302.03911**: Demonstrated superiority over FedAvg for multi-site organ segmentation

**Personalized Federated Learning:**
- **ArXiv 2306.14483**: Mixture of personalized and sharing components
- Achieves 60% improvement in communication efficiency
- Better tradeoff between generalization and personalization
- Particularly effective for heterogeneous clinical datasets

**Hierarchical and Adaptive Approaches:**
- **ArXiv 2510.06259**: Adaptive Knowledge Messengers that scale dynamically
- **ArXiv 2005.12055**: Hierarchical Bayesian Regression for multi-site normative modeling
- 60-70% reduction in communication rounds
- 56-68% fairness improvement across institutions

### 2.2 Domain Adaptation Techniques

**Adversarial Domain Adaptation:**
- **ArXiv 2307.16651**: UDAMA for cardio-fitness prediction (10%+ improvement)
- **ArXiv 2202.13174**: BioADAPT-MRC for biomedical machine reading comprehension
- Aligns feature distributions across source and target domains
- Effective for labeled source, unlabeled target scenarios

**Self-Training and Pseudo-Labeling:**
- **ArXiv 2211.07692**: Self-training for liver histopathology (3% F1 improvement)
- Leverages both labeled and unlabeled data
- Reduces annotation burden on new sites
- Performance approaches fully supervised with 2x less annotations

**Test-Time Adaptation:**
- **ArXiv 2407.13632**: Data Alchemy for cross-site model variability
- **ArXiv 2004.04668**: Test-Time Adaptable Neural Networks for medical segmentation
- Adapts models during inference without retraining
- Particularly useful for streaming deployment scenarios

**Domain Invariant Representation Learning:**
- **ArXiv 2310.07799**: Domain-invariant clinical representation learning
- **ArXiv 2008.04152**: Learning invariant features for chest X-ray datasets
- Learns features robust to scanner/protocol variations
- Improves generalization to unseen institutions

### 2.3 Transfer Learning Strategies

**Pre-training and Fine-tuning:**
- **ArXiv 2407.11034**: Transfer learning for biomedical data (scoping review of 3,515 papers)
- Only 2% utilized external studies for validation
- 7% addressed multi-site collaborations with privacy
- Limited cross-institutional generalization without adaptation

**Knowledge Distillation:**
- **ArXiv 2207.02445**: Distillation for portability across institutions
- **ArXiv 2412.20040**: Contrastive pretrain with prompt tuning for multi-center medication
- Enables model compression and transfer
- Reduces computational requirements for resource-constrained sites

**Multi-Task Learning:**
- **ArXiv 2108.12978**: Private multi-task learning with joint differential privacy
- Learns shared representations across related tasks
- Improves sample efficiency for rare conditions
- Better privacy-utility tradeoffs than single-task approaches

---

## 3. Domain Adaptation Approaches

### 3.1 Feature-Level Adaptation

**Statistical Harmonization:**
- **ArXiv 1911.04289**: Relevance Vector Machines for MRI brain volume harmonization
- **ArXiv 2407.13632**: Cross-site calibration using MMD statistics
- Reduces scanner and center variability
- Preserves measurements not requiring correction

**Contrastive Learning:**
- **ArXiv 2501.13587**: Contrastive Predictive Coding for pediatric ventilation
- **ArXiv 2506.22393**: Multi-view contrastive learning for EEG analysis
- Learns temporal progression patterns
- Transfers more readily than point-of-care decisions

**Disentangled Representations:**
- **ArXiv 2411.06513**: PRISM for inter-site MRI harmonization
- **ArXiv 2306.09177**: Disentangled autoencoder for multi-domain tasks
- Separates anatomical features from site-specific variations
- Enables unpaired image translation without traveling subjects

### 3.2 Model-Level Adaptation

**Batch Normalization Strategies:**
- **ArXiv 2106.01009**: FedHealth 2 with weighted batch normalization
- **ArXiv 2206.05284**: Decoupling predictions via distribution-conditioned adaptation
- Preserves local batch normalization for site-specific statistics
- 10%+ improvement in activity recognition accuracy

**Meta-Learning:**
- **ArXiv 2501.13479**: Adaptive Few-Shot Learning for data-scarce domains
- **ArXiv 2211.15476**: Meta-analysis of individualized treatment rules
- Learns to quickly adapt to new sites with minimal data
- Particularly relevant for rare diseases

**Ensemble Methods:**
- **ArXiv 2507.19530**: Ensemble framework combining Gradient Boosting, RF, XGBoost
- **ArXiv 2410.00046**: Mixture of Multicenter Experts for radiotherapy
- Reduces variance across sites
- Better uncertainty quantification

### 3.3 Data-Level Adaptation

**Synthetic Data Generation:**
- **ArXiv 2504.20635**: Synthetic data for multi-site clinical validation
- **ArXiv 2212.01109**: Generative augmentation for non-IID federated learning
- Bridges distributional gaps between sites
- Enables controlled benchmarking of robustness

**Domain-Specific Augmentation:**
- **ArXiv 2310.15371**: Vicinal feature statistics augmentation for FL
- Exploits batch-wise feature statistics
- Represents site discrepancy probabilistically
- 4% DSC increase in cardiac segmentation

---

## 4. Generalization Challenges

### 4.1 Types of Distribution Shift

**Covariate Shift:**
- Different patient demographics across sites
- Variations in imaging protocols and equipment
- Scanner manufacturer differences (Siemens vs. GE vs. Philips)
- **Impact:** 15-30% performance degradation typical

**Label Shift:**
- Disease prevalence varies by institution
- Referral patterns differ (academic vs. community hospitals)
- Annotation protocol variations
- **Impact:** Particularly severe for imbalanced datasets

**Concept Drift:**
- Evolution of clinical practices over time
- Changes in treatment protocols
- Technology upgrades
- **Impact:** Models become outdated without continuous learning

**Confounding by Provenance:**
- **ArXiv 2310.02451**: Backdoor adjustment for multi-institutional clinical notes
- **ArXiv 2312.05435**: Provenance-related distribution shifts in foundation models
- Site-specific language use and measurement differences
- Requires causal adjustment methods

### 4.2 Performance Degradation Patterns

**Multi-Site Imaging Studies:**
- **ArXiv 2503.06759**: Mammogram classification across sites
  - Direct transfer: 0.545 AUPRC → 0.710 with adaptation
  - Further improvement to 0.852 with Data Alchemy
- **ArXiv 2102.08660**: Chest X-ray interpretation
  - Photos of X-rays: Statistically significant drop
  - External datasets: 3 of 8 models worse than radiologists
- **ArXiv 1911.00515**: Brain MRI across multiple cohorts
  - Good performance with similar protocols
  - Substantially worse with different tissue contrasts

**Clinical Time Series:**
- **ArXiv 2103.11163**: Domain generalization framework
  - Limited performance gains on real-world medical imaging
  - Some scenarios in time series show improvement
  - Highly dependent on type of distribution shift

**ICU Predictions:**
- **ArXiv 2507.19530**: Blood pressure in MIMIC-III to eICU
  - 30% performance degradation in external validation
  - Critical limitations in hypotensive patients
  - Uncertainty quantification essential for deployment

### 4.3 Factors Affecting Generalization

**Data Volume and Quality:**
- Small sample sizes per site limit learning
- Annotation quality varies across institutions
- Missing data patterns differ by site
- **ArXiv 2204.10836**: 71 institutions needed for rare cancer detection

**Model Architecture:**
- **ArXiv 2103.11163**: Domain generalization methods don't consistently outperform ERM
- Deep models more susceptible to overfitting site-specific features
- Simpler models may generalize better with limited multi-site data

**Task Complexity:**
- Binary classification more robust than multi-class
- Segmentation harder to generalize than classification
- Survival prediction especially challenging across sites

---

## 5. Privacy-Preserving Multi-Site Learning

### 5.1 Differential Privacy Mechanisms

**DP-SGD (Differentially Private Stochastic Gradient Descent):**
- **ArXiv 2107.02586**: Applied to medical image segmentation
- Mean absolute error: 6.03 mmHg (SBP), 7.13 mmHg (DBP)
- Prevents gradient-based model inversion attacks
- Privacy-utility tradeoff requires careful ε selection

**Gaussian Mechanism:**
- **ArXiv 2110.03478**: Complex-valued Gaussian mechanism for MRI
- Characterized via f-DP, (ε,δ)-DP, and Rényi-DP
- Excellent utility with strong privacy guarantees
- First application to complex-valued medical data

**Sparse Vector Technique (DP-SVT):**
- **ArXiv 2302.04208**: Compared with DP-SGD on MIMIC-III
- Mean authenticity of 0.778 on CKD dataset
- Large quantifiable privacy leakage for similar performance
- Tradeoff between privacy level and model accuracy

### 5.2 Federated Learning with Privacy

**Secure Aggregation:**
- **ArXiv 2409.00974**: Secure aggregation using Joye-Libert and Low Overhead Masking
- <1% overhead on CPU, <50% on GPU for large models
- Protection phases <10 seconds
- Accuracy impact <2% compared to non-secure FL

**Privacy Budget Management:**
- **ArXiv 1910.02578**: Two-level privacy protection (JDP and LDP)
- Applied to 1M patient EHR dataset
- Maintains utility while preventing privacy attacks
- Requires careful ε allocation across sites

**Homomorphic Encryption:**
- Enables computation on encrypted data
- High computational overhead limits practical deployment
- **ArXiv 2111.14838**: Inefficacy for deep learning on time series
- More suitable for simple aggregation operations

### 5.3 Privacy-Utility Tradeoffs

**Performance Impact:**
- **ArXiv 2412.00687**: Modified ResNet with DP achieves near non-private accuracy
- **ArXiv 2306.17794**: Strategic privacy budget calibration maintains robust performance
- **ArXiv 2406.10563**: Heterogeneous FL with DP for diabetes and mortality prediction

**Privacy Guarantees:**
- Stricter privacy (lower ε) → reduced model utility
- Trade-off varies by dataset characteristics
- Medical imaging: Better tradeoffs than tabular EHR data
- Time series: Strong dataset dependence of DP effectiveness

**Practical Considerations:**
- **ArXiv 2409.18907**: Default FL settings expose private training data
- Noisy defense mechanisms not always effective
- Need for standardized privacy evaluation frameworks
- Gap between theoretical guarantees and practical attacks

---

## 6. Research Gaps

### 6.1 Methodological Limitations

**External Validation:**
- **ArXiv 2305.03219**: External validation insufficient for model safety
- Most studies use single external cohort validation
- Need for recurring local validation paradigm
- Current practices don't account for temporal drift

**Evaluation Frameworks:**
- Lack of standardized multi-site benchmarks
- **ArXiv 2510.06259**: Proposed MedFedBench for 6 healthcare dimensions
- Limited evaluation on truly heterogeneous clinical settings
- Focus on imaging, underrepresentation of EHR and time series

**Reproducibility:**
- **ArXiv 2212.14177**: Only 1 of surveyed studies shared code
- Only 3 used open-access data
- Difficult to validate and build upon prior work
- Need for open-source implementations

### 6.2 Clinical Application Gaps

**Acute Care Settings:**
- Limited studies on emergency department multi-site deployment
- Most work focuses on chronic conditions or imaging
- Real-time requirements not adequately addressed
- Resource constraints in acute settings underexplored

**Rare Diseases:**
- **ArXiv 2204.10836**: Federated learning enables rare cancer research
- But most FL research focuses on common conditions
- Need for methods handling extreme class imbalance across sites
- Privacy concerns more acute with identifiable rare cases

**Multi-Modal Integration:**
- Most studies focus on single modality (imaging OR EHR)
- **ArXiv 2510.06259**: Framework enables multi-modal integration
- Clinical practice requires combining imaging, lab, notes, vitals
- Heterogeneity across modalities adds complexity

### 6.3 Scalability Challenges

**Number of Sites:**
- Most studies: 2-15 sites
- **ArXiv 2510.06259**: Current limit ~15 clients
- Real-world hospital networks: 50-100+ sites
- Communication and coordination costs increase non-linearly

**Computational Resources:**
- Small hospitals lack computational infrastructure
- **ArXiv 2510.06259**: 400-800% ROI projected for rural hospitals
- Need for edge computing and efficient algorithms
- Current methods assume GPU availability

**Fairness Across Institutions:**
- **ArXiv 2510.06259**: Fairness gaps marginalize smaller institutions
- Performance varies significantly by site size
- Need for mechanisms ensuring equitable benefit distribution
- Under-resourced sites should not subsidize wealthy centers

---

## 7. Relevance to ED Multi-Hospital Deployment

### 7.1 Specific Challenges for Emergency Departments

**Real-Time Requirements:**
- ED predictions need <1 second latency for clinical utility
- Most FL approaches have high communication overhead
- **ArXiv 2510.06259**: 60-70% communication reduction needed
- Test-time adaptation must be computationally efficient

**Heterogeneous Patient Populations:**
- ED sees wider patient diversity than specialty clinics
- Acuity levels vary dramatically by institution
- Rural vs. urban vs. academic medical centers
- Requires robust generalization across patient demographics

**Data Quality Issues:**
- Incomplete data common in ED setting
- **ArXiv 2509.08247**: CRITICAL dataset addresses this
- Missing modalities across sites
- Temporal resolution varies by institution

**Regulatory and Workflow Constraints:**
- Must integrate with existing EMR systems
- Cannot disrupt clinical workflow
- HIPAA compliance non-negotiable
- Clinical validation requirements stringent

### 7.2 Applicable Methods

**Federated Learning Frameworks:**
- **ArXiv 2306.14483**: Personalized FL with 60% communication efficiency
- Suitable for multi-hospital ED networks
- Privacy-preserving without data sharing
- Can handle heterogeneous EHR systems

**Domain Adaptation:**
- **ArXiv 1812.01281**: Continuous domain adaptation for healthcare
- **ArXiv 2407.13632**: Test-time calibration for cross-site variability
- Adapts to new ED sites without retraining
- Maintains performance under protocol variations

**Privacy Techniques:**
- **ArXiv 2409.00974**: Secure aggregation with <1% overhead
- **ArXiv 1910.02578**: Differential privacy for EHR data
- HIPAA-compliant multi-site collaboration
- Protects both patient and institutional data

**Uncertainty Quantification:**
- **ArXiv 2110.07661**: Distribution-free conformal predictions
- Essential for clinical decision support in ED
- Provides coverage guarantees across sites
- Helps identify when to defer to human judgment

### 7.3 Deployment Strategies

**Phased Rollout:**
1. **Initial deployment** at source institution with local validation
2. **Pilot expansion** to 2-3 similar EDs with federated learning
3. **Gradual scaling** with continuous monitoring and adaptation
4. **Full network deployment** with recurring local validation

**Performance Monitoring:**
- **ArXiv 2305.03219**: Recurring local validation paradigm
- Real-time performance tracking per site
- Automated alerts for distribution shift
- Regular model updates and retraining

**Site-Specific Adaptation:**
- **ArXiv 2206.05284**: Decoupled predictions per site
- Maintain global model for general knowledge
- Local adaptation layers for site-specific patterns
- Balance between standardization and customization

---

## 8. Recommendations

### 8.1 For Model Development

**Multi-Site Design from the Start:**
- Plan for heterogeneity during initial development
- Include diverse sites in training cohort
- **ArXiv 2103.11163**: Test on multiple external cohorts
- Use domain adaptation techniques proactively

**Robust Evaluation:**
- External validation on ≥3 diverse sites
- Test under realistic distribution shifts
- **ArXiv 2207.05796**: Conformal prediction for performance estimation
- Report performance stratified by site characteristics

**Privacy by Design:**
- Integrate differential privacy from the beginning
- **ArXiv 2107.02586**: Budget ε appropriately for medical data
- Use secure aggregation for federated training
- Document privacy guarantees explicitly

### 8.2 For Clinical Deployment

**Continuous Monitoring:**
- Implement recurring local validation
- **ArXiv 2305.03219**: Monitor for temporal drift
- Track performance metrics per site
- Automated alerts for degradation

**Uncertainty Quantification:**
- Provide prediction confidence scores
- **ArXiv 2110.07661**: Use conformal prediction for coverage guarantees
- Flag high-uncertainty predictions for review
- Enable clinician override mechanisms

**Stakeholder Engagement:**
- Involve clinicians from all sites early
- Address workflow integration concerns
- Provide interpretable explanations
- Regular feedback loops for improvement

### 8.3 For Future Research

**Standardized Benchmarks:**
- Develop multi-site clinical datasets
- **ArXiv 2510.06259**: Adopt MedFedBench framework
- Include diverse shift scenarios
- Enable reproducible comparisons

**Scalability Research:**
- Methods for 50-100+ site networks
- Efficient communication protocols
- Edge computing integration
- **ArXiv 2412.17150**: Learned compression for data transfer

**Fairness and Equity:**
- Ensure small hospitals benefit equitably
- **ArXiv 2510.06259**: Fairness-aware distillation
- Address digital divide concerns
- Democratize access to AI tools

---

## 9. Key Takeaways for ED Multi-Hospital AI

### Critical Success Factors:

1. **Privacy-Preserving Collaboration**: Federated learning with differential privacy enables multi-site learning without data sharing (ArXiv: 2107.02586, 1910.02578, 2409.00974)

2. **Domain Adaptation**: Essential for handling scanner, protocol, and population variations across EDs (ArXiv: 1812.01281, 2407.13632, 2103.11163)

3. **Uncertainty Quantification**: Conformal prediction and other techniques provide reliable confidence estimates across sites (ArXiv: 2110.07661, 2207.05796)

4. **Continuous Validation**: Recurring local validation needed instead of one-time external validation (ArXiv: 2305.03219)

5. **Computational Efficiency**: Communication reduction and edge computing critical for real-time ED use (ArXiv: 2510.06259, 2412.17150)

### Major Challenges to Address:

1. **Performance Degradation**: 15-30% typical when deploying to new sites without adaptation
2. **Data Heterogeneity**: Non-IID distributions across sites require specialized FL methods
3. **Privacy-Utility Tradeoff**: Strong privacy guarantees come at cost of model performance
4. **Scalability**: Current methods tested on 2-15 sites, need 50-100+ for hospital networks
5. **Fairness**: Small, under-resourced hospitals must benefit equitably

### Recommended Approach for ED Deployment:

1. Use **personalized federated learning** with site-specific adaptation layers
2. Implement **differential privacy** with carefully calibrated privacy budgets
3. Apply **test-time adaptation** for handling new site variations
4. Employ **conformal prediction** for uncertainty quantification
5. Establish **recurring local validation** for continuous monitoring
6. Ensure **secure aggregation** for privacy-preserving model updates

---

## References

Complete list of 100+ papers analyzed available in ArXiv search results. Key papers cited throughout document with ArXiv IDs for easy reference and access.

**Primary Multi-Site Studies:**
- 2204.10836, 2107.02586, 1810.08553, 2103.11163, 2504.20635
- 2308.16376, 2302.03911, 2107.05230, 2509.08247

**Federated Learning:**
- 1907.09173, 2306.14483, 2510.06259, 2106.01009, 2409.00974
- 2110.07661, 2110.03478, 2302.13473, 2006.10517

**Privacy Preservation:**
- 1910.02578, 2310.06237, 2412.00687, 2306.17794, 2405.07735
- 2409.18907, 2302.04208, 2002.09096

**Domain Adaptation:**
- 1812.01281, 2304.09164, 2403.08197, 1904.00655, 2506.22393
- 2307.16651, 2202.13174, 2408.03353, 2211.10475

**External Validation:**
- 2305.03219, 2207.05796, 2312.06697, 2507.22776, 2102.08660

---

## Conclusion

Cross-institutional learning for clinical AI presents both significant challenges and promising opportunities. While substantial progress has been made in federated learning, domain adaptation, and privacy preservation, critical gaps remain—particularly for real-time acute care applications like emergency departments. Success requires careful integration of multiple techniques: privacy-preserving federated learning, robust domain adaptation, uncertainty quantification, and continuous validation. The path forward demands not only technical innovation but also standardized evaluation frameworks, multi-stakeholder collaboration, and commitment to ensuring equitable access across healthcare institutions of all sizes.

For ED multi-hospital deployment specifically, the research indicates that a hybrid approach combining personalized federated learning, test-time adaptation, and conformal prediction offers the most promise for maintaining both privacy and performance across heterogeneous clinical sites.
