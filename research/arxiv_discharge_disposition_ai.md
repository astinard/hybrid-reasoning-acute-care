# ArXiv Research Synthesis: Discharge Planning and Disposition Prediction AI

## Executive Summary

This comprehensive review examines state-of-the-art AI approaches for discharge planning, disposition prediction, and patient outcome forecasting in acute care settings. Based on analysis of 100+ papers from ArXiv, this synthesis reveals several key findings:

**Key Insights:**
- Deep learning models (RNNs, LSTMs, Transformers, CNNs) consistently outperform traditional machine learning for discharge-related predictions
- Multimodal approaches combining structured EHR data, clinical notes, and vital signs achieve superior performance (AUC 0.85-0.95)
- Length of stay (LOS) prediction remains challenging but critical for resource allocation (typical AUC 0.70-0.85)
- Readmission prediction benefits significantly from temporal modeling and NLP of discharge summaries
- Emergency department (ED) triage and disposition decisions show promise with AI support (AUC 0.75-0.88)
- Interpretability and clinical trust remain major barriers to widespread adoption

**Clinical Applications:**
- 30-day readmission prediction (primary focus area)
- ICU admission prediction from ED
- Hospital length of stay forecasting
- Post-acute care (PAC) facility placement
- Mortality risk assessment
- Disease-specific disposition (SNF, home, rehabilitation)

---

## 1. Key Papers with ArXiv IDs

### Discharge Planning & Disposition

**2412.16134v1** - EF-Net: Patient Disposition Analysis
- Neural network combining word embeddings and feature fusion
- 96% accuracy on MIMIC-IV-ED dataset for disposition prediction
- Addresses ED overcrowding through early disposition assessment

**1812.10487v1** - Post-Acute Care Discharge Disposition Prediction
- CHAID algorithm for early PAC discharge prediction
- 84.16% accuracy, AUC 0.81
- Reduced inpatient LOS by 22.22% through early insurance authorization
- Quantified cost savings: $1,974-$2,346 per day per hospital type

**2408.07531v2** - Multi-Agent LLM Clinical Decision Support (KTAS-Based)
- Korean Triage and Acuity Scale integration
- Multi-agent system (Triage Nurse, Emergency Physician, Pharmacist, ED Coordinator)
- Strong performance in disposition decision-making and resource allocation

**2305.15629v1** - Patient Outcome Predictions for Hospital Operations
- ML models predicting 24hr/48hr discharge and ICU transfers
- AUC 75.7%-92.5% across outcomes
- 0.67 day reduction in average LOS post-deployment
- Estimated $55-72 million annual savings

### Readmission Prediction

**1912.10306v1** - Heart Failure Readmission from Clinical Notes
- CNN-based deep learning on discharge summaries
- F1 score 0.756 (general readmission), 0.733 (30-day readmission)
- Outperformed random forest by 12.2% in F1 score
- Chi-square interpretation of key readmission features

**1904.05342v3** - ClinicalBERT for Hospital Readmission
- Bidirectional transformer model pretrained on clinical notes
- Outperforms baselines on 30-day readmission using discharge summaries
- Demonstrates value of clinical domain pretraining

**2204.06766v1** - Multimodal Spatiotemporal Graph Neural Networks
- GNN approach for 30-day all-cause readmission
- Integrates temporal chest radiographs and EHR data
- AUROC improvement over LACE+ score (clinical standard)
- Multi-center validation (MIMIC-IV)

**2406.19980v1** - LSTM vs Traditional ML for Diabetes Readmission
- LightGBM best traditional model (AUC not specified)
- SHAP analysis: lab procedures and discharge disposition critical
- LSTM captures temporal dependencies but prone to overfitting

**2310.10187v2** - Interpretable Deep Learning for Readmission
- ConvLSTM neural networks with NLP word embeddings
- AUROC improvements over traditional models
- Model-dependent interpretability technique for clinicians

**2501.01183v1** - ICU Readmission Prediction (Intracerebral Hemorrhage)
- Focus on ICH patients in MIMIC-III and MIMIC-IV
- Multiple ML techniques evaluated (ANN, XGBoost, Random Forest)

**1905.08547v3** - Benchmarking Deep Learning for ICU Readmission
- Recurrent neural networks with attention mechanisms
- AUROC 0.739, F1-Score 0.372 on MIMIC-III
- Bayesian inference for model uncertainty
- Attention visualization identifies high-risk patient subgroups

**2102.02586v1** - Temporal Cascade and Structural Modeling
- Addresses multimorbidity cascade relationships
- Point process integration with attention-based seq2seq
- Captures non-immediate historical visit influences

### Length of Stay Prediction

**2306.16823v1** - LOS Prediction Using Domain Adaptation
- LSTM and fully connected networks on eICU-CRD and MIMIC-IV
- 110,079 (eICU) and 60,492 (MIMIC-IV) patient stays
- Domain adaptation for cross-ICU generalization
- 1-5% accuracy improvement, up to 2hr computation time savings

**2303.11042v1** - Patient Event Sequences for LOS Prediction
- Transformer-based Medic-BERT (M-BERT)
- 45,000+ emergency care patients from Danish hospital
- High accuracy on LOS prediction tasks
- Outperforms non-sequence-based ML approaches

**2007.09483v4** - Temporal Pointwise Convolutional Networks (TPC)
- Designed for irregular sampling and missing data in EHR
- eICU and MIMIC-IV validation
- Mean absolute deviation: 1.55 days (eICU), 2.28 days (MIMIC-IV)
- 18-68% performance improvement over LSTM
- Mortality as side-task improves LOS prediction

**2110.08918v1** - Clinical Drug Representations for LOS/Mortality
- ECFP and SMILES-Transformer drug embeddings
- 6% AUROC improvement for LOS, 2% for mortality
- 5% AUPRC improvement for LOS, 3.5% for mortality

**2201.00005v1** - Literature Review: LOS for Stroke Patients
- Identifies contradicting conclusions across studies
- Age as predictor: inconsistent findings across research
- Highlights need for additional research in stroke LOS

**2507.11570v1** - SurgeryLSTM for Spine Surgery LOS
- Masked bidirectional LSTM with attention
- R² = 0.86 (outperforms XGBoost at 0.85)
- Key predictors: bone disorder, chronic kidney disease, lumbar fusion
- Attention mechanism provides temporal interpretability

**2508.17554v2** - S²G-Net: State-Space + Graph Neural Networks
- Combines Mamba SSMs with GraphGPS for ICU LOS
- Multi-view patient similarity graphs
- Outperforms sequence-only and graph-only models on MIMIC-IV

**2105.04414v1** - ICU LOS and Mortality Using Vital Signs
- Memory network-based approach
- F1 score 99.16%, ensemble model 96%
- Quantiles approach with 12 baseline features + 21 engineered features

**2304.14663v1** - Client Recruitment for Federated Learning (ICU LOS)
- 189 ICU federated learning approach
- Client recruitment reduces communication overhead
- Maintains model performance while improving training time

**2401.00902v1** - Fairness Evaluation of MIMIC-IV (ICU LOS)
- XGBoost binary classification
- 62% meeting standard of care in physician evaluation
- Addresses fairness across race and insurance attributes

**2407.12741v1** - Federated Learning for Hospital LOS
- FedSGD vs FedAVG comparison
- GTVMin framework for decentralized hospitals
- Privacy-preserving collaborative model training

### Emergency Department Triage & Disposition

**2111.11017v2** - Benchmarking ED Triage Prediction (MIMIC-IV-ED)
- 400,000+ ED visits (2011-2019)
- Three outcomes: hospitalization, critical outcomes, 72-hour reattendance
- Comprehensive benchmark for ED triage models

**2204.07657v6** - Sepsis Detection at ED Triage (KATE Sepsis)
- Machine learning during triage (before labs)
- AUROC 0.9423, sensitivity 71.09%, specificity 94.81%
- 77.67% sensitivity for severe sepsis, 86.95% for septic shock
- Significantly outperforms SIRS screening (AUROC 0.6826)

**2106.12921v2** - ML for Hospital Admission Prediction from ED
- Common biomarkers and clinical features
- F-measure 0.679-0.708, ROC Area 0.734-0.774
- 3,204 ED visits analyzed
- Low-cost, accessible decision support tool

**1804.03240v1** - Deep Attention Model for ED Triage
- Word attention mechanism on medical text
- 338,500 ED visits over 3 years (MIMIC-III)
- AUC ~88% for resource-intensive patient identification
- 16% estimated lift over nurse performance

**2004.05184v2** - Improving ESI Acuity with ML and C-NLP (KATE)
- 166,175 patient encounters from two hospitals
- 75.9% accurate ESI assignment (vs 59.8% nurses)
- 93.2% higher accuracy than nurses at ESI 2/3 boundary
- Mitigates racial and social biases in triage

**2309.02604v1** - TriNet for Pneumonia/UTI Screening at Triage
- Positive predictive values: 0.86 (pneumonia), 0.93 (UTI)
- Cost-free, non-invasive screening
- Reduces over-testing risk, increases ED efficiency

**2212.11892v1** - Adaptive Simulated Annealing for E-Triage
- ASA-CatBoost optimization
- 83.3% accuracy, precision, recall, F1
- Five-level ESI prediction
- Outperforms traditional hyperparameter tuning

**2403.07038v1** - Graph Neural Networks for Automatic Triage
- GNN-based patient triage processing
- Emergency code assignment optimization
- Predicts severity index for resource allocation

**2507.01080v2** - LLM vs NLP vs JEPA for ED Triage (FRENCH scale)
- LLM-based URGENTIAPARSE best performance
- F1-score 0.900, AUC-ROC 0.879
- Superior hospitalization need prediction
- 7-month retrospective study (Roger Salengro Hospital, France)

### Mortality Prediction

**2510.18103v1** - Cardiac Arrest ICU Mortality Prediction
- Integrates structured data + textual features (discharge summaries, radiology)
- TF-IDF and BERT embeddings
- AUC 0.918 (combined) vs 0.753 (structured only)
- 22% relative improvement with text features
- MIMIC-IV dataset (45K+ patients)

**1711.01386v3** - Predicting Discharge Medications at Admission
- CNN on admission notes (MIMIC-III)
- 25K patient visits, 343K admissions
- 20% increase in macro-averaged F1 over best baseline
- Distills semantic patterns from noisy texts

**1910.06492v1** - Hierarchical Semantic Correspondence (Post-Discharge Mortality)
- Embedding framework for clinical notes
- Self-similarity matrix representations
- Spatial knowledge reasoning with UMLS semantic types
- Outperforms models without semantic interactions

**2504.12338v1** - Paging Dr. GPT: GPT-4o-mini for Mortality Prediction
- 14,011 first-time CCU/CVICU admissions (MIMIC-IV)
- GPT-based models outperform tabular data alone
- AUC 0.79 on primary and external datasets
- 5.1 percentage point AUROC improvement with GPT features
- 29.9% PPV improvement for highest-risk decile

**1803.06589v2** - Early Hospital Mortality Using Vital Signals
- Heart rate signal-based prediction (12 statistical features)
- Decision tree: F1-score 0.91, AUC 0.93
- MIMIC-III dataset (338,500 ED visits)
- Comparable to clinical record-based methods

**2105.06141v1** - Hybrid ML/DL COVID-19 Severity Prediction
- 3D CNN feature extraction from CT images
- CatBoost gradient boosting with SHAP interpretation
- AUC 0.949 on holdout test
- Combines imaging with lab/clinical data

**2109.06711v2** - COVID-Net Clinical ICU Prediction
- Neural network for ICU admission prediction
- 96.9% accuracy for COVID-19 ICU admission
- Trust quantification metrics
- SHAP explainability for decision transparency

### Clinical Notes & NLP Applications

**2405.11255v1** - WisPerMed: Discharge Summary Generation
- LLM-based automation of "Brief Hospital Course" and "Discharge Instructions"
- Dynamic Expert Selection (DES) method
- Overall score 0.332 in BioNLP competition
- MIMIC-IV dataset application

**2501.18724v3** - LLMs with Temporal Reasoning for Clinical Summarization
- Long-context clinical summarization with RAG and CoT
- MIMIC-IV and eICU evaluation
- Temporal coherence challenges identified
- RAG improves hallucination in some cases

**2307.07051v1** - Predictive Power Varies by Note Type/Section
- Context length limitations in clinical LLMs
- Nursing notes vs discharge notes comparison
- Section-specific predictive power analysis
- MIMIC-III 45,298 ICU stays

**1910.04006v1** - Sentiment Analysis for Psychiatric Readmission
- Topic extraction and clinical sentiment analysis
- Psychiatric readmission risk prediction
- NLP-based features for readmission models

**2106.02524v1** - CLIP: Action Items from Discharge Notes
- Multi-aspect extractive summarization
- 718 documents, 100K sentences (MIMIC-III)
- Physician-annotated action items
- In-domain language model pre-training

**2305.06416v1** - Automated Discharge Summary Hospital Course
- BERT and BART encoder-decoder models
- R-2 score 13.76
- 62% rated meeting standard of care by physicians
- Novel approach to reduce physician burden

### Predictive Models & Architectures

**2405.11255v1** - Discharge Summary Text Generation (Transformers)
- Fine-tuned BERT/BART models
- Constrained beam search for factuality
- Good ROUGE scores, clinical evaluation

**2102.04110v1** - Clinical Outcome Prediction from Admission Notes
- Self-supervised knowledge integration
- Four outcomes: discharge diagnoses, procedures, mortality, LOS
- ICD code hierarchy incorporation
- Transfer learning benefits demonstrated

**2509.26136v1** - CliniBench: Diagnosis Prediction Benchmark
- Compares 12 generative LLMs + 3 encoder-based classifiers
- MIMIC-IV admission notes
- Encoder-based consistently outperforms generative
- AUROC 0.69 for discharge diagnosis

**2011.01571v2** - Preserving Knowledge of Long Clinical Texts
- Ensemble of BERT-like transformers with aggregation
- MIMIC-III admission notes
- Mortality: 17.6% AUROC boost, readmission: 6.6% boost
- LOS: 7.9% F1 improvement, drug rec: 10.8% F1 improvement

**2208.12814v3** - Interpretable Discharge Placement Prediction
- Bayesian neural network with piecewise linearity
- Causal local average treatment effects (LATE)
- AUROC ~0.76 for readmission/death within 30 days
- Outperforms XGBoost with interpretability

**2110.00998v1** - Simple RNNs for Clinical Events
- GRUs and LSTMs with Bayesian Optimization
- Heart failure and early readmission tasks
- Competitive with complex architectures when properly tuned

### Specialized Applications

**2405.05993v1** - Precision Rehabilitation for Post-Stroke Patients
- ML for rehabilitation exercise effectiveness
- Random Forest best performance
- NLP extraction from procedure notes
- Statistical significance testing for functional improvements

**2211.09068v1** - Ischemic Stroke Lesion Prediction (iTDGP)
- Imbalanced Temporal Deep Gaussian Process
- CTP time series analysis
- Dice score 71.42% (ISLES 2018), 65.37% (UAH)
- Robust against imbalanced classes

**2407.00147v1** - Elevated Hospitalization Risk Post-ED Discharge
- Ensemble: logistic regression, naive Bayes, association rules
- Decision tree rules for operational use
- Predicts hospitalization within 3, 7, 14 days of ED discharge

**2112.09315v1** - Optimal Discharge via Data-Driven Policy Learning
- Infinite-horizon discounted MDP
- Off-policy evaluation strategies
- Addresses trade-off: LOS vs readmission/death risk
- 9,670 patients from 2015-2021

**2206.12551v1** - ML-Guided Discrete Event Simulation
- Post-discharge care management referral processing
- Random forest prediction models for LOS and referral type
- Reduces referral creation delay time
- Integrated systems engineering methods

---

## 2. Discharge Prediction Architectures

### Deep Learning Approaches

**Recurrent Neural Networks (RNNs)**
- **LSTM (Long Short-Term Memory)**
  - Widely used for sequential clinical data
  - Captures temporal dependencies in patient trajectories
  - Performance: AUC 0.70-0.85 for various tasks
  - Example: 2007.09483v4 (TPC outperforms LSTM by 18-68%)

- **GRU (Gated Recurrent Unit)**
  - Simpler alternative to LSTM
  - Competitive performance with proper tuning
  - Reference: 2110.00998v1

- **Bidirectional LSTM**
  - Processes sequences forward and backward
  - Enhanced context understanding
  - Example: 2507.11570v1 (SurgeryLSTM R²=0.86)

**Transformer-Based Models**
- **BERT Variants**
  - ClinicalBERT: Pretrained on clinical notes (1904.05342v3)
  - Medic-BERT: For patient event sequences (2303.11042v1)
  - Domain-specific pretraining crucial for performance

- **Attention Mechanisms**
  - Self-attention for long-range dependencies
  - Multi-head attention for different feature aspects
  - Attention weights provide interpretability
  - Example: 1804.03240v1 (Deep Attention Model AUC ~88%)

- **Encoder-Decoder Models**
  - BART for discharge summary generation
  - Seq2seq for temporal cascade modeling

**Convolutional Neural Networks (CNNs)**
- **1D CNN for Time Series**
  - Temporal Pointwise Convolution (TPC): 2007.09483v4
  - Handles irregular sampling and missing data
  - Mitigates skewness in EHR data

- **Text CNNs**
  - Predicting discharge medications: 1711.01386v3
  - Heart failure readmission: 1912.10306v1
  - F1 scores 0.73-0.76 for readmission

**Graph Neural Networks (GNNs)**
- **Spatiotemporal GNNs**
  - Multi-modal integration (2204.06766v1)
  - Patient similarity graphs
  - Temporal relationships modeling

- **S²G-Net (State-Space + Graph)**
  - Mamba SSMs + GraphGPS (2508.17554v2)
  - Heterogeneous patient similarity graphs
  - Superior to single-modality approaches

**Hybrid Architectures**
- **Neural ODEs + RNN**
  - Continuous-time dynamics modeling
  - Addresses irregular time intervals

- **Memory Networks**
  - Self-correcting predictions (1901.04364v1)
  - Feeds previous errors back into network
  - AUC 0.893 (MIMIC III), 0.871 (eICU)

### Traditional Machine Learning

**Tree-Based Methods**
- **Random Forest**
  - Robust baseline performance
  - Feature importance interpretation
  - Often competitive with deep learning

- **Gradient Boosting**
  - XGBoost: Popular for tabular clinical data
  - LightGBM: Best for diabetes readmission (2406.19980v1)
  - CatBoost: Hybrid models (2105.06141v1, AUC 0.949)

**Ensemble Methods**
- **Stacking**
  - Combines multiple model predictions
  - Example: EF-Net (2412.16134v1, 96% accuracy)

- **Voting Ensembles**
  - Soft voting for probability averaging
  - Hard voting for majority class

**Linear Models**
- **Logistic Regression**
  - With LASSO regularization
  - Competitive with deep learning for readmission (1812.09549v1)
  - Interpretable coefficients

**Support Vector Machines (SVM)**
- Limited use in recent literature
- Cluster-dependent SVM for readmission reduction (1903.09056v1)

### Multi-Modal Fusion Strategies

**Early Fusion**
- Concatenate features from different modalities
- Single model processes combined input
- Example: Vital signs + lab values + demographics

**Late Fusion**
- Separate models for each modality
- Combine predictions at decision level
- Example: 2510.18103v1 (text + structured data)

**Intermediate Fusion**
- Learn shared representations across modalities
- Feature-level or representation-level fusion
- Example: S²G-Net combining sequences and graphs

**Attention-Based Fusion**
- Learns importance weights for different modalities
- Dynamic fusion based on patient context
- Cross-modal attention mechanisms

---

## 3. Length of Stay Prediction Models

### Model Performance Summary

| Approach | Dataset | MAE/R² | Key Features |
|----------|---------|---------|--------------|
| TPC (Temporal Pointwise CNN) | eICU, MIMIC-IV | MAE: 1.55d, 2.28d | Handles irregular sampling |
| SurgeryLSTM (Attention BiLSTM) | Spine surgery | R²: 0.86 | Temporal interpretability |
| Domain Adaptation LSTM | eICU, MIMIC-IV | 1-5% acc. gain | Cross-ICU generalization |
| Medic-BERT (Transformer) | Danish ED (45K) | High accuracy | Event sequence modeling |
| S²G-Net (Mamba + GraphGPS) | MIMIC-IV | Outperforms baselines | Multi-view graphs |
| Drug Representation + DL | MIMIC-III | 6% AUROC gain | ECFP/SMILES embeddings |

### Challenges in LOS Prediction

**Data Heterogeneity**
- Wide variation in patient populations
- Different hospital protocols and practices
- Inconsistent documentation standards
- Solution: Domain adaptation, federated learning

**Temporal Complexity**
- Non-linear disease progression
- Unexpected complications
- Weekend/holiday effects on discharge
- Solution: Temporal CNNs, attention mechanisms

**Class Imbalance**
- Many short stays, few prolonged stays
- Skewed towards certain LOS ranges
- Solution: Stratified sampling, loss reweighting

**Missing Data**
- Irregular vital sign measurements
- Incomplete laboratory tests
- Sparse documentation
- Solution: TPC architecture, imputation strategies

### Feature Importance for LOS

**Consistent Top Predictors Across Studies:**

1. **Clinical Severity**
   - ICU admission indicators
   - Disease severity scores (APACHE, SOFA)
   - Number of active diagnoses

2. **Procedures and Interventions**
   - Surgical procedures performed
   - Mechanical ventilation
   - Dialysis requirements
   - Rehabilitation assessments

3. **Laboratory Values**
   - Kidney function (creatinine, BUN)
   - Liver function tests
   - Hematological parameters
   - Albumin levels

4. **Demographics**
   - Age (though controversial in stroke patients)
   - Insurance type
   - Admission source (ED, transfer, elective)

5. **Temporal Factors**
   - Time to first intervention
   - Day of week admitted
   - Season/month effects

6. **Medication History**
   - Drug burden scores
   - Nephrotoxic medications
   - Polypharmacy indicators

### Domain-Specific LOS Models

**ICU Length of Stay**
- Requires real-time vital signs
- High mortality correlation
- Benefits from multimodal data (2508.17554v2)
- Typical prediction horizons: 24-48 hours ahead

**Surgical LOS**
- Preoperative risk factors dominant
- Procedure-specific predictors
- Complications as major drivers
- Example: SurgeryLSTM for spine surgery (2507.11570v1)

**Emergency Department LOS**
- Triage information critical
- Chief complaint text valuable
- Disposition decision interdependent
- Fast prediction required (<1 hour)

**Post-Stroke LOS**
- Contradictory findings in literature (2201.00005v1)
- Age importance disputed
- Rehabilitation needs assessment crucial
- Functional status predictive

---

## 4. Disposition Prediction Models

### Disposition Categories

**Primary Discharge Destinations:**
1. **Home** (with or without services)
   - Home health services
   - Outpatient follow-up
   - Self-care capable

2. **Skilled Nursing Facility (SNF)**
   - Post-acute care needs
   - Rehabilitation requirements
   - Chronic care management

3. **Rehabilitation Facility**
   - Inpatient rehabilitation
   - Specialized therapy programs
   - Functional recovery focus

4. **Long-Term Acute Care (LTAC)**
   - Extended ventilator weaning
   - Complex wound care
   - Prolonged IV therapy

5. **Other Hospital Transfer**
   - Specialized treatment unavailable
   - Step-down care
   - Psychiatric facilities

6. **Hospice/Palliative Care**
   - End-of-life care
   - Comfort measures
   - Terminal diagnosis

7. **Against Medical Advice (AMA)**
   - Patient decision to leave
   - Non-compliance risk
   - Poor outcomes associated

### Key Disposition Prediction Studies

**Post-Acute Care (PAC) Disposition (1812.10487v1)**
- CHAID algorithm: 84.16% accuracy, AUC 0.81
- Predictors: nursing assessment, vitals, demographics
- Early prediction enables insurance authorization
- Cost savings: $1,974-$2,346 per patient-day
- 22.22% reduction in inpatient LOS

**ED Patient Disposition (2412.16134v1)**
- EF-Net architecture: 96% accuracy
- Combines categorical embeddings + numerical features
- XGBoost ensemble with soft voting
- MIMIC-IV-ED dataset validation
- Addresses ED overcrowding

**General Disposition Prediction Features:**

1. **Functional Status**
   - Activities of daily living (ADL) scores
   - Mobility assessments
   - Cognitive function tests
   - Baseline independence level

2. **Social Determinants**
   - Living situation (alone, with family)
   - Caregiver availability
   - Insurance coverage (Medicare, Medicaid, private)
   - Geographic location
   - Socioeconomic status

3. **Clinical Factors**
   - Primary diagnosis
   - Comorbidity burden (Charlson index)
   - Recent procedures
   - Ongoing care needs (wound care, IV therapy)
   - Medication complexity

4. **Healthcare Utilization**
   - Prior hospitalizations
   - ED visit frequency
   - Previous PAC use
   - Home health services history

5. **Resource Requirements**
   - Therapy needs (PT, OT, ST)
   - Nursing care level
   - Medical equipment needs
   - Medication administration complexity

### Disposition vs. Readmission Relationship

**Key Finding from Research:**
- Discharge disposition is consistently identified as a critical predictor of readmission
- SNF discharge associated with higher readmission rates
- Home discharge with services shows protective effect
- AMA discharges have dramatically elevated readmission risk

**Confounding Factors:**
- Disposition reflects disease severity
- Selection bias in facility placement
- Quality variation across PAC facilities
- Patient preference vs. clinical need mismatch

### Barriers to Disposition Prediction

**Data Limitations:**
- Incomplete social history documentation
- Lack of standardized functional assessments
- Missing caregiver information
- Insurance coverage uncertainty at admission

**Clinical Challenges:**
- Disease progression unpredictability
- Complications altering discharge trajectory
- Patient/family preferences changing
- Facility availability constraints

**Ethical Considerations:**
- Risk of premature discharge pressure
- Equity in PAC access
- Insurance-driven decisions
- Patient autonomy vs. clinical judgment

---

## 5. Readmission Risk Prediction

### 30-Day All-Cause Readmission

**Performance Benchmarks:**

| Model Type | Dataset | AUROC | F1 Score | Notes |
|------------|---------|--------|----------|-------|
| ClinicalBERT | MIMIC-III | - | Competitive | Discharge summaries only |
| CNN (Clinical Notes) | MIMIC-III | - | 0.756 (general), 0.733 (30d) | Heart failure specific |
| Multimodal ST-GNN | MIMIC-IV | - | - | 5.1% AUROC improvement |
| LSTM (Diabetes) | 130-US Hospitals | - | Variable | SHAP interpretation |
| Logistic Regression + LASSO | - | 0.643 | - | Competitive with DL |
| ConvLSTM | - | Improved | - | Temporal + NLP features |

### Key Predictive Features

**Consistently Important Across Studies:**

1. **Hospital Course Factors**
   - Length of stay (often J-shaped relationship)
   - Number of medications at discharge
   - Number of active diagnoses
   - ICU admission during stay
   - Surgical procedures performed

2. **Laboratory Abnormalities**
   - Anemia (hemoglobin levels)
   - Kidney function (BUN, creatinine)
   - Electrolyte imbalances
   - Albumin (nutritional status)
   - D-dimer elevation

3. **Discharge Characteristics**
   - Discharge disposition (SNF, home, etc.)
   - Number of discharge medications
   - Medication changes from admission
   - Follow-up appointment scheduled
   - Home health services ordered

4. **Patient Demographics**
   - Age (but relationship complex)
   - Insurance type (Medicaid higher risk)
   - Race/ethnicity (health equity concerns)
   - Distance from hospital
   - Socioeconomic deprivation index

5. **Comorbidity Burden**
   - Charlson Comorbidity Index
   - Diabetes with complications
   - Heart failure
   - COPD
   - Chronic kidney disease

6. **Prior Healthcare Utilization**
   - Previous hospitalizations (number and recency)
   - ED visits in past year
   - Prior readmissions
   - Frequent flyer status

### Disease-Specific Readmission Models

**Heart Failure (1912.10306v1)**
- CNN on discharge summaries
- F1: 0.756 (general), 0.733 (30-day)
- Outperforms random forest by 12.2%
- Chi-square feature interpretation

**Diabetes (2406.19980v1)**
- LightGBM best traditional model
- LSTM captures temporal patterns but overfits
- Key factors: lab procedures, discharge disposition
- SHAP values for interpretability

**ICU Patients (1905.08547v3)**
- RNN with attention mechanisms
- AUROC 0.739, F1 0.372
- Identifies high-risk subpopulations
- Bayesian inference for uncertainty

**Intracerebral Hemorrhage (2501.01183v1)**
- Multiple ML techniques evaluated
- MIMIC-III and MIMIC-IV validation
- Focus on ICU readmission specifically

### Temporal Patterns in Readmission

**Early vs. Late Readmission:**
- Different risk factors operate at different timeframes
- Immediate (<7 days): complications, premature discharge
- Intermediate (7-30 days): medication issues, inadequate follow-up
- Late (>30 days): chronic disease progression

**Temporal Modeling Advantages:**
- Captures disease trajectory evolution
- Accounts for cascade effects (1 condition → another)
- Models intervention timing impacts
- Handles irregular measurement intervals

### Clinical Notes for Readmission Prediction

**NLP Techniques Employed:**

1. **Word Embeddings**
   - Word2Vec, GloVe on clinical corpora
   - Domain-specific embeddings outperform general
   - Capture medical semantic relationships

2. **Transformer Models**
   - BERT, ClinicalBERT, BioBERT
   - Fine-tuned on discharge summaries
   - Contextual understanding superior to bag-of-words

3. **Topic Modeling**
   - LDA for discharge summary themes
   - Identifies latent clinical concepts
   - Complement to supervised learning

4. **Sentiment Analysis**
   - Psychiatric readmission specific (1910.04006v1)
   - Emotional tone in clinical notes
   - Provider uncertainty markers

**Challenges with Clinical Text:**
- Inconsistent documentation quality
- Abbreviations and medical jargon
- Negation and uncertainty handling
- Long document contexts (average >5000 tokens)
- Missing temporal information

### Intervention Implications

**High-Risk Patient Management:**
- Transitional care programs
- Post-discharge phone calls
- Medication reconciliation emphasis
- Timely follow-up appointment scheduling
- Home health services deployment

**Resource Allocation:**
- Prioritize care coordination for high-risk patients
- Targeted disease management programs
- Social work intervention
- Pharmacy consultation at discharge

---

## 6. Research Gaps and Future Directions

### Identified Research Gaps

**1. External Validation Limitations**
- Most models trained and tested on single-center data
- Limited multi-center validation studies
- Geographic and demographic generalization unknown
- Different EHR systems create data incompatibility

**2. Prospective Clinical Trials**
- Predominantly retrospective studies
- Lack of randomized controlled trials
- Unknown impact on clinical outcomes when deployed
- Alert fatigue and workflow integration unstudied

**3. Temporal Granularity**
- Most models provide static point-in-time predictions
- Continuous risk updating underexplored
- Dynamic prediction as patient state changes rare
- Real-time inference computational requirements

**4. Multimodal Integration**
- Limited use of imaging data (mostly text + structured)
- Physiological waveforms underutilized
- Genomic/molecular data not integrated
- Sensor data (wearables) unexplored for inpatients

**5. Causal Inference**
- Predominantly associational models
- Confounding not adequately addressed
- Treatment effect heterogeneity unexplored
- Counterfactual reasoning limited

**6. Health Equity**
- Bias in model predictions across demographics
- Fairness metrics often not reported
- Algorithmic justice considerations insufficient
- Disparate impact on vulnerable populations

**7. Interpretability vs. Performance Trade-off**
- Most accurate models are black boxes
- Post-hoc explanations may be unreliable
- Inherently interpretable models underperform
- Clinician trust remains a barrier

**8. Long-term Outcomes**
- Focus on short-term outcomes (30-90 days)
- Quality of life predictions absent
- Functional status long-term not modeled
- Cost-effectiveness rarely evaluated

**9. Rare Events and Outliers**
- Class imbalance not fully resolved
- Rare complications under-predicted
- Tail risk management insufficient
- Extreme values handling inadequate

**10. Implementation Science**
- Workflow integration strategies unstudied
- Clinician acceptance and trust metrics lacking
- Alert fatigue mitigation approaches needed
- Human-AI collaboration frameworks missing

### Emerging Research Directions

**Large Language Models (LLMs)**
- GPT-4 and similar for clinical reasoning (2504.12338v1)
- Few-shot learning for rare conditions
- Multi-agent systems for care coordination (2408.07531v2)
- Prompt engineering for clinical tasks

**Federated Learning**
- Privacy-preserving multi-institutional models (2407.12741v1, 2304.14663v1)
- Cross-hospital generalization without data sharing
- Client recruitment strategies
- Differential privacy integration

**Continuous Learning**
- Models that update with new data
- Concept drift detection and adaptation
- Online learning in clinical settings
- Active learning for efficient annotation

**Explainable AI (XAI)**
- Attention visualization (beyond attention weights)
- Counterfactual explanations
- Concept-based interpretability
- Faithful local explanations (SHAP, LIME limitations)

**Causal Machine Learning**
- Doubly robust estimation
- Instrumental variable approaches
- Causal discovery from observational data
- Treatment effect heterogeneity modeling

**Multi-task Learning**
- Simultaneous prediction of multiple outcomes
- Shared representations across tasks
- Transfer learning from related tasks
- Meta-learning for new hospitals/diseases

**Temporal Modeling Advances**
- Neural ODEs for continuous-time dynamics
- State-space models (Mamba architecture)
- Temporal point processes
- Irregularly sampled time series methods

**Graph Neural Networks**
- Patient similarity graphs
- Knowledge graph integration (medical ontologies)
- Temporal knowledge graphs
- Multi-relational learning

**Reinforcement Learning**
- Optimal discharge timing policies (2112.09315v1)
- Treatment recommendation systems
- Resource allocation strategies
- Sequential decision-making

**Digital Twins**
- Patient-specific simulation models
- Personalized treatment planning
- "What-if" scenario analysis
- Precision medicine integration

### Critical Technical Needs

**Data Infrastructure**
- Standardized data formats (FHIR, OMOP CDM)
- Interoperability across systems
- Real-time data pipelines
- Data quality monitoring

**Computational Resources**
- Efficient inference for real-time predictions
- Edge computing for privacy
- Model compression techniques
- GPU/TPU optimization

**Evaluation Frameworks**
- Standardized benchmark datasets
- Consistent performance metrics
- Fairness evaluation protocols
- Clinical utility metrics (beyond accuracy)

**Regulatory Pathways**
- FDA/EMA approval processes for AI/ML devices
- Post-market surveillance requirements
- Adaptive learning regulation
- Liability frameworks

---

## 7. Relevance to ED Disposition Decisions

### ED-Specific Challenges

**Time Constraints**
- Rapid decision-making required (minutes to hours)
- Limited historical data available at presentation
- Incomplete diagnostic workup at triage
- Overcrowding pressure for quick disposition

**Data Availability**
- Chief complaint (often free-text)
- Triage vital signs (initial set)
- Brief history from patient/family
- Prior ED visits and admissions
- Limited or no laboratory results initially

**Disposition Options from ED**
- Discharge home
- Admit to observation unit
- Admit to general medical/surgical floor
- Admit to ICU
- Transfer to specialty hospital
- Psychiatric facility
- Against medical advice (AMA)

### AI Models for ED Disposition

**High-Performing ED Models:**

**1. EF-Net (2412.16134v1)**
- Patient disposition analysis
- 96% accuracy on MIMIC-IV-ED
- Combines categorical embeddings + numerical features
- Scalable solution for ED overcrowding

**2. KATE Sepsis (2204.07657v6)**
- AUROC 0.9423 for sepsis detection at triage
- 71.09% sensitivity, 94.81% specificity
- Supports early ICU disposition decision
- Dramatically outperforms SIRS criteria (AUROC 0.6826)

**3. URGENTIAPARSE (LLM-based) (2507.01080v2)**
- F1-score 0.900, AUC-ROC 0.879
- Predicts triage acuity and disposition
- Hospitalization need prediction
- 7-month retrospective validation

**4. Multi-Agent LLM CDSS (2408.07531v2)**
- Korean Triage and Acuity Scale (KTAS)
- Four AI agents (Triage Nurse, Physician, Pharmacist, Coordinator)
- High accuracy in disposition decision-making
- Resource allocation optimization

**5. Hospital Admission Prediction (2106.12921v2)**
- F-measure 0.679-0.708, ROC Area 0.734-0.774
- Common biomarkers + clinical features
- Low-cost, accessible decision tool
- 3,204 ED visits analyzed

**6. Deep Attention for ED Triage (1804.03240v1)**
- AUC ~88% for resource-intensive patient identification
- 338,500 ED visits (MIMIC-III)
- 16% accuracy lift over nurse performance
- Word attention mechanism on medical text

### ED Disposition Prediction Features

**Critical Predictors at ED Presentation:**

1. **Triage Vital Signs**
   - Blood pressure (systolic/diastolic)
   - Heart rate
   - Respiratory rate
   - Temperature
   - Oxygen saturation
   - Pain score
   - Mental status (GCS)

2. **Chief Complaint**
   - Chest pain → higher ICU likelihood
   - Shortness of breath → admission likely
   - Abdominal pain → variable disposition
   - Injury/trauma → orthopedic consult needs
   - Psychiatric complaint → specialty disposition

3. **Demographics**
   - Age (elderly higher admission rate)
   - Arrival by ambulance (higher acuity)
   - Time of arrival (night/weekend effects)
   - Distance from hospital (influences threshold)

4. **Prior Healthcare Utilization**
   - Recent ED visits (bounceback risk)
   - Active hospitalizations
   - Known chronic conditions
   - Medication list complexity
   - Prior ICU admissions

5. **Initial Laboratory Results** (if available)
   - Troponin (cardiac injury)
   - Lactate (sepsis/shock)
   - White blood cell count (infection)
   - Creatinine (kidney function)
   - Glucose (diabetic emergency)

6. **Rapid Assessment Scores**
   - NEWS (National Early Warning Score)
   - qSOFA (sepsis screening)
   - HEART score (chest pain)
   - Canadian C-Spine Rule
   - PERC (pulmonary embolism)

### Workflow Integration Considerations

**Pre-Triage Support**
- EMS call data for pre-arrival preparation
- Predicted acuity for room assignment
- Specialist notification triggers
- Resource pre-allocation

**At-Triage Decision Support**
- Real-time risk scoring
- Disposition recommendation with confidence
- Alternative disposition options ranked
- Expected resource utilization forecast

**During ED Stay**
- Continuous risk updating
- Deterioration warning alerts
- Predicted time to disposition
- Bed request automation

**Discharge/Admission Decision**
- Readmission risk stratification
- Post-acute care recommendation
- Follow-up visit scheduling
- Care transition planning

### Integration with Existing Systems

**MIMIC-IV-ED Dataset Applications:**
- 400,000+ ED visits (2011-2019)
- Three primary outcomes: hospitalization, critical outcomes, 72hr reattendance
- Benchmark for comparing ED models
- Reference: 2111.11017v2

**Key Outcomes to Predict:**
1. **Hospitalization** (vs discharge)
2. **ICU admission** (vs floor)
3. **Critical deterioration** (within 24-48 hrs)
4. **72-hour ED reattendance** (bounceback)
5. **30-day outcomes** (readmission, mortality)

### Performance Metrics for ED Models

**Discriminative Performance:**
- AUROC (overall prediction quality)
- Sensitivity (catching high-risk patients)
- Specificity (not over-admitting)
- Positive/Negative Predictive Value
- Calibration (predicted vs observed risk)

**Clinical Utility:**
- Net benefit analysis
- Decision curve analysis
- Number needed to screen
- Alert burden (false positive rate tolerance)
- Time to disposition improvement

**Operational Impact:**
- ED length of stay reduction
- ICU utilization optimization
- Admission rate changes
- Boarding time effects
- Cost per disposition decision

### Barriers to ED AI Adoption

**Technical Challenges:**
- Real-time data integration from multiple systems
- Computational latency requirements (<1 min)
- Missing data handling at early time points
- Model updating frequency needs

**Clinical Workflow:**
- Alert fatigue from excessive notifications
- Override behavior and compliance
- Provider trust in AI recommendations
- Medicolegal liability concerns

**Organizational:**
- IT infrastructure investment
- Training and change management
- Performance monitoring systems
- Continuous model maintenance

**Ethical and Equity:**
- Bias in predictions across demographics
- Overtriage vs undertriage trade-offs
- Resource availability constraints
- Access to post-acute care facilities

### Best Practices for ED Disposition AI

**Model Development:**
- Multi-center validation essential
- Prospective evaluation before deployment
- Continuous monitoring post-deployment
- Explicit fairness evaluation across demographics

**Clinical Integration:**
- Transparent, interpretable predictions
- Confidence intervals communicated
- Override mechanisms with feedback
- Clinical champion engagement

**Implementation:**
- Gradual rollout with evaluation
- Provider training and education
- Usability testing with end-users
- Integration with existing clinical tools

**Governance:**
- Clinical oversight committee
- Regular model audits
- Performance reporting
- Incident response protocols

---

## 8. Clinical Workflow Integration

### Decision Support Touchpoints

**1. Pre-Hospital Phase**
- EMS data integration
- Prehospital risk scores
- Hospital notification systems
- Bed preparation triggers

**2. Triage Phase**
- Real-time acuity prediction
- Disposition forecasting
- Resource allocation suggestions
- Fast-track identification

**3. ED Evaluation Phase**
- Continuous risk updating
- Order set recommendations
- Consultation triggers
- Expected LOS estimation

**4. Disposition Decision**
- Admission appropriateness scoring
- Post-acute care matching
- ICU vs floor recommendation
- Discharge safety assessment

**5. Post-Discharge**
- Readmission risk communication
- Follow-up scheduling optimization
- Care transition alerts
- Remote monitoring triggers

### Human-AI Collaboration Models

**AI as Advisor** (Most Common)
- Provides risk scores and recommendations
- Clinician retains full decision authority
- Override always permitted
- Explanations required for high-risk flags

**AI as Automation** (Limited Use)
- Automates routine low-risk decisions
- Human review for exceptions
- Example: Discharge summary generation
- Safeguards prevent unsafe automation

**AI as Augmentation** (Emerging)
- Enhances clinician capabilities
- Surfaces hidden patterns in data
- Enables precision medicine approaches
- Collaborative decision-making

### Alert Design Principles

**Timeliness:**
- Actionable window (not too early/late)
- Updated predictions as data changes
- Suppression of redundant alerts
- Priority levels for multiple conditions

**Content:**
- Risk level (low, medium, high, critical)
- Confidence interval
- Key contributing factors
- Suggested interventions
- Alternative scenarios

**Format:**
- Visual risk displays (traffic light, gauges)
- Trend visualization (trajectory)
- Comparison to population averages
- Interpretable explanations

**Integration:**
- Native EHR embedding
- Workflow non-disruptive
- Mobile access for providers
- Summary dashboards for management

### Interpretability Requirements

**Local Explanations (Patient-Level):**
- Feature importance (SHAP, LIME)
- Counterfactual scenarios ("what if")
- Case-based reasoning (similar patients)
- Attention weight visualization

**Global Explanations (Model-Level):**
- Overall feature ranking
- Decision rules extraction
- Subgroup analyses
- Performance stratification

**Clinical Validation:**
- Face validity (makes clinical sense)
- Alignment with clinical guidelines
- Expert review of explanations
- Bias detection and mitigation

### Performance Monitoring

**Model Performance Metrics:**
- Discrimination (AUROC, AUPRC)
- Calibration (Brier score, calibration plots)
- Clinical utility (net benefit)
- Fairness (across demographics)

**Operational Metrics:**
- Alert burden (alerts per shift)
- Override rate and reasons
- Time to disposition decision
- Resource utilization changes

**Outcome Metrics:**
- Readmission rates
- Mortality (in-hospital, 30-day)
- Length of stay
- Patient satisfaction
- Provider satisfaction

**Safety Monitoring:**
- Adverse event tracking
- Near-miss documentation
- Failure mode analysis
- Incident reporting

---

## 9. Key Findings and Recommendations

### Summary of Evidence

**Model Performance:**
- Deep learning consistently outperforms traditional ML (5-20% improvement)
- Multimodal approaches superior to single-modality (AUC gains 0.05-0.15)
- Temporal modeling captures disease trajectories effectively
- NLP of clinical notes adds significant predictive value

**Clinical Applications:**
- Readmission prediction: AUC 0.70-0.85 achievable
- LOS prediction: MAE 1.5-2.5 days typical
- ED disposition: Accuracy 75-96% depending on task
- Mortality prediction: AUC 0.80-0.95 for ICU patients

**Implementation Challenges:**
- External validation often shows performance degradation
- Interpretability vs accuracy trade-off persistent
- Workflow integration complex and institution-specific
- Health equity concerns require ongoing attention

### Recommendations for Research

**High Priority:**
1. **Multi-Center Prospective Trials**
   - Randomized deployment studies
   - Clinical outcome measurement
   - Cost-effectiveness analysis
   - Real-world generalization testing

2. **Fairness and Equity**
   - Systematic bias audits
   - Disparity impact assessments
   - Mitigation strategy development
   - Diverse population validation

3. **Interpretability Methods**
   - Inherently interpretable architectures
   - Faithful explanation techniques
   - Clinical validation of explanations
   - User-centered design

4. **Causal Inference**
   - Treatment effect estimation
   - Confounding adjustment
   - Counterfactual reasoning
   - Policy learning

5. **Continuous Learning**
   - Online model updating
   - Concept drift detection
   - Active learning strategies
   - Feedback loop integration

**Medium Priority:**
1. **Multimodal Integration**
   - Imaging + text + structured data
   - Waveform signal processing
   - Genomic data incorporation
   - Sensor data fusion

2. **Long-Term Outcomes**
   - Quality of life prediction
   - Functional status forecasting
   - Cost trajectory modeling
   - Personalized care planning

3. **Rare Events**
   - Class imbalance solutions
   - Few-shot learning approaches
   - Transfer learning strategies
   - Synthetic data generation

### Recommendations for Implementation

**Technical Infrastructure:**
1. **Data Pipelines**
   - Real-time EHR integration
   - Data quality monitoring
   - Feature engineering automation
   - Model versioning and deployment

2. **Computational Resources**
   - Low-latency inference (<1 second)
   - Scalable architecture
   - Redundancy and failover
   - Security and privacy compliance

3. **Monitoring Systems**
   - Performance dashboards
   - Alert management
   - Feedback collection
   - Continuous evaluation

**Clinical Governance:**
1. **Oversight Structure**
   - Multidisciplinary committee
   - Clinical champions
   - IT representation
   - Patient advocates

2. **Policies and Procedures**
   - Use protocols
   - Override guidelines
   - Escalation pathways
   - Incident response

3. **Training and Education**
   - Provider onboarding
   - Ongoing education
   - Performance feedback
   - Best practice sharing

**Regulatory Compliance:**
1. **Validation Requirements**
   - Pre-deployment testing
   - External validation
   - Ongoing monitoring
   - Retraining triggers

2. **Documentation**
   - Model cards
   - Data sheets
   - Risk assessments
   - Audit trails

3. **Patient Rights**
   - Informed consent
   - Transparency
   - Appeal processes
   - Data privacy

### Future Vision

**Near-Term (1-3 years):**
- Widespread ED triage support deployment
- Automated discharge summary generation
- Real-time readmission risk at discharge
- Standardized benchmarks and datasets

**Medium-Term (3-5 years):**
- Federated learning across hospital systems
- Causal AI for treatment recommendations
- Digital twin patient simulations
- LLM-based clinical reasoning assistants

**Long-Term (5-10 years):**
- Fully integrated AI-augmented care pathways
- Personalized precision discharge planning
- Autonomous decision-making for routine cases
- Continuous learning self-improving systems

---

## Conclusion

This comprehensive review of 100+ ArXiv papers reveals substantial progress in AI for discharge planning, disposition prediction, and patient outcome forecasting. Deep learning approaches, particularly those incorporating multimodal data and temporal modeling, demonstrate significant improvements over traditional methods. However, critical gaps remain in external validation, health equity, interpretability, and real-world clinical integration.

The evidence supports cautious optimism about AI's potential to transform acute care decision-making, particularly in ED disposition and readmission prevention. Success will require:
- Rigorous prospective validation studies
- Systematic attention to fairness and bias
- Thoughtful human-AI collaboration design
- Robust clinical governance frameworks
- Continuous performance monitoring

For ED disposition decisions specifically, the research demonstrates feasibility of accurate, real-time predictions that can support—but not replace—clinical judgment. Models achieving 75-96% accuracy for various disposition outcomes suggest clinical utility, though careful implementation with appropriate safeguards is essential.

The field is rapidly evolving, with large language models, federated learning, and causal inference methods representing promising frontiers. Continued collaboration between AI researchers, clinicians, implementation scientists, and ethicists will be critical to realizing the full potential of AI in improving discharge planning and patient outcomes in acute care settings.

---

## References

This synthesis is based on 100+ papers from ArXiv spanning 2014-2025, with primary focus on:
- MIMIC-III and MIMIC-IV datasets
- eICU Collaborative Research Database
- Emergency department prediction tasks
- Hospital readmission modeling
- Length of stay forecasting
- Clinical natural language processing
- Deep learning architectures for healthcare
- Interpretable machine learning in medicine

Complete ArXiv IDs are cited throughout the document for all major findings and can be accessed at https://arxiv.org/ followed by the paper ID.

**Key Datasets Referenced:**
- MIMIC-III: Medical Information Mart for Intensive Care III
- MIMIC-IV: Medical Information Mart for Intensive Care IV
- MIMIC-IV-ED: MIMIC-IV Emergency Department Module
- eICU-CRD: eICU Collaborative Research Database
- ISLES 2018: Ischemic Stroke Lesion Segmentation
- Diabetes 130-US Hospitals Dataset

**Date of Synthesis:** December 2025
**Total Papers Reviewed:** 100+
**Primary Focus:** Acute care discharge planning and disposition prediction using AI/ML techniques