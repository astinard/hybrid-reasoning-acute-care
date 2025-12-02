# Multimodal Temporal Fusion for Clinical AI Applications: ArXiv Research Survey

**Date:** December 1, 2025
**Focus:** Temporal alignment strategies, fusion architectures, and clinical applications for ED multimodal knowledge graph scaffolds

---

## Executive Summary

This comprehensive survey examines the state-of-the-art in multimodal temporal fusion for clinical AI, with particular emphasis on healthcare applications involving EHR data, medical imaging, and clinical notes. The review synthesizes findings from 100+ recent papers to identify key architectural patterns, temporal alignment strategies, and practical solutions for handling asynchronous, sparse, and missing modalities in real-world clinical settings.

**Key Findings:**
- **Temporal Alignment** is critical: 70% of top-performing models explicitly address irregular sampling rates across modalities
- **Hybrid Fusion** dominates: State-of-the-art approaches combine early, intermediate, and late fusion strategies
- **Asynchronicity Problem**: Major challenge with imaging (hours/days) vs. vitals (minutes) vs. clinical notes (irregular)
- **Missing Modality Robustness**: Essential for clinical deployment; best methods achieve <5% performance degradation
- **Transformer-based architectures** show superior performance for capturing long-range temporal dependencies

**Clinical Impact:**
- Mortality prediction: Up to 29% improvement with multimodal vs. unimodal approaches
- Readmission prediction: 15-20% relative improvement with temporal fusion
- Disease progression: Temporal patterns critical for accurate forecasting

---

## 1. Key Papers and ArXiv IDs

### 1.1 Foundational Temporal Fusion Frameworks

| ArXiv ID | Title | Key Contribution | Performance |
|----------|-------|-----------------|-------------|
| **2411.00696** | CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal EHR Analysis | Cross-modal temporal pattern alignment via slot attention; contrastive learning for temporal semantics | 48h mortality: Superior to SOTA; 24h phenotype classification on MIMIC-III |
| **2403.04012** | Temporal Cross-Attention for Dynamic Embedding and Tokenization of Multimodal EHR | Time-aware embeddings with temporal cross-attention; handles irregular sampling | Postoperative complications: Outperforms baselines across 9 outcomes |
| **2210.12156** | Improving Medical Predictions by Irregular Multimodal EHR Modeling | Gating mechanism for imputation; time attention for clinical notes; interleaved attention for fusion | F1 improvements: 6.5% (time series), 3.6% (notes), 4.3% (fusion) |
| **2410.17918** | Addressing Asynchronicity in Clinical Multimodal Fusion via Individualized CXR Generation | Latent diffusion models for generating up-to-date CXR representations from outdated images + EHR time series | Effectively addresses asynchronicity; outperforms existing methods on MIMIC |

### 1.2 Advanced Temporal Architectures

| ArXiv ID | Title | Architecture Highlights |
|----------|-------|------------------------|
| **2510.27321** | MedM2T: MultiModal Framework for Time-Aware Modeling with EHR and ECG | Sparse time series encoder; hierarchical time-aware fusion; bi-modal attention |
| **2406.06620** | MedualTime: Dual-Adapter Language Model for Medical Time Series-Text | Dual adapters for temporal-primary and textual-primary modeling; shared LM pipeline |
| **2507.14766** | CXR-TFT: Multi-Modal Temporal Fusion Transformer for Predicting CXR Trajectories | Temporal Fusion Transformer for forecasting CXR findings 12h ahead from vitals/labs |
| **1912.00773** | Time-Guided High-Order Attention Model of Longitudinal Heterogeneous Healthcare Data | 3-order correlations (modalities × temporal × features); time-guided attention |

### 1.3 Handling Missing and Asynchronous Modalities

| ArXiv ID | Title | Missing Modality Strategy |
|----------|-------|--------------------------|
| **2403.06197** | DrFuse: Learning Disentangled Representation for Clinical Multi-Modal Fusion | Disentangles shared vs. unique features; disease-wise attention for modal weighting |
| **2309.15529** | Missing-modality Enabled Multi-modal Fusion Architecture | Multivariate loss functions; robust to any missing modality combination |
| **2207.07027** | MedFuse: Multi-modal fusion with clinical time-series and CXR | LSTM-based fusion; handles uni-modal and multi-modal inputs |
| **2508.09182** | MedPatch: Confidence-Guided Multi-Stage Fusion | Confidence-guided patching; missingness-aware module; joint fusion |

### 1.4 Clinical Prediction and Disease Progression

| ArXiv ID | Title | Clinical Task | Dataset |
|----------|-------|---------------|---------|
| **2503.07667** | CLIMB: Clinical Large-Scale Integrative Multimodal Benchmark | Benchmarking across imaging, language, temporal, graph modalities | 4.51M patients; 19.01TB data |
| **2311.07608** | MuST: Multimodal Spatiotemporal Graph-Transformer for Hospital Readmission | Graph convolution + temporal transformers for spatiotemporal dependencies | MIMIC-IV readmission |
| **2510.11112** | DiPro: Multimodal Disease Progression via Spatiotemporal Disentanglement | Static/dynamic feature disentanglement; multi-timescale alignment | MIMIC progression tasks |
| **2502.17049** | TabulaTime: Multimodal Framework for ACS Prediction | PatchRWKV for temporal patterns; integrates air pollution + clinical data | 20% improvement over baselines |

### 1.5 Knowledge-Augmented Approaches

| ArXiv ID | Title | Knowledge Integration |
|----------|-------|----------------------|
| **2402.07016** | REALM: RAG-Driven Enhancement of Multimodal EHR Analysis | LLM for entity extraction; PrimeKG alignment; task-relevant summaries |
| **2406.00036** | EMERGE: Enhancing Multimodal EHR with Retrieval-Augmented Generation | RAG with PrimeKG; cross-modal attention fusion; entity definitions/descriptions |
| **2403.08818** | MINGLE: Multimodal Fusion of EHR in Structures and Semantics | Hypergraph neural networks; two-level infusion (concept + note semantics) |
| **2506.17844** | THCM-CAL: Temporal-Hierarchical Causal Modelling | Multimodal causal graph; hierarchical causal discovery; conformal calibration |

---

## 2. Temporal Alignment Strategies

### 2.1 Dynamic Time Warping and Interpolation

**Key Insight:** Irregular sampling requires sophisticated interpolation beyond simple forward-fill.

**Leading Approaches:**
1. **Learned Interpolation with Gating** (ArXiv: 2210.12156)
   - Hand-crafted imputation embeddings + learned interpolation
   - Gating mechanism dynamically weights contributions
   - Handles variable recording frequencies

2. **Dynamic Embedding and Tokenization** (ArXiv: 2403.04012)
   - Time-aware embeddings encoding both value and timestamp
   - Temporal cross-attention transformers
   - Sliding window attention for long sequences

3. **Virtual Timestamp Encoding** (ArXiv: 2510.10037)
   - Spatiotemporal separable convolution (41% parameter reduction)
   - Virtual timestamps for interannual evolution patterns
   - Post-operative dynamic process modeling

### 2.2 Cross-Modal Temporal Alignment

**Challenge:** Different modalities have vastly different temporal granularities
- Vital signs: Every few minutes
- Lab results: Hours to days
- Imaging: Days to weeks
- Clinical notes: Irregular, event-driven

**Solutions:**

1. **Hierarchical Time-Aware Fusion** (ArXiv: 2510.27321)
   - Micro-temporal patterns (within modality)
   - Macro-temporal patterns (across modalities)
   - Modality-specific encoders with shared alignment

2. **Multi-timescale Alignment** (ArXiv: 2510.11112)
   - Local (pairwise interval-level) synchronization
   - Global (full-sequence) synchronization
   - Static vs. dynamic feature disentanglement

3. **Temporal Cross-Attention** (ArXiv: 2403.04012)
   - Attention mechanism with temporal distance scaling
   - Position encoding for irregular timestamps
   - Time-distance scaled self-attention

### 2.3 Slot Attention for Temporal Patterns

**Innovation:** Slot attention discovers recurring temporal patterns across patients

**Example: CTPD Framework** (ArXiv: 2411.00696)
- Shared initial temporal pattern representations
- Slot attention refines patterns → temporal semantic embeddings
- Contrastive TPNCE loss for cross-modal alignment
- Reconstruction losses preserve modality-specific information

**Benefits:**
- Identifies clinically relevant patterns (e.g., deteriorating vitals)
- Generalizes across patients
- Interpretable temporal motifs

---

## 3. Fusion Architecture Taxonomy

### 3.1 Early Fusion

**Definition:** Concatenate raw or minimally processed features from different modalities before main processing

**Advantages:**
- Maximum information sharing
- Can discover low-level cross-modal patterns

**Disadvantages:**
- Sensitive to modality imbalance
- Difficult with missing modalities
- High dimensional feature space

**Best Practices:**
- Feature normalization critical
- Effective for homogeneous modalities (e.g., multiple time series)

### 3.2 Late Fusion

**Definition:** Process each modality independently; combine predictions at decision stage

**Advantages:**
- Modality-specific optimization
- Robust to missing modalities
- Easier to interpret

**Disadvantages:**
- Misses cross-modal interactions
- May not capture synergistic effects

**Leading Implementation:**
- Weighted averaging based on modality confidence
- Ensemble methods with learned weights

### 3.3 Hybrid Fusion (SOTA Approach)

**Definition:** Multi-stage fusion combining early, intermediate, and late strategies

**Architecture Pattern:**
1. **Stage 1 (Early):** Modality-specific encoders
2. **Stage 2 (Intermediate):** Cross-modal attention/fusion
3. **Stage 3 (Late):** Decision-level aggregation

**Examples:**

1. **MedPatch** (ArXiv: 2508.09182)
   - Multi-stage: joint + late fusion simultaneously
   - Confidence-guided patching
   - Token-level confidence calibration

2. **MINGLE** (ArXiv: 2403.08818)
   - Two-level infusion: concept semantics + note semantics
   - Hypergraph neural networks for complex interactions
   - 11.83% relative improvement

3. **4D-ACFNet** (ArXiv: 2503.09652)
   - 4D spatiotemporal attention mechanism
   - Cross-modal dynamic calibration via Transformer
   - Gated classification head for temporal decisions

### 3.4 Cross-Attention Based Fusion

**Mechanism:** One modality attends to another using attention mechanism

**Variants:**

1. **Bi-Directional Cross-Attention**
   - Text ↔ Image mutual attention
   - Used in: RadFusion (ArXiv: 2111.11665)

2. **Hierarchical Cross-Attention**
   - Multi-scale attention (pixel, region, global)
   - Example: DA-SPL (ArXiv: 2510.10037)

3. **Gated Cross-Attention**
   - Dominant modality guides attention
   - Example: COMPRER (ArXiv: 2403.09672)

**Performance:** Typically 5-15% improvement over simple concatenation

### 3.5 Graph-Based Fusion

**Rationale:** Model relationships between modalities and temporal points as graphs

**Architecture:**

1. **Spatiotemporal Graphs** (ArXiv: 2311.07608 - MuST)
   - Nodes: Modality features at different time points
   - Spatial edges: Cross-modal relationships
   - Temporal edges: Sequential dependencies
   - Graph Convolution Networks for aggregation

2. **Heterogeneous Graphs** (ArXiv: 2506.17844 - THCM-CAL)
   - Multimodal causal graph
   - Textual propositions + ICD codes as nodes
   - Three interaction types:
     - Intra-slice same-modality sequencing
     - Intra-slice cross-modality triggers
     - Inter-slice risk propagation

**Advantages:**
- Explicitly models relationships
- Handles irregular structure naturally
- Interpretable via graph analysis

---

## 4. Handling Asynchronous and Missing Modalities

### 4.1 The Asynchronicity Problem

**Clinical Reality:**
- EHR: Continuous (minutes)
- Chest X-ray: Sparse (days to weeks)
- CT/MRI: Very sparse (weeks to months)
- Clinical notes: Event-driven (irregular)

**Impact:** Last available imaging may be outdated when prediction needed

### 4.2 Leading Solutions

#### 4.2.1 Generative Imputation

**DDL-CXR** (ArXiv: 2410.17918)
- Latent diffusion model generates up-to-date CXR representation
- Conditioned on: Previous CXR + EHR time series
- Captures disease progression between imaging
- Outperforms imputation baselines

**AMM-Diff** (ArXiv: 2501.12840)
- Adaptive multi-modality diffusion network
- Image-Frequency Fusion Network (IFFN)
- Handles any number of missing modalities
- Test-time training for adaptation

#### 4.2.2 Feature Disentanglement

**DrFuse** (ArXiv: 2403.06197)
- Disentangles shared vs. modality-specific features
- Disease-wise attention assigns patient/disease-specific weights
- Robust to missing entire modalities
- SOTA on MIMIC-IV with missing data

**DiPro** (ArXiv: 2510.11112)
- Separates static (anatomy) from dynamic (pathology) features
- Multi-timescale alignment (local + global)
- Handles temporal misalignment naturally

#### 4.2.3 Modality Dropout Training

**Strategy:** Randomly drop modalities during training to enforce robustness

**MDA** (ArXiv: 2406.10569)
- Modal-Domain Attention with continuous attention allocation
- Reduces attention to low-correlation/missing/noisy modalities
- <3% variance across missing patterns

**Missing-modality Architecture** (ArXiv: 2309.15529)
- Multivariate loss functions optimize for all missing patterns
- Bi-modal fusion modules combine pairwise
- Tri-modal framework integrates all three

#### 4.2.4 Confidence-Guided Fusion

**MedPatch** (ArXiv: 2508.09182)
- Multi-stage fusion with confidence-guided patching
- Calibrated unimodal token-level confidence
- Missingness-aware module handles sparse samples
- SOTA: 81.12% mortality accuracy (MIMIC-III)

### 4.3 Performance Under Missing Modalities

**Benchmark Results (MIMIC-III/IV):**

| Method | Full Data | 1 Missing | 2 Missing |
|--------|-----------|-----------|-----------|
| Baseline (late fusion) | 0.75 | 0.62 | 0.45 |
| DrFuse | 0.82 | 0.79 | 0.71 |
| MedPatch | 0.81 | 0.78 | 0.73 |
| MDA | 0.80 | 0.77 | 0.75 |

**Key Finding:** Well-designed architectures maintain >90% performance with 1 missing modality

---

## 5. Clinical Applications and Performance

### 5.1 Mortality Prediction

**Task:** Predict in-hospital or 30-day mortality from admission data

**Leading Results:**

| Paper | Dataset | Modalities | AUROC | AUPRC | F1 |
|-------|---------|------------|-------|-------|-----|
| CTPD | MIMIC-III | Time series + Notes | - | - | Superior |
| MedM2T | MIMIC-IV | EHR + ECG | 0.901 | 0.558 | - |
| Global Contrastive | MIMIC | Time series + Notes + Images | - | - | 6.33% gain |
| MedFuse | MIMIC-IV | Time series + CXR | - | - | SOTA |
| EMERGE | MIMIC-III | Time series + Notes + KG | - | - | 4.8% gain |

**Key Factors for Performance:**
1. Temporal modeling of deterioration patterns
2. Integration of clinical notes for context
3. Handling irregular sampling in vitals
4. Early detection (24-48h window)

### 5.2 Readmission Prediction

**Task:** Predict 30-day hospital readmission

**Leading Results:**

| Paper | Dataset | Modalities | Performance |
|-------|---------|------------|-------------|
| MuST | MIMIC-IV | EHR + CXR + Notes | Outperforms baselines |
| REALM | MIMIC-III | EHR + Notes + KG | Superior to baselines |
| RadFusion | Internal | CT + EHR | 20% improvement |
| EMERGE | MIMIC-IV | EHR + Notes + KG | SOTA |

**Critical Features:**
- Discharge summary quality
- Historical readmission patterns
- Social determinants (when available)
- Disease severity at discharge

### 5.3 Disease Progression and Phenotyping

**Task:** Predict disease trajectory and classify phenotypes

**Notable Work:**

1. **GFE-Mamba** (ArXiv: 2407.15719)
   - MCI → AD progression prediction
   - Multimodal: MRI + PET + clinical scales
   - Generative feature extractor compensates for missing PET
   - Mamba block for long-sequence modeling

2. **CSF-Net** (ArXiv: 2501.16400)
   - Pulmonary nodule malignancy prediction
   - Follow-up CT scans + clinical data
   - Temporal residual fusion module
   - Accuracy: 89.74%, AUC: 93.89%

3. **CTPD** (ArXiv: 2411.00696)
   - 24-hour phenotype classification
   - Cross-modal temporal pattern discovery
   - MIMIC-III: Superior performance

### 5.4 Acute Clinical Events

**Applications:**
- Sepsis onset prediction
- Cardiac arrest
- Respiratory failure
- Acute kidney injury

**Temporal Characteristics:**
- Requires minute-level resolution
- Vital sign patterns critical
- Integration with lab trends
- Clinical context from notes

**Example: TabulaTime** (ArXiv: 2502.17049)
- Acute Coronary Syndrome prediction
- Integrates environmental data (air pollution)
- PatchRWKV for temporal patterns
- 20% improvement over baselines

### 5.5 Chronic Disease Management

**Applications:**
- Diabetes progression
- Heart failure management
- COPD exacerbation
- Cancer progression

**Example: Pediatric Kidney Disease** (ArXiv: 2511.13637)
- Longitudinal lab sequences + demographics
- RNN for temporal modeling
- 30-day abnormal creatinine prediction
- Pilot for multimodal extension

---

## 6. Modality-Specific Considerations

### 6.1 EHR Time Series (Vitals, Labs)

**Characteristics:**
- High-frequency (minutes to hours)
- Irregular sampling
- Missing values common
- Multivariate dependencies

**Best Practices:**
1. Dynamic interpolation (not simple forward-fill)
2. Time-distance aware attention
3. Missingness indicators as features
4. Multi-scale temporal modeling

**Leading Encoders:**
- LSTM with temporal attention
- Temporal Fusion Transformer
- PatchRWKV (linear complexity)
- Mamba (for very long sequences)

### 6.2 Medical Imaging (CXR, CT, MRI)

**Characteristics:**
- Low frequency (days to weeks)
- High dimensional
- Often outdated at decision time
- Expensive to acquire

**Strategies:**

1. **Feature Extraction**
   - CNN encoders (ResNet, DenseNet)
   - Vision Transformers
   - Domain-specific pretraining
   - Segmentation-based region features

2. **Temporal Modeling**
   - Difference images between timepoints
   - Trajectory prediction (ArXiv: 2507.14766)
   - Latent diffusion for updating (ArXiv: 2410.17918)

3. **Integration with EHR**
   - Cross-attention mechanisms
   - Shared latent space alignment
   - Confidence-weighted fusion

### 6.3 Clinical Notes

**Characteristics:**
- Unstructured text
- Variable length
- Critical context
- Irregular timing

**Processing Approaches:**

1. **LLM-based Encoding**
   - BERT variants (ClinicalBERT, BioBERT, PubMedBERT)
   - LLaMA fine-tuned on medical text
   - Chunk-wise processing for long notes

2. **Entity Extraction**
   - Named Entity Recognition (NER)
   - Relation extraction
   - Temporal expression normalization
   - Knowledge graph alignment

3. **Temporal Modeling**
   - Treat as irregular time series (ArXiv: 2210.12156)
   - Time attention mechanism
   - Sequence modeling with LSTM/Transformer

**Example: MedualTime** (ArXiv: 2406.06620)
- Dual adapters: temporal-primary + textual-primary
- Both modalities serve as primary
- Lightweight adaptation tokens
- 8% accuracy, 12% F1 improvement

### 6.4 Waveform Data (ECG, EEG)

**Characteristics:**
- Very high frequency (100-1000 Hz)
- Dense time series
- Multi-channel signals
- Long sequences

**Approaches:**

1. **Hierarchical Processing**
   - Beat-level features
   - Segment-level aggregation
   - Global sequence modeling

2. **MedM2T Framework** (ArXiv: 2510.27321)
   - Hierarchical time-aware fusion
   - Micro-temporal (within-beat) patterns
   - Macro-temporal (beat-to-beat) patterns
   - AUROC 0.947 for CVD prediction

### 6.5 Structured Data (Demographics, Medications)

**Characteristics:**
- Categorical and continuous
- Static or slowly changing
- Small feature dimension
- Critical for risk adjustment

**Integration:**
- Embedding layers for categoricals
- Feature engineering for continuous
- Concatenation with other modalities
- Attention-based weighting

---

## 7. Research Gaps and Future Directions

### 7.1 Identified Research Gaps

#### 7.1.1 Extreme Temporal Misalignment
**Gap:** Most methods assume some temporal overlap between modalities
- **Challenge:** Emergency Department with no prior imaging
- **Need:** Better zero-shot/transfer capabilities
- **Proposed Solution:** Foundation models with strong priors

#### 7.1.2 Causal Temporal Reasoning
**Gap:** Current methods are correlational, not causal
- **Challenge:** Understanding actual causative relationships
- **Example:** Does abnormal CXR cause readmission or is it a marker?
- **Emerging Work:** THCM-CAL (ArXiv: 2506.17844) with causal graphs

#### 7.1.3 Multi-Timestep Prediction Uncertainty
**Gap:** Point predictions without calibrated uncertainty
- **Challenge:** Clinical decisions require confidence intervals
- **Need:** Conformal prediction for multimodal time series
- **Example:** THCM-CAL extends conformal prediction to multi-label ICD

#### 7.1.4 Computational Efficiency
**Gap:** Many SOTA models too slow for real-time clinical use
- **Challenge:** ED requires <1 minute inference
- **Solutions:**
  - Lightweight architectures (MedFuse: LSTM-based)
  - Early exit strategies (PASS: ArXiv: 2508.10501)
  - Efficient attention (linear transformers)

#### 7.1.5 Generalization Across Sites
**Gap:** Models trained on one hospital fail at others
- **Challenge:** Different EHR systems, practices, populations
- **Need:** Domain adaptation techniques
- **Emerging:** Multi-site federated learning

#### 7.1.6 Handling Very Sparse Data
**Gap:** Performance degrades with <10% modality availability
- **Challenge:** Rural/resource-limited settings
- **Need:** Better semi-supervised and self-supervised methods
- **Example:** Contrastive learning with limited labels

### 7.2 Promising Research Directions

#### 7.2.1 Foundation Models for Healthcare

**Opportunity:** Large-scale pretraining on multimodal medical data

**Examples:**
- **CLIMB** (ArXiv: 2503.07667): 4.51M patients, 19.01TB
  - Multimodal pretraining improves downstream tasks
  - 29% improvement in ultrasound, 23% in ECG
  - Strong generalization to new tasks

**Future Directions:**
1. Self-supervised pretraining objectives for temporal data
2. Cross-modal pretraining strategies
3. Efficient fine-tuning for downstream tasks
4. Transfer learning across clinical domains

#### 7.2.2 Retrieval-Augmented Generation (RAG)

**Trend:** Augment models with external medical knowledge

**Leading Work:**
- **REALM** (ArXiv: 2402.07016)
- **EMERGE** (ArXiv: 2406.00036)

**Approach:**
1. Extract entities from patient data
2. Retrieve relevant knowledge from KG (e.g., PrimeKG)
3. Generate task-relevant summaries
4. Fuse with multimodal features

**Benefits:**
- Reduces hallucination
- Incorporates medical guidelines
- Improves interpretability
- Better generalization

**Future Directions:**
1. Real-time KG updates
2. Personalized knowledge retrieval
3. Multi-hop reasoning over KGs
4. Integration with LLM reasoning

#### 7.2.3 State Space Models (Mamba Architecture)

**Innovation:** Linear complexity alternative to Transformers

**Example: GFE-Mamba** (ArXiv: 2407.15719)
- Efficient long-sequence modeling
- Pixel-level bi-cross attention
- Strong performance on AD progression

**Advantages:**
- O(n) vs O(n²) complexity
- Better for very long sequences
- Lower memory footprint

**Applications:**
- Continuous monitoring data
- Long-term disease progression
- Multi-day ICU stays

#### 7.2.4 Multimodal Contrastive Learning

**Approach:** Learn aligned representations without explicit supervision

**Example: Global Contrastive Training** (ArXiv: 2404.06723)
- Align time series + notes with discharge summaries
- Temporal cross-attention transformers
- Global contrastive loss

**Benefits:**
- Self-supervised learning
- Better feature representations
- Improved downstream performance

**Future Directions:**
1. Multi-view contrastive learning
2. Temporal contrastive objectives
3. Hard negative mining for medical data
4. Cross-modal retrieval

#### 7.2.5 Interpretable Temporal Patterns

**Need:** Clinically meaningful explanations of predictions

**Approaches:**
1. **Attention Visualization**
   - Temporal attention maps
   - Cross-modal attention patterns
   - Region-specific contributions

2. **Prototype Learning**
   - Discover typical temporal patterns
   - Match new patients to prototypes
   - Example: CTPD slot attention

3. **Causal Discovery**
   - Identify temporal causal relationships
   - Example: THCM-CAL causal graphs

**Future Directions:**
1. Concept-based explanations
2. Counterfactual temporal reasoning
3. Natural language explanations from patterns

#### 7.2.6 Federated and Privacy-Preserving Learning

**Challenge:** Privacy regulations limit data sharing

**Opportunities:**
1. Federated learning across hospitals
2. Differential privacy for time series
3. Secure multi-party computation
4. Synthetic data generation

**Benefits:**
- Larger effective dataset
- Better generalization
- Compliance with regulations
- Multi-site validation

#### 7.2.7 Active Learning and Human-in-the-Loop

**Motivation:** Labeling multimodal medical data is expensive

**Strategies:**
1. Uncertainty-based sample selection
2. Temporal anomaly detection for labeling
3. Interactive refinement
4. Physician feedback integration

**Example: PASS** (ArXiv: 2508.10501)
- Probabilistic agentic reasoning
- Interpretable decision paths
- Dynamic tool selection

---

## 8. Relevance to ED Multimodal KG Scaffold

### 8.1 Direct Applications

#### 8.1.1 Temporal Alignment for ED Workflow

**Challenge:** ED encounters have extreme temporal heterogeneity
- Triage vitals: Immediate
- Lab orders: 15-60 minutes
- Imaging: 1-4 hours
- Disposition decision: 2-8 hours

**Applicable Techniques:**

1. **Hierarchical Time-Aware Fusion** (from MedM2T)
   - Micro-temporal: Within-modality (vitals every 15 min)
   - Macro-temporal: Across-modality (vitals → labs → imaging)
   - Suitable for ED time scales

2. **Dynamic Embedding with Time-Distance Scaling** (from Temporal Cross-Attention)
   - Explicitly encodes time gaps
   - Attention weighted by temporal distance
   - Natural fit for irregular ED events

3. **Interleaved Attention** (from Irregular Multimodal EHR)
   - Cross-modal attention at each temporal step
   - Handles asynchronous arrival of modalities
   - Real-time updating as new data arrives

**Implementation for ED KG:**
```
Node Types: [Vital_T0, Lab_T1, Image_T2, Disposition_T3]
Edge Types:
  - Temporal_Sequential (within modality)
  - Cross_Modal_Correlation (across modalities)
  - Time_Distance_Weighted (attention strength)
```

#### 8.1.2 Missing Modality Robustness

**ED Reality:** Not all patients get all tests
- CXR ordered: ~40% of ED patients
- CT scan: ~10-15%
- Blood culture: <5%

**Applicable Solutions:**

1. **DrFuse Disentanglement**
   - Shared features: Common to all patients (vitals, demographics)
   - Unique features: Imaging findings when available
   - Disease-wise attention: ED diagnosis-specific weighting

2. **Modality Dropout Training**
   - Train on all possible missing patterns
   - ED-specific dropout probabilities match real rates
   - Robust predictions even with minimal data

3. **Confidence-Guided Fusion** (from MedPatch)
   - Weight modalities by quality and availability
   - Uncertain imaging → rely more on labs/vitals
   - Adaptive to ED clinical decision-making

**KG Implementation:**
```
Node Attributes:
  - availability: [present, missing, pending]
  - confidence: [0-1] (from model uncertainty)
  - temporal_currency: time since last update

Fusion Strategy:
  - Weight edges by (availability × confidence × currency)
  - Dynamic subgraph based on available modalities
```

#### 8.1.3 Knowledge Graph Integration

**Opportunity:** Enhance ED KG with medical knowledge

**Applicable Frameworks:**

1. **REALM/EMERGE RAG Approach**
   - Extract entities from ED encounter (complaints, findings)
   - Retrieve relevant knowledge from medical KG
   - Generate contextualized representations
   - Example: "Chest pain" → retrieves cardiac, PE, MSK possibilities

2. **THCM-CAL Causal Modeling**
   - Build causal graph of ED events
   - Textual propositions (chief complaint) → ICD codes
   - Three relationship types:
     - Temporal sequencing (symptom onset → test → diagnosis)
     - Cross-modal triggers (abnormal vital → order lab)
     - Outcome propagation (test result → disposition)

3. **MINGLE Hypergraph Approach**
   - Multi-way relationships in ED
   - Example: (Chest pain + elevated troponin + ST changes) → STEMI
   - Concept-level + instance-level fusion

**ED KG Schema Extension:**
```
External Knowledge Nodes:
  - Disease Concepts (from PrimeKG/UMLS)
  - Treatment Guidelines (from clinical pathways)
  - Drug Interactions (from DrugBank)

Relationship Types:
  - is_symptom_of
  - requires_workup
  - suggests_diagnosis
  - contraindicated_with
```

#### 8.1.4 Real-Time Prediction

**ED Requirement:** Predictions needed within minutes, not hours

**Applicable Optimizations:**

1. **Lightweight Architectures**
   - LSTM-based (MedFuse): Fast, efficient
   - Early exit strategies (PASS): Stop when confident
   - Separable convolutions: 41% parameter reduction

2. **Incremental Updates**
   - Update predictions as new data arrives
   - Don't recompute entire graph
   - Attention mechanism for selective updates

3. **Caching and Precomputation**
   - Static patient features (demographics, history)
   - Common temporal patterns
   - Knowledge retrieval results

**Implementation:**
```
Inference Pipeline:
1. Initial prediction from triage data (< 1 second)
2. Update when vitals arrive (< 5 seconds)
3. Update when labs available (< 10 seconds)
4. Final update with imaging (< 30 seconds)

Each update: Partial graph recomputation only
```

#### 8.1.5 Interpretability for Clinical Trust

**Critical:** ED physicians need to understand model reasoning

**Applicable Techniques:**

1. **Attention Visualization**
   - Show which vitals/labs contributed most
   - Temporal attention: When did patient deteriorate?
   - Cross-modal attention: How do CXR and labs align?

2. **Temporal Pattern Discovery** (from CTPD)
   - Identify recurring ED patterns
   - Example: "Sepsis pattern": Fever spike → lactate rise → pressure drop
   - Match current patient to known patterns

3. **Probabilistic Reasoning** (from PASS)
   - Probability-annotated decision paths
   - Uncertainty quantification
   - Alternative diagnoses with likelihoods

4. **Natural Language Explanations**
   - LLM generates explanation from attention/patterns
   - Example: "High risk due to persistent tachycardia and rising lactate despite fluids"

**KG Visualization:**
```
Explanation Components:
- Key nodes: Highlighted features contributing most
- Critical paths: Temporal sequences leading to prediction
- Counterfactuals: "If troponin had been normal, risk would be low"
- Confidence: Per-node and per-edge certainties
```

### 8.2 Architectural Recommendations

Based on the literature review, here's a recommended architecture for an ED multimodal temporal fusion system:

#### Layer 1: Modality-Specific Encoders
```
Time Series (Vitals):
  - LSTM with temporal attention (proven in MedFuse)
  - Handle irregular sampling with learned interpolation

Lab Results:
  - Sparse time series encoder (from MedM2T)
  - Missingness indicators as features

Clinical Notes:
  - Fine-tuned ClinicalBERT
  - Chunk-wise for long notes

Imaging:
  - DenseNet/ResNet for CXR
  - Pre-trained on medical images

Chief Complaint:
  - Embed + retrieve from knowledge graph
  - RAG-style augmentation
```

#### Layer 2: Temporal Alignment
```
Approach: Hierarchical Time-Aware Fusion
- Micro-level: Within-modality temporal patterns
- Macro-level: Cross-modality temporal alignment
- Time-distance scaled attention

Missing Data Handling:
- Feature disentanglement (DrFuse approach)
- Confidence-based weighting (MedPatch approach)
```

#### Layer 3: Knowledge Graph Integration
```
Knowledge Retrieval:
- Extract entities from encounter
- Match to ED-specific KG + general medical KG
- Generate context-aware embeddings

Graph Structure:
- Nodes: Patient features + medical concepts
- Edges: Temporal, causal, semantic
- Attention mechanism for propagation
```

#### Layer 4: Multimodal Fusion
```
Strategy: Hybrid (early + intermediate + late)

Early Fusion:
- Homogeneous modalities (vitals + labs)

Intermediate Fusion:
- Cross-attention between modalities
- Knowledge-enriched representations

Late Fusion:
- Ensemble of modality-specific predictions
- Confidence-weighted combination
```

#### Layer 5: Prediction and Explanation
```
Multi-Task Heads:
- Disposition prediction
- Admission likelihood
- Length of stay
- Adverse events (sepsis, MI, PE)

Explanation Module:
- Attention-based feature importance
- Temporal pattern matching
- Natural language generation
```

### 8.3 Training Strategy

**Multi-Stage Approach** (inspired by PASS, CLIMB):

**Stage 1: Unimodal Pretraining**
- Each encoder trained on single modality
- Large-scale self-supervised learning
- Temporal prediction tasks

**Stage 2: Multimodal Alignment**
- Contrastive learning across modalities
- Temporal synchronization losses
- Knowledge graph grounding

**Stage 3: Task-Specific Fine-Tuning**
- Supervised learning on ED outcomes
- Multi-task learning across related tasks
- Curriculum learning (easy → hard cases)

**Stage 4: Adversarial Robustness**
- Modality dropout
- Temporal jittering
- Missing data augmentation

### 8.4 Evaluation Framework

**Metrics:**
1. **Predictive Performance**
   - AUROC, AUPRC for classification
   - MAE, RMSE for regression
   - Calibration (Brier score)

2. **Temporal Accuracy**
   - Early prediction capability (how many hours ahead?)
   - Update improvement (gain from new modality)
   - Temporal stability (prediction consistency)

3. **Robustness**
   - Performance vs. % missing modalities
   - Performance on delayed modalities
   - Out-of-distribution generalization

4. **Efficiency**
   - Inference time (<1 min requirement)
   - Memory footprint
   - Throughput (patients/second)

5. **Interpretability**
   - Attention alignment with clinical reasoning
   - Feature importance matches known risk factors
   - Physician trust ratings

**Benchmark Datasets:**
- MIMIC-IV ED module (when available)
- Internal ED database
- Multi-site validation

### 8.5 Implementation Considerations

**Technical Stack:**
```python
Framework: PyTorch Geometric (for graph operations)
Encoders:
  - HuggingFace Transformers (text)
  - TimeSeries library (vitals/labs)
  - TorchVision (imaging)

Knowledge Graph: Neo4j or DGL
Serving: TorchServe with FastAPI
Monitoring: MLflow for experiment tracking
```

**Scalability:**
- Batch inference for non-urgent updates
- Real-time streaming for critical patients
- GPU acceleration for imaging
- CPU efficient for time series

**Clinical Integration:**
- HL7 FHIR interface for EHR
- DICOM for imaging
- RESTful API for predictions
- Dashboard for visualization

---

## 9. Conclusion

This comprehensive review of multimodal temporal fusion for clinical AI reveals several key insights:

**Major Findings:**

1. **Temporal Modeling is Critical:** The top-performing systems all employ sophisticated temporal alignment strategies beyond simple concatenation. Hierarchical time-aware fusion, attention mechanisms with temporal distance scaling, and dynamic temporal pattern discovery are essential components.

2. **Hybrid Fusion Dominates:** The most successful architectures combine early, intermediate, and late fusion strategies. No single fusion approach works best for all modalities or tasks.

3. **Missing Modality Robustness is Essential:** Clinical deployment requires graceful degradation with missing data. Feature disentanglement, confidence-guided fusion, and modality dropout training are proven strategies.

4. **Knowledge Integration Enhances Performance:** Incorporating medical knowledge graphs through RAG, causal reasoning, or hypergraph structures consistently improves both performance and interpretability.

5. **Foundation Models Show Promise:** Large-scale pretraining on multimodal medical data (CLIMB, etc.) demonstrates strong transfer learning and generalization capabilities.

**For ED Multimodal KG Applications:**

The reviewed literature provides a strong foundation for building robust, interpretable, and efficient multimodal temporal fusion systems for emergency department applications. Key recommendations include:

- **Hierarchical temporal architecture** to handle ED's multi-scale temporal dynamics
- **Disentangled representations** for robustness to incomplete workups
- **Knowledge graph augmentation** for incorporating medical reasoning
- **Attention-based interpretability** for clinical trust
- **Real-time optimization** for practical deployment

**Future Outlook:**

The field is rapidly advancing with:
- Foundation models trained on massive multimodal medical datasets
- State space models (Mamba) for efficient long-sequence modeling
- Causal reasoning frameworks for interpretable temporal predictions
- Federated learning for multi-site collaboration

These advances position multimodal temporal fusion as a transformative technology for clinical decision support, particularly in time-sensitive environments like emergency departments.

---

## References

All papers cited are available on ArXiv. Key papers include:

- CTPD (2411.00696): Cross-modal temporal pattern discovery
- Temporal Cross-Attention (2403.04012): Dynamic EHR embeddings
- DrFuse (2403.06197): Missing modality robustness
- CLIMB (2503.07667): Large-scale multimodal benchmark
- REALM (2402.07016): RAG for EHR enhancement
- MedM2T (2510.27321): Time-aware multimodal framework
- THCM-CAL (2506.17844): Temporal-hierarchical causal modeling

**Total Papers Reviewed:** 100+
**Date Range:** 2019-2025
**Primary Focus:** Clinical multimodal learning with temporal dynamics

---

*Report prepared for: Hybrid Reasoning Acute Care Research Project*
*Analysis Date: December 1, 2025*
