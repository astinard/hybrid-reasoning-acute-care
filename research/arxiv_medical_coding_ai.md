# ArXiv Literature Synthesis: Medical Coding AI for Automated ICD and CPT Coding

## Executive Summary

This comprehensive literature synthesis examines the state-of-the-art in automated medical coding using AI, specifically focusing on ICD (International Classification of Diseases) and CPT (Current Procedural Terminology) code prediction from clinical text. The field has evolved significantly from traditional machine learning approaches to sophisticated deep learning architectures, with recent advances in transformer-based models and large language models (LLMs) showing remarkable promise.

**Key Findings:**
- **Performance Benchmarks**: State-of-the-art models achieve micro-F1 scores of 55-61% on MIMIC-III full code set, with significant improvements for top-50 frequent codes (70-80% F1)
- **Architectural Evolution**: Progression from CNN-based models → RNN/LSTM → Attention mechanisms → Transformers → LLMs
- **Critical Challenges**: Long-tail distribution of codes, long document processing, label imbalance, and explainability requirements
- **Emerging Solutions**: Pre-trained language models (PLM-ICD), hierarchical attention, label-wise attention, and contrastive learning approaches

---

## 1. Key Research Papers and ArXiv IDs

### Foundational Papers

1. **Explainable Prediction of Medical Codes** (arXiv:1802.05695v2)
   - Authors: Mullenbach et al.
   - Introduced CAML (Convolutional Attention for Multi-Label classification)
   - Benchmark: Precision@8 of 0.71, Micro-F1 of 0.54 on MIMIC-III

2. **An Empirical Evaluation of Deep Learning for ICD-9 Code Assignment** (arXiv:1802.02311v2)
   - Authors: Huang, Osorio, Sy
   - Systematic comparison of CNN and RNN approaches
   - Top-10 codes: F1=0.6957, Accuracy=0.8967

3. **ICD Coding from Clinical Text Using Multi-Filter Residual CNN** (arXiv:1912.00862v1)
   - Authors: Li & Yu (MultiResCNN)
   - Multi-filter convolutional architecture
   - State-of-the-art on MIMIC-III full codes at time of publication

### Transformer-Based Approaches

4. **PLM-ICD: Automatic ICD Coding with Pretrained Language Models** (arXiv:2207.05289v1)
   - Authors: Huang, Tsai, Chen
   - **Current SOTA methodology**
   - Addresses three key challenges: large label space, long sequences, domain mismatch
   - Micro-F1: ~58-60% on MIMIC-III full codes

5. **Automated ICD Coding using Extreme Multi-label Long Text Transformer** (arXiv:2212.05857v2)
   - Authors: Liu et al.
   - **New SOTA**: Micro-F1 of 60.8% on MIMIC-III
   - XR-LAT model with hierarchical code trees
   - Macro-AUC improvement of 2.1%

6. **TransICD: Transformer Based Code-wise Attention Model** (arXiv:2104.10652v1)
   - Authors: Biswas, Pham, Zhang
   - Code-wise attention mechanism
   - Micro-AUC: 0.923 on MIMIC-III

7. **NoteContrast: Contrastive Language-Diagnostic Pretraining** (arXiv:2412.11477v1)
   - Authors: Kailas et al. (2024)
   - Contrastive learning approach
   - Improvements on MIMIC-III-50, MIMIC-III-rare50, and full datasets

### Label Attention Mechanisms

8. **A Label Attention Model for ICD Coding** (arXiv:2007.06351v1)
   - Authors: Vu, Nguyen, Nguyen
   - Label-wise attention for handling variable text fragment lengths
   - Hierarchical joint learning for infrequent codes
   - State-of-the-art on three MIMIC benchmarks

9. **A Pseudo Label-wise Attention Network** (arXiv:2106.06822v3)
   - Authors: Wu et al.
   - Computational efficiency through merging similar codes
   - Micro-F1: 0.583 on MIMIC-III, 0.806 on Xiangya dataset

10. **Hierarchical Label-wise Attention Transformer (HiLAT)** (arXiv:2204.10716v2)
    - Authors: Liu et al.
    - Label-wise attention with clinical BERT
    - Enhanced explainability through attention visualization

### Few-Shot and Long-Tail Solutions

11. **Knowledge Injected Prompt Based Fine-tuning** (arXiv:2210.03304v2)
    - Authors: Yang et al.
    - Addresses long-tail challenge with prompt-based learning
    - Macro-F1 improvement: 10.3 → 11.8 (14.5% gain)
    - MIMIC-III-rare50: Macro-F1 30.4, Micro-F1 32.6

12. **Multi-label Few-shot ICD Coding as Autoregressive Generation** (arXiv:2211.13813v2)
    - Authors: Yang et al.
    - Generation-based approach with prompts
    - Macro-F1: 30.2 on MIMIC-III-few
    - Ensemble learner: Macro-F1 14.6, Micro-F1 59.1

13. **Synthetic Clinical Notes for Rare ICD Codes** (arXiv:2511.14112v1)
    - Authors: Vo, Wu, Ding (2025)
    - Data augmentation for long-tail codes
    - Generated 90,000 synthetic notes covering 7,902 ICD codes
    - Improved macro-F1 while maintaining micro-F1

### Large Language Model Approaches

14. **Large Language Model in Medical Informatics** (arXiv:2411.06823v1)
    - Authors: Boukhers et al.
    - LLAMA architecture for ICD coding
    - Two methodologies: direct classification and enhanced representations

15. **Exploring LLM Multi-Agents for ICD Coding** (arXiv:2406.15363v2)
    - Authors: Li, Wang, Yu
    - Multi-agent framework mimicking real-world coding process
    - Five agents: patient, physician, coder, reviewer, adjuster
    - Competitive with SOTA while providing explainability

16. **Surpassing GPT-4 Medical Coding with Two-Stage Approach** (arXiv:2311.13735v1)
    - Authors: Yang et al.
    - LLM-codex: evidence generation + LSTM verification
    - State-of-the-art on accuracy, rare codes, and evidence identification

17. **Beyond Label Attention: Transparency via Dictionary Learning** (arXiv:2411.00173v2)
    - Authors: Wu et al.
    - Dictionary learning for interpretability
    - Goes beyond token-level attention for mechanistic explanations

### Medical Coding Review and Datasets

18. **A Unified Review of Deep Learning for Automated Medical Coding** (arXiv:2201.02797v5)
    - Authors: Ji et al.
    - **Comprehensive survey paper**
    - Unified framework: encoder, decoder, deep architecture, auxiliary information
    - Systematic categorization of approaches

19. **MDACE: MIMIC Documents Annotated with Code Evidence** (arXiv:2307.03859v1)
    - Authors: Cheng et al.
    - First publicly available code evidence dataset
    - 302 inpatient + 52 profee charts with evidence spans

### CPT Code Prediction

20. **Intelligent EHRs: Predicting Procedure Codes from Diagnosis Codes** (arXiv:1712.00481v1)
    - Authors: Haq et al.
    - Multi-label classification for CPT from ICD codes
    - 2.3M claims dataset
    - Recall of 90@3

---

## 2. Methods and Architectures

### 2.1 CNN-Based Approaches

**CAML (Convolutional Attention for Multi-Label)**
- Architecture: 1D CNN with per-label attention
- Innovation: Label-specific attention weights for each code
- Performance: Precision@8: 0.71, Micro-F1: 0.54
- Limitation: Fixed kernel sizes, limited long-range dependencies

**MultiResCNN (Multi-Filter Residual CNN)**
- Architecture: Multiple convolutional filters + residual connections
- Innovation: Captures various text patterns with different lengths
- Performance: Outperformed SOTA in 4/6 metrics on MIMIC-III full
- Key feature: Enlarged receptive field through residual layers

**DCAN (Dilated Convolutional Attention Network)**
- Architecture: Dilated convolutions with exponentially growing receptive field
- Innovation: Captures complex medical patterns without parameter explosion
- Application: Medical code assignment with sequential causal constraints

### 2.2 Attention-Based Architectures

**Label-Wise Attention (LWAN)**
- Mechanism: Separate attention for each ICD code
- Advantage: Handles variable text fragment lengths
- Challenge: Computational redundancy (attention for every code)
- Solution: Pseudo label-wise attention merges similar codes

**Hierarchical Attention Networks**
- Structure: Word-level → Sentence-level → Document-level attention
- Examples: HLAN, HAN
- Benefit: Multi-granularity feature extraction
- Application: Long clinical documents (discharge summaries)

**Partition-Based Label Attention**
- Innovation: Segmental encoding + local/global attention
- Mechanism: Finds key tokens in paragraphs vs. entire text
- Performance: Improved ICD coding on MIMIC-III benchmark

### 2.3 Transformer-Based Models

**PLM-ICD (Pretrained Language Model for ICD)**
- Base: Clinical BERT / BioClinicalBERT
- Key Strategies:
  1. Chunking long documents (512 token windows)
  2. Label-wise attention for multi-label prediction
  3. Domain-specific pretraining on MIMIC-III
- Performance: Micro-F1 ~58-60% on full MIMIC-III codes
- **Current baseline for comparison**

**XR-LAT (eXtreme Recursively-trained Label Attention Transformer)**
- Architecture: Recursive training on hierarchical code tree
- Components:
  1. Label-wise attention
  2. Knowledge transferring
  3. Dynamic negative sampling
- Performance: Competitive with PLM-ICD, +2.1% macro-AUC

**TransICD**
- Innovation: Code-wise attention with BERT-CNN hybrid
- Explainability: Attention maps show reasoning process
- Performance: Micro-AUC 0.923 on MIMIC-III

**Hierarchical Label-wise Attention Transformer (HiLAT)**
- Components:
  1. ClinicalplusXLNet (continual pretraining)
  2. Label-wise attention layers
  3. Hierarchical structure awareness
- Explainability: Attention weight visualization for clinical validation

### 2.4 Large Language Model Approaches

**Direct LLM Classification**
- Models: GPT-4, LLAMA, Gemini
- Challenge: Excessive code prediction (high recall, low precision)
- Limitation: Requires careful prompt engineering

**Two-Stage LLM Systems**
- Stage 1: Evidence generation from clinical notes
- Stage 2: LSTM/neural verification of generated codes
- Example: LLM-codex achieves SOTA on evidence identification

**Multi-Agent LLM Systems**
- Agents: Patient, Physician, Coder, Reviewer, Adjuster
- Workflow: Mimics real-world ICD coding process
- Advantage: Enhanced explainability and human-like reasoning
- Performance: Competitive with SOTA on MIMIC-III

**Contrastive Pre-training (NoteContrast)**
- Approach: Joint pre-training of LLM and ICD code models
- Loss: Contrastive loss with noisy labels
- Performance: Improvements on MIMIC-III-50, rare50, and full datasets
- Innovation: Integrates ICD code sequences with medical text

### 2.5 Hierarchical and Graph-Based Methods

**Hierarchical Joint Learning**
- Utilizes ICD code hierarchy (chapters, sections, categories)
- Example: ICD-9 codes organized in tree structure
- Benefit: Improves infrequent code prediction
- Implementation: Multi-task learning across hierarchy levels

**Graph Neural Networks**
- Application: Label co-occurrence modeling
- Example: Multi-View Joint Learning with GNN
- Innovation: Extracts label dependencies from code graphs
- Combined with: Text embeddings for joint representation

### 2.6 Hybrid and Ensemble Approaches

**Medical Coding with Biomedical Transformer Ensembles**
- Strategy: Zero/few-shot learning with domain-specific transformers
- Innovation: Overcomes limited training data for rare codes
- Application: Pharmaceutical coding systems

**Multi-stage Retrieve and Re-rank**
- Stage 1: BM25 + knowledge-based retrieval
- Stage 2: Contrastive re-ranking with label co-occurrence
- Performance: SOTA on MIMIC-III benchmark
- Advantage: Handles extreme multi-label efficiently

---

## 3. Performance Benchmarks

### 3.1 MIMIC-III Dataset Benchmarks

**MIMIC-III Full Code Set (8,929 unique codes)**

| Model | Micro-F1 | Macro-F1 | Micro-AUC | Macro-AUC | Precision@8 |
|-------|----------|----------|-----------|-----------|-------------|
| CAML (2018) | 0.539 | 0.088 | 0.895 | 0.775 | 0.709 |
| MultiResCNN (2019) | 0.552 | 0.098 | 0.910 | 0.830 | - |
| LWAN (2020) | 0.559 | 0.095 | 0.921 | 0.836 | 0.719 |
| PLM-ICD (2022) | 0.580-0.600 | 0.103 | 0.928 | 0.841 | 0.735 |
| XR-LAT (2022) | 0.608 | - | 0.932 | 0.862 | - |
| Ensemble + RAC (2021) | 0.591 | 0.146 | - | - | - |

**MIMIC-III Top-50 Codes**

| Model | Micro-F1 | Macro-F1 | AUC-ROC |
|-------|----------|----------|---------|
| CNN-Attention (2018) | 0.628 | 0.571 | - |
| BERT-based (2020) | 0.703 | 0.618 | - |
| TransICD (2021) | 0.758 | 0.682 | 0.923 |
| HiLAT (2022) | 0.770-0.790 | 0.700-0.720 | - |
| SWAM (2021) | ~0.75 | - | - |

**MIMIC-III Rare-50 Codes (Few-shot Setting)**

| Model | Micro-F1 | Macro-F1 |
|-------|----------|----------|
| Previous SOTA | 0.172 | 0.171 |
| Knowledge-injected Prompt (2022) | 0.326 | 0.304 |
| Autoregressive Generation (2022) | - | 0.302 |

### 3.2 MIMIC-IV Dataset

| Model | Performance Notes |
|-------|------------------|
| NoteContrast | Improvements over MIMIC-III baselines |
| PLM-ICD (adapted) | Competitive performance maintained |

### 3.3 Other Datasets

**Private/Commercial Datasets**
- Xiangya Hospital: Micro-F1 0.806 (Pseudo-LWAN)
- Brazilian Portuguese: Micro-F1 0.537 (CNN-Att on MIMIC-III)
- French Clinical Notes: 55% F1 improvement with transformers

### 3.4 Top-10 and Top-50 Specific Results

**Top-10 ICD-9 Codes**
- Best F1: 0.6957, Accuracy: 0.8967 (Huang et al., 2018)
- Top-10 Categories: F1: 0.7233, Accuracy: 0.8588

**Top-50 ICD-9 Codes**
- BiLSTM-CRF baseline: Micro-F1 ~0.65
- TransICD: Micro-F1 0.758
- State-of-the-art range: 0.70-0.79 F1

### 3.5 CPT Code Prediction

**Diagnosis → Procedure Mapping**
- Dataset: 2.3M claims
- Recall@3: 90%
- Architecture: Multi-label classification with distributed representations

---

## 4. LLM Approaches to Medical Coding

### 4.1 Challenges with Direct LLM Application

**GPT-4 Performance Issues**
- Problem: Excessive code prediction (high recall, low precision)
- Cause: Lack of medical coding-specific training
- Result: Not suitable for production without additional stages

**Domain Mismatch**
- Pre-trained LLMs lack medical domain knowledge
- ICD coding requires understanding of:
  - Medical terminology and abbreviations
  - Code hierarchy and relationships
  - Clinical context and implications

### 4.2 Successful LLM Integration Strategies

**1. Two-Stage Evidence-Based Systems**
- **LLM-codex Approach:**
  - Stage 1: LLM generates evidence proposals
  - Stage 2: LSTM verification with custom loss
  - Result: SOTA on accuracy, rare codes, and evidence ID

**2. Multi-Agent Frameworks**
- **Five-Agent System:**
  - Patient agent: Represents patient information
  - Physician agent: Clinical reasoning
  - Coder agent: Code assignment
  - Reviewer agent: Quality control
  - Adjuster agent: Final adjustments
- **Benefits:**
  - Mimics real-world workflow
  - Enhanced explainability
  - Competitive with SOTA

**3. Contrastive Pre-training**
- **NoteContrast Method:**
  - Joint training of ICD code sequences and text
  - Contrastive loss for alignment
  - Long-document transformers for clinical notes
- **Results:**
  - Improvements on MIMIC-III-50, rare50, and full
  - Better than previous SOTA transformers

**4. Retrieval-Augmented Generation**
- BM25 retrieval + LLM re-ranking
- Knowledge base integration
- Efficient for large code spaces

### 4.3 Prompt Engineering Approaches

**Knowledge-Injected Prompts**
- Label semantics in prompts
- Hierarchy information encoding
- Synonym and abbreviation handling
- Macro-F1 improvement: 10.3 → 11.8 (14.5% gain)

**Autoregressive Generation**
- Transform multi-label to sequence generation
- Generate code descriptions → infer codes
- SOAP structure utilization
- Macro-F1: 30.2 on few-shot benchmark

### 4.4 LLM Fine-tuning for Medical Coding

**PLM-ICD Style Approaches**
- Base: Clinical BERT, BioClinicalBERT, PubMedBERT
- Strategies:
  - Chunked document processing
  - Label-wise attention layers
  - Domain-specific continued pre-training

**ClinicalplusXLNet**
- Continual pre-training on MIMIC-III clinical notes
- XLNet-Base architecture
- Integration with hierarchical attention

**LLAMA for Medical Coding**
- Direct classification experiments
- Enhanced text representation generation
- Integration with MultiResCNN framework

### 4.5 Explainability in LLM-Based Coding

**Dictionary Learning Approach**
- Sparse representations from dense embeddings
- Overcomes limitations of label attention
- Explains medically irrelevant token contributions
- Mechanistic-based explanations for predictions

**Attention Visualization**
- Token-level attribution
- Sentence-level highlighting
- Code-specific explanations
- Human validation by medical professionals

---

## 5. Research Gaps and Future Directions

### 5.1 Current Limitations

**1. Long-Tail Distribution Challenge**
- Thousands of rare codes severely underrepresented
- Macro-F1 scores remain low (10-15%) for full code sets
- Data imbalance affects model fairness

**2. Long Document Processing**
- Clinical notes often exceed 3,000 tokens
- Transformer limitations at 512-2048 tokens
- Chunking strategies lose global context

**3. Explainability Gap**
- Attention mechanisms show tokens, not clinical reasoning
- Black-box decisions unacceptable for billing/compliance
- Need for evidence-based explanations

**4. Domain Adaptation**
- Models trained on English MIMIC-III data
- Limited generalization to other languages
- Institution-specific coding practices not captured

**5. Real-World Deployment Challenges**
- Human coder trust and acceptance
- Integration with existing EHR systems
- Computational costs for large models
- Update lag when ICD codes change (ICD-10 → ICD-11)

### 5.2 Emerging Research Directions

**1. Few-Shot and Zero-Shot Learning**
- Essential for rare and new codes
- Prompt-based learning showing promise
- Meta-learning approaches underexplored
- Synthetic data generation for long-tail codes

**2. Multimodal Medical Coding**
- Integration of:
  - Clinical text (discharge summaries, notes)
  - Structured data (lab results, vitals)
  - Medical images (radiology, pathology)
  - Temporal information (visit sequences)

**3. Hierarchical and Graph-Based Methods**
- Better utilization of ICD code hierarchy
- Label co-occurrence modeling
- Knowledge graph integration
- Taxonomy-aware embeddings

**4. Continual Learning**
- Adaptation to ICD code updates
- Learning from coder feedback
- Domain adaptation across institutions
- Incremental learning without catastrophic forgetting

**5. Evidence-Based Coding**
- Linking predictions to specific text spans
- Sentence-level justifications
- Supporting human coders with highlighted evidence
- Quality assurance and error detection

**6. Cross-Lingual Medical Coding**
- Multilingual transformer models
- Language-agnostic code representations
- Transfer learning across languages
- Low-resource language support

### 5.3 Dataset and Evaluation Needs

**1. Annotated Evidence Datasets**
- MDACE is first but limited (302 inpatient charts)
- Need for larger-scale evidence annotations
- Multiple annotator perspectives
- Coverage of rare codes

**2. Diverse Clinical Settings**
- Emergency department notes (current gap)
- Outpatient clinic documentation
- Specialty-specific datasets (oncology, cardiology)
- Pediatric vs. adult populations

**3. Updated ICD-11 Benchmarks**
- ICD-10 → ICD-11 transition ongoing
- New code structures and relationships
- Backward compatibility challenges

**4. Standardized Evaluation Metrics**
- Beyond micro/macro F1 and AUC
- Clinical utility metrics
- Cost-benefit analysis
- Human-in-the-loop performance

### 5.4 Practical Application Gaps

**1. Real-Time Coding Support**
- Point-of-care code suggestions
- Interactive coding assistance
- Confidence scoring for deferral
- Integration with clinical workflows

**2. Audit and Compliance**
- Detection of coding errors
- Compliance with billing regulations
- Fraud detection
- Quality metrics tracking

**3. Coder Training and Support**
- Educational tools for new coders
- Continuing education support
- Difficult case assistance
- Consistency checking

**4. Cost-Effectiveness Studies**
- ROI analysis for deployment
- Time savings quantification
- Error reduction benefits
- Scalability to different institutions

---

## 6. Relevance to Emergency Department Auto-Coding

### 6.1 ED-Specific Challenges

**Unique Characteristics of ED Documentation:**

1. **Time Pressure and Brevity**
   - ED notes are typically shorter than inpatient discharge summaries
   - Less detailed documentation due to time constraints
   - Abbreviation-heavy language
   - Focus on chief complaint and immediate treatment

2. **Heterogeneity of Conditions**
   - Wide variety of presentations (trauma, cardiac, psychiatric, pediatric)
   - Mix of high-acuity (MI, stroke) and low-acuity (minor injuries) cases
   - Complex multi-system involvement common

3. **Procedure Coding Complexity**
   - Numerous minor procedures (suturing, splinting, IV placement)
   - CPT codes more relevant than in inpatient settings
   - E/M (Evaluation and Management) coding critical for billing

4. **Temporal Constraints**
   - Real-time or near-real-time coding needs
   - Rapid patient turnover
   - Documentation often incomplete at discharge

### 6.2 Applicable Methods from Literature

**1. Hierarchical Attention Models**
- **Why Relevant:** ED notes have clear structure (HPI, exam, assessment, plan)
- **Best Approaches:**
  - HLAN for section-wise attention
  - Encounter-level document attention (ED visits = multiple notes)
  - Partition-based label attention for paragraph-level focus
- **Expected Benefit:** Better handling of structured ED note templates

**2. Few-Shot Learning**
- **Why Relevant:** Many rare ED conditions/procedures
- **Best Approaches:**
  - Knowledge-injected prompt-based fine-tuning
  - Synthetic data generation for rare codes
  - Meta-learning for uncommon presentations
- **Expected Benefit:** Improved coding of infrequent ED diagnoses

**3. Multi-Label Classification**
- **Why Relevant:** ED patients often have multiple diagnoses/procedures
- **Best Approaches:**
  - Label-wise attention mechanisms
  - Multi-stage retrieve and re-rank
  - Label co-occurrence modeling
- **Expected Benefit:** Accurate capture of all relevant codes

**4. Lightweight Models for Speed**
- **Why Relevant:** Real-time coding support needs
- **Best Approaches:**
  - Efficient transformers (ALBERT, DistilBERT)
  - CNN-based models (faster than transformers)
  - Hybrid CNN-transformer architectures
- **Expected Benefit:** Sub-second inference for point-of-care use

### 6.3 Recommended Approach for ED Auto-Coding

**Phase 1: Foundation (Months 1-3)**
1. **Data Collection:**
   - Gather de-identified ED discharge summaries
   - Collect corresponding ICD-10 and CPT codes
   - Annotate with evidence spans (sample subset)
   - Include timestamps for temporal modeling

2. **Baseline Model:**
   - Start with PLM-ICD as baseline
   - Fine-tune on ED-specific data
   - Evaluate on top-50 ED codes initially
   - Benchmark: Target >75% F1 on frequent codes

**Phase 2: Enhancement (Months 4-6)**
1. **Architectural Improvements:**
   - Implement hierarchical attention for ED note structure
   - Add label-wise attention for multi-label accuracy
   - Integrate CPT-specific decoder for procedure codes
   - Test few-shot learning for rare conditions

2. **Feature Engineering:**
   - Incorporate chief complaint explicitly
   - Use triage acuity level
   - Encode vital signs as structured features
   - Add temporal features (time-of-day, day-of-week)

**Phase 3: Explainability (Months 7-9)**
1. **Evidence-Based Coding:**
   - Train evidence extraction layer
   - Implement attention visualization
   - Create human-readable explanations
   - Test with actual ED coders for validation

2. **Confidence Scoring:**
   - Develop uncertainty quantification
   - Set thresholds for automatic vs. human review
   - Implement active learning for edge cases

**Phase 4: Deployment (Months 10-12)**
1. **Integration:**
   - Real-time API for EHR integration
   - Batch processing for retrospective coding
   - Human-in-the-loop interface for review
   - Feedback collection system

2. **Evaluation Metrics:**
   - Coding accuracy vs. certified coders
   - Time savings quantification
   - Error reduction measurement
   - User satisfaction surveys

### 6.4 Expected Performance Targets

Based on literature review and ED-specific considerations:

| Metric | Target | Rationale |
|--------|--------|-----------|
| Top-20 ED Diagnoses (Micro-F1) | 80-85% | High-frequency codes well-represented |
| Top-50 ED Diagnoses (Micro-F1) | 75-80% | Comparable to SOTA on MIMIC |
| All ED Diagnoses (Micro-F1) | 60-65% | ED has fewer unique codes than inpatient |
| Top-20 CPT Procedures (Micro-F1) | 70-75% | Procedures more challenging than diagnoses |
| Rare Code Recall | 40-50% | Few-shot learning enhancement |
| Inference Time | <1 second | Real-time support requirement |
| Evidence Extraction Accuracy | 85-90% | Critical for coder trust |

### 6.5 Key Success Factors

1. **Data Quality:**
   - Clean, de-identified ED note corpus
   - Accurate gold-standard coding
   - Sufficient rare code examples
   - Evidence annotations for training

2. **Model Selection:**
   - Balance accuracy vs. speed
   - Explainability requirements met
   - Scalability to code updates
   - Resource constraints considered

3. **Human Integration:**
   - Coder feedback loop
   - Transparent explanations
   - Confidence-based routing
   - Continuous learning from corrections

4. **Institutional Factors:**
   - EHR system compatibility
   - Coding practice alignment
   - Regulatory compliance
   - Change management for adoption

### 6.6 Risk Mitigation

**Technical Risks:**
- Model drift with code updates → Implement continual learning
- Low accuracy on rare codes → Use few-shot learning + synthetic data
- Long inference time → Optimize model architecture, use model compression

**Operational Risks:**
- Coder resistance → Involve coders in design, emphasize assistance not replacement
- Billing errors → Implement confidence thresholds, human review for low-confidence
- Regulatory issues → Maintain human oversight, comprehensive audit trails

**Data Risks:**
- Privacy breaches → De-identification, secure infrastructure, compliance audits
- Bias in predictions → Fairness testing, diverse training data, bias mitigation
- Data quality issues → Validation pipelines, anomaly detection, data cleaning

---

## 7. Conclusions and Recommendations

### 7.1 State of the Field

Automated medical coding using AI has matured significantly:
- **Strong performance** on frequent codes (75-85% F1)
- **Remaining challenges** with rare codes and explainability
- **Clear path forward** with transformer models and LLMs
- **Increasing focus** on evidence-based, interpretable systems

### 7.2 Best Practices

1. **Start with Pre-trained Models:** PLM-ICD or similar as baseline
2. **Leverage Hierarchies:** Use ICD code structure for better rare code performance
3. **Prioritize Explainability:** Essential for clinical adoption
4. **Use Multi-Stage Approaches:** Retrieve → Re-rank → Verify
5. **Incorporate Domain Knowledge:** Medical ontologies, label relationships, code descriptions
6. **Address Long-Tail:** Few-shot learning, synthetic data, hierarchical joint learning
7. **Validate with Clinicians:** Ground-truth from medical professionals, not just metrics

### 7.3 Recommendations for ED Auto-Coding Project

**Immediate Actions:**
1. Replicate PLM-ICD on ED data as baseline
2. Collect evidence-annotated sample (100-200 notes)
3. Benchmark against human coders
4. Identify top-50 most common ED codes

**Short-Term (3-6 months):**
1. Implement hierarchical attention for ED note structure
2. Add few-shot learning for rare codes
3. Develop explainability module
4. Pilot with small coder group

**Long-Term (6-12 months):**
1. Scale to full ED code set
2. Integrate CPT procedure coding
3. Deploy real-time coding assistance
4. Measure clinical and financial impact

### 7.4 Future-Proofing

- **Monitor LLM developments:** GPT-5, Gemini Ultra, Claude Opus may enable new approaches
- **Prepare for ICD-11:** Transition strategies and model updates
- **Build feedback loops:** Continual learning from coder corrections
- **Invest in infrastructure:** Scalable, maintainable, secure systems

---

## References

This synthesis is based on 75+ papers from ArXiv spanning 2017-2025, focusing on:
- Deep learning for medical coding
- Transformer-based NLP methods
- Multi-label classification techniques
- Clinical NLP and healthcare AI
- Explainable AI for medical applications

Primary datasets referenced:
- MIMIC-III: 52,000+ admissions, 8,929 ICD-9 codes
- MIMIC-IV: Updated version with ICD-10 codes
- MDACE: Evidence-annotated coding dataset
- Private institutional datasets (Xiangya, TrialTrove, etc.)

**Compiled:** December 2025
**For:** Hybrid Reasoning Acute Care Project
**Status:** Comprehensive literature synthesis complete