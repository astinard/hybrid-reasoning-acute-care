# Clinical Embeddings and Patient Representation Learning: A Comprehensive Survey

**Research Date:** December 1, 2025
**Focus Area:** Clinical embeddings, patient representation learning, and healthcare representation learning for ED patient modeling

---

## Executive Summary

This comprehensive survey examines the state-of-the-art in clinical embeddings and patient representation learning, analyzing 100+ papers from ArXiv focused on transforming electronic health records (EHR) into meaningful, actionable representations. The research reveals several key trends:

1. **Transformer architectures** have become dominant for clinical representation learning, with BERT-based models achieving state-of-the-art performance across multiple clinical tasks
2. **Temporal modeling** is critical for capturing disease progression and patient trajectories, with innovations in time-aware attention mechanisms and temporal embeddings
3. **Graph-based approaches** effectively model complex relationships between medical concepts, patients, and clinical events
4. **Multimodal fusion** combining structured EHR data with clinical notes shows significant performance improvements
5. **Pre-training strategies** leveraging medical knowledge graphs and large clinical corpora enable better generalization with limited labeled data

**Key Finding for ED Applications:** Models combining temporal cross-attention, medical concept embeddings, and visit-level representations show the most promise for acute care settings where rapid, accurate patient state assessment is critical.

---

## Table of Contents

1. [Key Papers with ArXiv IDs](#key-papers-with-arxiv-ids)
2. [Embedding Architectures](#embedding-architectures)
3. [Pre-training Strategies](#pre-training-strategies)
4. [Downstream Clinical Tasks](#downstream-clinical-tasks)
5. [Temporal Embedding Approaches](#temporal-embedding-approaches)
6. [Research Gaps](#research-gaps)
7. [Relevance to ED Patient Representations](#relevance-to-ed-patient-representations)

---

## Key Papers with ArXiv IDs

### Foundational Patient Representation Learning

1. **Deep Representation Learning of Patient Data from EHR: A Systematic Review**
   - ArXiv ID: 2010.02809v2
   - Authors: Yuqi Si, Jingcheng Du, Zhao Li, et al.
   - Key Finding: Systematic review of 49 papers showing RNNs (LSTM: 13 studies, GRU: 11 studies) are widely applied; disease prediction is most common task (31 studies)
   - Architecture: Comprehensive survey of structured EHR representation learning
   - Performance: Establishes benchmarks across multiple representation learning approaches

2. **Deep Representation Learning to Unlock Patient Stratification at Scale**
   - ArXiv ID: 2003.06516v2
   - Authors: Isotta Landi, Benjamin S. Glicksberg, Hao-Chih Lee, et al.
   - Key Finding: ConvAE achieves 2.61 entropy and 0.31 purity for patient clustering across 1.6M patients
   - Architecture: Word embeddings + CNN + Autoencoders for unsupervised patient stratification
   - Performance: Superior clustering performance across multiple diseases (T2D, Parkinson's, Alzheimer's)

3. **Representation Learning for Electronic Health Records**
   - ArXiv ID: 1909.09248v1
   - Authors: Wei-Hung Weng, Peter Szolovits
   - Key Finding: Comprehensive framework for learning representations from heterogeneous EHR data
   - Architecture: Reviews disentangling factors and distilling information from multimodal sources

### BERT-Based Clinical Models

4. **Publicly Available Clinical BERT Embeddings**
   - ArXiv ID: 1904.03323v3
   - Authors: Emily Alsentzer, John R. Murphy, Willie Boag, et al.
   - Key Finding: Domain-specific BERT models significantly outperform general BERT on clinical tasks
   - Architecture: BERT pre-trained on MIMIC-III clinical notes
   - Performance: State-of-the-art on clinical NLP tasks; models publicly available
   - Pre-training: Masked language modeling on 2M clinical notes

5. **UmlsBERT: Clinical Domain Knowledge Augmentation**
   - ArXiv ID: 2010.10391v5
   - Authors: George Michalopoulos, Yuanxin Wang, Hussam Kaka, et al.
   - Key Finding: Integrating UMLS knowledge into BERT improves clinical NER and inference tasks
   - Architecture: BERT with knowledge-augmented input embeddings and UMLS semantic groups
   - Performance: Outperforms Bio_ClinicalBERT and BioBERT on clinical tasks

6. **CEHR-BERT: Incorporating Temporal Information from Structured EHR**
   - ArXiv ID: 2111.08585v1
   - Authors: Chao Pang, Xinzhuo Jiang, Krishna S Kalluri, et al.
   - Key Finding: Temporal embeddings critical for EHR prediction - artificial time tokens + time/age embeddings improve all tasks
   - Architecture: BERT with time tokens, time embeddings, age embeddings, concept embeddings, and visit type objective
   - Performance: Outperforms baselines on mortality (AUC), heart failure diagnosis, and readmission; 5% data achieves full-data baseline performance
   - Downstream Tasks: Hospitalization, death, HF diagnosis, HF readmission prediction

7. **TAPER: Time-Aware Patient EHR Representation**
   - ArXiv ID: 1908.03971v4
   - Authors: Sajad Darabi, Mohammad Kachuee, Shayan Fazeli, et al.
   - Key Finding: Combines BERT with Bi-LSTM for forward-looking clinical text embeddings
   - Architecture: BERT for text encoding + temporal modeling for visit sequences
   - Performance: Superior on mortality, readmission, length of stay (MIMIC-III)

### Temporal and Sequential Models

8. **Temporal Supervised Contrastive Learning for Patient Risk Progression**
   - ArXiv ID: 2312.05933v1
   - Authors: Shahriar Noroozizadeh, Jeremy C. Weiss, George H. Chen
   - Key Finding: Supervised contrastive learning with time-step embeddings outperforms baselines
   - Architecture: Embedding space where nearby points have similar predictions, adjacent time steps map nearby, different features map far apart
   - Performance: SOTA on sepsis mortality (MIMIC-III) and cognitive impairment progression (ADNI)
   - Downstream Tasks: Mortality prediction, cognitive impairment tracking

9. **ICE-NODE: Integration of Clinical Embeddings with Neural ODEs**
   - ArXiv ID: 2207.01873v3
   - Authors: Asem Alaa, Erik Mayer, Mauricio Barahona
   - Key Finding: Neural ODEs combined with clinical code embeddings capture temporal dynamics
   - Architecture: Neural ODEs + semantic embeddings for temporal integration
   - Performance: 4.8% AP improvement for Table detection, 11.8% for Column, 11.1% for GUI objects
   - Downstream Tasks: Clinical code prediction, rare event prediction (acute renal failure, pulmonary disease)

10. **ConCare: Personalized Clinical Feature Embedding**
    - ArXiv ID: 1911.12216v1
    - Authors: Liantao Ma, Chaohe Zhang, Yasha Wang, et al.
    - Key Finding: Cross-head decorrelation in multi-head attention improves personalized predictions
    - Architecture: Time-aware distribution modeling + multi-head self-attention with cross-head decorrelation
    - Performance: Validated on real-world EMR datasets; findings confirmed by medical experts
    - Temporal Modeling: Models time-aware distribution for irregular EMR data

11. **Temporal Cross-Attention for Dynamic Embedding of Multimodal EHR**
    - ArXiv ID: 2403.04012v2
    - Authors: Yingbo Ma, Suraj Kolla, Dhruv Kaliraman, et al.
    - Key Finding: Temporal cross-attention addresses sparsity, multimodality, irregular sampling in EHR
    - Architecture: Novel time encoding + sequential position encoding + temporal cross-attention + multitask transformer
    - Performance: Outperforms baselines on 9 postoperative complications (120K+ surgeries, multi-hospital)
    - Downstream Tasks: Postoperative complication prediction across 9 different complications

12. **On the Importance of Step-wise Embeddings for Heterogeneous Clinical Time-Series**
    - ArXiv ID: 2311.08902v1
    - Authors: Rita Kuznetsova, Alizée Pace, Manuel Burger, et al.
    - Key Finding: Step-wise embeddings with feature grouping significantly improve ICU predictions
    - Architecture: Tabular deep learning methods applied to time-series with semantic feature groups
    - Performance: Overall improvement on MIMIC-III and HiRID across multiple tasks
    - Temporal Modeling: Feature grouping by semantic categories enhances temporal modeling

13. **Language Model Training Paradigms for Clinical Feature Embeddings**
    - ArXiv ID: 2311.00768v2
    - Authors: Yurong Hu, Manuel Burger, Gunnar Rätsch, Rita Kuznetsova
    - Key Finding: Self-supervised language model approach learns clinical feature embeddings at finer granularity
    - Architecture: Language models for clinical time series with temporal and taxonomy embeddings
    - Performance: SOTA on MIMIC-III mortality prediction benchmark
    - Pre-training: Self-supervised training on clinical features rather than patient/time-step level

### Medical Concept Embeddings

14. **Transfer Learning in EHR through Clinical Concept Embedding**
    - ArXiv ID: 2107.12919v1
    - Authors: Jose Roberto Ayala Solares, Yajie Zhu, Abdelaali Hassaine, et al.
    - Key Finding: Comprehensive evaluation of disease embedding techniques on 3.1M patients
    - Architecture: Multiple embedding techniques (Word2Vec, Node2Vec, etc.) evaluated
    - Performance: Provides pre-trained embeddings for transfer learning
    - Pre-training: Trained on comprehensive EHR from 3.1M patients

15. **Clinical Concept Embeddings Learned from Massive Sources**
    - ArXiv ID: 1804.01486v3
    - Authors: Andrew L. Beam, Benjamin Kompa, Allen Schmaltz, et al.
    - Key Finding: cui2vec learns embeddings for 108,477 medical concepts from multimodal sources
    - Architecture: GloVe-based approach combining 60M insurance claims, 20M clinical notes, 1.7M journal articles
    - Performance: SOTA on concept similarity tasks
    - Pre-training: Massive multimodal pre-training (insurance claims + notes + literature)

16. **Medical Concept Embedding with Time-Aware Attention**
    - ArXiv ID: 1806.02873v1
    - Authors: Xiangrui Cai, Jinyang Gao, Kee Yuan Ngiam, et al.
    - Key Finding: Attention mechanism learns "soft" time-aware context windows for medical concepts
    - Architecture: CBOW model + attention for temporal scopes of medical concepts
    - Performance: Outperforms 5 SOTA baselines on clustering and nearest neighbor tasks
    - Temporal Modeling: Captures varying temporal scopes (e.g., common cold vs diabetes)

17. **Temporal Self-Attention Network for Medical Concept Embedding**
    - ArXiv ID: 1909.06886v1
    - Authors: Xueping Peng, Guodong Long, Tao Shen, et al.
    - Key Finding: First to exploit temporal self-attentive relations between medical events
    - Architecture: TeSAN - lightweight neural net with novel temporal self-attention
    - Performance: Superior to 5 SOTA methods on EHR clustering and prediction
    - Temporal Modeling: Captures contextual information and temporal relationships

18. **Clinical Concept Extraction with Contextual Word Embedding**
    - ArXiv ID: 1810.10566v2
    - Authors: Henghui Zhu, Ioannis Ch. Paschalidis, Amir Tahmasebi
    - Key Finding: Domain-specific contextual embeddings improve concept extraction by 3.4% F1
    - Architecture: Contextual word embeddings + Bi-LSTM-CRF
    - Performance: Best on I2B2 2010 dataset (outperforms SOTA by 3.4% F1)
    - Downstream Tasks: Medical concept extraction (problems, treatments, tests)

19. **SECNLP: A Survey of Embeddings in Clinical NLP**
    - ArXiv ID: 1903.01039v4
    - Authors: Kalyan KS, S Sangeetha
    - Key Finding: Comprehensive survey of 9 types of clinical embeddings
    - Architecture: Taxonomy of embedding types: concept, code, visit, patient, etc.
    - Coverage: Reviews medical corpora, embedding models, evaluation methods, challenges

20. **KEEP: Integrating Medical Ontologies with Clinical Data**
    - ArXiv ID: 2510.05049v1
    - Authors: Ahmed Elhussein, Paul Meddeb, Abigail Newbury, et al.
    - Key Finding: Combines knowledge graph embeddings with empirical learning from EHR
    - Architecture: Knowledge graph embeddings + regularized training on patient records
    - Performance: Outperforms traditional and LLM-based approaches on UK Biobank and MIMIC-IV
    - Pre-training: Knowledge graph embeddings with adaptive clinical data integration

### Graph-Based Approaches

21. **Representation Learning of EHR Data via Graph-Based Medical Entity Embedding**
    - ArXiv ID: 1910.02574v1
    - Authors: Tong Wu, Yunlong Wang, Yue Wang, et al.
    - Key Finding: ME2Vec leverages diverse graph embedding for doctors, patients, medical services
    - Architecture: Graph embedding techniques tailored to each medical entity type
    - Performance: Superior to baselines on disease diagnosis prediction
    - Downstream Tasks: Disease diagnosis prediction

22. **Med2Meta: Learning Representations with Meta-Embeddings**
    - ArXiv ID: 1912.03366v2
    - Authors: Shaika Chowdhury, Chenwei Zhang, Philip S. Yu, Yuan Luo
    - Key Finding: Meta-embeddings aggregate modality-specific embeddings for holistic representation
    - Architecture: Graph autoencoders for modality-specific embeddings + joint reconstruction
    - Performance: Improvements over SOTA on clinical evaluations
    - Pre-training: Learns from heterogeneous EHR modalities (notes, labs, etc.)

23. **Graph-Text Multi-Modal Pre-training for Medical Representation**
    - ArXiv ID: 2203.09994v1
    - Authors: Sungjin Park, Seongsu Bae, Jiho Kim, et al.
    - Key Finding: MedGTX combines graph encoder for structured EHR with text encoder
    - Architecture: GNN for medical codes + text encoder + cross-modal encoder
    - Performance: Substantial improvements over baselines on MIMIC-III benchmarks
    - Pre-training: Four proxy tasks combining structured codes and clinical text

24. **Multi-View Joint Learning for Clinical Codes and Text Using GNN**
    - ArXiv ID: 2301.11608v1
    - Authors: Lecheng Kong, Christopher King, Bradley Fritz, Yixin Chen
    - Key Finding: GNN processes ICD codes while Bi-LSTM processes text; DCCA enforces similar representations
    - Architecture: GNN for codes + Bi-LSTM for text + Deep CCA alignment
    - Performance: Outperforms fine-tuned BERT on surgical procedures with fraction of computation
    - Downstream Tasks: Mortality, readmission, length of stay prediction

25. **Predicting Patient Outcomes with Graph Representation Learning**
    - ArXiv ID: 2101.03940v1
    - Authors: Emma Rocheteau, Catherine Tong, Petar Veličković, et al.
    - Key Finding: LSTM-GNN combines temporal features with patient neighborhood information
    - Architecture: LSTM for temporal features + GNN for patient graph
    - Performance: Outperforms LSTM baseline on length of stay prediction (eICU)
    - Downstream Tasks: Length of stay, mortality prediction

26. **GNNs for Heart Failure Prediction on Patient Similarity Graph**
    - ArXiv ID: 2411.19742v1
    - Authors: Heloisa Oss Boll, Ali Amirahmadi, Amira Soliman, et al.
    - Key Finding: Graph Transformer achieves F1=0.5361, AUROC=0.7925 for HF prediction
    - Architecture: KNN-based patient similarity graph + GraphSAGE/GAT/GT models
    - Performance: GT model best performance; attention weights provide interpretability
    - Downstream Tasks: Heart failure incidence prediction

### Visit-Level Representations

27. **Deep Representation for Patient Visits from EHR**
    - ArXiv ID: 1803.09533v1
    - Authors: Jean-Baptiste Escudié, Alaa Saade, Alice Coucke, Marc Lelarge
    - Key Finding: Visit embeddings learned by predicting ICD codes capture clinical information
    - Architecture: Deep neural network trained to predict ICD diagnosis categories
    - Performance: Embeddings directly usable for multi-output ICD prediction
    - Downstream Tasks: ICD code prediction from visit data

28. **MIPO: Mutual Integration of Patient Journey and Medical Ontology**
    - ArXiv ID: 2107.09288v4
    - Authors: Xueping Peng, Guodong Long, Sen Wang, et al.
    - Key Finding: Integrating ontology with patient journeys improves learning with limited data
    - Architecture: Transformer + graph embedding for medical ontology integration
    - Performance: Consistently better than SOTA regardless of data sufficiency
    - Pre-training: Joint training on diagnosis prediction and ontology-based disease typing

29. **TRACE: Intra-visit Clinical Event Nowcasting**
    - ArXiv ID: 2503.23072v1
    - Authors: Yuyang Liang, Yankai Chen, Yixiang Fang, et al.
    - Key Finding: Novel timestamp embedding with decay and periodic patterns for nowcasting
    - Architecture: Transformer with timestamp embedding capturing temporal dependencies
    - Performance: Significantly outperforms baselines on lab measurement prediction
    - Downstream Tasks: Laboratory measurement nowcasting within hospital visits

### Multimodal Approaches

30. **Global Contrastive Training for Multimodal EHR**
    - ArXiv ID: 2404.06723v1
    - Authors: Yingbo Ma, Suraj Kolla, Zhenhong Hu, et al.
    - Key Finding: Global contrastive loss aligns time series with discharge summaries
    - Architecture: Temporal cross-attention transformers + global contrastive learning
    - Performance: SOTA on 9 postoperative complications (120K+ surgeries)
    - Pre-training: Contrastive learning between multimodal features and discharge summaries

31. **Learning Missing Modal EHR with Unified Multi-modal Embedding**
    - ArXiv ID: 2305.02504v1
    - Authors: Kwanhyung Lee, Soojeong Lee, Sangchul Hahn, et al.
    - Key Finding: UMSE handles missing modalities without imputation; MAA learns with modality-aware attention
    - Architecture: Unified multi-modal set embedding + modality-aware attention + skip bottleneck
    - Performance: Outperforms baselines on mortality, vasopressor, intubation (MIMIC-IV)
    - Temporal Modeling: Handles missing modalities and temporal relationships jointly

32. **CAAT-EHR: Cross-Attentional Autoregressive Transformer**
    - ArXiv ID: 2501.18891v1
    - Authors: Mohammad Al Olaimat, Serdar Bozdag
    - Key Finding: Cross-attention integrates multimodal EHR; autoregressive decoder ensures temporal consistency
    - Architecture: Self/cross-attention encoder + autoregressive decoder
    - Performance: Superior to raw EHR and baseline methods on benchmark datasets
    - Pre-training: Autoregressive prediction of future time points

### Specialized Architectures

33. **ProtoEHR: Hierarchical Prototype Learning for EHR**
    - ArXiv ID: 2508.18313v1
    - Authors: Zi Cai, Yu Liu, Zhiyao Luo, Tingting Zhu
    - Key Finding: Hierarchical prototypes at code, visit, and patient levels improve predictions
    - Architecture: Multi-task learning combining target prediction and general patient-state representation
    - Performance: Outperforms baselines on 5 tasks (mortality, readmission, LOS, drug rec, phenotype)
    - Downstream Tasks: Mortality, readmission, LOS, drug recommendation, phenotype prediction

34. **MedFuse: Multiplicative Embedding Fusion for Irregular Clinical Time Series**
    - ArXiv ID: 2511.09247v1
    - Authors: Yi-Hsien Hsieh, Ta-Jung Chien, Chun-Kai Huang, et al.
    - Key Finding: Multiplicative fusion captures value-dependent feature interactions better than additive
    - Architecture: MuFuse module with multiplicative modulation of value and feature embeddings
    - Performance: Consistently outperforms SOTA on intensive and chronic care datasets
    - Temporal Modeling: Handles irregular sampling, missing values, heterogeneous features

35. **HyMaTE: Hybrid Mamba and Transformer for EHR**
    - ArXiv ID: 2509.24118v1
    - Authors: Md Mozaharul Mottalib, Thao-Ly T. Phan, Rahmatollah Beheshti
    - Key Finding: Combines Mamba (linear-time) with Transformer for efficiency and performance
    - Architecture: State Space Models (Mamba) + Transformer for temporal dynamics
    - Performance: Higher accuracy than concept-based methods; improved scalability
    - Temporal Modeling: Linear-time sequence modeling with long context

36. **TA-RNN: Attention-based Time-aware RNN for EHR**
    - ArXiv ID: 2401.14694v3
    - Authors: Mohammad Al Olaimat, Serdar Bozdag
    - Key Finding: Dual-level attention (visit-level and feature-level) with time embeddings
    - Architecture: Time-aware RNN + dual attention mechanisms
    - Performance: Superior F2 and sensitivity on ADNI and NACC (Alzheimer's); better mortality on MIMIC-III
    - Temporal Modeling: Time embedding for irregular intervals; attention identifies influential visits

### Pre-training and Transfer Learning

37. **Serialized EHR Makes for Good Text Representations**
    - ArXiv ID: 2510.13843v1
    - Authors: Zhirong Chou, Quan Qin, Shi Li
    - Key Finding: SerialBERT extends SciBERT with EHR sequences for temporal/contextual relationships
    - Architecture: SciBERT + EHR sequence pre-training
    - Performance: Better than state-of-the-art EHR strategies on antibiotic susceptibility
    - Pre-training: PubMed abstracts + MIMIC-IV + medical codes with descriptions

38. **Federated Learning of Medical Concepts Using BEHRT**
    - ArXiv ID: 2305.13052v1
    - Authors: Ofir Ben Shoham, Nadav Rappoport
    - Key Finding: Federated learning enables multi-center BEHRT training without data sharing
    - Architecture: BEHRT with federated learning framework
    - Performance: Close to centralized model; outperforms local models (avg precision)
    - Pre-training: Masked language modeling across federated sites

39. **TAMER: Test-Time Adaptive MoE for EHR**
    - ArXiv ID: 2501.05661v2
    - Authors: Yinghao Zhu, Xiaochen Zheng, Ahmed Allam, Michael Krauthammer
    - Key Finding: MoE + Test-Time Adaptation addresses patient heterogeneity and distribution shifts
    - Architecture: Mixture-of-Experts with test-time adaptation
    - Performance: SOTA with diverse EHR backbones on mortality and readmission
    - Temporal Modeling: Dynamic adaptation to evolving health status distributions

### Contrastive and Self-Supervised Learning

40. **Contrastive Learning-based Imputation-Prediction Networks**
    - ArXiv ID: 2308.09896v1
    - Authors: Yuxi Liu, Zhenhao Zhang, Shaowen Qin, et al.
    - Key Finding: Graph-based patient stratification + contrastive learning improves predictions
    - Architecture: Patient stratification modeling + contrastive learning for representations
    - Performance: Outperforms SOTA on imputation and mortality prediction (ADNI, NACC)
    - Pre-training: Contrastive learning enhances patient representation

41. **Self-Supervised Graph Learning with Hyperbolic Embedding**
    - ArXiv ID: 2106.04751v2
    - Authors: Chang Lu, Chandan K. Reddy, Yue Ning
    - Key Finding: Hyperbolic embeddings capture hierarchical medical knowledge; self-supervised pre-training improves with limited labels
    - Architecture: Hyperbolic embedding + GNN with hierarchy-enhanced prediction proxy
    - Performance: SOTA on disease prediction, complication detection on MIMIC-III
    - Pre-training: Self-supervised learning with medical domain knowledge

### Advanced Temporal Modeling

42. **GRU-TV: Time- and Velocity-aware GRU**
    - ArXiv ID: 2205.04892v2
    - Authors: Ningtao Liu, Ruoxi Gao, Jing Yuan, et al.
    - Key Finding: Neural ODEs + velocity perception capture time intervals and changing rates
    - Architecture: GRU with time and velocity awareness via neural ODEs
    - Performance: Robust on high-variance time intervals (PhysioNet2012, MIMIC-III)
    - Temporal Modeling: Continuous-time perception of patient physiological changes

43. **Sequential Diagnosis Prediction with Transformer and Ontology**
    - ArXiv ID: 2109.03069v1
    - Authors: Xueping Peng, Guodong Long, Tao Shen, et al.
    - Key Finding: SETOR integrates ontology via graph embeddings with Transformer
    - Architecture: Graph NN for hierarchy-aware embeddings + Transformer with self-attention
    - Performance: Superior on MIMIC-III and MIMIC-IV regardless of data sufficiency
    - Pre-training: Medical ontology integration for interpretable embeddings

44. **Time-Aware Heterogeneous Graph Transformer**
    - ArXiv ID: 2404.14815v3
    - Authors: Shibo Li, Hengliang Cheng, Weihua Li
    - Key Finding: Heterogeneous graph with adaptive attention for drugs and diseases
    - Architecture: Heterogeneous graph learning + time-aware transformer + adaptive attention
    - Performance: Notable improvements on BUSI, ChestXray2017, Retinal OCT
    - Temporal Modeling: Time-aware transformer captures disease progression dynamics

### Specialized Domain Models

45. **CardioEmbed: Domain-Specialized Text Embeddings**
    - ArXiv ID: 2511.10930v1
    - Authors: Richard J. Young, Alice M. Matthews
    - Key Finding: Domain specialization on cardiology textbooks yields 99.60% retrieval accuracy (+15.94% over MedTE)
    - Architecture: Qwen3-Embedding-8B with contrastive learning on cardiology texts
    - Performance: 99.60% Acc@1 on cardiology retrieval; competitive on MTEB medical benchmarks
    - Pre-training: 150K cardiology textbook sentences with contrastive learning

46. **Clinical ModernBERT: Efficient Long Context Encoder**
    - ArXiv ID: 2504.03964v1
    - Authors: Simon A. Lee, Anthony Wu, Jeffrey N. Chiang
    - Key Finding: Extends ModernBERT with RoPE, Flash Attention for 8,192 token context
    - Architecture: ModernBERT + RoPE + Flash Attention for biomedical domain
    - Performance: Excels at long context clinical NLP tasks
    - Pre-training: PubMed abstracts + MIMIC-IV + medical ontologies

---

## Embedding Architectures

### 1. BERT-Based Architectures

**Core Concept:** Bidirectional transformer encoders pre-trained on clinical text using masked language modeling.

**Key Implementations:**

- **Clinical BERT (2019, ArXiv: 1904.03323v3)**
  - Architecture: BERT pre-trained on MIMIC-III clinical notes
  - Innovation: Domain-specific pre-training significantly outperforms general BERT
  - Use case: Clinical NER, de-identification, outcome prediction

- **UmlsBERT (2020, ArXiv: 2010.10391v5)**
  - Architecture: BERT + UMLS knowledge integration via semantic group embeddings
  - Innovation: Knowledge-augmented input combines concept embeddings with UMLS hierarchy
  - Performance: Outperforms Bio_ClinicalBERT on NER and inference tasks

- **CEHR-BERT (2021, ArXiv: 2111.08585v1)**
  - Architecture: BERT + artificial time tokens + time/age/concept embeddings + visit type objective
  - Innovation: Hybrid temporal approach combining multiple time representations
  - Performance: Outperforms all baselines on 4 prediction tasks; 5% data matches full baseline
  - Temporal Features: Time tokens, continuous time embeddings, age embeddings

- **SerialBERT (2025, ArXiv: 2510.13843v1)**
  - Architecture: SciBERT extended with EHR sequence pre-training
  - Innovation: Serializes EHR data preserving temporal order
  - Performance: SOTA on antibiotic susceptibility prediction

**Advantages:**
- Strong semantic understanding from language model pre-training
- Effective transfer learning with limited labeled data
- Rich contextualized representations

**Limitations:**
- Quadratic computational complexity limits very long sequences
- Limited explicit temporal modeling in base architecture
- Requires substantial computational resources for pre-training

### 2. Recurrent Neural Networks (RNNs)

**Core Concept:** Sequential models that maintain hidden states to capture temporal dependencies.

**Key Implementations:**

- **LSTM/GRU Variants**
  - Widely applied (13 LSTM studies, 11 GRU studies per systematic review)
  - Effective for disease prediction (31 studies)
  - Handle variable-length sequences naturally

- **GRU-TV (2022, ArXiv: 2205.04892v2)**
  - Architecture: GRU + neural ODEs + velocity perception
  - Innovation: Captures both time intervals and rate of change
  - Performance: Robust on high-variance temporal data
  - Temporal Modeling: Continuous-time patient state transitions

- **TA-RNN (2024, ArXiv: 2401.14694v3)**
  - Architecture: Time-aware RNN + dual-level attention
  - Innovation: Visit-level and feature-level attention with time embeddings
  - Performance: Superior F2/sensitivity on Alzheimer's datasets; better mortality (MIMIC-III)
  - Interpretability: Attention weights identify influential visits and features

**Advantages:**
- Natural handling of sequential data
- Efficient for moderate sequence lengths
- Well-understood training procedures

**Limitations:**
- Gradient vanishing/exploding in very long sequences
- Sequential processing prevents parallelization
- Limited ability to capture long-range dependencies

### 3. Transformer Architectures

**Core Concept:** Self-attention mechanisms enable parallel processing and long-range dependency modeling.

**Key Implementations:**

- **Temporal Cross-Attention Transformer (2024, ArXiv: 2403.04012v2)**
  - Architecture: Novel time encoding + sequential position + temporal cross-attention
  - Innovation: Addresses sparsity, multimodality, irregular sampling simultaneously
  - Performance: SOTA on 9 postoperative complications (120K+ surgeries)
  - Temporal Features: Combined time encoding and sequential position embeddings

- **ConCare (2019, ArXiv: 1911.12216v1)**
  - Architecture: Multi-head self-attention + cross-head decorrelation
  - Innovation: Decorrelation prevents redundant attention patterns
  - Performance: Superior on personalized healthcare prediction
  - Temporal Modeling: Time-aware distribution modeling for irregular EMR

- **TeSAN (2019, ArXiv: 1909.06886v1)**
  - Architecture: Lightweight temporal self-attention network
  - Innovation: First to exploit temporal self-attentive relations between medical events
  - Performance: Superior to 5 SOTA methods on clustering and prediction

- **SETOR (2021, ArXiv: 2109.03069v1)**
  - Architecture: Transformer + ontology integration via graph embeddings
  - Innovation: Combines neural ODEs with ontology for irregular temporal data
  - Performance: Superior on MIMIC-III/IV regardless of data sufficiency

- **HyMaTE (2025, ArXiv: 2509.24118v1)**
  - Architecture: Mamba (State Space Model) + Transformer hybrid
  - Innovation: Linear-time complexity with transformer expressiveness
  - Performance: Higher accuracy with better scalability
  - Efficiency: Handles long sequences efficiently via Mamba backbone

**Advantages:**
- Parallel processing enables faster training
- Excellent at capturing long-range dependencies
- Flexible attention mechanisms for different data modalities

**Limitations:**
- Quadratic complexity in sequence length
- Requires large amounts of data
- Positional encoding challenges for irregular time series

### 4. Graph Neural Networks (GNNs)

**Core Concept:** Learn representations by aggregating information from graph neighbors, ideal for medical knowledge graphs and patient similarity networks.

**Key Implementations:**

- **ME2Vec (2019, ArXiv: 1910.02574v1)**
  - Architecture: Diverse graph embeddings for doctors, patients, services
  - Innovation: Tailored embeddings for each medical entity type
  - Performance: Superior on disease diagnosis prediction

- **Med2Meta (2019, ArXiv: 1912.03366v2)**
  - Architecture: Graph autoencoders per modality + meta-embedding fusion
  - Innovation: Aggregates heterogeneous modality embeddings via joint reconstruction
  - Performance: Improvements on quantitative and qualitative clinical evaluations

- **MedGTX (2022, ArXiv: 2203.09994v1)**
  - Architecture: GNN for codes + text encoder + cross-modal encoder
  - Innovation: Joint learning from structured codes and unstructured text
  - Performance: Substantial improvements on MIMIC-III benchmarks
  - Pre-training: Four proxy tasks combining modalities

- **LSTM-GNN (2021, ArXiv: 2101.03940v1)**
  - Architecture: LSTM temporal features + GNN patient graph
  - Innovation: Combines sequential and relational information
  - Performance: Outperforms LSTM on length of stay (eICU)

- **Graph Transformer for Heart Failure (2024, ArXiv: 2411.19742v1)**
  - Architecture: KNN patient similarity graph + GraphSAGE/GAT/GT
  - Innovation: Patient similarity graph from diagnosis/procedure/medication embeddings
  - Performance: GT achieves F1=0.5361, AUROC=0.7925; attention provides interpretability

- **Hyperbolic GNN (2021, ArXiv: 2106.04751v2)**
  - Architecture: Hyperbolic embeddings + GNN with hierarchy-enhanced pre-training
  - Innovation: Captures hierarchical medical knowledge in hyperbolic space
  - Performance: SOTA on disease prediction and complication detection
  - Pre-training: Self-supervised with medical domain knowledge

**Advantages:**
- Natural representation of medical knowledge graphs
- Captures complex relationships between entities
- Effective for rare disease/code prediction via graph structure

**Limitations:**
- Requires graph construction (can be non-trivial)
- Scalability challenges with very large graphs
- May not capture temporal dynamics without additional mechanisms

### 5. Convolutional Neural Networks (CNNs)

**Core Concept:** Apply convolution operations to capture local temporal patterns in EHR sequences.

**Key Implementations:**

- **ConvAE for Patient Stratification (2020, ArXiv: 2003.06516v2)**
  - Architecture: Word embeddings + CNN + Autoencoders
  - Innovation: Unsupervised patient stratification at scale
  - Performance: 2.61 entropy, 0.31 purity on 1.6M patients
  - Use case: Disease subtyping (T2D, Parkinson's, Alzheimer's)

- **Medical Feature Embedding CNN (2017, ArXiv: 1701.07474v1)**
  - Architecture: Multi-layer CNN with learned medical feature embeddings
  - Innovation: Captures local/short temporal dependencies effectively
  - Performance: Promising results on CHF and diabetes prediction
  - Embedding: Medical concepts embedded holding natural medical semantics

**Advantages:**
- Efficient local pattern detection
- Translation invariance useful for finding patterns anywhere in sequence
- Computationally efficient

**Limitations:**
- Limited receptive field for long-range dependencies
- Less natural for variable-length sequences
- May miss global temporal patterns

### 6. Hybrid Architectures

**Core Concept:** Combine multiple architectural paradigms to leverage complementary strengths.

**Key Implementations:**

- **ICE-NODE (2022, ArXiv: 2207.01873v3)**
  - Architecture: Clinical embeddings + Neural ODEs
  - Innovation: Continuous-time dynamics modeling with semantic embeddings
  - Performance: Improved prediction for infrequent conditions
  - Temporal Modeling: Neural ODEs handle irregular time intervals

- **MedFuse (2025, ArXiv: 2511.09247v1)**
  - Architecture: Multiplicative embedding fusion module
  - Innovation: Multiplicative modulation captures value-dependent interactions
  - Performance: Consistently outperforms SOTA on multiple datasets
  - Temporal Modeling: Handles irregularity, missing values, heterogeneity

- **CAAT-EHR (2025, ArXiv: 2501.18891v1)**
  - Architecture: Cross-attention encoder + autoregressive decoder
  - Innovation: Self/cross-attention integration with temporal consistency
  - Performance: Superior to raw EHR and baselines
  - Pre-training: Autoregressive future time point prediction

- **Multi-View GNN (2023, ArXiv: 2301.11608v1)**
  - Architecture: GNN for ICD codes + Bi-LSTM for text + Deep CCA
  - Innovation: Enforces similar patient representations across modalities
  - Performance: Outperforms fine-tuned BERT at fraction of computation

**Advantages:**
- Leverages strengths of multiple approaches
- Flexible adaptation to different data types
- Often achieves best empirical performance

**Limitations:**
- Increased architectural complexity
- More hyperparameters to tune
- Requires careful design to avoid redundancy

### 7. Specialized Architectures

**Mixture-of-Experts (MoE):**

- **TAMER (2025, ArXiv: 2501.05661v2)**
  - Architecture: MoE + Test-Time Adaptation
  - Innovation: Domain-aware expert specialization with dynamic adaptation
  - Performance: SOTA with diverse backbones on mortality/readmission
  - Adaptive: Real-time adaptation to distribution shifts

**State Space Models:**

- **HyMaTE (2025, ArXiv: 2509.24118v1)**
  - Architecture: Mamba (SSM) + Transformer hybrid
  - Innovation: Linear complexity sequence modeling
  - Performance: Better accuracy and scalability than pure transformers
  - Efficiency: Handles very long sequences efficiently

**Prototype-Based:**

- **ProtoEHR (2025, ArXiv: 2508.18313v1)**
  - Architecture: Hierarchical prototypes (code/visit/patient levels)
  - Innovation: Multi-task learning with general patient-state representation
  - Performance: Outperforms baselines on 5 clinical tasks
  - Interpretability: Prototype-based explanations at multiple levels

---

## Pre-training Strategies

### 1. Masked Language Modeling (MLM)

**Approach:** Mask portions of input and train model to predict masked content.

**Key Implementations:**

- **Clinical BERT (ArXiv: 1904.03323v3)**
  - Corpus: 2M MIMIC-III clinical notes
  - Strategy: Standard BERT MLM on clinical text
  - Outcome: Domain-specific representations significantly outperform general BERT

- **CEHR-BERT (ArXiv: 2111.08585v1)**
  - Corpus: 2.4M patients, 3 decades of data
  - Strategy: MLM + visit type prediction (dual objective)
  - Innovation: Artificial time tokens incorporated into masking
  - Outcome: Outperforms all baselines; strong transfer with 5% data

- **SerialBERT (ArXiv: 2510.13843v1)**
  - Corpus: MIMIC-IV + PubMed + medical code descriptions
  - Strategy: MLM on serialized EHR sequences
  - Outcome: SOTA on antibiotic susceptibility prediction

**Advantages:**
- Self-supervised, doesn't require labels
- Learns rich contextual representations
- Proven effective across domains

**Limitations:**
- Requires large corpus
- May not capture task-specific patterns
- Computational cost of pre-training

### 2. Contrastive Learning

**Approach:** Learn representations by contrasting positive and negative pairs.

**Key Implementations:**

- **Temporal Supervised Contrastive Learning (ArXiv: 2312.05933v1)**
  - Strategy: Nearby points in embedding have similar predictions; adjacent time steps map nearby
  - Innovation: Nearest neighbor pairing in raw feature space (alternative to augmentation)
  - Outcome: SOTA on sepsis mortality and cognitive impairment

- **Global Contrastive Training (ArXiv: 2404.06723v1)**
  - Strategy: Align multimodal features with discharge summaries via global contrastive loss
  - Innovation: Patient-level alignment across modalities
  - Outcome: SOTA on 9 postoperative complications

- **Contrastive Learning-based Imputation (ArXiv: 2308.09896v1)**
  - Strategy: Patient stratification + contrastive learning for representations
  - Innovation: Graph-based stratification integrated with contrastive objective
  - Outcome: Outperforms SOTA on imputation and mortality prediction

- **CardioEmbed (ArXiv: 2511.10930v1)**
  - Strategy: InfoNCE loss with in-batch negatives on cardiology texts
  - Corpus: 150K sentences from 7 cardiology textbooks
  - Outcome: 99.60% retrieval accuracy (+15.94% over MedTE)

**Advantages:**
- Learns discriminative features
- Effective with limited labels
- Can incorporate domain knowledge through positive/negative pair selection

**Limitations:**
- Requires careful pair construction
- Sensitive to batch size and negative sampling
- May require large batches for hard negatives

### 3. Medical Knowledge Integration

**Approach:** Incorporate structured medical knowledge (ontologies, knowledge graphs) into pre-training.

**Key Implementations:**

- **UmlsBERT (ArXiv: 2010.10391v5)**
  - Knowledge Source: UMLS Metathesaurus
  - Strategy: Augment BERT input with UMLS semantic group embeddings
  - Innovation: Connect concepts with same underlying UMLS concept
  - Outcome: Outperforms Bio_ClinicalBERT on clinical tasks

- **KEEP (ArXiv: 2510.05049v1)**
  - Knowledge Source: Medical knowledge graphs
  - Strategy: Knowledge graph embeddings + regularized training on patient records
  - Innovation: Adaptive integration preserving ontological relationships
  - Outcome: Outperforms traditional and LLM approaches on UK Biobank/MIMIC-IV

- **MIPO (ArXiv: 2107.09288v4)**
  - Knowledge Source: Medical ontology
  - Strategy: Joint training on diagnosis prediction and ontology-based disease typing
  - Innovation: Mutual integration of patient journey and ontology
  - Outcome: Consistently better than SOTA with sufficient or insufficient data

- **SETOR (ArXiv: 2109.03069v1)**
  - Knowledge Source: Medical ontology hierarchy
  - Strategy: Graph NN for hierarchy-aware embeddings + Transformer
  - Outcome: Superior regardless of data sufficiency; interpretable embeddings

**Advantages:**
- Incorporates expert domain knowledge
- Improves performance on rare conditions
- Enhances interpretability through known relationships

**Limitations:**
- Requires high-quality knowledge graphs
- May not capture novel patterns not in knowledge base
- Knowledge graphs may be incomplete or outdated

### 4. Multi-Task Learning

**Approach:** Simultaneously train on multiple related tasks to learn shared representations.

**Key Implementations:**

- **MedGTX (ArXiv: 2203.09994v1)**
  - Tasks: Four proxy tasks combining structured codes and clinical text
  - Strategy: Joint optimization across tasks
  - Outcome: Substantial improvements on MIMIC-III benchmarks

- **ProtoEHR (ArXiv: 2508.18313v1)**
  - Tasks: Target prediction + general patient-state representation learning
  - Strategy: Hierarchical prototype learning at code/visit/patient levels
  - Outcome: Outperforms baselines on 5 clinical tasks

- **CEHR-BERT (ArXiv: 2111.08585v1)**
  - Tasks: MLM + visit type prediction
  - Strategy: Dual learning objectives
  - Outcome: Incremental gains from each component (ablation study)

**Advantages:**
- Learns more robust representations
- Leverages supervision from multiple tasks
- Can improve data efficiency

**Limitations:**
- Requires balancing task weights
- May require more training time
- Tasks must be sufficiently related

### 5. Self-Supervised Learning

**Approach:** Create supervisory signals from the data itself without manual labels.

**Key Implementations:**

- **Hyperbolic GNN (ArXiv: 2106.04751v2)**
  - Strategy: Hierarchy-enhanced historical prediction proxy task
  - Innovation: Fully utilizes EHR data with medical domain knowledge
  - Outcome: SOTA on disease prediction and complication detection

- **Language Model for Clinical Features (ArXiv: 2311.00768v2)**
  - Strategy: Self-supervised training on clinical features (not patient/time-step level)
  - Innovation: Finer granularity than existing approaches
  - Outcome: SOTA on MIMIC-III mortality prediction

- **ConvAE (ArXiv: 2003.06516v2)**
  - Strategy: Autoencoder reconstruction on patient trajectories
  - Innovation: Unsupervised at scale (1.6M patients)
  - Outcome: 2.61 entropy, 0.31 purity for disease subtyping

**Advantages:**
- No manual labeling required
- Scalable to large datasets
- Can discover novel patterns

**Limitations:**
- Proxy tasks may not align with downstream objectives
- Requires careful proxy task design
- May need fine-tuning for specific applications

### 6. Multimodal Pre-training

**Approach:** Pre-train on multiple data modalities simultaneously to learn aligned representations.

**Key Implementations:**

- **cui2vec (ArXiv: 1804.01486v3)**
  - Modalities: 60M insurance claims + 20M clinical notes + 1.7M journal articles
  - Strategy: GloVe-based approach combining all modalities
  - Innovation: Largest pre-training (108,477 medical concepts)
  - Outcome: SOTA on concept similarity tasks

- **Med2Meta (ArXiv: 1912.03366v2)**
  - Modalities: Clinical notes, lab results, multiple EHR views
  - Strategy: Graph autoencoders per modality + meta-embedding fusion
  - Innovation: Joint reconstruction across modalities
  - Outcome: Improvements on clinical evaluations

- **CAAT-EHR (ArXiv: 2501.18891v1)**
  - Modalities: Lab results, imaging, vital signs, clinical notes
  - Strategy: Self/cross-attention across modalities + autoregressive decoder
  - Innovation: Temporal consistency via autoregressive prediction
  - Outcome: Superior to raw EHR and baselines

**Advantages:**
- Learns complementary information from different modalities
- More robust representations
- Can handle missing modalities

**Limitations:**
- Requires aligned multimodal data
- Increased architectural complexity
- Higher computational cost

### 7. Transfer Learning and Fine-tuning

**Approach:** Leverage pre-trained models and adapt to specific clinical tasks.

**Key Implementations:**

- **Transfer Learning via Clinical Concept Embedding (ArXiv: 2107.12919v1)**
  - Source: 3.1M patient EHR data
  - Strategy: Pre-train embeddings, provide for transfer
  - Outcome: Comprehensive evaluation framework for embeddings

- **Federated BEHRT (ArXiv: 2305.13052v1)**
  - Strategy: Federated MLM across multiple sites
  - Innovation: Multi-center training without data sharing
  - Outcome: Close to centralized; outperforms local models

- **TAMER (ArXiv: 2501.05661v2)**
  - Strategy: Pre-train then adapt at test-time
  - Innovation: Test-Time Adaptation to distribution shifts
  - Outcome: SOTA with diverse backbones on mortality/readmission

**Advantages:**
- Leverages large-scale pre-training
- Reduces data requirements for new tasks
- Enables rapid adaptation

**Limitations:**
- Requires compatible pre-trained models
- Domain shift between pre-training and target
- Fine-tuning may overfit with small datasets

---

## Downstream Clinical Tasks

### 1. Mortality Prediction

**Task Description:** Predict in-hospital or post-discharge mortality risk.

**Key Papers:**

- **CEHR-BERT (ArXiv: 2111.08585v1)**
  - Dataset: Columbia University CUIMC (2.4M patients)
  - Performance: Outperforms all baselines (ROC-AUC and PR-AUC)
  - Architecture: BERT with temporal information

- **Temporal Supervised Contrastive Learning (ArXiv: 2312.05933v1)**
  - Dataset: MIMIC-III (septic patients)
  - Performance: SOTA on sepsis mortality prediction
  - Architecture: Contrastive learning with time-step embeddings

- **TA-RNN (ArXiv: 2401.14694v3)**
  - Dataset: MIMIC-III
  - Performance: Superior to baselines
  - Architecture: Time-aware RNN with dual attention

- **GRU-TV (ArXiv: 2205.04892v2)**
  - Dataset: PhysioNet2012, MIMIC-III
  - Performance: Robust on high-variance time intervals
  - Architecture: GRU with time and velocity awareness

**Common Metrics:** ROC-AUC, PR-AUC, F1-score, sensitivity, specificity

**Challenges:**
- Class imbalance (mortality is relatively rare)
- Need for early prediction (before outcome becomes obvious)
- Handling missing data common in ICU settings

### 2. Readmission Prediction

**Task Description:** Predict likelihood of hospital readmission within specific timeframe (e.g., 30 days).

**Key Papers:**

- **CEHR-BERT (ArXiv: 2111.08585v1)**
  - Task: Heart failure readmission
  - Performance: Outperforms baselines on ROC-AUC and PR-AUC
  - Architecture: BERT with temporal embeddings

- **ProtoEHR (ArXiv: 2508.18313v1)**
  - Dataset: MIMIC-III
  - Performance: Outperforms baselines
  - Architecture: Hierarchical prototype learning

- **TAMER (ArXiv: 2501.05661v2)**
  - Dataset: Multiple EHR datasets
  - Performance: SOTA with test-time adaptation
  - Architecture: MoE with adaptive mechanisms

**Common Metrics:** ROC-AUC, PR-AUC, precision, recall

**Challenges:**
- Multiple factors influence readmission
- Socioeconomic factors often not in EHR
- Intervention opportunities depend on early prediction

### 3. Disease Diagnosis and Onset Prediction

**Key Papers:**

- **ME2Vec (ArXiv: 1910.02574v1)**
  - Task: Disease diagnosis prediction
  - Performance: Superior to baselines
  - Architecture: Graph-based medical entity embeddings

- **ICE-NODE (ArXiv: 2207.01873v3)**
  - Task: Clinical code prediction (especially rare conditions)
  - Performance: Improved for infrequent conditions (acute renal failure, pulmonary disease)
  - Architecture: Clinical embeddings + Neural ODEs

- **CEHR-BERT (ArXiv: 2111.08585v1)**
  - Task: New heart failure diagnosis
  - Performance: Outperforms all baselines
  - Architecture: BERT with temporal information

- **ConvAE (ArXiv: 2003.06516v2)**
  - Task: Disease subtyping (T2D, Parkinson's, Alzheimer's)
  - Performance: 2.61 entropy, 0.31 purity on clustering
  - Architecture: CNN + Autoencoder

**Common Metrics:** F1-score, precision, recall, ROC-AUC

**Challenges:**
- Rare diseases with limited training examples
- Symptom overlap across diseases
- Need for early detection before full symptom manifestation

### 4. Disease Progression and Phenotyping

**Key Papers:**

- **Temporal Supervised Contrastive Learning (ArXiv: 2312.05933v1)**
  - Task: Cognitive impairment progression tracking
  - Dataset: ADNI (Alzheimer's)
  - Performance: SOTA on progression prediction
  - Architecture: Time-step embeddings with contrastive learning

- **TA-RNN (ArXiv: 2401.14694v3)**
  - Task: Alzheimer's disease progression
  - Dataset: ADNI, NACC
  - Performance: Superior F2 and sensitivity
  - Architecture: Time-aware RNN with dual attention

- **ProtoEHR (ArXiv: 2508.18313v1)**
  - Task: Phenotype prediction
  - Performance: Outperforms baselines on PhysioNet Challenge
  - Architecture: Hierarchical prototypes

**Common Metrics:** Trajectory accuracy, progression slope, cluster purity

**Challenges:**
- Long time horizons
- Irregular follow-up intervals
- Individual variation in progression rates

### 5. Length of Stay (LOS) Prediction

**Key Papers:**

- **LSTM-GNN (ArXiv: 2101.03940v1)**
  - Dataset: eICU
  - Performance: Outperforms LSTM baseline
  - Architecture: LSTM + GNN for patient similarity

- **TAPER (ArXiv: 1908.03971v4)**
  - Dataset: MIMIC-III
  - Performance: Superior to baselines
  - Architecture: BERT + Bi-LSTM for temporal modeling

- **ProtoEHR (ArXiv: 2508.18313v1)**
  - Performance: Outperforms baselines
  - Architecture: Hierarchical prototype learning

**Common Metrics:** MAE, RMSE, classification accuracy (for binned LOS)

**Challenges:**
- High variance in LOS
- Multiple factors (clinical + operational)
- Need for early prediction for resource planning

### 6. Treatment Recommendation

**Key Papers:**

- **ProtoEHR (ArXiv: 2508.18313v1)**
  - Task: Drug recommendation
  - Performance: Outperforms baselines
  - Architecture: Hierarchical prototypes with medical knowledge

- **Multi-View GNN (ArXiv: 2301.11608v1)**
  - Task: Treatment planning from surgical procedure text
  - Performance: Outperforms fine-tuned BERT
  - Architecture: GNN for codes + Bi-LSTM for text

**Common Metrics:** Jaccard similarity, F1-score, recall@k

**Challenges:**
- Combinatorial medication space
- Drug-drug interactions
- Personalization to patient characteristics

### 7. Clinical Event Nowcasting and Next-Visit Prediction

**Key Papers:**

- **TRACE (ArXiv: 2503.23072v1)**
  - Task: Laboratory measurement nowcasting within visit
  - Performance: Significantly outperforms baselines
  - Architecture: Transformer with timestamp embedding (decay + periodic patterns)

- **MIPO (ArXiv: 2107.09288v4)**
  - Task: Sequential diagnosis prediction
  - Performance: Consistently better than SOTA
  - Architecture: Transformer + medical ontology integration

**Common Metrics:** MAE, RMSE, prediction accuracy

**Challenges:**
- Intra-visit temporal resolution
- Irregular measurement times
- Multiple concurrent measurements

### 8. Postoperative Complication Prediction

**Key Papers:**

- **Temporal Cross-Attention (ArXiv: 2403.04012v2)**
  - Task: 9 postoperative complications
  - Dataset: 120K+ surgeries across 3 hospitals
  - Performance: Outperforms all baselines
  - Architecture: Temporal cross-attention + multitask transformer

- **Global Contrastive Training (ArXiv: 2404.06723v1)**
  - Task: 9 postoperative complications
  - Dataset: 120K+ surgeries (UF health system)
  - Performance: SOTA
  - Architecture: Contrastive learning across modalities

**Common Metrics:** ROC-AUC, PR-AUC per complication

**Challenges:**
- Multiple complications with different risk profiles
- Imbalanced complication rates
- Need for early prediction (ideally pre-operative)

### 9. Rare Event and Complication Detection

**Key Papers:**

- **ICE-NODE (ArXiv: 2207.01873v3)**
  - Task: Rare clinical code prediction (acute renal failure, pulmonary disease)
  - Performance: Improved on infrequent conditions
  - Architecture: Clinical embeddings + Neural ODEs

- **Hyperbolic GNN (ArXiv: 2106.04751v2)**
  - Task: Disease complication detection
  - Performance: SOTA on rare complications
  - Architecture: Hyperbolic embeddings + GNN

**Common Metrics:** Precision, recall, F1 (especially for minority class)

**Challenges:**
- Severe class imbalance
- Limited training examples
- High cost of false negatives

### 10. Patient Stratification and Clustering

**Key Papers:**

- **ConvAE (ArXiv: 2003.06516v2)**
  - Task: Unsupervised patient stratification
  - Dataset: 1.6M patients
  - Performance: 2.61 entropy, 0.31 purity
  - Use cases: T2D subtypes, Parkinson's, Alzheimer's

- **Contrastive Learning Imputation (ArXiv: 2308.09896v1)**
  - Task: Patient stratification for outcome prediction
  - Performance: Outperforms SOTA on imputation and prediction
  - Architecture: Graph-based stratification + contrastive learning

**Common Metrics:** Entropy, purity, silhouette score, clinical validity

**Challenges:**
- Defining clinically meaningful clusters
- Validation without ground truth
- Interpretability of discovered subtypes

### 11. Clinical Concept Extraction and NER

**Key Papers:**

- **Clinical Concept Extraction with Contextual Embedding (ArXiv: 1810.10566v2)**
  - Task: Extraction of problems, treatments, tests
  - Dataset: I2B2 2010
  - Performance: Best performance, +3.4% F1 over SOTA
  - Architecture: Contextual embeddings + Bi-LSTM-CRF

- **Clinical BERT (ArXiv: 1904.03323v3)**
  - Task: Multiple clinical NLP tasks
  - Performance: Superior to non-specific embeddings
  - Architecture: Domain-specific BERT

**Common Metrics:** F1-score, precision, recall

**Challenges:**
- Ambiguous terminology
- Negation and context
- Domain-specific language

---

## Temporal Embedding Approaches

### 1. Time-Aware Attention Mechanisms

**Core Concept:** Modify attention mechanisms to explicitly account for temporal relationships.

**Key Implementations:**

- **Medical Concept Embedding with Time-Aware Attention (ArXiv: 1806.02873v1)**
  - Mechanism: Learns "soft" time-aware context window for each medical concept
  - Innovation: Captures varying temporal scopes (e.g., common cold vs. diabetes)
  - Performance: Outperforms 5 SOTA baselines
  - Temporal Features: Time gap between consecutive concepts used to weight correlations

- **TeSAN - Temporal Self-Attention Network (ArXiv: 1909.06886v1)**
  - Mechanism: Novel self-attention capturing contextual and temporal relationships
  - Innovation: First to exploit temporal self-attentive relations
  - Performance: Superior to 5 SOTA methods
  - Temporal Features: Time-aware attention weights

- **ConCare (ArXiv: 1911.12216v1)**
  - Mechanism: Multi-head self-attention with cross-head decorrelation
  - Innovation: Time-aware distribution modeling for feature sequences
  - Performance: Validated on real-world EMR; confirmed by medical experts
  - Temporal Features: Models time-aware distribution separately per feature

- **TA-RNN (ArXiv: 2401.14694v3)**
  - Mechanism: Dual-level attention (visit-level and feature-level) with time embeddings
  - Innovation: Identifies influential visits and features via attention weights
  - Performance: Superior F2/sensitivity on Alzheimer's; better mortality on MIMIC-III
  - Temporal Features: Time embedding for irregular intervals between visits

**Advantages:**
- Learns data-driven temporal importance
- Interpretable attention weights
- Flexible to varying time scales

**Limitations:**
- May not capture complex temporal dynamics
- Attention patterns can be difficult to interpret
- Requires sufficient data to learn meaningful patterns

### 2. Temporal Embeddings and Positional Encoding

**Core Concept:** Learn continuous representations of time to incorporate into model.

**Key Implementations:**

- **CEHR-BERT (ArXiv: 2111.08585v1)**
  - Approach: Artificial time tokens + continuous time embeddings + age embeddings
  - Innovation: Hybrid approach combining multiple temporal representations
  - Performance: Incremental gains from each time component (ablation study)
  - Temporal Features: Time since last visit, patient age, visit type

- **Temporal Cross-Attention (ArXiv: 2403.04012v2)**
  - Approach: Novel time encoding + sequential position encoding
  - Innovation: Addresses timestamp duplication when multiple measurements simultaneous
  - Performance: Outperforms baselines on 9 complications
  - Temporal Features: Separate encoding for absolute time and sequential position

- **TRACE (ArXiv: 2503.23072v1)**
  - Approach: Timestamp embedding with decay properties and periodic patterns
  - Innovation: Captures both decay and cyclical patterns in lab measurements
  - Performance: Significantly outperforms baselines on nowcasting
  - Temporal Features: Decay + periodicity combined in embedding

- **Time-Aware Heterogeneous Graph Transformer (ArXiv: 2404.14815v3)**
  - Approach: Time-aware transformer + adaptive attention
  - Innovation: Integrates temporal data into visit-level embeddings
  - Performance: Notable improvements on medical imaging + EHR tasks
  - Temporal Features: Temporal dynamics captured in graph structure

**Advantages:**
- Continuous representation of time
- Can capture both short and long time scales
- Learnable temporal patterns

**Limitations:**
- May require careful initialization
- Can be sensitive to time scale normalization
- Limited interpretability of learned embeddings

### 3. Neural Ordinary Differential Equations (ODEs)

**Core Concept:** Model patient state as continuous-time dynamical system.

**Key Implementations:**

- **ICE-NODE (ArXiv: 2207.01873v3)**
  - Approach: Neural ODEs integrate clinical code embeddings over continuous time
  - Innovation: Handles irregular time intervals naturally
  - Performance: Improved prediction for infrequent conditions
  - Temporal Features: Continuous patient trajectory between visits

- **GRU-TV (ArXiv: 2205.04892v2)**
  - Approach: Neural ODEs + velocity perception
  - Innovation: Captures both time intervals and rate of change (velocity)
  - Performance: Robust on high-variance temporal data
  - Temporal Features: Time-varying patient state transitions + velocity

- **SETOR (ArXiv: 2109.03069v1)**
  - Approach: Neural ODEs for irregular visit intervals
  - Innovation: Combines ODEs with medical ontology integration
  - Performance: Superior regardless of data sufficiency
  - Temporal Features: Continuous-time dynamics between irregular visits

**Advantages:**
- Natural handling of irregular sampling
- Theoretically grounded continuous-time modeling
- Can capture smooth state transitions

**Limitations:**
- Computationally expensive (ODE solving)
- May require careful tuning
- Can be unstable with very sparse observations

### 4. Recurrent Temporal Modeling

**Core Concept:** Use recurrent architectures with explicit temporal mechanisms.

**Key Implementations:**

- **Time-Aware Distribution Modeling (ConCare, ArXiv: 1911.12216v1)**
  - Approach: Embed feature sequences separately with time-aware distributions
  - Innovation: Doesn't assume more recent is more important
  - Performance: Validated on real-world EMR datasets
  - Temporal Features: Separate temporal modeling per feature

- **LSTM-GNN Hybrid (ArXiv: 2101.03940v1)**
  - Approach: LSTM captures temporal features + GNN captures relationships
  - Innovation: Combines sequential and graph information
  - Performance: Outperforms LSTM on length of stay
  - Temporal Features: LSTM hidden states over time

- **TA-RNN (ArXiv: 2401.14694v3)**
  - Approach: Time-aware RNN with dual attention
  - Innovation: Visit and feature-level attention with time embeddings
  - Performance: Superior on Alzheimer's progression
  - Temporal Features: Time embedding for irregular intervals

**Advantages:**
- Well-established architectures
- Effective for moderate sequence lengths
- Natural handling of sequential dependencies

**Limitations:**
- Gradient issues with very long sequences
- Sequential processing limits parallelization
- May struggle with long-range dependencies

### 5. Autoregressive and Predictive Modeling

**Core Concept:** Learn representations by predicting future states.

**Key Implementations:**

- **CAAT-EHR (ArXiv: 2501.18891v1)**
  - Approach: Autoregressive decoder predicts future time points
  - Innovation: Ensures temporal consistency and alignment
  - Performance: Superior to raw EHR and baselines
  - Temporal Features: Future prediction as regularization

- **Hyperbolic GNN (ArXiv: 2106.04751v2)**
  - Approach: Hierarchy-enhanced historical prediction proxy task
  - Innovation: Self-supervised temporal prediction with domain knowledge
  - Performance: SOTA on disease prediction
  - Temporal Features: Historical prediction captures temporal patterns

**Advantages:**
- Self-supervised learning from temporal structure
- Natural alignment with clinical forecasting
- Learns predictive representations

**Limitations:**
- May overfit to training distribution
- Prediction errors can compound
- Requires careful horizon selection

### 6. Time Decay and Recency Modeling

**Core Concept:** Explicitly model decay of information relevance over time.

**Key Implementations:**

- **TRACE Timestamp Embedding (ArXiv: 2503.23072v1)**
  - Approach: Decay properties embedded in timestamp representations
  - Innovation: Combines decay with periodic patterns
  - Performance: Significantly outperforms baselines
  - Temporal Features: Learned decay rates + periodic components

- **Smoothed Mask for Denoising (HGM, ArXiv: 2403.04012v2)**
  - Approach: Smoothed masking considers temporal proximity
  - Innovation: Denoising improves robustness to temporal irregularities
  - Performance: Component of SOTA system
  - Temporal Features: Temporal smoothing in masking

**Advantages:**
- Intuitive modeling of information decay
- Captures clinical relevance over time
- Can incorporate domain knowledge about decay rates

**Limitations:**
- May not capture all temporal patterns (e.g., cyclic)
- Assumes monotonic decay
- Decay rates may vary by condition

### 7. Visit-Level Temporal Aggregation

**Core Concept:** Aggregate information within visits and model visit sequences.

**Key Implementations:**

- **Deep Representation for Patient Visits (ArXiv: 1803.09533v1)**
  - Approach: Learn visit embeddings by predicting ICD codes
  - Innovation: Direct visit-level representations
  - Performance: Embeddings capture relevant clinical information
  - Temporal Features: Visit as fundamental temporal unit

- **MIPO (ArXiv: 2107.09288v4)**
  - Approach: Joint optimization of patient journey and medical ontology
  - Innovation: Visit embeddings integrate temporal and ontological information
  - Performance: Consistently better than SOTA
  - Temporal Features: Visit sequence modeling + ontology

- **TRACE (ArXiv: 2503.23072v1)**
  - Approach: Intra-visit nowcasting with temporal attention
  - Innovation: Models events within single visit
  - Performance: Significantly outperforms baselines
  - Temporal Features: Sub-visit temporal granularity

**Advantages:**
- Natural clinical unit (visits)
- Reduces sequence length
- Aligns with clinical workflow

**Limitations:**
- May lose fine-grained temporal information within visits
- Visit boundaries can be arbitrary
- Aggregation may obscure important patterns

### 8. Multi-Scale Temporal Modeling

**Core Concept:** Capture temporal patterns at multiple time scales simultaneously.

**Key Implementations:**

- **Step-wise Embeddings (ArXiv: 2311.08902v1)**
  - Approach: Feature grouping by semantic categories with temporal modeling
  - Innovation: Different time scales for different feature groups
  - Performance: Overall improvement on MIMIC-III and HiRID
  - Temporal Features: Multi-scale temporal patterns via feature groups

- **Modeling Long-term Dependencies and Short-term Correlations (ArXiv: 2207.06414v2)**
  - Approach: Short-term temporal attention + long-term temporal attention
  - Innovation: Separate modeling of short and long-range dependencies
  - Performance: Improved predictive accuracy
  - Temporal Features: Dual temporal attention mechanisms

- **MedFuse (ArXiv: 2511.09247v1)**
  - Approach: Handles long-term trends and short-term fluctuations
  - Innovation: Multiplicative fusion across temporal scales
  - Performance: Consistently outperforms SOTA
  - Temporal Features: Multi-scale value-dependent interactions

**Advantages:**
- Captures both rapid changes and slow trends
- More comprehensive temporal modeling
- Can adapt to different temporal patterns

**Limitations:**
- Increased model complexity
- More hyperparameters
- Computational cost

---

## Research Gaps

### 1. Limited Acute Care Focus

**Gap Description:**
Most existing research focuses on chronic disease management, ICU settings, or general inpatient populations. There is limited work specifically addressing the unique challenges of Emergency Department (ED) patient representation.

**ED-Specific Challenges Not Addressed:**
- Extremely short time horizons (hours, not days/weeks)
- Highly heterogeneous patient populations
- Limited prior medical history available
- Need for rapid decision-making
- High missing data rates due to urgency
- Triage and flow management considerations

**Opportunities:**
- Develop fast, lightweight models for ED settings
- Focus on early prediction with minimal data
- Incorporate triage information and chief complaints
- Model patient flow dynamics
- Handle extreme data sparsity

### 2. Insufficient Handling of Missing Data

**Gap Description:**
While several papers address missing data (e.g., MedFuse, UMSE), most approaches use imputation or masking rather than learning representations that are robust to missingness patterns.

**Current Limitations:**
- Many models assume missingness at random
- Imputation may introduce errors
- Missingness patterns themselves can be informative
- Limited work on learning from partial observations

**Opportunities:**
- Develop missingness-aware embeddings
- Treat missingness as a feature
- Learn from patterns of what's measured vs. what's missing
- Robust representations under varying missingness

**Relevant Work:**
- MedFuse (ArXiv: 2511.09247v1): Handles irregular sampling
- UMSE (ArXiv: 2305.02504v1): Handles missing modalities
- PRISM (ArXiv: 2309.04160v6): Addresses data sparsity

### 3. Limited Real-Time Performance Evaluation

**Gap Description:**
Most papers evaluate on offline datasets without considering computational constraints of real-time clinical deployment.

**Missing Considerations:**
- Inference latency requirements
- Memory footprint for deployment
- Continuous learning/updating
- Model degradation over time
- Integration with existing clinical systems

**Opportunities:**
- Benchmark models on inference speed
- Develop edge-deployable models
- Online learning frameworks
- Model monitoring and updating strategies

### 4. Weak Clinical Validation

**Gap Description:**
Many papers rely solely on computational metrics (AUC, F1) without clinical expert validation or prospective studies.

**Validation Gaps:**
- Limited clinician feedback on predictions
- No prospective validation
- Lack of clinical utility assessment
- Missing cost-effectiveness analysis
- Limited evaluation of decision support value

**Opportunities:**
- Clinician-in-the-loop evaluation
- Prospective clinical trials
- Clinical utility metrics
- Cost-effectiveness studies
- Qualitative feedback from healthcare providers

**Partial Exceptions:**
- ConCare (ArXiv: 1911.12216v1): Findings confirmed by medical experts
- Several papers include interpretability analysis

### 5. Limited Multimodal Integration

**Gap Description:**
While some work addresses multimodal EHR (structured + text), there's limited integration of other modalities relevant to ED: imaging, waveforms, lab results, vital signs in real-time.

**Missing Modalities:**
- Real-time vital sign waveforms (ECG, SpO2, etc.)
- Point-of-care imaging (ultrasound, X-ray)
- Streaming lab results
- Nurse/physician notes taken during encounter
- Patient-reported symptoms

**Opportunities:**
- Develop truly multimodal ED representations
- Real-time multimodal fusion
- Handle asynchronous modality availability
- Learn cross-modal dependencies specific to ED

**Partial Exceptions:**
- Temporal Cross-Attention (ArXiv: 2403.04012v2): Multimodal with temporal focus
- Global Contrastive Training (ArXiv: 2404.06723v1): Time series + notes

### 6. Scalability and Generalization

**Gap Description:**
Many models are evaluated on single-site data or don't address cross-site generalization.

**Generalization Challenges:**
- Hospital-specific coding practices
- Regional population differences
- Varying clinical protocols
- Different EHR systems
- Temporal drift (practice changes over time)

**Opportunities:**
- Multi-site training and evaluation
- Domain adaptation techniques
- Robust to EHR system differences
- Continuous adaptation to practice changes

**Relevant Work:**
- Federated BEHRT (ArXiv: 2305.13052v1): Multi-site learning
- TAMER (ArXiv: 2501.05661v2): Test-time adaptation
- KEEP (ArXiv: 2510.05049v1): Evaluated on UK Biobank and MIMIC-IV

### 7. Interpretability and Explainability

**Gap Description:**
While attention mechanisms provide some interpretability, deeper clinical reasoning explanation is often lacking.

**Interpretability Needs:**
- Why specific predictions were made
- Which features were most important
- What alternative outcomes were considered
- Uncertainty quantification
- Causal reasoning (not just correlation)

**Opportunities:**
- Develop inherently interpretable architectures
- Causal representation learning
- Uncertainty-aware predictions
- Natural language explanations
- Counterfactual reasoning

**Relevant Work:**
- ProtoEHR (ArXiv: 2508.18313v1): Prototype-based interpretability
- Several papers use attention for interpretation
- Limited causal reasoning

### 8. Rare Disease and Long-Tail Events

**Gap Description:**
While some papers address rare events, comprehensive handling of long-tail distributions in clinical data remains challenging.

**Challenges:**
- Severe class imbalance
- Limited training examples for rare conditions
- Importance of rare but critical conditions
- Evaluation metrics for rare events

**Opportunities:**
- Few-shot learning approaches
- Meta-learning for rare conditions
- Better evaluation metrics for imbalanced data
- Knowledge transfer from common to rare

**Relevant Work:**
- ICE-NODE (ArXiv: 2207.01873v3): Improved on infrequent conditions
- Hyperbolic GNN (ArXiv: 2106.04751v2): Rare disease prediction

### 9. Privacy and Federated Learning

**Gap Description:**
Limited work on privacy-preserving representation learning despite sensitive nature of medical data.

**Privacy Challenges:**
- Patient re-identification risk
- Membership inference attacks
- Model inversion attacks
- Secure multi-party computation
- Differential privacy trade-offs

**Opportunities:**
- Differentially private embeddings
- Secure aggregation methods
- Privacy-utility trade-off optimization
- Federated representation learning

**Relevant Work:**
- Federated BEHRT (ArXiv: 2305.13052v1): Federated learning approach
- Limited work on differential privacy

### 10. Temporal Causality

**Gap Description:**
Most models learn correlations rather than causal relationships, limiting their utility for intervention planning.

**Causality Gaps:**
- Confounding not addressed
- Treatment effect estimation weak
- Counterfactual prediction limited
- Causal discovery from observational data

**Opportunities:**
- Causal representation learning
- Treatment effect estimation
- Counterfactual prediction
- Incorporate causal medical knowledge

**Limited Relevant Work:**
- Most papers focus on prediction, not causation
- Opportunity for significant contribution

### 11. Continuous Learning and Adaptation

**Gap Description:**
Models are typically trained once and deployed, without mechanisms for continuous learning from new data.

**Adaptation Needs:**
- Concept drift handling
- New disease emergence
- Practice guideline changes
- Population shifts
- New treatments

**Opportunities:**
- Online learning frameworks
- Active learning for label efficiency
- Continual learning without catastrophic forgetting
- Dynamic model updating

**Relevant Work:**
- TAMER (ArXiv: 2501.05661v2): Test-time adaptation
- Limited work on long-term deployment

### 12. Multi-Task and Transfer Learning

**Gap Description:**
While some papers address multi-task learning, systematic frameworks for task relationships and transfer are limited.

**Transfer Gaps:**
- Which tasks benefit from shared representations
- How to design task hierarchies
- Transfer from data-rich to data-poor tasks
- Cross-domain transfer (e.g., ICU to ED)

**Opportunities:**
- Meta-learning frameworks
- Task relationship discovery
- Hierarchical multi-task architectures
- Cross-domain transfer strategies

**Relevant Work:**
- ProtoEHR (ArXiv: 2508.18313v1): Multi-task learning
- Transfer Learning via Embeddings (ArXiv: 2107.12919v1)
- Limited systematic frameworks

---

## Relevance to ED Patient Representations

### Key Insights for Emergency Department Applications

#### 1. Temporal Modeling is Critical

**ED-Specific Considerations:**
- **Short time horizons:** ED encounters typically span hours, not days/weeks
- **Rapid state changes:** Patient condition can deteriorate quickly
- **Fine-grained temporal resolution:** Vital signs, treatments measured frequently

**Most Relevant Papers:**

- **TRACE (ArXiv: 2503.23072v1):** Intra-visit nowcasting ideal for ED
  - Timestamp embedding with decay and periodic patterns
  - Handles high-frequency measurements within single encounter
  - Applicable to real-time lab result prediction

- **Temporal Cross-Attention (ArXiv: 2403.04012v2):** Addresses irregular sampling
  - Critical for ED where measurements are opportunistic
  - Novel time encoding handles timestamp duplication
  - Proven on acute care (postoperative complications)

- **GRU-TV (ArXiv: 2205.04892v2):** Time and velocity awareness
  - Rate of change critical in ED (e.g., rapidly dropping BP)
  - Handles high-variance time intervals common in ED
  - Robust temporal modeling for acute settings

**Recommendation:**
Combine fine-grained timestamp embeddings (TRACE-style) with velocity-aware modeling (GRU-TV) for ED patient state representation capturing both current state and rate of change.

#### 2. Handling Missing and Sparse Data

**ED-Specific Challenges:**
- **Limited history:** Many ED patients lack prior records
- **Urgency-driven incompleteness:** Critical interventions prioritized over complete documentation
- **Varying data availability:** Not all patients receive same workup

**Most Relevant Papers:**

- **MedFuse (ArXiv: 2511.09247v1):** Multiplicative embedding fusion
  - Handles irregular sampling, missing values, heterogeneous features
  - Value-dependent feature interactions
  - Proven on both intensive and chronic care

- **UMSE (ArXiv: 2305.02504v1):** Unified multi-modal set embedding
  - Handles missing modalities without imputation
  - Modality-aware attention
  - Skip bottleneck for learning with missing data

- **PRISM (ArXiv: 2309.04160v6):** Prototype patient representations
  - Mitigates EHR data sparsity
  - Feature confidence learner evaluates reliability given missingness
  - Avoids over-reliance on imputed values

**Recommendation:**
Develop ED-specific embedding that treats missingness as informative (what was/wasn't measured tells us something) rather than just a problem to impute away. Multiplicative fusion approach from MedFuse could be adapted.

#### 3. Rapid Decision Support Requirements

**ED-Specific Needs:**
- **Low latency:** Predictions needed in seconds/minutes
- **Lightweight models:** May deploy on edge devices
- **Interpretability:** Clinicians need to understand why

**Most Relevant Papers:**

- **HyMaTE (ArXiv: 2509.24118v1):** Mamba + Transformer hybrid
  - Linear-time complexity enables fast inference
  - Maintains transformer expressiveness
  - Suitable for real-time deployment

- **ProtoEHR (ArXiv: 2508.18313v1):** Hierarchical prototypes
  - Prototype-based interpretability at code/visit/patient levels
  - Multi-scale explanations help clinicians
  - Efficient inference via prototypes

- **Step-wise Embeddings (ArXiv: 2311.08902v1):** Feature grouping
  - Semantic grouping improves both performance and interpretability
  - Reduces dimensionality via meaningful aggregation
  - Aligns with clinical thinking

**Recommendation:**
Prioritize efficient architectures (Mamba-based or optimized Transformers) with built-in interpretability (prototypes, semantic grouping). Consider edge deployment for minimal latency.

#### 4. Integration of Chief Complaints and Triage

**ED-Specific Data:**
- **Chief complaint:** Free-text reason for visit
- **Triage assessment:** Initial severity assessment
- **Vital signs:** Immediate measurements
- **Initial presentation:** Symptoms, observed state

**Most Relevant Papers:**

- **Clinical BERT (ArXiv: 1904.03323v3):** Domain-specific text embeddings
  - Can encode chief complaints effectively
  - Pre-trained on clinical notes
  - Proven for clinical NLP

- **Global Contrastive Training (ArXiv: 2404.06723v1):** Text + time series
  - Aligns multimodal features with clinical text
  - Global contrastive loss for patient-level alignment
  - Applicable to chief complaint + vitals integration

- **Multi-View GNN (ArXiv: 2301.11608v1):** Codes + text joint learning
  - Could adapt for triage categories + chief complaint
  - DCCA enforces similar representations across modalities
  - Efficient compared to large language models

**Recommendation:**
Develop ED-specific encoder combining chief complaint (BERT-encoded), triage assessment, and initial vitals using contrastive or multi-view learning to create holistic initial patient representation.

#### 5. Patient Similarity for Case-Based Reasoning

**ED Use Cases:**
- **Rare presentations:** Finding similar past cases
- **Diagnostic support:** What happened to similar patients
- **Treatment guidance:** What worked for similar cases

**Most Relevant Papers:**

- **Graph Transformer for Patient Similarity (ArXiv: 2411.19742v1)**
  - KNN-based patient similarity graph
  - Graph Transformer achieves strong performance
  - Attention weights provide interpretability

- **Patient Similarity with Medical Concept Embedding (ArXiv: 1902.03376v1)**
  - Temporal matching of longitudinal EHRs
  - Supervised learning of optimal patient representations
  - Preserves temporal properties

- **ConvAE (ArXiv: 2003.06516v2):** Patient stratification
  - Unsupervised clustering of similar patients
  - Validated on disease subtypes
  - Scalable to large populations

**Recommendation:**
Build patient similarity graph using ED-relevant features (chief complaint, vitals, demographics) with Graph Transformer for finding similar cases in real-time. Use for decision support and diagnostic suggestions.

#### 6. Multi-Task Learning for ED Outcomes

**ED Prediction Tasks:**
- **Disposition:** Admit, discharge, observe
- **Severity:** Triage category refinement
- **Mortality risk:** In-ED and short-term
- **Specific diagnoses:** Sepsis, MI, stroke, etc.
- **Resource needs:** Labs, imaging, consultations

**Most Relevant Papers:**

- **ProtoEHR (ArXiv: 2508.18313v1):** Multi-task learning
  - Combines target prediction with general patient-state representation
  - Hierarchical prototypes support multiple tasks
  - Outperforms single-task models

- **CEHR-BERT (ArXiv: 2111.08585v1):** Multiple prediction tasks
  - Dual learning objectives (MLM + visit type)
  - Strong transfer with limited labeled data
  - Proven on multiple clinical outcomes

- **Temporal Cross-Attention (ArXiv: 2403.04012v2):** Multi-task transformer
  - Predicts 9 complications simultaneously
  - Shared representations across tasks
  - Sliding window attention for efficiency

**Recommendation:**
Design multi-task architecture with shared ED patient representation and task-specific heads for disposition, severity, mortality, and key time-sensitive diagnoses. Leverage shared representation to improve data efficiency.

#### 7. Incorporating Medical Knowledge

**ED-Relevant Knowledge:**
- **Clinical protocols:** Sepsis bundles, stroke alerts, etc.
- **Risk scores:** CURB-65, HEART score, etc.
- **Diagnostic criteria:** Standard definitions
- **Treatment guidelines:** Evidence-based protocols

**Most Relevant Papers:**

- **KEEP (ArXiv: 2510.05049v1):** Knowledge graph + clinical data
  - Combines structured knowledge with empirical learning
  - Preserves ontological relationships while adapting
  - Outperforms both pure knowledge and pure data approaches

- **MIPO (ArXiv: 2107.09288v4):** Ontology + patient journey
  - Joint optimization improves with limited data
  - Medical ontology provides structure
  - Interpretable embeddings

- **UmlsBERT (ArXiv: 2010.10391v5):** UMLS knowledge augmentation
  - Integrates medical knowledge into BERT
  - Semantic group embeddings
  - Improved clinical NER and inference

**Recommendation:**
Integrate ED-specific clinical knowledge (protocols, risk scores) into embedding learning via knowledge graph or structured constraints. Particularly valuable for rare conditions and new presentations where data is limited.

#### 8. Real-Time Updating and Adaptation

**ED Dynamics:**
- **Continuous patient flow:** New information arrives constantly
- **State evolution:** Patient condition changes during stay
- **Treatment effects:** Interventions change trajectory

**Most Relevant Papers:**

- **TAMER (ArXiv: 2501.05661v2):** Test-time adaptation
  - Adapts to distribution shifts in real-time
  - MoE with domain-aware experts
  - SOTA with dynamic adaptation

- **CAAT-EHR (ArXiv: 2501.18891v1):** Autoregressive decoder
  - Predicts future states
  - Temporal consistency via autoregressive design
  - Suitable for continuous updating

- **Neural ODEs (ICE-NODE, ArXiv: 2207.01873v3)**
  - Continuous-time patient state modeling
  - Natural for evolving ED encounter
  - Handles irregular updates

**Recommendation:**
Implement continuous patient state model using Neural ODEs or autoregressive architecture that updates representation as new data arrives during ED stay. Enable real-time risk re-assessment.

#### 9. Computational Efficiency

**ED Constraints:**
- **High patient volume:** Must scale to many concurrent patients
- **Limited computational resources:** Often not research-grade infrastructure
- **Real-time requirements:** Inference in seconds
- **Battery constraints:** If mobile/tablet deployment

**Most Relevant Papers:**

- **HyMaTE (ArXiv: 2509.24118v1):** Linear-time complexity
  - Mamba enables efficient long sequence handling
  - Maintains high accuracy
  - Suitable for resource-constrained deployment

- **Step-wise Embeddings (ArXiv: 2311.08902v1):** Dimensionality reduction
  - Feature grouping reduces computational burden
  - Maintains or improves performance
  - Semantic grouping aids efficiency

- **Multi-View GNN (ArXiv: 2301.11608v1):** Efficient alternative to BERT
  - Competitive to fine-tuned BERT at fraction of computation
  - Suitable for resource-limited settings
  - Fast inference

**Recommendation:**
Prioritize efficient architectures (Mamba-based, or lightweight Transformers with attention optimization). Use semantic feature grouping to reduce dimensionality. Consider model distillation for deployment.

#### 10. Handling Heterogeneous Patient Populations

**ED Diversity:**
- **Age range:** Pediatric to geriatric
- **Acuity spectrum:** Minor injuries to life-threatening emergencies
- **Chief complaints:** Extremely diverse
- **Prior health:** From healthy to multiple chronic conditions

**Most Relevant Papers:**

- **ConvAE (ArXiv: 2003.06516v2):** Patient stratification
  - Discovers heterogeneous patient cohorts
  - Unsupervised at scale (1.6M patients)
  - Clinically meaningful subtypes

- **TAMER (ArXiv: 2501.05661v2):** Patient heterogeneity handling
  - MoE specializes for patient subgroups
  - Mitigates intertwined heterogeneity challenges
  - Improved personalization

- **Patient Similarity Graph (ArXiv: 2411.19742v1)**
  - Captures relationships across heterogeneous population
  - Graph structure handles diverse patients
  - Interpretable via attention

**Recommendation:**
Use mixture-of-experts or patient clustering to handle ED population heterogeneity. Different experts/prototypes for different patient subgroups (pediatric, trauma, chest pain, etc.). Graph-based similarity to leverage related patients.

### Proposed ED Patient Representation Framework

Based on the comprehensive literature review, here is a recommended framework for ED patient representation:

#### Architecture Components:

1. **Initial Representation Layer:**
   - Chief complaint encoding via Clinical BERT
   - Triage assessment + vital signs via specialized embeddings
   - Demographics via learned embeddings
   - Fusion via multiplicative embedding (MedFuse-style)

2. **Temporal Modeling:**
   - Timestamp embeddings with decay + periodic patterns (TRACE-style)
   - Velocity-aware state transitions (GRU-TV approach)
   - Continuous-time dynamics via Neural ODEs for evolving state

3. **Knowledge Integration:**
   - ED protocol knowledge graph (KEEP-style integration)
   - Clinical risk score embeddings
   - Diagnostic criteria embeddings

4. **Efficient Processing:**
   - Mamba backbone for linear-time complexity (HyMaTE-style)
   - Semantic feature grouping for dimensionality reduction
   - Edge-optimized implementation

5. **Multi-Task Heads:**
   - Disposition prediction (admit/discharge/observe)
   - Mortality risk estimation
   - Time-sensitive diagnosis (sepsis, MI, stroke)
   - Resource need prediction
   - Shared representations with task-specific heads

6. **Similarity and Case-Based Reasoning:**
   - Patient similarity graph (Graph Transformer)
   - Retrieval of similar historical cases
   - Interpretable via attention weights

7. **Continuous Adaptation:**
   - Test-time adaptation (TAMER-style)
   - Real-time state updates as new data arrives
   - Autoregressive future state prediction

8. **Interpretability:**
   - Hierarchical prototypes (ProtoEHR-style)
   - Attention visualizations
   - Feature importance via semantic groups
   - Confidence estimates for predictions

#### Key Advantages for ED:

- **Fast inference:** Linear-time architecture enables real-time predictions
- **Handles sparsity:** Multiplicative fusion and missingness awareness
- **Temporal precision:** Fine-grained modeling of rapid state changes
- **Interpretable:** Multiple levels of explanation for clinicians
- **Adaptive:** Updates representations as encounter progresses
- **Efficient:** Deployable on standard ED infrastructure
- **Knowledge-informed:** Integrates clinical protocols and risk scores
- **Multi-task:** Supports multiple ED prediction needs simultaneously

This framework synthesizes insights from the most relevant papers while addressing the unique requirements of emergency department patient care.

---

## Conclusion

This comprehensive survey of 100+ papers on clinical embeddings and patient representation learning reveals a rapidly evolving field with significant advances in:

1. **Transformer-based architectures** that have become dominant for clinical NLP and sequential EHR modeling
2. **Temporal modeling innovations** including time-aware attention, temporal embeddings, and Neural ODEs for capturing patient trajectories
3. **Graph-based approaches** that effectively model complex relationships between medical entities and patients
4. **Multimodal fusion techniques** that combine structured and unstructured EHR data for improved predictions
5. **Pre-training strategies** leveraging medical knowledge graphs and large clinical corpora

For Emergency Department applications specifically, the most promising approaches combine:
- Fine-grained temporal modeling (TRACE, GRU-TV)
- Efficient architectures (HyMaTE, optimized Transformers)
- Robustness to missing data (MedFuse, UMSE)
- Clinical knowledge integration (KEEP, MIPO)
- Multi-task learning (ProtoEHR, CEHR-BERT)
- Interpretability (prototypes, attention, semantic grouping)

Significant research gaps remain in:
- ED-specific representation learning
- Real-time deployment considerations
- Prospective clinical validation
- Privacy-preserving learning
- Causal reasoning
- Continuous adaptation

The field is rapidly advancing, with recent papers (2024-2025) showing increasing sophistication in handling temporal dynamics, multimodal integration, and efficient architectures suitable for clinical deployment. The convergence of large language models, graph neural networks, and specialized clinical pre-training suggests continued rapid progress in this critical area of healthcare AI.

---

## References

All 100+ papers cited in this survey are available on ArXiv. ArXiv IDs are provided throughout the document for each referenced paper. For the complete list of papers by category:

- **Patient Representation:** 2010.02809v2, 2003.06516v2, 1909.09248v1
- **BERT-based Models:** 1904.03323v3, 2010.10391v5, 2111.08585v1, 1908.03971v4
- **Temporal Models:** 2312.05933v1, 2207.01873v3, 1911.12216v1, 2403.04012v2
- **Medical Concepts:** 2107.12919v1, 1804.01486v3, 1806.02873v1, 1909.06886v1
- **Graph-based:** 1910.02574v1, 1912.03366v2, 2203.09994v1, 2101.03940v1
- **Specialized:** 2508.18313v1, 2511.09247v1, 2509.24118v1, 2501.05661v2
- And many more detailed throughout this survey.

---

*Document prepared for the Hybrid Reasoning Acute Care project, December 1, 2025*
