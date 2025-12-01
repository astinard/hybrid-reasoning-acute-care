# Hybrid Reasoning for Acute Care: Temporal Knowledge Graphs and Clinical Constraints

## A Comprehensive Literature Review

---

## Table of Contents

1. [Introduction and Background](#1-introduction-and-background)
2. [Temporal Knowledge Graphs as World Models in Healthcare](#2-temporal-knowledge-graphs-as-world-models-in-healthcare)
3. [Neuro-Symbolic Constraints and Clinical Rules as Priors](#3-neuro-symbolic-constraints-and-clinical-rules-as-priors)
4. [Generative Modeling of Clinical Trajectories](#4-generative-modeling-of-clinical-trajectories)
5. [Multimodal Data Fusion with Knowledge Graph Scaffold](#5-multimodal-data-fusion-with-knowledge-graph-scaffold)
6. [Applications: ED Auto-Coding, Documentation, Decision Support](#6-applications-ed-auto-coding-documentation-decision-support)
7. [Privacy, Safety, and Deployment](#7-privacy-safety-and-deployment)
8. [Research Gaps and Future Directions](#8-research-gaps-and-future-directions)
9. [References](#references)

---

## 1. Introduction and Background

### 1.1 The Crisis in Emergency Department Care

Emergency departments (EDs) worldwide face unprecedented challenges characterized by increasing patient volumes, resource constraints, and growing complexity of care delivery. The demand for ED services continues to escalate globally, particularly evident during the COVID-19 pandemic, which placed extraordinary strain on healthcare systems 【Xie2021†L1-L15】. This growth has precipitated ED crowding and delays in care delivery, resulting in measurably increased morbidity and mortality 【Xie2021†L16-L25】. The ED triage process—centered on risk stratification—represents a complex clinical judgment integrating factors such as the patient's likely acute course, availability of medical resources, and local practices 【Xie2021†L26-L35】.

The widespread adoption of Electronic Health Records (EHRs) has fundamentally transformed healthcare data availability. According to the Office of the National Coordinator for Health Information Technology, nearly 84% of hospitals have adopted at least a basic EHR system, representing a 9-fold increase since 2008 【Shickel2017†L1-L25】. EHR systems store comprehensive data associated with each patient encounter, including demographic information, diagnoses, laboratory tests and results, prescriptions, radiological images, clinical notes, and more 【Shickel2017†L26-L45】. While primarily designed for improving healthcare efficiency from an operational standpoint, researchers have found substantial secondary use for clinical informatics applications.

### 1.2 The Promise and Limitations of AI in Acute Care

Machine learning has demonstrated tremendous potential in ED triage prediction models, with various approaches including traditional machine learning, deep learning, and interpretable methods being explored 【Xie2021†L45-L60】. However, researchers often develop ad-hoc models for single clinical prediction tasks using isolated datasets, creating a lack of comparative studies among different methods and models to predict the same ED outcomes 【Xie2021†L61-L75】. This undermines the generalizability of any single model and highlights the need for standardized benchmarks and more sophisticated approaches.

The application of deep learning to EHR data has yielded notable successes across several clinical informatics tasks including information extraction, representation learning, outcome prediction, phenotyping, and de-identification 【Shickel2017†L46-L70】. Traditional machine learning and statistical techniques such as logistic regression, support vector machines, and random forests dominated the field until recently, but deep learning techniques have achieved greater success through hierarchical feature construction and effective capture of long-range dependencies in data 【Shickel2017†L71-L95】.

### 1.3 The Need for Hybrid Reasoning Approaches

Despite these advances, significant challenges persist. Deep learning models, while powerful, often lack interpretability—a crucial requirement in clinical settings where understanding the rationale behind a diagnosis is essential 【Lu2024†L1-L30】. The "black-box" nature of neural networks has raised concerns about the transparency of decision-making processes. Traditional machine learning models require substantial feature engineering, and while some models like decision trees are interpretable, others such as SVMs or ensemble methods can become opaque as model complexity increases 【Lu2024†L31-L60】.

Neuro-symbolic AI represents an emerging paradigm that integrates the interpretability and structured reasoning of symbolic methods with the powerful learning capabilities of neural networks 【Lu2024†L61-L90】. Symbolic reasoning enables incorporation of domain knowledge and logical rules, providing transparency that pure neural networks often lack. Meanwhile, neural networks contribute the ability to learn from data and handle complex patterns, making neuro-symbolic AI a compelling candidate for tasks like diagnosis prediction where explainability is critical.

### 1.4 Scope and Organization of This Review

This literature review examines the convergence of temporal knowledge graphs, clinical constraint reasoning, and generative modeling for acute care applications. We survey recent advances (2018-2025) across seven interconnected domains: (1) temporal knowledge graph representations for healthcare, (2) neuro-symbolic approaches integrating clinical rules, (3) generative modeling of patient trajectories, (4) multimodal data fusion architectures, (5) practical applications in ED coding and decision support, (6) privacy-preserving deployment strategies, and (7) critical research gaps requiring future investigation.

---

## 2. Temporal Knowledge Graphs as World Models in Healthcare

### 2.1 Foundations of Healthcare Knowledge Graphs

Knowledge graphs have emerged as powerful representations for modeling complex healthcare data, capturing entities (patients, conditions, medications, procedures) and their relationships in structured, queryable formats. The fundamental premise is that representing patient data as interconnected graphs rather than isolated tabular features enables more sophisticated reasoning about clinical trajectories and outcomes.

Recent work by Jhee et al. demonstrates that graph-based representations of clinical data can significantly outperform traditional tabular approaches for predictive modeling 【Jhee2025†L1-L35】. Their study on patients with intracranial aneurysm compared various classification approaches on tabular data versus graph-based representations of the same data, finding that Graph Convolutional Network (GCN) embeddings achieved the best performance for predicting clinical outcomes from observational data. The study achieved an AUC of 0.91 with graph representations compared to 0.71 with traditional tabular approaches 【Jhee2025†L36-L70】.

The importance of schema design in knowledge graph construction cannot be overstated. Jhee et al. emphasize that the adopted schema for representing individual data and temporal data significantly impacts predictive performance 【Jhee2025†L71-L100】. Their work utilized two complementary ontological frameworks: the Swiss Personalized Health Network (SPHN) ontology for data harmonization across institutions, and the CARE-SM (Common Data Model for RarE Disease Registration) specification built upon SPHN for representing clinical information 【Jhee2025†L101-L135】.

### 2.2 Temporal Representation Learning

A critical challenge in healthcare knowledge graphs is the representation of temporal dynamics—how patient states evolve over time and how interventions affect trajectories. The temporal aspect of EHR data presents unique challenges including irregular sampling intervals, variable-specific recording frequencies, and timestamp duplication when multiple measurements are recorded simultaneously 【Ma2024†L1-L35】.

Ma et al. introduced a dynamic embedding and tokenization framework for precise representation of multimodal clinical time series that combines novel methods for encoding time and sequential position with temporal cross-attention 【Ma2024†L36-L70】. Their approach addresses the high dimensionality, sparsity, and multimodality challenges inherent in EHR data while accounting for irregular and variable-specific recording frequencies. When integrated into a multitask transformer classifier with sliding window attention, their embedding and tokenization framework outperformed baseline approaches on predicting nine postoperative complications across data from three hospitals and two academic health centers 【Ma2024†L71-L110】.

For temporal knowledge graph construction, multiple encoding strategies have been explored. Jhee et al. investigated three temporal encoding approaches: (1) Timestamp Encoding using positional encoding, (2) Time2Vec learning periodic and linear time representations, and (3) Interval Encoding capturing time intervals between events 【Jhee2025†L136-L175】. Their results moderated the relative impact of various time encodings on GCN performance, suggesting that schema design may be more consequential than specific temporal encoding choices.

### 2.3 Patient Journey Knowledge Graphs

The concept of Patient Journey Knowledge Graphs (PJKGs) represents a novel approach to addressing fragmented healthcare data by integrating diverse patient information into unified, structured representations. Al Khatib et al. present a methodology for constructing PJKGs using Large Language Models to process and structure both formal clinical documentation and unstructured patient-provider conversations 【AlKhatib2025†L1-L40】.

These graphs encapsulate temporal and causal relationships among clinical encounters, diagnoses, treatments, and outcomes, enabling advanced temporal reasoning and personalized care insights 【AlKhatib2025†L41-L80】. The research evaluated four different LLMs—Claude 3.5, Mistral, Llama 3.1, and GPT-4o—in their ability to generate accurate and computationally efficient knowledge graphs. Results demonstrated that while all models achieved perfect structural compliance, they exhibited variations in medical entity processing and computational efficiency 【AlKhatib2025†L81-L120】.

The PJKG framework defines several key entity types critical for representing patient journeys: Patient Demographics, Medical History, Symptoms, Diagnoses, Medications, Procedures, Lab Tests, Vital Signs, and Follow-up Plans 【AlKhatib2025†L121-L160】. Relationships between entities capture clinical semantics including HAS_SYMPTOM, DIAGNOSED_WITH, PRESCRIBED, UNDERWENT_PROCEDURE, HAD_TEST, MEASURED_VITAL, and REQUIRES_FOLLOWUP. This ontological structure enables sophisticated queries about patient trajectories and supports clinical decision-making.

### 2.4 Graph Neural Network Architectures for Healthcare

The application of Graph Neural Networks (GNNs) to healthcare knowledge graphs has demonstrated significant promise. Jhee et al. employed Relational Graph Convolutional Networks (RGCNs) that extend standard GCNs to handle multiple edge types—essential for healthcare applications where relationships between entities are semantically diverse 【Jhee2025†L176-L210】.

The HealthGAT framework introduces a hierarchical approach to generating embeddings from EHR data, iteratively refining embeddings for medical codes through graph attention mechanisms 【HealthGAT2024†L1-L40】. The model incorporates customized EHR-centric auxiliary pre-training tasks to leverage rich medical knowledge embedded within the data, providing comprehensive analysis of complex medical relationships that offers significant advancement over standard data representation techniques.

GraphCare represents another important advance, using external knowledge graphs to build patient-specific knowledge graphs that are then used to train a Bi-attention Augmented (BAT) graph neural network for healthcare predictions 【GraphCare2023†L1-L45】. The framework extracts knowledge from large language models and external biomedical knowledge graphs, demonstrating substantial improvements on MIMIC-III and MIMIC-IV across mortality prediction, readmission prediction, length of stay, and drug recommendation tasks—with particularly strong performance in limited data scenarios.

### 2.5 Ontological Foundations and Standardization

Successful healthcare knowledge graphs require robust ontological foundations. Several classification schema and controlled vocabularies exist for recording medical information, including ICD (International Classification of Diseases) for diagnosis codes, CPT (Current Procedural Terminology) for procedure codes, LOINC (Logical Observation Identifiers Names and Codes) for laboratory observations, and RxNorm for medication codes 【Shickel2017†L96-L130】. These codes vary between institutions, with partial mappings maintained by resources such as UMLS (Unified Medical Language System) and SNOMED CT.

The challenge of harmonizing data across terminologies and between institutions remains an active area of research. Several deep EHR systems propose forms of clinical code representation that facilitate cross-institution analysis and applications 【Shickel2017†L131-L165】. Knowledge graph approaches offer a natural framework for integrating multiple ontologies and enabling semantic interoperability across healthcare systems.

---

## 3. Neuro-Symbolic Constraints and Clinical Rules as Priors

### 3.1 The Case for Neuro-Symbolic Integration in Healthcare

Healthcare AI faces a fundamental tension between the pattern recognition capabilities of neural networks and the need for transparent, trustworthy decision-making. Neuro-symbolic AI addresses this by integrating symbolic reasoning with neural learning, offering a promising balance between accuracy and explainability 【Lu2024†L91-L125】.

Traditional symbolic methods in healthcare relied on predefined rules and logic to draw conclusions from data. These methods are highly interpretable as their reasoning process is transparent, but their performance is limited by the rigidity of human-defined rules which struggle to capture the complexity and variability of real-world medical data 【Lu2024†L126-L160】. Conversely, deep learning models often surpass traditional models in accuracy but are criticized for their "black-box" nature, raising concerns about transparency in clinical settings.

The neuro-symbolic paradigm enables incorporation of domain-specific knowledge through logical rules with learnable weights and thresholds. Lu et al. demonstrate that Logical Neural Network (LNN)-based models, particularly their multi-pathway and comprehensive configurations, achieve superior performance over traditional models including Logistic Regression, SVM, and Random Forest, reaching accuracy up to 80.52% and AUROC scores up to 0.8457 in diabetes prediction 【Lu2024†L161-L200】.

### 3.2 Logical Neural Networks for Clinical Reasoning

Logical Neural Networks represent a neuro-symbolic framework designed to learn interpretable models by expressing rules in first-order logic (FOL), a powerful language with clear semantics and rich operators 【Lu2024†L201-L240】. Unlike traditional neuro-symbolic methods that rely on non-learnable t-norms, LNNs incorporate learnable parameters that allow the network to adjust logical rules based on data.

The LNN conjunction operator extends standard boolean logic with learnable weights:

```
LNN-∧(x, y) = max(0, min(1, β - w₁(1-x) - w₂(1-y)))
```

where β, w₁, w₂ are learnable parameters and x, y ∈ [0, 1] are inputs 【Lu2024†L241-L280】. This formulation ensures the output behaves similarly to classical logic when inputs are near 0 or 1, while allowing smooth transitions for intermediate values. Constraints maintain the crispness of FOL semantics while enabling gradient-based optimization.

For diagnosis prediction, LNNs learn thresholds using smooth transition logic functions that enable data-driven flexibility while maintaining interpretability. For example, a hypertension rule becomes:

```
p(diagnosis=hypertension | patient) = LNN-∧(TL(f_blood_pressure, θ₁), TL(f_age, θ₂))
```

The learned weights and thresholds provide direct insight into feature contributions, aligning with clinical knowledge—for instance, learned thresholds for glucose and BMI are consistent with established risk factors for diabetes 【Lu2024†L281-L320】.

### 3.3 Knowledge Graph-Enhanced Reasoning

The integration of knowledge graphs with neural reasoning creates powerful hybrid systems. The medIKAL framework (Integrating Knowledge Graphs as Assistants of LLMs) combines Large Language Models with knowledge graphs to enhance diagnostic capabilities 【medIKAL2024†L1-L45】. The system assigns weighted importance to entities in medical records based on their type, enabling precise localization of candidate diseases within knowledge graphs. It employs a residual network-like approach allowing initial LLM diagnosis to merge with KG search results, with a path-based reranking algorithm refining the diagnostic process.

For clinical decision support, the MSATT-KG model combines Multi-Scale Feature Attention with Structured Knowledge Graph Propagation 【Xie2019†L1-L40】. The approach uses graph convolutional neural networks to capture hierarchical relationships among medical codes, achieving state-of-the-art ICD coding results by leveraging the ontological structure of medical classification systems as a form of structured prior knowledge.

### 3.4 Clinical Constraint Integration

Clinical guidelines, protocols, and domain expertise can be formulated as constraints that guide neural network predictions. The KAT-GNN (Knowledge-Augmented Temporal Graph Neural Network) framework integrates clinical knowledge and temporal dynamics for risk prediction 【KATGNN2024†L1-L45】. The approach constructs modality-specific patient graphs from EHRs, then augments them using ontology-driven edges derived from SNOMED CT and co-occurrence priors extracted from EHRs. A time-aware transformer captures longitudinal dynamics from graph-encoded representations, achieving state-of-the-art performance in coronary artery disease prediction (AUROC: 0.9269) and strong results in mortality prediction on MIMIC-III and MIMIC-IV.

The hierarchical nature of medical coding systems (ICD, SNOMED-CT) provides natural constraint structures. Vu et al. propose a hierarchical joint learning mechanism that uses the hierarchical relationships among ICD codes to improve prediction of infrequent codes 【Vu2020†L1-L45】. Their approach, JointLAAT, first predicts normalized codes (first three characters) then uses this prediction to guide full code prediction, demonstrating significant improvements in macro-averaged metrics that emphasize rare-label performance.

### 3.5 Explainability Through Symbolic Structure

The integration of symbolic reasoning provides multiple avenues for explainability. Maximum activation analysis examines which inputs result in maximum activation of hidden units, assigning importance to raw features 【Shickel2017†L166-L200】. Constraint-based approaches enforce non-negativity, sparsity, or ontological smoothness in learned representations, making the resulting models more interpretable.

The concept of "mimic learning" offers another path to interpretability—training interpretable models (like gradient boosting trees) on the probability outputs of complex neural networks, thereby transferring predictive power while maintaining feature transparency 【Shickel2017†L201-L235】. This approach achieves similar or better performance than both baseline linear and deep models for phenotyping and mortality prediction while retaining desired feature transparency.

---

## 4. Generative Modeling of Clinical Trajectories

### 4.1 Diffusion Models for Electronic Health Records

Denoising Diffusion Probabilistic Models have emerged as a powerful class of generative models, achieving state-of-the-art performance across image and speech synthesis domains. He et al. introduce MedDiff, the first successful application of diffusion models to electronic health record generation 【He2023†L1-L45】.

The diffusion framework operates through a forward process that gradually adds noise to original data until it becomes entirely noisy, followed by a reverse process that learns to denoise and generate new samples 【He2023†L46-L90】. MedDiff employs a modified 1D U-Net architecture with larger model depth/width, positional embeddings, and residual blocks to capture neighboring feature correlations in patient records. The model proposes a mechanism for class-conditional sampling to preserve label information—ensuring generated records match the distribution of specified clinical labels 【He2023†L91-L135】.

A key innovation in MedDiff is the use of Anderson Acceleration to speed up the generation process, addressing a major limitation of diffusion models. The acceleration technique leverages fixed-point iteration theory to predict subsequent denoised samples, achieving 2× speedup while maintaining sample quality 【He2023†L136-L180】. Experimental results on MIMIC-III and the Patient Treatment Classification dataset demonstrate that MedDiff outperforms existing state-of-the-art methods including MedGAN and CorGAN in dimension-wise probability matching (correlation ρ = 0.98, lowest SAE = 4.16).

### 4.2 GANs and Autoencoders for EHR Synthesis

Prior to diffusion models, Generative Adversarial Networks dominated synthetic EHR generation. MedGAN combines autoencoders with GANs to generate high-dimensional discrete patient records, proposing minibatch averaging to avoid mode collapse and batch normalization to increase learning efficiency 【Choi2017†L1-L40】. The framework generates synthetic records that achieve comparable performance to real data on distribution statistics, predictive modeling tasks, and medical expert review.

CorGAN (Correlation-capturing Convolutional GAN) extends MedGAN by using convolutional architectures to better capture correlations between medical features 【Torfi2020†L1-L35】. However, both approaches rely heavily on pre-trained autoencoders to reduce latent variable dimensionality, creating challenges when generalizing to different institutions with varying patient distributions.

Autoencoders have been extensively used for patient representation learning. The DeepPatient framework employs stacked denoising autoencoders to transform raw patient vectors into compressed representations, achieving better generalized disease prediction performance compared to raw patient features 【Miotto2016†L1-L45】. The three-layer autoencoder network produces patient representations useful for predicting a wide variety of ICD-9-based diagnoses.

### 4.3 Counterfactual Reasoning in Healthcare

Counterfactual analysis—asking "what would have happened under different conditions?"—provides powerful tools for causal inference in healthcare. While less explored than predictive modeling, counterfactual reasoning offers insights into treatment effects and enables "what-if" simulations of clinical trajectories.

Generative models can enable counterfactual trajectory simulation by learning the distribution of patient outcomes and generating alternative scenarios under different intervention conditions. This capability supports clinical decision-making by illustrating potential consequences of treatment choices. However, substantial methodological challenges remain in ensuring that counterfactual generations respect causal constraints and medical plausibility.

### 4.4 Trajectory Prediction and Temporal Modeling

Recurrent neural networks, particularly LSTM and GRU variants, have been extensively applied to temporal prediction in healthcare. The Doctor AI framework uses sequences of (event, time) pairs across patient admissions as input to GRU networks, predicting future diagnoses and medication interventions 【Choi2016†L1-L45】. The system performs differential diagnosis with accuracy comparable to physicians, achieving up to 79% recall@30.

The DeepCare framework generates separate vectors for diagnosis and intervention codes per admission, using modified LSTM cells that model time, admission methods, and complete illness history 【Pham2016†L1-L40】. The concatenated vectors enable prediction of next diagnosis, next intervention, and future readmission for conditions including diabetes and mental health.

Temporal Cross-Attention mechanisms offer sophisticated approaches to multimodal EHR time series. Ma et al. introduce methods that combine novel time and sequential position encoding with temporal cross-attention, precisely representing multimodal clinical time series while handling irregular sampling 【Ma2024†L111-L150】. Their approach outperforms baselines on predicting postoperative complications using multimodal data from multiple institutions.

### 4.5 Clinical Plausibility Constraints

A critical challenge in generative healthcare models is ensuring clinical plausibility of generated trajectories. Unlike image generation where visual inspection can assess quality, synthetic patient records require domain expertise to evaluate medical validity. MedDiff addresses this partially through class-conditional sampling that preserves label distributions, but broader constraints capturing medical knowledge remain an open challenge 【He2023†L181-L220】.

The integration of knowledge graphs with generative models offers promising directions. Graph structures can encode valid medical relationships, constraining generation to produce trajectories that respect clinical semantics. Future work at the intersection of diffusion models and knowledge graphs may enable more medically plausible synthetic data generation.

---

## 5. Multimodal Data Fusion with Knowledge Graph Scaffold

### 5.1 The Multimodal Challenge in Healthcare

Electronic health records encompass heterogeneous information types including structured data in tabular form, unstructured data in textual notes, medical images, and physiological waveforms. Different modalities can complement each other and provide a more complete picture of patient health status, but fusion presents significant methodological challenges 【Cui2024†L1-L40】.

The complexity stems from several factors: complex medical coding systems, noise and redundancy in written notes, temporal misalignment across modalities, and varying granularity of information 【Cui2024†L41-L80】. While substantial research addresses representation learning of structured EHR data, multimodal fusion remains comparatively under-studied.

### 5.2 The MINGLE Framework

Cui et al. propose MINGLE (Multimodal INtegration via Graph Learning and LLM Enhancement), a framework that effectively integrates both structures and semantics in EHR 【Cui2024†L81-L120】. The framework employs a two-level infusion strategy combining medical concept semantics and clinical note semantics into hypergraph neural networks, which learn complex interactions between different data types to generate visit representations for downstream prediction.

The MINGLE architecture processes structured EHR data through hypergraph construction where nodes represent medical codes and hyperedges capture higher-order relationships (e.g., all codes from a single visit). Clinical notes are processed through LLM-based semantic extraction, with the resulting embeddings integrated into the hypergraph structure 【Cui2024†L121-L160】. Experiments on MIMIC-III and the private CRADLE dataset demonstrate 11.83% relative improvement in predictive performance through effective semantic integration and multimodal fusion.

Key innovations include: (1) hypergraph formulation capturing beyond-pairwise medical concept relationships, (2) LLM-based extraction of semantic information from clinical notes, and (3) attention mechanisms that learn optimal fusion strategies for different prediction tasks 【Cui2024†L161-L200】.

### 5.3 Temporal Multimodal Integration

Handling temporal dynamics across modalities presents unique challenges. Zhang et al. address irregular multimodal EHR modeling by dynamically incorporating hand-crafted imputation embeddings into learned interpolation embeddings via a gating mechanism for time series, while treating clinical note sequences as multivariate irregular time series addressed via time attention 【Zhang2022†L1-L45】.

Their interleaved attention mechanism integrates irregularity in multimodal fusion across temporal steps—to our knowledge, the first work to thoroughly model irregularity across multimodalities for medical predictions 【Zhang2022†L46-L90】. Results demonstrate relative improvements of 6.5%, 3.6%, and 4.3% in F1 for time series, clinical notes, and multimodal fusion respectively, demonstrating the importance of considering irregularity in multimodal EHRs.

### 5.4 Knowledge Graph as Fusion Scaffold

Knowledge graphs provide natural scaffolding for multimodal fusion by encoding semantic relationships that guide integration of diverse data types. The GraphCare framework demonstrates this approach, extracting knowledge from LLMs and external biomedical knowledge graphs to build patient-specific graphs 【GraphCare2023†L46-L90】.

The process involves: (1) extracting relevant entities from patient records across modalities, (2) linking entities to external knowledge bases (UMLS, drug databases), (3) constructing patient-specific subgraphs, and (4) using graph neural networks to generate unified patient representations. This approach particularly excels in limited data scenarios where knowledge graph priors compensate for sparse training examples.

### 5.5 Clinical Text Integration

Clinical notes contain rich information about patient conditions, treatment plans, and clinical reasoning that is difficult to capture in structured codes. Several approaches address text integration:

**Embedding-based fusion**: Clinical notes are embedded using language models (Word2Vec, BERT, clinical LLMs) and combined with structured embeddings through concatenation, attention, or gating mechanisms 【Shickel2017†L236-L270】.

**Knowledge extraction**: LLMs extract structured information from notes—entities, relationships, temporal expressions—that can be represented in knowledge graph format for integration with structured data 【AlKhatib2025†L161-L200】.

**Hierarchical attention**: Document-level and sentence-level attention mechanisms identify relevant text passages for specific prediction tasks, enabling selective information extraction 【Baumel2018†L1-L35】.

The ambient scribe application demonstrates practical integration of transcription and clinical note generation, using Whisper for transcription and GPT-4o for SOAP note generation, achieving quality that exceeds expert-written notes as evaluated by LLM-as-judge 【Morse2024†L1-L45】.

---

## 6. Applications: ED Auto-Coding, Documentation, Decision Support

### 6.1 Automated Medical Coding

Medical coding—assigning diagnosis and procedure codes to clinical documentation—represents a costly, labor-intensive process prone to human error. Automated medical coding has attracted substantial research attention, with the task typically formulated as multi-label classification 【Vu2020†L46-L90】.

The scale of the challenge is substantial: ICD-9-CM contains approximately 17,000 codes while ICD-10-CM/PCS contains approximately 140,000 codes 【Edin2023†L1-L45】. The problem is further complicated by highly long-tailed distributions where some codes are frequently used but the majority may have only a few instances due to disease rarity.

### 6.2 State-of-the-Art Coding Models

**CAML (Convolutional Attention for Multi-Label classification)** introduced label-wise attention mechanisms where each code learns specific attention weights over the document, achieving strong results on MIMIC benchmarks 【Mullenbach2018†L1-L40】.

**LAAT (Label Attention Model)** extends CAML by using bidirectional LSTM encoding and a modified attention mechanism that transforms hidden representations into label-specific vectors 【Vu2020†L91-L135】. The model achieves new state-of-the-art results on MIMIC-III with notable improvements: macro-AUC 91.9%, micro-F1 57.5%, demonstrating the value of capturing contextual information across input words.

**PLM-ICD** leverages pre-trained language models (BERT) for ICD coding, processing documents in chunks and applying label-wise attention 【Huang2022†L1-L40】. The approach achieves the highest performance on revised MIMIC benchmarks, though pre-training benefits in medical coding remain smaller than in other NLP domains.

**MultiResCNN** combines multi-filter convolutional layers with residual connections to capture variable-length text patterns, achieving competitive performance with faster training through the residual architecture 【Li2020†L1-L35】.

### 6.3 Critical Analysis of Coding Benchmarks

Edin et al. provide a critical review of automated medical coding research, revealing significant methodological issues 【Edin2023†L46-L90】. Their key findings include:

1. **Flawed evaluation metrics**: The macro F1 score was calculated sub-optimally in previous work, resulting in misleading underestimates. Their correction approximately doubles reported macro F1 scores on MIMIC-III.

2. **Dataset split problems**: The original MIMIC-III split results in 54% of codes being absent from the test set, biasing evaluation. They introduce MIMIC-III "clean" with stratified sampling ensuring full class representation.

3. **Hyperparameter sensitivity**: Models previously reported as low-performing improved considerably with proper hyperparameter tuning and decision boundary optimization, demonstrating the importance of fair experimental setup.

4. **Error analysis insights**: All models struggle severely with rare codes, with more than 50% of ICD-10 codes never predicted correctly by any model. Contrary to previous claims, document length has only negligible impact on performance 【Edin2023†L91-L135】.

### 6.4 Clinical Decision Support Systems

Decision support systems aim to identify high-risk patients and prioritize limited medical resources. The ED triage benchmark by Xie et al. establishes standardized evaluation across three outcomes: hospitalization, critical outcomes (ICU transfer or mortality), and 72-hour ED reattendance 【Xie2021†L76-L120】.

Performance comparison across methods reveals important patterns:
- Gradient boosting achieved AUC 0.881 for critical outcomes and 0.820 for hospitalization
- Deep learning (MLP, LSTM, Med2Vec) did not outperform simpler models
- Traditional scoring systems (NEWS, MEWS, REMS, CART) achieved substantially lower discrimination
- Interpretable machine learning (AutoScore) achieved AUC 0.846 for critical outcomes with only 7 variables

The finding that complex deep learning models do not necessarily improve over simpler approaches with relatively simple, low-dimensional ED data suggests that overly complex models may be unnecessary—and their interpretability disadvantages more problematic—in this setting 【Xie2021†L121-L165】.

### 6.5 Documentation Automation

AI-powered ambient scribes represent an emerging application reducing clinical documentation burden. The AMIE system (Articulate Medical Intelligence Explorer) demonstrates LLM-based diagnostic dialogue capabilities, outperforming primary care physicians on 28 of 32 evaluation axes in text-based consultations 【Tu2024†L1-L50】.

The guardrailed-AMIE (g-AMIE) framework introduces physician-centered oversight, performing history taking within guardrails while abstaining from individualized medical advice 【Vedadi2024†L1-L45】. The system conveys assessments to overseeing physicians through a clinician cockpit interface, decoupling oversight from intake to enable asynchronous review. Results demonstrate that g-AMIE outperforms nurse practitioners and physician assistants in intake quality, case summarization, and proposed diagnoses.

Clinical documentation quality improvements from ambient scribes include reduced cognitive load during visits (94% of surveyed clinicians) and decreased documentation burden (97%) 【Morse2024†L46-L90】. Post-processing notes with fine-tuned models further improves conciseness, demonstrating the potential for AI systems to ease administrative burdens while supporting high-quality care delivery.

---

## 7. Privacy, Safety, and Deployment

### 7.1 Privacy Challenges in Healthcare AI

Patient privacy concerns represent a fundamental barrier to healthcare AI research and deployment. High-quality, realistic synthetic EHRs can accelerate methodological developments while mitigating privacy concerns associated with data sharing 【He2023†L221-L260】. Traditional de-identification approaches through perturbation and randomization remain vulnerable to re-identification attacks, motivating the development of synthetic data approaches.

MedGAN pioneered privacy-preserving synthetic EHR generation, demonstrating limited privacy risk in both identity and attribute disclosure 【Choi2017†L41-L80】. The medGAN framework generates records that are statistically similar to real data while being "beyond de-identification"—synthetic rather than perturbed real records.

### 7.2 Knowledge Distillation for PHI-Safe Models

Knowledge distillation offers approaches for creating deployable models that don't require access to protected health information at inference time. The interpretable mimic learning framework trains interpretable models (gradient boosting trees) on probability outputs from complex neural networks 【Shickel2017†L271-L305】.

This approach enables:
1. Training complex models on full PHI data
2. Distilling predictions into interpretable, PHI-free models
3. Deploying lightweight models that maintain predictive power without requiring sensitive data

The Curiosity foundation model demonstrates scaling-law relationships for medical event data, pre-training on 118 million patients representing 115 billion discrete medical events 【Waxler2024†L1-L50】. Such large-scale pre-trained models can enable transfer learning where downstream tasks require less institution-specific PHI data.

### 7.3 Federated Learning for Healthcare

The GAME algorithm addresses multi-institutional EHR challenges through federated learning that preserves data privacy 【Zhou2025†L1-L45】. The approach integrates data at multiple levels: institutional level with knowledge graphs, between institutions using language models, and quantifying relationship strength using graph attention networks. Jointly trained embeddings are created using transfer and federated learning to preserve data privacy while enabling harmonized multi-institutional analysis.

### 7.4 Safety Considerations

Clinical deployment of AI systems requires careful attention to safety. Several considerations are paramount:

**Model interpretability**: The ability to understand and validate model decisions is essential for clinical trust. Neuro-symbolic approaches that provide explicit reasoning chains offer advantages over black-box models 【Lu2024†L321-L360】.

**Guardrails and oversight**: The g-AMIE framework demonstrates how AI systems can operate within defined guardrails while maintaining physician oversight and accountability 【Vedadi2024†L46-L90】.

**Error characterization**: Understanding failure modes is critical. Edin et al. demonstrate that models struggle systematically with rare codes and certain semantic categories (body mass index, location codes, similar concept disambiguation) 【Edin2023†L136-L180】.

**Calibration**: Model confidence should correlate with actual accuracy. Poorly calibrated models may provide overconfident predictions on cases where they are likely to fail.

### 7.5 Regulatory and Deployment Considerations

Real-world deployment of clinical AI requires navigating regulatory frameworks including FDA oversight of clinical decision support software. Key considerations include:

1. **Intended use specification**: Clearly defining what the system does and doesn't do
2. **Clinical validation**: Prospective validation in target clinical settings
3. **Human factors**: Ensuring appropriate integration into clinical workflows
4. **Monitoring**: Post-deployment surveillance for performance degradation or safety signals

The AI Consult tool demonstrates successful real-world deployment, with evaluation showing 16% fewer diagnostic errors and 13% fewer treatment errors compared to clinicians without AI support 【Korom2024†L1-L45】. The implementation required workflow-aligned design and active deployment strategies to encourage clinician uptake.

---

## 8. Research Gaps and Future Directions

### 8.1 Technical Challenges

**Unified Representation Learning**: Current approaches often process different data types (codes, notes, images, waveforms) in isolation. Truly unified patient representations that naturally integrate heterogeneous modalities remain elusive. As Shickel et al. note, "a truly unified patient representation appears to be one of the holy grails of clinical deep learning research" 【Shickel2017†L306-L340】.

**Temporal Reasoning**: While progress has been made on temporal encoding, sophisticated reasoning about causality, intervention effects, and counterfactual outcomes remains limited. Integrating temporal knowledge graphs with causal inference frameworks represents a promising direction.

**Rare Event Prediction**: Models consistently struggle with rare codes and infrequent events. The long-tailed distribution of medical phenomena means that the majority of codes may have insufficient training data for reliable prediction 【Edin2023†L181-L220】.

**Irregular Sampling**: EHR data exhibits highly irregular sampling with variable-specific recording frequencies. Current approaches often rely on imputation or binning strategies that may lose important temporal information.

### 8.2 Clinical Integration Challenges

**Workflow Integration**: Technical performance improvements don't guarantee clinical adoption. Understanding and designing for clinical workflows is essential for successful deployment 【Korom2024†L46-L90】.

**Trust and Explainability**: Clinicians need to understand and trust AI recommendations. While neuro-symbolic approaches offer improved interpretability, translating technical explanations into clinically meaningful insights remains challenging.

**Validation Paradigms**: Most research uses retrospective evaluation on benchmark datasets. Prospective validation in clinical settings with appropriate outcome measures is needed to establish real-world value.

**Heterogeneous Healthcare Settings**: Models trained on academic medical center data may not generalize to community hospitals, outpatient settings, or different geographic regions. The ED benchmark is based on single-center data, limiting generalizability claims 【Xie2021†L166-L200】.

### 8.3 Methodological Directions

**Neuro-Symbolic Scaling**: Current neuro-symbolic approaches operate at relatively small scale. Scaling logical reasoning to match the capabilities of large language models while maintaining interpretability is an open challenge.

**Generative Models for Clinical Trajectories**: Diffusion models show promise for EHR synthesis, but generating complete, clinically plausible patient trajectories with temporal dynamics and intervention effects remains early-stage.

**Hybrid Retrieval-Generation**: Combining knowledge retrieval from graphs with generative capabilities could enable systems that ground predictions in established medical knowledge while adapting to novel situations.

**Multi-Task Learning**: Healthcare naturally involves many related prediction tasks. Joint learning across tasks with appropriate knowledge sharing could improve data efficiency and performance, particularly for rare events.

### 8.4 Emerging Opportunities

**Foundation Models for Healthcare**: Large-scale pre-training on medical events demonstrates power-law scaling relationships, suggesting that larger models trained on more data will continue to improve 【Waxler2024†L51-L100】. The Curiosity models provide a framework for personalized prediction that generalizes across diverse clinical tasks.

**Real-Time Clinical Integration**: Moving from retrospective prediction to real-time decision support that integrates with EHR systems and clinical workflows represents a critical frontier.

**Patient-Centered AI**: Most current work focuses on clinician-facing tools. Patient-facing applications that support shared decision-making and health management represent an underexplored opportunity.

**Global Health Applications**: Demonstrating value in resource-limited settings could dramatically expand impact. The AI Consult study in Kenya demonstrates potential for LLM-based decision support to reduce clinical errors in primary care 【Korom2024†L91-L130】.

### 8.5 Synthesis: Toward Hybrid Reasoning Systems

The convergence of temporal knowledge graphs, neuro-symbolic reasoning, and generative modeling offers a promising path toward more capable, interpretable, and trustworthy healthcare AI. Key elements of this synthesis include:

1. **Knowledge graphs as semantic backbone**: Providing structured representation of medical knowledge that grounds predictions in established understanding

2. **Neuro-symbolic integration**: Combining the pattern recognition of neural networks with the transparency of symbolic reasoning

3. **Generative capabilities**: Enabling simulation of clinical trajectories, counterfactual reasoning, and synthetic data generation

4. **Multimodal fusion**: Integrating diverse data types through graph-based scaffolding

5. **Clinical constraints as priors**: Incorporating domain expertise and guidelines to ensure medically plausible outputs

Realizing this vision requires addressing the technical and clinical integration challenges outlined above, but the potential to transform acute care delivery justifies sustained research investment.

---

## References

### Section 1: Introduction and Background
- Shickel B, Tighe P, Bihorac A, Rashidi P. Deep EHR: A Survey of Recent Advances in Deep Learning Techniques for Electronic Health Record (EHR) Analysis. IEEE J Biomed Health Inform. 2017.
- Xie F, Zhou J, Lee JW, et al. Benchmarking Emergency Department Triage Prediction Models with Machine Learning and Large Public Electronic Health Records. Sci Data. 2021.
- Lu Q, Li R, Sagheb E, et al. Explainable Diagnosis Prediction through Neuro-Symbolic Integration. arXiv:2410.01855. 2024.

### Section 2: Temporal Knowledge Graphs
- Jhee JH, Megina A, Constant Dit Beaufils P, et al. Predicting clinical outcomes from patient care pathways represented with temporal knowledge graphs. arXiv:2502.21138. 2025.
- Al Khatib HS, Mittal S, Rahimi S, et al. From Patient Consultations to Graphs: Leveraging LLMs for Patient Journey Knowledge Graph Construction. arXiv:2503.16533. 2025.
- Ma Y, Kolla S, Kaliraman D, et al. Temporal Cross-Attention for Dynamic Embedding and Tokenization of Multimodal Electronic Health Records. arXiv:2403.04012. 2024.
- Piya FL, Gupta M, Beheshti R. HealthGAT: Node Classifications in Electronic Health Records using Graph Attention Networks. arXiv:2403.18128. 2024.
- Jiang P, Xiao C, Cross A, Sun J. GraphCare: Enhancing Healthcare Predictions with Personalized Knowledge Graphs. arXiv:2305.12788. 2023.

### Section 3: Neuro-Symbolic Constraints
- Lu Q, Li R, Sagheb E, et al. Explainable Diagnosis Prediction through Neuro-Symbolic Integration. arXiv:2410.01855. 2024.
- Jia M, Duan J, Song Y, Wang J. medIKAL: Integrating Knowledge Graphs as Assistants of LLMs for Enhanced Clinical Diagnosis on EMRs. arXiv:2406.14326. 2024.
- Xie X, Xiong Y, Yu PS, Zhu Y. EHR Coding with Multi-scale Feature Attention and Structured Knowledge Graph Propagation. CIKM 2019.
- Lin KW, Kuo YC, Wang HY, Tseng YJ. KAT-GNN: A Knowledge-Augmented Temporal Graph Neural Network for Risk Prediction in Electronic Health Records. arXiv:2511.01249. 2024.
- Hossain D, Chen JY. A Study on Neuro-Symbolic Artificial Intelligence: Healthcare Perspectives. arXiv:2503.18213. 2025.

### Section 4: Generative Modeling
- He H, Zhao S, Xi Y, Ho JC. MedDiff: Generating Electronic Health Records using Accelerated Denoising Diffusion Model. arXiv:2302.04355. 2023.
- Yuan H, Zhou S, Yu S. EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models. arXiv:2303.05656. 2023.
- Choi E, Biswal S, Malin B, et al. Generating Multi-label Discrete Patient Records using Generative Adversarial Networks. MLHC 2017.
- Choi E, Bahadori MT, Sun J. Doctor AI: Predicting Clinical Events via Recurrent Neural Networks. arXiv:1511.05942. 2016.
- Pham T, Tran T, Phung D, Venkatesh S. DeepCare: A Deep Dynamic Memory Model for Predictive Medicine. arXiv:1602.00357. 2016.

### Section 5: Multimodal Fusion
- Cui H, Fang X, Xu R, et al. Multimodal Fusion of EHR in Structures and Semantics: Integrating Clinical Records and Notes with Hypergraph and LLM. arXiv:2403.08818. 2024.
- Zhang X, Li S, Chen Z, et al. Improving Medical Predictions by Irregular Multimodal Electronic Health Records Modeling. arXiv:2210.12156. 2022.
- Feng Y, Chan TH, Yin G, Yu L. Democratizing Large Language Model-Based Graph Data Augmentation via Latent Knowledge Graphs. arXiv:2502.13555. 2025.

### Section 6: Applications
- Vu T, Nguyen DQ, Nguyen A. A Label Attention Model for ICD Coding from Clinical Text. IJCAI 2020.
- Edin J, Junge A, Havtorn JD, et al. Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study. SIGIR 2023.
- Mullenbach J, Wiegreffe S, Duke J, Sun J, Eisenstein J. Explainable Prediction of Medical Codes from Clinical Text. NAACL 2018.
- Li F, Yu H. ICD Coding from Clinical Text Using Multi-Filter Residual Convolutional Neural Network. AAAI 2020.
- Huang CW, Tsai SC, Chen YN. PLM-ICD: Automatic ICD Coding with Pretrained Language Models. ClinicalNLP 2022.
- Tu T, Palepu A, Schaekermann M, et al. Towards Conversational Diagnostic AI. arXiv:2401.05654. 2024.
- Vedadi E, Barrett D, Harris N, et al. Towards physician-centered oversight of conversational diagnostic AI. arXiv:2507.15743. 2024.
- Morse J, Gilbert K, Shin K, et al. A Custom-Built Ambient Scribe Reduces Cognitive Load and Documentation Burden for Telehealth Clinicians. arXiv:2507.17754. 2024.
- Korom R, Kiptinness S, Adan N, et al. AI-based Clinical Decision Support for Primary Care: A Real-World Study. arXiv:2507.16947. 2024.

### Section 7: Privacy and Safety
- Zhou D, Tong H, Wang L, et al. Representation Learning to Advance Multi-institutional Studies with Electronic Health Record Data. arXiv:2502.08547. 2025.
- Waxler S, Blazek P, White D, et al. Generative Medical Event Models Improve with Scale. arXiv:2508.12104. 2024.

### Additional References
- Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. Sci Data. 2016;3:160035.
- Johnson A, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data. 2023;10:1.
- Yuan Z, Tan C, Huang S. Code Synonyms Do Matter: Multiple Synonyms Matching Network for Automatic ICD Coding. ACL 2022.
- Kumthekar A, Tilley Z, Duong H, et al. Second Opinion Matters: Towards Adaptive Clinical AI via the Consensus of Expert Model Ensemble. arXiv:2505.23075. 2024.
- Vladika J, Domres A, Nguyen M, et al. Improving Reliability and Explainability of Medical Question Answering through Atomic Fact Checking in Retrieval-Augmented LLMs. arXiv:2505.24830. 2025.
- Yu Y, Hu X, Rajaganapathy S, et al. Launching Insights: A Pilot Study on Leveraging Real-World Observational Data from the Mayo Clinic Platform. arXiv:2504.16090. 2024.

---

*This literature review was compiled as part of the Hybrid Reasoning for Acute Care research project. Last updated: November 2025.*
