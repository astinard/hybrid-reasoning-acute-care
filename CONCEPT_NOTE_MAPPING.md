# Concept Note → Research Repository Mapping

**Purpose:** Map the CRCV Lab Concept Note structure to supporting research documents.

**Repository:** [github.com/astinard/hybrid-reasoning-acute-care](https://github.com/astinard/hybrid-reasoning-acute-care)
**Generated:** December 2025
**Documents Mapped:** 157 research documents across 16 categories

---

## How to Use This Document

This mapping follows the **exact hierarchy** of the Concept Note for Collaboration. When reviewing the concept note, reference this document to find supporting research literature for each section.

**Document paths updated:** All files now organized in topic-based subdirectories under `research/`.

---

## Table of Contents

1. [Executive Summary Support](#1-executive-summary-support)
2. [Thread 1: Neuro-Symbolic Constraints](#2-thread-1-neuro-symbolic-constraints)
3. [Thread 2: Temporal KGs as World Models](#3-thread-2-temporal-kgs-as-world-models)
4. [Thread 3: Diffusion Over Clinical Trajectories](#4-thread-3-diffusion-over-clinical-trajectories)
5. [Thread 4: Multimodal Fusion](#5-thread-4-multimodal-fusion)
6. [Technical Annex A: Temporal KG World Model](#6-technical-annex-a-temporal-kg-world-model)
7. [Technical Annex B: Diffusion Models](#7-technical-annex-b-diffusion-models)
8. [Technical Annex C: Multimodal Fusion Architecture](#8-technical-annex-c-multimodal-fusion-architecture)
9. [Technical Annex D: Clinical Tasks & Evaluation](#9-technical-annex-d-clinical-tasks--evaluation)
10. [Technical Annex E: Evaluation Framework](#10-technical-annex-e-evaluation-framework)
11. [Technical Annex F: Risks and Mitigations](#11-technical-annex-f-risks-and-mitigations)
12. [Foundations & Institutional Context](#12-foundations--institutional-context)

---

## 1. Executive Summary Support

| Claim | Supporting Document(s) | Location |
|-------|----------------------|----------|
| Temporal KGs as world models | `arxiv_temporal_kg_2024.md`, `allen_temporal_algebra.md` | [01_temporal_methods/](research/01_temporal_methods/), [_foundations/](research/_foundations/) |
| Neuro-symbolic constraints | `arxiv_neurosymbolic_clinical_reasoning.md`, `ibm_lnn_framework.md` | [03_hybrid_neurosymbolic/](research/03_hybrid_neurosymbolic/), [_foundations/](research/_foundations/) |
| Diffusion over trajectories | `arxiv_sequence_diffusion.md`, `arxiv_diffusion_healthcare.md` | [01_temporal_methods/](research/01_temporal_methods/), [09_learning_methods/](research/09_learning_methods/) |
| Multimodal fusion | `arxiv_multimodal_temporal_fusion.md`, `arxiv_multimodal_fusion.md` | [06_medical_imaging/](research/06_medical_imaging/) |
| Teacher-student distillation | `arxiv_teacher_student_clinical.md` | [09_learning_methods/](research/09_learning_methods/) |
| ED auto-coding evaluation | `arxiv_medical_coding_ai.md`, `arxiv_cdi_documentation_ai.md` | [05_clinical_nlp/](research/05_clinical_nlp/) |

---

## 2. Thread 1: Neuro-Symbolic Constraints

**Concept Note Section:** *Neuro-symbolic constraints for uncertain clinical reasoning*

### Core Documents

| Document | Key Finding | Location |
|----------|-------------|----------|
| `arxiv_neurosymbolic_clinical_reasoning.md` | 120+ papers; GraphCare +17.6% AUROC | [03_hybrid_neurosymbolic/](research/03_hybrid_neurosymbolic/) |
| `ibm_lnn_framework.md` | LNN: 80.52% accuracy, 0.8457 AUROC | [_foundations/](research/_foundations/) |
| `arxiv_hybrid_symbolic_neural.md` | Neural Theorem Provers +17% | [03_hybrid_neurosymbolic/](research/03_hybrid_neurosymbolic/) |
| `arxiv_constraint_satisfaction.md` | CP-SAT 18 seconds for OR scheduling | [03_hybrid_neurosymbolic/](research/03_hybrid_neurosymbolic/) |
| `arxiv_constraint_guided_generation.md` | 100% constraint satisfaction | [03_hybrid_neurosymbolic/](research/03_hybrid_neurosymbolic/) |

### Clinical Rule Encoding

| Document | Supports | Location |
|----------|----------|----------|
| `arxiv_guideline_encoding.md` | CQL, Arden Syntax, FHIR CQL | [03_hybrid_neurosymbolic/](research/03_hybrid_neurosymbolic/) |
| `fda_cds_guidance_current.md` | FDA 4-criteria CDS exemption | [_foundations/](research/_foundations/) |
| `arxiv_clinical_pathways.md` | Clinical protocol adherence | [03_hybrid_neurosymbolic/](research/03_hybrid_neurosymbolic/) |

### Time-Sensitive Protocols

| Protocol | Document | Key Metric | Location |
|----------|----------|------------|----------|
| Sepsis bundles | `arxiv_sepsis_prediction.md` | AUROC 0.88-0.97 | [04_clinical_prediction/](research/04_clinical_prediction/) |
| Stroke windows | `arxiv_stroke_prediction.md` | 85-93% LVO sensitivity | [04_clinical_prediction/](research/04_clinical_prediction/) |
| ACS rule-out | `arxiv_cardiac_arrest.md` | HEART score validation | [04_clinical_prediction/](research/04_clinical_prediction/) |

### Billing Policy (AMA E/M, NCCI)

| Document | Supports | Location |
|----------|----------|----------|
| `arxiv_cdi_documentation_ai.md` | E/M level prediction, CDI query generation | [05_clinical_nlp/](research/05_clinical_nlp/) |
| `arxiv_medical_coding_ai.md` | ICD-10 coding: 60.8% micro-F1 SOTA | [05_clinical_nlp/](research/05_clinical_nlp/) |

---

## 3. Thread 2: Temporal KGs as World Models

**Concept Note Section:** *Temporal Knowledge Graphs as world models*

### Core Documents

| Document | Key Finding | Location |
|----------|-------------|----------|
| `arxiv_temporal_kg_2024.md` | KAT-GNN AUROC 0.9269 | [01_temporal_methods/](research/01_temporal_methods/) |
| `allen_temporal_algebra.md` | 13 interval relations | [_foundations/](research/_foundations/) |
| `arxiv_temporal_reasoning.md` | LTL, CTL, Event Calculus | [01_temporal_methods/](research/01_temporal_methods/) |
| `arxiv_clinical_graph_construction.md` | 90-96% NER, 73-96% RE | [02_graph_neural_networks/](research/02_graph_neural_networks/) |
| `arxiv_temporal_gnn.md` | DyRep, TGN architectures | [01_temporal_methods/](research/01_temporal_methods/) |

### Graph Neural Networks

| Document | Application | Location |
|----------|------------|----------|
| `arxiv_gnn_clinical_2024.md` | AUROC 70-94% clinical prediction | [02_graph_neural_networks/](research/02_graph_neural_networks/) |
| `arxiv_kg_reasoning_clinical.md` | ComplEx MRR 0.50 | [02_graph_neural_networks/](research/02_graph_neural_networks/) |
| `arxiv_graph_embeddings_healthcare.md` | Snomed2Vec, Node2Vec, RotatE | [02_graph_neural_networks/](research/02_graph_neural_networks/) |

### Event Sequence Modeling

| Document | Key Finding | Location |
|----------|-------------|----------|
| `arxiv_event_sequence_clinical.md` | Hawkes processes, Neural TPPs | [01_temporal_methods/](research/01_temporal_methods/) |
| `arxiv_time_series_clinical.md` | AUC 0.85-0.93 | [01_temporal_methods/](research/01_temporal_methods/) |

---

## 4. Thread 3: Diffusion Over Clinical Trajectories

**Concept Note Section:** *Diffusion models over clinical trajectories*

### Core Documents

| Document | Key Finding | Location |
|----------|-------------|----------|
| `arxiv_sequence_diffusion.md` | SEDD 25-75% perplexity reduction | [01_temporal_methods/](research/01_temporal_methods/) |
| `arxiv_diffusion_healthcare.md` | CSDI 40-65% imputation improvement | [09_learning_methods/](research/09_learning_methods/) |
| `arxiv_graph_diffusion.md` | DiGress 3x validity improvement | [02_graph_neural_networks/](research/02_graph_neural_networks/) |
| `arxiv_counterfactual_clinical.md` | TraCE trajectory explanations | [09_learning_methods/](research/09_learning_methods/) |
| `arxiv_clinical_world_models.md` | ED as ideal testbed | [09_learning_methods/](research/09_learning_methods/) |

### Constraint-Guided Generation

| Document | Application | Location |
|----------|------------|----------|
| `arxiv_constraint_guided_generation.md` | 100% constraint satisfaction | [03_hybrid_neurosymbolic/](research/03_hybrid_neurosymbolic/) |
| `arxiv_clinical_rl_simulation.md` | Conservative Q-learning, medDreamer | [09_learning_methods/](research/09_learning_methods/) |

### Data Augmentation

| Document | Application | Location |
|----------|------------|----------|
| `arxiv_clinical_data_augmentation.md` | Rare disease handling | [12_data_quality/](research/12_data_quality/) |
| `arxiv_synthetic_clinical.md` | Privacy-preserving generation | [12_data_quality/](research/12_data_quality/) |

---

## 5. Thread 4: Multimodal Fusion

**Concept Note Section:** *Multimodal fusion over a KG scaffold*

### Core Documents

| Document | Key Finding | Location |
|----------|-------------|----------|
| `arxiv_multimodal_temporal_fusion.md` | +29% mortality prediction | [06_medical_imaging/](research/06_medical_imaging/) |
| `arxiv_multimodal_fusion.md` | +3-8% AUROC multimodal | [06_medical_imaging/](research/06_medical_imaging/) |
| `arxiv_multimodal_clinical.md` | Cross-modal learning | [06_medical_imaging/](research/06_medical_imaging/) |

### Modality-Specific Encoders

| Modality | Document | Key Metric | Location |
|----------|----------|------------|----------|
| ECG/Waveforms | `arxiv_ecg_deep_learning.md` | 95-99% arrhythmia accuracy | [06_medical_imaging/](research/06_medical_imaging/) |
| Chest X-ray | `arxiv_chest_xray_ai.md` | Foundation models | [06_medical_imaging/](research/06_medical_imaging/) |
| Vitals | `arxiv_vital_signs_ml.md` | Continuous monitoring | [06_medical_imaging/](research/06_medical_imaging/) |
| Clinical Notes | `arxiv_clinical_nlp.md` | NER F1 88.8% | [05_clinical_nlp/](research/05_clinical_nlp/) |
| Waveforms | `arxiv_waveform_signal_clinical.md` | <100ms real-time | [06_medical_imaging/](research/06_medical_imaging/) |

### Missing Data Handling

| Document | Application | Location |
|----------|------------|----------|
| `arxiv_clinical_missing_data.md` | Imputation methods | [12_data_quality/](research/12_data_quality/) |
| `arxiv_ehr_data_quality.md` | GRU-D +3-5% | [12_data_quality/](research/12_data_quality/) |

---

## 6. Technical Annex A: Temporal KG World Model

### Node Types

| Document | Coverage | Location |
|----------|----------|----------|
| `arxiv_clinical_graph_construction.md` | Entity extraction | [02_graph_neural_networks/](research/02_graph_neural_networks/) |
| `arxiv_medical_ontology_integration.md` | SNOMED/UMLS | [02_graph_neural_networks/](research/02_graph_neural_networks/) |
| `ohdsi_omop_cdm.md` | 39-table CDM | [_foundations/](research/_foundations/) |

### Edge Types

| Document | Coverage | Location |
|----------|----------|----------|
| `allen_temporal_algebra.md` | 13 temporal relations | [_foundations/](research/_foundations/) |
| `arxiv_temporal_reasoning.md` | Composition table | [01_temporal_methods/](research/01_temporal_methods/) |
| `arxiv_healthcare_causal_discovery.md` | Causal edge learning | [09_learning_methods/](research/09_learning_methods/) |

---

## 7. Technical Annex B: Diffusion Models

| Application | Document | Location |
|-------------|----------|----------|
| KG state sequences | `arxiv_sequence_diffusion.md` | [01_temporal_methods/](research/01_temporal_methods/) |
| Constraint guidance | `arxiv_constraint_guided_generation.md` | [03_hybrid_neurosymbolic/](research/03_hybrid_neurosymbolic/) |
| Counterfactuals | `arxiv_counterfactual_clinical.md` | [09_learning_methods/](research/09_learning_methods/) |
| Rare case augmentation | `arxiv_clinical_data_augmentation.md` | [12_data_quality/](research/12_data_quality/) |
| RL simulation | `arxiv_clinical_rl_simulation.md` | [09_learning_methods/](research/09_learning_methods/) |

---

## 8. Technical Annex C: Multimodal Fusion Architecture

### Encoder Documents

| Encoder Type | Document | Location |
|--------------|----------|----------|
| Imaging | `arxiv_chest_xray_ai.md`, `arxiv_clinical_imaging_text.md` | [06_medical_imaging/](research/06_medical_imaging/) |
| Waveforms | `arxiv_ecg_deep_learning.md`, `arxiv_waveform_signal_clinical.md` | [06_medical_imaging/](research/06_medical_imaging/) |
| Text | `arxiv_clinical_nlp.md`, `arxiv_clinical_note_generation.md` | [05_clinical_nlp/](research/05_clinical_nlp/) |
| Structured | `arxiv_clinical_embeddings_representations.md` | [02_graph_neural_networks/](research/02_graph_neural_networks/) |

### Cross-Modal Attention

| Document | Coverage | Location |
|----------|----------|----------|
| `arxiv_attention_mechanisms_medical.md` | Multi-head, cross-modal | [02_graph_neural_networks/](research/02_graph_neural_networks/) |
| `arxiv_multimodal_temporal_fusion.md` | Temporal alignment | [06_medical_imaging/](research/06_medical_imaging/) |

---

## 9. Technical Annex D: Clinical Tasks & Evaluation

### D.1 ICD-10 Diagnosis Coding

| Document | Key Metric | Location |
|----------|------------|----------|
| `arxiv_medical_coding_ai.md` | 60.8% micro-F1 (XR-LAT) | [05_clinical_nlp/](research/05_clinical_nlp/) |
| `arxiv_cdi_documentation_ai.md` | CDI query generation | [05_clinical_nlp/](research/05_clinical_nlp/) |
| `arxiv_clinical_nlp.md` | Clinical BERT NER | [05_clinical_nlp/](research/05_clinical_nlp/) |

### D.2 E/M & Critical Care

| Document | Coverage | Location |
|----------|----------|----------|
| `arxiv_clinical_risk_scores.md` | SOFA, APACHE | [04_clinical_prediction/](research/04_clinical_prediction/) |
| `arxiv_icu_outcomes.md` | Severity classification | [04_clinical_prediction/](research/04_clinical_prediction/) |
| `arxiv_cdi_documentation_ai.md` | E/M level prediction | [05_clinical_nlp/](research/05_clinical_nlp/) |

### D.3 Explanation Quality

| Document | Coverage | Location |
|----------|----------|----------|
| `arxiv_evidence_extraction_clinical.md` | PICO extraction 90.7% | [05_clinical_nlp/](research/05_clinical_nlp/) |
| `arxiv_clinical_interpretability.md` | SHAP, LIME | [11_interpretability_safety/](research/11_interpretability_safety/) |
| `arxiv_explainable_ai_clinical.md` | 45 papers on XAI | [11_interpretability_safety/](research/11_interpretability_safety/) |

---

## 10. Technical Annex E: Evaluation Framework

| Metric Type | Document | Location |
|-------------|----------|----------|
| Quantitative (F1, AUROC) | `arxiv_clinical_benchmarks.md` | [10_implementation_deployment/](research/10_implementation_deployment/) |
| Data quality | `arxiv_ehr_data_quality.md` | [12_data_quality/](research/12_data_quality/) |
| Expert validation | `arxiv_clinical_validation.md` | [10_implementation_deployment/](research/10_implementation_deployment/) |
| Constraint satisfaction | `arxiv_constraint_satisfaction.md` | [03_hybrid_neurosymbolic/](research/03_hybrid_neurosymbolic/) |
| Deployment metrics | `arxiv_clinical_ai_deployment.md` | [10_implementation_deployment/](research/10_implementation_deployment/) |

---

## 11. Technical Annex F: Risks and Mitigations

| Risk | Mitigation Document | Location |
|------|---------------------|----------|
| KG construction noise | `arxiv_ehr_data_quality.md` | [12_data_quality/](research/12_data_quality/) |
| Label noise | `arxiv_clinical_validation.md` | [10_implementation_deployment/](research/10_implementation_deployment/) |
| Scope creep | `arxiv_clinical_world_models.md` | [09_learning_methods/](research/09_learning_methods/) |
| Generative safety | `arxiv_constraint_guided_generation.md` | [03_hybrid_neurosymbolic/](research/03_hybrid_neurosymbolic/) |
| Alert fatigue | `arxiv_clinical_alert_fatigue.md` | [08_clinical_operations/](research/08_clinical_operations/) |

---

## 12. Foundations & Institutional Context

### Datasets

| Document | Coverage | Location |
|----------|----------|----------|
| `mimic_iv_dataset_details.md` | 364,627 patients, 35,239 sepsis | [_foundations/](research/_foundations/) |
| `arxiv_clinical_benchmarks.md` | eICU, PhysioNet | [10_implementation_deployment/](research/10_implementation_deployment/) |

### Institutional Partnership

| Document | Coverage | Location |
|----------|----------|----------|
| `hca_ucf_partnership.md` | 182+ hospitals, 1,000+ residents | [_institutional/](research/_institutional/) |
| `ucf_faculty_profiles.md` | CRCV, IAI collaborators | [_institutional/](research/_institutional/) |
| `orlando_health_ai_initiatives.md` | Regional health system | [_institutional/](research/_institutional/) |

### Regulatory & Standards

| Document | Coverage | Location |
|----------|----------|----------|
| `fda_cds_guidance_current.md` | 4-criteria CDS exemption | [_foundations/](research/_foundations/) |
| `fhir_clinical_standards.md` | FHIR R4 integration | [_foundations/](research/_foundations/) |
| `ohdsi_omop_cdm.md` | OMOP v5.4 standardization | [_foundations/](research/_foundations/) |

### Funding

| Document | Coverage | Location |
|----------|----------|----------|
| `nsf_smart_health_awards_2024.md` | $1.2M awards, Oct 3 deadline | [_institutional/](research/_institutional/) |
| `clinical_trials_ai.md` | 3,106 AI/ML trials | [04_clinical_prediction/](research/04_clinical_prediction/) |

---

## Quick Reference: Top Documents by Thread

| Thread | Top 3 Documents | Folder |
|--------|----------------|--------|
| **1. Neuro-Symbolic** | `arxiv_neurosymbolic_clinical_reasoning.md`, `ibm_lnn_framework.md`, `arxiv_constraint_guided_generation.md` | `03_hybrid_neurosymbolic/`, `_foundations/` |
| **2. Temporal KGs** | `arxiv_temporal_kg_2024.md`, `allen_temporal_algebra.md`, `arxiv_clinical_graph_construction.md` | `01_temporal_methods/`, `_foundations/`, `02_graph_neural_networks/` |
| **3. Diffusion** | `arxiv_sequence_diffusion.md`, `arxiv_diffusion_healthcare.md`, `arxiv_counterfactual_clinical.md` | `01_temporal_methods/`, `09_learning_methods/` |
| **4. Multimodal** | `arxiv_multimodal_temporal_fusion.md`, `arxiv_multimodal_fusion.md`, `arxiv_ecg_deep_learning.md` | `06_medical_imaging/` |

---

## Repository Statistics

| Category | Count |
|----------|-------|
| Total Documents | 157 |
| Total Lines | 197,000+ |
| ArXiv Papers Synthesized | 2,000+ |
| Topic Categories | 16 |
| Concept Note Coverage | 100% |

---

<p align="center">
  <strong>Last Updated:</strong> December 2025 &nbsp;|&nbsp; <strong>Version:</strong> 2.0
</p>
