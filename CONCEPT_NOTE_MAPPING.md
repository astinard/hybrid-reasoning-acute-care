# Concept Note → Research Repository Mapping

**Purpose:** Map the CRCV Lab Concept Note structure to supporting research documents in this repository.

**Repository:** `/Users/alexstinard/hybrid-reasoning-acute-care/`
**Generated:** December 2025
**Documents Mapped:** 157 research documents → Concept Note structure

---

## How to Use This Document

This mapping follows the **exact hierarchy** of the Concept Note for Collaboration. When reviewing the concept note, reference this document to find supporting research literature for each section.

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

The concept note's core claims are supported by these foundational documents:

| Claim | Supporting Document(s) |
|-------|----------------------|
| Temporal KGs as world models | `arxiv_temporal_kg_2024.md`, `allen_temporal_algebra.md` |
| Neuro-symbolic constraints | `arxiv_neurosymbolic_clinical_reasoning.md`, `ibm_lnn_framework.md` |
| Diffusion over trajectories | `arxiv_sequence_diffusion.md`, `arxiv_diffusion_healthcare.md` |
| Multimodal fusion | `arxiv_multimodal_temporal_fusion.md`, `arxiv_multimodal_fusion.md` |
| Teacher-student distillation | `arxiv_teacher_student_clinical.md` |
| ED auto-coding evaluation | `arxiv_medical_coding_ai.md`, `arxiv_cdi_documentation_ai.md` |

---

## 2. Thread 1: Neuro-Symbolic Constraints

**Concept Note Section:** *Neuro-symbolic constraints for uncertain clinical reasoning*

### Core Documents (Must-Read)

| Document | Key Finding | Lines |
|----------|-------------|-------|
| [arxiv_neurosymbolic_clinical_reasoning.md](research/arxiv_neurosymbolic_clinical_reasoning.md) | 120+ papers; GraphCare +17.6% AUROC | 1,200+ |
| [ibm_lnn_framework.md](research/ibm_lnn_framework.md) | LNN: 80.52% accuracy, 0.8457 AUROC on diabetes | 800+ |
| [arxiv_hybrid_symbolic_neural.md](research/arxiv_hybrid_symbolic_neural.md) | Neural Theorem Provers +17% | 1,400+ |
| [arxiv_constraint_satisfaction.md](research/arxiv_constraint_satisfaction.md) | CP-SAT 18 seconds for OR scheduling | 900+ |
| [arxiv_constraint_guided_generation.md](research/arxiv_constraint_guided_generation.md) | 100% constraint satisfaction for safety | 1,100+ |

### Clinical Rule Encoding

| Document | Supports |
|----------|----------|
| [arxiv_guideline_encoding.md](research/arxiv_guideline_encoding.md) | CQL, Arden Syntax, FHIR CQL |
| [fda_cds_guidance_current.md](research/fda_cds_guidance_current.md) | FDA 4-criteria CDS exemption |
| [arxiv_clinical_pathways.md](research/arxiv_clinical_pathways.md) | Clinical protocol adherence |

### Time-Sensitive Protocols

| Protocol | Document | Key Metric |
|----------|----------|------------|
| Sepsis bundles | [arxiv_sepsis_prediction.md](research/arxiv_sepsis_prediction.md) | AUROC 0.88-0.97 |
| Stroke windows | [arxiv_stroke_prediction.md](research/arxiv_stroke_prediction.md) | 85-93% LVO sensitivity |
| ACS rule-out | [arxiv_cardiac_arrest.md](research/arxiv_cardiac_arrest.md) | HEART score validation |

### Billing Policy (AMA E/M, NCCI)

| Document | Supports |
|----------|----------|
| [arxiv_cdi_documentation_ai.md](research/arxiv_cdi_documentation_ai.md) | E/M level prediction, CDI query generation |
| [arxiv_medical_coding_ai.md](research/arxiv_medical_coding_ai.md) | ICD-10 coding: 60.8% micro-F1 SOTA |

---

## 3. Thread 2: Temporal KGs as World Models

**Concept Note Section:** *Temporal Knowledge Graphs as world models*

### Core Documents (Must-Read)

| Document | Key Finding | Lines |
|----------|-------------|-------|
| [arxiv_temporal_kg_2024.md](research/arxiv_temporal_kg_2024.md) | KAT-GNN AUROC 0.9269 | 1,100+ |
| [allen_temporal_algebra.md](research/allen_temporal_algebra.md) | 13 interval relations, composition table | 900+ |
| [arxiv_temporal_reasoning.md](research/arxiv_temporal_reasoning.md) | LTL, CTL, Event Calculus | 1,000+ |
| [arxiv_clinical_graph_construction.md](research/arxiv_clinical_graph_construction.md) | 90-96% NER, 73-96% RE | 900+ |
| [arxiv_temporal_gnn.md](research/arxiv_temporal_gnn.md) | DyRep, TGN architectures | 1,200+ |

### Graph Neural Networks

| Document | Application |
|----------|------------|
| [arxiv_gnn_clinical_2024.md](research/arxiv_gnn_clinical_2024.md) | AUROC 70-94% clinical prediction |
| [arxiv_kg_reasoning_clinical.md](research/arxiv_kg_reasoning_clinical.md) | ComplEx MRR 0.50 |
| [arxiv_graph_embeddings_healthcare.md](research/arxiv_graph_embeddings_healthcare.md) | Snomed2Vec, Node2Vec, RotatE |

### Event Sequence Modeling

| Document | Key Finding |
|----------|-------------|
| [arxiv_event_sequence_clinical.md](research/arxiv_event_sequence_clinical.md) | Hawkes processes, Neural TPPs |
| [arxiv_time_series_clinical.md](research/arxiv_time_series_clinical.md) | AUC 0.85-0.93 |

---

## 4. Thread 3: Diffusion Over Clinical Trajectories

**Concept Note Section:** *Diffusion models over clinical trajectories*

### Core Documents (Must-Read)

| Document | Key Finding | Lines |
|----------|-------------|-------|
| [arxiv_sequence_diffusion.md](research/arxiv_sequence_diffusion.md) | SEDD 25-75% perplexity reduction | 1,600+ |
| [arxiv_diffusion_healthcare.md](research/arxiv_diffusion_healthcare.md) | CSDI 40-65% imputation improvement | 1,400+ |
| [arxiv_graph_diffusion.md](research/arxiv_graph_diffusion.md) | DiGress 3x validity improvement | 1,100+ |
| [arxiv_counterfactual_clinical.md](research/arxiv_counterfactual_clinical.md) | TraCE trajectory explanations | 700+ |
| [arxiv_clinical_world_models.md](research/arxiv_clinical_world_models.md) | ED as ideal testbed for world models | 1,000+ |

### Constraint-Guided Generation

| Document | Application |
|----------|------------|
| [arxiv_constraint_guided_generation.md](research/arxiv_constraint_guided_generation.md) | 100% constraint satisfaction |
| [arxiv_clinical_rl_simulation.md](research/arxiv_clinical_rl_simulation.md) | Conservative Q-learning, medDreamer |

### Data Augmentation

| Document | Application |
|----------|------------|
| [arxiv_clinical_data_augmentation.md](research/arxiv_clinical_data_augmentation.md) | Rare disease handling |
| [arxiv_synthetic_clinical.md](research/arxiv_synthetic_clinical.md) | Privacy-preserving generation |

---

## 5. Thread 4: Multimodal Fusion

**Concept Note Section:** *Multimodal fusion over a KG scaffold*

### Core Documents (Must-Read)

| Document | Key Finding | Lines |
|----------|-------------|-------|
| [arxiv_multimodal_temporal_fusion.md](research/arxiv_multimodal_temporal_fusion.md) | +29% mortality prediction | 1,100+ |
| [arxiv_multimodal_fusion.md](research/arxiv_multimodal_fusion.md) | +3-8% AUROC multimodal | 2,100+ |
| [arxiv_multimodal_clinical.md](research/arxiv_multimodal_clinical.md) | Cross-modal learning | 1,100+ |

### Modality-Specific Encoders

| Modality | Document | Key Metric |
|----------|----------|------------|
| ECG/Waveforms | [arxiv_ecg_deep_learning.md](research/arxiv_ecg_deep_learning.md) | 95-99% arrhythmia accuracy |
| Chest X-ray | [arxiv_chest_xray_ai.md](research/arxiv_chest_xray_ai.md) | Foundation models |
| Vitals | [arxiv_vital_signs_ml.md](research/arxiv_vital_signs_ml.md) | Continuous monitoring |
| Clinical Notes | [arxiv_clinical_nlp.md](research/arxiv_clinical_nlp.md) | NER F1 88.8% |
| Waveforms | [arxiv_waveform_signal_clinical.md](research/arxiv_waveform_signal_clinical.md) | <100ms real-time |

### Missing Data Handling

| Document | Application |
|----------|------------|
| [arxiv_clinical_missing_data.md](research/arxiv_clinical_missing_data.md) | Imputation methods |
| [arxiv_ehr_data_quality.md](research/arxiv_ehr_data_quality.md) | GRU-D +3-5% |

---

## 6. Technical Annex A: Temporal KG World Model

**Concept Note Section:** *Temporal Knowledge Graph: The World Model*

### Node Types (Symptoms, Labs, Imaging, Meds, Vitals)

| Document | Coverage |
|----------|----------|
| [arxiv_clinical_graph_construction.md](research/arxiv_clinical_graph_construction.md) | Entity extraction from EHR |
| [arxiv_medical_ontology_integration.md](research/arxiv_medical_ontology_integration.md) | SNOMED/UMLS integration |
| [ohdsi_omop_cdm.md](research/ohdsi_omop_cdm.md) | 39-table CDM schema |

### Edge Types (precedes, supports, contradicts)

| Document | Coverage |
|----------|----------|
| [allen_temporal_algebra.md](research/allen_temporal_algebra.md) | 13 temporal relations |
| [arxiv_temporal_reasoning.md](research/arxiv_temporal_reasoning.md) | Composition table |
| [arxiv_healthcare_causal_discovery.md](research/arxiv_healthcare_causal_discovery.md) | Causal edge learning |

### Example Trajectories

| Trajectory | Document |
|------------|----------|
| Sepsis | [arxiv_sepsis_prediction.md](research/arxiv_sepsis_prediction.md) |
| Stroke | [arxiv_stroke_prediction.md](research/arxiv_stroke_prediction.md) |
| ACS | [arxiv_cardiac_arrest.md](research/arxiv_cardiac_arrest.md) |
| Respiratory failure | [arxiv_respiratory_ai.md](research/arxiv_respiratory_ai.md) |

---

## 7. Technical Annex B: Diffusion Models

**Concept Note Section:** *Diffusion Models Over Clinical Trajectories*

| Application | Document |
|-------------|----------|
| KG state sequences | [arxiv_sequence_diffusion.md](research/arxiv_sequence_diffusion.md) |
| Constraint guidance | [arxiv_constraint_guided_generation.md](research/arxiv_constraint_guided_generation.md) |
| Counterfactuals | [arxiv_counterfactual_clinical.md](research/arxiv_counterfactual_clinical.md) |
| Rare case augmentation | [arxiv_clinical_data_augmentation.md](research/arxiv_clinical_data_augmentation.md) |
| RL simulation | [arxiv_clinical_rl_simulation.md](research/arxiv_clinical_rl_simulation.md) |

---

## 8. Technical Annex C: Multimodal Fusion Architecture

**Concept Note Section:** *Multimodal Fusion Architecture*

### Encoder Documents

| Encoder Type | Document |
|--------------|----------|
| Imaging (CT, X-ray) | [arxiv_chest_xray_ai.md](research/arxiv_chest_xray_ai.md), [arxiv_clinical_imaging_text.md](research/arxiv_clinical_imaging_text.md) |
| Waveforms (ECG, vitals) | [arxiv_ecg_deep_learning.md](research/arxiv_ecg_deep_learning.md), [arxiv_waveform_signal_clinical.md](research/arxiv_waveform_signal_clinical.md) |
| Text (notes) | [arxiv_clinical_nlp.md](research/arxiv_clinical_nlp.md), [arxiv_clinical_note_generation.md](research/arxiv_clinical_note_generation.md) |
| Structured (labs, meds) | [arxiv_clinical_embeddings_representations.md](research/arxiv_clinical_embeddings_representations.md) |

### Cross-Modal Attention

| Document | Coverage |
|----------|----------|
| [arxiv_attention_mechanisms_medical.md](research/arxiv_attention_mechanisms_medical.md) | Multi-head, cross-modal |
| [arxiv_multimodal_temporal_fusion.md](research/arxiv_multimodal_temporal_fusion.md) | Temporal alignment |

---

## 9. Technical Annex D: Clinical Tasks & Evaluation

**Concept Note Section:** *Clinical Tasks & Evaluation Signals*

### D.1 ICD-10 Diagnosis Coding

| Document | Key Metric |
|----------|------------|
| [arxiv_medical_coding_ai.md](research/arxiv_medical_coding_ai.md) | 60.8% micro-F1 (XR-LAT) |
| [arxiv_cdi_documentation_ai.md](research/arxiv_cdi_documentation_ai.md) | CDI query generation |
| [arxiv_clinical_nlp.md](research/arxiv_clinical_nlp.md) | Clinical BERT NER |

### D.2 E/M & Critical Care

| Document | Coverage |
|----------|----------|
| [arxiv_clinical_risk_scores.md](research/arxiv_clinical_risk_scores.md) | SOFA, APACHE complexity |
| [arxiv_icu_outcomes.md](research/arxiv_icu_outcomes.md) | Severity classification |
| [arxiv_cdi_documentation_ai.md](research/arxiv_cdi_documentation_ai.md) | E/M level prediction |

### D.3 Explanation Quality

| Document | Coverage |
|----------|----------|
| [arxiv_evidence_extraction_clinical.md](research/arxiv_evidence_extraction_clinical.md) | PICO extraction 90.7% NER |
| [arxiv_clinical_interpretability.md](research/arxiv_clinical_interpretability.md) | SHAP, LIME |
| [arxiv_explainable_ai_clinical.md](research/arxiv_explainable_ai_clinical.md) | 45 papers on XAI |

---

## 10. Technical Annex E: Evaluation Framework

**Concept Note Section:** *Evaluation Framework*

| Metric Type | Document |
|-------------|----------|
| Quantitative (F1, AUROC) | [arxiv_clinical_benchmarks.md](research/arxiv_clinical_benchmarks.md) |
| Data quality | [arxiv_ehr_data_quality.md](research/arxiv_ehr_data_quality.md) |
| Expert validation | [arxiv_clinical_validation.md](research/arxiv_clinical_validation.md) |
| Constraint satisfaction | [arxiv_constraint_satisfaction.md](research/arxiv_constraint_satisfaction.md) |
| Deployment metrics | [arxiv_clinical_ai_deployment.md](research/arxiv_clinical_ai_deployment.md) |

---

## 11. Technical Annex F: Risks and Mitigations

**Concept Note Section:** *Risks and Mitigations*

| Risk | Mitigation Document |
|------|---------------------|
| KG construction noise | [arxiv_ehr_data_quality.md](research/arxiv_ehr_data_quality.md) |
| Label noise | [arxiv_clinical_validation.md](research/arxiv_clinical_validation.md) |
| Scope creep | [arxiv_clinical_world_models.md](research/arxiv_clinical_world_models.md) |
| Generative safety | [arxiv_constraint_guided_generation.md](research/arxiv_constraint_guided_generation.md) |
| Alert fatigue | [arxiv_clinical_alert_fatigue.md](research/arxiv_clinical_alert_fatigue.md) |

---

## 12. Foundations & Institutional Context

**Concept Note Sections:** *MIMIC-IV, UCF/HCA, FDA, FHIR/OMOP, Funding*

### Datasets

| Document | Coverage |
|----------|----------|
| [mimic_iv_dataset_details.md](research/mimic_iv_dataset_details.md) | 364,627 patients, 35,239 sepsis cases |
| [arxiv_clinical_benchmarks.md](research/arxiv_clinical_benchmarks.md) | eICU, PhysioNet |

### Institutional Partnership

| Document | Coverage |
|----------|----------|
| [hca_ucf_partnership.md](research/hca_ucf_partnership.md) | 182+ hospitals, 1,000+ residents |
| [ucf_faculty_profiles.md](research/ucf_faculty_profiles.md) | CRCV, IAI collaborators |
| [orlando_health_ai_initiatives.md](research/orlando_health_ai_initiatives.md) | Regional health system |

### Regulatory & Standards

| Document | Coverage |
|----------|----------|
| [fda_cds_guidance_current.md](research/fda_cds_guidance_current.md) | 4-criteria CDS exemption |
| [fhir_clinical_standards.md](research/fhir_clinical_standards.md) | FHIR R4 integration |
| [ohdsi_omop_cdm.md](research/ohdsi_omop_cdm.md) | OMOP v5.4 standardization |

### Prior Art Lessons

| Document | Coverage |
|----------|----------|
| [epic_sepsis_model_analysis.md](research/epic_sepsis_model_analysis.md) | External validation failure (AUROC 0.63) |

### Funding

| Document | Coverage |
|----------|----------|
| [nsf_smart_health_awards_2024.md](research/nsf_smart_health_awards_2024.md) | $1.2M awards, Oct 3 deadline |
| [clinical_trials_ai.md](research/clinical_trials_ai.md) | 3,106 AI/ML trials |

---

## Quick Reference: Top Documents by Thread

### Thread 1: Neuro-Symbolic
1. `arxiv_neurosymbolic_clinical_reasoning.md`
2. `ibm_lnn_framework.md`
3. `arxiv_constraint_guided_generation.md`

### Thread 2: Temporal KGs
1. `arxiv_temporal_kg_2024.md`
2. `allen_temporal_algebra.md`
3. `arxiv_clinical_graph_construction.md`

### Thread 3: Diffusion
1. `arxiv_sequence_diffusion.md`
2. `arxiv_diffusion_healthcare.md`
3. `arxiv_counterfactual_clinical.md`

### Thread 4: Multimodal
1. `arxiv_multimodal_temporal_fusion.md`
2. `arxiv_multimodal_fusion.md`
3. `arxiv_ecg_deep_learning.md`

---

## Repository Statistics

| Category | Count |
|----------|-------|
| Total Documents | 157 |
| Total Lines | 197,000+ |
| ArXiv Papers Synthesized | 2,000+ |
| Concept Note Threads Covered | 4/4 (100%) |
| Technical Annex Sections | 6/6 (100%) |
| Foundation Areas | 6/6 (100%) |

---

**Last Updated:** December 2025
