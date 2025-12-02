# Research Literature Corpus

<p align="center">
  <strong>158 documents</strong> &nbsp;|&nbsp; <strong>200,000+ lines</strong> &nbsp;|&nbsp; <strong>2,000+ ArXiv papers</strong>
</p>

Comprehensive literature syntheses covering every major topic in clinical AI for acute care.

---

## Quick Navigation

| Category | Docs | Focus |
|----------|------|-------|
| [_foundations/](_foundations/) | 12 | MIMIC-IV, FHIR, OMOP, FDA, IBM LNN |
| [_institutional/](_institutional/) | 6 | UCF/HCA partnership, funding |
| [_domain_synthesis/](_domain_synthesis/) | 2 | Cross-cutting analysis |
| [01_temporal_methods/](01_temporal_methods/) | 7 | Temporal KGs, time series, event sequences |
| [02_graph_neural_networks/](02_graph_neural_networks/) | 10 | GNN, embeddings, KG reasoning |
| [03_hybrid_neurosymbolic/](03_hybrid_neurosymbolic/) | 7 | Neuro-symbolic, constraints |
| [04_clinical_prediction/](04_clinical_prediction/) | 19 | Sepsis, AKI, mortality, risk |
| [05_clinical_nlp/](05_clinical_nlp/) | 10 | NLP, coding, summarization |
| [06_medical_imaging/](06_medical_imaging/) | 9 | ECG, X-ray, multimodal fusion |
| [07_specialty_domains/](07_specialty_domains/) | 19 | Oncology, pediatric, surgical |
| [08_clinical_operations/](08_clinical_operations/) | 8 | Triage, ED, alerts, workflow |
| [09_learning_methods/](09_learning_methods/) | 18 | Transfer, RL, causal, diffusion |
| [10_implementation_deployment/](10_implementation_deployment/) | 12 | MLOps, federated, edge, validation |
| [11_interpretability_safety/](11_interpretability_safety/) | 10 | XAI, fairness, safety |
| [12_data_quality/](12_data_quality/) | 6 | Missing data, synthetic, augmentation |
| [13_emerging_technology/](13_emerging_technology/) | 9 | LLMs, digital twins, quantum |
| [14_agentic_medical_ai/](14_agentic_medical_ai/) | 1 | Agentic reasoning, evidence-based diagnosis |

---

## Foundations & Context

### _foundations/ (12 documents)
Core standards, datasets, and frameworks.

| Document | Description | Key Metric |
|----------|-------------|------------|
| `allen_temporal_algebra.md` | Allen's 13 interval relations | Temporal logic foundation |
| `ibm_lnn_framework.md` | IBM Logical Neural Networks | 17% improvement, 100x compression |
| `fhir_clinical_standards.md` | FHIR R4 data standards | Interoperability |
| `ohdsi_omop_cdm.md` | OMOP Common Data Model | 39 tables, 810M patients |
| `mimic_iv_dataset_details.md` | MIMIC-IV dataset | 364,627 patients |
| `fda_cds_guidance_current.md` | FDA CDS regulatory pathway | 4-criteria exemption |
| `epic_sepsis_model_analysis.md` | Epic model failure analysis | AUROC 0.63 (why it failed) |
| `arxiv_fda_ai_guidance.md` | FDA AI/ML guidance | Regulatory framework |

### _institutional/ (6 documents)
Partnership and funding landscape.

| Document | Description |
|----------|-------------|
| `hca_ucf_partnership.md` | HCA/UCF partnership (182+ hospitals) |
| `ucf_faculty_profiles.md` | UCF collaborator profiles |
| `orlando_health_ai_initiatives.md` | Regional health system AI |
| `nsf_smart_health_awards_2024.md` | NSF funding opportunities |
| `nih_funding_mechanisms.md` | NIH R01/R21 pathways |
| `commercial_cds_vendors.md` | Competitive landscape |

---

## Core Technical Methods

### 01_temporal_methods/ (7 documents)
Temporal reasoning for clinical events.

| Document | Key Finding |
|----------|-------------|
| `arxiv_temporal_kg_2024.md` | KAT-GNN AUROC 0.9269 |
| `arxiv_temporal_reasoning.md` | Allen algebra formalization |
| `arxiv_temporal_gnn.md` | Dynamic temporal GNNs |
| `arxiv_time_series_clinical.md` | AUC 0.85-0.93 |
| `arxiv_clinical_time_series_forecasting.md` | Forecasting methods |
| `arxiv_event_sequence_clinical.md` | Event sequence modeling |
| `arxiv_sequence_diffusion.md` | SEDD, trajectory generation |

### 02_graph_neural_networks/ (10 documents)
Graph-based learning for healthcare.

| Document | Key Finding |
|----------|-------------|
| `arxiv_gnn_clinical_2024.md` | AUROC 70-94% |
| `arxiv_kg_reasoning_clinical.md` | ComplEx MRR 0.50 |
| `arxiv_graph_embeddings_healthcare.md` | Patient similarity |
| `arxiv_graph_diffusion.md` | DiGress, graph generation |
| `arxiv_clinical_graph_construction.md` | Graph construction methods |
| `arxiv_ddi_knowledge_graphs.md` | Drug-drug interactions |
| `arxiv_clinical_embeddings.md` | Patient representations |
| `arxiv_clinical_embeddings_representations.md` | Deep embeddings |
| `arxiv_attention_mechanisms_medical.md` | Self/cross attention |
| `arxiv_medical_ontology_integration.md` | SNOMED, LOINC integration |

### 03_hybrid_neurosymbolic/ (7 documents)
Neuro-symbolic AI with constraints.

| Document | Key Finding |
|----------|-------------|
| `arxiv_neurosymbolic_healthcare.md` | 60+ papers synthesized |
| `arxiv_neurosymbolic_clinical_reasoning.md` | Clinical reasoning patterns |
| `arxiv_hybrid_symbolic_neural.md` | 17% improvement |
| `arxiv_constraint_satisfaction.md` | CP-SAT 18 seconds |
| `arxiv_constraint_guided_generation.md` | Constrained generation |
| `arxiv_guideline_encoding.md` | CQL, Arden Syntax |
| `arxiv_clinical_pathways.md` | Protocol encoding |

---

## Clinical Applications

### 04_clinical_prediction/ (19 documents)
Risk prediction and outcome forecasting.

| Document | Key Finding |
|----------|-------------|
| `arxiv_sepsis_prediction.md` | AUROC 0.88-0.97, 6hr advance |
| `arxiv_mortality_prediction_icu.md` | AUROC 0.85-0.98 |
| `arxiv_aki_prediction.md` | AUROC 0.75-0.93 |
| `arxiv_cardiac_arrest.md` | 6-hour early warning |
| `arxiv_stroke_prediction.md` | Time-critical protocols |
| `arxiv_clinical_risk_scores.md` | Risk stratification |
| `arxiv_risk_stratification_models.md` | Model comparison |
| `arxiv_diagnosis_prediction.md` | Diagnostic AI |
| `arxiv_patient_outcome_prediction.md` | Outcome modeling |
| `arxiv_icu_outcomes.md` | ICU-specific outcomes |
| `arxiv_clinical_phenotyping.md` | Patient phenotypes |
| `arxiv_lab_prediction.md` | Lab value forecasting |
| `arxiv_patient_similarity_cohort.md` | Cohort identification |
| `arxiv_medication_treatment_ai.md` | Treatment optimization |
| `arxiv_sdoh_ai.md` | Social determinants |
| `arxiv_cost_prediction_ai.md` | Cost modeling |
| `arxiv_clinical_trial_ai_methods.md` | Trial optimization |
| `clinical_trials_ai.md` | 3,106 AI/ML trials |

### 05_clinical_nlp/ (10 documents)
Natural language processing for clinical text.

| Document | Key Finding |
|----------|-------------|
| `arxiv_clinical_nlp.md` | Comprehensive NLP survey |
| `arxiv_medical_coding_ai.md` | 89%+ F1 ICD-10 |
| `arxiv_cdi_documentation_ai.md` | E/M level prediction |
| `arxiv_clinical_information_extraction.md` | Entity extraction |
| `arxiv_clinical_summarization.md` | Note summarization |
| `arxiv_clinical_note_generation.md` | Note generation |
| `arxiv_clinical_qa.md` | Question answering |
| `arxiv_clinical_question_answering.md` | QA systems |
| `arxiv_clinical_nlg.md` | Natural language generation |
| `arxiv_evidence_extraction_clinical.md` | Evidence extraction |

### 06_medical_imaging/ (9 documents)
Imaging, waveforms, and multimodal fusion.

| Document | Key Finding |
|----------|-------------|
| `arxiv_chest_xray_ai.md` | CXR analysis |
| `arxiv_ecg_deep_learning.md` | 95-99% accuracy |
| `arxiv_multimodal_fusion.md` | Cross-modal attention |
| `arxiv_multimodal_temporal_fusion.md` | Temporal alignment |
| `arxiv_multimodal_clinical.md` | Multimodal architectures |
| `arxiv_clinical_imaging_text.md` | Image-text fusion |
| `arxiv_waveform_signal_clinical.md` | Waveform analysis |
| `arxiv_vital_signs_ml.md` | Vital sign monitoring |
| `arxiv_wearables_monitoring.md` | Wearable devices |

### 07_specialty_domains/ (19 documents)
Medical specialty-specific AI.

| Document | Specialty |
|----------|-----------|
| `arxiv_oncology_ai.md` | Cancer/Oncology |
| `arxiv_pediatric_ai.md` | Pediatrics |
| `arxiv_psychiatry_ai.md` | Mental Health |
| `arxiv_surgical_ai.md` | Surgery |
| `arxiv_surgical_robotics.md` | Robotic Surgery |
| `arxiv_cardiac_surgery_ai.md` | Cardiac Surgery |
| `arxiv_obstetrics_ai.md` | Obstetrics |
| `arxiv_geriatric_ai.md` | Geriatrics |
| `arxiv_trauma_ai.md` | Trauma |
| `arxiv_infectious_ai.md` | Infectious Disease |
| `arxiv_respiratory_ai.md` | Respiratory |
| `arxiv_liver_ai.md` | Hepatology |
| `arxiv_neuro_icu_ai.md` | Neuro-ICU |
| `arxiv_coagulopathy_ai.md` | Coagulopathy |
| `arxiv_electrolyte_ai.md` | Electrolyte Management |
| `arxiv_fluid_ai.md` | Fluid Management |
| `arxiv_nutrition_ai.md` | Clinical Nutrition |
| `arxiv_pain_ai.md` | Pain Management |
| `arxiv_genomics_clinical.md` | Genomics |

### 08_clinical_operations/ (8 documents)
ED and hospital operations.

| Document | Key Finding |
|----------|-------------|
| `arxiv_triage_ml.md` | Triage optimization |
| `arxiv_ed_crowding.md` | ED flow management |
| `arxiv_clinical_alerts.md` | Alert systems |
| `arxiv_clinical_alert_fatigue.md` | 50%→75% precision |
| `arxiv_discharge_disposition_ai.md` | Discharge planning |
| `arxiv_hospital_resource_optimization.md` | Resource allocation |
| `arxiv_workflow_ai.md` | Workflow optimization |
| `arxiv_healthcare_process_mining.md` | Process mining |

---

## Methods & Implementation

### 09_learning_methods/ (18 documents)
Advanced ML techniques.

| Document | Topic |
|----------|-------|
| `arxiv_transfer_learning_clinical.md` | Domain adaptation |
| `arxiv_clinical_transfer_learning.md` | Transfer methods |
| `arxiv_contrastive_learning_medical.md` | Contrastive learning |
| `arxiv_reinforcement_learning_clinical.md` | RL for treatment |
| `arxiv_clinical_rl_simulation.md` | RL simulation |
| `arxiv_clinical_simulation.md` | Clinical simulators |
| `arxiv_causal_inference_ehr.md` | Causal inference |
| `arxiv_healthcare_causal_discovery.md` | Causal discovery |
| `arxiv_counterfactual_clinical.md` | Counterfactual reasoning |
| `arxiv_diffusion_healthcare.md` | Diffusion models |
| `arxiv_clinical_world_models.md` | World models |
| `arxiv_teacher_student_clinical.md` | Knowledge distillation |
| `arxiv_ensemble_clinical.md` | Ensemble methods |
| `arxiv_ssm_clinical.md` | State-space models (Mamba) |
| `arxiv_automl_healthcare.md` | AutoML |
| `arxiv_healthcare_active_learning.md` | Active learning |
| `arxiv_healthcare_continual_learning.md` | Continual learning |
| `arxiv_clinical_decision_theory.md` | Decision theory |

### 10_implementation_deployment/ (12 documents)
Production deployment.

| Document | Topic |
|----------|-------|
| `arxiv_clinical_ai_deployment.md` | Deployment patterns |
| `arxiv_clinical_deployment.md` | Implementation |
| `arxiv_mlops_healthcare.md` | MLOps practices |
| `arxiv_federated_healthcare.md` | Federated learning |
| `arxiv_privacy_preserving_clinical.md` | Privacy methods |
| `arxiv_edge_ai_healthcare.md` | Edge deployment |
| `arxiv_clinical_validation.md` | Validation frameworks |
| `arxiv_clinical_benchmarks.md` | Benchmarking |
| `arxiv_cross_institutional_learning.md` | Multi-site learning |
| `arxiv_realtime_clinical_ai.md` | Real-time systems |
| `arxiv_streaming_realtime_clinical.md` | Streaming inference |
| `arxiv_healthcare_interop.md` | Interoperability |

### 11_interpretability_safety/ (10 documents)
Explainability, fairness, and safety.

| Document | Topic |
|----------|-------|
| `arxiv_explainable_ai_clinical.md` | XAI methods |
| `arxiv_clinical_interpretability.md` | Interpretability |
| `arxiv_uncertainty_medical.md` | Uncertainty quantification |
| `arxiv_clinical_uncertainty_quantification.md` | Calibration |
| `arxiv_healthcare_fairness_equity.md` | Fairness |
| `arxiv_clinical_safety_monitoring.md` | Safety monitoring |
| `arxiv_adverse_events.md` | Adverse event detection |
| `arxiv_human_ai_clinical.md` | Human-AI collaboration |
| `arxiv_healthcare_human_factors.md` | Human factors |
| `arxiv_clinical_decision_thresholds.md` | Decision thresholds |

### 12_data_quality/ (6 documents)
Data handling and quality.

| Document | Topic |
|----------|-------|
| `arxiv_ehr_data_quality.md` | EHR quality |
| `arxiv_clinical_missing_data.md` | Missing data |
| `arxiv_synthetic_clinical.md` | Synthetic data |
| `arxiv_clinical_data_augmentation.md` | Augmentation |
| `arxiv_medical_entity_resolution.md` | Entity resolution |
| `arxiv_clinical_anomaly_detection.md` | Anomaly detection |

### 13_emerging_technology/ (9 documents)
Cutting-edge methods.

| Document | Topic |
|----------|-------|
| `arxiv_foundation_models.md` | Foundation models |
| `arxiv_llm_clinical.md` | Clinical LLMs |
| `arxiv_llm_clinical_reasoning.md` | LLM reasoning |
| `arxiv_llm_agents_healthcare.md` | LLM agents |
| `arxiv_medical_llm_evaluation.md` | LLM evaluation |
| `arxiv_digital_twins_healthcare.md` | Digital twins |
| `arxiv_quantum_healthcare.md` | Quantum computing |
| `arxiv_ambient_ai_healthcare.md` | Ambient AI |
| `arxiv_voice_ai_healthcare.md` | Voice AI |

### 14_agentic_medical_ai/ (1 document)
LLM-based agentic systems for clinical reasoning.

| Document | Key Finding |
|----------|-------------|
| `arxiv_medagent_pro.md` | Hierarchical agentic reasoning: +34% over GPT-4o on glaucoma |

**Key insight:** Agentic systems that mirror clinical workflows (disease-level planning → patient-level reasoning) dramatically outperform both VLMs and existing medical agents. Validates structured reasoning approach.

---

## Document Format

Each research document follows a consistent structure:

1. **Executive Summary** — Key findings in 2-3 paragraphs
2. **Key Papers** — ArXiv IDs with citations
3. **Performance Metrics** — AUROC, F1, accuracy tables
4. **Architectural Details** — Model architectures, training
5. **Clinical Applications** — Real-world use cases
6. **Research Gaps** — Open problems identified
7. **Relevance to Project** — Connection to our goals

---

<p align="center">
  <strong>Last Updated:</strong> December 2025
</p>
