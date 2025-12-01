# Research Gaps Matrix for Hybrid Reasoning in Acute Care

**Analysis Date:** November 30, 2025
**Research Corpus:** 22 documents across clinical AI, neuro-symbolic methods, temporal reasoning, and healthcare standards
**Project:** Hybrid Reasoning for Acute Care Clinical Decision Support

---

## Executive Summary

This matrix identifies critical gaps across five categories based on systematic analysis of the research corpus. The gaps represent opportunities where UCF can make unique contributions by combining neuro-symbolic AI, temporal knowledge graphs, and clinical domain expertise for acute care applications.

**Key Finding:** Despite significant progress in clinical AI, there remains a fundamental disconnect between research capabilities and clinical deployment requirements. The most critical gaps lie at the intersection of real-time temporal reasoning, explainable decision-making, and regulatory compliance for time-critical acute care settings.

---

## 1. METHODOLOGICAL GAPS

### Gap 1.1: Real-Time Temporal Knowledge Graph Reasoning

**Current State:**
- Papers demonstrate temporal KG construction (Paper: arxiv_temporal_kg_2024.md) with RGCN achieving AUC 0.91 on retrospective data
- Time series models handle irregular sampling (Paper: arxiv_time_series_clinical.md) but lack symbolic reasoning layer
- No systems demonstrate sub-second inference on streaming clinical data

**Gap Description:**
Existing temporal KG approaches operate on static/retrospective data with inference times of seconds to minutes. Acute care requires sub-second decision support on continuously updating patient data streams.

**Evidence:**
- DynST model: No real-time deployment metrics reported (arxiv_causal_inference_ehr.md)
- HIT-GNN: Evaluated on historical data only (arxiv_temporal_kg_2024.md)
- All 7 temporal graph papers lack streaming architecture discussion

**Required Research:**
1. Incremental TKG update algorithms maintaining graph consistency
2. Event-driven reasoning triggered by critical pattern detection (e.g., vital sign deterioration)
3. Temporal constraint propagation for real-time inference
4. Memory-efficient graph pruning for continuous operation

**Severity:** HIGH - Fundamental barrier to clinical deployment

**UCF Opportunity:**
- Combine CRCV's real-time video processing expertise with TKG reasoning
- Leverage Dr. Brattain's edge AI for medical devices experience
- Develop streaming TKG update mechanisms for ICU monitoring

**Resources Required:**
- 2 PhD students (CS + Biomedical Engineering)
- Access to MIMIC-IV streaming API or UCF Health partnership
- GPU cluster for real-time inference benchmarking
- 18-24 months development + validation

**Timeline:** 18-24 months to prototype, 36 months to clinical validation

---

### Gap 1.2: Deep Temporal Logic for Clinical Reasoning

**Current State:**
- LNN framework (ibm_lnn_framework.md) supports first-order logic but lacks rich temporal operators
- Temporal papers model sequences but not complex temporal relationships (e.g., Allen's interval algebra)
- No integration of temporal logic with neural temporal models

**Gap Description:**
Clinical guidelines contain complex temporal constraints ("if A occurs within 6 hours of B, then C is required within 1 hour") that current systems cannot natively represent or reason over.

**Evidence:**
- Sepsis Hour-1 Bundle requires precise temporal sequencing (ed_triage_sepsis_protocols.md)
- LNN paper shows only simple sequential rules (ibm_lnn_framework.md, diabetes example)
- None of the neuro-symbolic papers demonstrate Allen interval algebra integration

**Required Research:**
1. Extend LNN with temporal logic operators (Before, During, Overlaps, Meets)
2. Differentiable temporal constraint satisfaction
3. Temporal rule learning from clinical guideline text
4. Hybrid neural-symbolic temporal embeddings

**Severity:** HIGH - Critical for guideline compliance

**UCF Opportunity:**
- Extend IBM LNN framework with temporal operators
- Integrate with Dr. Gurupur's clinical decision support expertise
- Collaborate with Dr. Chen Chen on multi-modal temporal reasoning

**Resources Required:**
- 1-2 PhD students specializing in logic + deep learning
- IBM LNN collaboration/extension
- Clinical guideline corpus (CMS, SSC)
- 12-18 months theoretical + implementation work

**Timeline:** 12-18 months for framework extension, 24 months for clinical validation

---

### Gap 1.3: Uncertainty-Aware Neuro-Symbolic Integration

**Current State:**
- Neuro-symbolic papers report point accuracies (80.52% for LNN diabetes, arxiv_neurosymbolic_healthcare.md)
- Uncertainty quantification papers (arxiv_uncertainty_medical.md context) address neural networks but not symbolic reasoning
- No systematic approach to propagating uncertainty through hybrid architectures

**Gap Description:**
Clinical deployment requires calibrated confidence intervals and explicit uncertainty quantification, especially for hybrid models where neural and symbolic components have different uncertainty characteristics.

**Evidence:**
- LNN diabetes paper: No confidence intervals reported (arxiv_neurosymbolic_healthcare.md)
- Explainable AI review: Uncertainty quantification identified as critical gap (arxiv_explainable_ai_clinical.md)
- Only 1/7 temporal KG papers mentions calibration

**Required Research:**
1. Probabilistic LNN with Bayesian parameter estimation
2. Uncertainty propagation through TKG reasoning
3. Conformal prediction for neuro-symbolic models
4. Calibration methods specific to hybrid architectures

**Severity:** VERY HIGH - Essential for patient safety and FDA approval

**UCF Opportunity:**
- Dr. Ali Siahkoohi's uncertainty quantification expertise (ucf_faculty_profiles.md)
- Combine probabilistic graphical models with neuro-symbolic reasoning
- Novel contribution to neuro-symbolic AI field

**Resources Required:**
- 1 PhD student (statistics/ML background)
- Collaboration with Dr. Siahkoohi
- Clinical validation cohorts for calibration testing
- 18 months theoretical + empirical validation

**Timeline:** 18 months to develop methods, 12 months to validate clinically

---

### Gap 1.4: Multi-Modal Neuro-Symbolic Fusion

**Current State:**
- Multi-modal papers combine imaging + EHR (arxiv_multimodal_clinical.md context, arxiv_llm_clinical.md)
- KG-DG framework integrates vision + knowledge graphs (arxiv_neurosymbolic_healthcare.md)
- No comprehensive neuro-symbolic fusion of imaging, vitals, labs, notes, and biomedical KGs

**Gap Description:**
Acute care decisions integrate diverse data modalities, but current neuro-symbolic systems focus on single or dual modalities. No framework demonstrates comprehensive multi-modal symbolic reasoning.

**Evidence:**
- KG-DG: Vision + lesion ontology only (diabetic retinopathy)
- NeuroSymAD: MRI + clinical features, but symbolic rules are feature-based, not multi-modal
- Targeted-BEHRT: EHR only, no imaging integration

**Required Research:**
1. Cross-modal symbolic rule extraction
2. Multi-modal knowledge graph construction
3. Attention mechanisms guided by symbolic constraints
4. Fusion strategies preserving interpretability

**Severity:** MEDIUM-HIGH - Important for comprehensive clinical reasoning

**UCF Opportunity:**
- Dr. Chen Chen: Multi-modal biomedical AI (BiomedGPT)
- Dr. Mubarak Shah: Vision-language models
- Dr. Brattain: Ultrasound imaging AI
- Unique multi-faculty integration capability

**Resources Required:**
- 2-3 PhD students across vision, NLP, and knowledge representation
- Multi-modal clinical datasets (MIMIC-IV + MIMIC-CXR linkage)
- 24-30 months collaborative development

**Timeline:** 24-30 months for framework development, 12 months for clinical validation

---

## 2. CLINICAL TRANSLATION GAPS

### Gap 2.1: FDA Regulatory Pathway for Neuro-Symbolic CDS

**Current State:**
- FDA guidance addresses AI/ML medical devices (fda_cds_guidance_current.md)
- Time-critical CDS fails Criterion 3 (automation bias concern)
- No established regulatory precedent for neuro-symbolic hybrid systems

**Gap Description:**
Hybrid neuro-symbolic systems don't fit neatly into existing regulatory categories. FDA guidance addresses black-box ML or rule-based systems, but not interpretable learned logic.

**Evidence:**
- FDA four criteria: Time-critical CDS doesn't qualify for non-device exemption
- GMLP principles: Designed for black-box neural networks
- PCCP (Predetermined Change Control Plans): Focus on continuous learning, not hybrid architectures
- No approved neuro-symbolic medical devices identified

**Required Research:**
1. Regulatory science framework for interpretable AI
2. Validation protocols demonstrating both neural learning and logical soundness
3. Change management for systems with learned symbolic rules
4. Post-market surveillance specific to hybrid systems

**Severity:** VERY HIGH - Blocks clinical deployment path

**UCF Opportunity:**
- Dr. Gurupur: Healthcare IT regulatory experience
- Partner with FDA through Orlando Health
- First-mover advantage in regulatory framework development
- Potential OHDSI network validation

**Resources Required:**
- Regulatory consultant (former FDA reviewer)
- Clinical trial design expertise
- Multi-site validation study infrastructure
- 24-36 months regulatory pathway development

**Timeline:** 24-36 months to establish pathway with FDA engagement

---

### Gap 2.2: Clinician Trust and Adoption of Hybrid AI

**Current State:**
- XAI research shows mixed clinician trust (arxiv_explainable_ai_clinical.md)
- LLM co-pilot mode doubles accuracy over autonomous (arxiv_llm_clinical.md: 54.1% vs 31.1%)
- Small sample sizes limit generalizability (median: 16 clinicians)

**Gap Description:**
Despite theoretical interpretability advantages, no studies demonstrate that clinicians trust neuro-symbolic systems more than pure neural networks or understand hybrid reasoning traces.

**Evidence:**
- LNN diabetes paper: No clinician evaluation reported
- XAI review: Attention mechanisms ≠ clinical explanations
- Only 9/31 XAI studies had ≥30 participants
- Gap: "What XAI provides ≠ What clinicians need"

**Required Research:**
1. Clinician-centered evaluation of neuro-symbolic explanations
2. Think-aloud protocols comparing hybrid vs black-box explanations
3. Trust calibration studies (appropriate vs over/under-reliance)
4. Workflow integration feasibility studies

**Severity:** HIGH - Adoption barrier regardless of technical performance

**UCF Opportunity:**
- School of Global Health Management & Informatics partnerships (700+ healthcare orgs)
- Dr. Brattain's Orlando Health collaborations
- Dr. Zraick: Health communication expertise
- HCI evaluation capabilities at UCF

**Resources Required:**
- Mixed-methods research team (HCI + clinical informatics)
- Orlando Health partnership for clinician access
- 40-60 clinician participants (across specialties)
- 18-24 months for comprehensive evaluation

**Timeline:** 18-24 months from prototype to clinician evaluation completion

---

### Gap 2.3: EHR Integration Architecture for Hybrid Reasoning

**Current State:**
- FHIR enables standardized data exchange (fhir_clinical_standards.md)
- OMOP CDM supports network research (ohdsi_omop_cdm.md)
- No reference architecture for deploying neuro-symbolic models via FHIR APIs

**Gap Description:**
Theoretical frameworks exist for FHIR + ML, but no practical architecture demonstrates real-time neuro-symbolic reasoning integrated with EHR workflows, handling both FHIR data ingestion and symbolic inference output.

**Evidence:**
- FHIR-ML review: 18 CDSS systems, but none use neuro-symbolic methods
- Challenges: "Limited EHR interoperability, nonstandardized formats"
- No papers describe FHIR → TKG → LNN → FHIR prediction pipeline
- Missing: Temporal FHIR extensions for continuous reasoning

**Required Research:**
1. FHIR to temporal knowledge graph ETL pipeline
2. Real-time FHIR resource streaming to incremental TKG updates
3. Symbolic reasoning output as FHIR Observations with provenance
4. Event-driven architecture triggering inference on critical FHIR updates

**Severity:** HIGH - Essential for health system deployment

**UCF Opportunity:**
- Dr. Gurupur: Center for Decision Support Systems & Informatics
- FHIR expertise + EHR integration experience
- Partner with UCF Health or Orlando Health EHR systems
- Potential open-source contribution (OHDSI community)

**Resources Required:**
- Software engineering team (FHIR + graph databases)
- EHR sandbox environment (Epic/Cerner test instance)
- HL7 FHIR certification
- 12-18 months architectural development + testing

**Timeline:** 12-18 months for architecture + prototype, 12 months for pilot deployment

---

### Gap 2.4: Prospective Clinical Trial Design for Hybrid AI

**Current State:**
- All neuro-symbolic healthcare papers use retrospective data
- LLM clinical paper: Quality improvement study, not RCT (Penda Health)
- Recommendation: "Prospective validation whenever possible" but rarely done

**Gap Description:**
No randomized controlled trials (RCTs) evaluate neuro-symbolic clinical decision support in prospective acute care settings. All evidence is retrospective or observational.

**Evidence:**
- 0/60+ neuro-symbolic papers report prospective RCT
- 0/7 temporal KG papers have prospective validation
- Gap explicitly noted in multiple reviews
- Challenge: High cost, regulatory barriers, patient safety concerns

**Required Research:**
1. RCT design for hybrid AI decision support in acute care
2. Endpoints: Patient outcomes (mortality, LOS) vs process metrics
3. Allocation strategies (cluster randomization at ICU level)
4. Sample size calculations accounting for hybrid model uncertainty
5. Safety monitoring and stopping rules

**Severity:** VERY HIGH - Required for evidence-based adoption

**UCF Opportunity:**
- Partner with Orlando Health for multi-ICU trial
- Leverage UCF College of Medicine clinical trial infrastructure
- Dr. Brattain's clinical research experience
- NSF Smart Health funding for clinical trials (NSF 25-542)

**Resources Required:**
- Clinical trial coordinator
- Biostatistician for trial design
- IRB approval + patient consent infrastructure
- $500K-$1M budget (depending on scope)
- 36-48 months (design, recruit, follow-up, analysis)

**Timeline:** 12 months design + IRB, 24-36 months trial execution + analysis

---

## 3. DATA GAPS

### Gap 3.1: Annotated Acute Care Temporal Trajectories

**Current State:**
- MIMIC-IV: 73,141 ICU stays, rich but not trajectory-labeled (mimic_iv_dataset_details.md)
- MIMIC-Sepsis: 35,239 ICU stays with Sepsis-3 criteria
- No dataset with expert-annotated temporal reasoning paths

**Gap Description:**
Training and validating neuro-symbolic temporal reasoning requires datasets with explicit temporal relationship annotations (e.g., "deterioration began 4 hours before sepsis diagnosis, triggered by rising lactate at 2 hours").

**Evidence:**
- MIMIC-IV: Timestamps exist but temporal relationships not annotated
- Temporal KG papers infer relationships algorithmically, not ground truth
- No gold-standard dataset for clinical temporal logic validation
- Existing datasets: Static snapshots or raw time series, not structured temporal knowledge

**Required Research:**
1. Annotation schema for clinical temporal relationships (Allen algebra + clinical semantics)
2. Expert annotation workflow (clinician + informaticist collaboration)
3. Inter-rater reliability measurement for temporal annotations
4. Minimum viable dataset size for temporal reasoning validation

**Severity:** HIGH - Limits supervised learning and validation

**UCF Opportunity:**
- MIMIC-IV access + UCF Health data for annotation
- Dr. Gurupur: Clinical data quality expertise
- Medical student annotators (supervised by Dr. Brattain)
- Contribution to OHDSI/PhysioNet community

**Resources Required:**
- 2-3 clinical expert annotators (physicians)
- Annotation platform development/customization
- 500-1000 ICU stay annotations (estimated 200 hours clinician time)
- $50K-$75K for annotation effort
- 12-18 months

**Timeline:** 12-18 months to create, validate, and publish dataset

---

### Gap 3.2: Multi-Site Acute Care Data with Consistent Semantics

**Current State:**
- MIMIC-IV: Beth Israel Deaconess only
- eICU-CRD: 200+ hospitals but heterogeneous data collection
- No federated acute care dataset with harmonized sepsis phenotypes

**Gap Description:**
Validating generalizability across healthcare systems requires multi-site data with consistent clinical phenotype definitions. Existing datasets either single-site or semantically heterogeneous.

**Evidence:**
- MIMIC vs eICU: Different sampling frequencies, missingness patterns
- Federated Markov Imputation paper: Different ICUs use 1h/2h/3h intervals
- Domain adaptation papers confirm cross-ICU distribution shift
- OMOP CDM adoption incomplete for acute care temporal data

**Required Research:**
1. Standardized acute care phenotype definitions (via OMOP + FHIR)
2. Federated data network with temporal harmonization
3. Cross-site validation protocols for neuro-symbolic models
4. Privacy-preserving methods for multi-site temporal reasoning

**Severity:** MEDIUM-HIGH - Needed for external validation and generalization

**UCF Opportunity:**
- OHDSI network participation (100+ databases, 20+ countries)
- Partner with multiple Florida health systems
- Federated learning expertise (emerging at UCF)
- FHIR-DHP pipeline adaptation

**Resources Required:**
- Data use agreements with 3-5 health systems
- OMOP CDM implementation support
- Privacy-preserving infrastructure
- 18-24 months to establish network

**Timeline:** 18-24 months for network establishment, ongoing for research

---

### Gap 3.3: Ground Truth for Symbolic Rule Validation

**Current State:**
- Clinical guidelines exist (CMS SEP-1, Surviving Sepsis Campaign)
- LNN learns rule weights but no gold standard for "correct" weights
- No benchmark dataset for clinical logic validation

**Gap Description:**
To validate that learned symbolic rules align with medical knowledge, we need curated test cases with expert consensus on correct logical reasoning paths—currently unavailable.

**Evidence:**
- LNN diabetes paper: Compares to baselines but not clinical gold standard
- No "clinical logic benchmark" analogous to CLEVR for visual reasoning
- Clinical guidelines provide rules but not ground truth inferences
- Expert disagreement on edge cases not quantified

**Required Research:**
1. Clinical reasoning test suite (similar to medical board exam cases)
2. Multi-expert annotation of logical inference paths
3. Consensus methods for resolving expert disagreement
4. Benchmark metrics for symbolic rule quality (beyond accuracy)

**Severity:** MEDIUM - Important for validating symbolic component

**UCF Opportunity:**
- Medical education collaboration (UCF College of Medicine)
- USMLE-style case development for logic validation
- Partnership with medical board organizations
- Novel contribution to neuro-symbolic AI evaluation

**Resources Required:**
- 5-10 physician experts across specialties
- Case development + annotation platform
- 100-200 reasoning scenarios
- $40K-$60K for expert time
- 12 months

**Timeline:** 12 months to develop and validate benchmark

---

### Gap 3.4: Real-Time Streaming Data Infrastructure

**Current State:**
- MIMIC-IV: Static dataset, no streaming API
- Research uses retrospective replay, not true real-time
- No publicly available streaming acute care data

**Gap Description:**
Developing real-time reasoning systems requires streaming data infrastructure to simulate ICU data flows. Current datasets are static archives.

**Evidence:**
- All temporal papers use batch processing on historical data
- No papers report latency benchmarks on streaming data
- Real-time requirement: Sub-second inference on continuous vitals
- Gap: Static datasets → models optimized for batch, not stream

**Required Research:**
- Not research gap, but infrastructure gap requiring:
  1. Streaming MIMIC-IV replay infrastructure
  2. Kafka/Kinesis data pipelines
  3. Benchmark for streaming inference latency
  4. Simulated ICU environment for testing

**Severity:** MEDIUM - Infrastructure enabler for real-time methods

**UCF Opportunity:**
- Build open-source MIMIC-IV streaming simulator
- Contribution to PhysioNet community
- UCF's engineering + CS collaboration

**Resources Required:**
- 1 software engineer (data engineering background)
- Cloud infrastructure (AWS/GCP credits for streaming)
- 6-9 months development
- $30K-$50K

**Timeline:** 6-9 months to build, validate, and open-source

---

## 4. EVALUATION GAPS

### Gap 4.1: Clinically Meaningful Metrics for Neuro-Symbolic Models

**Current State:**
- Papers report AUROC, accuracy, F1 (technical metrics)
- Clinical utility metrics rare (NNT, decision curve analysis)
- No metrics for symbolic reasoning quality

**Gap Description:**
Standard ML metrics don't capture clinical value or assess quality of learned symbolic rules. Need metrics that evaluate both neural performance AND symbolic interpretability.

**Evidence:**
- LNN: Reports accuracy but not rule plausibility
- Penda Health: 16% reduction in errors (NNT: 18.1) - rare example of clinical metric
- Explainable AI review: "What XAI provides ≠ what clinicians need"
- Missing: Metrics for rule consistency with medical knowledge

**Required Research:**
1. Rule plausibility scoring (alignment with clinical guidelines)
2. Symbolic reasoning evaluation metrics
3. Clinical utility metrics (net benefit, NNT, cost-effectiveness)
4. Combined metrics balancing neural accuracy + symbolic quality

**Severity:** HIGH - Misaligned evaluation → clinically irrelevant models

**UCF Opportunity:**
- Dr. Gurupur: Clinical decision support evaluation expertise
- Clinician partnership for metric validation
- Novel contribution to neuro-symbolic AI evaluation

**Resources Required:**
- Clinical outcomes research expertise
- Retrospective cohort for decision curve analysis
- Expert panel for rule plausibility validation
- 9-12 months

**Timeline:** 9-12 months to develop, validate, and publish metrics

---

### Gap 4.2: Fairness and Bias Auditing for Hybrid Models

**Current State:**
- Fairness in ML: Well-studied for neural networks
- Neuro-symbolic fairness: Understudied
- No frameworks for auditing bias in learned logical rules

**Gap Description:**
Hybrid models can encode bias in both neural components (learned from data) and symbolic components (learned rules or human-defined rules reflecting historical biases). No comprehensive fairness framework exists.

**Evidence:**
- Neuro-symbolic papers: No fairness analysis
- LNN diabetes paper: No subgroup analysis by demographics
- Explainable AI gap: "Bias detection through symbolic rule inspection" noted as opportunity
- Clinical AI gap: Racial/ethnic biases in prediction algorithms identified as concern (NSF SCH)

**Required Research:**
1. Fairness metrics for symbolic rules (e.g., "high glucose threshold" varies by population)
2. Bias propagation through neuro-symbolic pipelines
3. Debiasing methods preserving logical soundness
4. Subgroup performance analysis for learned rules

**Severity:** VERY HIGH - Ethical imperative and regulatory requirement

**UCF Opportunity:**
- Health equity focus aligns with NSF priorities
- Dr. Gurupur: Healthcare disparities research
- MIMIC-IV demographics for fairness analysis
- Novel contribution to trustworthy AI

**Resources Required:**
- Fairness researcher (CS or biostatistics)
- Diverse patient cohorts for analysis
- Ethics consultation
- 12-18 months

**Timeline:** 12-18 months to develop framework and conduct audits

---

### Gap 4.3: Long-Term Deployment Performance Monitoring

**Current State:**
- Papers evaluate on held-out test sets (snapshot)
- No longitudinal studies of deployed neuro-symbolic systems
- Model drift not addressed for hybrid architectures

**Gap Description:**
Clinical AI performance degrades over time due to distribution shift, but no methods exist for monitoring and maintaining neuro-symbolic systems in production, particularly detecting when learned rules become outdated.

**Evidence:**
- All papers: Static evaluation, no deployment monitoring
- Model drift: Known problem for neural networks, unexplored for hybrid systems
- Gap: How to detect when symbolic rules no longer valid?
- Missing: Continuous validation frameworks for hybrid AI

**Required Research:**
1. Drift detection for symbolic rules (not just neural weights)
2. Continuous calibration monitoring
3. Automated rule update vs retrain decision framework
4. A/B testing protocols for hybrid model updates

**Severity:** HIGH - Essential for sustained clinical deployment

**UCF Opportunity:**
- Partner with Orlando Health for deployment monitoring
- Software engineering + clinical informatics collaboration
- Potential contribution to FDA PCCP framework for hybrid AI

**Resources Required:**
- Deployment platform access (production EHR environment)
- Monitoring infrastructure
- 2-3 years of longitudinal data
- $75K-$100K for infrastructure + analysis

**Timeline:** 24-36 months minimum for meaningful longitudinal study

---

### Gap 4.4: Interpretability Benchmarks for Hybrid Reasoning

**Current State:**
- Attention weights, SHAP values used for neural interpretability
- Rule inspection used for symbolic interpretability
- No standardized benchmarks for hybrid system interpretability

**Gap Description:**
Cannot systematically compare interpretability across neuro-symbolic architectures or validate that hybrid explanations are more useful than pure neural explanations.

**Evidence:**
- Explainable AI review: Inconsistent evaluation methods across studies
- No "interpretability benchmark" for clinical AI
- Claims of neuro-symbolic interpretability advantages lack empirical validation
- Expert evaluation studies: Small samples, non-standardized protocols

**Required Research:**
1. Standardized interpretability evaluation protocol for hybrid AI
2. Human-subject studies with sufficient power (N≥30 clinicians)
3. Comparative evaluation: hybrid vs black-box vs rule-based
4. Task-based evaluation (not just subjective ratings)

**Severity:** MEDIUM-HIGH - Important for demonstrating neuro-symbolic value

**UCF Opportunity:**
- Dr. Rawat: Trustworthy AI research
- HCI capabilities at UCF
- Multi-clinician access via health system partnerships
- Potential NSF funding (Smart Health, Trustworthy AI)

**Resources Required:**
- Human subjects research team
- 40-60 clinician participants
- Experimental platform development
- $60K-$80K for participant compensation + platform
- 18-24 months

**Timeline:** 18-24 months from protocol development to publication

---

## 5. UCF OPPORTUNITY GAPS

### Gap 5.1: Neuro-Symbolic Framework for Emergency Ultrasound

**Current State:**
- Dr. Brattain: AI for medical ultrasound, surgical robotics
- No neuro-symbolic approaches for ultrasound interpretation
- Opportunity: Combine imaging AI with clinical decision rules

**Gap Description:**
Emergency ultrasound (FAST exam, lung ultrasound, cardiac) requires rapid interpretation with clinical context integration. Neuro-symbolic approach could combine image analysis with patient history, vitals, and clinical rules.

**Evidence:**
- Dr. Brattain publications: Transfer learning for COVID B-lines, but pure neural approach
- No papers combine ultrasound CNNs with clinical logic networks
- Unique UCF strength: Dr. Brattain (ultrasound) + Dr. Chen Chen (medical imaging) + neuro-symbolic expertise

**Required Research:**
1. Vision Transformer + LNN integration for ultrasound
2. Clinical guideline encoding (ACEP ultrasound guidelines)
3. Multi-modal fusion: ultrasound + vitals + history → diagnosis
4. Real-time inference on portable ultrasound devices (edge AI)

**Severity:** N/A (Opportunity, not gap) - HIGH IMPACT potential

**UCF Opportunity:**
- **Dr. Brattain (Lead)**: Ultrasound AI, Orlando Health partnerships, edge AI
- **Dr. Chen Chen**: Medical imaging, multi-modal AI (BiomedGPT)
- **Dr. Gurupur**: Clinical decision support frameworks
- **Unique Differentiator**: Only team with ultrasound + neuro-symbolic + clinical expertise

**Resources Required:**
- Ultrasound video dataset (Orlando Health collaboration)
- Portable ultrasound devices for edge deployment
- Clinical guideline formalization (ACEP, AIUM)
- 2-3 PhD students
- $300K-$500K (NSF Smart Health proposal target)

**Timeline:**
- Months 1-6: Dataset collection + guideline encoding
- Months 7-18: Model development + integration
- Months 19-30: Clinical validation at Orlando Health
- Months 31-36: Prospective pilot study

**Funding Target:** NSF Smart Health ($1.2M/4 years), NIH R01

---

### Gap 5.2: Multi-Agent Neuro-Symbolic ICU Monitoring

**Current State:**
- Dr. Sukthankar: Multi-agent systems, plan recognition
- Multi-agent LLM systems outperform single-agent (KTAS study)
- No neuro-symbolic multi-agent clinical systems

**Gap Description:**
ICU teams involve multiple specialists (intensivist, cardiologist, nephrologist) with distributed decision-making. Multi-agent neuro-symbolic system could model collaborative reasoning with each agent representing specialist knowledge domain.

**Evidence:**
- Korean KTAS study: Multi-agent (triage nurse, ED physician, pharmacist, ED director) perfect scores
- Dr. Sukthankar: 150+ publications on multi-agent systems, no healthcare applications
- Unique opportunity: Combine UCF multi-agent expertise with clinical AI

**Required Research:**
1. Multi-agent LNN architecture (organ system specialists as agents)
2. Agent communication protocols using symbolic messages
3. Distributed reasoning with consistency maintenance
4. Explanation generation from multi-agent deliberation

**Severity:** N/A (Opportunity) - MEDIUM-HIGH IMPACT

**UCF Opportunity:**
- **Dr. Sukthankar (Lead)**: Multi-agent systems, plan recognition
- **Dr. Chen Chen**: Medical AI
- **Dr. Brattain**: Clinical context
- **Unique Differentiator**: Multi-agent + neuro-symbolic + clinical domain

**Resources Required:**
- Multi-agent framework development
- Clinical workflow analysis (ICU team interactions)
- 1-2 PhD students
- MIMIC-IV or UCF Health data
- $200K-$300K

**Timeline:**
- Months 1-12: Framework development
- Months 13-24: ICU workflow integration
- Months 25-36: Validation study

**Funding Target:** NSF Smart Health, DARPA (AI-assisted healthcare)

---

### Gap 5.3: Hybrid Reasoning for Patient Activity Monitoring

**Current State:**
- Dr. Rawat: Video action recognition under occlusion, privacy-preserving AI
- Patient fall detection, activity of daily living (ADL) assessment needs
- No neuro-symbolic activity monitoring systems

**Gap Description:**
Patient activity monitoring (fall detection, mobility assessment, delirium detection) requires both visual pattern recognition and clinical rule application. Privacy-preserving neuro-symbolic approach could satisfy both performance and privacy requirements.

**Evidence:**
- Dr. Rawat: Privacy-preserving surveillance (NSF-funded), action recognition expertise
- UCF patent: "Self-Supervised Privacy Preservation Action Recognition System"
- Clinical need: Fall detection (joint commission safety goal), delirium screening
- No papers combine privacy-preserving vision with clinical logic

**Required Research:**
1. Privacy-preserving vision transformers + LNN clinical rules
2. Activity recognition → clinical risk assessment pipeline
3. Temporal logic for fall risk (posture changes, gait patterns)
4. Edge deployment for real-time monitoring

**Severity:** N/A (Opportunity) - MEDIUM-HIGH IMPACT

**UCF Opportunity:**
- **Dr. Rawat (Lead)**: Action recognition, privacy-preserving AI
- **Dr. Brattain**: Clinical monitoring context
- **Dr. Sukthankar**: Activity recognition
- **Unique Differentiator**: Privacy + neuro-symbolic + clinical safety

**Resources Required:**
- Video dataset (simulated falls, delirium behaviors)
- UCF Health or nursing home partnership
- Privacy evaluation framework
- 1-2 PhD students
- $250K-$350K

**Timeline:**
- Months 1-9: Privacy-preserving vision + LNN integration
- Months 10-24: Clinical rule encoding + validation
- Months 25-36: Pilot deployment study

**Funding Target:** NSF Smart Health, NIH NIA (aging), AHRQ (patient safety)

---

### Gap 5.4: Digital Twin with Neuro-Symbolic Reasoning

**Current State:**
- Dr. Xishun Liao: AI-powered digital twins, human-centered intelligent systems
- NSF Award #2123900: "Causal AI Digital Twin Framework for Critical Care"
- No neuro-symbolic digital twin implementations

**Gap Description:**
Patient digital twins simulate physiological responses to interventions. Neuro-symbolic approach could combine mechanistic models (symbolic) with data-driven learning (neural) for personalized medicine.

**Evidence:**
- Digital twin papers: Pure simulation or pure ML, not hybrid
- Opportunity: Combine physiological equations (symbolic) with patient-specific learning (neural)
- Dr. Liao expertise aligns with emerging NSF digital twin focus

**Required Research:**
1. Physiological model encoding as symbolic constraints (cardiac output equations, fluid balance)
2. Neural component learns patient-specific deviations from population model
3. Counterfactual reasoning: "What if we increase vasopressor dose?"
4. Temporal prediction: Patient trajectory under different treatment plans

**Severity:** N/A (Opportunity) - MEDIUM IMPACT (longer-term)

**UCF Opportunity:**
- **Dr. Liao (Lead)**: Digital twins, control theory + AI
- **Dr. Kassab**: Cardiovascular modeling
- **Dr. Brattain**: Clinical validation
- **Unique Differentiator**: Digital twin + neuro-symbolic + personalized medicine

**Resources Required:**
- Physiological modeling expertise
- Longitudinal ICU data (treatment sequences + outcomes)
- Simulation environment
- 2 PhD students
- $300K-$400K

**Timeline:**
- Months 1-12: Physiological model symbolic encoding
- Months 13-30: Patient-specific neural learning
- Months 31-48: Clinical validation + counterfactual evaluation

**Funding Target:** NSF (Digital Twin program when announced), NIH R01

---

### Gap 5.5: Federated Neuro-Symbolic Learning Across Florida Health Systems

**Current State:**
- Dr. Gurupur: 700+ healthcare organization affiliations via SGHMI
- Federated learning emerging (arxiv_federated_healthcare.md context)
- No federated neuro-symbolic approaches

**Gap Description:**
Multi-site validation requires federated learning to preserve patient privacy. Federated neuro-symbolic learning could share symbolic rules while keeping patient data local.

**Evidence:**
- Gap identified: "Cross-hospital transfer with domain adaptation"
- Privacy-preserving methods needed for multi-site temporal reasoning
- Dr. Gurupur's network provides unique access to Florida health systems
- Symbolic rules more privacy-preserving than raw neural weights

**Required Research:**
1. Federated LNN training (share rule weights, not patient data)
2. Symbolic rule aggregation across institutions
3. Local TKG construction + federated reasoning
4. Privacy-preserving temporal pattern detection

**Severity:** N/A (Opportunity) - MEDIUM-HIGH IMPACT

**UCF Opportunity:**
- **Dr. Gurupur (Lead)**: Healthcare network, decision support expertise
- **SGHMI Infrastructure**: 700+ healthcare partnerships
- **Dr. Chen Chen**: Federated learning for medical imaging
- **Unique Differentiator**: Florida-wide health network + neuro-symbolic

**Resources Required:**
- Data use agreements with 3-5 Florida health systems
- Federated infrastructure (secure multi-party computation)
- IRB approvals across sites
- 1-2 PhD students + software engineer
- $250K-$400K

**Timeline:**
- Months 1-12: Federated infrastructure + agreements
- Months 13-30: Federated neuro-symbolic training
- Months 31-42: Multi-site validation

**Funding Target:** NSF Smart Health, NIH (multi-site research), Florida Dept of Health

---

## Summary Table: Gaps by Severity and Feasibility

| Gap ID | Category | Severity | UCF Feasibility | Funding Likelihood | Timeline | Priority Score |
|--------|----------|----------|-----------------|-------------------|----------|----------------|
| 1.1 | Methodology | HIGH | Medium | High (NSF) | 18-24 mo | 9/10 |
| 1.2 | Methodology | HIGH | Medium | Medium (NSF) | 12-18 mo | 8/10 |
| 1.3 | Methodology | VERY HIGH | High | High (NSF, NIH) | 18 mo | 10/10 |
| 1.4 | Methodology | MEDIUM-HIGH | Very High | High (NSF) | 24-30 mo | 9/10 |
| 2.1 | Translation | VERY HIGH | Medium | Medium (FDA collab) | 24-36 mo | 7/10 |
| 2.2 | Translation | HIGH | High | High (NSF) | 18-24 mo | 8/10 |
| 2.3 | Translation | HIGH | High | Medium (NIH) | 12-18 mo | 8/10 |
| 2.4 | Translation | VERY HIGH | Medium | High (NSF SCH) | 36-48 mo | 8/10 |
| 3.1 | Data | HIGH | High | Medium (PhysioNet) | 12-18 mo | 7/10 |
| 3.2 | Data | MEDIUM-HIGH | Medium | Medium (OHDSI) | 18-24 mo | 6/10 |
| 3.3 | Data | MEDIUM | Medium | Low | 12 mo | 5/10 |
| 3.4 | Data | MEDIUM | High | Medium | 6-9 mo | 6/10 |
| 4.1 | Evaluation | HIGH | High | High (NSF) | 9-12 mo | 8/10 |
| 4.2 | Evaluation | VERY HIGH | High | High (NSF, NIH) | 12-18 mo | 9/10 |
| 4.3 | Evaluation | HIGH | Medium | Medium | 24-36 mo | 6/10 |
| 4.4 | Evaluation | MEDIUM-HIGH | High | High (NSF) | 18-24 mo | 7/10 |
| 5.1 | UCF Opportunity | HIGH | Very High | Very High | 36 mo | 10/10 |
| 5.2 | UCF Opportunity | MEDIUM-HIGH | High | High | 36 mo | 8/10 |
| 5.3 | UCF Opportunity | MEDIUM-HIGH | High | High | 36 mo | 8/10 |
| 5.4 | UCF Opportunity | MEDIUM | Medium | Medium | 48 mo | 6/10 |
| 5.5 | UCF Opportunity | MEDIUM-HIGH | Very High | High | 42 mo | 9/10 |

**Priority Score Formula:** (Severity × 0.4) + (UCF Feasibility × 0.3) + (Funding Likelihood × 0.2) + (1/Timeline × 0.1), normalized to 10

---

## Recommended Research Roadmap

### Phase 1: Foundation (Years 1-2)

**Focus:** Core methodological gaps + initial UCF opportunity exploration

**Priority Projects:**
1. **Gap 1.3:** Uncertainty-aware neuro-symbolic integration (Dr. Siahkoohi lead) - CRITICAL
2. **Gap 2.3:** EHR integration architecture (Dr. Gurupur lead) - ENABLING
3. **Gap 3.1:** Annotated temporal trajectories (Multi-faculty) - DATA FOUNDATION
4. **Gap 5.1:** Emergency ultrasound pilot (Dr. Brattain lead) - UCF DIFFERENTIATOR

**Deliverables:**
- Probabilistic LNN framework with calibrated uncertainty
- FHIR → TKG → LNN → FHIR reference architecture
- 500 annotated ICU temporal trajectories (MIMIC-IV subset)
- Proof-of-concept ultrasound neuro-symbolic system

**Funding Targets:** NSF Smart Health ($1.2M), NIH R21 exploratory ($275K), UCF seed funding

---

### Phase 2: Integration & Validation (Years 2-4)

**Focus:** Clinical translation gaps + multi-modal integration

**Priority Projects:**
1. **Gap 1.1:** Real-time temporal reasoning (Building on Phase 1 architecture)
2. **Gap 1.4:** Multi-modal neuro-symbolic fusion (Dr. Chen Chen + Dr. Brattain)
3. **Gap 2.2:** Clinician trust evaluation (SGHMI partnerships)
4. **Gap 4.2:** Fairness and bias auditing (Health equity focus)

**Deliverables:**
- Streaming TKG with sub-second inference
- Multi-modal (imaging + EHR + biomedical KG) neuro-symbolic framework
- Clinician evaluation study (N=50) with trust calibration results
- Fairness audit framework + demographic subgroup analysis

**Funding Targets:** NSF CAREER ($500K/5yr), NIH R01 ($1.5M/5yr), AHRQ ($500K)

---

### Phase 3: Clinical Deployment (Years 4-6)

**Focus:** Prospective trials + regulatory pathway + scale-out

**Priority Projects:**
1. **Gap 2.4:** Prospective RCT (Orlando Health multi-ICU)
2. **Gap 2.1:** FDA regulatory pathway development
3. **Gap 5.5:** Federated learning across Florida health systems
4. **Gap 4.3:** Long-term deployment monitoring

**Deliverables:**
- Completed RCT with patient outcome data
- FDA 510(k) submission or De Novo pathway application
- Federated neuro-symbolic framework deployed across 3-5 sites
- 2-year deployment monitoring report

**Funding Targets:** NIH R01 renewal/supplement, CDC/AHRQ implementation science, Florida Dept of Health, Commercial partnerships

---

## Conclusion

This analysis identifies **20 critical gaps** across methodology, clinical translation, data, evaluation, and UCF-specific opportunities. The most urgent gaps center on:

1. **Uncertainty quantification for hybrid models** (Gap 1.3) - Essential for patient safety
2. **Real-time temporal reasoning** (Gap 1.1) - Core capability for acute care
3. **Prospective clinical validation** (Gap 2.4) - Required for evidence-based adoption
4. **Fairness and bias auditing** (Gap 4.2) - Ethical and regulatory imperative

**UCF's unique positioning** lies at the intersection of:
- Medical imaging AI (Dr. Brattain ultrasound, Dr. Chen Chen multi-modal)
- Clinical decision support infrastructure (Dr. Gurupur, SGHMI partnerships)
- Advanced AI methods (neuro-symbolic, multi-agent, privacy-preserving)
- Clinical access (Orlando Health, UCF Health, 700+ healthcare affiliations)

The **recommended strategy** is a phased approach starting with foundational methodological work (uncertainty quantification, EHR integration) while simultaneously piloting a high-impact UCF differentiator (emergency ultrasound neuro-symbolic system). This balances scientific rigor with demonstrable clinical value, positioning UCF for sustained NSF Smart Health and NIH funding while creating a pathway to FDA-approved clinical deployment.

**Total estimated funding target:** $3-5M over 6 years across multiple awards (NSF Smart Health, NIH R01/R21, AHRQ, state/local partnerships).

---

**Document Prepared By:** Research synthesis across 22 documents
**Date:** November 30, 2025
**Next Review:** Quarterly updates as new research emerges or funding opportunities arise
**Contact:** Hybrid Reasoning for Acute Care Research Team
