# KG4Diagnosis: A Hierarchical Multi-Agent LLM System with Knowledge Graph Enhancement for Medical Diagnosis

**Source:** arXiv:2412.16833 [cs.AI] December 2024
**Authors:** Kaiwen Zuo et al.
**Code:** https://github.com/traveler-leon/KG4Diagnosis

---

## Executive Summary

KG4Diagnosis introduces a **hierarchical multi-agent framework** integrating a **Neo4j knowledge graph** with specialized LLM agents for comprehensive medical diagnosis. The system models **362 diseases** across 12 body systems with **8,409 relations** connecting symptoms, examinations, and treatments.

**Key Innovation:** Two-tier agent architecture where a GP-LLM (General Practitioner) agent routes patients to specialized Consultant-LLM agents based on initial assessment and knowledge graph lookup.

---

## Architecture

### Hierarchical Multi-Agent Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      PATIENT INPUT                               │
│              (Chief complaint, symptoms, history)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GP-LLM AGENT                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ Initial     │───▶│ KG Query    │───▶│ Department  │          │
│  │ Assessment  │    │ (Neo4j)     │    │ Routing     │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CONSULTANT-LLM AGENTS (12)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │Cardiology│ │Neurology │ │ Pulmonary│ │Nephrology│  ...       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
│                                                                   │
│  Each agent:                                                      │
│  • Queries department-specific KG subgraph                       │
│  • Applies specialist diagnostic reasoning                        │
│  • Recommends examinations and treatments                        │
└─────────────────────────────────────────────────────────────────┘
```

### Knowledge Graph Schema

```
Nodes:
├── Disease (362 total)
├── Symptom
├── Examination
├── Treatment
└── Department (12 body systems)

Relations (8,409 total):
├── Disease → HAS_SYMPTOM → Symptom
├── Disease → REQUIRES_EXAM → Examination
├── Disease → TREATED_BY → Treatment
├── Disease → BELONGS_TO → Department
└── Symptom → SUGGESTS → Disease
```

### 12 Medical Departments

| Department | Disease Count | Example Conditions |
|------------|---------------|-------------------|
| Cardiology | 35 | MI, arrhythmia, heart failure |
| Neurology | 42 | Stroke, epilepsy, Parkinson's |
| Pulmonology | 31 | Pneumonia, COPD, asthma |
| Gastroenterology | 38 | GERD, hepatitis, IBD |
| Nephrology | 22 | CKD, AKI, glomerulonephritis |
| Endocrinology | 28 | Diabetes, thyroid disorders |
| Hematology | 24 | Anemia, leukemia, coagulopathy |
| Rheumatology | 31 | RA, lupus, gout |
| Infectious Disease | 45 | Sepsis, tuberculosis, HIV |
| Oncology | 33 | Various cancers |
| Dermatology | 18 | Psoriasis, eczema |
| Psychiatry | 15 | Depression, schizophrenia |

---

## Performance Results

### Diagnostic Accuracy by Department

| Department | GPT-4 Baseline | KG4Diagnosis | Improvement |
|------------|----------------|--------------|-------------|
| Cardiology | 72.3% | **84.1%** | +11.8% |
| Neurology | 68.5% | **79.2%** | +10.7% |
| Pulmonology | 74.1% | **85.6%** | +11.5% |
| Infectious | 71.8% | **83.4%** | +11.6% |
| **Average** | 71.2% | **82.8%** | **+11.6%** |

### Multi-Agent vs Single-Agent

| Configuration | Accuracy | Reasoning Quality |
|--------------|----------|-------------------|
| Single LLM (GPT-4) | 71.2% | Limited depth |
| Single LLM + KG | 76.4% | Better coverage |
| Multi-Agent no KG | 74.8% | Specialist focus |
| **KG4Diagnosis** | **82.8%** | **Comprehensive** |

### Knowledge Graph Impact

| Metric | Without KG | With KG | Δ |
|--------|-----------|---------|---|
| Symptom coverage | 64% | 89% | +25% |
| Exam recommendations | 58% | 84% | +26% |
| Treatment alignment | 61% | 81% | +20% |
| Differential breadth | 3.2 avg | 5.8 avg | +81% |

---

## Key Technical Components

### 1. GP-LLM Agent (Router)

```python
# Pseudocode for GP routing logic
def gp_assessment(patient_input):
    # Extract chief complaint and symptoms
    symptoms = extract_symptoms(patient_input)

    # Query KG for symptom-department associations
    candidate_depts = kg.query("""
        MATCH (s:Symptom)-[:SUGGESTS]->(d:Disease)-[:BELONGS_TO]->(dept:Department)
        WHERE s.name IN $symptoms
        RETURN dept.name, count(DISTINCT d) as relevance
        ORDER BY relevance DESC
        LIMIT 3
    """, symptoms=symptoms)

    # LLM reasoning with KG context
    routing_decision = llm.reason(
        prompt=GP_PROMPT,
        symptoms=symptoms,
        kg_context=candidate_depts
    )

    return routing_decision.primary_department
```

### 2. Consultant-LLM Agents (Specialists)

Each specialist agent has:
- **Department-specific KG subgraph** (filtered view)
- **Diagnostic protocol prompts** (specialty-tuned)
- **Examination ordering logic**
- **Treatment recommendation patterns**

### 3. Knowledge Graph Queries

```cypher
-- Find differential diagnoses for symptom set
MATCH (s:Symptom)-[:SUGGESTS]->(d:Disease)-[:BELONGS_TO]->(dept:Department)
WHERE s.name IN ['chest pain', 'shortness of breath', 'fatigue']
  AND dept.name = 'Cardiology'
WITH d, collect(DISTINCT s.name) as matched_symptoms
WHERE size(matched_symptoms) >= 2
RETURN d.name, matched_symptoms, d.severity
ORDER BY size(matched_symptoms) DESC

-- Get diagnostic workup for suspected condition
MATCH (d:Disease {name: 'Acute Myocardial Infarction'})
MATCH (d)-[:REQUIRES_EXAM]->(e:Examination)
MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
RETURN d, collect(DISTINCT e.name) as exams, collect(DISTINCT s.name) as symptoms
```

---

## Relevance to Your Project

### Direct Architectural Parallels

| KG4Diagnosis | Your Architecture |
|--------------|-------------------|
| Neo4j knowledge graph | Temporal Knowledge Graph |
| GP → Specialist routing | Severity-based escalation |
| Department-specific agents | Condition-specific reasoning modules |
| Symptom-disease relations | Temporal symptom-event relations |

### What KG4Diagnosis Validates

1. **Knowledge graphs improve LLM diagnosis** — +11.6% accuracy with KG integration
2. **Multi-agent > single agent** — Specialist routing matters
3. **Structured relations > unstructured retrieval** — Graph queries outperform RAG alone
4. **362 diseases manageable** — Scale is achievable

### Gaps Your Project Addresses

| KG4Diagnosis Gap | Your Innovation |
|------------------|-----------------|
| Static symptom-disease mapping | Temporal evolution of symptoms |
| No time dimension | Allen's 13 interval relations |
| Single-timepoint assessment | Trajectory-based prediction |
| No constraint satisfaction | Clinical protocol constraints |
| General diagnosis | Acute care time-criticality |

---

## Implementation Insights

### Neo4j Graph Construction

- **Source data:** Medical textbooks, clinical guidelines, ICD-10
- **Entity extraction:** NER models + manual curation
- **Relation validation:** Clinical expert review
- **Update frequency:** Quarterly refresh recommended

### Agent Orchestration

- **LangChain/LangGraph** for agent coordination
- **Async execution** for parallel specialist consultation
- **Memory management** for multi-turn diagnosis
- **Confidence thresholds** for routing decisions

### Prompt Engineering

Key elements of effective diagnostic prompts:
1. Role definition (GP vs specialist)
2. KG context injection
3. Structured output format
4. Uncertainty expression
5. Examination justification

---

## Citation

```bibtex
@article{zuo2024kg4diagnosis,
  title={KG4Diagnosis: A Hierarchical Multi-Agent LLM System with Knowledge Graph Enhancement for Medical Diagnosis},
  author={Zuo, Kaiwen and others},
  journal={arXiv preprint arXiv:2412.16833},
  year={2024}
}
```

---

## Bottom Line

KG4Diagnosis demonstrates that **structured knowledge graphs combined with hierarchical multi-agent LLM systems** achieve significantly better diagnostic accuracy than single-agent approaches. The 362-disease, 12-department architecture provides a scalable template.

**For your project:** This validates KG-enhanced reasoning. Your temporal dimension and constraint satisfaction add the **time-critical acute care layer** that KG4Diagnosis lacks — essential for sepsis, AKI, and cardiac arrest prediction where **when** matters as much as **what**.
