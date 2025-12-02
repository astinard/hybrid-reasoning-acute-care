# Hybrid Reasoning for Acute Care

*Temporal Knowledge Graphs and Clinical Constraints for Emergency Department Decision Support*

<p align="center">
  <img src="https://img.shields.io/badge/documents-157-blue" alt="Documents">
  <img src="https://img.shields.io/badge/lines-197K+-green" alt="Lines">
  <img src="https://img.shields.io/badge/papers-2000+-orange" alt="Papers">
  <img src="https://img.shields.io/badge/research_score-9.8/10-brightgreen" alt="Score">
</p>

---

## The Problem

Emergency departments face critical decision gaps that cost lives and money:

| Challenge | Impact | Current State |
|-----------|--------|---------------|
| **Sepsis Detection** | 1-hour delay = 7% mortality increase | Median 3.5 hour delays |
| **Alert Fatigue** | Clinicians ignore 49-96% of alerts | 50% false positive rate |
| **Clinical Coding** | $7K+ cost per discharge | 15-20% error rate |
| **Time-Critical Protocols** | Stroke: 4.5hr window, ACS: 90min door-to-balloon | Frequent misses |

## The Solution

This repository synthesizes **2,000+ ArXiv papers** into a comprehensive research corpus for **hybrid AI approaches** that combine:

- **Temporal Reasoning** — When events occur matters more than what (Allen's 13 interval relations)
- **Neuro-Symbolic Logic** — Rules + neural networks, not either-or (IBM LNN achieves 17% improvement)
- **Clinical Constraints** — Valid outputs, explainability, regulatory compliance

## Results We Can Achieve

| Application | Metric | Evidence |
|-------------|--------|----------|
| Sepsis Prediction | AUROC 0.90-0.94, 6hr advance | [04_clinical_prediction/](research/04_clinical_prediction/) |
| Temporal KG | AUROC 0.9269 (KAT-GNN) | [01_temporal_methods/](research/01_temporal_methods/) |
| ICD-10 Coding | 89%+ F1 score | [05_clinical_nlp/](research/05_clinical_nlp/) |
| Alert Precision | 75%+ (vs 50% baseline) | [08_clinical_operations/](research/08_clinical_operations/) |
| ECG Analysis | 95-99% accuracy | [06_medical_imaging/](research/06_medical_imaging/) |

---

## Quick Start

<table>
<tr>
<td width="33%">

### For Clinical Researchers
1. [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md)
2. [CONCEPT_NOTE_MAPPING.md](CONCEPT_NOTE_MAPPING.md)
3. [docs/PRELIMINARY_RESULTS.md](docs/PRELIMINARY_RESULTS.md)

</td>
<td width="33%">

### For ML Engineers
1. [01_temporal_methods/](research/01_temporal_methods/)
2. [03_hybrid_neurosymbolic/](research/03_hybrid_neurosymbolic/)
3. [10_implementation_deployment/](research/10_implementation_deployment/)

</td>
<td width="33%">

### For Grant Writers
1. [_institutional/](research/_institutional/)
2. [docs/IRB_REGULATORY_PATHWAY.md](docs/IRB_REGULATORY_PATHWAY.md)
3. [docs/COMPETITIVE_DIFFERENTIATION.md](docs/COMPETITIVE_DIFFERENTIATION.md)

</td>
</tr>
</table>

---

## Repository Structure

```
hybrid-reasoning-acute-care/
│
├── README.md                          # You are here
├── RESEARCH_BRIEF.md                  # Executive summary (9.5/10)
├── CONCEPT_NOTE_MAPPING.md            # Maps concept note → research
│
├── docs/                              # Strategic documentation
│   ├── TECHNICAL_ARCHITECTURE.md
│   ├── PRELIMINARY_RESULTS.md
│   ├── IRB_REGULATORY_PATHWAY.md
│   └── COMPETITIVE_DIFFERENTIATION.md
│
└── research/                          # 157 documents in 16 categories
    │
    ├── _foundations/          (12)    # MIMIC-IV, FHIR, OMOP, FDA guidance
    ├── _institutional/        (6)     # UCF/HCA partnership, funding
    ├── _domain_synthesis/     (2)     # Cross-cutting analysis
    │
    ├── 01_temporal_methods/   (7)     # Temporal KGs, time series
    ├── 02_graph_neural_networks/ (10) # GNN, embeddings, KG reasoning
    ├── 03_hybrid_neurosymbolic/ (7)   # Neuro-symbolic, constraints
    ├── 04_clinical_prediction/ (19)   # Sepsis, AKI, mortality
    ├── 05_clinical_nlp/       (10)    # NLP, coding, summarization
    ├── 06_medical_imaging/    (9)     # ECG, X-ray, multimodal
    ├── 07_specialty_domains/  (19)    # Oncology, pediatric, surgical
    ├── 08_clinical_operations/ (8)    # Triage, ED, alerts
    ├── 09_learning_methods/   (18)    # Transfer, RL, causal, diffusion
    ├── 10_implementation_deployment/ (12)  # MLOps, federated, edge
    ├── 11_interpretability_safety/ (10)    # XAI, fairness, safety
    ├── 12_data_quality/       (6)     # Missing data, synthetic
    └── 13_emerging_technology/ (9)    # LLMs, digital twins
```

---

## Key Technical Innovations

<table>
<tr>
<td width="50%">

### Temporal Knowledge Graphs
- Allen's 13 interval relations
- KAT-GNN: AUROC 0.9269 on MIMIC-IV
- Dynamic graph updates for real-time
- **See:** [01_temporal_methods/](research/01_temporal_methods/)

### Neuro-Symbolic Constraints
- IBM LNN: 17% improvement + 100x compression
- Clinical protocol encoding
- Guaranteed constraint satisfaction
- **See:** [03_hybrid_neurosymbolic/](research/03_hybrid_neurosymbolic/)

</td>
<td width="50%">

### Multimodal Fusion
- Cross-modal attention (ECG + notes + labs)
- Temporal alignment across modalities
- Missing modality handling
- **See:** [06_medical_imaging/](research/06_medical_imaging/)

### Diffusion Over Trajectories
- Sequence diffusion for trajectory generation
- Constraint-guided sampling
- Counterfactual analysis
- **See:** [09_learning_methods/](research/09_learning_methods/)

</td>
</tr>
</table>

---

## Institutional Context

### UCF/HCA Partnership
| Asset | Scale |
|-------|-------|
| Hospitals | 182+ across 20 states |
| GME Residents | 1,000+ (nation's largest consortium) |
| Annual Encounters | 35M+ patients |
| Data Access | Direct path through partnership |

### Datasets Available
- **MIMIC-IV**: 364,627 patients, 35,239 sepsis cases
- **eICU**: 200,859 stays, 139 hospitals
- **HCA Internal**: Available through partnership

### Regulatory Pathway
- FDA CDS guidance analyzed (4-criteria exemption)
- 510(k) predicate devices identified
- IRB protocol templates prepared

---

## Funding Targets

| Agency | Program | Award | Fit |
|--------|---------|-------|-----|
| NSF | Smart Health (SCH) | $300K-1M | ★★★ |
| NIH | R21 Exploratory | $275K | ★★★ |
| NIH | R01 | $500K/yr | ★★★ |
| AHRQ | R01 Health IT | $400K/yr | ★★★ |
| NSF | CAREER | $500K-600K | ★★ |

---

## Corpus Statistics

| Metric | Value |
|--------|-------|
| Total Documents | 157 |
| Total Lines | 197,000+ |
| ArXiv Papers | 2,000+ |
| Research Categories | 16 |
| Concept Note Coverage | 95%+ |

---

## Citation

```bibtex
@misc{hybrid_reasoning_acute_care_2025,
  title={Hybrid Reasoning for Acute Care: Research Literature Corpus},
  author={Stinard, Alex and UCF CRCV Lab},
  year={2025},
  publisher={GitHub},
  url={https://github.com/astinard/hybrid-reasoning-acute-care}
}
```

---

## Contact

**Primary:** Alex Stinard, UCF Computer Science
**Partnership:** HCA Healthcare, UCF College of Medicine, CRCV Lab
**Repository:** [github.com/astinard/hybrid-reasoning-acute-care](https://github.com/astinard/hybrid-reasoning-acute-care)

---

<p align="center">
  <strong>Last Updated:</strong> December 2025 &nbsp;|&nbsp; <strong>Version:</strong> 2.0 &nbsp;|&nbsp; <strong>License:</strong> MIT
</p>
