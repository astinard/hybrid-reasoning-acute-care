# MedAgent-Pro: Evidence-Based Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow

**Source:** arXiv:2503.18968v3 [cs.AI] 2 Jul 2025
**Authors:** Wang et al. (NUS, Oxford, Harbin IT, Zhejiang)
**Code:** https://github.com/jinlab-imvr/MedAgent-Pro

---

## Executive Summary

MedAgent-Pro introduces a **hierarchical agentic reasoning paradigm** that mirrors real clinical workflows: disease-level standardized planning followed by patient-level personalized step-by-step reasoning. This is the first agentic system to achieve systematic, evidence-based medical diagnosis rather than empirical one-hop VQA.

**Key Result:** Outperforms GPT-4o by **34% on glaucoma** and **22% on heart disease** diagnosis. Beats all existing medical agentic systems (MedAgents, MMedAgent, MDAgent) across 10+ imaging modalities, 20+ anatomies, and 50+ diseases.

---

## The Problem This Solves

Current approaches fail clinical standards:

| Approach | Limitation |
|----------|------------|
| **VLMs (GPT-4o, LLaVA-Med)** | Direct empirical answers without quantitative analysis |
| **Reasoning models (o1, DeepSeek-R1)** | Limited fine-grained visual perception |
| **Existing medical agents** | "Toolbox" approach — tools glued together, not clinically-oriented workflow |

**Core Issue:** Treating diagnosis as one-hop QA instead of the **standardized, step-by-step process** that real clinical practice demands.

---

## The MedAgent-Pro Architecture

### Hierarchical Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    DISEASE-LEVEL PLANNING                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ RAG Agent   │───▶│ Knowledge   │───▶│ Diagnostic  │          │
│  │ (MedlinePlus)│    │ Retrieval   │    │ Plan P      │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                   │
│  • 1,000+ diseases in knowledge base                             │
│  • 4,000+ expert-reviewed articles                               │
│  • Generates clinical indicators I = {I₁, I₂, ..., Iₘ}          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PATIENT-LEVEL REASONING                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ Orchestrator│───▶│ Tool Agents │───▶│ Evidence-   │          │
│  │ (selects    │    │ (quant.     │    │ Based       │          │
│  │  steps)     │    │  analysis)  │    │ Verification│          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                   │
│  Tools: MedSAM, Medical SAM Adapter, Cellpose, Maira-2, Copilot  │
│  Status: Continue | Terminate | Complete                         │
└─────────────────────────────────────────────────────────────────┘
```

### Key Innovation: Evidence-Based Reasoning Paradigm

At each step Pᵢ, the system evaluates output reliability:

```
sᵢ = {
  Complete,   if krᵢ ∈ I                      (indicator computed)
  Terminate,  if krᵢ ∉ I ∧ ¬φ(vrᵢ)           (unreliable result)
  Continue,   if krᵢ ∉ I ∧ φ(vrᵢ)            (reliable, proceed)
}
```

Where φ is a state assessment function evaluating:
- Quality of input data
- Plausibility of result

**This prevents error propagation** — unreliable intermediate results halt the pipeline rather than corrupting downstream reasoning.

---

## Performance Results

### vs. General VLMs (REFUGE2, MITEA, NEJM datasets)

| Method | Glaucoma bAcc | Glaucoma F1 | Heart Disease bAcc | Heart Disease F1 |
|--------|---------------|-------------|-------------------|------------------|
| GPT-4o | 56.4% | 21.1% | 56.8% | 28.1% |
| Janus-Pro-7B | 53.4% | 13.3% | 52.3% | 10.7% |
| LLaVA-Med | 50.0% | 0.0% | 50.0% | 0.0% |
| Qwen2.5-7B-VL | 54.3% | 16.3% | 50.0% | 0.0% |
| InternVL2.5-8B | 51.8% | 13.8% | 49.7% | 3.6% |
| **MedAgent-Pro** | **90.4%** | **76.4%** | **77.8%** | **72.3%** |

**Δ vs GPT-4o:** +34.0% bAcc, +55.3% F1 (glaucoma); +21.0% bAcc, +44.2% F1 (heart disease)

### vs. Medical Agentic Systems

| Method | Glaucoma bAcc/F1 | Heart Disease bAcc/F1 | NEJM Acc |
|--------|------------------|----------------------|----------|
| MedAgents (ACL'24) | 52.1% / 8.9% | 51.1% / 15.9% | 66.1% |
| MMedAgent (EMNLP'24) | 52.4% / 16.3% | 55.0% / 26.7% | 71.7% |
| MDAgent (NeurIPS'24) | 56.8% / 22.2% | 57.2% / 30.3% | 73.8% |
| **MedAgent-Pro** | **90.4% / 76.4%** | **77.8% / 72.3%** | **81.7%** |

### vs. Task-Specific Expert Models

| Domain | Expert Model | Expert Performance | MedAgent-Pro |
|--------|--------------|-------------------|--------------|
| Glaucoma | REFUGE2 Winners | 88.3% AUC | **95.1% AUC** |
| Glaucoma | VisionUnite | 85.8% bAcc | **90.4% bAcc** |
| Chest X-Ray | CheXagent | 69.1% bAcc | **72.0% bAcc** |

**Key Finding:** Zero-shot agentic system outperforms trained task-specific models.

### MIMIC-IV Chest X-Ray (12 conditions)

| Condition | GPT-4o | MedAgent-Pro | Δ |
|-----------|--------|--------------|---|
| Atelectasis | 68.7% | **85.5%** | +16.8% |
| Cardiomegaly | 61.2% | **74.2%** | +13.0% |
| Pneumonia | 59.6% | **62.1%** | +2.5% |
| Supporting Devices | 63.4% | **89.2%** | +25.8% |
| **Average (12 tasks)** | 58.3% | **72.0%** | **+13.7%** |

---

## Why This Architecture Works

### 1. Quantitative > Qualitative

Ablation study finding:
- Replacing GPT-4o with domain-specific VisionUnite for qualitative analysis: **marginal improvement**
- Improving segmentation accuracy for quantitative analysis: **consistent gains**

**Insight:** In multi-modal diagnosis, **evidence-based quantitative analysis** matters more than **experience-driven qualitative assessment**.

### 2. Structured Fusion > Flat Fusion

Two decision strategies tested:
- **Flat fusion:** Feed all indicators to VLM for decision
- **Structured fusion:** Assign risk-based weights guided by clinical guidelines

**Result:** Structured fusion consistently outperforms across varying indicator counts. VLMs focus on partial cues; structured weighting ensures comprehensive decisions.

### 3. Clinical Workflow Alignment

Human evaluation by clinicians rated diagnostic complexity 1-12 across chest X-ray tasks:

| Task | Clinician Complexity Rank | MedAgent-Pro Steps |
|------|---------------------------|-------------------|
| Fracture | 12 (highest) | Highest step count |
| Support Devices | 1 (lowest) | Lowest step count |
| Cardiomegaly | Mid (has visual tool) | Reduced by automation |

**Correlation:** Plan complexity aligns with actual clinical difficulty.

---

## Relevance to Your Project

### Direct Architectural Alignment

Your hybrid reasoning approach maps cleanly to MedAgent-Pro's structure:

| MedAgent-Pro Component | Your Architecture |
|------------------------|-------------------|
| Disease-level planning with RAG | Temporal KG + clinical guidelines |
| Patient-level step-by-step reasoning | Neuro-symbolic constraint satisfaction |
| Evidence-based verification | Uncertainty quantification + thresholds |
| Tool integration for quantitative analysis | Specialized models for sepsis indicators |

### Key Validations for Your Grant

1. **Hierarchical reasoning works** — 34% improvement over flat VQA
2. **Clinical workflow mirroring matters** — Plans that match real complexity
3. **Quantitative evidence > empirical judgment** — Supports your constraint-based approach
4. **Zero-shot generalization possible** — Beat task-specific models without training

### What MedAgent-Pro Doesn't Do (Your Differentiation)

| Gap in MedAgent-Pro | Your Innovation |
|---------------------|-----------------|
| No temporal reasoning | Allen's 13 interval relations |
| Static indicators | Dynamic trajectory analysis |
| Single-timepoint diagnosis | Longitudinal pattern detection |
| Generic RAG knowledge | Clinical constraint satisfaction |

**Your pitch:** "MedAgent-Pro shows hierarchical agentic reasoning works. We add **temporal dynamics** and **constraint-based logic** for acute care where **when matters as much as what**."

---

## Implementation Details

### Knowledge Base
- **MedlinePlus:** 1,000+ diseases, 4,000+ articles
- **Two-step retrieval:** Keyword filter → Vector search → Top 5 chunks
- **Output:** Procedural guideline G with clinical indicators I

### Tool Integration
- **Segmentation:** MedSAM, Medical SAM Adapter, Cellpose
- **Grounding:** Maira-2
- **Coding:** GitHub Copilot for quantitative computation

### Evaluation Datasets
- **REFUGE2:** Glaucoma diagnosis (fundus images)
- **MITEA:** Heart disease (3D echocardiography)
- **MIMIC-IV:** 12 chest X-ray conditions
- **NEJM:** 992 real-world cases, 50+ diseases, 10+ modalities

---

## Limitations Acknowledged

1. **Tool dependency** — Requires visual tools; some domains lack coverage
2. **VLM hallucination** — Qualitative steps still rely on VLM internal knowledge
3. **Static guidelines** — No adaptive learning from outcomes

**These are exactly where your temporal + neuro-symbolic approach adds value.**

---

## Citation

```bibtex
@article{wang2025medagentpro,
  title={MedAgent-Pro: Towards Evidence-based Multi-modal Medical Diagnosis via Reasoning Agentic Workflow},
  author={Wang, Ziyue and Wu, Junde and Cai, Linghan and Low, Chang Han and Yang, Xihong and Li, Qiaxuan and Jin, Yueming},
  journal={arXiv preprint arXiv:2503.18968},
  year={2025}
}
```

---

## Bottom Line

MedAgent-Pro proves that **structured, evidence-based agentic reasoning dramatically outperforms empirical VLM approaches** in medical diagnosis. The hierarchical disease-level → patient-level architecture mirrors real clinical workflows and achieves state-of-the-art across diverse conditions.

**For your project:** This validates the core premise that clinical AI needs structured reasoning, not just bigger models. Your temporal + constraint-based additions address the gaps MedAgent-Pro leaves open — making your approach the natural next step.
