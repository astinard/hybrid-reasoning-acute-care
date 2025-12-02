# Agentic Medical AI

LLM-based agent systems for clinical reasoning and diagnosis.

## Documents (5)

| Document | Topic | Key Finding |
|----------|-------|-------------|
| `arxiv_medagent_pro.md` | Evidence-based multi-modal diagnosis | +34% over GPT-4o on glaucoma |
| `arxiv_kg4diagnosis.md` | Hierarchical multi-agent with KG | 362 diseases, +11.6% accuracy |
| `arxiv_mdagents.md` | Adaptive complexity-based collaboration | SOTA on 7/10 benchmarks |
| `arxiv_pathfinder.md` | Multi-agent histopathology | 74% vs 65% human pathologists |
| `arxiv_mmedagent.md` | Tool-learning multi-modal agent | 6 tools, 5 modalities |

---

## Key Insights

### 1. Hierarchical Reasoning Works
MedAgent-Pro and KG4Diagnosis both show that **disease-level planning → patient-level reasoning** dramatically outperforms flat approaches. This mirrors real clinical workflows.

### 2. Adaptive Complexity Routing
MDAgents proves that **matching reasoning architecture to case complexity** (solo vs ICT vs MDT) achieves higher accuracy at lower compute cost than fixed multi-agent structures.

### 3. Role Specialization > Generic Agents
PathFinder demonstrates **role-specialized sequential agents** (Triage → Navigation → Description → Diagnosis) outperform both single VLMs and generic multi-agent systems.

### 4. Tool Learning > End-to-End
MMedAgent shows **training agents to select specialized tools** outperforms training one model to do everything. Mirrors how physicians use diagnostic equipment.

### 5. Knowledge Graphs Improve Diagnosis
KG4Diagnosis achieves +11.6% accuracy by integrating Neo4j knowledge graph with LLM agents — structured relations beat unstructured retrieval.

---

## Relevance to Your Project

All 5 papers validate core architectural choices in your temporal reasoning approach:

| Paper Finding | Your Architecture |
|---------------|-------------------|
| Hierarchical planning (MedAgent-Pro) | Temporal KG + clinical constraints |
| Adaptive routing (MDAgents) | Severity-based escalation |
| Role specialization (PathFinder) | Specialized temporal agents |
| Tool orchestration (MMedAgent) | Constraint satisfaction tools |
| KG enhancement (KG4Diagnosis) | Temporal knowledge graph |

**Your differentiation:** All these papers operate on **single-timepoint** data. Your temporal reasoning adds the **time dimension** — where severity evolves, patterns emerge over time, and **when matters as much as what**.

---

## Architecture Synthesis

Combining insights from all 5 papers into a unified temporal agentic framework:

```
┌────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL ACUTE CARE AGENT                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. COMPLEXITY ASSESSMENT (MDAgents)                                │
│     └─ Dynamic severity classification as patient state evolves     │
│                                                                      │
│  2. HIERARCHICAL PLANNING (MedAgent-Pro)                            │
│     └─ Condition-level protocols → Patient-level reasoning          │
│                                                                      │
│  3. KNOWLEDGE GRAPH INTEGRATION (KG4Diagnosis)                      │
│     └─ Temporal KG with Allen's interval relations                  │
│                                                                      │
│  4. SPECIALIZED AGENTS (PathFinder)                                 │
│     ├─ Triage Agent: Initial severity assessment                    │
│     ├─ Trajectory Agent: Temporal pattern extraction                │
│     ├─ Constraint Agent: Protocol compliance checking               │
│     └─ Alert Agent: Risk synthesis and recommendation               │
│                                                                      │
│  5. TOOL ORCHESTRATION (MMedAgent)                                  │
│     ├─ TemporalKG: Trajectory extraction                            │
│     ├─ AllenReasoner: Interval relation analysis                    │
│     ├─ SepsisScorer: SOFA/qSOFA calculation                         │
│     └─ ConstraintChecker: Guideline compliance                      │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

---

**Bottom Line:** The agentic medical AI field is converging on hierarchical, tool-using, knowledge-enhanced architectures. Your temporal reasoning framework slots naturally into this paradigm — adding the time dimension that all current systems lack.
