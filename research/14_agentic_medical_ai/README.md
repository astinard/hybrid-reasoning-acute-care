# Agentic Medical AI

LLM-based agent systems for clinical reasoning and diagnosis.

## Documents (6)

| Document | Topic | Key Finding |
|----------|-------|-------------|
| `arxiv_medagent_pro.md` | Evidence-based multi-modal diagnosis | +34% over GPT-4o on glaucoma |
| `arxiv_kg4diagnosis.md` | Hierarchical multi-agent with KG | 362 diseases, +11.6% accuracy |
| `arxiv_mdagents.md` | Adaptive complexity-based collaboration | SOTA on 7/10 benchmarks |
| `arxiv_pathfinder.md` | Multi-agent histopathology | 74% vs 65% human pathologists |
| `arxiv_mmedagent.md` | Tool-learning multi-modal agent | 6 tools, 5 modalities |
| `arxiv_ace_context_engineering.md` | **Evolving context playbooks** | **+10.6% agents, 86.9% faster** |

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

### 6. Contexts Should Evolve, Not Be Static (NEW - Game Changer)
ACE proves that **evolving playbooks** that accumulate strategies through Generator→Reflector→Curator loops dramatically outperform static prompts. The system achieves self-improvement **without labeled supervision** using execution feedback alone. Open-source DeepSeek matches proprietary GPT-4.1 agents on AppWorld leaderboard.

---

## Relevance to Your Project

All 6 papers validate and **extend** core architectural choices in your temporal reasoning approach:

| Paper Finding | Your Architecture |
|---------------|-------------------|
| Hierarchical planning (MedAgent-Pro) | Temporal KG + clinical constraints |
| Adaptive routing (MDAgents) | Severity-based escalation |
| Role specialization (PathFinder) | Specialized temporal agents |
| Tool orchestration (MMedAgent) | Constraint satisfaction tools |
| KG enhancement (KG4Diagnosis) | Temporal knowledge graph |
| **Evolving playbooks (ACE)** | **Self-improving temporal patterns** |

**Your differentiation:** Previous papers operate on **single-timepoint** data. Your temporal reasoning adds the **time dimension**.

**ACE adds a new dimension:** Your system shouldn't have static guidelines — it should have **evolving clinical playbooks** that accumulate temporal patterns, refine thresholds, and learn from patient outcomes. This enables **self-improving acute care AI**.

---

## Architecture Synthesis

Combining insights from all 6 papers into a unified **self-improving** temporal agentic framework:

```
┌────────────────────────────────────────────────────────────────────┐
│              SELF-IMPROVING TEMPORAL ACUTE CARE AGENT               │
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
│  6. EVOLVING PLAYBOOK SYSTEM (ACE) ← NEW                            │
│     ├─ Generator: Process patient streams, flag helpful patterns    │
│     ├─ Reflector: Analyze outcomes, extract temporal insights       │
│     ├─ Curator: Update clinical playbook with delta entries         │
│     └─ Result: System improves with every patient encounter         │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## The ACE Evolution: Why This Changes Everything

ACE (arXiv:2510.04618) introduces a paradigm shift that should fundamentally change your approach:

### Before ACE
- Static clinical guidelines
- Fixed constraint rules
- Pre-defined alert thresholds
- One-time knowledge graph training

### After ACE
- **Evolving clinical playbooks** that accumulate temporal patterns
- **Learned constraint refinements** from patient outcomes
- **Adaptive thresholds** that improve with experience
- **Continuously enriched TKG** through reflection loops

### The Self-Improvement Loop

```
Patient Data → Generator (apply current playbook)
                    ↓
            Reflector (analyze outcomes)
                    ↓
            Curator (delta updates to playbook)
                    ↓
            Improved playbook for next patient
```

**Key finding:** ACE achieves this self-improvement **without labeled supervision** — using only execution feedback (patient outcomes).

---

**Bottom Line:** The agentic medical AI field is converging on hierarchical, tool-using, knowledge-enhanced architectures. ACE adds **self-improvement** to this paradigm. Your temporal reasoning framework should evolve from static constraints to **evolving temporal playbooks** — enabling acute care AI that gets better with every patient.
