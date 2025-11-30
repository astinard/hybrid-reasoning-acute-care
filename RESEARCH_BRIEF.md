# ACADEMIC RESEARCH BRIEF: Hybrid Reasoning for Acute Care
## University of Central Florida - Computer Science & College of Medicine
## Interdisciplinary Research Direction Assessment

**Date:** November 2025
**Prepared For:** UCF Department of Computer Science & UCF College of Medicine
**Purpose:** Doctoral/Faculty Research Direction Viability Assessment

---

## EXECUTIVE SUMMARY

This brief assesses the viability of "Hybrid Reasoning for Acute Care: Temporal Knowledge Graphs and Clinical Constraints" as an interdisciplinary research program between UCF Computer Science and UCF College of Medicine.

### RECOMMENDATION: **STRONG CANDIDATE FOR INTERDISCIPLINARY RESEARCH**

This research direction offers:
1. **High publication potential** across top venues (NeurIPS, ICML, JAMIA, Nature Digital Medicine)
2. **Novel contributions** at intersection of knowledge representation, neural-symbolic AI, and clinical informatics
3. **Natural interdisciplinary collaboration** between CS and Medicine
4. **Access to public datasets** (MIMIC-III/IV, eICU) eliminating data access barriers
5. **Alignment with NSF/NIH funding priorities** in trustworthy AI and clinical decision support

**Estimated Timeline:** 3-5 years for dissertation-level contribution
**Funding Sources:** NSF CAREER, NIH R01/R21, DoD CDMRP, industry partnerships

---

## PART 1: RESEARCH SIGNIFICANCE

### 1.1 Scientific Gaps Addressable

| Gap | Current State | UCF Contribution Potential |
|-----|---------------|---------------------------|
| Temporal clinical reasoning | Ad-hoc temporal encoding | Formal temporal KG framework with Allen's interval algebra |
| Explainable acute care AI | Black-box models (67% miss rate for Epic sepsis) | Neuro-symbolic reasoning with interpretable decision paths |
| Benchmark standardization | Fragmented evaluation protocols | Unified temporal clinical benchmarks |
| Multi-modal fusion | Separate modality processing | KG-scaffolded integration architecture |

### 1.2 Why This Problem Matters

**Clinical Impact:**
- ED visits: 140M+ annually in US
- Sepsis mortality: 270,000 deaths/year in US
- Current AI systems failing: Epic Sepsis Model external validation shows AUC 0.63 vs claimed 0.76-0.83

**Scientific Impact:**
- Neuro-symbolic AI is experiencing resurgence (NeurIPS 2024 workshop, major industry adoption)
- Temporal knowledge graphs remain under-explored in healthcare
- Intersection of KR+R, ML, and clinical informatics is publication-rich

### 1.3 Alignment with UCF Strengths

**UCF Computer Science:**
- Strong AI/ML research groups
- Knowledge representation expertise
- NLP and healthcare informatics capabilities

**UCF College of Medicine:**
- Clinical partnerships with Orlando Health, AdventHealth, Nemours
- Emergency medicine and critical care faculty
- Access to clinical expertise for validation

**Interdisciplinary Opportunity:**
- Joint CS-Medicine dissertations
- Shared graduate students
- Collaborative grant applications (NIH values interdisciplinary teams)

---

## PART 2: LITERATURE LANDSCAPE

### 2.1 Publication Venue Analysis

| Venue | Relevant Topics | Recent Acceptance Rate | Fit |
|-------|-----------------|----------------------|-----|
| **NeurIPS** | Neuro-symbolic AI, graph learning | ~26% | High |
| **ICML** | Temporal models, representation learning | ~28% | High |
| **KDD** | Healthcare AI, knowledge graphs | ~20% | High |
| **AAAI** | Knowledge representation, clinical AI | ~23% | High |
| **JAMIA** | Clinical informatics, decision support | ~15% | Very High |
| **Nature Digital Medicine** | Healthcare AI applications | ~8% | High impact |
| **npj Digital Medicine** | Clinical AI validation | ~12% | High impact |
| **JMIR** | Health informatics | ~30% | Accessible |

### 2.2 Key Research Groups to Position Against/With

| Group | Institution | Focus | Collaboration Potential |
|-------|-------------|-------|------------------------|
| Jimeng Sun Lab | UIUC | Healthcare AI, temporal modeling | High - complementary |
| David Sontag Lab | MIT | Clinical ML, causal inference | Medium - competitive |
| Nigam Shah Lab | Stanford | Clinical NLP, knowledge graphs | High - datasets |
| Fei Wang Lab | Cornell | Healthcare AI, federated learning | High - complementary |
| IBM Research | Industry | Neuro-symbolic AI, LNNs | High - framework access |

### 2.3 State-of-the-Art Performance Benchmarks

**Temporal Knowledge Graphs (Target to Beat):**
- KAT-GNN (2024): AUROC 0.9269 for CAD prediction
- RGCN+Literals (Jhee 2025): AUC 0.91 for outcome prediction
- GraphCare (2023): Strong MIMIC-III/IV results

**Neuro-Symbolic Clinical AI (Target to Beat):**
- LNN Diabetes (Lu 2024): 80.52% accuracy, 0.8457 AUROC
- NeuroSymAD (He 2025): 88.58% accuracy for Alzheimer's

**ICD Coding (Context, not primary target):**
- PLM-ICD: 60-62% micro-F1 (demonstrates ceiling of current approaches)

---

## PART 3: PROPOSED RESEARCH DIRECTIONS

### 3.1 Dissertation-Level Research Questions

**RQ1: Temporal Knowledge Graph Representation**
> How can clinical event sequences be represented as temporal knowledge graphs that capture causal and temporal relationships while supporting real-time inference?

*Novelty:* First systematic framework for ED-specific temporal KGs with formal temporal semantics (Allen's interval algebra + OWL-Time)

**RQ2: Neuro-Symbolic Clinical Reasoning**
> How can logical neural networks be extended to encode clinical guidelines as soft constraints while maintaining differentiability?

*Novelty:* Extension of IBM's LNN framework to multi-pathway clinical reasoning with hierarchical medical ontologies

**RQ3: Explainable Acute Care Decision Support**
> What explanation modalities (reasoning chains, counterfactuals, feature importance) are most effective for clinician trust and adoption?

*Novelty:* Human-centered evaluation of neuro-symbolic explanations with ED physicians

**RQ4: Multi-Modal Temporal Fusion**
> How can knowledge graphs serve as scaffolds for integrating heterogeneous clinical data (vitals, labs, notes, imaging) with temporal alignment?

*Novelty:* KG-guided attention mechanisms for asynchronous multi-modal clinical data

### 3.2 Potential Paper Contributions

**Year 1-2 Papers:**
1. "Temporal Knowledge Graphs for Emergency Department Risk Stratification" → **KDD/AAAI**
2. "Benchmarking Temporal Clinical Reasoning: A MIMIC-IV Study" → **JAMIA**
3. "Survey: Neuro-Symbolic AI in Healthcare" → **ACM Computing Surveys**

**Year 2-3 Papers:**
1. "Logical Neural Networks for Interpretable Sepsis Prediction" → **NeurIPS/ICML**
2. "Clinician-Centered Evaluation of AI Explanations in Acute Care" → **CHI/CSCW**
3. "Knowledge Graph-Scaffolded Multi-Modal EHR Fusion" → **Nature Digital Medicine**

**Year 3-4 Papers:**
1. "Hybrid Reasoning Framework for ED Decision Support" → **AAAI/IJCAI**
2. "Prospective Validation of Temporal Knowledge Graph-Based Clinical AI" → **NEJM AI/Lancet Digital Health**

### 3.3 Research Thrusts for Multiple Students

| Thrust | CS Focus | Medicine Focus | Joint Work |
|--------|----------|----------------|------------|
| **Thrust 1: Representation** | Temporal KG architecture | Clinical ontology design | Schema validation |
| **Thrust 2: Reasoning** | LNN implementation | Guideline formalization | Rule extraction |
| **Thrust 3: Explanation** | XAI methods | Clinician studies | User evaluation |
| **Thrust 4: Validation** | Benchmark development | Clinical protocols | Prospective studies |

---

## PART 4: DATASETS AND RESOURCES

### 4.1 Publicly Available Datasets

| Dataset | Size | Access | Relevance |
|---------|------|--------|-----------|
| **MIMIC-III** | 46K+ ICU stays | PhysioNet (free) | Primary benchmark |
| **MIMIC-IV** | 300K+ admissions | PhysioNet (free) | Latest benchmark |
| **MIMIC-IV-ED** | 448K+ ED visits | PhysioNet (free) | ED-specific |
| **eICU** | 200K+ ICU stays | PhysioNet (free) | Multi-center validation |
| **MIMIC-CXR** | 377K+ chest X-rays | PhysioNet (free) | Multi-modal |

**Access Process:** PhysioNet credentialing (~1-2 weeks), CITI training required

### 4.2 Knowledge Resources

| Resource | Content | Access |
|----------|---------|--------|
| **SNOMED-CT** | 350K+ clinical concepts | UMLS license (free for research) |
| **UMLS** | Unified medical terminology | NLM license (free) |
| **ICD-10** | Diagnosis codes | Public |
| **RxNorm** | Medication ontology | NLM (free) |
| **DrugBank** | Drug interactions | Academic license |

### 4.3 Computational Resources Needed

| Resource | Specification | UCF Availability |
|----------|---------------|------------------|
| GPU Compute | 4x A100 or equivalent | STOKES cluster / AWS credits |
| Storage | 5-10TB for datasets | Available |
| Software | PyTorch, DGL, IBM LNN | Open source |

---

## PART 5: FUNDING OPPORTUNITIES

### 5.1 Federal Funding

| Agency | Program | Fit | Award Size | Duration |
|--------|---------|-----|------------|----------|
| **NSF** | CAREER | High | $500K-600K | 5 years |
| **NSF** | RI: Medium | High | $600K-1.2M | 3-4 years |
| **NSF** | Smart Health | Very High | $300K-1M | 3-4 years |
| **NIH** | R21 (Exploratory) | High | $275K | 2 years |
| **NIH** | R01 | High | $500K/yr | 5 years |
| **NIH** | K99/R00 (Postdoc) | High | $250K | 5 years |
| **AHRQ** | R01 Health IT | Very High | $400K/yr | 4 years |

### 5.2 Specific Program Fit

**NSF 24-582: Smart Health and Biomedical Research in the Era of AI**
- Explicitly calls for "AI and machine learning for clinical decision support"
- Requires interdisciplinary teams (CS + clinical)
- UCF CS + Medicine is perfect fit

**NIH NOT-OD-24-004: AI/ML for Healthcare**
- Emphasis on trustworthy, explainable AI
- Neuro-symbolic approaches directly aligned
- Clinical validation requirements match our prospective validation thrust

### 5.3 Industry Partnerships

| Company | Interest | Partnership Type |
|---------|----------|------------------|
| **Epic** | Improve sepsis model | Data/validation partnership |
| **IBM Research** | LNN healthcare applications | Framework collaboration |
| **Google Health** | Clinical AI | Research grant |
| **Microsoft Research** | Healthcare AI | Azure credits, collaboration |
| **Oracle Health** | Clinical decision support | Data partnership |

### 5.4 Foundation Funding

| Foundation | Program | Fit |
|------------|---------|-----|
| **Gordon and Betty Moore** | Patient Safety | High |
| **Robert Wood Johnson** | Healthcare Innovation | Medium |
| **PCORI** | Patient-Centered Research | High |

---

## PART 6: COLLABORATION STRUCTURE

### 6.1 Proposed Team Structure

**UCF Computer Science:**
- PI: Faculty with AI/ML expertise
- Co-PI: Faculty with knowledge representation expertise
- PhD Students: 2-3 (representation learning, neuro-symbolic, systems)

**UCF College of Medicine:**
- Co-PI: Faculty with clinical informatics expertise
- Clinical Collaborators: ED physicians, critical care specialists
- Medical Students/Residents: Clinical validation support

### 6.2 External Collaborations

| Institution | Role | Value |
|-------------|------|-------|
| **MIT CSAIL** | Benchmark validation | Credibility, comparison |
| **Stanford BMIR** | Dataset expertise | MIMIC expertise |
| **IBM Research** | LNN framework | Technical support |
| **Orlando Health** | Clinical validation | Local ED data |
| **AdventHealth** | Prospective studies | Multi-site validation |

### 6.3 Advisory Board (Recommended)

- Academic: 2-3 senior faculty from peer institutions
- Clinical: 2 practicing ED physicians
- Industry: 1 representative from EHR vendor or healthcare AI company

---

## PART 7: TIMELINE AND MILESTONES

### 7.1 Year 1: Foundation

| Quarter | Milestone | Deliverable |
|---------|-----------|-------------|
| Q1 | Dataset access, literature review | PhysioNet credentials, survey draft |
| Q2 | Temporal KG framework design | Architecture document, initial code |
| Q3 | MIMIC-III/IV benchmarking | Baseline results, benchmark paper draft |
| Q4 | Grant submission (NSF CAREER/R21) | Submitted proposals |

### 7.2 Year 2: Core Contributions

| Quarter | Milestone | Deliverable |
|---------|-----------|-------------|
| Q1 | Temporal KG implementation | Open-source framework release |
| Q2 | Neuro-symbolic integration | LNN clinical reasoning module |
| Q3 | MIMIC-IV-ED validation | Top venue paper submission |
| Q4 | Clinician evaluation design | IRB approval, study protocol |

### 7.3 Year 3: Validation and Extension

| Quarter | Milestone | Deliverable |
|---------|-----------|-------------|
| Q1 | Clinician user studies | CHI/CSCW paper submission |
| Q2 | Multi-modal fusion | Architecture paper |
| Q3 | Multi-site validation (eICU) | Generalization study |
| Q4 | Grant renewal/expansion | R01 submission |

### 7.4 Years 4-5: Impact and Translation

- Prospective clinical validation (with hospital partners)
- High-impact clinical venue publications (NEJM AI, Lancet Digital Health)
- Dissertation completions
- Technology transfer discussions

---

## PART 8: RISK ASSESSMENT (ACADEMIC)

### 8.1 Research Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| MIMIC data limitations | Low | Medium | eICU multi-site validation |
| Neuro-symbolic scaling issues | Medium | Medium | Focus on interpretability over scale |
| Negative clinician evaluation | Medium | High | Iterative design with clinician input |
| Scooped by larger groups | Medium | High | Focus on clinical validation (our advantage) |

### 8.2 Funding Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| NSF/NIH rejection | Medium | High | Multiple submissions, industry backup |
| Budget constraints | Low | Medium | Leverage cloud credits, open datasets |
| Personnel turnover | Medium | Medium | Document everything, multiple students |

### 8.3 Collaboration Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Clinical partner disengagement | Low | High | Formal agreements, regular meetings |
| IRB delays | Medium | Medium | Early submission, standard protocols |
| Data access issues | Low | Medium | Use only public datasets initially |

---

## PART 9: SUCCESS METRICS (ACADEMIC)

### 9.1 Publication Metrics

| Timeframe | Target |
|-----------|--------|
| Year 1 | 1-2 workshop papers, 1 survey submission |
| Year 2 | 2 top venue submissions (NeurIPS/ICML/KDD/AAAI) |
| Year 3 | 1 clinical venue (JAMIA/Nature DM), 2 CS venues |
| Year 4-5 | 1 high-impact clinical (NEJM AI/Lancet DH) |
| Total | 8-12 peer-reviewed publications |

### 9.2 Funding Metrics

| Timeframe | Target |
|-----------|--------|
| Year 1 | Submit NSF CAREER + NIH R21 |
| Year 2 | Secure $300K+ funding |
| Year 3 | Submit R01, secure $500K+ cumulative |
| Year 5 | $1M+ cumulative funding |

### 9.3 Impact Metrics

| Metric | Target |
|--------|--------|
| Citations (5 year) | 500+ |
| GitHub stars (framework) | 200+ |
| Benchmark adoption | 3+ external papers using |
| Clinical validation | 1 prospective study completed |

### 9.4 Student Outcomes

| Outcome | Target |
|---------|--------|
| PhD dissertations | 2-3 |
| MS theses | 3-4 |
| Student publications | 4+ per student |
| Industry placements | Top tech/healthcare AI |

---

## PART 10: RECOMMENDATION

### 10.1 Overall Assessment

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Scientific novelty | 8/10 | Unique intersection of TKG + neuro-symbolic + clinical |
| Publication potential | 9/10 | Multiple venues across CS and clinical |
| Funding alignment | 9/10 | Direct fit with NSF Smart Health, NIH AI priorities |
| Feasibility | 7/10 | Public datasets available, manageable scope |
| UCF fit | 8/10 | Leverages CS-Medicine collaboration |
| Student training | 9/10 | Rich dissertation topics, industry-relevant skills |

**Overall Score: 8.3/10 - STRONGLY RECOMMENDED**

### 10.2 Recommended Next Steps

1. **Immediate (Month 1):**
   - Identify faculty collaborators in CS and Medicine
   - Apply for PhysioNet MIMIC access
   - Draft NSF CAREER concept

2. **Near-term (Months 2-3):**
   - Recruit initial PhD student(s)
   - Establish clinical advisory relationship
   - Submit internal seed funding application

3. **Medium-term (Months 4-6):**
   - Complete literature survey
   - Develop initial temporal KG prototype
   - Submit NSF CAREER proposal

### 10.3 Potential Challenges for UCF Specifically

| Challenge | Mitigation |
|-----------|------------|
| Limited clinical informatics history | Partner with established groups, leverage Medicine school |
| Competition from R1 powerhouses | Focus on clinical validation (requires hospital access) |
| Resource constraints | Use cloud credits, open datasets, lean team |

### 10.4 Unique UCF Advantages

| Advantage | Leverage Strategy |
|-----------|-------------------|
| Orlando healthcare ecosystem | Partner with Orlando Health, AdventHealth, Nemours |
| Growing CS department | Position as flagship interdisciplinary initiative |
| Medical school maturation | Joint programs, shared students |
| Florida healthcare market | State funding opportunities, industry partnerships |

---

## APPENDIX A: KEY CITATIONS FOR PROPOSAL WRITING

### Foundational Papers (Must Cite)
1. Jhee et al. (2025). Temporal knowledge graphs for clinical outcomes. arXiv:2502.21138
2. Lu et al. (2024). Neuro-symbolic diagnosis prediction. arXiv:2410.01855
3. Cui et al. (2024). MINGLE: Multimodal EHR fusion. arXiv:2403.08818
4. Edin et al. (2023). Medical coding critical review. arXiv:2304.10909
5. Xie et al. (2021). ED triage benchmark. Scientific Data.

### Background Papers
6. Shickel et al. (2017). Deep EHR survey. IEEE JBHI.
7. He et al. (2023). MedDiff: Diffusion for EHR. arXiv:2302.04355
8. Vu et al. (2020). LAAT for ICD coding. arXiv:2007.06351

### Neuro-Symbolic AI
9. Hossain & Chen (2025). Neuro-symbolic healthcare perspectives. arXiv:2503.18213
10. IBM LNN documentation and papers

---

## APPENDIX B: SAMPLE GRANT LANGUAGE

### Project Summary (NSF Style)
> This project develops a hybrid reasoning framework for acute care clinical decision support that combines temporal knowledge graphs with neuro-symbolic AI. Unlike current "black-box" deep learning approaches that lack interpretability and fail to generalize across patient populations, our approach integrates structured clinical knowledge with neural learning to produce explainable, trustworthy predictions. We will: (1) develop a temporal knowledge graph representation for emergency department patient trajectories, (2) extend logical neural networks to encode clinical guidelines as differentiable constraints, (3) evaluate explanation effectiveness with practicing clinicians, and (4) validate on multi-site public datasets. This interdisciplinary collaboration between computer science and medicine will advance both AI methodology and clinical practice.

### Intellectual Merit (NSF Style)
> This research makes fundamental contributions to knowledge representation, machine learning, and clinical informatics. Specifically, we introduce (1) a novel temporal knowledge graph schema for acute care that formalizes clinical event sequences using Allen's interval algebra, (2) extensions to logical neural networks that incorporate hierarchical medical ontologies as soft constraints, and (3) the first systematic evaluation of neuro-symbolic explanations with emergency department physicians.

### Broader Impacts (NSF Style)
> Emergency departments serve as the healthcare safety net, providing 28% of all acute care visits. Improving ED decision support has direct implications for health equity, as underserved populations disproportionately rely on emergency care. This project will produce open-source tools, public benchmarks, and educational materials. Graduate students will receive interdisciplinary training at the intersection of AI and medicine. We will engage undergraduate researchers through REU supplements and partner with UCF's diverse student body to broaden participation in computing.

---

*This research brief was prepared to assess the viability of "Hybrid Reasoning for Acute Care" as an academic research direction for UCF Computer Science and College of Medicine. Analysis based on comprehensive literature review of 100+ papers across temporal knowledge graphs, neuro-symbolic AI, generative models, multimodal fusion, ICD coding, and privacy-preserving ML.*
