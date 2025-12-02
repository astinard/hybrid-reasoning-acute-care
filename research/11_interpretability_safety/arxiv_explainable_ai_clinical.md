# Explainable AI in Clinical Settings: A Research Synthesis

## Executive Summary

This synthesis examines recent research on explainable AI (XAI) methods in clinical decision support systems (CDSS), with emphasis on explanation methods, clinician evaluation studies, and trust/adoption findings. Based on systematic analysis of peer-reviewed literature from 2022-2025, key findings reveal that while XAI shows promise for improving clinical trust and diagnostic confidence, significant challenges remain around cognitive load, alignment with clinical reasoning, and real-world adoption.

**Key Takeaways:**
- SHAP and Grad-CAM are the dominant explanation methods (80%+ of studies use post-hoc, model-agnostic approaches)
- High AI confidence scores increase trust but can lead to overreliance and reduced diagnostic accuracy
- Explanations generally improve trust but frequently increase cognitive load and stress
- Small sample sizes (median: 16 clinicians) limit generalizability of findings
- Critical gap between what XAI provides and what clinicians actually need for decision-making

---

## 1. Overview of Explanation Methods

### 1.1 Dominant XAI Approaches in Clinical Settings

**Post-Hoc Model-Agnostic Methods (>80% of studies)**

The literature reveals strong preference for post-hoc, model-agnostic explanation methods:

1. **SHAP (SHapley Additive exPlanations)** - Most widely adopted (12/31 studies surveyed)
   - Quantifies feature contributions using game theory
   - Provides both local (patient-level) and global (population-level) explanations
   - Applications: ICU mortality prediction, sepsis treatment, postpartum depression risk
   - **Clinical perception:** Generally positive, though concerns about complexity for non-technical users

2. **Gradient-based Methods (Grad-CAM, Integrated Gradients)**
   - Primarily used for medical imaging (radiology, pathology)
   - Highlights image regions influencing predictions
   - **Clinical perception:** Mixed - radiologists prefer simple binary feedback over complex heatmaps in some contexts

3. **LIME (Local Interpretable Model-agnostic Explanations)**
   - Approximates model behavior locally through perturbation
   - Used in sepsis prediction, cardiac arrhythmia detection
   - **Clinical perception:** 78% satisfaction in sepsis studies, but inconsistencies limit trust

### 1.2 Intrinsic Interpretability vs Post-Hoc Explanations

**Intrinsic Methods:**
- Decision trees, linear models, attention mechanisms
- Directly interpretable without additional processing
- One study found decision trees most explainable, but increased experience degraded performance
- Suggests no "one-size-fits-all" approach

**Key Finding:** Transparency alone insufficient - explanations must align with clinical reasoning patterns

### 1.3 Emerging Trends

- **Hybrid approaches:** Combining LLMs with traditional ML for natural language explanations
- **Counterfactual explanations:** "What would need to change for a different prediction?"
- **Multi-modal explanations:** Text + visualization + example-based reasoning
- **Layered systems:** Varying detail levels to minimize cognitive load

---

## 2. Clinician Evaluation Studies: Methods and Findings

### 2.1 Study Characteristics

**Sample Sizes and Methods (from systematic survey of 31 studies):**
- **Median sample size:** 16 participants
- **Only 9 studies had ≥30 participants** - raises concerns about statistical validity
- **Evaluation methods:**
  - Surveys (16 studies) - most common
  - Think-aloud protocols (5 studies)
  - Interviews (4 studies)
  - Focus groups (1 study)
  - Mixed methods (5 studies)

**Medical Specialties:**
- Critical Care/ICU (10 studies) - most represented
- Radiology (7 studies)
- Endocrinology (3 studies)
- Cardiology, Oncology, Neurology (2 each)

### 2.2 Key Evaluation Framework: Four Dimensions of Explainability

Research by Kibria et al. (2025) identified **four critical themes** for XAI acceptance:

#### Theme 1: Understandability
- Clinicians with epidemiology/research backgrounds find systems intuitive
- Novice users struggle without background knowledge
- **Learning curve exists:** More interactions improve comprehension
- **Risk of misinterpretation:** Oversimplified representations (e.g., color-coded risk zones) can mislead

**Representative Quote:**
> "I have a lot of experience. But the answers to these the average person might not agree with, or they don't have the background knowledge." - Participant 1, age 29

#### Theme 2: Trust
- **Pre-use skepticism** about AI accuracy, bias, and privacy
- **Post-use confidence increase** when using actual systems
- **Validation behavior:** Clinicians create scenarios to test model reliability against their experience
- **Bias concerns:** Awareness of potential algorithmic bias in training data

**Critical Finding:** Trust decreased significantly after interaction in pre-study interviews (χ² = 14.277, p = 0.002), but not after actual use

#### Theme 3: Usability
- **Positive factors:**
  - Intuitive design and easy navigation
  - Clear feature representation
  - Dynamic visualization of contributions
  - Real-time responsiveness
  - Color-coded risk levels

- **Improvement needs:**
  - Direct links between related screens
  - More contextual explanations
  - Integration with existing EMR systems
  - Reduction in number of separate systems

#### Theme 4: Usefulness
- **Demand for actionable recommendations,** not just risk scores
- Interest in patient-facing applications
- Desire for EMR integration to save time
- Broader applicability to multiple conditions
- Support for automated referrals and patient education

**Critical Gap:** "Once I know high risk is kind of understandable. But what do I do with that information?" - Participant 4, age 40

### 2.3 Impact on Clinical Performance and Behavior

**Diagnostic Accuracy:**
- MyRiskSurgery system: +12-16% net reclassification improvement
- High AI confidence: -1.5% performance decrease (overreliance effect)
- Low AI confidence: No significant performance change, but +13.1% longer diagnosis time

**Cognitive Load Effects:**
- Mental demand showed slight decrease with XAI (not statistically significant)
- **Stress levels increased significantly** with tumor localization + probability (β = 0.393, p = 0.015)
- 3rd intervention (complex visualization) most cognitively demanding

**Trust Dynamics:**
- Low confidence scores: -16.3% trust (p = 0.023)
- High confidence scores: +10.3% trust (not significant)
- **Asymmetric effect:** Negative confidence more impactful than positive

**Behavioral Patterns (Sivaraman et al., 2023):**
Four distinct clinician behaviors identified:
1. **Ignore:** Disregard AI recommendations
2. **Negotiate:** Weigh and prioritize aspects selectively
3. **Consider:** Treat as additional evidence
4. **Rely:** Follow recommendations directly

**Most common:** "Consider" - viewing AI as supplementary rather than decisive

---

## 3. Trust and Adoption Findings

### 3.1 Factors Promoting Trust

**Technical Factors:**
1. **High model accuracy** (81% in breast cancer detection studies)
2. **Explainability features** - particularly when aligned with clinical knowledge
3. **Consistent performance** across multiple test cases
4. **Transparent model information** (data sources, preprocessing, performance metrics)

**Human Factors:**
1. **Alignment with clinical reasoning** - explanations match expected features
2. **Personal validation** - ability to test system with known cases
3. **Gradual exposure** - learning through repeated interactions
4. **Peer confirmation** - seeing colleagues use successfully

**Design Factors:**
1. **Appropriate framing** - suggestions vs. directives
2. **Evidence-based reasoning** - showing supporting data
3. **Actionable outputs** - not just predictions but next steps
4. **Workflow integration** - seamless EMR incorporation

### 3.2 Barriers to Trust and Adoption

#### Technical Barriers:
- **Black-box opacity** - 89% of surveyed clinicians express concerns
- **Algorithmic bias** from training data
- **Inconsistent explanations** (particularly with LIME)
- **Performance limitations** - 80% accuracy insufficient for high-stakes decisions

#### Human Barriers:
- **Preference for established practice** - "Basic reasoning showed higher understandability than data-driven + explanation"
- **Automation bias risk** - overreliance on high-confidence predictions
- **Cognitive load** - additional systems increase mental burden
- **Learning curve** - especially for global SHAP models

#### Organizational Barriers:
- **EMR integration complexity** - clinicians report "fatigue with different software systems"
- **Time constraints** - 15-20 minute visits insufficient for system interaction
- **Staff training requirements**
- **Regulatory uncertainty**

### 3.3 The Overreliance Problem

**Critical Finding (Rezaeian et al., 2025):**
- High confidence scores substantially increased trust
- **But led to overreliance, reducing diagnostic accuracy**
- Clinicians became less critical of AI suggestions
- "Automation abuse" - reduced human oversight

**Mitigation Strategies:**
- Display confidence scores but frame appropriately
- Encourage critical evaluation through interface design
- Provide "forcing functions" that require explicit reasoning
- Balance transparency with appropriate skepticism

### 3.4 Demographic Influences on Trust

**Age:**
- Older clinicians (55+) rate AI role and usefulness higher
- Younger clinicians (25-34) report higher mental demand and stress
- Experience may reduce cognitive load

**Gender:**
- Female clinicians perceive tasks as more complex and stressful
- Male clinicians rate AI role higher
- Different communication style preferences

**Professional Role:**
- Radiologists view AI as more integral (frequent exposure to AI tools)
- Oncologists find AI more complex and demanding
- Nurses prioritize usability over technical sophistication

**Experience Level:**
- Senior clinicians struggle integrating into established workflows
- Junior clinicians find systems cognitively demanding
- Mid-level clinicians show highest acceptance

---

## 4. Clinical Decision-Making Context (ICU Case Study)

### 4.1 Complexity of Clinical Decision-Making

**Wide Range of Factors (Clark et al., 2024):**

1. **Patient Journey**
   - Care planning and escalation decisions
   - Patient trajectory and history
   - Recent data and trends
   - "Direction of travel" - treatment vs. palliative care

2. **Physiological Data**
   - Organ systems categorization
   - Diagnosis-specific parameters
   - **Context-dependent "normal" values**
   - Trends more important than isolated points

3. **"End-of-Bed" Information**
   - Visual assessment cannot be captured in data
   - Behavioral factors (e.g., delirium)
   - Clinical gestalt from experienced observation
   - **"Eyeballing the patient"** - critical for decision confidence

4. **Operational Factors**
   - Bed availability and hospital flow
   - Staff availability and expertise
   - Time of week (weekend vs. weekday coverage)
   - Resource constraints

**Implication for XAI:** Systems must account for factors beyond physiological data

### 4.2 Collaborative Decision-Making Challenges

**Information Sharing:**
- Handovers between shifts create familiarization delays
- Multiple documentation systems (CareFlow, SBAR, ICU paperwork)
- "Changing clinicians really slows things down"

**XAI Opportunity:**
- Structured handover support
- Organ-system based summaries
- Historical trajectory visualization
- Alerting to missed information

**Risk:**
- Over-simplification may miss nuances
- Bias introduction by limiting options presented
- Reduced critical thinking if over-relied upon

---

## 5. Design Recommendations for Clinical XAI

### 5.1 Framework for Explainability (Synthesized from Literature)

**Operational Definition of Clinical XAI:**
> Explainability refers to the extent to which clinicians can **understand** the input data, interact with the **decision-making** processes, interpret the **outputs and suggestions**, and **trust** the predictions of the AI tool.

**Four-Dimensional Framework:**

1. **Understandability** of data input, meaning, outputs, and decision processes
2. **Trust** in data security, privacy, and confidence in AI methods/outputs
3. **Usability** through seamless interactions and efficient outputs
4. **Usefulness** via personalized actionable recommendations

### 5.2 Specific Design Principles

#### Principle 1: Minimize Cognitive Load
- **Integrate into existing systems** - avoid proliferation of separate tools
- **Layered explanations** - summary view with drill-down capability
- **Progressive disclosure** - show details on demand
- **Visual hierarchy** - prioritize most important information

**Example Implementation:**
- Top-level: Color-coded risk score with top 3 contributing factors
- Mid-level: SHAP waterfall plot showing all feature contributions
- Deep-level: Raw data trends and historical comparisons

#### Principle 2: Provide Contextual Information
- **Patient journey context** - not just current snapshot
- **Trend visualization** - temporal patterns over isolated values
- **Comparative baselines** - patient-specific normal ranges
- **Operational factors** - bed availability, resource constraints

#### Principle 3: Frame for Independent Thought
- **Suggestions, not directives** - "Factors supporting discharge" vs. "Patient should be discharged"
- **Evidence-based presentation** - show supporting data
- **Highlight uncertainty** - confidence ranges, not just point predictions
- **Prompt consideration** - "Have you considered...?" rather than "Do this"

**Critical Quote:**
> "I'd rather be presented the data that's there or things that happen and then let me make that decision." - ICU Doctor

#### Principle 4: Support Collaborative Decision-Making
- **Structured handover tools** - align with SBAR framework
- **Organ system organization** - match clinical thought processes
- **Shared decision views** - family involvement support
- **Audit trail** - who made decisions and why

#### Principle 5: Ensure Clinical Plausibility
- **Align with domain knowledge** - explanations match clinical understanding
- **Highlight anomalies** - flag when recommendations conflict with standards
- **Provide mechanisms for feedback** - clinicians can correct/annotate
- **Continuous validation** - performance monitoring in deployment

### 5.3 Evaluation Requirements

**Application-Grounded Evaluation (Essential for Clinical XAI):**
- Real tasks with real clinical experts (not proxy tasks with laypersons)
- Sufficient sample sizes (≥30 participants minimum)
- Longitudinal assessment (not just initial impressions)
- Multiple evaluation methods (surveys + think-aloud + interviews)

**Metrics to Assess:**

*Technical:*
- Predictive accuracy, calibration, fairness
- Explanation fidelity and consistency
- Computational efficiency

*Human-Centered:*
- Trust calibration (appropriate reliance)
- Cognitive workload (NASA-TLX)
- Task performance (accuracy, time)
- Subjective satisfaction

*Clinical:*
- Diagnostic accuracy changes
- Time to decision
- Confidence in decisions
- Patient outcomes (when measurable)

---

## 6. Socio-Technical Gap Analysis

### 6.1 The Core Disconnect

**Human Reality:**
- Flexible, nuanced, context-dependent reasoning
- Relies on tacit knowledge and experience
- Collaborative and communicative processes
- Values efficiency and minimal disruption

**Technological Reality:**
- Rigid, brittle, data-dependent systems
- Requires formalized, quantifiable inputs
- Often isolated from workflow
- Adds cognitive overhead

**Result:** Socio-technical gap where XAI capabilities ≠ clinical needs

### 6.2 Bridging Strategies

**T1: Human Requirement Realism**
- Identify all stakeholders (clinicians, patients, administrators, regulators)
- Understand diverse and sometimes conflicting needs
- Negotiate trade-offs explicitly
- Include patients as active stakeholders (currently missing)

**T2: Context Realism**
- Evaluate XAI in actual clinical workflows
- Assess whether explanations support real decision needs
- Measure impact on decision quality, not just user satisfaction
- Iterative refinement based on deployment experience

### 6.3 Stakeholder-Centric Development Protocol

**Phase 1: Stakeholder Identification**
- Map all affected parties (clinicians, developers, hospitals, regulators, payers, patients)
- Elicit requirements from each group
- Identify conflicts and negotiate trade-offs

**Phase 2: Use Case Refinement**
- Define specific clinical problem collaboratively
- Establish evaluation criteria jointly
- Inform non-technical stakeholders about system capabilities/limitations
- Set realistic expectations

**Phase 3: Iterative Prototype Development**
- Low-fidelity prototypes for structure exploration
- Retrospective studies for rapid iteration
- High-fidelity prototypes matching final functionality
- Continuous clinician feedback integration

**Phase 4: Comprehensive Evaluation**
- Technical validation (accuracy, robustness, fairness)
- Human-centered evaluation (trust, cognitive load, usability)
- Ethical-legal assessment (accountability, transparency, bias)
- **Iterate back to Phase 3 until criteria met**

---

## 7. Critical Insights and Contradictions

### 7.1 The Explainability Paradox

**Finding 1:** More explainability doesn't always help
- Basic clinical reasoning outperformed ML + explanations in one diabetes study
- SHAP viewed as "too complex" - clinicians preferred simple bar charts
- Feature importance less actionable than counterfactual recommendations

**Finding 2:** Experience can degrade XAI effectiveness
- Increasing neurology experience + perceived explainability → worse performance
- Senior clinicians struggle integrating XAI into established workflows
- Novices benefit more from decision support

**Implication:** XAI must be **adaptive** to user expertise and context

### 7.2 The Confidence Paradox

**High Confidence:**
- ✓ Increases trust and agreement
- ✗ Leads to overreliance and reduced accuracy
- ✗ Decreases critical evaluation

**Low Confidence:**
- ✓ Promotes cautious, thorough evaluation
- ✓ Increases diagnostic time (more careful)
- ✗ Decreases trust and willingness to use system

**Implication:** Optimal confidence display is nuanced, not maximized

### 7.3 The Detail Paradox

**Clinicians want:**
- Minimal cognitive load and simple summaries
- **AND** ability to drill down into detailed explanations
- **AND** comprehensive data access for validation

**Solution:** Layered architecture with progressive disclosure

### 7.4 Gaps Between Research and Practice

**What Research Shows:**
- 80%+ positive or neutral perceptions of XAI
- SHAP and Grad-CAM effective explanation methods
- Improved diagnostic confidence with explanations

**What Practice Reveals:**
- Limited real-world adoption despite positive studies
- Small sample sizes limit generalizability
- Short-term evaluations miss long-term effects
- Lab settings don't reflect clinical complexity

**Missing Elements:**
- Long-term deployment studies
- Integration with existing workflows
- Impact on patient outcomes (not just clinician satisfaction)
- Cost-effectiveness analysis
- Regulatory compliance pathways

---

## 8. Future Research Directions

### 8.1 Methodological Improvements Needed

1. **Larger, More Diverse Samples**
   - Minimum 30 participants per study
   - Multiple institutions and healthcare systems
   - Range of experience levels and specialties

2. **Longitudinal Studies**
   - Track XAI effectiveness over months/years
   - Measure adaptation and learning curves
   - Assess sustained impact on decision quality

3. **Real-World Deployment Studies**
   - Move beyond controlled experiments
   - Integrate with actual EMR systems
   - Measure patient outcomes, not just process metrics

4. **Comparative Effectiveness Research**
   - Head-to-head XAI method comparisons
   - Cost-effectiveness analysis
   - Identification of optimal methods for specific contexts

### 8.2 Technical Research Priorities

1. **Personalized Explanations**
   - Adapt to user expertise level
   - Learn from individual interaction patterns
   - Support different explanation preferences

2. **Temporal Explanations**
   - Show patient trajectory and trend predictions
   - Explain why now vs. earlier/later
   - Dynamic adaptation as patient state changes

3. **Uncertainty Quantification**
   - Better calibration of confidence scores
   - Explanation of uncertainty sources
   - Visualization of prediction intervals

4. **Multi-Modal Integration**
   - Combine imaging, text, structured data explanations
   - Coherent cross-modal narratives
   - Handling of conflicting information

### 8.3 Human Factors Research Needs

1. **Cognitive Load Optimization**
   - Empirically determine optimal information density
   - Design patterns that minimize mental burden
   - Interruption management strategies

2. **Trust Calibration**
   - Mechanisms to prevent both under- and over-trust
   - Feedback on appropriateness of reliance
   - Training interventions for optimal use

3. **Communication Effectiveness**
   - Best practices for clinician-patient discussion of AI recommendations
   - Family involvement in AI-informed decisions
   - Handover protocols incorporating AI insights

4. **Workflow Integration**
   - Seamless EMR embedding
   - Minimal context switching
   - Just-in-time information delivery

### 8.4 Stakeholder Inclusion Gaps

**Critical Missing Stakeholder: Patients**
- Current research focuses almost exclusively on clinician perspectives
- Patients directly affected by AI-informed decisions
- Need for patient-facing explanations
- Shared decision-making requires patient understanding

**Research Questions:**
- How should AI recommendations be communicated to patients?
- What explanation formats do patients find trustworthy?
- How does AI disclosure affect patient satisfaction and adherence?
- Can patient feedback improve XAI systems?

---

## 9. Conclusions and Recommendations

### 9.1 State of the Field

**Maturity Assessment:**
- XAI methods technically sophisticated
- Clinical evaluation methodology improving but limited
- Real-world deployment sparse
- Regulatory frameworks emerging but incomplete

**Readiness for Clinical Adoption:**
- **Limited contexts:** Low-stakes, decision-support (not decision-making)
- **Requires:** Extensive validation, regulatory approval, clinician training
- **Not ready:** High-stakes autonomous decisions

### 9.2 Key Recommendations

**For Researchers:**
1. Prioritize application-grounded evaluation with adequate sample sizes
2. Conduct longitudinal studies in real clinical environments
3. Include diverse stakeholders, especially patients
4. Report negative results and limitations transparently
5. Move beyond accuracy to measure decision quality and patient outcomes

**For Developers:**
1. Adopt stakeholder-centric design from project inception
2. Build layered explanation systems with progressive disclosure
3. Integrate seamlessly with existing clinical workflows
4. Provide mechanisms for clinician feedback and system refinement
5. Design for appropriate trust, not maximum trust

**For Clinicians:**
1. Engage early in XAI development processes
2. Provide detailed feedback on explanation usefulness
3. Demand evidence of clinical benefit, not just technical performance
4. Participate in validation studies
5. Advocate for patient-centered explanation design

**For Healthcare Organizations:**
1. Establish clear evaluation criteria for XAI systems
2. Provide resources for clinician training
3. Support incremental, evidence-based deployment
4. Monitor long-term impact on decision quality
5. Create feedback loops for continuous improvement

**For Regulators:**
1. Develop clear guidance on XAI requirements
2. Balance innovation with patient safety
3. Require human-centered evaluation evidence
4. Establish post-market surveillance requirements
5. Support research on real-world XAI effectiveness

### 9.3 The Path Forward

**Short-term (1-2 years):**
- Standardize XAI evaluation methodologies
- Conduct larger-scale validation studies
- Develop reference implementations for common use cases
- Establish best practices for clinical XAI design

**Medium-term (3-5 years):**
- Deploy XAI systems in low-risk clinical contexts
- Build evidence base for effectiveness and safety
- Refine based on real-world experience
- Develop clinician training programs

**Long-term (5+ years):**
- Integrate XAI as standard component of clinical AI
- Extend to higher-stakes decision contexts
- Establish XAI as patient right (regulatory requirement)
- Achieve measurable improvements in patient outcomes

---

## 10. References and Key Papers

### Foundational Studies

1. **Kibria, M.G., Kucirka, L., & Mostafa, J. (2025).** "Usability Testing of an Explainable AI-enhanced Tool for Clinical Decision Support: Insights from the Reflexive Thematic Analysis." *IEEE Explore.*
   - Establishes four-dimensional explainability framework
   - 20 U.S. clinicians, qualitative thematic analysis
   - Key themes: understandability, trust, usability, usefulness

2. **Rezaeian, O., Bayrak, A.E., & Asan, O. (2025).** "Explainability and AI Confidence in Clinical Decision Support Systems: Effects on Trust, Diagnostic Performance, and Cognitive Load in Breast Cancer Care." *arXiv preprint.*
   - 28 healthcare professionals, interrupted time series design
   - High confidence → overreliance → reduced accuracy
   - Cognitive load and stress measurements

3. **Gambetti, A., Han, Q., Shen, H., & Soares, C. (2025).** "A Survey on Human-Centered Evaluation of Explainable AI Methods in Clinical Decision Support Systems." *PRISMA-guided systematic review.*
   - 31 studies analyzed comprehensively
   - 80%+ use SHAP/Grad-CAM (post-hoc, model-agnostic)
   - Sample size median: 16 participants
   - Identifies socio-technical gap

4. **Clark, J.N., Wragg, M., Nielsen, E., et al. (2024).** "Exploring the Requirements of Clinicians for Explainable AI Decision Support Systems in Intensive Care." *arXiv preprint.*
   - 7 ICU clinicians, group interviews
   - Three themes: decision factors, communication challenges, XAI requirements
   - Critical insight: "end-of-bed" information irreplaceable by data

### Methodology and Framework Papers

5. **Jin, W., Li, X., & Hamarneh, G. (2022).** "Evaluating Explainable AI on a Multi-Modal Medical Imaging Task: Can Existing Algorithms Fulfill Clinical Requirements?" *IEEE CVPR.*
   - Modality-Specific Feature Importance (MSFI) metric
   - 16 heatmap algorithms fail clinical requirements
   - Importance of multi-modal explanation

6. **Jin, W., Li, X., Fatehi, M., & Hamarneh, G. (2022).** "Guidelines and Evaluation of Clinical Explainable AI in Medical Image Analysis." *arXiv preprint.*
   - Clinical XAI Guidelines: 5 criteria framework
   - G1: Understandability, G2: Clinical relevance
   - G3: Truthfulness, G4: Informative plausibility, G5: Computational efficiency

7. **Wysocki, O., Davies, J.K., Vigo, M., et al. (2022).** "Assessing the communication gap between AI models and healthcare professionals: explainability, utility and trust in AI-driven clinical decision-making." *arXiv preprint.*
   - Pragmatic evaluation framework
   - Contradictory findings: explanations can reduce automation bias BUT also cause confirmation bias
   - Benefit for less experienced clinicians

### Application Studies

8. **Solomon, J., Jalilian, L., Vilesov, A., et al. (2024).** "2-Factor Retrieval for Improved Human-AI Decision Making in Radiology." *arXiv preprint.*
   - Novel 2FR approach: retrieval without processing
   - Radiologist accuracy improvement, especially with low confidence
   - Importance of physician verification

### Meta-Analyses and Surveys

9. **Bienefeld, N., Boss, J.M., Lüthy, R.L., et al. (2023).** "Solving the Explainable AI Conundrum by Bridging Clinicians' Needs and Developers' Goals." *npj Digital Medicine.*
   - Design tensions between developers (interpretability) and clinicians (plausibility)
   - 112 participants (clinicians and developers)
   - Highlights need for early stakeholder alignment

10. **Abraham, J., Bartek, B., Meng, A., et al. (2023).** "Integrating Machine Learning Predictions for Perioperative Risk Management." *Journal of Biomedical Informatics.*
    - 17 clinicians, cognitive walkthroughs
    - High agreement between ML and manual risk assessments
    - Streamlines report preparation and care planning

---

## Appendix A: XAI Methods Glossary

**SHAP (SHapley Additive exPlanations)**
- Game-theoretic approach to feature attribution
- Quantifies each feature's contribution to prediction
- Provides additive explanations: f(x) = E[f(x)] + Σ φᵢ
- Can be applied locally or globally

**LIME (Local Interpretable Model-agnostic Explanations)**
- Approximates model locally via surrogate interpretable model
- Perturbs input and observes output changes
- Fits simple model (e.g., linear) to local behavior
- Model-agnostic but computationally intensive

**Grad-CAM (Gradient-weighted Class Activation Mapping)**
- Uses gradients of target output w.r.t. feature maps
- Creates heatmap highlighting important image regions
- Specific to convolutional neural networks
- Widely used in medical imaging

**Integrated Gradients**
- Attributes prediction to input features via gradient accumulation
- Accumulates gradients from baseline to actual input
- Axiomatic (satisfies completeness and sensitivity)
- More stable than basic gradient methods

**Attention Mechanisms**
- Built into model architecture (intrinsic)
- Weights indicating importance of different inputs
- Common in Transformers and recurrent networks
- Directly interpretable as "what model attends to"

**Counterfactual Explanations**
- "What would need to change for different prediction?"
- Provides actionable insights
- Example-based reasoning
- Valued by clinicians for supporting decision alternatives

---

## Appendix B: Evaluation Frameworks Summary

### NASA Task Load Index (NASA-TLX)
- Mental demand, Physical demand, Temporal demand
- Performance, Effort, Frustration
- 5-point Likert scales
- Standard for cognitive load assessment

### CLIX-M (Clinician-Informed XAI Evaluation Checklist)
14 evaluation metrics across three categories:
- **Clinical attributes:** Clinical relevance, actionability, plausibility
- **Decision attributes:** Comprehensibility, trust, confidence
- **Model attributes:** Fidelity, robustness, efficiency

### Doshi-Velez & Kim Framework
Three evaluation levels:
- **Application-grounded:** Real tasks, real experts (required for clinical)
- **Human-grounded:** Proxy tasks, human participants
- **Proxy:** No human involvement, automated metrics

### Human-Centered Evaluation Methods
- **Think-Aloud (TA):** Verbalize thoughts during interaction
- **Interviews (I):** Structured, semi-structured, or unstructured
- **Surveys (S):** Quantitative questionnaires (Likert scales)
- **Focus Groups (FG):** Small group discussions

---

## Document Metadata

**Created:** November 30, 2025
**Sources:** 15+ peer-reviewed papers from arXiv (2022-2025)
**Primary Focus:** Explainable AI in clinical decision support
**Target Audience:** Researchers, developers, clinicians working on hybrid reasoning systems for acute care

**Search Strategy:**
- Query 1: "explainable AI" AND "clinical decision support"
- Query 2: "interpretable machine learning" AND "healthcare"
- Query 3: "attention mechanism" AND "medical diagnosis"
- Categories: cs.AI, cs.LG, cs.HC, cs.CV
- Sort: Relevance-based ranking

**Key Finding Summary:**
While XAI shows promise, the field needs larger validation studies, better workflow integration, focus on appropriate trust calibration rather than maximization, and inclusion of patient perspectives. The socio-technical gap between human needs and technical capabilities remains the central challenge to widespread clinical adoption.
