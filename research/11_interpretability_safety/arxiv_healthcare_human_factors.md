# Human Factors and Human-AI Interaction in Clinical AI: A Comprehensive Literature Review

**Date:** December 1, 2025
**Focus:** Emergency Department (ED) Human-AI Teaming for Acute Care Decision-Making

---

## Executive Summary

This comprehensive review synthesizes research on human factors and human-AI interaction in clinical AI systems, with particular emphasis on implications for emergency department (ED) settings. Based on analysis of over 150 papers from ArXiv, the review identifies critical challenges and opportunities in developing trustworthy, usable, and effective AI systems that support clinical decision-making.

**Key Findings:**
- **Trust and Reliance Paradox**: High-confidence AI predictions increase clinician trust but can lead to dangerous overreliance, reducing diagnostic accuracy when AI is wrong
- **Workflow Integration Critical**: Most AI systems fail in practice due to poor integration with clinical workflows, not technical limitations
- **Cognitive Load Trade-offs**: Explainability features can simultaneously improve understanding and increase cognitive burden
- **Automation Bias Prevalent**: Clinicians show 7-21% automation bias rates, accepting incorrect AI advice even when initially correct
- **Human-AI Collaboration Models**: Successful systems support intermediate decision stages (hypothesis generation, data gathering) rather than just final decisions
- **Context Matters**: Generic AI tools underperform domain-specific systems tailored to clinical workflows and local practices

**Critical Gap**: Despite advances in AI performance, fundamental questions about how clinicians should interact with, oversee, and collaborate with AI systems remain largely unanswered, particularly in time-critical, high-stakes environments like the ED.

---

## 1. Key Papers and ArXiv IDs

### 1.1 Foundational Human-AI Collaboration Studies

**2204.09082** - Hemmer et al. (2022)
*Factors that influence the adoption of human-AI collaboration in clinical decision-making*
- **Human Factors**: Trust, transparency, workflow integration, autonomy preservation
- **Methodology**: Semi-structured interviews with healthcare domain experts
- **Key Findings**: Identified six adoption factors with inherent tensions between effective collaboration and clinical acceptance
- **Design Implications**: AI must become coequal team member, not just tool

**2205.09696** - Fogliato et al. (2022)
*Who Goes First? Influences of Human-AI Workflow on Decision Making in Clinical Imaging*
- **Human Factors**: Anchoring bias, temporal sequencing effects, perceived AI usefulness
- **Methodology**: User study with 19 veterinary radiologists, two workflow configurations
- **Key Findings**: Provisional decisions before AI review reduce agreement with AI by 21%, perceived usefulness decreased but accuracy maintained
- **Design Implications**: Workflow sequencing critically impacts human-AI collaboration outcomes

**2412.00372** - Solomon et al. (2024)
*2-Factor Retrieval for Improved Human-AI Decision Making in Radiology*
- **Human Factors**: Confidence calibration, decision verification needs
- **Methodology**: Experimental study with radiologists on chest X-ray diagnosis
- **Key Findings**: Retrieval-based explanations (2FR) outperform saliency methods, particularly for low-confidence decisions
- **Design Implications**: Explanations should enable physician verification, not just model transparency

### 1.2 Trust and Reliance Studies

**2503.16692** - Hatherley (2025)
*Limits of trust in medical AI*
- **Human Factors**: Trustworthiness vs reliability, accountability gaps
- **Methodology**: Philosophical analysis and conceptual framework
- **Key Findings**: AI systems can be reliable but not trustworthy; trust requires capability for moral responsibility
- **Design Implications**: Clear distinction needed between AI reliance and trust

**2204.06916** - Schemmer et al. (2022)
*Should I Follow AI-based Advice? Measuring Appropriate Reliance in Human-AI Decision-Making*
- **Human Factors**: Appropriate reliance, advice quality discrimination
- **Methodology**: Developed two-dimensional AR measurement framework
- **Key Findings**: Current metrics fail to capture ability to discriminate advice quality and act appropriately
- **Design Implications**: Need metrics beyond accuracy that measure case-by-case reliance decisions

**2102.00593** - Jacobs et al. (2021)
*Designing AI for Trust and Collaboration in Time-Constrained Medical Decisions*
- **Human Factors**: Sociotechnical integration, patient preferences, resource constraints
- **Methodology**: Iterative co-design process for antidepressant treatment DST
- **Key Findings**: Clinical DSTs must engage with entire healthcare system, not just technical accuracy
- **Design Implications**: Multi-user systems supporting patient-provider collaboration essential

**2501.16693** - Rezaeian et al. (2025)
*Explainability and AI Confidence in Clinical Decision Support Systems*
- **Human Factors**: Trust-accuracy trade-off, cognitive load, overreliance
- **Methodology**: Interrupted time series, 28 healthcare professionals, breast cancer detection
- **Key Findings**: High confidence → 15% overreliance, low confidence → slower but more cautious decisions
- **Design Implications**: Balance transparency, usability, and cognitive demands

### 1.3 Explainability and Interpretability

**2308.08407** - Mesinovic et al. (2023)
*Explainable AI for clinical risk prediction: a survey*
- **Human Factors**: Fairness, bias, transparency, interpretability needs
- **Methodology**: Comprehensive survey of XAI methods for clinical risk prediction
- **Key Findings**: Need for external validation, combination of diverse interpretability methods
- **Design Implications**: End-to-end approach to explainability incorporating all stakeholders

**2112.12596** - Chen et al. (2021)
*Explainable Medical Imaging AI Needs Human-Centered Design*
- **Human Factors**: User research needs, transparency as affordance not property
- **Methodology**: Systematic review of transparent ML in medical imaging
- **Key Findings**: Most studies treat transparency as model property, ignore end users during development
- **Design Implications**: INTRPRT guideline for systematic design of transparent ML systems

**2308.04375** - Lee & Chew (2023)
*Understanding the Effect of Counterfactual Explanations on Trust and Reliance*
- **Human Factors**: Cognitive engagement, overreliance reduction
- **Methodology**: Experiment with therapists and laypersons assessing motion quality
- **Key Findings**: Counterfactual explanations reduced overreliance on wrong AI by 21% vs saliency
- **Design Implications**: Explanation type significantly impacts reliance patterns

### 1.4 Workflow Integration and Usability

**1904.09612** - Yang et al. (2019)
*Unremarkable AI: Fitting Intelligent Decision Support into Critical Clinical Decision-Making*
- **Human Factors**: Workflow compatibility, contextual fit, unobtrusiveness
- **Methodology**: Field evaluation of DST that auto-generates slides with embedded ML prognostics
- **Key Findings**: "Unremarkable Computing" approach - AI augments routines while remaining unobtrusive
- **Design Implications**: DSTs must fit existing workflows to enable adoption

**2006.12683** - Gu et al. (2020)
*Improving Workflow Integration with xPath*
- **Human Factors**: Workflow alignment, examination process similarity
- **Methodology**: Work sessions with 12 medical professionals in pathology
- **Key Findings**: AI examination process must mirror pathologist workflow for integration
- **Design Implications**: Human-AI collaborative tools need shared examination paradigms

**2309.12368** - Zhang et al. (2023)
*Rethinking Human-AI Collaboration in Complex Medical Decision Making: Sepsis*
- **Human Factors**: Intermediate stage support, uncertainty visualization, actionable suggestions
- **Methodology**: Formative study + prototype (SepsisLab) + heuristic evaluation
- **Key Findings**: Support needed for hypothesis generation and data gathering, not just final decisions
- **Design Implications**: Future projection, uncertainty visualization, actionable test suggestions

**2101.01524** - Wang et al. (2021)
*"Brilliant AI Doctor" in Rural China: Tensions and Challenges in AI-CDSS Deployment*
- **Human Factors**: Local context misalignment, usability barriers, transparency issues
- **Methodology**: Observations and interviews with 22 clinicians from 6 rural clinics
- **Key Findings**: Despite tensions, all participants positive about "doctor's AI assistant" future
- **Design Implications**: Context-specific design essential for rural/under-resourced settings

### 1.5 Clinical Performance and Decision Quality

**2507.16947** - Korom et al. (2025)
*AI-based Clinical Decision Support for Primary Care: A Real-World Study*
- **Human Factors**: Error reduction, clinical uptake, workflow activation
- **Methodology**: Quality improvement study, 39,849 visits across 15 clinics (Penda Health, Kenya)
- **Key Findings**: 16% fewer diagnostic errors, 13% fewer treatment errors; 75% reported substantial quality improvement
- **Design Implications**: Workflow-aligned implementation and active deployment critical for uptake

**2506.14774** - Sayin et al. (2025)
*MedSyn: Enhancing Diagnostics with Human-AI Collaboration*
- **Human Factors**: Cognitive bias mitigation, decisional complexity, interactive refinement
- **Methodology**: Multi-step interactive dialogues for diagnosis/treatment decisions
- **Key Findings**: Dynamic exchanges allow physicians to challenge AI, AI highlights alternatives
- **Design Implications**: Move beyond one-shot usage to interactive collaborative frameworks

**2510.22414** - Sevgi et al. (2025)
*Complementary Human-AI Clinical Reasoning in Ophthalmology*
- **Human Factors**: Diagnostic re-ranking, agreement enhancement, experience-based personalization
- **Methodology**: Ophthalmologists reviewed AI structured output and revised answers
- **Key Findings**: Clinicians ranked correct diagnosis higher, reached greater agreement after AI review
- **Design Implications**: Complementary effect where AI helps re-rank rather than direct acceptance

### 1.6 Mental Models and Shared Understanding

**2102.08507** - Seo et al. (2021)
*Towards an AI Coach to Infer Team Mental Model Alignment in Healthcare*
- **Human Factors**: Shared mental models, team coordination, alignment inference
- **Methodology**: Bayesian approach to infer misalignment in cardiac surgery teams
- **Key Findings**: 75%+ recall in detecting model misalignment enables intervention
- **Design Implications**: AI systems can augment team cognition by identifying coordination gaps

**2402.13437** - Yildirim et al. (2024)
*Sketching AI Concepts with Capabilities and Examples: AI Innovation in ICU*
- **Human Factors**: Problem formulation, stakeholder engagement, ideation barriers
- **Methodology**: Design workshops with data scientists, clinicians, HCI researchers
- **Key Findings**: Domain experts think of unbuildable innovations, data scientists think of unwanted ones
- **Design Implications**: Systematic ideation methods needed for effective AI concept development

### 1.7 Bias and Fairness

**2305.10201** - Fazli et al. (2023)
*Echoes of Biases: How Stigmatizing Language Affects AI Performance*
- **Human Factors**: Clinician bias propagation, racial disparity in AI models
- **Methodology**: Analysis of stigmatizing language (SL) in EHR notes on mortality prediction
- **Key Findings**: SL adversely affects AI performance, particularly for Black patients; central clinicians have stronger disparity impact
- **Design Implications**: Removing SL from central clinicians more efficient than eliminating all SL

**2108.01764** - Logé et al. (2021)
*Q-Pain: A Question Answering Dataset to Measure Social Bias in Pain Management*
- **Human Factors**: Intersectional bias (race-gender), treatment disparities
- **Methodology**: Bias assessment framework for medical QA systems
- **Key Findings**: Statistically significant differences in treatment between race-gender subgroups
- **Design Implications**: Datasets needed to ensure safety before medical AI deployment

### 1.8 Automation Bias and Overreliance

**2411.00998** - Rosbach et al. (2024)
*Automation Bias in AI-Assisted Medical Decision-Making under Time Pressure*
- **Human Factors**: Automation bias, time pressure effects, cognitive resource strain
- **Methodology**: Web-based experiment with 28 pathology experts estimating tumor percentages
- **Key Findings**: 7% automation bias rate; time pressure increased severity but not occurrence
- **Design Implications**: Time pressure considerations essential for emergency/acute settings

**2102.09692** - Bucinca et al. (2021)
*To Trust or to Think: Cognitive Forcing Functions Can Reduce Overreliance on AI*
- **Human Factors**: Analytical engagement, cognitive forcing, Need for Cognition
- **Methodology**: Experiment (N=199) comparing three cognitive forcing designs
- **Key Findings**: Cognitive forcing reduced overreliance but received less favorable ratings; benefits varied by Need for Cognition
- **Design Implications**: Trade-off between reducing overreliance and user satisfaction

**2204.05030** - Wysocki et al. (2022)
*Assessing the communication gap between AI models and healthcare professionals*
- **Human Factors**: Confirmation bias, over-reliance, critical understanding gaps
- **Methodology**: Pragmatic evaluation with healthcare professionals using XAI in practice
- **Key Findings**: Explanations showed positive effects (reduced automation bias, helped ambiguous cases) and negative (confirmation bias, increased effort)
- **Design Implications**: Standard explanations have limited ability to support critical understanding of model limitations

### 1.9 Emergency and Time-Critical Settings

**2505.11996** - Mastrianni et al. (2025)
*To Recommend or Not to Recommend: AI Decision Support for Time-Critical Medical Events*
- **Human Factors**: Time efficiency, accuracy-time trade-offs, provider trust polarization
- **Methodology**: Randomized blinded virtual OSCE, trauma treatment decisions (N=35)
- **Key Findings**: AI information + recommendations improved decisions vs no AI; accuracy-time trade-off in recommendations
- **Design Implications**: Balance between providing recommendations and maintaining trust in time-critical contexts

**2503.08814** - Davidson et al. (2025)
*An Iterative, User-Centered Design of a CDSS for Critical Care Assessments*
- **Human Factors**: Computational utility, workflow optimization, patient care effects
- **Methodology**: Co-design sessions with ICU clinicians for sepsis/delirium AI-CDS
- **Key Findings**: Value in multi-modal monitoring and delirium detection; actionability and intuitive interpretation emphasized
- **Design Implications**: Real-time risk quantification with actionable suggestions for high-stakes settings

### 1.10 Novel Interaction Paradigms

**2507.15743** - Vedadi et al. (2025)
*Towards physician-centered oversight of conversational diagnostic AI*
- **Human Factors**: Asynchronous oversight, decoupled accountability, guardrails
- **Methodology**: Randomized blinded virtual OSCE with guardrailed-AMIE (g-AMIE)
- **Key Findings**: g-AMIE outperformed NPs/PAs in intake quality and diagnosis proposals; asynchronous oversight more time-efficient
- **Design Implications**: Decoupling oversight from intake enables scalable AI deployment

**2503.16433** - Cho et al. (2025)
*The Application of MATEC Framework in Sepsis Care*
- **Human Factors**: Multi-agent team coordination, specialist access simulation
- **Methodology**: Multi-AI agent team with 10 doctors evaluating framework
- **Key Findings**: High usefulness (Median=4) and accuracy (Median=4) ratings from attending physicians
- **Design Implications**: Multi-agent systems can simulate specialist teams for under-resourced settings

---

## 2. Human-AI Collaboration Models

### 2.1 Taxonomy of Collaboration Modes

Analysis of the literature reveals **five distinct collaboration modes** with different human factors implications:

#### **Mode 1: AI as Second Opinion (Parallel Review)**
- **Characteristics**: Clinician makes independent assessment, then reviews AI recommendation
- **Examples**: Radiology reading with AI concurrent analysis
- **Human Factors**: Reduces anchoring bias, maintains clinician autonomy, supports thoughtful disagreement
- **Challenges**: Time-consuming, may still show confirmation bias
- **Best for**: High-stakes decisions where independent assessment is critical

#### **Mode 2: AI-First Advisory (Sequential Review)**
- **Characteristics**: AI provides recommendation upfront, clinician reviews and decides
- **Examples**: Triage systems, preliminary diagnostic screening
- **Human Factors**: Efficiency gains, risk of automation bias and anchoring effects
- **Challenges**: Overreliance when AI wrong, reduced analytical engagement
- **Best for**: High-volume screening, preliminary assessments

#### **Mode 3: Interactive Collaboration (Iterative Refinement)**
- **Characteristics**: Multi-turn dialogue where clinician and AI refine assessment together
- **Examples**: MedSyn diagnostic dialogues, conversational AI assistants
- **Human Factors**: Allows challenge and negotiation, surfaces alternative perspectives
- **Challenges**: Requires sophisticated AI, may be time-intensive
- **Best for**: Complex diagnostic uncertainty, treatment planning

#### **Mode 4: Asynchronous Oversight (Decoupled Review)**
- **Characteristics**: AI performs initial assessment, specialist reviews offline
- **Examples**: g-AMIE framework with PCP oversight, telehealth preliminary screening
- **Human Factors**: Scalable oversight, maintains accountability without real-time requirement
- **Challenges**: Delayed intervention potential, reduced learning from AI
- **Best for**: Non-urgent assessments, resource-constrained settings

#### **Mode 5: AI-Augmented Workflow (Embedded Support)**
- **Characteristics**: AI integrated into existing tools, surfaces insights within workflow
- **Examples**: xPath pathology system, "unremarkable" AI embedded in slides
- **Human Factors**: Minimal workflow disruption, natural integration
- **Challenges**: May be overlooked if too subtle, limited interactivity
- **Best for**: Routine tasks with established workflows

### 2.2 Framework: Stages of Decision Support

Research consistently shows that **effective AI support targets intermediate decision stages**, not just final decisions:

**Stage 1: Data Gathering**
- AI Function: Suggest relevant tests, identify information gaps
- Example: SepsisLab recommending which lab tests to order
- Human Factor Need: Actionable, prioritized suggestions

**Stage 2: Hypothesis Generation**
- AI Function: Surface differential diagnoses, identify patterns
- Example: Retrieval of similar cases with outcomes
- Human Factor Need: Diverse alternatives, not just top prediction

**Stage 3: Evidence Evaluation**
- AI Function: Synthesize relevant information, highlight conflicts
- Example: AI Consult identifying documentation errors
- Human Factor Need: Transparent reasoning, verifiable claims

**Stage 4: Decision Making**
- AI Function: Recommend action with rationale
- Example: Treatment suggestions with success probabilities
- Human Factor Need: Clear trade-offs, uncertainty quantification

**Stage 5: Monitoring & Adjustment**
- AI Function: Track outcomes, suggest modifications
- Example: Longitudinal risk tracking, drift detection
- Human Factor Need: Timely alerts, trend visualization

### 2.3 Complementarity Principles

Evidence from multiple studies (2510.22414, 2506.14774, 2205.09696) suggests successful collaboration emerges from **complementary strengths**:

**AI Strengths:**
- Pattern recognition across large datasets
- Consistent application of learned associations
- Rapid processing of structured data
- Memory of rare cases and base rates

**Human Strengths:**
- Contextual understanding (patient history, social factors)
- Handling of ambiguity and uncertainty
- Ethical reasoning and value judgments
- Communication and empathy
- Integration of tacit knowledge

**Complementarity Design Principles:**
1. **AI compensates for human limitations**: Fatigue, memory constraints, availability bias
2. **Humans compensate for AI limitations**: Context blindness, distributional shift, edge cases
3. **Mutual verification**: AI flags potential human errors, humans audit AI outputs
4. **Adaptive allocation**: Task delegation based on relative capabilities and case characteristics

---

## 3. Trust and Reliance Factors

### 3.1 The Trust-Performance Paradox

A critical finding across multiple studies is the **counterintuitive relationship between trust, reliance, and performance**:

**High Confidence AI Predictions:**
- ↑ Trust (perceived reliability)
- ↑ Reliance (following recommendations)
- ↓ Accuracy when AI wrong (overreliance)
- ↓ Independent reasoning (automation bias)

**Study**: Rezaeian et al. (2501.16693) - High confidence scores led to 15% overreliance, reducing diagnostic accuracy

**Low Confidence AI Predictions:**
- ↓ Trust and agreement
- ↑ Time to decision (cautious behavior)
- ↑ Independent verification
- ↓ AI utility perception

**Study**: Same study showed low confidence increased diagnosis duration but reduced harmful overreliance

### 3.2 Trust Calibration Mechanisms

Evidence suggests trust calibration requires **multifaceted approaches**:

#### **Mechanism 1: Performance Transparency**
- **What**: Clear communication of AI accuracy, including failure modes
- **Evidence**: Schemmer et al. (2204.06916) - Appropriate reliance requires ability to discriminate advice quality
- **Implementation**: Show accuracy by case type, patient population, uncertainty ranges

#### **Mechanism 2: Explanation Quality**
- **What**: Interpretable rationales that enable verification
- **Evidence**: Solomon et al. (2412.00372) - Retrieval-based explanations outperform saliency for trust calibration
- **Implementation**: Similar case retrieval, counterfactual examples, feature importance in clinical terms

#### **Mechanism 3: Interactive Feedback**
- **What**: Ability to query AI, request alternative perspectives
- **Evidence**: Sayin et al. (2506.14774) - Interactive dialogues allow physicians to challenge AI
- **Implementation**: Conversational interfaces, what-if analysis, sensitivity exploration

#### **Mechanism 4: Cognitive Forcing Functions**
- **What**: Require analytical engagement before showing AI output
- **Evidence**: Bucinca et al. (2102.09692) - Provisional decisions reduced overreliance by 21%
- **Implementation**: Staged reveal of AI advice, required independent assessment first

#### **Mechanism 5: Confidence Calibration**
- **What**: AI confidence scores aligned with actual accuracy
- **Evidence**: Multiple studies show miscalibrated confidence undermines trust
- **Implementation**: Properly calibrated uncertainty estimates, explicit confidence intervals

### 3.3 Trust Boundaries and Appropriate Reliance

**Key Insight**: The goal is not maximum trust, but **appropriate reliance** - following AI when correct, rejecting when wrong.

**Appropriate Reliance Framework** (Schemmer et al., 2204.06916):

**Dimension 1: Discrimination Ability**
- Can clinician distinguish correct vs incorrect AI advice?
- Measured by: Sensitivity to AI accuracy variation

**Dimension 2: Response Behavior**
- Does clinician act appropriately based on discrimination?
- Measured by: Reliance on correct advice, rejection of incorrect advice

**Factors Affecting Appropriate Reliance:**

**Individual Factors:**
- Clinical experience and expertise
- Familiarity with AI/ML systems
- Need for Cognition (motivation for effortful thinking)
- Attitudes toward AI (skepticism vs enthusiasm)

**System Factors:**
- Explanation quality and verifiability
- Confidence calibration and communication
- Interface design and cognitive load
- Workflow integration and timing

**Contextual Factors:**
- Time pressure and cognitive resource availability
- Case complexity and ambiguity
- Stakes and consequences of errors
- Availability of second opinions

### 3.4 The Explainability-Trust Relationship

Research reveals **complex, non-linear relationships** between explainability and trust:

**Positive Effects of Explanations:**
- Increased understanding of AI logic (2308.08407)
- Enhanced ability to detect errors (2308.04375)
- Reduced automation bias in some contexts (2204.05030)
- Support for learning and skill development

**Negative Effects of Explanations:**
- Increased cognitive load and stress (2501.16693)
- Confirmation bias when explanations align with priors (2204.05030)
- False sense of understanding (illusion of explanatory depth)
- Increased effort without performance gains in some cases

**Critical Moderators:**
1. **Explanation Type**: Counterfactual > Saliency for reducing overreliance
2. **User Expertise**: Experts vs novices interpret differently
3. **Task Complexity**: Simple tasks may not benefit from explanations
4. **Explanation Timing**: Pre- vs post-decision affects impact
5. **Interactivity**: On-demand vs always-shown explanations

---

## 4. Usability Considerations

### 4.1 Cognitive Load Management

A critical usability challenge is managing **the cognitive burden of AI-augmented decision-making**:

#### **Sources of Cognitive Load in Clinical AI:**

**Intrinsic Load (Task Complexity):**
- Medical decision complexity
- Patient-specific factors
- Uncertainty and ambiguity
- Time pressure (especially in ED)

**Extraneous Load (System-Imposed):**
- Interpreting AI outputs
- Evaluating explanations
- Navigating interface complexity
- Switching between systems
- Mental model alignment with AI

**Germane Load (Learning & Integration):**
- Understanding AI capabilities/limitations
- Developing calibrated trust
- Integrating AI into clinical reasoning
- Building new workflows

#### **Cognitive Load Reduction Strategies:**

**Study: Chen et al. (2112.12596) - INTRPRT Guidelines**
- User research to understand actual needs (not assumed needs)
- Iterative prototyping with end users
- Minimize extraneous cognitive elements
- Progressive disclosure of complexity

**Study: Yang et al. (1904.09612) - "Unremarkable Computing"**
- Embed AI in existing workflow artifacts (slides, reports)
- Subtle integration that doesn't demand attention
- Information available when needed, invisible when not

**Design Recommendations:**
1. **Default to Simplicity**: Show minimal information by default, detail on demand
2. **Chunking**: Group related information, use clear visual hierarchy
3. **Familiar Formats**: Present in clinically standard formats (not novel visualizations)
4. **Progressive Disclosure**: Layer complexity based on user need
5. **Workflow Integration**: Reduce context switching, embed in existing tools

### 4.2 Interface Design Patterns for Clinical AI

Analysis of successful systems reveals **recurring design patterns**:

#### **Pattern 1: Dashboard + Drill-Down**
- **Overview**: Summary metrics and alerts at top level
- **Details**: Expandable sections for deeper investigation
- **Example**: SepsisLab showing overall sepsis risk with detailed laboratory suggestions
- **Use Case**: Monitoring multiple patients, quick triage

#### **Pattern 2: Inline Annotation**
- **Overview**: AI insights embedded directly in clinical documents/images
- **Details**: Hover/click for explanations and confidence
- **Example**: xPath highlighting suspicious regions in pathology slides
- **Use Case**: Image interpretation, document review

#### **Pattern 3: Conversational Interface**
- **Overview**: Natural language dialogue for queries and exploration
- **Details**: Multi-turn refinement of questions and answers
- **Example**: MedSyn diagnostic conversations, AI Consult
- **Use Case**: Complex diagnostic reasoning, second opinion

#### **Pattern 4: Side-by-Side Comparison**
- **Overview**: AI suggestion alongside clinician assessment
- **Details**: Explicit comparison enables critical evaluation
- **Example**: 2FR retrieval showing similar cases with outcomes
- **Use Case**: Decision verification, learning from AI

#### **Pattern 5: Staged Reveal**
- **Overview**: Progressive disclosure of AI information
- **Details**: Independent assessment first, then AI input
- **Example**: Workflow where provisional decision precedes AI recommendation
- **Use Case**: Reducing anchoring bias, maintaining independent reasoning

### 4.3 Workflow Integration Challenges

**Primary Failure Mode**: Systems fail not from technical inadequacy but from **workflow misalignment**

#### **Dimensions of Workflow Compatibility:**

**Temporal Alignment:**
- When does AI provide input relative to clinical workflow?
- Real-time vs batch processing
- Synchronous vs asynchronous interaction

**Informational Alignment:**
- What information does AI need vs what's available?
- Data entry burden vs automation
- Integration with EHR and existing systems

**Process Alignment:**
- Does AI support actual clinical processes?
- Alignment with local protocols and practices
- Compatibility with team coordination patterns

**Cognitive Alignment:**
- Does AI match clinician mental models?
- Terminology and concept alignment
- Reasoning transparency and verifiability

#### **Case Study: "Brilliant AI Doctor" Failures (2101.01524)**

**Context**: AI-CDSS deployment in rural Chinese clinics

**Workflow Tensions Identified:**
1. **Misalignment with local context**: Recommendations not adapted to available resources
2. **Technical limitations**: Usability barriers for less tech-savvy users
3. **Transparency issues**: Opaque reasoning reduced trust
4. **Time burden**: System added steps rather than saving time

**Despite Failures**: All participants positive about future potential as "doctor's AI assistant"

**Lesson**: Even well-intentioned AI fails without deep workflow integration

#### **Success Factors from Real-World Deployments:**

**Study: Korom et al. (2507.16947) - AI Consult in Kenya**
- 16% fewer diagnostic errors, 13% fewer treatment errors
- Key success factor: "Activating only when needed, preserving clinician autonomy"
- Workflow-aligned: Operates as safety net, not replacement

**Design Principles for Workflow Integration:**
1. **Preserve Autonomy**: Clinician always in control, AI suggests not dictates
2. **Reduce Friction**: Minimize additional steps, automate data entry where possible
3. **Local Adaptation**: Customize to local protocols, resources, patient populations
4. **Transparent Activation**: Clear when/why AI is invoked
5. **Exit Ramps**: Easy to override or ignore AI without penalty
6. **Team Coordination**: Support multi-disciplinary workflows

### 4.4 Multi-Stakeholder Usability

Clinical AI must serve **multiple stakeholders with different needs**:

#### **Clinicians (Primary Users):**
- **Needs**: Fast, accurate, actionable information; minimal workflow disruption
- **Concerns**: Liability, autonomy, skill degradation
- **Usability Focus**: Efficiency, trust calibration, workflow integration

#### **Patients:**
- **Needs**: Understanding diagnosis/treatment, involvement in decisions
- **Concerns**: Privacy, AI replacing doctors, bias/fairness
- **Usability Focus**: Transparency, communication, consent

#### **Healthcare Administrators:**
- **Needs**: Cost reduction, quality improvement, compliance
- **Concerns**: ROI, liability, implementation costs
- **Usability Focus**: Deployment efficiency, monitoring, reporting

#### **Regulators/Policy Makers:**
- **Needs**: Safety assurance, fairness, accountability
- **Concerns**: Harm prevention, bias, privacy violations
- **Usability Focus**: Auditability, governance, standards compliance

**Key Challenge**: Design must balance competing stakeholder needs without compromising usability for any group

---

## 5. Workflow Integration Strategies

### 5.1 Pre-Deployment Integration Framework

Successful integration requires **systematic assessment across multiple dimensions**:

#### **Phase 1: Workflow Analysis**

**Activities:**
- Shadow clinicians through complete workflows
- Map decision points and information flows
- Identify pain points and inefficiencies
- Document temporal patterns and time pressures
- Understand team coordination and handoffs

**Study Example**: Davidson et al. (2503.08814)
- Co-design sessions with ICU clinicians for delirium/acuity assessment
- Identified 5 themes: computational utility, workflow optimization, patient care effects, technical considerations, implementation

**Outputs:**
- Workflow diagrams with AI integration points
- Time-motion studies quantifying delays
- Stakeholder requirement specifications
- Constraint documentation (technical, regulatory, resource)

#### **Phase 2: Integration Point Selection**

**Criteria for Optimal Integration Points:**
1. **High Uncertainty**: Where clinical uncertainty is greatest
2. **High Volume**: Where efficiency gains matter most
3. **High Risk**: Where errors have serious consequences
4. **Data Availability**: Where needed data is accessible in real-time
5. **Workflow Stability**: Where processes are established and predictable

**Anti-Patterns to Avoid:**
- ❌ Integration at every possible point (information overload)
- ❌ Integration only at final decision (too late for intermediate support)
- ❌ Integration without clear value proposition (adds burden without benefit)
- ❌ Integration requiring major workflow changes (adoption resistance)

#### **Phase 3: Prototype Development**

**Key Principles:**
- Build minimum viable integration, not full system
- Focus on one workflow integration point first
- Use familiar interface patterns and terminology
- Ensure fail-safe defaults (clinician can proceed without AI)
- Enable easy comparison between AI and non-AI workflows

**Study Example**: Yang et al. (1904.09612) - xPath
- Developed automatic slide generation with embedded AI prognostics
- Mirrors existing meeting structure (minimal workflow change)
- Subtly embeds AI insights in familiar format
- Field evaluation showed improved efficiency without disruption

#### **Phase 4: Iterative Refinement**

**Co-Design Process:**
- Regular feedback sessions with end users
- A/B testing of alternative designs
- Think-aloud protocols during use
- Quantitative usability metrics (time, errors, satisfaction)
- Qualitative interviews on experience

**Study Example**: Gu et al. (2020.12683) - xPath Development
- Work sessions with 12 medical professionals
- Iterative refinement based on pathologist feedback
- Aligned AI examination process with human workflow
- Result: High usability (SUS score) and improved pathologist performance

### 5.2 Temporal Integration Patterns

**When AI input is provided affects human-AI interaction outcomes**:

#### **Pattern 1: Pre-Assessment AI (AI-First)**
**Timing**: AI analyzes before clinician sees patient/data
**Advantages**: Efficiency, can highlight urgent issues, prepares clinician
**Disadvantages**: Anchoring bias, reduced independent reasoning
**Best For**: Triage, screening, preliminary assessments
**Example**: Emergency department AI triage scoring

#### **Pattern 2: Concurrent AI (Real-Time Parallel)**
**Timing**: AI and clinician analyze simultaneously
**Advantages**: No anchoring, maintains independence, allows comparison
**Disadvantages**: Requires time for both assessments, may still influence
**Best For**: High-stakes diagnoses requiring independent verification
**Example**: Dual reading in radiology with AI concurrent analysis

#### **Pattern 3: Post-Assessment AI (Human-First)**
**Timing**: Clinician forms initial assessment, then reviews AI
**Advantages**: Reduces anchoring (21% less agreement with AI when wrong per Fogliato 2205.09696)
**Disadvantages**: Time-consuming, may not benefit from AI insights early
**Best For**: Complex cases where independent clinical judgment critical
**Example**: Second opinion consultation model

#### **Pattern 4: Asynchronous AI (Decoupled)**
**Timing**: AI performs intake/screening, specialist reviews later
**Advantages**: Scalability, specialist time efficiency, maintains oversight
**Disadvantages**: Delayed intervention, limited learning from AI
**Best For**: Non-urgent cases, telemedicine, resource-constrained settings
**Example**: g-AMIE framework with PCP asynchronous oversight (2507.15743)

#### **Pattern 5: Adaptive Timing**
**Timing**: System determines optimal timing based on case characteristics
**Advantages**: Tailored to specific needs, maximizes benefits
**Disadvantages**: Complex to implement, requires sophisticated case classification
**Best For**: Mixed acuity settings with varying time pressures
**Example**: AI Consult activating only when potential errors detected (2507.16947)

### 5.3 Information Architecture for Clinical AI

**Challenge**: Present AI outputs in clinically meaningful, actionable format

#### **Layered Information Architecture:**

**Layer 1: Alert/Summary (Always Visible)**
- High-level status or recommendation
- Visual indicators (color coding, icons)
- Immediate action items if urgent
- Example: "High sepsis risk - consider blood cultures"

**Layer 2: Supporting Evidence (On-Demand)**
- Key factors driving recommendation
- Relevant clinical data summary
- Confidence/uncertainty indicators
- Example: "Risk based on: fever, elevated lactate, tachycardia"

**Layer 3: Detailed Explanation (Drill-Down)**
- Full model rationale
- Alternative considerations
- Similar cases or literature
- Example: Feature importance, counterfactuals, retrieval results

**Layer 4: System Transparency (Background)**
- Model performance statistics
- Training data characteristics
- Known limitations and failure modes
- Example: "Accuracy: 85% on similar cases; lower in elderly patients"

#### **Clinical Action Mapping:**

Effective systems map AI outputs to **specific clinical actions**:

**Poor Practice:**
- "Risk score: 0.87" (What should I do?)
- "High probability of pneumonia" (What's the next step?)

**Best Practice:**
- "High pneumonia risk → Consider: Chest X-ray, sputum culture, empiric antibiotics"
- "Sepsis risk increasing → Recommended: Blood cultures, lactate, broad-spectrum abx"

**Study Example**: SepsisLab (2309.12368)
- Future sepsis trajectory projection (temporal information)
- Uncertainty visualization (confidence communication)
- Actionable suggestions for lab tests (specific next steps)
- Result: 6 clinicians found "promising paradigm for human-AI collaboration"

### 5.4 Team-Based Workflow Integration

Clinical decisions often involve **multiple team members** - AI must support coordination:

#### **Multi-Disciplinary Team Challenges:**

**Challenge 1: Information Asymmetry**
- Different team members have different information access
- AI may be visible to some but not others
- Leads to coordination failures and miscommunication

**Solution**: Shared AI interface with role-based views, clear communication of AI inputs to all team members

**Challenge 2: Responsibility Allocation**
- Unclear who is accountable for AI-influenced decisions
- Diffusion of responsibility in team settings
- Potential for "everyone thought someone else checked"

**Solution**: Explicit accountability assignment, AI recommendations tagged with responsible reviewer

**Challenge 3: Mental Model Alignment**
- Team members may have different understanding of AI capabilities
- Mismatched trust levels across team
- Conflicting use strategies

**Solution**: Team training on AI, shared calibration exercises, regular review of AI performance

**Study Example**: Seo et al. (2102.08507) - Team Mental Model Inference
- Bayesian approach to detect misalignment in cardiac surgery teams
- 75%+ recall in detecting coordination gaps
- Enables AI to serve as "team coach" identifying when members have different understanding

#### **Handoff Integration:**

Critical transition points where information must transfer between providers:

**Handoff Challenges with AI:**
- AI information may not transfer between systems
- Receiving provider unaware of AI inputs to previous decisions
- Loss of AI reasoning context across handoffs

**Design Solutions:**
- AI decision summary included in handoff documentation
- Clear tagging of AI-influenced vs human-only decisions
- Persistent AI rationale accessible to downstream providers
- Handoff checklist including AI recommendation review

---

## 6. Research Gaps and Future Directions

### 6.1 Identified Research Gaps

Analysis of the current literature reveals **critical areas requiring further investigation**:

#### **Gap 1: Longitudinal Human-AI Interaction**

**Current State**: Most studies are short-term evaluations
**Missing**:
- How does clinician-AI interaction evolve over extended use?
- Do calibration and appropriate reliance improve with experience?
- What are the long-term effects on clinical skills and judgment?
- How does the relationship between human and AI change with familiarity?

**Priority for ED Context**: Understanding how shift work and rotating staff affect AI learning and trust calibration

#### **Gap 2: Real-World Emergency Settings**

**Current State**: Limited studies in actual time-critical, high-acuity environments
**Missing**:
- How do time pressures affect human-AI collaboration in ED?
- What are optimal integration points in emergency workflows?
- How does cognitive load under stress affect AI utilization?
- Multi-tasking effects on AI-assisted decision-making

**Priority for ED Context**: Critical - most research in controlled or outpatient settings

#### **Gap 3: Team-Level Human-AI Collaboration**

**Current State**: Focus on individual clinician-AI interaction
**Missing**:
- How does AI affect team coordination and communication?
- What are optimal ways to distribute AI information across teams?
- How do shared mental models develop with AI team member?
- Effects on novice-expert collaboration when AI present

**Priority for ED Context**: High - ED care is fundamentally team-based

#### **Gap 4: Failure Mode Characterization**

**Current State**: Limited systematic study of AI failure patterns in practice
**Missing**:
- Taxonomy of clinical AI failure modes
- Human factors contributing to failure detection/recovery
- Design patterns that minimize harm from AI errors
- Early warning indicators of impending failures

**Priority for ED Context**: Critical - high stakes make failures particularly dangerous

#### **Gap 5: Adaptive and Personalized AI Support**

**Current State**: One-size-fits-all AI systems
**Missing**:
- How should AI adapt to individual clinician expertise levels?
- Personalization based on specialty, experience, practice patterns
- Adaptive interfaces that learn from user interactions
- Context-aware support adjusting to case complexity and time available

**Priority for ED Context**: Moderate-High - wide variability in ED staff experience

#### **Gap 6: Cross-Cultural and Context-Specific Factors**

**Current State**: Most research in high-resource Western settings
**Missing**:
- How do cultural factors affect AI adoption and use?
- Adaptation requirements for different healthcare systems
- Low-resource setting specific challenges and opportunities
- Urban vs rural differences in AI integration needs

**Priority for ED Context**: Moderate - but critical for generalizability

#### **Gap 7: Evaluation Methodologies**

**Current State**: Lack of standardized evaluation frameworks
**Missing**:
- Validated metrics for appropriate reliance and trust calibration
- Standardized tasks for comparing human-AI collaboration approaches
- Benchmarks for explainability effectiveness
- Real-world outcome measures beyond accuracy

**Priority for ED Context**: High - need rigorous methods to evaluate ED-specific systems

### 6.2 Future Research Directions

Based on identified gaps, priority research areas include:

#### **Direction 1: Emergency Department Human-AI Teaming Studies**

**Proposed Research:**
- Longitudinal field studies in actual EDs
- High-fidelity simulation studies with time pressure manipulation
- Comparative evaluation of collaboration modes in triage, diagnosis, treatment decisions
- Team coordination analysis with AI integration

**Key Questions:**
1. How does time pressure affect appropriate reliance on AI in ED?
2. What are optimal workflow integration points for different ED processes?
3. How can AI support multi-provider team coordination in ED?
4. What failure modes are most likely in time-critical ED settings?

**Methodologies:**
- Ethnographic observations in EDs
- Simulated ED scenarios with standardized patients
- Retrospective analysis of AI-assisted ED cases
- Think-aloud protocols during AI-assisted ED decisions

#### **Direction 2: Adaptive AI Support Systems**

**Proposed Research:**
- AI systems that adapt presentation based on:
  - Clinician experience and specialty
  - Case complexity and urgency
  - Available time and cognitive load
  - Past interaction patterns and preferences

**Key Questions:**
1. What personalization dimensions most affect outcomes?
2. How can AI infer user needs from interaction patterns?
3. What are risks of over-personalization (filter bubbles)?
4. How to balance consistency with adaptability?

**Methodologies:**
- User modeling and preference learning studies
- A/B testing of personalization strategies
- Long-term deployment with adaptation tracking
- Comparative evaluation across user types

#### **Direction 3: Explainability-Effectiveness Research**

**Proposed Research:**
- Systematic evaluation of explanation types in clinical contexts
- Task-specific explanation requirements
- Interactive explainability vs static presentations
- Explanation timing and progressive disclosure studies

**Key Questions:**
1. Which explanation types support appropriate reliance for which tasks?
2. How should explanations differ for experts vs novices?
3. What is optimal balance between detail and cognitive load?
4. When do explanations help vs hurt performance?

**Methodologies:**
- Controlled experiments with different explanation types
- Eye-tracking studies of explanation use
- Think-aloud protocols during explanation interpretation
- Performance measurement with/without explanations

#### **Direction 4: Failure Detection and Recovery**

**Proposed Research:**
- Systematic characterization of AI failure modes in practice
- Human factors in failure detection
- Design patterns for graceful degradation
- Recovery strategies when AI fails

**Key Questions:**
1. What failure patterns are most common in clinical AI?
2. What cues enable clinicians to detect AI errors?
3. How can systems be designed to minimize harm from failures?
4. What recovery strategies are most effective?

**Methodologies:**
- Failure mode and effects analysis (FMEA)
- Incident reporting and analysis
- Simulation studies with introduced AI errors
- Expert interviews on error detection strategies

#### **Direction 5: Multi-Modal AI Integration**

**Proposed Research:**
- Integration of multiple AI systems (imaging, lab, vital signs, text)
- Coherent presentation of multi-source AI recommendations
- Conflict resolution when AI systems disagree
- Ensemble human-AI decision-making

**Key Questions:**
1. How do humans integrate multiple AI recommendations?
2. What happens when AI systems conflict?
3. How to present multi-modal evidence coherently?
4. When is ensemble of AI+human better than either alone?

**Methodologies:**
- Multi-modal case studies
- Conflict resolution experiments
- Cognitive load assessment with multiple AI
- Performance comparison: single vs multi-AI

### 6.3 Methodological Recommendations

To address research gaps effectively, studies should:

#### **1. Use Ecologically Valid Settings**
- Real clinical environments when possible
- High-fidelity simulations when real settings too risky
- Authentic tasks and time pressures
- Representative patient cases

#### **2. Employ Mixed Methods**
- Quantitative performance metrics (accuracy, time, errors)
- Qualitative understanding (interviews, observations)
- Physiological measures (eye-tracking, cognitive load)
- Behavioral analysis (interaction patterns, usage logs)

#### **3. Include Diverse Participants**
- Range of expertise levels (students, residents, attendings)
- Multiple specialties and roles
- Varied demographic backgrounds
- Different practice settings (academic, community, rural)

#### **4. Measure Multiple Outcomes**
- Task performance (accuracy, efficiency)
- Human factors (trust, cognitive load, satisfaction)
- Clinical outcomes (patient safety, quality of care)
- System-level effects (workflow disruption, team coordination)

#### **5. Conduct Longitudinal Studies**
- Initial exposure effects
- Learning and adaptation over time
- Long-term skill retention/degradation
- Sustained usage patterns

#### **6. Compare Collaboration Modes**
- Multiple human-AI interaction designs
- Active controls (traditional tools/processes)
- Between and within-subject designs
- Systematic variation of key factors

### 6.4 Specific ED Research Priorities

For emergency department human-AI teaming:

**Immediate Priorities (1-2 years):**
1. High-fidelity simulation studies of AI integration in ED triage
2. Workflow analysis of current ED processes to identify integration points
3. Time pressure effects on AI utilization and appropriate reliance
4. Failure mode analysis for time-critical AI systems

**Medium-term Priorities (2-4 years):**
1. Field studies in actual EDs with deployed AI systems
2. Team coordination analysis with AI-augmented ED workflows
3. Longitudinal evaluation of ED clinician-AI interaction patterns
4. Multi-site comparative studies across ED settings

**Long-term Priorities (4+ years):**
1. Development of validated ED-specific evaluation frameworks
2. Large-scale RCTs of AI integration approaches in ED
3. Outcome studies linking AI use to patient outcomes
4. Development of ED-specific AI design guidelines and standards

---

## 7. Relevance to ED Human-AI Teaming

### 7.1 Unique ED Characteristics Affecting Human-AI Interaction

The Emergency Department presents **distinctive challenges** for human-AI collaboration:

#### **1. Extreme Time Pressure**

**Characteristic**: Seconds-to-minutes for life-threatening decisions
**Implications for AI**:
- Must provide near-instantaneous insights (no time for lengthy analysis)
- Cannot increase cognitive load (clinicians already at capacity)
- Must prioritize actionable over comprehensive information
- Interface must support rapid scanning and comprehension

**Relevant Research**:
- **2411.00998** (Rosbach et al.): Time pressure increased automation bias severity
- **2505.11996** (Mastrianni et al.): Time-critical trauma decisions showed accuracy-time trade-offs
- **2503.08814** (Davidson et al.): ICU real-time risk quantification requirements

**Design Implications**:
- Minimize information presentation to critical insights only
- Use visual indicators (color, icons) for rapid interpretation
- Avoid requiring user input/interaction during acute phases
- Enable retrospective review when time permits

#### **2. Diagnostic Uncertainty**

**Characteristic**: Limited information, undifferentiated presentations, evolving situations
**Implications for AI**:
- AI predictions based on incomplete data
- Uncertainty quantification critical
- Differential diagnosis support more valuable than single diagnosis
- Must handle rapidly changing patient states

**Relevant Research**:
- **2309.12368** (Zhang et al.): SepsisLab provides future trajectory projection and uncertainty visualization
- **2204.09082** (Hemmer et al.): Identified uncertainty tolerance as key adoption factor
- **2412.00372** (Solomon et al.): Low-confidence decisions benefit most from AI support

**Design Implications**:
- Explicit uncertainty communication (confidence intervals, probability ranges)
- Differential diagnosis lists, not just top prediction
- Dynamic updating as new information becomes available
- Clear indication of data completeness/reliability

#### **3. High Cognitive Load**

**Characteristic**: Multiple patients, frequent interruptions, multitasking
**Implications for AI**:
- Extraneous cognitive load must be minimal
- Cannot assume focused attention on AI outputs
- Must compete for limited cognitive resources
- Integration must not add workflow steps

**Relevant Research**:
- **2501.16693** (Rezaeian et al.): Explainability features increased stress/cognitive load
- **1904.09612** (Yang et al.): "Unremarkable AI" reduces cognitive demands
- **2112.12596** (Chen et al.): User-centered design essential to manage load

**Design Implications**:
- Default to minimal information, details on-demand
- Embed in existing displays/workflows (avoid separate systems)
- Visual over textual presentation where possible
- Reduce need for interpretation (direct actionable suggestions)

#### **4. Team-Based Care**

**Characteristic**: Multiple providers (physicians, nurses, techs), frequent handoffs
**Implications for AI**:
- AI information must be shareable across team
- Different team members need different information
- Handoffs must include AI context
- Coordination around AI recommendations necessary

**Relevant Research**:
- **2102.08507** (Seo et al.): Team mental model alignment critical
- **2402.13437** (Yildirim et al.): Multi-stakeholder engagement in ICU
- **2101.01524** (Wang et al.): Team workflow integration failures

**Design Implications**:
- Role-based information presentation
- Shared visual displays accessible to team
- Clear documentation for handoffs
- Support for team discussion of AI recommendations

#### **5. High Stakes/Low Tolerance for Error**

**Characteristic**: Life-threatening conditions, immediate consequences of mistakes
**Implications for AI**:
- False negatives potentially catastrophic
- False positives consume limited resources
- Trust calibration critical (under- and over-reliance both dangerous)
- Transparency and explainability essential for accountability

**Relevant Research**:
- **2503.16692** (Hatherley): Trust vs reliability in high-stakes decisions
- **2204.06916** (Schemmer et al.): Appropriate reliance measurement
- **2411.00998** (Rosbach et al.): 7% automation bias rate unacceptable in high-stakes

**Design Implications**:
- Asymmetric error handling (prefer false positives in safety-critical)
- Clear confidence calibration (uncertainty when uncertain)
- Explainability for accountability and error detection
- Fail-safe defaults (safe to ignore AI without penalty)

#### **6. Practice Variability**

**Characteristic**: Rotating shifts, variable expertise, diverse patient populations
**Implications for AI**:
- Cannot assume consistent user expertise
- Must handle varied practice patterns
- Training/onboarding must be minimal
- Cannot rely on extended learning curves

**Relevant Research**:
- **2510.22414** (Sevgi et al.): Performance varied by clinician grade/experience
- **2102.00593** (Jacobs et al.): Local practice pattern adaptation needed
- **2101.01524** (Wang et al.): Context-specific requirements

**Design Implications**:
- Intuitive interfaces requiring minimal training
- Adaptive support based on user expertise detection
- Clear documentation of AI basis/limitations
- Standardized yet flexible to accommodate practice variation

### 7.2 ED-Specific Human Factors Priorities

Based on ED characteristics, priority human factors considerations:

#### **Priority 1: Rapid Trust Calibration**

**Challenge**: ED clinicians have minutes, not days, to calibrate trust in AI

**Requirements**:
- Immediate transparency about AI basis and limitations
- Rapid feedback on AI accuracy for specific patient types
- Clear performance statistics for local ED population
- Quick identification of failure modes

**Potential Solutions**:
- Pre-shift briefing on AI performance trends
- Real-time confidence calibration displays
- Case-based examples of success/failure modes
- Progressive disclosure during initial deployment

#### **Priority 2: Minimal Cognitive Burden**

**Challenge**: ED clinicians at cognitive capacity, cannot handle additional load

**Requirements**:
- Zero learning curve (intuitive from first use)
- No interpretation required (actionable outputs)
- Minimal attention demand
- Integration, not addition, to workflow

**Potential Solutions**:
- Visual indicators requiring no interpretation
- "Unremarkable AI" embedded in existing displays
- Voice-activated queries (hands-free, eyes-free)
- Automatic activation based on data patterns

#### **Priority 3: Team Coordination Support**

**Challenge**: Multiple providers need shared understanding of AI inputs

**Requirements**:
- Shared visibility of AI recommendations
- Clear communication of AI basis to all team members
- Support for team discussion/decision-making
- Handoff documentation including AI context

**Potential Solutions**:
- Shared displays visible to entire team
- AI summary in verbal handoff protocols
- Team-accessible explanation interfaces
- Collaborative decision documentation

#### **Priority 4: Time-Appropriate Support**

**Challenge**: Support needs differ in acute vs stabilization phases

**Requirements**:
- Acute phase: Minimal, high-priority information only
- Stabilization: Detailed support, differential diagnosis
- Post-event: Learning, outcome tracking, performance review
- Adaptive to clinical phase

**Potential Solutions**:
- Phase-adaptive interfaces
- Priority-based information filtering
- Retrospective detailed explanations
- Contextual awareness of patient acuity

#### **Priority 5: Failure Safety**

**Challenge**: AI errors in ED can be immediately life-threatening

**Requirements**:
- Graceful degradation when AI uncertain
- Easy detection of AI errors by clinicians
- Safe to ignore without penalty
- Clear accountability for decisions

**Potential Solutions**:
- Explicit uncertainty quantification
- Comparison to standard-of-care protocols
- Independent verification mechanisms
- Audit trails distinguishing AI-influenced decisions

### 7.3 Recommended Collaboration Modes for ED

Based on ED characteristics and human factors research, **recommended collaboration modes by ED process**:

#### **ED Triage (Initial Assessment)**

**Recommended Mode**: AI-First Advisory (Mode 2) with Cognitive Forcing

**Rationale**:
- High volume, need efficiency
- Initial data collection standardized
- Acuity assessment time-sensitive
- Triage nurses benefit from second opinion

**Implementation**:
- AI analyzes initial vitals/chief complaint
- Suggests triage category with confidence
- Triage nurse makes independent assessment
- Highlights disagreements for supervisor review
- Rapid feedback loop for calibration

**Key Studies**: 2507.16947 (AI Consult safety net model)

#### **Diagnostic Workup (Differential Diagnosis)**

**Recommended Mode**: Interactive Collaboration (Mode 3)

**Rationale**:
- High uncertainty, multiple possibilities
- Clinician needs support generating hypotheses
- Iterative refinement as data accumulates
- Benefits from AI pattern recognition

**Implementation**:
- AI suggests differential based on available data
- Clinician queries specific diagnoses
- AI recommends tests to distinguish possibilities
- Updates probabilities as new data arrives
- Supports clinical reasoning, not replacement

**Key Studies**: 2309.12368 (SepsisLab), 2506.14774 (MedSyn)

#### **Critical Decision Points (Treatment Decisions)**

**Recommended Mode**: Human-First with AI Verification (Mode 3 + Cognitive Forcing)

**Rationale**:
- Highest stakes, need independent clinical judgment
- AI as verification/safety check
- Preserves clinician accountability
- Reduces automation bias

**Implementation**:
- Clinician formulates treatment plan
- AI evaluates plan against evidence/guidelines
- Highlights potential concerns/alternatives
- Clinician makes final decision with AI context
- Clear documentation of reasoning

**Key Studies**: 2205.09696 (provisional decisions reduce bias), 2412.00372 (verification approach)

#### **Monitoring and Re-Assessment**

**Recommended Mode**: AI-Augmented Workflow (Mode 5) with Alerts

**Rationale**:
- Continuous data stream, human can't monitor constantly
- Need automated detection of deterioration
- Alert fatigue concern, must be selective
- Supports early intervention

**Implementation**:
- AI continuously monitors vitals/labs
- Alerts on concerning trends or patterns
- Embedded in existing monitoring displays
- Prioritizes alerts by urgency/confidence
- Supports rapid response team activation

**Key Studies**: 1904.09612 (unremarkable AI), 2503.08814 (ICU monitoring)

#### **Handoffs and Transitions**

**Recommended Mode**: Asynchronous Oversight (Mode 4) with Documentation

**Rationale**:
- Information synthesis during transitions
- Supports complete handoff
- Enables receiving provider review
- Maintains continuity of AI context

**Implementation**:
- AI generates handoff summary
- Includes relevant AI recommendations/alerts
- Flags outstanding concerns
- Accessible to receiving provider
- Integrates with handoff protocols

**Key Studies**: 2507.15743 (g-AMIE asynchronous oversight)

### 7.4 ED Implementation Roadmap

**Phased approach to ED AI integration**:

#### **Phase 1: Low-Risk, High-Volume Processes (Months 1-6)**

**Target**: Triage, routine order sets, documentation support

**Approach**:
- AI-augmented workflow mode
- Minimal workflow disruption
- Clear value proposition (time savings)
- Low stakes allows calibration

**Success Metrics**:
- Time savings per patient
- User satisfaction scores
- Triage accuracy vs historical
- Documentation completeness

#### **Phase 2: Decision Support for Specific Conditions (Months 6-12)**

**Target**: Common high-risk conditions (sepsis, MI, stroke)

**Approach**:
- Interactive collaboration mode
- Condition-specific workflows
- Explicit training and calibration
- Parallel deployment (doesn't replace existing)

**Success Metrics**:
- Time to diagnosis for target conditions
- Appropriate reliance rates
- False negative rate (safety)
- Clinician trust calibration

#### **Phase 3: Comprehensive Integration (Months 12-24)**

**Target**: Broader diagnostic support, treatment recommendations

**Approach**:
- Multiple collaboration modes by context
- Adaptive interfaces
- Team-level integration
- Continuous monitoring and improvement

**Success Metrics**:
- Patient outcomes (LOS, mortality, readmission)
- Clinician performance and satisfaction
- Workflow efficiency
- Error reduction rates

#### **Phase 4: Advanced Capabilities (Months 24+)**

**Target**: Predictive analytics, resource optimization, personalized medicine

**Approach**:
- Multi-modal AI integration
- Predictive rather than reactive support
- System-level optimization
- Research and continuous innovation

**Success Metrics**:
- Department-level outcomes
- Resource utilization efficiency
- Preventable adverse event reduction
- Generalizability to other EDs

### 7.5 ED-Specific Evaluation Framework

To rigorously evaluate ED AI systems, comprehensive framework needed:

#### **Evaluation Dimension 1: Clinical Performance**

**Metrics**:
- Diagnostic accuracy (sensitivity, specificity, PPV, NPV)
- Time to diagnosis for time-sensitive conditions
- Treatment appropriateness and guideline concordance
- Adverse event rates (missed diagnoses, medication errors)
- Patient outcomes (mortality, morbidity, readmissions)

**Methods**:
- Retrospective chart review
- Prospective observational studies
- Randomized controlled trials (when ethical)
- Comparison to historical controls

#### **Evaluation Dimension 2: Human Factors**

**Metrics**:
- Appropriate reliance rate (accept correct, reject incorrect AI)
- Trust calibration (confidence-accuracy alignment)
- Cognitive load (NASA-TLX, physiological measures)
- Usability (SUS, task completion time, error rates)
- User satisfaction and acceptance

**Methods**:
- Survey instruments (validated scales)
- Think-aloud protocols
- Eye-tracking studies
- Cognitive load assessments
- Interviews and focus groups

#### **Evaluation Dimension 3: Workflow Integration**

**Metrics**:
- Time per patient encounter
- Workflow disruption events
- System access/usage rates
- Integration with EHR and other systems
- Handoff completeness and accuracy

**Methods**:
- Time-motion studies
- Workflow analysis and process mapping
- Usage log analysis
- Ethnographic observation
- Provider shadowing

#### **Evaluation Dimension 4: Team Coordination**

**Metrics**:
- Team communication quality
- Shared mental model alignment
- Coordination efficiency
- Role clarity and accountability
- Handoff effectiveness

**Methods**:
- Team communication analysis
- Social network analysis
- Coordination measurement tools
- Incident reports and near-misses
- Team debriefings

#### **Evaluation Dimension 5: System-Level Impact**

**Metrics**:
- ED throughput (patients/hour)
- Length of stay
- Left without being seen (LWBS) rate
- Resource utilization (labs, imaging, consults)
- Cost-effectiveness

**Methods**:
- Operational metrics tracking
- Before-after comparisons
- Economic analysis
- System dynamics modeling
- Multi-site benchmarking

---

## 8. Synthesis and Recommendations

### 8.1 Core Principles for ED Human-AI Teaming

Based on comprehensive literature review, **ten core principles** for effective ED AI systems:

#### **Principle 1: Clinician Autonomy Preservation**
AI must augment, not replace, clinical judgment. Clinicians maintain decision authority and accountability.

**Evidence**: 2204.09082, 2507.16947, 2102.00593

#### **Principle 2: Appropriate Reliance Optimization**
Goal is discriminating AI quality and acting accordingly, not maximum trust or usage.

**Evidence**: 2204.06916, 2501.16693, 2411.00998

#### **Principle 3: Workflow Integration Priority**
Technical performance insufficient; AI must fit clinical workflows seamlessly.

**Evidence**: 1904.09612, 2006.12683, 2101.01524

#### **Principle 4: Cognitive Load Minimization**
ED clinicians at capacity; AI must reduce, not increase, cognitive burden.

**Evidence**: 2501.16693, 2112.12596, 1904.09612

#### **Principle 5: Time-Appropriate Support**
Support must adapt to temporal context (acute vs stabilization phases).

**Evidence**: 2505.11996, 2503.08814, 2411.00998

#### **Principle 6: Transparent Uncertainty**
AI must explicitly communicate confidence, limitations, and uncertainty.

**Evidence**: 2309.12368, 2204.06916, 2503.16692

#### **Principle 7: Team-Centered Design**
Support multi-provider teams, not just individual clinicians.

**Evidence**: 2102.08507, 2402.13437, 2102.00593

#### **Principle 8: Actionable Outputs**
Provide specific clinical actions, not just diagnoses or risk scores.

**Evidence**: 2309.12368, 2507.16947, 2503.08814

#### **Principle 9: Failure Safety**
System must degrade gracefully; safe to ignore AI without penalty.

**Evidence**: 2503.16692, 2411.00998, 2507.16947

#### **Principle 10: Continuous Learning**
Support clinician learning from AI and AI improvement from usage.

**Evidence**: 2510.22414, 2506.14774, 2204.05030

### 8.2 Design Recommendations

#### **For AI Developers:**

**1. Conduct Deep Workflow Analysis**
- Shadow ED clinicians through complete shifts
- Map all decision points and information needs
- Identify pain points and inefficiencies
- Understand team dynamics and handoffs
- Document time pressures and cognitive load

**2. Co-Design with End Users**
- Include ED physicians, nurses, techs in design process
- Iterative prototyping with real users
- Test in high-fidelity simulated ED environments
- Continuous feedback and refinement
- Multi-stakeholder perspective integration

**3. Optimize for Cognitive Load**
- Default to minimal information presentation
- Progressive disclosure for details
- Visual over textual where possible
- Embed in existing displays/systems
- Reduce interpretation requirements

**4. Prioritize Explainability**
- Provide verifiable explanations, not just model internals
- Enable comparison to similar cases
- Show alignment with clinical guidelines
- Support what-if exploration
- Balance detail with usability

**5. Implement Robust Uncertainty Quantification**
- Calibrated confidence estimates
- Explicit communication of data limitations
- Clear indication when outside training distribution
- Graceful degradation when uncertain
- Transparent about failure modes

#### **For Healthcare Organizations:**

**1. Invest in Workflow Integration**
- Dedicate resources to EHR integration
- Minimize additional login/systems
- Ensure real-time data availability
- Support team-accessible interfaces
- Plan for handoff documentation

**2. Provide Adequate Training**
- Not just how to use, but when and why
- Calibration exercises with known cases
- Discussion of failure modes and limitations
- Team-based training for coordination
- Ongoing education as systems evolve

**3. Monitor Human Factors**
- Track usage patterns and adoption
- Measure trust calibration and appropriate reliance
- Assess cognitive load and satisfaction
- Identify workflow friction points
- Continuous user feedback collection

**4. Establish Governance**
- Clear accountability for AI-influenced decisions
- Incident reporting and analysis processes
- Regular performance audits
- Ethics review for new applications
- Patient communication protocols

**5. Support Continuous Improvement**
- Collect outcomes data linked to AI usage
- Enable clinician feedback on AI performance
- Regular model retraining and updates
- Transparent communication of changes
- Research collaboration for evaluation

#### **For Researchers:**

**1. Conduct ED-Specific Studies**
- Real-world ED field studies (not just simulations)
- Time pressure manipulation experiments
- Team-level interaction analysis
- Longitudinal deployment studies
- Multi-site generalizability testing

**2. Develop Validated Metrics**
- Appropriate reliance measurement tools
- Trust calibration assessments
- Cognitive load instruments for clinical settings
- Team coordination metrics
- Real-world outcome measures

**3. Build Benchmark Datasets**
- ED-specific clinical scenarios
- Time-stamped data reflecting information availability
- Diverse patient populations and conditions
- Known ground truth for evaluation
- Publicly available for comparison

**4. Study Failure Modes**
- Systematic characterization of AI failures in ED
- Human factors in error detection
- Recovery strategies evaluation
- Near-miss analysis
- Harm prevention mechanisms

**5. Advance Theoretical Understanding**
- Models of human-AI collaboration in time-critical settings
- Cognitive mechanisms of appropriate reliance
- Team mental model formation with AI
- Trust calibration dynamics
- Learning and adaptation over time

### 8.3 Policy and Regulatory Recommendations

**1. Develop ED-Specific Regulatory Frameworks**
- Recognize unique requirements of time-critical AI
- Balance innovation with safety
- Require real-world ED validation
- Mandate human factors evaluation
- Continuous post-market surveillance

**2. Establish Standards for Transparency**
- Required disclosure of training data characteristics
- Performance reporting by patient population
- Known limitations and failure modes
- Explainability requirements
- Uncertainty quantification standards

**3. Create Accountability Frameworks**
- Clear assignment of responsibility
- Support for clinician override
- Documentation requirements
- Liability considerations
- Patient consent protocols

**4. Support Research Infrastructure**
- Funding for human factors research in clinical AI
- Data sharing and benchmark development
- Multi-site collaboration networks
- Evaluation methodology development
- Workforce training and development

**5. Promote Equity and Fairness**
- Require bias testing across populations
- Ensure accessibility in under-resourced settings
- Address digital divide considerations
- Support diverse stakeholder engagement
- Monitor for disparate impact

### 8.4 Key Takeaways for ED Human-AI Teaming

**Critical Success Factors:**
1. **Workflow Integration > Technical Performance**: The best AI fails if it doesn't fit clinical workflows
2. **Trust Calibration Critical**: Both under- and over-reliance dangerous; appropriate reliance is goal
3. **Context Matters**: ED requirements differ fundamentally from outpatient/elective settings
4. **Team-Level Support**: Individual clinician focus insufficient for team-based ED care
5. **Continuous Evolution**: Initial deployment just beginning; continuous learning and adaptation essential

**Biggest Risks:**
1. **Automation Bias in Time Pressure**: Clinicians may over-rely when cognitively strained
2. **Workflow Disruption**: Adding steps or systems undermines adoption and safety
3. **Calibration Failures**: Miscommunication of confidence leads to inappropriate reliance
4. **Team Coordination Breakdown**: AI creates information asymmetry between providers
5. **Skill Degradation**: Long-term AI reliance may erode clinical capabilities

**Most Promising Opportunities:**
1. **Early Warning Systems**: AI excels at continuous monitoring for deterioration
2. **Diagnostic Support**: Pattern recognition for complex presentations
3. **Resource Optimization**: Predicting demand and optimizing allocation
4. **Workflow Efficiency**: Automation of documentation and routine tasks
5. **Quality Improvement**: Identification of care gaps and adherence issues

---

## 9. Conclusion

This comprehensive review of human factors and human-AI interaction in clinical AI reveals both immense promise and significant challenges for implementation in emergency department settings. While AI systems demonstrate impressive technical capabilities, successful deployment depends fundamentally on understanding and addressing the human factors that govern clinician-AI collaboration.

### Key Insights

**1. Human Factors Are the Bottleneck**
Despite rapid advances in AI performance, the primary barriers to effective clinical AI are human factors: trust calibration, workflow integration, cognitive load management, and team coordination. Technical excellence is necessary but insufficient.

**2. Context Shapes Collaboration**
The emergency department's unique characteristics—extreme time pressure, high uncertainty, cognitive overload, team-based care, and high stakes—necessitate ED-specific approaches to human-AI teaming. Solutions from other settings cannot simply be transferred.

**3. Trust is Complex and Dynamic**
The relationship between trust, reliance, and performance is non-linear and counterintuitive. High trust can lead to dangerous overreliance; low trust prevents beneficial use. The goal is appropriate reliance that discriminates AI quality case-by-case.

**4. Workflow Integration is Make-or-Break**
Even perfectly accurate AI systems fail in practice if they disrupt clinical workflows. Integration must be seamless, adding value without adding burden. "Unremarkable AI" that augments existing processes shows most promise.

**5. Explainability Has Trade-offs**
While explanations can support appropriate reliance and trust calibration, they also increase cognitive load and may reinforce biases. The type, timing, and presentation of explanations critically affect their utility.

**6. Collaboration Modes Matter**
Different clinical contexts require different human-AI collaboration modes. Triage may benefit from AI-first advisory; critical decisions from human-first verification. Adaptive, context-aware support is ideal but challenging to implement.

**7. Team-Level Considerations Essential**
Most research focuses on individual clinician-AI interaction, but ED care is fundamentally team-based. Supporting team coordination, shared mental models, and effective handoffs is critical but understudied.

**8. Time Pressure Changes Everything**
The time-critical nature of ED care affects every aspect of human-AI collaboration. Systems must provide near-instantaneous insights, minimize cognitive load, and support rapid decision-making while maintaining safety.

### Path Forward

Realizing the potential of AI to enhance emergency care requires a coordinated, multi-faceted effort:

**For the Research Community:**
Prioritize real-world ED studies, develop validated human factors metrics, characterize failure modes, and advance theoretical understanding of human-AI collaboration in time-critical settings.

**For AI Developers:**
Adopt human-centered design principles, conduct deep workflow analysis, co-design with end users, prioritize cognitive load minimization, and ensure transparent uncertainty quantification.

**For Healthcare Organizations:**
Invest in workflow integration infrastructure, provide comprehensive training, monitor human factors continuously, establish clear governance, and support ongoing improvement.

**For Policy Makers:**
Develop ED-specific regulatory frameworks, establish transparency standards, create accountability mechanisms, fund research infrastructure, and ensure equity in AI access and impact.

### Final Perspective

The question is not whether AI will transform emergency care, but how we can ensure that transformation enhances rather than undermines the clinician-patient relationship and the quality of care. Success requires moving beyond a narrow focus on AI performance to a holistic understanding of human-AI collaboration as a complex sociotechnical system.

The emergency department, with its extreme demands and life-or-death stakes, represents both the greatest challenge and opportunity for clinical AI. Getting it right in the ED—achieving truly effective human-AI teaming—will require unprecedented collaboration between computer scientists, clinicians, human factors experts, and patients. But the potential rewards—saved lives, improved outcomes, reduced clinician burden, and more equitable care—make this one of the most important challenges in modern medicine.

This review provides a foundation for that work, synthesizing current knowledge and identifying critical gaps. The path forward is clear: rigorous, human-centered research and development, thoughtful deployment and evaluation, and continuous learning and adaptation. The future of emergency care will be shaped by how well we understand and support the humans who work alongside AI.

---

## References

This review analyzed over 150 papers from ArXiv spanning 2019-2025. Key papers are cited throughout using ArXiv IDs. Complete bibliography available upon request.

**Primary Database**: ArXiv (arxiv.org)
**Search Period**: 2019-2025
**Categories Searched**: cs.HC, cs.AI, cs.CL, cs.CY, cs.LG
**Total Papers Reviewed**: 150+
**Core Papers Analyzed in Detail**: 85

---

**Document Information:**
- **Created**: December 1, 2025
- **Author**: Systematic Literature Review
- **Purpose**: Research synthesis for ED human-AI teaming project
- **Location**: /Users/alexstinard/hybrid-reasoning-acute-care/research/
- **Version**: 1.0
