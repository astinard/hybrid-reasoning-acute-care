# LLMs in Clinical Settings: A Research Synthesis

## Executive Summary

This synthesis examines recent research on large language models (LLMs) in clinical decision support, with particular focus on clinical NLP tasks, hallucination concerns, and integration with structured data. The analysis covers 45 papers spanning clinical decision support systems, medical diagnosis, medication safety, and emergency care applications.

**Key Findings:**
- LLMs show significant promise in clinical decision support when properly grounded with clinical practice guidelines and structured knowledge
- Hallucination remains a critical concern, but can be mitigated through RAG (Retrieval-Augmented Generation), clinical guideline grounding, and human-in-the-loop approaches
- Integration with EHR/structured data through standardized medical terminologies (SNOMED CT, LOINC, RxNorm) substantially improves performance
- Multi-agent LLM systems outperform single-agent approaches for complex clinical tasks
- Co-pilot modes (LLM + human clinician) consistently outperform fully autonomous LLM deployment

---

## 1. Clinical Decision Support Systems (CDSS)

### 1.1 Real-World Deployment and Impact

**AI Consult at Penda Health (Kenya)** - Korom et al., 2025
- **Study Design:** Quality improvement study across 39,849 patient visits at 15 primary care clinics
- **System:** LLM-based safety net integrated into EMR workflow with traffic-light interface (green/yellow/red)
- **Results:**
  - 16% reduction in diagnostic errors (NNT: 18.1)
  - 13% reduction in treatment errors (NNT: 13.9)
  - 32% reduction in history-taking errors (NNT: 11.3)
  - Would avert 22,000 diagnostic errors and 29,000 treatment errors annually at Penda alone
  - All clinicians reported AI Consult improved quality of care (75% said "substantial")
- **Critical Success Factors:**
  1. Clinical workflow alignment (asynchronous, event-driven architecture)
  2. Active deployment strategies (peer champions, measurement, incentives)
  3. Tiered alert system to minimize alert fatigue
- **Limitations:** No significant impact on patient-reported outcomes during study period

**Key Insight:** The study demonstrates that the "model-implementation gap" is often more critical than model performance. Even with capable models, success requires clinically-aligned implementation and active change management.

### 1.2 Grounding LLMs with Clinical Practice Guidelines

**COVID-19 Treatment Support** - Oniani et al., 2024
- **Methods Compared:**
  1. Binary Decision Tree (BDT) - recursive algorithm guided by CPGs
  2. Chain-of-Thought-Few-Shot Prompting (CoT-FSP) - algorithmic if-else description
  3. Program-Aided Graph Construction (PAGC) - networkx graph representation
  4. Zero-Shot Prompting (ZSP) - baseline
- **Models Tested:** GPT-4, GPT-3.5 Turbo, LLaMA, PaLM 2
- **Results:**
  - All CPG-enhanced methods significantly outperformed ZSP baseline
  - BDT achieved highest F-score in automatic evaluation
  - GPT-4 CoT-FSP, GPT-4 PAGC, and PaLM 2 BDT achieved perfect scores (2.0) in human evaluation
  - Zero-shot prompting inadequate for complex clinical reasoning
- **Implications:** Explicit incorporation of CPGs dramatically improves LLM performance in clinical decision support

### 1.3 Multi-Agent Systems for Emergency Department Triage

**Korean Triage and Acuity Scale (KTAS) System** - Han & Choi, 2024
- **Architecture:** Multi-agent system with 4 specialized agents:
  1. Triage Nurse (KTAS classification)
  2. Emergency Physician (diagnosis and treatment)
  3. Pharmacist (medication safety via RxNorm)
  4. ED Doctor in Charge (final decisions and coordination)
- **Base Model:** Llama-3-70b orchestrated by CrewAI
- **Performance vs Single-Agent:**
  - Multi-agent: Consistent, definitive KTAS classifications
  - Single-agent: Frequent range classifications (e.g., "1 or 2") or N/A outputs
  - Multi-agent achieved perfect scores (1.0) on immediate action, medication, diagnostic tests, consultation, and monitoring
- **Clinical Implications:**
  - Tendency to over-triage (erring on side of caution)
  - Could reduce risk of overlooking critically ill patients
  - Mimics collaborative nature of real ED teams

**Key Finding:** Multi-agent approaches that mirror actual clinical team structures consistently outperform monolithic LLM systems for complex clinical tasks.

---

## 2. Medication Safety and Drug-Related Problems

### 2.1 RAG-Enhanced Medication Error Detection

**Medication Safety CDSS** - Ong et al., 2024
- **Study:** 61 prescribing error scenarios across 23 complex vignettes, 12 clinical specialties
- **System:** RAG-LLM with institutional medication protocols and drug monographs
- **Deployment Modes Compared:**
  - Autonomous mode (RAG-LLM alone): 31.1% accuracy
  - Co-pilot mode (Junior pharmacist + RAG-LLM): 54.1% accuracy
  - Human alone baseline
- **DRP Categories Evaluated:**
  - Adverse drug reactions
  - Drug-drug interactions
  - Duplication of therapy
  - Inappropriate choice of therapy
  - Inappropriate dosage regimen
  - Omission of therapy
- **Performance by Severity:**
  - Strongest performance on serious/moderate harm DRPs
  - Co-pilot mode showed improvements across most categories
  - Declined performance in "inappropriate dosage regimen" and "no indication" categories
- **Challenges Identified:**
  1. Dosing recommendations in table format (difficult for LLMs to process)
  2. Incomplete off-label prescribing documentation
  3. Institution-specific variations in practice

**Critical Insight:** Co-pilot deployment doubled the accuracy compared to autonomous LLM use, emphasizing the importance of human oversight for complex reasoning tasks.

### 2.2 Limitations and Practical Considerations

**Common Challenges Across Studies:**
1. **Formulary Alignment:** LLMs sometimes recommend medications not in hospital formularies or prohibitively expensive
2. **Documentation Format:** Structured tables and images within documents pose processing challenges
3. **Context Specificity:** Generic drug monographs may not capture institution-specific protocols
4. **Update Frequency:** Knowledge bases require regular updates to reflect evolving guidelines

**Recommendations:**
- Integrate hospital-specific medication databases
- Include cost information in knowledge bases
- Use standardized terminologies (RxNorm, SNOMED CT)
- Implement regular knowledge base updates
- Maintain human pharmacist oversight for final recommendations

---

## 3. Hallucination Mitigation Strategies

### 3.1 Grounding Techniques

**Clinical Practice Guideline Grounding** (Multiple Studies)
- **Approach:** Embed CPGs directly in prompts or through RAG
- **Evidence:**
  - Penda Health: AI Consult grounded in Kenyan clinical guidelines and local epidemiology
  - COVID-19 study: All CPG-enhanced methods significantly reduced errors vs baseline
  - CliCARE study: Guideline knowledge graphs for cancer EHR decision support
- **Effectiveness:** Consistently reduces hallucinations and improves clinical appropriateness

**Retrieval-Augmented Generation (RAG)**
- **Version 1 (Simple):**
  - Vector storage: Pinecone
  - Embedding: OpenAI text-embedding-ada-002
  - Chunk size: 1000, k=5
- **Version 2 (Advanced):**
  - Framework: Llamaindex with auto-merging retrieval
  - Embedding: HuggingFace bge-small-en-v1.5
  - Hierarchical chunking: 2048, 512, 123
  - Manual drug name indexing, k=20
- **Results:** RAG improved accuracy by 22% over base GPT-4 for medication safety

**Temporal Knowledge Graphs** (CliCARE - Li et al., 2025)
- **Method:** Transform longitudinal EHRs into patient-specific Temporal Knowledge Graphs (TKGs)
- **Grounding:** Align patient trajectories with normative guideline knowledge graph
- **Benefits:**
  - Captures long-range dependencies in patient history
  - Provides evidence-grounded decision support
  - Significantly outperforms standard RAG methods
- **Validation:** Tested on Chinese cancer dataset and MIMIC-IV

### 3.2 Human-in-the-Loop Approaches

**Safety Net Architecture** (Penda Health)
- LLM runs asynchronously in background
- Only surfaces alerts when material risk identified
- Clinician maintains final decision authority
- No cases of AI-caused harm in patient safety reports
- AI prevented harm in 50% of serious adverse events when guidance was followed

**Expert Oversight Models**
- Multi-rater physician review (Penda: 108 physicians)
- Board-certified specialist validation (medication safety study)
- Emergency medicine specialist assessment (KTAS study)

**Co-Pilot Mode Best Practices:**
1. Clear delineation of AI vs human responsibilities
2. Transparent presentation of AI reasoning
3. Easy override mechanisms
4. Continuous monitoring and feedback loops
5. Regular performance audits

### 3.3 Evaluation and Validation

**LLM-as-Evaluator Paradox** (Penda Health findings)
- GPT-4.1 and o3 as evaluators showed higher inter-rater agreement with physicians than physician-physician agreement
- LLM evaluators found larger effect sizes than human raters
- Potential for Goodhart's Law: systems optimized for LLM evaluation may not optimize for clinical outcomes
- Recommendation: Use LLMs for scale, but validate with physician agreement on subset

**Error Analysis Categories** (Common across studies)
1. **Comprehension errors** (49.3% of LLM errors in calculator recommendation study)
2. **Knowledge gaps** (7.1% of errors)
3. **Reasoning failures** (particularly for multi-step clinical reasoning)
4. **Context misunderstanding** (especially for clinical nuance)

---

## 4. Integration with Structured Data

### 4.1 EHR Integration Approaches

**Temporal Knowledge Graph Integration** (CliCARE)
- **Input:** Unstructured, multilingual longitudinal EHRs
- **Processing:** Transform to patient-specific TKGs
- **Integration:** Align with guideline knowledge graphs
- **Output:** Clinical summary + actionable recommendations
- **Performance:** Significantly outperforms long-context LLMs and KG-enhanced RAG baselines

**Direct EHR Integration Challenges:**
1. Multilingual nature of records
2. Extensive length and temporal complexity
3. Missing structured data elements
4. Inconsistent documentation formats
5. Privacy and security requirements

**Successful Integration Strategies:**
- Standardized medical terminologies (SNOMED CT, LOINC, RxNorm)
- Event-driven architecture (trigger on specific EMR field updates)
- Asynchronous processing to avoid workflow disruption
- Local deployment to maintain data privacy (e.g., Llama models)

### 4.2 Clinical NLP Tasks

**Entity Extraction and Recognition**
- **Common Tasks:**
  - Medication names and dosages
  - Disease diagnoses and ICD codes
  - Vital signs and laboratory values
  - Temporal information (onset, duration, frequency)
  - Social determinants of health (SDOH)

**NLP Models for Clinical Text:**
- **General LLMs:** GPT-4, Gemini Pro, Claude
- **Medical-Specific:** Med-PaLM 2, GatorTron, BioClinical BERT
- **Domain Adaptation:** Fine-tuning on clinical notes significantly improves performance

**Clinical NLP Performance:**
- BERT-based models: Strong for entity recognition and classification
- Transformer models: Excel at understanding clinical context
- LLMs: Superior for generation and complex reasoning tasks
- Trade-offs: Accuracy vs speed vs computational resources vs cost

### 4.3 Standardized Terminologies

**SNOMED CT Integration**
- Systematized nomenclature for clinical concepts
- Enables precise semantic representation
- Facilitates cross-institutional interoperability
- Used in delirium symptom extraction (97.6% F1 for best systems)

**RxNorm for Medication Management**
- Standard nomenclature for clinical drugs
- Enables drug interaction checking
- Supports dosing recommendations
- Critical for medication safety CDSS

**LOINC for Laboratory Results**
- Standardized identifiers for observations
- Enables consistent lab result interpretation
- Supports clinical decision rules based on lab values

**Integration Benefits:**
1. Reduced ambiguity in clinical concepts
2. Improved interoperability across systems
3. More reliable automated reasoning
4. Easier knowledge base maintenance
5. Better support for clinical quality measures

---

## 5. Clinical NLP Tasks and Performance

### 5.1 Clinical Reasoning and Diagnosis

**Medical Calculator Recommendation** (Wan et al., 2024)
- **Task:** Select appropriate medical calculators for clinical scenarios
- **Dataset:** 1,009 multiple-choice questions across 35 clinical calculators
- **Best LLM Performance:** OpenAI o1 - 66.0% accuracy (CI: 56.7-75.3%)
- **Human Performance:** 79.5% accuracy (CI: 73.5-85.0%)
- **Error Analysis:**
  - Comprehension errors: 49.3%
  - Calculator knowledge gaps: 7.1%
  - Other errors: 43.6%
- **Conclusion:** LLMs not yet superior to humans in specialized clinical tool selection

**Differential Diagnosis**
- **CliCARE:** Outperformed baselines for cancer diagnosis from longitudinal EHRs
- **Multi-Agent ED System:** High accuracy in primary diagnosis (perfect 5/5 ratings for 41/43 cases)
- **Challenges:** Rare diseases, atypical presentations, comorbid conditions

### 5.2 Clinical Text Summarization and Documentation

**Discharge Summary Generation**
- LLMs show promise but require careful validation
- Risk of omitting critical information
- MED-OMIT metric: Measures clinically relevant omissions
- Recommendation: Use for draft generation with mandatory physician review

**Triage Documentation**
- Multi-agent system: Generated comprehensive triage reports
- Included KTAS level, critical findings, justification
- High concordance with expert assessments (42/43 perfect scores for critical findings)

### 5.3 Social Determinants of Health (SDOH) Extraction

**BERT-based Extraction** (Yu et al., 2021 - Lung Cancer Patients)
- **Task:** Extract smoking, education, employment from clinical narratives
- **Performance:** F1-score 0.8791 (strict), 0.8999 (lenient)
- **Findings:**
  - Much more SDOH information in narratives than structured EHR fields
  - Critical for comprehensive patient understanding
  - Requires both structured and unstructured data analysis

**Importance for Clinical Care:**
- Essential for holistic patient assessment
- Influences treatment planning and resource allocation
- Often missing from structured EHR data
- NLP enables capture of this critical information

---

## 6. Model Performance Comparisons

### 6.1 LLM Model Rankings (Across Studies)

**For Clinical Decision Support:**
1. **GPT-4 / GPT-4o:** Consistently highest performance
   - Best in medication safety (when using RAG)
   - Strong clinical reasoning
   - High cost may limit deployment
2. **Med-PaLM 2:** Strong for medical-specific tasks
   - Lower precision than GPT-4 in medication safety
   - Better suited for question-answering than complex reasoning
3. **Gemini Pro 1.0:** Middle-tier performance
   - Lower accuracy than GPT-4 in most evaluations
   - More cost-effective alternative
4. **Llama-3-70b:** Open-source option
   - Excellent for multi-agent systems when properly prompted
   - Local deployment enables HIPAA compliance
   - Requires significant computational resources

**For Specialized Clinical NLP:**
1. **GatorTron:** Domain-specific for clinical text
2. **BioClinical BERT:** Strong entity recognition
3. **PubMedBERT:** Good for biomedical literature

### 6.2 RAG vs Fine-Tuning vs Prompting

**RAG (Retrieval-Augmented Generation):**
- **Advantages:**
  - No retraining required for knowledge updates
  - Can incorporate hospital-specific protocols
  - Reduces hallucinations
  - Maintains privacy with local vector stores
- **Disadvantages:**
  - Requires well-structured knowledge base
  - Retrieval quality critical to performance
  - Additional latency from retrieval step
- **Best Use Cases:** Medication safety, protocol adherence, guideline implementation

**Fine-Tuning:**
- **Advantages:**
  - Better task-specific performance
  - No external knowledge base needed at inference
  - Lower latency
- **Disadvantages:**
  - Requires labeled training data
  - Expensive to update
  - Risk of overfitting to training distribution
- **Best Use Cases:** Entity extraction, classification tasks, standardized workflows

**Prompting (Zero-Shot, Few-Shot, CoT):**
- **Advantages:**
  - No training required
  - Highly flexible
  - Easy to update instructions
- **Disadvantages:**
  - Inconsistent performance
  - Higher token costs
  - May require extensive prompt engineering
- **Best Use Cases:** Exploratory tasks, rapid prototyping, simple queries

**Hybrid Approaches (Most Common in Clinical Settings):**
- RAG + Few-Shot Prompting
- Fine-Tuned Base Model + RAG
- Multi-Agent with Specialized Fine-Tuned Agents

---

## 7. Deployment Considerations

### 7.1 Modes of Deployment

**1. Fully Autonomous**
- **Use Cases:** Low-stakes tasks, well-defined problems
- **Examples:** Appointment scheduling, simple triage
- **Performance:** Generally 30-40% accuracy for complex tasks
- **Risk Level:** High - requires extensive validation

**2. Co-Pilot / Assistive**
- **Use Cases:** Complex clinical reasoning, decision support
- **Examples:** Medication review, diagnosis support
- **Performance:** 50-85% accuracy depending on task complexity
- **Risk Level:** Moderate - human oversight required
- **Most Successful Model:** Demonstrated 2x improvement over autonomous

**3. Human-in-the-Loop Safety Net**
- **Use Cases:** Background monitoring, error detection
- **Examples:** Penda Health AI Consult
- **Performance:** 13-32% error reduction
- **Risk Level:** Low - clinician maintains full control
- **Alert Strategy:** Tiered (green/yellow/red) to minimize fatigue

### 7.2 Implementation Success Factors

**Critical Elements (from Penda Health study):**
1. **Clinically-Aligned Implementation:**
   - Asynchronous, event-driven architecture
   - Integration at natural workflow decision points
   - Minimal cognitive load (tiered alerts)
   - Preserve clinician autonomy
2. **Active Deployment:**
   - Peer champions at each site
   - Performance metrics and feedback
   - Recognition and incentives
   - Continuous iteration based on user feedback
3. **Model Performance:**
   - Adequate base model capability
   - Appropriate grounding/RAG
   - Local adaptation (epidemiology, formularies, costs)

**Common Implementation Failures:**
1. Excessive alert burden leading to fatigue
2. Poor integration with existing workflows
3. Insufficient training and change management
4. Lack of local adaptation
5. Over-reliance on model without validation

### 7.3 Privacy and Security

**HIPAA/GDPR Compliance Strategies:**
1. **Local Deployment:** Open-source models (Llama) on-premise
2. **De-identification:** Remove PHI before processing
3. **Secure APIs:** Encrypted connections, access controls
4. **Audit Trails:** Log all system interactions
5. **Business Associate Agreements:** For cloud-based solutions

**Data Governance:**
- Regular security audits
- Minimum necessary principle
- Role-based access control
- Incident response procedures
- Patient consent frameworks

---

## 8. Cost and Resource Considerations

### 8.1 Computational Costs

**Training Costs (from studies):**
- Me-LLaMA 70B: >100,000 A100 GPU hours
- Domain fine-tuning: $500-$150 per model (GPT-4 vs GPT-3.5)
- Open-source models (LLaMA, PaLM): Free except infrastructure

**Inference Costs:**
- GPT-4: ~$0.03-0.06 per request (varies by context length)
- GPT-3.5: ~$0.001-0.002 per request
- Local models: Infrastructure costs only
- Multi-agent systems: 3-4x single-agent costs

**Cost-Effectiveness Analysis:**
- Penda Health: Would prevent 51,000 errors annually across 400,000 visits
- Medication safety: Average error cost €111,727 per serious error
- ED overcrowding: Improved throughput reduces costly delays
- ROI typically positive if system reduces errors by >5%

### 8.2 Resource Requirements

**Personnel:**
- Clinical experts for validation and oversight
- ML engineers for deployment and maintenance
- Clinical informaticists for workflow integration
- Quality assurance specialists
- Training and support staff

**Technical Infrastructure:**
- GPU servers for local deployment (if using open-source models)
- Vector databases for RAG systems
- EMR integration APIs
- Monitoring and logging systems
- Backup and disaster recovery

**Maintenance:**
- Regular model updates (knowledge drift)
- Performance monitoring and retraining
- Knowledge base updates
- Bug fixes and improvements
- Regulatory compliance updates

---

## 9. Ethical and Regulatory Considerations

### 9.1 Key Ethical Concerns

**Accountability:**
- Who is responsible when AI makes an error?
- Clear documentation of AI vs human decisions
- Legal liability frameworks still evolving
- Recommendation: Maintain human as final decision-maker

**Transparency:**
- "Black box" nature of LLMs problematic in healthcare
- Explainable AI techniques improving but not perfect
- Patients deserve to know when AI is involved in their care
- Need for clear communication strategies

**Bias and Fairness:**
- Training data biases can perpetuate healthcare disparities
- Regular audits across demographic groups required
- Particular concern for underrepresented populations
- Example: GPT-4 shown to have gender and racial biases

**Patient Autonomy:**
- Informed consent for AI-assisted care
- Right to opt-out of AI involvement
- Understanding how AI influences treatment recommendations

### 9.2 Regulatory Landscape

**Current Status:**
- FDA regulates AI as medical devices (SaMD)
- Clinical decision support may be exempt if:
  - Human in the loop
  - Transparent reasoning
  - Evidence-based recommendations
- Rapidly evolving regulatory framework

**Best Practices:**
- Comprehensive validation studies before deployment
- Continuous monitoring post-deployment
- Adverse event reporting systems
- Regular regulatory compliance reviews
- Documentation of intended use and limitations

---

## 10. Limitations and Future Directions

### 10.1 Current Limitations

**Technical:**
1. Hallucination remains problematic despite mitigation strategies
2. Difficulty processing complex structured data (tables, images)
3. Temporal reasoning challenges for longitudinal data
4. Computational costs limit widespread deployment
5. Knowledge cutoff dates create information gaps

**Clinical:**
1. Limited validation in diverse patient populations
2. Most studies use synthetic or retrospective data
3. Rare diseases and edge cases poorly handled
4. Integration with existing workflows challenging
5. Alert fatigue from excessive or irrelevant recommendations

**Practical:**
1. Requires significant technical expertise to deploy
2. High initial implementation costs
3. Resistance from clinicians
4. Liability and regulatory uncertainties
5. Limited generalizability across institutions

### 10.2 Research Gaps

**High-Priority Research Needs:**
1. **Prospective Clinical Trials:**
   - Patient outcomes as primary endpoints
   - Randomized controlled trials of AI-assisted care
   - Long-term follow-up studies
   - Multi-site validation

2. **Rare Disease and Edge Cases:**
   - Performance on uncommon presentations
   - Few-shot learning for rare conditions
   - Transfer learning across related diseases

3. **Longitudinal Decision Support:**
   - Temporal reasoning improvements
   - Disease progression prediction
   - Treatment response monitoring

4. **Bias and Fairness:**
   - Systematic evaluation across demographic groups
   - Debiasing techniques for clinical LLMs
   - Health equity impact assessments

5. **Human-AI Collaboration:**
   - Optimal division of labor
   - Trust calibration
   - Cognitive load impact
   - Training methodologies

### 10.3 Promising Future Directions

**1. Multimodal Integration:**
- Combining text, images, lab values, vital signs
- Integration with medical imaging AI
- Wearable device data incorporation
- Genomic data integration

**2. Advanced Grounding Techniques:**
- Real-time guideline updates
- Personalized knowledge bases
- Federated learning across institutions
- Causal reasoning frameworks

**3. Voice and Conversational AI:**
- Speech-to-text for emergency calls (ED study suggestion)
- Voice-activated documentation
- Patient-facing conversational agents
- Real-time interpretation during patient encounters

**4. Specialized Domain Models:**
- Emergency medicine-specific LLMs
- Subspecialty fine-tuned models
- Procedure-specific guidance systems

**5. Regulatory and Policy Innovation:**
- Frameworks for adaptive AI systems
- Continuous learning with regulatory oversight
- International harmonization
- Insurance and reimbursement models

---

## 11. Recommendations for Practitioners

### 11.1 For Health Systems Considering LLM Deployment

**Start Small, Scale Gradually:**
1. Begin with low-risk, high-volume tasks
2. Pilot in controlled environment
3. Extensive validation before expansion
4. Continuous monitoring and evaluation

**Essential Pre-Deployment Steps:**
1. Comprehensive needs assessment
2. Stakeholder engagement (clinicians, patients, IT, legal)
3. Pilot study with clear success metrics
4. Privacy and security review
5. Regulatory compliance assessment
6. Training program development
7. Incident response plan

**Key Success Factors:**
1. Clinical leadership and buy-in
2. Workflow integration expertise
3. Adequate technical resources
4. Change management strategy
5. Continuous quality improvement

### 11.2 For Researchers

**Study Design Recommendations:**
1. Prospective validation whenever possible
2. Diverse patient populations
3. Realistic data conditions (missing values, noise)
4. Head-to-head comparisons with current standard of care
5. Patient-centered outcomes as endpoints
6. Economic analyses (cost-effectiveness)

**Reporting Standards:**
1. Detailed model architecture and hyperparameters
2. Complete dataset descriptions
3. Inter-rater reliability metrics
4. Error analysis and failure modes
5. Computational requirements
6. Code and data availability
7. Limitations and generalizability

### 11.3 For Clinicians

**Critical Evaluation Framework:**
1. **Validation:** What evidence supports this tool's safety and effectiveness?
2. **Transparency:** Can I understand how it reaches conclusions?
3. **Bias:** Has it been tested across diverse patient populations?
4. **Integration:** Will it fit my workflow or disrupt it?
5. **Accountability:** Who is responsible if it makes an error?
6. **Override:** Can I easily disagree with its recommendations?

**Best Practices for Use:**
1. Maintain clinical skepticism
2. Verify recommendations against current guidelines
3. Document when and why you disagree with AI
4. Report errors and near-misses
5. Participate in ongoing evaluation
6. Advocate for adequate training

---

## 12. Conclusion

Large language models show substantial promise for clinical decision support, but successful implementation requires careful attention to:

1. **Grounding and Hallucination Mitigation:** RAG, clinical practice guideline integration, and temporal knowledge graphs significantly reduce errors

2. **Human-AI Collaboration:** Co-pilot and safety net modes consistently outperform autonomous deployment, often by 2x or more

3. **Multi-Agent Architectures:** Systems that mirror actual clinical team structures show superior performance for complex tasks

4. **Implementation Strategy:** Technology alone is insufficient; clinical workflow alignment and active deployment are equally critical

5. **Integration with Structured Data:** Standardized terminologies (SNOMED CT, RxNorm, LOINC) and proper EHR integration are essential

6. **Continuous Validation:** Ongoing monitoring, error analysis, and iterative improvement are necessary for safe deployment

**The Path Forward:**

The evidence suggests LLMs are ready for carefully designed clinical decision support applications, particularly in co-pilot modes with appropriate safeguards. However, they are not yet suitable for fully autonomous clinical decision-making. Success will require:

- Interdisciplinary collaboration (clinicians, informaticists, ML engineers, ethicists)
- Rigorous prospective validation
- Thoughtful regulatory frameworks
- Commitment to continuous improvement
- Focus on augmenting rather than replacing clinical expertise

As models continue to improve and implementation science matures, LLM-based CDSS has the potential to significantly enhance clinical care quality, reduce errors, and alleviate clinician burden. The key is responsible, evidence-based deployment with patient safety as the paramount concern.

---

## References

This synthesis is based on 45 papers identified through ArXiv searches on:
- "large language model" AND "clinical" AND "decision support"
- ("GPT" OR "LLaMA") AND "medical" AND "diagnosis"
- "clinical NLP" AND "transformer" AND "EHR"

### Key Papers Reviewed in Detail:

1. **Korom R, et al. (2025).** AI-based Clinical Decision Support for Primary Care: A Real-World Study. arXiv:2507.16947v1

2. **Oniani D, et al. (2024).** Enhancing Large Language Models for Clinical Decision Support by Incorporating Clinical Practice Guidelines. arXiv:2401.11120v2

3. **Ong JCL, et al. (2024).** Development and Testing of a Novel Large Language Model-Based Clinical Decision Support Systems for Medication Safety in 12 Clinical Specialties. arXiv:2402.01741v2

4. **Han S, Choi W (2024).** Development of a Large Language Model-based Multi-Agent Clinical Decision Support System for Korean Triage and Acuity Scale (KTAS)-Based Triage and Treatment Planning in Emergency Departments. arXiv:2408.07531v2

5. **Li D, et al. (2025).** CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records. arXiv:2507.22533v1

### Additional Papers Reviewed:
- Medication safety and prescribing error detection (6 papers)
- Clinical NLP and entity extraction (8 papers)
- Emergency medicine applications (5 papers)
- EHR integration and temporal reasoning (7 papers)
- Model evaluation and validation (6 papers)
- Medical diagnosis and differential diagnosis (8 papers)

---

*Document prepared: November 30, 2025*
*Total papers analyzed: 45*
*Focus areas: Clinical decision support, medication safety, emergency medicine, clinical NLP, EHR integration*
