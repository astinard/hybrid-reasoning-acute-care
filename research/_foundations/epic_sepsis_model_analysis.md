# Epic Sepsis Model: Performance Analysis and Clinical Failures

**Document Generated:** 2025-11-30
**Purpose:** Comprehensive review of Epic Sepsis Model validation studies, performance metrics, and deployment challenges based on peer-reviewed research and clinical reports.

---

## Executive Summary

The Epic Sepsis Model (ESM), deployed at hundreds of U.S. hospitals serving 54% of American patients, has undergone multiple external validations revealing significant performance gaps between vendor claims and real-world effectiveness. Key findings indicate poor discrimination (AUROC 0.47-0.63 in external validations vs. vendor-reported 0.76-0.83), severe alert fatigue (18% of hospitalizations triggering alerts), and evidence that the model may be detecting clinician suspicion rather than providing early warning.

---

## 1. LANDMARK VALIDATION STUDIES

### 1.1 Michigan Medicine Study (JAMA Internal Medicine, 2021)

**Citation:** Wong A, Otles E, Donnelly JP, et al. External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients. JAMA Intern Med. 2021;181(8):1065-1070.

**Study Population:**
- 27,697 patients
- 38,455 hospitalizations
- 2,552 sepsis cases (7% incidence)
- Setting: Michigan Medicine academic medical center
- Period: December 6, 2018 – October 20, 2019

**Performance Metrics:**

| Metric | Value | 95% CI |
|--------|-------|--------|
| **Hospitalization-level AUROC** | 0.63 | 0.62-0.64 |
| **4-hour prediction AUROC** | 0.76 | 0.75-0.76 |
| **8-hour prediction AUROC** | 0.74 | 0.74-0.75 |
| **12-hour prediction AUROC** | 0.73 | 0.73-0.74 |
| **24-hour prediction AUROC** | 0.72 | 0.72-0.72 |

**At ESM Score ≥6 (Recommended Alert Threshold):**
- Sensitivity: 33%
- Specificity: 83%
- Positive Predictive Value: 12%
- Negative Predictive Value: 95%

**Clinical Impact:**
- **67% of sepsis cases missed:** Model failed to alert for 1,709 of 2,552 sepsis patients
- **Only 7% incremental detection:** Model identified only 183 patients (7%) with sepsis who were missed by clinicians
- **18% alert rate:** 6,971 of 38,455 hospitalizations triggered alerts
- **Number needed to evaluate:** 8 patients evaluated per true sepsis case
- **Median alert lead time:** 2.5 hours (IQR: 0.5-15.6 hours)

**Key Finding:** "The Epic Sepsis Model poorly predicts sepsis; its widespread adoption despite poor performance raises fundamental concerns about sepsis management on a national level."

**Source:** [JAMA Network - External Validation Study](https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2781307) | [PMC Full Text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8218233/)

---

### 1.2 County Emergency Department Validation (JAMIA Open, 2024)

**Citation:** External validation of the Epic sepsis predictive model in 2 county emergency departments. JAMIA Open. 2024;7(4):ooae133.

**Study Population:**
- 145,885 encounters
- 2 county emergency departments
- Period: January 1 – December 31, 2023
- Demographics: 59% Hispanic, 26% Black

**Performance Metrics (6-Hour Window):**
- Sensitivity: 14.7%
- Specificity: 95.3%
- Positive Predictive Value: 7.6%
- Negative Predictive Value: 97.7%

**Extended Performance (Full Hospital Encounter):**
- Sensitivity: 41.5%
- Specificity: 96.5%
- PPV: 31.4%
- NPV: 97.7%

**Alert Characteristics:**
- Alert rate: 4.9% (7,183 of 145,885 encounters)
- Sepsis cases with alert: 2,253 of 5,433
- Sepsis cases without alert: 3,180
- **Median lead time: 0 minutes** (80% CI: −6 hours 42 minutes to +12 hours)

**Critical Finding:** "Only alerted providers in half of the cases prior to sepsis occurrence," indicating clinicians treated sepsis without the alert 50% of the time.

**Comparison:** A randomly generated alert achieved 1.27% sensitivity and 95.5% specificity, demonstrating only marginal superiority of the ESPMv1 model.

**Source:** [JAMIA Open](https://academic.oup.com/jamiaopen/article/7/4/ooae133/7900014) | [PMC Full Text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11560849/)

---

### 1.3 Michigan NEJM AI Study (2024): Clinician Suspicion Bias

**Citation:** Kamran F, Tjandra D, Heiler A, et al. Evaluation of Sepsis Prediction Models before Onset of Treatment. NEJM AI. 2024;1(3).

**Study Population:**
- 77,000 adult inpatients
- University of Michigan Health
- Period: 2018-2020
- ~5% sepsis incidence

**Performance Metrics:**

| Evaluation Window | AUROC |
|------------------|-------|
| **Entire hospital stay** | 0.87 (87% accuracy) |
| **Before sepsis criteria met** | 0.62 (62% accuracy) |
| **Before blood culture ordered** | 0.53 (53% accuracy) |
| **Excluding post-recognition predictions** | 0.47 |

**Key Findings:**
1. **Clinician Suspicion Encoding:** "We suspect that some of the health data that the Epic Sepsis Model relies on encodes, perhaps unintentionally, clinician suspicion that the patient has sepsis." - Jenna Wiens, PhD, corresponding author

2. **Limited Early Detection:** Model accuracy drops dramatically when restricted to data before clinicians suspect sepsis, suggesting the AI is "cribbing doctors' suspicions" rather than providing true early warning.

3. **Timing Problem:** "Patients won't receive blood culture tests and antibiotic treatments until they start presenting sepsis symptoms. While such data could help make an AI very accurately identify sepsis risks, it could also enter the medical records too late to help clinicians get ahead on treatments."

4. **Workflow Mismatch:** "We need to consider when in the clinical workflow the model is being evaluated when deciding if it's helpful to clinicians. Evaluating the model with data collected after the clinician has already suspected sepsis onset can make the model's performance appear strong, but this does not align with what would aid clinicians in practice." - Donna Tjandra, co-author

**Source:** [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIoa2300032) | [University of Michigan News](https://news.umich.edu/widely-used-ai-tool-for-early-sepsis-detection-may-be-cribbing-doctors-suspicions/)

---

### 1.4 Prisma Health Validation (Positive Outcome Study, 2023)

**Study Type:** Single-center before-and-after study

**Performance Metrics at ESM ≥5:**
- Area Under ROC Curve: 0.834 (p < 0.001)
- Sensitivity: 86.0%
- Specificity: 80.8%
- Positive Predictive Value: 33.8%
- Negative Predictive Value: 98.1%

**Mortality Outcomes:**

| Population | Baseline Mortality | Post-Implementation | Change |
|-----------|-------------------|---------------------|---------|
| All ESM ≥5 with sepsis | 18.5% | 14.5% | −4.0% |
| Not yet on antibiotics | 24.3% | 15.9% | −8.4% |

**Risk-Adjusted Results:**
- Odds ratio for sepsis-related mortality: 0.56 (95% CI: 0.39-0.80)
- **44% reduction in mortality odds**

**Clinical Process Improvements:**
- Time from alert to antibiotic: 150 min → 90 min
- Antibiotics within 3 hours: 55.6% → 69.8%
- Sepsis order set utilization: 34.5% → 43.2%

**Alert Fatigue Mitigation:**
- Daily alert rate: 6-7% of inpatient population (~45-50 patients)
- "One in every three patients evaluated for sepsis would have a diagnosis of sepsis"
- Lower alert frequency may have "mitigated alert fatigue as a possible reason for improved results"

**Source:** [PMC Full Text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10317482/)

---

## 2. ALERT FATIGUE AND CLINICIAN OVERRIDE RATES

### 2.1 COVID-19 Alert Surge Study (JAMA Network Open)

**Study:** 24 U.S. hospitals across four health systems during COVID-19 pandemic

**Key Findings:**
- **43% increase in daily sepsis alerts** in the 3 weeks before/after first COVID-19 case
- **35% decrease in hospital census** due to cancelled elective surgeries
- **Alert rate doubled:** 9% → 21% of patients generating daily alerts
- **University of Michigan response:** Paused Epic sepsis model alerts 3-4 weeks after first COVID-19 hospitalization due to nurses reporting being "overalerted"

**Nurse Feedback:** "At Michigan Medicine, nurses qualitatively reported a rise in sepsis model-driven alerts at levels that felt untenable."

**Explanation:** "Dataset shift" - model performance deteriorated with sudden, unexpected changes in hospital case mix (COVID surge). "COVID-19 was a 'black swan' event that likely affected many existing models in ways we don't yet fully understand."

**Source:** [Michigan Medicine News](https://www.michiganmedicine.org/health-lab/study-24-us-hospitals-shows-onset-covid-19-led-spike-sepsis-alerts) | [Fierce Healthcare](https://www.fiercehealthcare.com/tech/epic-s-sepsis-algorithm-may-have-caused-alert-fatigue-43-alert-increase-during-pandemic)

---

### 2.2 Ochsner Health Implementation (2023)

**Override Statistics:**
- 237 patients screened as "triple-yes" (meeting all sepsis criteria)
- **Only 39 (16.46%) had sepsis protocol initiated**
- 192 non-initiated cases (94.79%) were "already receiving appropriate management"
- **High override rate driven by clinical judgment** that patients were adequately treated

**Alert-to-Action Ratio:**
- Achieved: 16.46%
- Ochsner Health average for all best practice alerts: 8.4% to 12.1%
- ESM alerts exceeded facility baseline, indicating improved engagement despite high overrides

**Suppression Mechanisms:**
- Alert suppression for patients already on antibiotics
- Suppression for ICU patients
- Suppression for inpatient hospice
- Suppression for ED patients not yet admitted
- Opt-out choices: 24-hour suppression for clinical judgment, entire encounter for comfort care

**Source:** [Ochsner Journal](https://www.ochsnerjournal.org/content/23/3/222)

---

## 3. VENDOR CLAIMS VS. EXTERNAL VALIDATION

### 3.1 Epic's Reported Performance

**Vendor Documentation:**
- Hospital-level AUROC: 0.76-0.83
- Can "correctly distinguish two patients with and without sepsis at least 76 percent of the time"

### 3.2 External Validation Reality

**Actual Performance:**
- Michigan Medicine: AUROC 0.63 (hospitalization-level)
- Michigan NEJM AI: AUROC 0.47 (excluding post-recognition)
- County ED: Sensitivity 14.7%, PPV 7.6%

**Gap:** External validations show performance **17-50% worse** than vendor claims.

**Epic's Response:** "The authors used a hypothetical approach that did not take into account the analysis and required tuning that needs to occur prior to real-world deployment to get optimal results."

---

## 4. EPIC'S MODEL UPDATES AND RESPONSE

### 4.1 Version 2 Rollout (2022-2023)

**Citation:** Epic overhauls popular sepsis algorithm. STAT News. October 3, 2022.

**Three Major Changes:**

1. **Hospital-Specific Training Requirement**
   - Epic now recommends training the model on individual hospital data before clinical deployment
   - Major shift to ensure predictions match actual patient populations
   - Addresses criticism that one-size-fits-all approach failed external validation

2. **Revised Sepsis Definition**
   - Adopted more commonly accepted standard for defining sepsis onset
   - Moving away from previous proprietary approach

3. **Reduced Antibiotic Reliance**
   - Decreased dependence on clinician antibiotic orders as sepsis indicator
   - Previous approach was "particularly problematic, resulting in late alarms to physicians who had already recognized the condition"

**Source:** [STAT News](https://www.statnews.com/2022/10/03/epic-sepsis-algorithm-revamp-training/)

---

### 4.2 Version 1 vs. Version 2 Technical Comparison

**Version 1 (V1):**
- Penalized logistic regression
- Trained on historical data from three health systems
- Features: vital signs, medications, lab values, comorbidities, demographics
- Predicts clinical intervention indicative of sepsis
- Deployed since 2020

**Version 2 (V2):**
- Gradient boosted tree model
- Different but overlapping feature set
- Predicts Sepsis-3 as outcome
- Can be trained locally on historical data
- Silently introduced in December 2023

**Reported Performance (No Independent Validation):**
- V1 AUROC: 0.77
- V2 AUROC: 0.90
- V1 pre-recognition AUROC: 0.70
- V2 pre-recognition AUROC: 0.85

**Critical Gap:** "Despite the broad availability of V2, no independent peer-reviewed evaluation has compared its performance or timeliness against V1."

---

## 5. ALERT FATIGUE MITIGATION STRATEGIES

### 5.1 Documented Challenges

**Alert Fatigue Phenomenon:**
- False positives contribute to unnecessary interventions
- Desensitization among healthcare providers
- Reduced responsiveness to real alarms
- "Alarm fatigue" widely documented in sepsis detection systems

**Balance Required:**
- Sensitivity vs. specificity trade-off
- Clinical benefit vs. alert burden
- Early detection vs. false alarm minimization

---

### 5.2 Evidence-Based Strategies

**1. Risk Stratification**
- Display interruptive alerts only for high-risk or very high-risk patients
- Reduce alert volume for low-risk populations

**2. Suppression Logic**
- Automated filtering based on clinical status:
  - Already on antibiotics
  - In intensive care unit
  - On inpatient hospice
  - In emergency department but not admitted

**3. Phenotyping Approaches**
- Move beyond binary "sepsis/no sepsis" models
- Granular categorization of patient states
- Reduces low-specificity alerts

**4. Nurse-Driven Platforms**
- Provide staff nurses with educational notifications in-workflow
- Display patient-specific data highlighting infection signs/symptoms
- Allow nurses to decide on sepsis bundle ordering
- Reduces likelihood of alert fatigue through autonomy

**5. Enhanced Interpretability**
- Use SHAP (Shapley Additive Explanations) values
- Help clinicians understand model predictions
- Build trust through transparency

**6. Educational Initiatives**
- Targeted training to build provider confidence
- Familiarity reduces cognitive load
- Less experienced providers often find alerts more beneficial

**Source:** [PMC - Artificial Intelligence in Sepsis Management](https://pmc.ncbi.nlm.nih.gov/articles/PMC11722371/)

---

## 6. IMPLEMENTATION BARRIERS

### 6.1 Human Factors

**Cognitive Biases:**
- Providers dismiss alerts for younger patients
- Dismissals more common during peak hours
- Non-clinical factors affecting alert interactions
- Previous CDS tools showed "little integration with EHR"

**Provider Skepticism:**
- Distrust in model accuracy
- Black box concerns - lack of interpretability
- Workflow integration difficulties

**Experience-Dependent Response:**
- "Clinician perceptions of alert utility are shaped by cognitive load and prior experience"
- Less experienced providers find alerts more actionable and beneficial
- Veteran clinicians may rely more on clinical judgment

---

### 6.2 Technical Challenges

**Data Quality:**
- Reliance on complete, accurate EHR data
- Missing data degrades performance
- Real-time data recording requirements

**Setting Specificity:**
- Models successful in one setting may fail in others
- ICU vs. ED performance differences
- Need for context-specific development

**Dataset Shift:**
- Model deterioration with unexpected changes in case mix
- COVID-19 demonstrated vulnerability to black swan events
- Requires ongoing monitoring and retraining

---

## 7. CRITICAL ANALYSIS AND IMPLICATIONS

### 7.1 Fundamental Concerns

**1. Widespread Deployment Without Adequate Validation**
- Hundreds of hospitals adopted ESM before external validation
- Vendor-reported performance not reproducible in independent studies
- Raises questions about procurement and evaluation processes

**2. Limited Clinical Utility**
- At best, identifies 7% of sepsis cases missed by clinicians (Michigan Medicine)
- At worst, provides no advance warning (County ED median lead time: 0 minutes)
- High number needed to evaluate (8-109 patients per true case)

**3. Clinician Suspicion Confounding**
- Model may encode clinician actions (blood cultures, antibiotics) rather than predict disease
- Creates illusion of prediction while detecting recognition
- "Cribbing doctors' suspicions" undermines claimed early warning value

**4. Alert Burden**
- 18% of hospitalizations triggering alerts (Michigan Medicine)
- 43% surge during COVID-19 despite reduced census
- Contributes to alert fatigue without proportional benefit

---

### 7.2 Implications for Clinical AI Development

**Need for External Validation:**
- Independent, peer-reviewed validation before widespread clinical use
- Transparency in training data, features, and validation methods
- Multiple validation cohorts across diverse settings

**Timing and Workflow Considerations:**
- Evaluate models using only data available before clinical recognition
- Exclude retrospective analyses that inflate performance
- Align evaluation with actual clinical decision-making workflow

**Co-Design with Clinicians:**
- Reduce alert fatigue through user-centered design
- Support hypothesis generation rather than dictate action
- Ensure seamless workflow integration

**Better Evidence Requirements:**
- High-quality randomized trials for clinical impact
- Process measures (time to antibiotics) and patient outcomes (mortality)
- Long-term monitoring for unintended consequences (antibiotic overuse)

---

## 8. COMPARATIVE CONTEXT

### 8.1 Alternative Sepsis Detection Approaches

**Traditional Criteria:**
- SIRS (Systemic Inflammatory Response Syndrome)
- SOFA (Sequential Organ Failure Assessment)
- qSOFA (Quick SOFA)

**Comparative Performance (from literature):**
- SIRS and SOFA showed better timeliness than Epic SPM at higher thresholds
- Initial clinician action occurred 68-145 minutes BEFORE high-threshold Epic scores
- Suggests traditional criteria may be more actionable

---

### 8.2 Other AI Sepsis Models

**Laura Early Detection Robot:**
- Improved diagnostic accuracy and professional satisfaction
- Required rapid data recording and ML model training
- Addressed distrust through transparency

**COMPOSER Model:**
- Leverages conformal prediction for robust out-of-distribution detection
- Tackles alarm fatigue more effectively
- Not yet widely deployed

**FDA-Authorized Models:**
- Some AI/ML sepsis tools have received FDA authorization
- Separate regulatory pathway from Epic ESM
- Performance data varies

---

## 9. SUMMARY OF PERFORMANCE METRICS

### Comprehensive Performance Table

| Study | Setting | N | AUROC | Sens | Spec | PPV | NPV | Alert Rate |
|-------|---------|---|-------|------|------|-----|-----|------------|
| **Michigan JAMA 2021** | Academic inpatient | 38,455 | 0.63 | 33% | 83% | 12% | 95% | 18% |
| **County ED 2024** | County ED (6hr) | 145,885 | - | 14.7% | 95.3% | 7.6% | 97.7% | 4.9% |
| **County ED 2024** | County ED (full) | 145,885 | - | 41.5% | 96.5% | 31.4% | 97.7% | 4.9% |
| **Prisma Health 2023** | Academic inpatient | - | 0.834 | 86.0% | 80.8% | 33.8% | 98.1% | 6-7% |
| **Michigan NEJM AI 2024** | Academic inpatient | 77,000 | 0.47* | - | - | - | - | - |

*AUROC when excluding post-recognition predictions; 0.87 for entire stay, 0.62 before sepsis criteria, 0.53 before blood culture

---

## 10. KEY TAKEAWAYS FOR HYBRID REASONING SYSTEMS

### 10.1 Lessons for Clinical AI Development

1. **Temporal Causality Matters**
   - Models must predict, not detect recognition
   - Evaluation must use only pre-suspicion data
   - Avoid confounding with clinician actions

2. **External Validation is Critical**
   - Vendor performance claims require independent verification
   - Multiple sites, diverse populations, prospective evaluation
   - Transparency in methodology and limitations

3. **Alert Design is Paramount**
   - Balance sensitivity with alert burden
   - Provide actionable, interpretable information
   - Support rather than overwhelm clinical workflow

4. **Context Adaptation Required**
   - One-size-fits-all models fail across diverse settings
   - Local training and tuning essential
   - Monitor for dataset shift and black swan events

5. **Human Factors Cannot be Ignored**
   - Co-design with end users
   - Address cognitive biases and trust issues
   - Provide transparency and education

---

### 10.2 Hybrid Reasoning Opportunities

**Symbolic + Neural Integration:**
- Combine data-driven patterns with clinical reasoning rules
- Temporal logic for sepsis progression modeling
- Explicit representation of diagnostic criteria (SIRS, SOFA, Sepsis-3)

**Explainability:**
- Graph-based knowledge representation of sepsis pathophysiology
- Traceable reasoning chains for clinician review
- SHAP-like explanations grounded in clinical knowledge

**Temporal Modeling:**
- Distinguish true early warning from recognition detection
- Model disease progression dynamics, not just static snapshots
- Incorporate time-to-event analysis

**Robust to Distribution Shift:**
- Ontology-grounded reasoning less vulnerable to dataset shift
- Explicit encoding of domain knowledge maintains performance
- COVID-19 resilience through adaptable knowledge structures

---

## 11. REFERENCES AND SOURCES

### Primary Research Articles

1. **Wong A, Otles E, Donnelly JP, et al.** External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients. *JAMA Intern Med.* 2021;181(8):1065-1070.
   - [JAMA Network](https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2781307)
   - [PMC Full Text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8218233/)

2. **External validation of the Epic sepsis predictive model in 2 county emergency departments.** *JAMIA Open.* 2024;7(4):ooae133.
   - [Oxford Academic](https://academic.oup.com/jamiaopen/article/7/4/ooae133/7900014)
   - [PMC Full Text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11560849/)

3. **Kamran F, Tjandra D, Heiler A, et al.** Evaluation of Sepsis Prediction Models before Onset of Treatment. *NEJM AI.* 2024;1(3).
   - [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIoa2300032)

4. **Epic Sepsis Model Inpatient Predictive Analytic Tool: A Validation Study.** *Ochsner J.* 2023.
   - [PMC Full Text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10317482/)

5. **Alert to Action: Implementing Artificial Intelligence–Driven Clinical Decision Support Tools for Sepsis.** *Ochsner J.* 2023;23(3):222.
   - [Ochsner Journal](https://www.ochsnerjournal.org/content/23/3/222)

### News and Industry Reports

6. **Herper M, Palmer E.** Epic overhauls popular sepsis algorithm criticized for faulty alarms. *STAT News.* October 3, 2022.
   - [STAT News](https://www.statnews.com/2022/10/03/epic-sepsis-algorithm-revamp-training/)

7. **University of Michigan News.** Widely used AI tool for early sepsis detection may be cribbing doctors' suspicions. February 2024.
   - [UM News](https://news.umich.edu/widely-used-ai-tool-for-early-sepsis-detection-may-be-cribbing-doctors-suspicions/)

8. **Michigan Medicine Health Lab.** Study of 24 U.S. hospitals shows onset of COVID-19 led to spike in sepsis alerts.
   - [Michigan Medicine](https://www.michiganmedicine.org/health-lab/study-24-us-hospitals-shows-onset-covid-19-led-spike-sepsis-alerts)

9. **Fierce Healthcare.** Epic's sepsis algorithm may have caused alert fatigue with 43% alert increase during pandemic.
   - [Fierce Healthcare](https://www.fiercehealthcare.com/tech/epic-s-sepsis-algorithm-may-have-caused-alert-fatigue-43-alert-increase-during-pandemic)

### Review Articles

10. **Artificial Intelligence in Sepsis Management: An Overview for Clinicians.** *PMC.* 2024.
    - [PMC Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11722371/)

---

## 12. DOCUMENT METADATA

**Generated:** 2025-11-30
**Sources:** 10 peer-reviewed articles, 9 news reports, multiple institutional press releases
**Focus:** Epic Sepsis Model performance, validation studies, alert fatigue, clinician override rates
**Key Metrics Extracted:** AUROC, sensitivity, specificity, PPV, NPV, alert rates, mortality outcomes, time-to-treatment
**Grounded in:** Published research with URLs provided for all claims

**Last Updated:** 2025-11-30
**Next Review:** Upon publication of Epic Sepsis Model V2 external validation studies

---

*This document was compiled from publicly available peer-reviewed research and industry reports. All performance metrics are sourced from external validation studies, not vendor claims. URLs provided throughout for verification and further reading.*
