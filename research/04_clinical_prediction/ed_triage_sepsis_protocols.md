# Emergency Department Triage Systems and Sepsis Protocols

## Executive Summary

This document provides a comprehensive overview of emergency department (ED) triage systems and sepsis management protocols, focusing on the Emergency Severity Index (ESI), Sepsis-3 definitions, CMS SEP-1 bundle requirements, and the Surviving Sepsis Campaign Hour-1 Bundle. These evidence-based tools and protocols are critical for rapid identification, risk stratification, and timely treatment of critically ill patients in acute care settings.

---

## Table of Contents

1. [Emergency Severity Index (ESI) Triage System](#emergency-severity-index-esi-triage-system)
2. [Sepsis-3: SOFA and qSOFA Criteria](#sepsis-3-sofa-and-qsofa-criteria)
3. [SIRS Criteria and Legacy Sepsis Definitions](#sirs-criteria-and-legacy-sepsis-definitions)
4. [CMS SEP-1 Sepsis Bundle Requirements](#cms-sep-1-sepsis-bundle-requirements)
5. [Surviving Sepsis Campaign Hour-1 Bundle](#surviving-sepsis-campaign-hour-1-bundle)
6. [Septic Shock Management](#septic-shock-management)
7. [Clinical Implications and Implementation](#clinical-implications-and-implementation)
8. [References](#references)

---

## Emergency Severity Index (ESI) Triage System

### Overview

The [Emergency Severity Index (ESI)](https://www.ahrq.gov/patient-safety/settings/emergency-dept/esi.html) is a five-level emergency department triage algorithm, initially developed in 1998 by emergency physicians Richard Wuerz and David Eitel, along with emergency nurses Nicki Gilboy, Taula Tanabe, and Debbie Travers. In 2019, the Emergency Nurses Association acquired the ESI five-level emergency triage system.

The ESI is a five-level ED triage algorithm that provides clinically relevant stratification of patients into five groups from **Level 1 (most urgent) to Level 5 (least urgent)** on the basis of acuity and resource needs. As of 2019, **94% of United States EDs** use the ESI algorithm in triage.

### ESI Level Definitions

**Levels 1-2:** Based on **patient acuity** (severity of condition)
**Levels 3-5:** Based on **resource prediction** (anticipated number of resources needed)

| ESI Level | Description | Criteria |
|-----------|-------------|----------|
| **Level 1** | Immediate life-saving intervention required | Requires immediate intervention such as CPR, endotracheal intubation, or emergent procedures |
| **Level 2** | High-risk situation; should not wait | Patient at risk for deterioration or experiencing severe pain/distress; requires immediate physician evaluation |
| **Level 3** | Multiple resources needed | Stable patient predicted to require **2 or more resources** |
| **Level 4** | One resource needed | Stable patient predicted to require **1 resource** |
| **Level 5** | No resources needed | Stable patient predicted to require **0 resources** |

### The Four Decision Points

The ESI algorithm uses **four decision points (A, B, C, and D)** to sort patients into one of the five triage levels. Decision points must be performed in sequential order:

#### **Decision Point A: Does this patient require immediate life-saving intervention?**
- If YES → **ESI Level 1**
- Interventions include: CPR, emergent airway management, respiratory failure requiring immediate intervention, cardiac arrest, unresponsive patient, severe respiratory distress, signs of shock

#### **Decision Point B: Is this a high-risk patient who should not wait?**
- If YES → **ESI Level 2**
- High-risk situations include:
  - Confused/lethargic/disoriented
  - Severe pain or distress
  - Signs of potential deterioration
  - High-risk presentations (chest pain, new-onset stroke symptoms)

#### **Decision Point C: How many resources will this patient need?**
- **2 or more resources** → **ESI Level 3**
- **1 resource** → **ESI Level 4**
- **0 resources** → **ESI Level 5**

Resources are predicted based on typical care for similar patient presentations

#### **Decision Point D: What are the patient's vital signs?**
- Identifies more subtle high-risk patients who may have been misclassified
- Abnormal vital signs may warrant re-evaluation and potential upgrade to Level 2

### ESI Resources Definition

**What COUNTS as a Resource:**
- Laboratory tests (blood, urine)
- Electrocardiogram (ECG)
- Radiologic imaging (X-ray, CT, MRI, ultrasound, angiography)
- Intravenous fluids (for hydration)
- IV, IM, or nebulized medications
- Specialty consultation
- Procedures (suturing, casting, etc.)

**What DOES NOT COUNT as a Resource:**
- Oral medications
- Simple wound care
- Tetanus immunization
- Prescription refills
- Crutches or splints
- Point-of-care testing (e.g., bedside glucose)

### Special Considerations

**Pediatric Patients:**
- The ESI should be used in conjunction with the **PAT (Pediatric Assessment Triangle)** and focused pediatric history
- Age-appropriate vital signs must be considered

**Mass Casualty Incidents:**
- ESI should NOT be used for mass casualty or major trauma incidents
- Use **START (Simple Triage and Rapid Treatment)** or **JumpSTART** for pediatric patients instead

### Performance and Validation

- The ESI has been found to be **reliable, consistent, and accurate** in multiple studies, languages, age groups, and countries
- The Emergency Severity Index was updated to Version 5, which emphasizes identification of abnormal vital signs in low-acuity patients (ESI Levels 3, 4, and 5) to prevent undertriage
- Research indicates that **mistriage occurred in almost one-third of patients** based on ESI Version 4, with studies demonstrating **59% accuracy** in assigning acuity

---

## Sepsis-3: SOFA and qSOFA Criteria

### Background and Definitions

In 2016, the [Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4968574/) were published, representing a significant update to previous sepsis definitions.

**Sepsis (Sepsis-3 Definition):**
> "Life-threatening organ dysfunction caused by a dysregulated host response to infection"

Operationally defined as:
- **Suspected or documented infection** PLUS
- **Acute increase in SOFA score ≥2 points**

A SOFA score ≥2 reflects an overall **mortality risk of approximately 10%** in a general hospital population with suspected infection.

### SOFA Score (Sequential Organ Failure Assessment)

The [SOFA score](https://en.wikipedia.org/wiki/SOFA_score) is used to track a person's status during an ICU stay to determine the extent of organ function or rate of failure. The score is based on **six different organ systems**, with each receiving a score from **0 (normal) to 4 (most abnormal)**.

**Total SOFA Score Range: 0-24**

#### SOFA Score Components

| Organ System | Parameter | Score 0 | Score 1 | Score 2 | Score 3 | Score 4 |
|--------------|-----------|---------|---------|---------|---------|---------|
| **Respiration** | PaO₂/FiO₂ (mmHg) | ≥400 | <400 | <300 | <200 with ventilation | <100 with ventilation |
| **Coagulation** | Platelets (×10³/μL) | ≥150 | <150 | <100 | <50 | <20 |
| **Hepatic** | Bilirubin (mg/dL) | <1.2 | 1.2-1.9 | 2.0-5.9 | 6.0-11.9 | >12.0 |
| **Cardiovascular** | Mean Arterial Pressure (MAP) or Vasopressor Use | MAP ≥70 mmHg | MAP <70 mmHg | Dopamine ≤5 or dobutamine (any dose) | Dopamine >5 or epi/norepi ≤0.1 | Dopamine >15 or epi/norepi >0.1 |
| **Central Nervous System** | Glasgow Coma Scale (GCS) | 15 | 13-14 | 10-12 | 6-9 | <6 |
| **Renal** | Creatinine (mg/dL) or Urine Output | <1.2 | 1.2-1.9 | 2.0-3.4 | 3.5-4.9 or <500 mL/day | >5.0 or <200 mL/day |

**Note:** Vasopressor doses in μg/kg/min for at least 1 hour

#### SOFA Score Interpretation

| SOFA Score | Mortality Risk | Clinical Interpretation |
|------------|----------------|------------------------|
| 0-5 | <10% | Low mortality risk; relatively normal organ function |
| 6-9 | 10-40% | Moderate mortality risk; impaired organ function requiring aggressive monitoring |
| 10-13 | 30-70% | High mortality risk; significant organ dysfunction |
| >14 | >80% | Very high mortality risk; multi-organ failure |

**Key Clinical Points:**
- **Baseline SOFA score can be assumed to be zero** in patients not known to have preexisting organ dysfunction
- **Change in SOFA score ≥2 points** is now a defining characteristic of sepsis
- Mean total maximum SOFA score shows very good correlation to ICU outcome, with mortality ranging from **3.2% in patients without organ failure to 91.3% in patients with failure of all six organs**
- SOFA score may be calculated on admission to ICU and at each 24-hour period thereafter

### qSOFA (Quick SOFA) Criteria

The [qSOFA score](https://www.mdcalc.com/calc/2654/qsofa-quick-sofa-score-sepsis) is a **bedside tool** for rapid identification of adult patients with suspected infection who are likely to have poor outcomes **outside the ICU setting**.

#### qSOFA Criteria (1 point each)

1. **Respiratory Rate ≥22 breaths/min**
2. **Altered Mental Status** (GCS <15)
3. **Systolic Blood Pressure ≤100 mmHg**

**Score Range: 0-3 points**

#### qSOFA Interpretation

- **qSOFA ≥2:** High-risk patient; consider sepsis and possible ICU admission
- **qSOFA <2:** Lower risk; continue monitoring

**Clinical Context:**
- qSOFA was designed for use in **non-ICU settings** (emergency department, hospital floors, pre-hospital)
- Does NOT require laboratory values
- Provides **simple bedside criteria** to identify patients at risk for poor outcomes
- A positive qSOFA (≥2) should prompt further evaluation with full SOFA score and consideration of sepsis protocols

#### qSOFA Performance

- The qSOFA score has an **area under the receiver operator characteristic curve (AUC) of 0.81** for predicting in-hospital mortality
- More specific but less sensitive than SIRS criteria
- Should be used as a **screening tool**, not a definitive diagnostic criterion

---

## SIRS Criteria and Legacy Sepsis Definitions

### Systemic Inflammatory Response Syndrome (SIRS)

[SIRS](https://www.ncbi.nlm.nih.gov/books/NBK547669/) is diagnosed when **≥2 of the following criteria** are present:

1. **Temperature:** >38°C (100.4°F) OR <36°C (96.8°F)
2. **Heart Rate:** >90 beats per minute
3. **Respiratory Rate:** >20 breaths/min OR PaCO₂ <32 mmHg
4. **White Blood Cell Count:**
   - >12,000/μL (12 × 10⁹/L) OR
   - <4,000/μL (4 × 10⁹/L) OR
   - >10% immature forms (bands)

### Legacy Sepsis Definitions (Pre-Sepsis-3)

**Sepsis (Old Definition):**
- SIRS PLUS suspected or documented infection

**Severe Sepsis (Old Definition):**
- Sepsis PLUS organ dysfunction, hypoperfusion, or hypotension
- Indicators include:
  - Lactic acid above upper limit of normal
  - Systolic blood pressure <90 mmHg
  - Systolic blood pressure drop >40 mmHg from normal

**Septic Shock (Old Definition):**
- Severe sepsis with hypotension despite adequate fluid resuscitation

### SIRS Limitations

- **Overall sensitivity for detecting sepsis: only 50-60%**
- **One in eight patients admitted to ICU with sepsis does not meet SIRS criteria**
- Can be triggered by noninfectious causes (trauma, burns, pancreatitis)
- Led to development of Sepsis-3 definitions with SOFA/qSOFA

**Note:** While Sepsis-3 definitions are preferred, SIRS criteria are still used in some clinical protocols, including **CMS SEP-1** measure.

---

## CMS SEP-1 Sepsis Bundle Requirements

### Overview

The [CMS SEP-1 (Severe Sepsis and Septic Shock: Early Management Bundle)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9448659/) has been measured by CMS since **October 2015**. The measure's target population is **adult inpatients 18 years and older** with a diagnosis of severe sepsis or septic shock.

**Important Update:** SEP-1 has been added to CMS's **Hospital Value-Based Purchasing (HVBP) Program** in **FY 2026**, making SEP-1 a **pay-for-performance measure**.

### SEP-1 Definitions

SEP-1 uses **legacy sepsis definitions** (SIRS-based), NOT Sepsis-3 criteria:

**Sepsis (SEP-1):**
- **Source of infection** PLUS
- **≥2 SIRS criteria:**
  - Temperature >38°C or <36°C
  - Heart rate >90 bpm
  - Respiratory rate >20 or PaCO₂ <32 mmHg
  - WBC >12,000/mm³ or <4,000/mm³ or >10% bands

**Severe Sepsis (SEP-1):**
- **Sepsis** PLUS
- **Organ dysfunction:**
  - Serum lactic acid above upper limit of normal, OR
  - Systolic blood pressure <90 mmHg, OR
  - Systolic blood pressure drop >40 mmHg from normal

**Septic Shock (SEP-1):**
- **Severe sepsis** with hypotension despite adequate fluid resuscitation

### SEP-1 Bundle Components

There are **two bundles** included in SEP-1:

#### 1. Severe Sepsis Bundle (3-Hour and 6-Hour Elements)

**Within 3 hours of sepsis presentation:**
1. Measure lactate level
2. Obtain blood cultures **before** administering antibiotics
3. Administer broad-spectrum antibiotics

**Within 6 hours of sepsis presentation:**
4. Remeasure lactate if initial lactate is elevated (>2 mmol/L)

#### 2. Septic Shock Bundle (Additional Requirements)

**All severe sepsis elements PLUS:**

**Within 3 hours:**
5. Administer **30 mL/kg** crystalloid for hypotension or lactate ≥4 mmol/L

**Within 6 hours:**
6. Apply vasopressors (for hypotension that does not respond to initial fluid resuscitation) to maintain **MAP ≥65 mmHg**
7. Reassess volume status and tissue perfusion with:
   - Repeat focused exam (after initial fluid resuscitation), OR
   - Two of the following:
     - Measure CVP
     - Measure ScvO₂
     - Bedside cardiovascular ultrasound
     - Dynamic assessment of fluid responsiveness with passive leg raise or fluid challenge

### SEP-1 Compliance Requirements

- SEP-1 is an **"all-or-nothing" measure**: hospitals must demonstrate adherence to **ALL** bundle elements to receive credit
- Hospitals report compliance to CMS as part of the **Inpatient Quality Reporting (IQR) program**
- Failure to report can result in payment reductions

### Clinical Impact of SEP-1

**Mortality Impact:**
- Each hour of delay before treatment is associated with a **4-9% increased risk of mortality**
- Studies show mortality rate of **14.87% in bundle-compliant patients** vs. **27.69% in non-compliant patients**

**Challenges:**
- The rigid "all-or-nothing" structure has been controversial
- The Infectious Diseases Society of America has called for revisions to SEP-1
- Compliance can be difficult to achieve, particularly the 3-hour antibiotic administration in patients with unclear infection source

---

## Surviving Sepsis Campaign Hour-1 Bundle

### Background

In [April 2018, the Surviving Sepsis Campaign (SSC)](https://sccm.org/survivingsepsiscampaign/guidelines-and-resources) released an updated sepsis bundle, which combined directives previously listed in the three-hour and six-hour bundles into the **"Hour-1 Bundle"** (updated June 2018).

**Objective:** Begin resuscitation and management **immediately**, recognizing that some measures may require more than one hour to complete.

### Hour-1 Bundle Elements

**To be initiated within 1 hour of sepsis recognition:**

1. **Measure lactate level.** Remeasure if initial lactate is >2 mmol/L

2. **Obtain blood cultures before administering antibiotics**

3. **Administer broad-spectrum antibiotics**

4. **Begin rapid administration of 30 mL/kg crystalloid** for hypotension or lactate ≥4 mmol/L

5. **Apply vasopressors** if hypotensive during or after fluid resuscitation to maintain MAP ≥65 mmHg

### Clinical Context

- The Hour-1 Bundle should be viewed as a **medical emergency** requiring rapid diagnosis and immediate intervention
- The "hour-1" timeframe begins at **recognition of sepsis**, not from ED arrival
- Both sepsis and septic shock should be treated as **medical emergencies** comparable to STEMI or stroke

### Compliance Rates and Outcomes

**Compliance Data:**
- Compliance with Sepsis Bundles is associated with a **25% reduction in the risk of death**
- However, compliance rates vary significantly:
  - Some settings report only **56.3% compliance** with Hour-1 Bundle
  - Post-intervention studies show improvement from **14% to 76% compliance** with quality improvement initiatives
  - ICU-acquired sepsis shows **31.9% total bundle compliance**

**Mortality Benefits:**
- Hour-1 Bundle adherence associated with **reduction in in-hospital mortality**
- Time-to-treatment is critical: delays increase mortality risk significantly

### Controversy and Debate

The 1-hour window recommendation has been **intensively debated** and remains controversial:

**Concerns:**
- May lead to inappropriate antibiotic use if infection not clearly present
- Potential for alert fatigue and resource strain
- Difficulty achieving 1-hour timeframe in real-world settings

**Support:**
- Emphasizes urgency of sepsis treatment
- Aligns with evidence showing mortality increases with treatment delays
- Functions as a quality improvement goal moving toward ideal state

### Implementation Resources

The Surviving Sepsis Campaign provides implementation tools including:
- [Hour-1 Bundle Implementation Guide](https://sccm.org/survivingsepsiscampaign/hour-1-bundle-implementation-guide)
- Early Identification of Sepsis on Hospital Floors: Insights for Implementation
- Best practices for sustaining quality improvement programs

---

## Septic Shock Management

### Septic Shock Definition (Sepsis-3)

[Septic shock](https://pmc.ncbi.nlm.nih.gov/articles/PMC4968574/) is defined as:

> **Sepsis with persisting hypotension requiring vasopressors to maintain MAP ≥65 mmHg AND having a serum lactate level >2 mmol/L (18 mg/dL) despite adequate volume resuscitation**

With these criteria, **hospital mortality is in excess of 40%**.

### Hemodynamic Management

#### Mean Arterial Pressure (MAP) Targets

**Standard Target: MAP ≥65 mmHg**

- The [Surviving Sepsis Campaign](https://sccm.org/survivingsepsiscampaign/guidelines-and-resources/surviving-sepsis-campaign-adult-guidelines) recommends targeting **MAP ≥65 mmHg** during initial resuscitation (Grade 1C recommendation)
- This target is based on minimal MAP needed to maintain organ perfusion
- In patients with chronic hypertension, a higher MAP target may be necessary due to rightward shift of autoregulation curve

**High vs. Low MAP Targets:**
- Major multicenter trial compared MAP targets of 80-85 mmHg vs. 65-70 mmHg
- **No significant difference in 28-day mortality:** 36.6% (high-target) vs. 34.0% (low-target)
- Higher targets may benefit specific populations but increase vasopressor requirements

#### Vasopressor Therapy

**First-Line Agent: Norepinephrine**

- [Norepinephrine is positioned as the first-line vasopressor](https://pmc.ncbi.nlm.nih.gov/articles/PMC7333107/) in septic shock
- Early administration allows achieving MAP target faster
- Reduces risk of fluid overload

**Shock Control Endpoints (at 6 hours):**
- MAP ≥65 mmHg PLUS either:
  - Urine output ≥0.5 mL/kg/hour for 2 consecutive hours, OR
  - Serum lactate decreased ≥10% from baseline

### Lactate Monitoring

**Lactate as Prognostic Marker:**

- [Serum lactate >2 mmol/L](https://pmc.ncbi.nlm.nih.gov/articles/PMC4958885/) is an emerging vital sign of septic shock
- Lactate ≥4 mmol/L indicates severe hypoperfusion requiring aggressive resuscitation

**Mortality by Lactate Level:**
- Hypotension + lactate ≥4 mmol/L: **46.1% mortality**
- Hypotension alone: **36.7% mortality**
- Lactate ≥4 mmol/L alone: **30% mortality**

**Lactate Clearance:**
- Repeat lactate measurement if initial value >2 mmol/L
- Goal: ≥10% decrease from baseline
- Decreasing lactate indicates improved tissue perfusion
- Increasing MAP above 65 mmHg with norepinephrine associated with decreased blood lactate concentrations

### Fluid Resuscitation

**Initial Fluid Bolus: 30 mL/kg crystalloid**

- Administer within first 3 hours for:
  - Hypotension (SBP <90 mmHg), OR
  - Lactate ≥4 mmol/L
- Use balanced crystalloids (lactated Ringer's) or normal saline
- Reassess volume status after initial bolus to guide further fluid administration

**Dynamic Assessment:**
- Passive leg raise
- Fluid challenge with hemodynamic monitoring
- Bedside ultrasound assessment

---

## Clinical Implications and Implementation

### Integration of Triage and Sepsis Protocols

**ED Workflow Integration:**

1. **Triage (ESI):** Rapid identification of potentially septic patients
   - Sepsis patients often present as **ESI Level 2** (high-risk, should not wait)
   - May present as **ESI Level 3** if vitals stable but multiple resources needed
   - Use **Decision Point D** (vital sign review) to catch subtle sepsis presentations

2. **Sepsis Screening:** Apply qSOFA at bedside
   - qSOFA ≥2 triggers sepsis protocol activation
   - Obtain full SOFA score if ICU admission likely

3. **Bundle Implementation:** Activate Hour-1 Bundle or SEP-1 Bundle
   - Time zero = recognition of sepsis
   - Simultaneous initiation of all bundle elements

### Quality Improvement Considerations

**Challenges:**
- Balancing speed with diagnostic accuracy
- Avoiding inappropriate antibiotic use
- Resource allocation in crowded EDs
- Documentation requirements for compliance

**Success Factors:**
- Automated sepsis alerts in EHR systems
- Standardized order sets
- Multidisciplinary team training
- Real-time compliance monitoring
- Feedback loops to clinicians

### Special Populations

**Pediatric Patients:**
- Modified SIRS criteria (mandatory abnormal temperature or WBC)
- Age-appropriate vital signs
- Use PAT with ESI for triage

**Immunocompromised Patients:**
- May not mount typical SIRS response
- Lower threshold for sepsis suspicion
- Consider broader antibiotic coverage

**Elderly Patients:**
- Atypical presentations common
- May have chronic organ dysfunction affecting SOFA baseline
- Higher mortality risk

---

## References

### Emergency Severity Index (ESI)

1. Agency for Healthcare Research and Quality. [Emergency Severity Index (ESI): A Triage Tool for Emergency Departments](https://www.ahrq.gov/patient-safety/settings/emergency-dept/esi.html)

2. Wikipedia. [Emergency Severity Index](https://en.wikipedia.org/wiki/Emergency_Severity_Index)

3. AHRQ. [Emergency Severity Index Handbook, Version 5](https://media.emscimprovement.center/documents/Emergency_Severity_Index_Handbook.pdf)

4. Nursing CE Central. [Emergency Severity Index](https://nursingcecentral.com/lessons/emergency-severity-index/)

5. BMC Medical Informatics and Decision Making. [Predicting Emergency Severity Index (ESI) level, hospital admission, and admitting ward](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-025-02941-9)

### Sepsis-3 SOFA and qSOFA

6. PMC. [The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4968574/)

7. Wikipedia. [SOFA score](https://en.wikipedia.org/wiki/SOFA_score)

8. MDCalc. [qSOFA (Quick SOFA) Score for Sepsis](https://www.mdcalc.com/calc/2654/qsofa-quick-sofa-score-sepsis)

9. MDCalc. [Sequential Organ Failure Assessment (SOFA) Score](https://www.mdcalc.com/calc/691/sequential-organ-failure-assessment-sofa-score)

10. ASPR TRACIE. [SOFA Score: What it is and How to Use it in Triage](https://files.asprtracie.hhs.gov/documents/aspr-tracie-sofa-score-fact-sheet.pdf)

11. PMC. [Comparison of SOFA Score, SIRS, qSOFA, and qSOFA + L Criteria in the Diagnosis and Prognosis of Sepsis](https://pmc.ncbi.nlm.nih.gov/articles/PMC7929579/)

12. Critical Care. [The SOFA score—development, utility and challenges of accurate assessment in clinical trials](https://ccforum.biomedcentral.com/articles/10.1186/s13054-019-2663-7)

### SIRS Criteria

13. NCBI Bookshelf. [Systemic Inflammatory Response Syndrome - StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK547669/)

14. American Academy of Family Physicians. [Sepsis: Diagnosis and Management](https://www.aafp.org/pubs/afp/issues/2020/0401/p409.html)

15. MDCalc. [SIRS, Sepsis, and Septic Shock Criteria](https://www.mdcalc.com/calc/1096/sirs-sepsis-septic-shock-criteria)

16. Cleveland Clinic. [SIRS (Systemic Inflammatory Response Syndrome): What It Is](https://my.clevelandclinic.org/health/diseases/25132-sirs-systemic-inflammatory-response-syndrome)

17. PMC. [SIRS, qSOFA and new sepsis definition](https://pmc.ncbi.nlm.nih.gov/articles/PMC5418298/)

### CMS SEP-1

18. PMC. [Improving Compliance with the CMS SEP-1 Sepsis Bundle at a Community-Based Teaching Hospital Emergency Department](https://pmc.ncbi.nlm.nih.gov/articles/PMC9448659/)

19. Sepsis Alliance. [The SEP-1 Measure: What Is It, and How Does It Impact Sepsis Patients & Their Families?](https://www.sepsis.org/news/the-sep-1-measure-what-is-it-and-how-does-it-impact-sepsis-patients-their-families/)

20. PMC. [Compliance with SEP-1 guidelines is associated with improved outcomes for septic shock but not for severe sepsis](https://pmc.ncbi.nlm.nih.gov/articles/PMC9924005/)

21. Oxford Academic. [Infectious Diseases Society of America Position Paper: Recommended Revisions to the National Severe Sepsis and Septic Shock Early Management Bundle (SEP-1) Sepsis Quality Measure](https://academic.oup.com/cid/article/72/4/541/5831166)

22. HealthLeaders Media. [Expert: CMS Takes Significant Step to Improve Sepsis Care in 2024 IPPS Rule](https://www.healthleadersmedia.com/clinical-care/expert-cms-takes-significant-step-improve-sepsis-care-2024-ipps-rule)

23. Quality Reporting Center. [Severe Sepsis and Septic Shock: Management Bundle (Composite Measure)](https://www.qualityreportingcenter.com/globalassets/iqr-2023-events/iqr32423/march2023_sep_1_npc_final508.pdf)

### Surviving Sepsis Campaign Hour-1 Bundle

24. Society of Critical Care Medicine. [Guidelines and Bundles](https://www.sccm.org/survivingsepsiscampaign/guidelines-and-resources)

25. SCCM. [Surviving Sepsis Campaign 2021 Adult Guidelines](https://sccm.org/survivingsepsiscampaign/guidelines-and-resources/surviving-sepsis-campaign-adult-guidelines)

26. SCCM. [Hour-1 Bundle Implementation Guide](https://sccm.org/survivingsepsiscampaign/hour-1-bundle-implementation-guide)

27. American Nurse. [Surviving Sepsis Campaign hour-1 bundle](https://www.myamericannurse.com/surviving-sepsis-campaign-hour-1-bundle/)

28. PLOS One. [Hour-1 bundle adherence was associated with reduction of in-hospital mortality among patients with sepsis in Japan](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0263936)

29. PMC. [The 28-Day Mortality Outcome of the Complete Hour-1 Sepsis Bundle in the Emergency Department](https://pmc.ncbi.nlm.nih.gov/articles/PMC8579988/)

30. PMC. [Challenging the One-hour Sepsis Bundle](https://pmc.ncbi.nlm.nih.gov/articles/PMC6404723/)

31. medRxiv. [Compliance with the Surviving Sepsis Campaign Hour-1 Bundle and impact on patient outcomes in a resource-limited setting](https://www.medrxiv.org/content/10.1101/2025.08.06.25333121v1.full)

### Septic Shock Management

32. PMC. [New clinical criteria for septic shock: serum lactate level as new emerging vital sign](https://pmc.ncbi.nlm.nih.gov/articles/PMC4958885/)

33. PMC. [Vasopressors in septic shock: which, when, and how much?](https://pmc.ncbi.nlm.nih.gov/articles/PMC7333107/)

34. New England Journal of Medicine. [High versus Low Blood-Pressure Target in Patients with Septic Shock](https://www.nejm.org/doi/full/10.1056/nejmoa1312173)

35. PMC. [Outcome of patients with septic shock and high-dose vasopressor therapy](https://pmc.ncbi.nlm.nih.gov/articles/PMC5397393/)

---

## Document Metadata

**Document Title:** Emergency Department Triage Systems and Sepsis Protocols
**Version:** 1.0
**Date Created:** 2025-11-30
**Purpose:** Comprehensive reference for ED triage and sepsis management protocols
**Target Audience:** Healthcare professionals, quality improvement teams, clinical decision support system developers
**Keywords:** Emergency Severity Index, ESI, Sepsis-3, SOFA, qSOFA, CMS SEP-1, Hour-1 Bundle, septic shock, triage, emergency department

---

*This document is intended for educational and informational purposes. Clinical decisions should be made in consultation with appropriate medical professionals and institutional protocols.*
