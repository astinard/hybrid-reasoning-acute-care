# IRB AND REGULATORY PATHWAY
## Hybrid Reasoning for Acute Care: Clinical Research Compliance Strategy

**Document Purpose:** This section demonstrates regulatory readiness and provides actionable guidance for executing clinical research at UCF across all phases of development.

**Regulatory Philosophy:** Progressive validation approach that begins with zero-IRB public data research, advances through expedited retrospective validation, and culminates in prospective clinical deployment under appropriate regulatory oversight.

---

## PHASE 1: PUBLIC DATA RESEARCH (NO IRB REQUIRED)

### 1.1 Regulatory Status: Exempt from IRB Review

**Rationale for Exemption:**
- MIMIC-III/IV/ED and eICU datasets are **fully de-identified** under HIPAA Safe Harbor method (45 CFR 164.514(b)(2))
- Data has been **irreversibly anonymized** by source institutions (Beth Israel Deaconess Medical Center, Philips Healthcare)
- UCF researchers receive **secondary data** with no ability to re-identify subjects
- PhysioNet serves as **honest broker** for data distribution

**Regulatory Determination:**
Per 45 CFR 46.104(d)(4), research involving only **de-identified private information** or **specimens** is NOT considered human subjects research and does not require IRB review.

### 1.2 Data Use Agreement Requirements

**PhysioNet Credentialed Access:**
- Individual researcher CITI training (Human Research Protections)
- Signed Data Use Agreement (DUA) for each dataset
- Institutional acknowledgment (UCF signatory authority)
- No cost for academic research use

**DUA Key Terms:**
```
- Purpose limitation: Research use only
- No re-identification attempts
- No data redistribution
- Attribution requirements in publications
- Annual usage reports (optional for most datasets)
- 1-year renewal cycle (automatic for continued research)
```

**Timeline for Access:**
- CITI training completion: 4-8 hours
- PhysioNet application: 1-2 weeks review
- Total time to data access: 2-3 weeks

### 1.3 Research Activities Permitted Without IRB

**Algorithm Development:**
- Temporal knowledge graph framework design and implementation
- Neuro-symbolic reasoning model training
- Multi-modal fusion architecture development
- Explainability method development

**Validation Studies:**
- MIMIC-III/IV benchmarking against published baselines
- Cross-dataset validation (MIMIC → eICU)
- Ablation studies and hyperparameter optimization
- Comparison with state-of-the-art methods

**Publications Permitted:**
- All major CS venues (NeurIPS, ICML, KDD, AAAI)
- Clinical informatics venues (JAMIA, AMIA)
- High-impact journals (Nature Digital Medicine, npj Digital Medicine)

**Limitation:**
Cannot make claims about **prospective clinical effectiveness** or **real-world performance** without additional validation (Phase 2-3).

### 1.4 Deliverables from Phase 1 (12-18 months)

| Deliverable | Target Venue | Regulatory Status |
|-------------|--------------|-------------------|
| Temporal KG framework paper | KDD/AAAI | No IRB required |
| Neuro-symbolic reasoning paper | NeurIPS/ICML | No IRB required |
| MIMIC-IV-ED benchmark paper | JAMIA | No IRB required |
| Multi-modal fusion paper | Nature Digital Medicine | No IRB required |
| Open-source software release | GitHub/Zenodo | No IRB required |

---

## PHASE 2: RETROSPECTIVE CLINICAL VALIDATION (EXPEDITED IRB)

### 2.1 IRB Category and Regulatory Pathway

**IRB Review Type:** Expedited Review under 45 CFR 46.110

**Applicable Expedited Category:**
- Category 5: "Research involving materials (data, documents, records, or specimens) that have been collected, or will be collected solely for nonresearch purposes (such as medical treatment or diagnosis)"

**Risk Classification:** Minimal Risk
- No intervention with patients
- No prospective data collection
- Analysis of existing medical records
- Waiver of informed consent likely

### 2.2 UCF IRB Application Template Language

**Study Title:**
"Retrospective Validation of Hybrid Reasoning AI for Emergency Department Risk Stratification"

**Study Purpose:**
To validate the performance of a hybrid temporal knowledge graph and neuro-symbolic AI system for predicting clinical outcomes (sepsis, mortality, ICU admission) in emergency department patients using retrospective medical record review.

**Study Design:**
- **Type:** Retrospective chart review
- **Setting:** Orlando Health/AdventHealth Emergency Departments
- **Population:** Adult ED patients (age ≥18) presenting between [date range]
- **Sample Size:** 5,000-10,000 patient encounters
- **Data Elements:** Demographics, vital signs, laboratory results, medications, diagnoses, outcomes
- **Analysis:** Comparison of AI predictions against actual clinical outcomes

**Inclusion Criteria:**
- Adult patients (≥18 years)
- Complete ED visit record with discharge disposition
- Minimum 4-hour ED length of stay (excludes fast-track patients)

**Exclusion Criteria:**
- Pediatric patients (<18 years)
- Incomplete medical records
- Patients who left against medical advice without outcome data

### 2.3 HIPAA Compliance Strategy

**Option A: Limited Dataset (Preferred for Expedited Approval)**

Remove 16 direct identifiers per 45 CFR 164.514(e)(2):
```
REMOVE:
- Names (patient, relatives, employers)
- Geographic subdivisions smaller than state (use ZIP3 only)
- Dates (except year) - use time offsets from index date
- Telephone/fax numbers
- Email addresses
- Social Security numbers
- Medical record numbers
- Account numbers
- Certificate/license numbers
- Vehicle identifiers
- Device identifiers/serial numbers
- URLs
- IP addresses
- Biometric identifiers
- Full-face photographs
- Any other unique identifying number/characteristic

RETAIN (for research purposes):
- Town/city, state, ZIP3
- Dates offset from index event
- Ages (including >89 years if necessary)
```

**Justification for Limited Dataset:**
- Temporal analysis requires relative date preservation (offset from ED arrival)
- Geographic variation analysis requires ZIP3-level data
- Clinical validation requires precise age (not grouped)

**Data Security Plan:**
- Encrypted storage (AES-256) on UCF-approved research servers
- Access control: Named researchers only, 2FA authentication
- No data on personal laptops or unencrypted drives
- De-identification performed by hospital honest broker before transmission to UCF
- Data destruction after study completion (or retention per hospital agreement)

**Option B: Fully De-identified Dataset (Faster, but less flexible)**

If hospital provides fully HIPAA-compliant de-identified data:
- IRB may determine this is NOT human subjects research
- May qualify for exempt status (faster approval)
- Trade-off: Less granular temporal/geographic data

### 2.4 Waiver of Informed Consent

**Regulatory Basis:** 45 CFR 46.116(d) - Waiver of Informed Consent

**Criteria to Satisfy:**
1. **Minimal risk:** Retrospective chart review with limited dataset poses minimal risk beyond privacy breach, which is mitigated by data security measures
2. **No adverse impact:** Waiver will not adversely affect rights and welfare of subjects (no intervention, no contact)
3. **Impracticable:** Research could not practicably be carried out without waiver (locating 10,000 ED patients from historical records is not feasible)
4. **Privacy protections:** Limited dataset, data security plan, honest broker de-identification
5. **Debriefing (N/A):** Subjects will not be provided with pertinent information after participation (no ongoing relationship)

**Template Waiver Language:**
> We request a waiver of informed consent for this retrospective chart review study. Obtaining consent from thousands of patients who presented to the ED over multiple years is not practicable, as many cannot be located and contact attempts would impose greater privacy risk than the research itself. The study poses minimal risk as it involves only analysis of de-identified data with robust security protections. The waiver will not adversely affect patient rights as there is no intervention or ongoing contact.

### 2.5 Timeline for UCF IRB Approval

**Expedited Review Timeline:**

| Stage | Duration | Notes |
|-------|----------|-------|
| **Protocol Development** | 2-4 weeks | Template provided above |
| **IRB Submission** | 1 day | UCF eIRB online portal |
| **Administrative Review** | 3-5 business days | Completeness check |
| **Expedited Review** | 10-15 business days | Single reviewer |
| **Contingencies** | 1-2 weeks | If revisions requested |
| **Approval to Data Access** | 1-2 weeks | Hospital DUA, BAA execution |
| **Total Timeline** | **8-12 weeks** | Concurrent BAA negotiation recommended |

**Optimization Strategies:**
- Submit IRB and BAA applications concurrently
- Use template language provided above
- Pre-consultation with UCF IRB (available for complex studies)
- Designate experienced co-investigator with prior IRB approvals

### 2.6 Hospital Partnership Requirements for Phase 2

**Business Associate Agreement (BAA) - Required under HIPAA**

**BAA Key Elements:**
```
1. Permitted Uses and Disclosures:
   - UCF is Business Associate for limited purpose of research study
   - Hospital is Covered Entity providing Limited Dataset
   - Use restricted to study protocol approved by IRBs

2. Safeguards:
   - UCF maintains administrative, physical, technical safeguards (45 CFR 164.308-312)
   - Encrypted storage and transmission
   - Access logs and audit trails
   - Incident response plan

3. Subcontractors:
   - Cloud computing providers (AWS, Azure) must have BAAs with UCF
   - No offshore data storage or processing

4. Breach Notification:
   - UCF reports any suspected breach to Hospital within 24 hours
   - Hospital reports to HHS if required

5. Data Return/Destruction:
   - Data returned or destroyed within 90 days of study completion
   - Certificate of destruction provided

6. Term and Termination:
   - Initial term: 3 years
   - Renewal by mutual agreement
   - Termination for breach with 30-day cure period
```

**Data Use Agreement (DUA) - Research-Specific Terms**

**DUA Key Elements:**
```
1. Purpose and Scope:
   - Specific to IRB-approved protocol [UCF-IRB-XXXX]
   - Amendment requires IRB modification approval

2. Data Elements:
   - Itemized list matching IRB protocol
   - Limited dataset with 16 identifiers removed
   - Honest broker de-identification by Hospital

3. No Re-identification:
   - UCF will not attempt to identify subjects
   - Will not contact patients
   - Will not link to other datasets without IRB approval

4. Publication and IP:
   - Hospital acknowledgment required in publications
   - No proprietary claims to data
   - Joint publication review (10-day turnaround typical)

5. Compliance:
   - Both institutions' IRB approval required before data transfer
   - Continuing review compliance (annual renewals)
   - Study closure notification

6. Liability and Indemnification:
   - Mutual indemnification for respective breaches
   - UCF maintains research liability insurance
   - Hospital not liable for data quality issues if disclosed
```

### 2.7 Orlando Health Partnership Pathway

**Current UCF-Orlando Health Relationship:**
- Existing affiliation through UCF College of Medicine
- Research collaboration infrastructure in place
- Streamlined IRB process (Orlando Health recognizes UCF IRB via reliance agreement)

**Recommended Approach:**
1. **Initial Meeting:** Clinical informatics director + ED medical director
2. **Concept Approval:** Present 1-page research summary to Research Institute
3. **Dual IRB Submission:**
   - Submit to UCF IRB first (lead IRB)
   - Orlando Health relies on UCF IRB review (via IRB Authorization Agreement)
4. **BAA/DUA Negotiation:** Concurrent with IRB review (8-12 weeks)
5. **Data Transfer:** Hospital honest broker creates limited dataset

**Timeline:** 3-4 months from first meeting to data access

### 2.8 AdventHealth Partnership Pathway

**Current UCF-AdventHealth Relationship:**
- Clinical rotation site for UCF medical students
- Research collaboration framework exists
- Graduate medical education partnerships

**Recommended Approach:**
1. **Entry Point:** AdventHealth Research Institute + CREATION Health Research
2. **Proposal Review:** Research Institute Scientific Review Committee (meets monthly)
3. **Dual IRB:**
   - AdventHealth IRB (Advarra commercial IRB) OR
   - UCF IRB as single IRB of record (recommended)
4. **Contract Negotiation:** DUA template available, typically faster than Orlando Health

**Timeline:** 3-4 months (similar to Orlando Health)

**Multi-Site Advantage:**
- Validates generalizability across health systems
- Different EHR systems (Orlando Health: Epic; AdventHealth: Cerner/Oracle)
- Strengthens publication claims

### 2.9 Deliverables from Phase 2 (12-18 months)

| Deliverable | Clinical Partner | Regulatory Basis |
|-------------|------------------|------------------|
| Single-site retrospective validation | Orlando Health | Expedited IRB |
| Multi-site external validation | AdventHealth | Expedited IRB (or reliance) |
| Subpopulation analysis (age, race, sex) | Combined dataset | Approved in protocol |
| Algorithm bias audit | Combined dataset | Approved in protocol |
| Clinical venue publication | JAMIA, JAMA Network Open | IRB-approved research |

---

## PHASE 3: PROSPECTIVE CLINICIAN STUDIES (FULL IRB)

### 3.1 Study Design: Clinician Evaluation of AI Explanations

**Study Type:** Human subjects research - healthcare providers

**IRB Review Type:** Full Board Review (not eligible for expedited)

**Rationale for Full Board:**
- Novel intervention (AI decision support exposure)
- Evaluation of clinical decision-making processes
- Potential impact on patient care (indirect)

### 3.2 Research Questions for Clinician Studies

**Primary Research Questions:**
1. Do neuro-symbolic AI explanations improve clinician understanding of model predictions compared to feature importance methods?
2. What explanation modalities (reasoning chains, counterfactuals, temporal graphs) are most effective for ED physician decision-making?
3. How does AI explanation transparency affect clinician trust and willingness to act on recommendations?

**Study Design:**
- **Type:** Prospective, within-subjects experimental design
- **Participants:** Emergency medicine physicians, PA/NPs (N=30-50)
- **Setting:** Simulated clinical cases (not real patients - reduced risk)
- **Intervention:** Exposure to AI predictions with different explanation types
- **Measurements:**
  - Explanation comprehension (quiz scores)
  - Decision concordance (with vs without AI)
  - Trust scales (validated instruments)
  - Cognitive load (NASA-TLX)
  - Time to decision
- **Control:** Cases without AI assistance (baseline)

### 3.3 Informed Consent Requirements

**Subjects:** Healthcare providers (not patients)

**Consent Process:**
- Written informed consent required
- No waiver applicable (provider participation is voluntary)
- Low risk to providers (no patient care involvement)

**Consent Form Elements (45 CFR 46.116):**

```
REQUIRED ELEMENTS:

1. Research statement:
   "You are being asked to participate in a research study evaluating
   explanation methods for artificial intelligence in emergency medicine."

2. Purpose:
   "This study aims to determine which AI explanation approaches are most
   effective for emergency physicians when assessing patient risk."

3. Duration:
   "Participation will require approximately 90 minutes in a single session."

4. Procedures:
   "You will review 20 simulated ED patient cases and make clinical assessments.
   For each case, you will see AI predictions with different types of
   explanations. You will answer questions about your understanding and
   decision-making process."

5. Risks:
   "Risks are minimal. You may experience mild fatigue from the 90-minute
   session. There are no real patients involved, so no risk to patient care."

6. Benefits:
   "You may gain insight into AI decision support tools. There is no direct
   compensation, but light refreshments will be provided."

7. Confidentiality:
   "Your responses will be de-identified and reported only in aggregate. Your
   name will not appear in any publications. Data will be stored securely and
   destroyed after 5 years per UCF policy."

8. Voluntary nature:
   "Participation is completely voluntary. You may withdraw at any time
   without penalty."

9. Contact information:
   Principal Investigator: [Name, Phone, Email]
   UCF IRB: 407-823-2901, irb@ucf.edu

10. Consent statement:
    "By signing below, you indicate that you have read this consent form,
    your questions have been answered, and you agree to participate."
```

### 3.4 Safety Monitoring Plan

**Risk Level:** Minimal risk study (no patient involvement, simulated cases only)

**Monitoring Requirements:**
- No formal Data Safety Monitoring Board (DSMB) required for minimal risk
- Annual continuing review by UCF IRB
- Adverse event reporting (if any participant experiences distress)

**Participant Safety Protections:**
```
1. Screen for conflicts of interest:
   - No participation by clinicians employed by researchers
   - No evaluation of own clinical performance

2. Fatigue mitigation:
   - Session limited to 90 minutes
   - Breaks offered every 30 minutes
   - Option to complete in two sessions

3. Psychological safety:
   - No evaluation of clinical competence
   - Performance data not shared with employers
   - No "correct answer" framing (avoid performance anxiety)

4. Withdrawal process:
   - Participants may withdraw at any time
   - Partial data may be retained with permission
```

### 3.5 Data Management Plan

**Data Collection:**
- Electronic survey platform (Qualtrics, REDCap)
- Secure UCF-licensed platform with HIPAA compliance
- No protected health information (providers are subjects, not patients)

**De-identification:**
- Participant assigned random ID at enrollment
- Consent forms stored separately from data
- No names in analysis dataset

**Data Security:**
```
1. Storage:
   - Encrypted database on UCF research servers
   - Access restricted to named investigators
   - Two-factor authentication required

2. Transmission:
   - HTTPS for web-based surveys
   - Encrypted email for file transfers
   - No data on personal devices

3. Retention:
   - Raw data: 5 years post-publication (UCF policy)
   - De-identified data: May be shared in repository (with consent)
   - Consent forms: 3 years post-study closure (regulatory requirement)

4. Destruction:
   - Secure deletion of electronic files
   - Certificate of destruction documented
```

**Data Sharing Plan (for publication):**
- De-identified data may be deposited in repository (OSF, Zenodo)
- Consent form includes optional checkbox for data sharing
- Metadata will include: study design, measures, codebook

### 3.6 Timeline for Full IRB Approval

**Full Board Review Timeline:**

| Stage | Duration | Notes |
|-------|----------|-------|
| **Protocol Development** | 4-6 weeks | Detailed consent forms, study instruments |
| **IRB Submission** | 1 day | UCF eIRB portal |
| **Administrative Review** | 5-7 business days | Completeness check |
| **Full Board Review** | 4-6 weeks | Board meets monthly |
| **Contingencies** | 2-4 weeks | Revisions likely for first submission |
| **Approval to Enrollment** | 1 week | Consent form finalization |
| **Total Timeline** | **12-16 weeks** | Longer than expedited review |

**Optimization Strategies:**
- Submit 6-8 weeks before target board meeting
- Request pre-review consultation with IRB staff
- Use validated instruments (scales, surveys) where possible
- Provide sample consent form with submission

### 3.7 Recruitment Strategy

**Target Population:**
- Emergency medicine attending physicians
- Emergency medicine residents (PGY-2 and above)
- Advanced practice providers (PA, NP) with ED experience

**Recruitment Methods:**
```
1. Professional networks:
   - Florida College of Emergency Physicians (FCEP)
   - UCF College of Medicine alumni network
   - Orlando Health/AdventHealth ED medical directors

2. Conference presentations:
   - Florida ACEP chapter meetings
   - UCF COM Grand Rounds
   - Regional emergency medicine symposia

3. Direct outreach:
   - Email invitation to ED faculty at partner hospitals
   - Flyers in ED physician lounges (with hospital approval)
   - Snowball sampling (participant referrals)

4. Incentives (if budget allows):
   - $50-100 gift card for 90-minute session
   - CME credit (if study approved for CME)
   - Contribution to departmental education fund
```

**Sample Size Justification:**
- N=30-50 clinicians
- Sufficient for within-subjects design (each participant sees all conditions)
- Power analysis: 80% power to detect medium effect size (d=0.5) with paired t-test at α=0.05 requires N=34

### 3.8 Deliverables from Phase 3 (12-18 months)

| Deliverable | Target Venue | Regulatory Basis |
|-------------|--------------|-------------------|
| Clinician explanation study | CHI, CSCW, AMIA | Full IRB approval |
| Trust and adoption analysis | JAMIA, J Patient Safety | Full IRB approval |
| XAI design guidelines | IEEE Intelligent Systems | Full IRB approval |
| Cognitive load study | Human Factors | Full IRB approval |

---

## PHASE 4: CLINICAL DECISION SUPPORT DEPLOYMENT (FDA/REGULATORY)

### 4.1 FDA Software as a Medical Device (SaMD) Framework

**FDA Categorization:** Clinical Decision Support Software

**Regulatory Question:** Does this software require FDA premarket review?

**Answer:** Likely EXEMPT under 21st Century Cures Act Section 3060

### 4.2 21st Century Cures Act CDS Exemptions

**Statutory Exemption (21 U.S.C. 360j(o)):**

Software functions are NOT devices if they meet **ALL FOUR criteria**:

```
1. NOT intended to acquire, process, or analyze medical images or signals:
   ✓ MEETS: Our system processes structured EHR data (vitals, labs, text notes)
   ✗ Does not process ECG waveforms, radiology images as primary input

2. For displaying, analyzing, or printing medical information:
   ✓ MEETS: System displays risk predictions and explanations to clinicians

3. For supporting or providing recommendations about prevention, diagnosis, or
   treatment:
   ✓ MEETS: System provides risk stratification recommendations (sepsis,
   deterioration, disposition)

4. For enabling clinician to independently review basis for recommendations:
   ✓ MEETS: Neuro-symbolic explanations show reasoning chains, temporal
   relationships, and allow clinician override
```

**Critical Requirement for Exemption:**
The software must allow clinicians to **independently review the basis** for the recommendations so they are **not intended to replace clinical judgment**.

**How Our System Satisfies This:**
- Explainable AI provides transparent reasoning chains
- Clinicians see input data, knowledge graph relationships, and logic rules
- System recommends but does not automate clinical actions
- Clinicians must actively choose to accept or reject recommendations

### 4.3 FDA Guidance: Clinical Decision Support Software (2022)

**FDA's Four Categories of CDS:**

| Category | FDA Review? | Our System Classification |
|----------|-------------|---------------------------|
| **Category 1:** Simple tools (calculators, alerts based on single data point) | Exempt | Not applicable - we're more sophisticated |
| **Category 2:** Analysis of multiple data sources, not time-critical, clinician review | **Exempt** | **YES - This is our category** |
| **Category 3:** Time-critical or complex diagnostic/treatment | May require review | Not applicable - ED decisions allow time for review |
| **Category 4:** Autonomous or minimal human oversight | Requires review | Not applicable - clinician must review |

**Determination:** Our system falls under **Category 2 (Exempt)**

**Rationale:**
- Analyzes multiple EHR data sources (vitals, labs, medications, notes)
- Provides recommendations for sepsis risk, deterioration, disposition
- NOT time-critical: ED physicians have time to review (minutes to hours, not seconds)
- Transparent basis: Neuro-symbolic explanations enable independent review
- Does not automate actions: Clinician must order tests, admit patient, etc.

### 4.4 What Would Trigger FDA Review?

**Scenarios That WOULD Require FDA Premarket Submission:**

```
1. Locked Algorithm:
   - If we prevented clinicians from seeing reasoning basis
   - If system auto-populated orders without review
   - If explanations were not interpretable

2. Autonomous Actions:
   - If system automatically administered medications
   - If system directly controlled medical devices (ventilators, IV pumps)
   - If system bypassed clinician decision-making

3. Critical Time Constraints:
   - If system made split-second decisions where review is impractical
   - Example: Automated defibrillator shock decisions

4. Primary Diagnostic Function:
   - If system analyzed radiology images and provided diagnosis without
     radiologist review
   - If system interpreted ECG waveforms as primary diagnostic tool

5. Replacement of Clinical Judgment:
   - If system was marketed as "replacing" physician decision-making
   - If system prevented clinician override
```

**How to MAINTAIN Exemption Status:**

```
1. Marketing and Labeling:
   - ALWAYS market as "clinical decision SUPPORT" not "clinical decision MAKING"
   - Emphasize clinician review and independent judgment
   - Never claim to "replace" physicians

2. User Interface Design:
   - Always show basis for recommendations
   - Allow easy override/dismissal
   - Require explicit clinician acceptance of recommendations
   - Log all overrides for quality assurance

3. Documentation:
   - Maintain records showing clinician review and decision-making
   - Document that system does not automate clinical actions
   - Publish validation studies showing clinician oversight

4. Training:
   - Train clinicians that system is advisory only
   - Emphasize professional responsibility for final decisions
   - Document training completion
```

### 4.5 FDA Quality System Regulation (QSR) Considerations

**Even if EXEMPT from premarket review, good practice follows QSR principles:**

**Design Controls (21 CFR 820.30):**
```
1. Design Input:
   - Clinical requirements documented (Phase 3 clinician studies)
   - Performance specifications defined (sensitivity, specificity, AUC targets)
   - Regulatory requirements identified (HIPAA, patient safety)

2. Design Output:
   - Software requirements specification
   - Architecture documentation
   - Interface specifications (HL7 FHIR for EHR integration)

3. Design Verification:
   - MIMIC-IV validation (Phase 1)
   - Retrospective clinical validation (Phase 2)
   - Unit testing, integration testing

4. Design Validation:
   - Prospective clinician studies (Phase 3)
   - Simulation studies with real clinicians
   - User acceptance testing in clinical environment

5. Design Review:
   - Regular reviews by interdisciplinary team (CS + Medicine)
   - Clinical advisory board review
   - Documentation of decisions and risk assessments
```

**Risk Management (ISO 14971):**
```
1. Hazard Analysis:
   - False positive: Unnecessary testing, patient anxiety
   - False negative: Missed diagnosis, delayed treatment
   - Software failure: System unavailable, data loss
   - Misinterpretation: Clinician misunderstands explanation

2. Risk Mitigation:
   - Conservative thresholds (high sensitivity over specificity)
   - Graceful degradation (system provides explanation or fails safe)
   - Redundant systems (backup clinical judgment)
   - Training and clear labeling

3. Post-Market Surveillance:
   - Monitor accuracy in real-world deployment
   - Track override rates (high override may indicate poor performance)
   - Adverse event reporting (if system contributes to patient harm)
```

### 4.6 Post-Deployment Monitoring Plan

**Regulatory Requirement:** Even exempt CDS should have quality assurance

**Monitoring Metrics:**

| Metric | Target | Action Threshold |
|--------|--------|------------------|
| **Performance** | AUROC >0.85 | <0.80 triggers retraining |
| **Calibration** | Brier score <0.15 | >0.20 triggers recalibration |
| **Override Rate** | 10-30% | >50% suggests poor utility |
| **Alert Fatigue** | <5 alerts per patient | >10 triggers threshold adjustment |
| **Time to Review** | <2 minutes median | >5 minutes suggests poor UX |
| **User Satisfaction** | >4/5 on Likert scale | <3/5 triggers redesign |

**Adverse Event Reporting:**
```
1. Definition of Adverse Event:
   - Patient harm potentially attributable to system failure or misuse
   - Serious system malfunction (crash, data corruption)
   - Privacy breach or security incident

2. Reporting Process:
   - Clinician reports event to hospital quality/safety team
   - Hospital quality team investigates
   - If device-related, report to FDA MedWatch (even if exempt, good practice)
   - Research team notified for root cause analysis

3. Response:
   - Immediate: Disable system if safety threat identified
   - Short-term: Issue software patch or configuration change
   - Long-term: Retrain model, update clinical guidelines
```

**Algorithm Update Governance:**
```
1. Minor Updates (no FDA submission needed):
   - Bug fixes
   - UI improvements
   - Performance optimizations (no logic changes)
   - Database updates (new drug interactions)

2. Major Updates (may require re-validation):
   - New prediction targets (e.g., add cardiac arrest prediction)
   - Architecture changes (new ML model)
   - New data sources (imaging, genomics)
   - Changes to decision thresholds that affect clinical recommendations

3. Update Approval Process:
   - Clinical advisory board review
   - Regression testing on validation datasets
   - Pilot deployment with monitoring
   - Full deployment after 30-day observation period
```

### 4.7 Hospital Deployment Requirements

**Beyond FDA: Institutional Requirements for Clinical Deployment**

**Hospital IT Security and Privacy:**
```
1. HIPAA Security Rule Compliance:
   - Encryption at rest and in transit (AES-256, TLS 1.3)
   - Access controls (role-based, least privilege)
   - Audit logs (all data access logged and monitored)
   - Disaster recovery (backups, redundancy)

2. Hospital Information Security Assessment:
   - Penetration testing by hospital IT security
   - Vulnerability scanning (quarterly)
   - Security risk assessment (annual)

3. Integration with Hospital Infrastructure:
   - HL7 FHIR interface for EHR integration
   - Single sign-on (SSO) using hospital Active Directory
   - Network segmentation (isolated research subnet)
```

**Clinical Governance:**
```
1. Medical Staff Approval:
   - Presentation to Medical Executive Committee
   - Approval by Department of Emergency Medicine
   - Inclusion in medical staff bylaws (if required)

2. Quality and Safety Review:
   - Patient Safety Committee review
   - Quality Improvement Committee oversight
   - Ongoing monitoring plan approved

3. Nursing and Allied Health Buy-In:
   - Nursing leadership approval
   - Pharmacy review (if medication recommendations)
   - Case management review (if disposition recommendations)
```

**Training and Competency:**
```
1. Mandatory Training for All Users:
   - System overview and intended use (30 min)
   - Interpretation of predictions and explanations (30 min)
   - Override procedures and documentation (15 min)
   - Limitations and known failure modes (15 min)
   - Total: 90 minutes initial training

2. Competency Assessment:
   - Post-training quiz (80% pass threshold)
   - Observed use with supervisor for first 10 cases
   - Annual refresher training

3. Documentation:
   - Training completion tracked in hospital LMS
   - Competency documentation in personnel file
```

### 4.8 Liability and Insurance Considerations

**Who is Liable if AI Recommendation is Wrong?**

**Legal Framework:**
- **Clinician retains professional liability** for patient care decisions
- **Institution (hospital) may be liable** for negligent credentialing of technology
- **UCF/researchers may be liable** for design defects or failure to warn

**Mitigation Strategies:**

```
1. Clear Labeling and Warnings:
   "This system is intended to support, not replace, clinical judgment.
   Clinicians are responsible for all clinical decisions."

2. Professional Liability Insurance:
   - UCF maintains research liability insurance
   - Clinical investigators maintain malpractice insurance
   - Hospital maintains institutional insurance

3. Informed Consent (if research continues during deployment):
   - Patients informed that AI is being used in their care
   - Opportunity to opt out (with standard care maintained)

4. Clinical Decision Documentation:
   - Clinician documents review of AI recommendation
   - Documents reasons for agreement or override
   - Creates auditable trail of decision-making

5. Limitation of Liability Clauses in Agreements:
   - Hospital-UCF agreement limits liability to actual damages
   - Mutual indemnification for respective actions
   - No consequential damages
```

---

## PHASE 5: HOSPITAL PARTNERSHIP REQUIREMENTS

### 5.1 Business Associate Agreement (BAA) Template Elements

**HIPAA Business Associate Agreement for Clinical AI Deployment**

**Parties:**
- Covered Entity: [Hospital Name]
- Business Associate: University of Central Florida Board of Trustees

**Key Provisions:**

```
ARTICLE 1: DEFINITIONS
- Business Associate: UCF researchers and systems processing PHI
- Covered Entity: Hospital providing PHI for AI system
- PHI: Protected Health Information as defined in 45 CFR 160.103
- Services: Operation of clinical decision support AI system

ARTICLE 2: PERMITTED USES AND DISCLOSURES
UCF may use and disclose PHI only to:
a) Provide clinical decision support services to Hospital
b) Perform quality improvement and validation studies
c) Comply with legal requirements
d) De-identify data for research (per 45 CFR 164.514)

Prohibited Uses:
- Marketing or commercial purposes
- Sale of PHI
- Disclosure to third parties without Hospital authorization

ARTICLE 3: OBLIGATIONS OF BUSINESS ASSOCIATE (UCF)

3.1 Safeguards:
UCF shall implement administrative, physical, and technical safeguards
to prevent use or disclosure of PHI other than as permitted, including:
- Encryption: AES-256 for data at rest, TLS 1.3 for transmission
- Access Controls: Role-based access, multi-factor authentication
- Audit Controls: Comprehensive logging of all PHI access
- Integrity Controls: Hash verification, version control
- Transmission Security: VPN, encrypted channels only

3.2 Subcontractors:
UCF shall ensure any subcontractors (e.g., cloud providers) agree to same
restrictions via written agreement. Current subcontractors:
- Amazon Web Services (AWS) - HIPAA-compliant BAA in place
- [Other vendors as applicable]

3.3 Breach Notification:
UCF shall report any suspected breach to Hospital within 24 hours of discovery
and provide:
- Description of breach (date, time, scope)
- PHI involved (number of individuals, data elements)
- Mitigation steps taken
- Assistance with Hospital's breach notification obligations

3.4 Access Rights:
UCF shall provide access to PHI to individuals upon Hospital's request within
30 days, or provide Hospital with copies of PHI for Hospital to fulfill access
requests.

3.5 Amendment Rights:
UCF shall amend PHI upon Hospital's request within 30 days.

3.6 Accounting of Disclosures:
UCF shall document all disclosures of PHI and provide accounting to Hospital
upon request within 30 days.

ARTICLE 4: OBLIGATIONS OF COVERED ENTITY (HOSPITAL)

4.1 Notice of Privacy Practices:
Hospital shall provide UCF with copy of current Notice of Privacy Practices
and notify UCF of any changes that affect UCF's obligations.

4.2 Restrictions:
Hospital shall notify UCF of any restrictions to use/disclosure of PHI that
Hospital has agreed to, and UCF shall comply.

4.3 Authorizations:
Hospital shall obtain any required authorizations for uses/disclosures and
provide copies to UCF.

ARTICLE 5: TERM AND TERMINATION

5.1 Term:
This Agreement is effective as of [Date] and shall terminate on [Date] or
upon termination of clinical AI deployment, whichever is earlier.

5.2 Termination for Breach:
- Hospital may terminate if UCF breaches material provision and fails to cure
  within 30 days of written notice
- If cure is not possible, Hospital may terminate immediately or:
  a) Report to Secretary of HHS, or
  b) Suspend UCF's access to PHI pending corrective action

5.3 Effect of Termination:
Upon termination, UCF shall:
a) Return all PHI to Hospital, or
b) If return is not feasible, destroy PHI and provide certification
c) Retain PHI only if required by law, with written explanation

ARTICLE 6: MISCELLANEOUS

6.1 Regulatory References:
All references to HIPAA regulations include amendments by HITECH Act and
Omnibus Final Rule.

6.2 Amendment:
Parties agree to amend Agreement to comply with changes in HIPAA regulations.

6.3 Survival:
Obligations regarding return/destruction of PHI survive termination.

6.4 Interpretation:
Any ambiguity shall be resolved to permit compliance with HIPAA.
```

**Timeline for BAA Execution:**
- Initial draft: Hospital legal provides template (1 week)
- UCF review: Office of Research legal review (2-3 weeks)
- Negotiation: Typically 2-4 rounds (4-8 weeks)
- Execution: Signature by authorized officials (1-2 weeks)
- **Total: 8-14 weeks** (should be started concurrently with IRB)

### 5.2 Data Use Agreement (DUA) Structure

**Research Data Use Agreement - Clinical AI Study**

**Distinguishing DUA from BAA:**
- **BAA:** Required under HIPAA for all Business Associates handling PHI
- **DUA:** Research-specific terms for dataset use, publication, IP

**Both are typically required for clinical research partnerships**

**Key DUA Provisions:**

```
ARTICLE 1: PURPOSE
Data will be used solely for research study "Hybrid Reasoning for Acute Care"
approved by UCF IRB [protocol number] and Hospital IRB [protocol number].

ARTICLE 2: DATA ELEMENTS
Appendix A lists all data elements to be provided:
- Demographics: Age, sex, race, ethnicity, ZIP3
- Clinical data: Vital signs, laboratory results, medications, procedures
- Outcomes: Diagnoses, disposition, mortality, readmission
- Temporal data: Timestamps (offset from ED arrival)
- Free text: Clinical notes (de-identified)

ARTICLE 3: DE-IDENTIFICATION
Hospital will provide data as:
[Choose one]
☐ Limited Dataset (16 identifiers removed per 45 CFR 164.514(e))
☐ De-identified Dataset (all 18 identifiers removed per Safe Harbor method)
☐ Identifiable Dataset (requires IRB waiver of authorization)

Hospital's honest broker will perform de-identification before transmission to UCF.

ARTICLE 4: DATA SECURITY (references BAA)
UCF shall maintain safeguards per Business Associate Agreement dated [Date].
Additionally:
- Data stored on UCF-approved secure research servers only
- No data on personal laptops, tablets, or removable media
- Access limited to named investigators: [List names]
- Two-factor authentication required for all access

ARTICLE 5: NO RE-IDENTIFICATION
UCF agrees:
- Will not attempt to re-identify individuals
- Will not contact patients
- Will not link dataset to other databases without Hospital approval and IRB
  amendment

ARTICLE 6: PUBLICATION RIGHTS

6.1 Academic Freedom:
UCF retains right to publish research findings in peer-reviewed journals and
present at academic conferences.

6.2 Hospital Review:
UCF shall provide Hospital with draft manuscripts 30 days before submission
for review to:
- Ensure no patient re-identification risk
- Verify accuracy of hospital/data descriptions
- Identify any confidential or proprietary information

Hospital shall provide feedback within 15 business days or be deemed approved.
Hospital may request delay of publication for up to 90 days to file patent
applications if applicable.

6.3 Acknowledgment:
All publications shall acknowledge Hospital's contribution:
"Data provided by [Hospital Name] in collaboration with University of Central
Florida."

6.4 Authorship:
If Hospital clinicians make substantial intellectual contributions (per ICMJE
criteria), they shall be offered co-authorship. Data provision alone does not
warrant authorship.

ARTICLE 7: INTELLECTUAL PROPERTY

7.1 Data Ownership:
Hospital retains all ownership rights to patient data. UCF acquires no
ownership through this Agreement.

7.2 Inventions:
Any inventions, algorithms, or software developed by UCF using Hospital data
are owned by UCF per Bayh-Dole Act and Florida law. Hospital receives
non-exclusive, royalty-free license for clinical use within Hospital.

7.3 Patents:
If patentable inventions arise, parties will negotiate in good faith for
joint patenting and revenue sharing if Hospital contributed more than data.

ARTICLE 8: DATA RETENTION AND DESTRUCTION

8.1 Retention Period:
UCF may retain data for [duration]:
☐ Duration of study + 5 years (UCF research record retention policy)
☐ Duration of study + 3 years (minimum regulatory requirement)
☐ Other: _______________

8.2 Destruction:
At end of retention period, UCF shall:
- Securely delete all electronic files (DoD 5220.22-M standard)
- Provide Certificate of Destruction to Hospital
- Destruction verified by UCF IT Security

8.3 Public Data Sharing:
De-identified data may be shared in public repository (e.g., PhysioNet,
Dryad) for scientific transparency if:
☐ Permitted (Hospital approves)
☐ Not permitted (Hospital requires exclusive use)
☐ Requires Hospital approval per dataset

ARTICLE 9: COMPLIANCE AND AUDIT

9.1 Regulatory Compliance:
Both parties shall comply with:
- HIPAA Privacy and Security Rules
- 21 CFR Part 11 (if FDA-regulated)
- State privacy laws (Florida Information Protection Act)
- Institutional policies

9.2 Audit Rights:
Hospital reserves right to audit UCF's data security and usage annually or
upon reasonable suspicion of breach. UCF shall cooperate and provide access
within 30 days of written request.

9.3 Reporting:
UCF shall provide Hospital with annual progress reports including:
- Summary of research activities
- Number of individuals whose data was accessed
- Any adverse events or protocol deviations
- Publications or presentations resulting from data

ARTICLE 10: LIABILITY AND INDEMNIFICATION

10.1 Data Quality:
Hospital makes no representations regarding accuracy, completeness, or quality
of data. UCF accepts data "as is" and is responsible for data validation.

10.2 Mutual Indemnification:
- UCF indemnifies Hospital for UCF's breach of data security or unauthorized
  use/disclosure
- Hospital indemnifies UCF for inaccurate data representation or lack of
  patient authorization (if required)

10.3 Limitation of Liability:
Neither party liable for consequential, incidental, or punitive damages.
Maximum liability limited to $[amount] or actual damages, whichever is less.

10.4 Insurance:
UCF shall maintain:
- General liability insurance: $1M per occurrence
- Professional liability insurance: $1M per occurrence
- Cyber liability insurance: $2M per occurrence (covers data breaches)

Certificates of insurance provided to Hospital upon request.

ARTICLE 11: TERM AND TERMINATION

11.1 Term:
Effective [Date] and continuing until earlier of:
- Study completion date: [Date]
- [Number] years from effective date
- Termination per Section 11.2

11.2 Termination:
Either party may terminate with 60 days written notice for:
- Material breach not cured within 30 days
- Convenience (Hospital may terminate any time; UCF may terminate with
  Hospital consent)

11.3 Effect of Termination:
Upon termination:
- UCF returns or destroys data per BAA requirements
- Publications in progress may be completed
- Data from completed analyses may be retained per retention policy
```

**DUA Negotiation Timeline:**
- Often negotiated concurrently with BAA: 8-12 weeks total
- Some hospitals have template DUAs that expedite process: 4-6 weeks

### 5.3 Orlando Health Partnership Pathway

**Institutional Background:**
- Level I Trauma Center (Orlando Regional Medical Center)
- Comprehensive stroke center
- 3,200+ beds across Orlando Health system
- Epic EHR system

**Research Infrastructure:**
- Orlando Health Research Institute
- Existing UCF College of Medicine affiliation
- Clinical trial experience (Phase I-IV)

**Partnership Development Pathway:**

**Phase 1: Initial Engagement (Months 1-2)**
```
1. Identify Champion:
   - Emergency Medicine Medical Director or
   - Chief Medical Information Officer (CMIO) or
   - Research Institute Director

2. Preliminary Meeting:
   - 30-minute presentation: Research overview, benefits to Orlando Health
   - Discuss: Data needs, timeline, clinician involvement
   - Assess: Interest level, potential barriers

3. Concept Approval:
   - Submit 2-page research concept to Orlando Health Research Institute
   - Include: Objectives, data elements, sample size, timeline
   - Review by Research Institute leadership (typically 2-4 weeks)
```

**Phase 2: Formal Proposal (Months 2-4)**
```
1. Detailed Proposal Submission:
   - Full IRB protocol
   - Data dictionary (specific fields requested)
   - Statistical analysis plan
   - Budget (if requesting Hospital resources beyond data)

2. Scientific Review Committee:
   - Monthly meetings
   - 15-minute presentation by PI
   - Focus on: Scientific merit, feasibility, Hospital benefit

3. Conditional Approval:
   - Subject to IRB and contract execution
   - Designate Hospital collaborators (often ED physicians)
```

**Phase 3: Regulatory and Contractual (Months 3-6)**
```
1. IRB Submission:
   - Option A: Dual submission (UCF + Orlando Health IRBs)
   - Option B: Single IRB reliance (UCF as sIRB)
   - Timeline: 8-12 weeks

2. Contract Negotiation:
   - BAA: 8-14 weeks (Hospital template available)
   - DUA: Concurrent with BAA
   - Bottleneck: Legal review cycles at both institutions

3. IT Security Assessment:
   - UCF provides security documentation
   - Hospital IT reviews and approves
   - May require penetration testing: 2-4 weeks
```

**Phase 4: Data Transfer (Months 6-7)**
```
1. Honest Broker De-identification:
   - Hospital analyst creates limited dataset
   - Review sample data for quality check
   - Execute data transfer via secure method (SFTP, secure USB)

2. Data Validation:
   - UCF team verifies data completeness
   - Query resolution with Hospital analyst
   - Final dataset locked for analysis
```

**Total Timeline: 6-7 months** from first contact to data access

**Key Success Factors:**
- Clinical champion engagement (ED physicians excited about AI)
- Demonstrate Hospital benefit (quality improvement, publications)
- Leverage UCF-Orlando Health existing relationship
- Start BAA/DUA negotiation early (don't wait for IRB approval)

### 5.4 AdventHealth Partnership Pathway

**Institutional Background:**
- Faith-based healthcare system (800+ locations, 9 states)
- 50,000+ employees, 82,000+ physicians
- Cerner/Oracle Health EHR system (different from Orlando Health's Epic)
- Corporate headquarters in Altamonte Springs, FL

**Research Infrastructure:**
- CREATION Health Research Institute
- Translational Research Institute
- Strong research culture (>300 active studies)

**Partnership Development Pathway:**

**Phase 1: Initial Engagement (Months 1-2)**
```
1. Entry Point:
   - AdventHealth Research Institute
   - Contact: Research Institute leadership or
   - Emergency Services Service Line medical director

2. Research Concept Submission:
   - Online submission portal
   - 3-page concept document required
   - Include: Objectives, methodology, resources needed, timeline

3. Initial Review:
   - Research Institute staff review for alignment
   - Feasibility assessment (data availability, clinician time)
   - Decision: Invite full proposal or decline (typically 3-4 weeks)
```

**Phase 2: Full Proposal (Months 2-4)**
```
1. Full Protocol Development:
   - Detailed study protocol (IRB-ready)
   - Budget (if requesting funds or significant resources)
   - Letters of support from clinical departments

2. Scientific Review:
   - Research Institute Scientific Review Committee
   - Meets quarterly
   - Criteria: Scientific merit, feasibility, alignment with AdventHealth mission

3. Approval to Proceed:
   - Conditional on IRB and contracts
   - Designated AdventHealth co-investigators assigned
```

**Phase 3: Regulatory and Contractual (Months 3-7)**
```
1. IRB Strategy:
   - AdventHealth uses Advarra (commercial IRB)
   - Option A: Advarra reviews study
   - Option B: UCF IRB as sIRB (AdventHealth relies on UCF)
   - Recommendation: UCF sIRB (faster, lower cost)

2. IRB Authorization Agreement:
   - AdventHealth relies on UCF IRB per 21 CFR 50.104
   - UCF IRB reviews and approves
   - AdventHealth accepts UCF determination
   - Timeline: 2-4 weeks for agreement execution + UCF IRB timeline

3. Contract Negotiation:
   - AdventHealth has template BAA and DUA
   - Generally more streamlined than Orlando Health
   - Timeline: 6-10 weeks (faster than Orlando Health in our experience)
```

**Phase 4: Data Transfer (Months 7-8)**
```
1. Data Extraction:
   - AdventHealth has dedicated research data team
   - Cerner/Oracle Health data warehouse
   - De-identification performed by AdventHealth analysts

2. Data Quality:
   - AdventHealth data often well-curated for research
   - Includes data dictionary and codebook
   - Support for query resolution
```

**Total Timeline: 7-8 months** from first contact to data access

**Comparison with Orlando Health:**

| Factor | Orlando Health | AdventHealth |
|--------|----------------|--------------|
| **Timeline** | 6-7 months | 7-8 months |
| **IRB** | Dual or sIRB | Advarra or sIRB |
| **Contract** | Slower (14 weeks) | Faster (10 weeks) |
| **EHR** | Epic | Cerner/Oracle |
| **Data Team** | Variable | Dedicated research team |
| **Culture** | Academic-affiliated | Faith-based mission |

**Strategic Value of Both Partnerships:**
- **Orlando Health:** Epic EHR (most common in US), Level I trauma
- **AdventHealth:** Different EHR, multi-state system (generalizability)
- **Combined:** Strong evidence of external validity for publications

---

## PHASE 6: TIMELINE SUMMARY

### 6.1 Realistic IRB Approval Timelines

**Phase 1: Public Data Research**
```
Activity                          Timeline        IRB Status
─────────────────────────────────────────────────────────────
PhysioNet CITI training           1-2 weeks       N/A
PhysioNet application             1-2 weeks       N/A
Data access granted               Immediate       No IRB required
Research and publications         12-24 months    No IRB required
─────────────────────────────────────────────────────────────
TOTAL TO START RESEARCH:          2-4 weeks
```

**Phase 2: Retrospective Clinical Validation**
```
Activity                          Timeline        Notes
─────────────────────────────────────────────────────────────
Hospital partnership meetings     1-2 months      Identify champion
Protocol development              1 month         Use templates above
UCF IRB submission (expedited)    2-3 weeks       Administrative review
UCF IRB approval                  2-4 weeks       Expedited review
Hospital IRB reliance             2-4 weeks       Authorization agreement
BAA negotiation (concurrent)      8-14 weeks      Legal review cycles
DUA negotiation (concurrent)      8-12 weeks      With BAA
Data extraction and transfer      2-4 weeks       Hospital analyst
─────────────────────────────────────────────────────────────
TOTAL TO DATA ACCESS:             6-8 months      (Critical path: contracts)
```

**Phase 3: Prospective Clinician Studies**
```
Activity                          Timeline        Notes
─────────────────────────────────────────────────────────────
Protocol and materials dev.       1-2 months      Consent forms, surveys
UCF IRB submission (full board)   1 week          Complete application
Administrative review             1-2 weeks       Completeness check
Full board review                 4-6 weeks       Monthly meetings
Contingent approval               2-4 weeks       Revisions likely
Final approval                    1 week          After revisions
Recruitment                       2-4 months      N=30-50 clinicians
Data collection                   3-6 months      90-min sessions
Analysis                          2-3 months      Statistical analysis
─────────────────────────────────────────────────────────────
TOTAL TO RESULTS:                 12-18 months    (Including recruitment)
```

**Phase 4: Clinical Decision Support Deployment**
```
Activity                          Timeline        Notes
─────────────────────────────────────────────────────────────
FDA determination                 1-2 weeks       Self-assessment (likely exempt)
Hospital governance approval      2-3 months      Medical staff committees
IT security assessment            1-2 months      Penetration testing
EHR integration development       3-6 months      HL7 FHIR interfaces
User training development         1-2 months      Materials and competency
Pilot deployment                  1-2 months      Small ED subset
Full deployment                   1 month         Rollout plan
Post-deployment monitoring        Ongoing         Quality assurance
─────────────────────────────────────────────────────────────
TOTAL TO DEPLOYMENT:              9-15 months     (From Phase 3 completion)
```

### 6.2 Critical Path Analysis

**Bottlenecks to Anticipate:**

1. **Hospital Contracts (Biggest Delay):**
   - BAA/DUA negotiation: 8-14 weeks
   - Mitigation: Start contract discussions before IRB submission
   - UCF Office of Research can expedite if marked "high priority"

2. **Full Board IRB (Phase 3):**
   - Monthly meeting cycle can add 4-6 weeks if submission timing is poor
   - Mitigation: Submit 6-8 weeks before target board date
   - UCF IRB meets 2nd Tuesday of each month (verify current schedule)

3. **EHR Integration (Phase 4):**
   - Hospital IT prioritization can delay 3-6 months
   - Mitigation: Frame as quality improvement (not just research)
   - Align with Hospital's strategic IT initiatives

4. **Clinician Recruitment (Phase 3):**
   - Busy ED physicians are hard to schedule
   - Mitigation: Offer flexible times, generous compensation ($100/session)
   - Snowball sampling from early participants

**Recommended Parallel Activities:**

```
TIMELINE OPTIMIZATION STRATEGY:

Months 1-3:
├─ PhysioNet access (Phase 1) [PARALLEL]
├─ Hospital partnership meetings (Phase 2) [PARALLEL]
└─ Protocol drafting for Phase 2 IRB [PARALLEL]

Months 4-6:
├─ Phase 1 research continues [ONGOING]
├─ UCF IRB submission (Phase 2) [START]
├─ BAA/DUA negotiation (Phase 2) [START - DON'T WAIT FOR IRB]
└─ Protocol development (Phase 3) [START]

Months 7-9:
├─ Phase 2 IRB approval received [COMPLETE]
├─ BAA/DUA finalization [COMPLETE]
├─ Phase 2 data analysis begins [START]
└─ Phase 3 IRB submission [SUBMIT]

Months 10-12:
├─ Phase 2 validation continues [ONGOING]
├─ Phase 3 IRB approval [COMPLETE]
├─ Phase 3 recruitment begins [START]
└─ Phase 2 paper submissions [SUBMIT]

This parallel approach reduces total timeline from 48 months (serial) to 24-30
months (parallel) for Phases 1-3.
```

### 6.3 Budget Implications by Phase

**Regulatory Costs to Budget:**

| Phase | Regulatory Activity | Estimated Cost | Notes |
|-------|---------------------|----------------|-------|
| **Phase 1** | PhysioNet access | $0 | Free for academic research |
| | CITI training | $0 | UCF subscription |
| | Storage (5TB) | $500/year | AWS/Azure research credits available |
| **Phase 2** | UCF IRB fee | $0 | No fee for UCF investigators |
| | Hospital IRB fee | $0-$5,000 | Waived for unfunded studies typically |
| | Legal review (BAA/DUA) | $0 | UCF Office of Research service |
| | Data de-identification | $0-$10,000 | Hospital analyst time (negotiate) |
| **Phase 3** | UCF IRB fee | $0 | No fee |
| | Participant compensation | $3,000-$5,000 | $100 × 30-50 participants |
| | Survey software (REDCap) | $0 | UCF license available |
| **Phase 4** | FDA submission | $0 | Likely exempt (no fee) |
| | Hospital IT assessment | $0-$5,000 | Negotiate as part of partnership |
| | Liability insurance | $2,000-$5,000/year | May be covered by UCF policy |
| | Training materials dev. | $5,000-$10,000 | Staff time, video production |

**TOTAL REGULATORY COSTS: $10,500-$35,500** over 3-4 years
(Primarily Phase 3 participant compensation and Phase 4 training)

**Funding to Seek for Regulatory Activities:**
- NSF budgets can include participant compensation, data costs
- NIH allows "human subjects" costs including IRB fees, consent
- Industry partnerships may cover deployment costs (Phase 4)

---

## APPENDIX: IRB TEMPLATES AND CHECKLISTS

### A.1 UCF IRB Protocol Template (Expedited - Phase 2)

**Use UCF's eIRB system; this is a content guide, not format**

```
1. STUDY TITLE
Retrospective Validation of Hybrid Reasoning AI for Emergency Department Risk
Stratification

2. PRINCIPAL INVESTIGATOR
Name: [Faculty Name]
Department: Computer Science / College of Medicine (joint)
Email: [email]
Phone: [phone]

3. CO-INVESTIGATORS
[List all UCF and hospital collaborators with roles]

4. FUNDING SOURCE
☐ Unfunded pilot study
☐ Internal seed grant: [UCF source]
☐ External grant: [NSF/NIH application pending]

5. STUDY SUMMARY (250 words)
[Use language from Section 2.2 above]

6. STUDY PURPOSE AND OBJECTIVES
Primary Objective: Validate hybrid temporal knowledge graph and neuro-symbolic
AI system for predicting sepsis, mortality, and ICU admission in ED patients.

Secondary Objectives:
- Assess performance across demographic subgroups (age, sex, race)
- Compare performance to standard ML models (XGBoost, LSTM)
- Evaluate algorithm fairness and bias

7. BACKGROUND AND SIGNIFICANCE
[Brief literature review - 500 words]
- Current state: Black-box AI in clinical care lacks interpretability
- Gap: Need for explainable AI in acute care settings
- Approach: Neuro-symbolic reasoning provides transparency
- Impact: Improved clinician trust and adoption of AI tools

8. STUDY DESIGN
Type: Retrospective chart review
Setting: [Hospital Name] Emergency Department
Duration: [Date range of medical records]
Sample Size: 5,000-10,000 patient encounters

9. STUDY POPULATION
Inclusion Criteria:
- Adult patients (≥18 years)
- ED visit between [dates]
- Minimum 4-hour ED length of stay

Exclusion Criteria:
- Pediatric patients (<18 years)
- Incomplete medical records
- Left against medical advice without outcome data

10. RECRUITMENT
N/A - Retrospective chart review, no recruitment

11. STUDY PROCEDURES
Step 1: Hospital honest broker creates limited dataset
Step 2: Dataset transferred to UCF via secure SFTP
Step 3: UCF researchers perform analysis
Step 4: Results aggregated for publication

No contact with patients.

12. RISKS TO SUBJECTS
Primary Risk: Privacy breach if data is compromised

Likelihood: Low
Magnitude: Moderate (embarrassment, discrimination if re-identified)

Mitigation:
- Limited dataset (16 identifiers removed)
- Encryption (AES-256 at rest, TLS in transit)
- Access controls (named investigators only, 2FA)
- No re-identification attempts
- Data destruction after study completion

Secondary Risk: None (no patient contact or intervention)

Risk Classification: Minimal Risk

13. BENEFITS TO SUBJECTS
Direct Benefits: None (patients will not be contacted)

Societal Benefits: Development of improved clinical decision support tools
that may benefit future patients

14. DATA MANAGEMENT AND CONFIDENTIALITY
[Use language from Section 2.3 and 2.5 above]

Data Storage: UCF research servers, encrypted
Access: [List names of personnel with access]
Retention: 5 years post-publication per UCF policy
Destruction: Secure deletion with certificate

15. WAIVER OF INFORMED CONSENT
Requested: YES

Justification per 45 CFR 46.116(d):
[Use language from Section 2.4 above]

16. HIPAA COMPLIANCE
Dataset Type: Limited Dataset per 45 CFR 164.514(e)
Authorization: Waiver requested (impracticable to obtain)
Business Associate Agreement: Executed between UCF and [Hospital]

17. DATA AND SAFETY MONITORING PLAN
Risk Level: Minimal risk - no patient intervention
DSMB: Not required
Monitoring: PI reviews data quality and security quarterly
Adverse Events: Report any data breaches to IRB within 24 hours

18. STATISTICAL ANALYSIS PLAN
Primary Outcome: AUROC for sepsis, mortality, ICU admission prediction
Sample Size: N=5,000-10,000 provides >90% power to detect AUC difference
of 0.05 vs baseline models
Analysis: Stratified by demographics, DeLong test for AUC comparison

19. DISSEMINATION OF RESULTS
Publications: JAMIA, Nature Digital Medicine, NeurIPS (planned)
Presentations: AMIA Annual Symposium
Data Sharing: De-identified results may be shared in public repository

20. STUDY TIMELINE
IRB Approval: [Month 0]
Data Transfer: [Month 1]
Analysis: [Months 1-6]
Manuscript Submission: [Month 8]
Study Closure: [Month 12]

21. INVESTIGATOR QUALIFICATIONS
[Brief CV highlights demonstrating expertise in AI and clinical research]

22. CONFLICTS OF INTEREST
None. / [Disclose any relevant conflicts]

23. REFERENCES
[Key citations supporting study rationale]

24. ATTACHMENTS
- Hospital IRB approval letter (if available) or reliance agreement
- Business Associate Agreement (executed or in negotiation)
- Data Use Agreement (executed or in negotiation)
- Data dictionary (list of all fields requested)
```

### A.2 Informed Consent Template (Phase 3 - Clinician Study)

```
UNIVERSITY OF CENTRAL FLORIDA
INFORMED CONSENT TO PARTICIPATE IN RESEARCH

Title: Clinician Evaluation of Explainable AI for Emergency Department
       Decision Support

Principal Investigator: [Name], PhD
                        Department of Computer Science
                        University of Central Florida

Co-Investigator:        [Name], MD
                        College of Medicine
                        University of Central Florida

─────────────────────────────────────────────────────────────────────────────

INTRODUCTION

You are being asked to participate in a research study because you are an
emergency medicine physician, physician assistant, or nurse practitioner with
experience in acute care. This study is evaluating different ways of
explaining artificial intelligence predictions to clinicians.

This consent form will provide information about the research study. Please
read this form carefully and ask questions before deciding whether to
participate.

Participation is voluntary. You may refuse to participate or withdraw at any
time without penalty.

PURPOSE OF THE STUDY

Artificial intelligence (AI) is increasingly being used to help doctors make
clinical decisions. However, many AI systems are "black boxes" that don't
explain their reasoning. This study evaluates different explanation methods
to determine which are most helpful for emergency physicians.

NUMBER OF PARTICIPANTS

We plan to enroll 30-50 emergency medicine clinicians in this study.

DURATION OF PARTICIPATION

Your participation will involve one 90-minute session. You may also be asked
to participate in an optional 30-minute follow-up interview 2-4 weeks later.

PROCEDURES

If you agree to participate, you will:

1. Complete a brief demographic survey (5 minutes)
   - Questions about your training, experience, and familiarity with AI

2. Review 20 simulated emergency department patient cases (60 minutes)
   - Each case includes patient vital signs, lab results, and clinical notes
   - You will see AI predictions for patient risk (sepsis, deterioration)
   - Different cases will show different types of AI explanations:
     * Feature importance (which data points mattered most)
     * Reasoning chains (logical steps the AI followed)
     * Temporal graphs (how patient condition changed over time)
     * Counterfactuals (what would change the prediction)

3. For each case, answer questions about (total 60 minutes):
   - Do you understand the AI explanation?
   - Do you agree with the AI prediction?
   - How confident are you in your decision?
   - How much do you trust this AI system?

4. Complete post-study survey (5 minutes)
   - Overall impressions of different explanation types
   - Preferences for clinical use

5. OPTIONAL: Follow-up interview (30 minutes, 2-4 weeks later)
   - In-depth discussion about what makes explanations useful

Important: These are SIMULATED cases based on de-identified public data. You
will NOT be making real clinical decisions, and no actual patients are
involved. This is purely a research study to understand how explanations work.

RISKS AND DISCOMFORTS

The risks of this study are minimal.

Potential risks include:
- Mild fatigue from the 90-minute session
- Frustration if AI explanations are confusing (this is what we're studying)

There are NO risks to patient safety because this study does NOT involve real
patient care. All cases are simulated.

BENEFITS

You may not directly benefit from participating in this study.

Potential benefits to you:
- Gaining exposure to AI decision support tools that may be used in future
  clinical practice
- Opportunity to shape how AI systems are designed for clinicians

Potential benefits to society:
- Improved design of AI explanation systems for healthcare
- Better understanding of clinician needs for AI adoption

COMPENSATION

You will receive a $100 Amazon gift card for completing the 90-minute session.

If you participate in the optional follow-up interview, you will receive an
additional $50 Amazon gift card.

Gift cards will be distributed via email within 2 weeks of session completion.

If you withdraw before completing the session, you will receive prorated
compensation based on time spent ($1.11 per minute).

CONFIDENTIALITY

Your identity and responses will be kept confidential to the extent allowed
by law.

How we will protect your privacy:
- You will be assigned a random participant ID number (e.g., "P001")
- Your name will NOT be recorded in the study data
- Your responses will be stored separately from consent forms
- Only researchers on this study will have access to identifiable information
- Published results will report only group averages, not individual responses

Data storage and security:
- Electronic data stored on secure UCF research servers (encrypted, password-
  protected, two-factor authentication)
- Paper consent forms (this document) stored in locked file cabinet in PI's
  office
- Data retained for 5 years after publication, then securely destroyed

De-identified data sharing:
We may share de-identified study data in a public repository (e.g., Open
Science Framework) to allow other researchers to verify our findings. This
helps advance science. De-identified data will include your responses but NO
information that could identify you (no name, employer, demographics beyond
age/gender/years of experience).

Do you consent to de-identified data sharing?
☐ YES, I consent to de-identified data sharing
☐ NO, I do not consent to de-identified data sharing

Note: You can still participate in the study even if you check "NO" above.

VOLUNTARY PARTICIPATION

Participation in this study is completely voluntary.

You may:
- Decline to participate without any consequences
- Refuse to answer any question
- Withdraw from the study at any time

If you withdraw:
- Your data will be destroyed unless you give permission to retain it
- You will receive compensation for time spent up to withdrawal
- There will be no penalty or loss of benefits

CONTACT INFORMATION

If you have questions about this study, contact:

Principal Investigator:
[Name], PhD
Department of Computer Science, UCF
Email: [email]
Phone: [phone]

If you have questions about your rights as a research participant, contact:

UCF Institutional Review Board (IRB)
University of Central Florida
12201 Research Parkway, Suite 501
Orlando, FL 32826-3246
Phone: 407-823-2901
Email: irb@ucf.edu

CONSENT TO PARTICIPATE

I have read this consent form (or it has been read to me). I have had the
opportunity to ask questions, and my questions have been answered to my
satisfaction.

By signing below, I voluntarily agree to participate in this research study.

I will receive a copy of this signed consent form for my records.

_________________________________     _______________
Participant Signature                  Date

_________________________________
Participant Name (printed)

_________________________________     _______________
Person Obtaining Consent Signature     Date

_________________________________
Person Obtaining Consent Name (printed)
```

### A.3 IRB Submission Checklist

**Before submitting to UCF IRB, verify you have:**

Phase 2 (Expedited Review - Retrospective Chart Review):
```
☐ Complete protocol (use template in Appendix A.1)
☐ Data dictionary listing all data elements requested
☐ Hospital IRB approval or reliance agreement letter
☐ BAA executed or letter confirming in negotiation
☐ DUA executed or letter confirming in negotiation
☐ CITI training certificates for all investigators (current within 3 years)
☐ PI CV/biosketch
☐ Conflict of interest disclosure for all investigators
☐ Waiver of consent justification (Section 2.4)
☐ HIPAA waiver justification (Section 2.3)
☐ Data security plan (Section 2.3)
☐ Statistical analysis plan with sample size justification
```

Phase 3 (Full Board Review - Clinician Study):
```
☐ Complete protocol
☐ Informed consent form (use template in Appendix A.2)
☐ Recruitment materials (email script, flyer)
☐ Study instruments (surveys, case vignettes)
☐ CITI training certificates for all investigators
☐ PI CV/biosketch
☐ Conflict of interest disclosure
☐ Data and safety monitoring plan
☐ Data management plan (Section 3.5)
☐ Statistical analysis plan with power calculation
☐ Compensation justification ($100 for 90 min is ~$67/hr, reasonable)
```

---

## CONCLUSION

This IRB and regulatory pathway section demonstrates that the research team has:

1. **Thoroughly analyzed regulatory requirements** across all phases of clinical AI research from public data studies to clinical deployment

2. **Provided actionable templates and timelines** that enable immediate execution without regulatory uncertainty

3. **Identified the optimal pathway** through complex FDA, HIPAA, and institutional requirements

4. **Established realistic expectations** for approval timelines (8-12 weeks for expedited IRB, 12-16 weeks for full board)

5. **De-risked hospital partnerships** by providing template BAA and DUA language that aligns with institutional norms

**Key Strategic Advantages:**

- **Phase 1 requires NO IRB**, allowing immediate research productivity with MIMIC/eICU datasets
- **Phase 2 qualifies for expedited review**, not full board (faster approval)
- **Phase 3 separates human subjects risk** by studying clinicians, not patients (lower risk)
- **Phase 4 likely FDA-exempt** under 21st Century Cures Act (no premarket burden)

**This demonstrates to reviewers:**
- Research team has domain expertise in clinical research regulations
- Project is shovel-ready with clear execution path
- Timeline is realistic and accounts for regulatory complexity
- Risk mitigation strategies are in place

**Next Actions:**
1. Customize templates with specific investigator names and dates
2. Initiate PhysioNet access (2-3 weeks to data)
3. Schedule exploratory meetings with Orlando Health/AdventHealth (Month 1)
4. Submit Phase 2 IRB protocol (Month 3-4, after hospital partnership secured)

This regulatory roadmap elevates the research brief from "interesting idea" to "ready to execute" - a critical differentiator for funding and institutional support.
