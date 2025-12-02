# FDA Clinical Decision Support Software Guidance - Current Regulatory Framework

**Document Compiled:** November 30, 2025
**Sources:** FDA.gov, Federal Register, Official FDA Guidance Documents
**Status:** Current regulatory guidance as of November 2025

---

## Executive Summary

This document compiles current FDA regulatory guidance on Clinical Decision Support (CDS) software, AI/ML-enabled medical devices, and the 21st Century Cures Act exemption criteria. All information is sourced from official FDA publications and Federal Register notices accessed in November 2025.

---

## 1. CLINICAL DECISION SUPPORT SOFTWARE - FINAL GUIDANCE (September 2022)

### 1.1 Regulatory Authority

**Source:** [FDA Clinical Decision Support Software Guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software)
**Federal Register:** [September 28, 2022 Notice](https://www.federalregister.gov/documents/2022/09/28/2022-20993/clinical-decision-support-software-guidance-for-industry-and-food-and-drug-administration-staff)
**Accessed:** November 30, 2025

The FDA published the final guidance "Clinical Decision Support Software" in September 2022 to clarify FDA oversight of clinical decision support software intended for healthcare professionals. This guidance interprets section 520(o) of the Federal Food, Drug & Cosmetic Act (FD&C Act), as amended by Section 3060(a) of the 21st Century Cures Act.

### 1.2 The Four Statutory Criteria for Non-Device CDS

**Legal Citation:** Section 520(o)(1)(E) of the FD&C Act

A software function must meet **ALL FOUR** criteria to be excluded from the definition of a medical device:

#### **Criterion 1: No Image/Signal Processing**
The software is **NOT** intended to acquire, process, or analyze:
- A medical image, OR
- A signal from an in vitro diagnostic device, OR
- A pattern or signal from a signal acquisition system

**FDA Interpretation:** The Agency broadly interprets this to include assessments of "clinical implications or clinical relevance" of medical images or signals.

**Source:** [FDA Step 6 Decision Framework](https://www.fda.gov/medical-devices/digital-health-center-excellence/step-6-software-function-intended-provide-clinical-decision-support)
**Accessed:** November 30, 2025

#### **Criterion 2: Display Medical Information**
The software is intended for the purpose of displaying, analyzing, or printing medical information about a patient or other medical information (such as peer-reviewed clinical studies and clinical practice guidelines).

**FDA Interpretation:** This includes information "normally communicated between health care professionals" with "well understood and accepted relevance" to clinical decisions, including:
- Patient demographics
- Test results
- Clinical guidelines
- Peer-reviewed studies
- Symptoms
- Discharge summaries
- Approved drug labeling

**Source:** [FDA CDS FAQs](https://www.fda.gov/medical-devices/software-medical-device-samd/clinical-decision-support-software-frequently-asked-questions-faqs)
**Accessed:** November 30, 2025

#### **Criterion 3: Provide Recommendations (Not Directives)**
The software is intended for the purpose of supporting or providing recommendations to a health care professional about prevention, diagnosis, or treatment of a disease or condition.

**FDA Interpretation:**
- Software must provide "recommendations (information/options)" rather than "specific output or directive"
- Must enhance, inform, or influence healthcare decisions WITHOUT replacing or directing HCP judgment
- Software providing condition-, disease-, or patient-specific recommendations qualifies IF it doesn't direct HCP judgment

**Examples of Compliant Recommendations:**
- Drug-interaction alerts
- Clinician order sets
- Duplicate testing prevention warnings

**Examples of NON-Compliant (Device) Functions:**
- Risk probability scores for individual patients
- Specific preventive, diagnostic, or treatment courses
- Time-critical alerts or alarms

**Critical Exclusion - Time-Critical Decision Making:**
> "Software functions intended for use in 'time-critical' decision-making will not meet Criterion 3 because, in such settings, HCPs are more likely to suffer from automation bias and place undue reliance on the software's suggestions or information rather than their own medical judgment."

**Source:** [Federal Register Notice - September 28, 2022](https://www.federalregister.gov/documents/2022/09/28/2022-20993/clinical-decision-support-software-guidance-for-industry-and-food-and-drug-administration-staff)
**Accessed:** November 30, 2025

#### **Criterion 4: Independent Review of Basis**
The software is intended for the purpose of enabling the healthcare professional to independently review the basis for such recommendations that such software presents so that it is not the intent that such health care professional rely primarily on any of such recommendations to make a clinical diagnosis or treatment decision regarding an individual patient.

**FDA Interpretation:**
- Software must enable HCPs to independently evaluate the recommendation basis "so that...they do not rely primarily" on the system
- Requires transparent disclosure of:
  - Algorithm logic
  - Validation data
  - Input requirements
  - Clinical evidence supporting recommendations

**Statutory Language from FD&C Act Section 520(o)(1)(E)(iii):**
> "enabling such health care professional to independently review the basis of such recommendations that such software presents so that it is not the intent that such health care professional rely primarily on any such recommendations to make a clinical diagnosis or treatment decision regarding an individual patient."

**Source:** [FDA Guidance Analysis - Nixon Law Group](https://nixongwiltlaw.com/nlg-blog/2022/10/10/fda-issues-final-guidance-on-clinical-decision-support-software-and-software-as-a-medical-device-key-takeaways-and-what-it-means-for-digital-health-companies)
**Accessed:** November 30, 2025

### 1.3 Critical Regulatory Changes in Final Guidance

**Elimination of Risk-Based Enforcement Discretion:**
The Final Guidance removed the International Medical Device Regulators Forum (IMDRF) risk categorization and the risk-based enforcement discretion policy that appeared in the 2019 Revised Draft Guidance. This creates a more binary classification system:
- Non-Device CDS (meets all four criteria)
- Device CDS (fails any of the four criteria)

**Scope Narrowing:**
The final guidance significantly narrowed the scope of exempt clinical decision support software compared to prior drafts, bringing more CDS software under device regulation.

**Source:** [The Incredible Shrinking Exemption - FDA Law Blog](https://www.thefdalawblog.com/2022/10/the-incredible-shrinking-exemption-fda-final-cds-guidance-would-significantly-narrow-the-scope-of-exempt-clinical-decision-support-software-under-the-cures-act/)
**Accessed:** November 30, 2025

### 1.4 Additional CDS Interpretations

**Patient-Facing Software:**
Software functions intended for patients or caregivers do NOT meet the criteria in section 520(o)(1)(E) since the software is not limited to supporting or providing recommendations to health care professionals.

**Multi-Function Products:**
Products can contain both non-device CDS functions and regulated device functions simultaneously, with each function evaluated independently.

**Platform Developers:**
Infrastructure providers are not considered device manufacturers, though those distributing medical software as a service may be.

**Source:** [FDA CDS FAQs](https://www.fda.gov/medical-devices/software-medical-device-samd/clinical-decision-support-software-frequently-asked-questions-faqs)
**Accessed:** November 30, 2025

---

## 2. 21ST CENTURY CURES ACT - CDS EXEMPTION CRITERIA

### 2.1 Legislative Background

**Public Law:** 114-255
**Signed:** December 13, 2016
**Purpose:** Accelerate medical product development and bring innovations to patients faster and more efficiently

**Source:** [FDA 21st Century Cures Act Overview](https://www.fda.gov/regulatory-information/selected-amendments-fdc-act/21st-century-cures-act)
**Accessed:** November 30, 2025

### 2.2 Section 3060(a) - Clarifying Medical Software Regulation

Section 3060(a) of the Cures Act amended Section 520 of the Federal Food, Drug, and Cosmetic Act (FD&C Act), adding subsection (o), which describes specific software functions that are excluded from the definition of "device."

**Statutory Purpose:** Remove certain software functions from the definition of a device subject to FDA regulation.

### 2.3 Complete Statutory Text - Section 520(o)(1)(E)

The Cures Act exempts certain software functions from the statutory definition of "medical device" under FDCA Section 201(h)(1):

> "unless the function is intended to acquire, process, or analyze a medical image or a signal from an in vitro diagnostic device or a pattern or signal from a signal acquisition system, for the purposes of —
>
> (i) displaying, analyzing, or printing medical information about a patient or other medical information;
>
> (ii) supporting or providing recommendations to a health care professional about prevention, diagnosis, or treatment of a disease or condition; and
>
> (iii) enabling such health care professional to independently review the basis of such recommendations that such software presents so that it is not the intent that such health care professional rely primarily on any such recommendations to make a clinical diagnosis or treatment decision regarding an individual patient."

**Source:** [FDA Section 520(o) Analysis - Crowell & Moring](https://www.crowell.com/en/insights/client-alerts/fda-issues-final-guidance-on-clinical-decision-support-software)
**Accessed:** November 30, 2025

### 2.4 Congressional Concerns (2024-2025)

In recent correspondence to Dr. Michelle Tarver, newly appointed director of the Center for Devices and Radiological Health, seven members of Congress raised concerns that FDA's guidance does not reflect its typical risk-based approach or consider the significant clinical oversight under which CDS tools are configured.

**Key Concerns:**
> "FDA's guidance indicates that CDS tools cannot qualify for the exemption unless they provide multiple recommendations. CDS tools often provide a single recommendation when users determine there is only one appropriate option based on clinical practice guidelines. This guidance would seem to make much of CDS used throughout the healthcare system ineligible for the exemption."

**Source:** [Healthcare IT News - Congressional Letter](https://www.healthcareitnews.com/news/lawmakers-ask-cdrh-revisit-its-cds-guidance)
**Accessed:** November 30, 2025

---

## 3. FDA AI/ML MEDICAL DEVICE GUIDANCE (2024-2025)

### 3.1 Regulatory Framework Overview

The FDA reviews AI/ML medical devices through standard premarket pathways:
- **510(k) clearance**
- **De Novo classification**
- **Premarket approval (PMA)**

However, the FDA acknowledges its traditional regulatory model wasn't designed for adaptive AI/ML technologies, necessitating new approaches.

**Source:** [FDA Artificial Intelligence in Software as a Medical Device](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-software-medical-device)
**Accessed:** November 30, 2025

### 3.2 Major Guidance Documents (2021-2025)

#### **October 2021: Good Machine Learning Practice for Medical Device Development**

The FDA, Health Canada, and UK MHRA jointly established **10 guiding principles** for medical device development using AI/ML:

1. **Multi-Disciplinary Expertise** - Leveraged throughout the total product life cycle
2. **Software Engineering & Security** - Implement proven practices for robust development
3. **Representative Participants & Data** - Ensure study groups reflect intended patient population
4. **Independent Datasets** - Keep training and test data separate to avoid bias
5. **Reference Datasets** - Base selections on "Best Available Methods"
6. **Tailored Model Design** - Align design with available data and device purpose
7. **Human-AI Team Performance** - Emphasize how humans and systems work together
8. **Clinically Relevant Testing** - Validate performance under realistic conditions
9. **Clear User Information** - Users are provided clear, essential information
10. **Post-Deployment Monitoring** - Deployed models are monitored for performance and re-training risks are managed

**Source:** [FDA Good Machine Learning Practice](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles)
**Accessed:** November 30, 2025

#### **June 2024: Transparency for Machine Learning-Enabled Medical Devices**

The FDA published "Transparency for Machine Learning-Enabled Medical Devices: Guiding Principle," establishing guiding principles for transparency for ML-enabled medical devices. The recommendations built upon overarching GMLP guiding principles.

**Source:** [Bipartisan Policy Center - FDA AI Oversight](https://bipartisanpolicy.org/issue-brief/fda-oversight-understanding-the-regulation-of-health-ai-tools/)
**Accessed:** November 30, 2025

#### **December 2024: Final Guidance - Predetermined Change Control Plans (PCCP) for AI-Enabled Devices**

**Document:** "Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions"
**Published:** December 3, 2024
**Status:** Final Guidance

**Purpose:**
Enable device manufacturers to implement device modifications that have been pre-authorized by FDA as part of an initial marketing application instead of submitting a new marketing application for each change. This is particularly useful for AI-enabled devices designed to evolve over time.

**Scope and Applicability:**
- AI-enabled devices reviewed through 510(k), De Novo, and PMA pathways
- Device constituent part of device-led combination products
- **NOTE:** PCCPs would NOT be accepted for drug-led combination products

**Key Content Requirements:**
A PCCP must describe:
1. Planned AI-enabled device software function (AI-DSF) modifications
2. Associated methodology to develop, validate, and implement modifications
3. Assessment of the impact of those modifications

**Notable Changes from Draft Guidance:**
- **Expanded Scope:** "Machine Learning Device Software Functions (ML-DSFs)" renamed to "AI-Enabled Device Software Functions (AI-DSFs)"
- **Data Categories Modified:** Changed from "training and testing" to "training, tuning, and testing"
- **Combination Products:** New example PCCP scenario for device-led combination product: imaging system co-packaged with approved optical imaging drug

**Implementation Considerations:**
Manufacturers must assess impact on:
- Design control
- Data management practices
- Risk management plans
- Adverse event reporting
- Labeling

**Source:** [FDA PCCP Guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence)
**Federal Register:** [December 4, 2024 Notice](https://www.federalregister.gov/documents/2024/12/04/2024-28361/marketing-submission-recommendations-for-a-predetermined-change-control-plan-for-artificial)
**Accessed:** November 30, 2025

#### **January 2025: Draft Guidance - AI-Enabled Device Software Functions Lifecycle Management**

**Document:** "Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations"
**Published:** January 6, 2025
**Status:** Draft Guidance (Non-binding recommendations)
**Comment Period:** Through April 7, 2025
**Webinar:** February 18, 2025

**Purpose:**
Provide comprehensive recommendations for AI-enabled devices throughout the total product lifecycle (TPLC), including:
- Contents of marketing submissions for devices with AI-enabled device software functions
- Documentation and information to support FDA evaluation of safety and effectiveness
- Design, development, and implementation considerations throughout TPLC
- Strategies to address transparency and bias

**Significance:**
If finalized, this would be the **first guidance to provide comprehensive recommendations for AI-enabled devices throughout the total product lifecycle.**

**Key Focus Areas:**
- Lifecycle management across the total product lifecycle
- Marketing submission requirements for AI-integrated medical devices
- Design and development recommendations manufacturers should consider
- Safety and effectiveness evaluation support for FDA reviewers
- Transparency and bias mitigation strategies

**Source:** [FDA AI-Enabled Device Software Functions Guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/artificial-intelligence-enabled-device-software-functions-lifecycle-management-and-marketing)
**Federal Register:** [January 7, 2025 Notice](https://www.federalregister.gov/documents/2025/01/07/2024-31543/artificial-intelligence-enabled-device-software-functions-lifecycle-management-and-marketing)
**Press Release:** [FDA Issues Comprehensive Draft Guidance](https://www.fda.gov/news-events/press-announcements/fda-issues-comprehensive-draft-guidance-developers-artificial-intelligence-enabled-medical-devices)
**Accessed:** November 30, 2025

### 3.3 FDA AI/ML Strategic Initiatives

#### **AI/ML SaMD Action Plan (January 2021)**
The FDA published an action plan to systematically address regulatory challenges posed by AI/ML-driven software as a medical device.

#### **Cross-Agency Coordination (March 2024)**
The FDA published "Artificial Intelligence and Medical Products: How CBER, CDER, CDRH, and OCP are Working Together," representing the FDA's coordinated approach to AI across:
- Center for Biologics Evaluation and Research (CBER)
- Center for Drug Evaluation and Research (CDER)
- Center for Devices and Radiological Health (CDRH)
- Office of Combination Products (OCP)

This document represents a commitment to drive alignment and share learnings applicable to AI in medical products broadly.

**Source:** [FDA AI/ML Overview](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-software-medical-device)
**Accessed:** November 30, 2025

#### **Digital Health Advisory Committee (DHAC)**
The FDA established an external advisory committee providing expert input on fast-moving digital health and AI issues:
- **Inaugural Meeting:** November 2024
- **Next Meeting:** Scheduled November 2025

#### **AI Governance Councils (2025)**
The FDA created two cross-agency councils:
1. **External Policy Council:** Establishes principles and policies for AI in regulated products
2. **Internal Use Council:** Oversees how FDA uses AI internally to improve efficiency

**Source:** [Bipartisan Policy Center - FDA AI Oversight](https://bipartisanpolicy.org/issue-brief/fda-oversight-understanding-the-regulation-of-health-ai-tools/)
**Accessed:** November 30, 2025

### 3.4 FDA-Authorized AI/ML Medical Devices

**Market Growth Statistics:**
- **July 2025:** Over 1,250 AI-enabled medical devices authorized for marketing in the United States
- **August 2024:** 950 AI-enabled medical devices authorized
- **Growth:** Approximately 300 new authorizations in less than one year (31% increase)

**Source:** [Bipartisan Policy Center - FDA AI Oversight](https://bipartisanpolicy.org/issue-brief/fda-oversight-understanding-the-regulation-of-health-ai-tools/)
**Accessed:** November 30, 2025

---

## 4. REGULATORY DECISION FRAMEWORK

### 4.1 FDA Digital Health Policy Navigator

The FDA provides a Digital Health Policy Navigator tool - a collection of guidances describing how the FDA intends to apply its device regulatory authority to software functions. This tool can be used to walk through policies with specific products.

**Source:** [FDA Digital Health Center of Excellence](https://www.fda.gov/medical-devices/digital-health-center-excellence/step-6-software-function-intended-provide-clinical-decision-support)
**Accessed:** November 30, 2025

### 4.2 Key Decision Points for CDS Software

#### **Step 1: Does the software acquire, process, or analyze medical images or signals?**
- If YES → Likely a medical device (Criterion 1 fails)
- If NO → Proceed to Step 2

#### **Step 2: Does the software display medical information normally communicated between HCPs?**
- If NO → Likely a medical device (Criterion 2 fails)
- If YES → Proceed to Step 3

#### **Step 3: Does the software provide recommendations (not directives)?**
- If provides specific directives → Medical device (Criterion 3 fails)
- If provides multiple options/recommendations → Proceed to Step 4
- If used in time-critical settings → Medical device (Criterion 3 fails)

#### **Step 4: Can HCPs independently review the basis for recommendations?**
- If NO (black box algorithm) → Medical device (Criterion 4 fails)
- If YES (transparent basis provided) → May qualify as Non-Device CDS

**All four criteria must be met for Non-Device CDS classification.**

### 4.3 Red Flags for Medical Device Classification

The following characteristics indicate software is likely a medical device:
- Time-critical alerts or alarms
- Single specific output or directive
- Risk probability scores for individual patients
- Specific preventive, diagnostic, or treatment courses
- Patient-facing or caregiver-facing (not HCP-facing)
- Processes medical images or diagnostic signals
- Black box algorithm without transparent basis
- Intended for HCP to rely primarily on recommendations

**Source:** [FDA Step 6 Decision Framework](https://www.fda.gov/medical-devices/digital-health-center-excellence/step-6-software-function-intended-provide-clinical-decision-support)
**Accessed:** November 30, 2025

---

## 5. PRACTICAL EXAMPLES AND INTERPRETATIONS

### 5.1 Non-Device CDS Examples (from FDA FAQs)

#### **Example 1: Digitized Clinical Surveys**
Electronic versions of established clinical instruments (e.g., PHQ-9, GAD-7) are non-device functions when they merely automate distribution and completion of familiar assessment methods.

#### **Example 2: Routine Medical Calculations**
Software automating standard calculations receives FDA enforcement discretion:
- Body Mass Index (BMI)
- Glasgow Coma Scale
- APGAR score

**Rationale:** These are technically device-adjacent but pose minimal risk since clinicians can calculate manually.

**Source:** [FDA CDS FAQs](https://www.fda.gov/medical-devices/software-medical-device-samd/clinical-decision-support-software-frequently-asked-questions-faqs)
**Accessed:** November 30, 2025

### 5.2 Device CDS Examples (Fails Exemption Criteria)

#### **Example 1: Time-Critical Decision Support**
Software intended to support urgent decisions cannot qualify as non-device CDS, as clinicians may rely primarily on recommendations in emergencies.

**Source:** [FDA CDS FAQs - Q3](https://www.fda.gov/medical-devices/software-medical-device-samd/clinical-decision-support-software-frequently-asked-questions-faqs)
**Accessed:** November 30, 2025

#### **Example 2: Predictive Decision Support Interventions**
Some predictive decision support interventions (as defined by ASTP/ONC) meet device definitions and should be evaluated under FDA CDS guidance.

**Source:** [FDA CDS FAQs - Q7](https://www.fda.gov/medical-devices/software-medical-device-samd/clinical-decision-support-software-frequently-asked-questions-faqs)
**Accessed:** November 30, 2025

---

## 6. CURRENT REGULATORY DEBATES AND CONCERNS

### 6.1 Industry and Congressional Pushback

Multiple stakeholders have raised concerns that the September 2022 final guidance significantly narrows the Cures Act exemption beyond Congressional intent:

**Congressional Letter to CDRH (2024):**
Seven members of Congress raised specific concerns:
- FDA's guidance does not reflect typical risk-based approach
- Does not consider significant clinical oversight under which CDS tools are configured
- Single-recommendation tools may be excluded despite clinical validity
- Much CDS used throughout healthcare system may be ineligible

**Source:** [Healthcare IT News - Congressional Letter](https://www.healthcareitnews.com/news/lawmakers-ask-cdrh-revisit-its-cds-guidance)
**Accessed:** November 30, 2025

### 6.2 Legal and Policy Analysis

**"The Incredible Shrinking Exemption":**
Legal analysts have characterized the final guidance as significantly narrowing the scope of exempt clinical decision support software under the Cures Act compared to earlier drafts and potentially Congressional intent.

**Key Changes from Earlier Drafts:**
- Removal of IMDRF risk categorization
- Elimination of risk-based enforcement discretion
- More restrictive interpretation of "recommendations" vs "directives"
- Strict application of time-critical exclusion
- Binary classification system (Non-Device vs Device)

**Source:** [FDA Law Blog Analysis](https://www.thefdalawblog.com/2022/10/the-incredible-shrinking-exemption-fda-final-cds-guidance-would-significantly-narrow-the-scope-of-exempt-clinical-decision-support-software-under-the-cures-act/)
**Accessed:** November 30, 2025

---

## 7. RESOURCES AND TOOLS

### 7.1 Official FDA Resources

#### **Primary Guidance Documents:**
1. Clinical Decision Support Software (Final - September 2022)
   - https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
   - Download: https://www.fda.gov/media/109618/download

2. AI-Enabled Device Software Functions: Lifecycle Management (Draft - January 2025)
   - https://www.fda.gov/regulatory-information/search-fda-guidance-documents/artificial-intelligence-enabled-device-software-functions-lifecycle-management-and-marketing

3. Marketing Submission Recommendations for PCCP for AI-Enabled Devices (Final - December 2024)
   - https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence

4. Good Machine Learning Practice for Medical Device Development (October 2021)
   - https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles

#### **Interactive Tools:**
1. Digital Health Policy Navigator
   - https://www.fda.gov/medical-devices/digital-health-center-excellence

2. CDS Decision Framework (Step 6)
   - https://www.fda.gov/medical-devices/digital-health-center-excellence/step-6-software-function-intended-provide-clinical-decision-support

3. Clinical Decision Support Software FAQs
   - https://www.fda.gov/medical-devices/software-medical-device-samd/clinical-decision-support-software-frequently-asked-questions-faqs

#### **AI/ML Resources:**
1. Artificial Intelligence in Software as a Medical Device
   - https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-software-medical-device

2. Predetermined Change Control Plans Overview
   - https://www.fda.gov/regulatory-information/search-fda-guidance-documents/predetermined-change-control-plans-medical-devices

### 7.2 Legislative Resources

1. 21st Century Cures Act Overview
   - https://www.fda.gov/regulatory-information/selected-amendments-fdc-act/21st-century-cures-act

2. 21st Century Cures Act Deliverables
   - https://www.fda.gov/regulatory-information/21st-century-cures-act/21st-century-cures-act-deliverables

### 7.3 Federal Register Notices

1. Clinical Decision Support Software (September 28, 2022)
   - https://www.federalregister.gov/documents/2022/09/28/2022-20993/clinical-decision-support-software-guidance-for-industry-and-food-and-drug-administration-staff

2. PCCP for AI-Enabled Devices (December 4, 2024)
   - https://www.federalregister.gov/documents/2024/12/04/2024-28361/marketing-submission-recommendations-for-a-predetermined-change-control-plan-for-artificial

3. AI-Enabled Device Software Functions (January 7, 2025)
   - https://www.federalregister.gov/documents/2025/01/07/2024-31543/artificial-intelligence-enabled-device-software-functions-lifecycle-management-and-marketing

---

## 8. SUMMARY OF KEY REGULATORY REQUIREMENTS

### 8.1 For Clinical Decision Support Software (Non-AI)

**To qualify as Non-Device CDS, software MUST:**
1. ✓ NOT process medical images or diagnostic signals
2. ✓ Display medical information normally shared between HCPs
3. ✓ Provide recommendations (multiple options), NOT directives
4. ✓ Enable independent review of recommendation basis
5. ✓ Target healthcare professionals, NOT patients
6. ✓ NOT be intended for time-critical decision-making

**If ANY criterion fails → Software is a medical device requiring FDA review**

### 8.2 For AI/ML-Enabled Medical Devices

**Manufacturers must consider:**
1. **Premarket Pathway:** 510(k), De Novo, or PMA
2. **GMLP Principles:** All 10 principles for development
3. **Transparency Requirements:** Clear disclosure of algorithm basis
4. **PCCP Option:** Pre-authorized modification plan for adaptive systems
5. **Lifecycle Management:** Comprehensive TPLC approach (per January 2025 draft guidance)
6. **Data Management:** Training, tuning, and testing dataset separation
7. **Post-Market Monitoring:** Performance monitoring and re-training risk management
8. **Bias Mitigation:** Representative data and testing populations

### 8.3 Documentation Requirements (AI/ML Devices)

**Marketing Submissions should include:**
- Algorithm design and development methodology
- Validation and testing data (training, tuning, testing datasets)
- Clinical evidence supporting safety and effectiveness
- Transparency documentation (algorithm basis, limitations)
- Risk management and mitigation strategies
- Post-deployment monitoring plan
- PCCP (if applicable) with modification methodology

**Source:** [FDA Draft Guidance - AI-Enabled Device Software Functions (January 2025)](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/artificial-intelligence-enabled-device-software-functions-lifecycle-management-and-marketing)
**Accessed:** November 30, 2025

---

## 9. IMPLICATIONS FOR ACUTE CARE HYBRID REASONING SYSTEMS

### 9.1 Regulatory Classification Analysis

For a hybrid reasoning system in acute care combining ontological reasoning and graph neural networks:

**Likely Device Classification Triggers:**
- ❌ **Time-critical use case** (acute care) → Fails Criterion 3
- ❌ **Risk scoring for individual patients** → Likely directive, not recommendation
- ❌ **Clinical urgency may lead to primary reliance** → Fails Criterion 4

**Potential Device Classification:**
- Most likely classified as **medical device** requiring premarket review
- May qualify for **510(k), De Novo, or PMA** pathway depending on risk level
- Should consider **PCCP** for model updates and retraining

### 9.2 Regulatory Strategy Recommendations

#### **Option 1: Medical Device Pathway (Most Likely)**
1. Determine predicate device or risk classification
2. Select appropriate pathway (510(k), De Novo, PMA)
3. Develop comprehensive GMLP-compliant development process
4. Consider PCCP for model evolution
5. Implement lifecycle management per January 2025 draft guidance
6. Plan post-market surveillance

#### **Option 2: Non-Device CDS (Challenging)**
To potentially qualify as Non-Device CDS, system would need to:
1. Explicitly NOT be intended for time-critical decision-making
2. Provide multiple recommendation options (not single risk score)
3. Enable transparent review of ontological + GNN reasoning basis
4. Target non-urgent clinical support scenarios
5. Clearly position as decision support, not decision automation

**Caution:** Given acute care context, Non-Device CDS classification is likely very difficult to achieve under current FDA interpretation.

### 9.3 Development Recommendations

Regardless of pathway, follow GMLP principles:
1. Multi-disciplinary team (clinical, ML, ontology experts)
2. Representative training data reflecting acute care populations
3. Independent test datasets
4. Transparent hybrid reasoning process
5. Clinically relevant testing protocols
6. Clear user information and limitations
7. Post-deployment performance monitoring

---

## 10. DOCUMENT METADATA

### 10.1 Compilation Information

- **Document Created:** November 30, 2025
- **Primary Author:** Research compilation from official FDA sources
- **Purpose:** Regulatory guidance reference for hybrid reasoning acute care system development
- **Version:** 1.0

### 10.2 Source Authentication

All information in this document is sourced from:
- Official FDA.gov websites
- Federal Register notices
- Published FDA guidance documents
- FDA press releases and announcements

**Access Dates:** All sources accessed November 30, 2025

### 10.3 Currency and Updates

**Most Recent Guidance:**
- **Draft:** AI-Enabled Device Software Functions (January 2025) - Comment period through April 7, 2025
- **Final:** PCCP for AI-Enabled Devices (December 2024)
- **Final:** Clinical Decision Support Software (September 2022)

**Upcoming Events:**
- February 18, 2025: FDA webinar on AI-Enabled Device Software Functions draft guidance
- April 7, 2025: End of comment period for January 2025 draft guidance
- November 2025: Next Digital Health Advisory Committee meeting

### 10.4 Limitations and Disclaimers

This document:
- Compiles publicly available FDA guidance as of November 30, 2025
- Does not constitute legal or regulatory advice
- Should be supplemented with consultation from FDA regulatory experts
- May not reflect unpublished FDA positions or enforcement priorities
- Will require updates as new guidance is finalized

For specific regulatory questions, manufacturers should:
1. Consult with FDA through Pre-Submission (Q-Submission) program
2. Engage qualified regulatory consultants
3. Monitor FDA website for guidance updates
4. Participate in FDA public comment periods
5. Attend FDA workshops and webinars

---

## END OF DOCUMENT

**Next Review Date:** April 2025 (following finalization of AI-Enabled Device Software Functions guidance)

---

## Complete Source List

### Primary FDA Sources
1. [Clinical Decision Support Software Guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software)
2. [FDA CDS FAQs](https://www.fda.gov/medical-devices/software-medical-device-samd/clinical-decision-support-software-frequently-asked-questions-faqs)
3. [FDA Step 6 Decision Framework](https://www.fda.gov/medical-devices/digital-health-center-excellence/step-6-software-function-intended-provide-clinical-decision-support)
4. [AI-Enabled Device Software Functions Draft Guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/artificial-intelligence-enabled-device-software-functions-lifecycle-management-and-marketing)
5. [PCCP for AI-Enabled Devices Final Guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence)
6. [Good Machine Learning Practice](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles)
7. [FDA AI/ML Overview](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-software-medical-device)
8. [21st Century Cures Act](https://www.fda.gov/regulatory-information/selected-amendments-fdc-act/21st-century-cures-act)

### Federal Register
1. [CDS Guidance Federal Register Notice - September 28, 2022](https://www.federalregister.gov/documents/2022/09/28/2022-20993/clinical-decision-support-software-guidance-for-industry-and-food-and-drug-administration-staff)
2. [PCCP Federal Register Notice - December 4, 2024](https://www.federalregister.gov/documents/2024/12/04/2024-28361/marketing-submission-recommendations-for-a-predetermined-change-control-plan-for-artificial)
3. [AI-DSF Federal Register Notice - January 7, 2025](https://www.federalregister.gov/documents/2025/01/07/2024-31543/artificial-intelligence-enabled-device-software-functions-lifecycle-management-and-marketing)

### News and Analysis
1. [Bipartisan Policy Center - FDA AI Oversight](https://bipartisanpolicy.org/issue-brief/fda-oversight-understanding-the-regulation-of-health-ai-tools/)
2. [Healthcare IT News - Congressional Letter](https://www.healthcareitnews.com/news/lawmakers-ask-cdrh-revisit-its-cds-guidance)

### Legal Analysis
1. [FDA Law Blog - Shrinking Exemption Analysis](https://www.thefdalawblog.com/2022/10/the-incredible-shrinking-exemption-fda-final-cds-guidance-would-significantly-narrow-the-scope-of-exempt-clinical-decision-support-software-under-the-cures-act/)
2. [Nixon Law Group - CDS Guidance Analysis](https://nixongwiltlaw.com/nlg-blog/2022/10/10/fda-issues-final-guidance-on-clinical-decision-support-software-and-software-as-a-medical-device-key-takeaways-and-what-it-means-for-digital-health-companies)
3. [Crowell & Moring - Section 520(o) Analysis](https://www.crowell.com/en/insights/client-alerts/fda-issues-final-guidance-on-clinical-decision-support-software)

All sources accessed November 30, 2025.
