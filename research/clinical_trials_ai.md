# Clinical Trials Involving AI/ML in Healthcare: Research Summary

## Executive Summary

This document summarizes key findings from clinical trials and studies involving artificial intelligence and machine learning in healthcare settings. The research covers AI/ML applications in clinical decision support systems, sepsis prediction, and EHR-based predictive modeling.

---

## 1. Overview of AI/ML Clinical Trials on ClinicalTrials.gov

### Registration Trends (2010-2023)

**Study**: Cross-sectional analysis of AI/ML studies on ClinicalTrials.gov

**Key Statistics**:
- **Total AI/ML studies registered**: 3,106 (January 2010 - December 2023)
- **Results reporting rate**: Only 5.6% of completed studies reported results
- **Geographic distribution**: Predominantly high-income countries, with modest increase in upper-middle-income countries (primarily China)
- **Underrepresentation**: Lower-middle-income and low-income countries remain poorly represented

**Challenges Identified**:
- Recruitment delays affecting 80% of studies
- Escalating costs exceeding $200 billion annually in pharmaceutical R&D
- Success rates below 12%
- Data quality issues affecting 50% of datasets

**Source**: [Studies of AI/ML Registered on ClinicalTrials.gov (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11549584/)

---

### NIH TrialGPT Algorithm

**Study**: AI algorithm for matching patients to clinical trials

**Design**: Algorithm development and validation study

**Purpose**: Speed up the process of matching potential volunteers to relevant clinical research trials on ClinicalTrials.gov

**Methodology**:
1. Processes patient summary containing medical and demographic information
2. Identifies relevant clinical trials from ClinicalTrials.gov
3. Determines patient eligibility and exclusions
4. Explains how patient meets enrollment criteria
5. Produces annotated list of trials ranked by relevance and eligibility

**Publication**: Nature Communications

**Source**: [NIH-developed AI algorithm (NIH)](https://www.nih.gov/news-events/news-releases/nih-developed-ai-algorithm-matches-potential-volunteers-clinical-trials)

---

### AI Applications in Clinical Trial Risk Assessment

**Study**: Scoping review of AI in clinical trial risk assessment

**Period**: 2013-2024

**Studies Analyzed**: 142 studies

**Risk Categories**:
- Safety prediction: n=55
- Efficacy prediction: n=46
- Operational risk prediction: n=45

**Source**: [AI in Clinical Trial Risk Assessment (Nature)](https://www.nature.com/articles/s41746-025-01886-7)

---

## 2. Clinical Decision Support System (CDSS) Trials

### 2.1 CDS Design Best Practices Study (2021)

**Trial ID**: Not specified

**Design**: Cluster randomized controlled trial

**Setting**: 28 primary care clinics

**Intervention**: Comparison of IS-enhanced CDS alert vs. commercial CDS alert

**Target**: Beta-blocker prescribing for heart failure

**Framework**: Enhanced alert informed by CDS best practices and Practical, Robust, Implementation, and Sustainability Model (PRISM)

**Source**: [JMIR Medical Informatics Study](https://medinform.jmir.org/2021/3/e24359)

---

### 2.2 Integrated Clinical Prediction Rule (iCPR) Trial

**Design**: Randomized clinical trial (RCT)

**Purpose**: Determine if clinical prediction rules (CPRs) could be efficiently integrated into workflow within commercial EHR

**Primary Outcome**: Antibiotic ordering changes

**Results**:
- **Adoption rate**: 57.5% of intervention users
- **Acceptance rate**: 42.4%

**Source**: [iCPR Framework Study (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4537146/)

---

### 2.3 Pediatric Fever Management CDSS

**Design**: Clinical trial

**Setting**: Emergency Department (ED)

**Population**: Children 1-36 months with fever without apparent source

**Intervention**: CDSS used by ED nursing staff to register children and provide patient-specific diagnostic management advice

**Outcomes Measured**:
- Compliance with CDSS
- Time spent in ED
- Number of laboratory tests

**Source**: [Pediatric Fever CDSS Trial (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2273109/)

---

### 2.4 Medical Imaging CDSS Trial (2025)

**Publication**: JAMA

**Design**: Cluster randomized clinical trial

**Intervention**: Clinical decision support system for medical imaging ordering

**Comparison**: CDS system vs. no CDS system

**Primary Outcome**: Appropriateness of medical imaging ordering behavior

**Source**: [JAMA Imaging Study](https://jamanetwork.com/journals/jama/fullarticle/2830127)

---

### 2.5 Problem List Completeness Multi-Site Trial (2023)

**Design**: Multi-site cluster randomized trial

**Sample Size**: 288,832 opportunities to add a problem in intervention arm

**Conditions Targeted**: 12 clinically significant heart, lung, and blood diseases

**Intervention**: CDS that suggests adding missing problems to EHR problem list using inference algorithms

**Results**:
- **Problems added in intervention arm**: 63,777 times
- **Acceptance rate**: 22.1%
- **Comparative effectiveness**: 4.6 times as many problems added vs. control arm

**Source**: [Problem List Completeness Trial (Oxford Academic)](https://academic.oup.com/jamia/article/30/5/899/7048709)

---

### 2.6 Antimicrobial Management CDSS Trial

**Design**: Randomized controlled trial

**Intervention**: Computerized CDS system for antimicrobial utilization management

**Purpose**: Alert antimicrobial management team of potentially inadequate antimicrobial therapy

**Results**:
- **Hospital antimicrobial expenditure savings**: $84,194 (23% reduction)
- **Per-patient savings**: $37.64 in intervention arm

**Source**: [JAMA Network Open CDSS Study](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2757368)

---

### 2.7 LIGHT Hypertension Trial (China)

**Trial Name**: Learning Implementation of Guideline-based decision support system for Hypertension Treatment (LIGHT)

**Design**: Pragmatic, four-stage, cluster-randomized trial

**Setting**: 94 primary care sites in China

**Significance**: Largest pragmatic randomized trial exploring feasibility and effectiveness of decision support tool for hypertension in primary care

**Source**: [LIGHT Trial Protocol](https://trialsjournal.biomedcentral.com/articles/10.1186/s13063-022-06374-x)

---

## 3. Sepsis AI Clinical Trials

### 3.1 NAVOY Sepsis Algorithm - Pivotal Trial

**Design**: Randomized clinical trial (largest RCT for ML sepsis prediction algorithm to date)

**Intervention**: NAVOY Sepsis algorithm using 4 hours of routinely collected clinical data

**Primary Outcome**: Sepsis prediction up to 3 hours before onset

**Performance Metrics**:
- **Accuracy**: 0.79
- **Sensitivity**: 0.80 (identifying 80% of patients who would develop sepsis)
- **Specificity**: 0.78

**Regulatory Status**: FDA-authorized

**Source**: [FDA-Authorized AI/ML Tool for Sepsis (NEJM AI)](https://ai.nejm.org/doi/full/10.1056/AIoa2400867)

---

### 3.2 Deep Learning Sepsis Prediction - Multi-Hospital Study

**Design**: Prospective clinical outcomes evaluation

**Setting**: 9 hospitals

**Duration**: 2 years

**Intervention**: AI algorithm for sepsis prediction

**Results**:
- **In-hospital mortality reduction**: 39.50%
- **Length of stay reduction**: 32.27%
- **30-day readmission reduction**: 22.74%

**Source**: [Impact of Deep Learning Sepsis Model (npj Digital Medicine)](https://www.nature.com/articles/s41746-023-00986-6)

---

### 3.3 Shimabukuro et al. ICU Sepsis Prediction Trial

**Design**: Small randomized trial

**Setting**: Intensive Care Unit (ICU)

**Sample Size**: 142 patients

**Intervention**: Machine learning algorithm to predict severe sepsis

**Results**:
- Decrease in in-hospital mortality (intervention group)
- Decrease in length of stay (intervention group)

**Source**: [AI in Sepsis Management Overview (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11722371/)

---

### 3.4 Sepsis ImmunoScore - FDA-Authorized Tool

**Regulatory Status**: FDA-authorized AI/ML tool

**Design**: Validation study with external validation

**Method**: Bayesian approach for EMR integration

**Performance Metrics**:
- **Diagnostic performance (AUC)**: 0.84
- **Prognostic performance (AUC)**: 0.76

**Purpose**: Adjunct to clinical decision-making

**Source**: [FDA-Authorized Sepsis Tool (NEJM AI)](https://ai.nejm.org/doi/full/10.1056/AIoa2400867)

---

### 3.5 Epic Sepsis Model (ESM) Validation Study (2021)

**Study**: Wong and colleagues validation

**Design**: External validation study

**Implementation**: Hundreds of hospitals

**Results**:
- **Actual AUC**: 0.63 (poor discrimination)
- **Initially reported AUC**: 0.76-0.83
- **Conclusion**: Performance far worse than initially reported

**Key Finding**: High incidence of false positives, lack of external validation

**Source**: [AI for Early Sepsis Detection Caution (AJRCCM)](https://www.atsjournals.org/doi/10.1164/rccm.202212-2284VP)

---

### 3.6 MLASA Sepsis Alert System - Cluster RCT

**Design**: Cluster-randomized trial

**Intervention**: Machine Learning Alert for Sepsis Antibiotics (MLASA) system

**Process Measure Results**:
- Increased prompt antibiotic administration

**Clinical Outcome Results**:
- No significant effect on 30-day mortality
- No significant effect on length of stay

**Key Finding**: Improved process measures but not patient-centered outcomes

**Source**: [AI in Sepsis Management (MDPI)](https://www.mdpi.com/2077-0383/14/1/286)

---

## 4. EHR-Based Machine Learning Prospective Studies

### 4.1 Incident Hypertension Prediction - Maine HIE Study (2018)

**Design**: Prospective validation study

**Data Source**: Maine Health Information Exchange network

**Sample Sizes**:
- **Retrospective cohort**: N=823,627 (2013)
- **Prospective cohort**: N=680,810 (2014)

**Algorithm**: XGBoost (machine learning)

**Prediction Target**: 1-year incident essential hypertension risk

**Performance Metrics**:
- **Retrospective AUC**: 0.917
- **Prospective AUC**: 0.870

**Source**: [Hypertension Prediction Using EHR and ML (PubMed)](https://pubmed.ncbi.nlm.nih.gov/29382633/)

---

### 4.2 Dysphagia Prediction Tool - Prospective Cohort Study (2023)

**Design**: Prospective observational cohort study

**Data Source**: Pre-existing documented EHR data

**Intervention**: ML-based prediction algorithm for dysphagia and aspiration pneumonia

**Significance**: First study to evaluate ML-based dysphagia prediction tool in prospective clinical setting

**Source**: [ML-Based Dysphagia Prediction (PubMed)](https://pubmed.ncbi.nlm.nih.gov/36625964/)

---

### 4.3 Multi-Disease Deep Learning Prediction

**Design**: Deep learning study using EHR data

**Target Conditions**: Diabetes, COPD, Hypertension, Myocardial Infarction (MI)

**Methodology**: Included binned observations and wider determinants of health

**Performance Metrics (AUC)**:
- **Diabetes**: 0.92
- **COPD**: 0.94
- **Hypertension**: 0.92
- **Myocardial Infarction**: 0.94

**Key Finding**: Increasing data scope improved predictive performance

**Source**: [Disease Prediction Using Deep Learning (Frontiers)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1287541/full)

---

### 4.4 Alzheimer's Disease and Related Dementias (ADRD) Prediction (2024)

**Design**: ML model development and validation study

**Population**: Over 6 million patients affected by ADRD in the United States

**Data Source**: De-identified EHR data

**Purpose**: Early diagnosis and prediction of ADRD

**Source**: [ADRD Prediction Using EHR and ML (medRxiv)](https://www.medrxiv.org/content/10.1101/2024.12.09.24318740v1)

---

## 5. Key Findings and Trends

### Positive Outcomes

1. **High Predictive Performance**: Several studies demonstrated AUCs >0.85 for disease prediction
2. **Process Improvements**: Increased compliance with clinical guidelines and faster treatment initiation
3. **Cost Savings**: Demonstrated reduction in antimicrobial expenditures (23% in one study)
4. **Clinical Outcomes**: Some sepsis studies showed significant reductions in mortality (up to 39.5%)
5. **High Adoption**: Some CDS interventions achieved >50% adoption rates among clinicians

### Challenges and Limitations

1. **Poor Results Reporting**: Only 5.6% of completed AI/ML trials reported results
2. **External Validation Failures**: Some widely-implemented algorithms (e.g., Epic Sepsis Model) showed poor performance in validation
3. **Process vs. Outcome Gap**: Improved process measures don't always translate to improved patient outcomes
4. **High False Positive Rates**: Alert fatigue from excessive false positives
5. **Geographic Disparities**: Limited representation from low and middle-income countries
6. **Inconsistent Mortality Benefits**: While some studies show dramatic reductions, others show no significant effect

### Study Design Considerations

1. **Need for Large-Scale RCTs**: Researchers emphasize need for high-quality, system-wide trials following CONSORT-AI standards
2. **Prospective Validation**: Critical to validate retrospectively-developed models prospectively
3. **Implementation Science**: Importance of frameworks like PRISM for successful CDS implementation
4. **Multi-Site Studies**: Increasing recognition of need for multi-site validation

---

## 6. Common ML/AI Approaches in Clinical Trials

### Algorithms and Methods

- **XGBoost**: Used for feature selection and model building
- **Recurrent Neural Networks (RNNs)**: Modeling long-term temporal dependencies
- **Time-aware Attention Mechanisms**: Handling sequential EHR data
- **Self-Attention and Transformers**: Representing inner-visit relations
- **Convolutional Neural Networks (CNNs)**: Pattern recognition in clinical data
- **Graph Neural Networks**: Representing complex clinical relationships
- **Bayesian Approaches**: Probabilistic decision support (e.g., Sepsis ImmunoScore)

### Data Sources

- Electronic Health Records (EHR)
- Health Information Exchanges (HIE)
- Routinely collected vital signs and lab values
- Clinical notes (structured and unstructured data)
- Temporal trajectories of patient data

---

## 7. Recommendations for Future Research

1. **Standardized Reporting**: Adherence to CONSORT-AI guidelines for AI/ML clinical trials
2. **External Validation**: Mandatory external validation before widespread implementation
3. **Patient-Centered Outcomes**: Focus on mortality, length of stay, and quality of life, not just process measures
4. **Implementation Science Integration**: Use established frameworks (e.g., PRISM) to optimize real-world deployment
5. **Health Equity**: Increase representation from diverse geographic and socioeconomic settings
6. **Results Transparency**: Improve results reporting rates from current 5.6%
7. **Alert Optimization**: Balance sensitivity and specificity to minimize alert fatigue
8. **Long-Term Follow-Up**: Evaluate sustained effects beyond immediate outcomes

---

## Sources

### AI/ML Clinical Trials Overview
- [Studies of AI/ML on ClinicalTrials.gov (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11549584/)
- [NIH TrialGPT Algorithm](https://www.nih.gov/news-events/news-releases/nih-developed-ai-algorithm-matches-potential-volunteers-clinical-trials)
- [AI in Clinical Trial Risk Assessment (Nature)](https://www.nature.com/articles/s41746-025-01886-7)
- [Public Disclosure of AI/ML Research Results (JMIR)](https://www.jmir.org/2025/1/e60148)
- [AI in Clinical Trials Review (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1386505625003582)

### Clinical Decision Support Trials
- [CDS Design Best Practices RCT (JMIR)](https://medinform.jmir.org/2021/3/e24359)
- [iCPR Framework Trial (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4537146/)
- [Pediatric Fever CDSS (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2273109/)
- [Medical Imaging CDSS Trial (JAMA)](https://jamanetwork.com/journals/jama/fullarticle/2830127)
- [Problem List Completeness Trial (Oxford Academic)](https://academic.oup.com/jamia/article/30/5/899/7048709)
- [LIGHT Hypertension Trial (Trials)](https://trialsjournal.biomedcentral.com/articles/10.1186/s13063-022-06374-x)
- [Hospital-Based CDSS Trial (JAMA Network Open)](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2757368)

### Sepsis AI Trials
- [FDA-Authorized Sepsis Tool (NEJM AI)](https://ai.nejm.org/doi/full/10.1056/AIoa2400867)
- [AI for Sepsis Prediction Review (NCBI Bookshelf)](https://www.ncbi.nlm.nih.gov/books/NBK596676/)
- [AI in Sepsis Management (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11722371/)
- [Deep Learning Sepsis Model Impact (npj Digital Medicine)](https://www.nature.com/articles/s41746-023-00986-6)
- [AI Early Sepsis Detection Caution (AJRCCM)](https://www.atsjournals.org/doi/10.1164/rccm.202212-2284VP)
- [AI-Driven Sepsis Detection (JMIR)](https://www.jmir.org/2025/1/e56155)
- [AI in Sepsis Using Unstructured Data (Nature Communications)](https://www.nature.com/articles/s41467-021-20910-4)

### EHR-Based ML Studies
- [Hypertension Prediction Prospective Study (PubMed)](https://pubmed.ncbi.nlm.nih.gov/29382633/)
- [Dysphagia Prediction Tool (PubMed)](https://pubmed.ncbi.nlm.nih.gov/36625964/)
- [Deep Learning EHR Trajectories Review (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S153204642300151X)
- [Disease Prediction Using Deep Learning (Frontiers)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1287541/full)
- [ML and DL for Early Disease Detection (JMIR)](https://www.jmir.org/2024/1/e48320)
- [ADRD Prediction Using ML (medRxiv)](https://www.medrxiv.org/content/10.1101/2024.12.09.24318740v1)
- [EHR-ML Framework (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1386505625000334)

---

**Document Created**: 2025-11-30
**Location**: /Users/alexstinard/hybrid-reasoning-acute-care/research/clinical_trials_ai.md
