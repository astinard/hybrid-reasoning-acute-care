# MIMIC-IV Dataset Details for Clinical AI Research

**Last Updated:** November 30, 2025

## Executive Summary

MIMIC-IV (Medical Information Mart for Intensive Care IV) is a freely accessible, deidentified electronic health record (EHR) database sourced from Beth Israel Deaconess Medical Center in Boston, MA. It contains comprehensive clinical data for critical care and emergency department patients, making it an invaluable resource for clinical AI research, particularly for sepsis prediction, mortality forecasting, and other acute care applications.

---

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Database Schema Structure](#database-schema-structure)
3. [MIMIC-IV-ED Emergency Department Module](#mimic-iv-ed-emergency-department-module)
4. [Access Requirements](#access-requirements)
5. [Sepsis Research Applications](#sepsis-research-applications)
6. [Common Research Tasks](#common-research-tasks)
7. [Code Examples](#code-examples)
8. [References](#references)

---

## Dataset Overview

### Patient Statistics (v3.1)

| Metric | Count | Description |
|--------|-------|-------------|
| **Unique Patients** | 364,627 | Individual patients (subject_id) |
| **Hospitalizations** | 546,028 | Total hospital admissions (hadm_id) |
| **ICU Stays** | 94,458 | Unique ICU admissions (stay_id) |
| **ED Visits** | 422,500 | Emergency department stays (2011-2019) |
| **ICU Patients** | 50,934 | Unique patients with ICU stays |
| **Total ICU Stays** | 73,141 | All ICU stays across all patients |

**Key Statistics:**
- Only 17% of hospital patients required ICU admission
- Coverage period: 2008-2022 (inclusive)
- Over 65,000 ICU admissions and 200,000 ED admissions available

### Data Source

- **Institution:** Beth Israel Deaconess Medical Center (tertiary academic medical center)
- **Location:** Boston, MA
- **Time Period:** 2008-2022
- **Latest Version:** MIMIC-IV v3.1 (released July 23, 2024)

---

## Database Schema Structure

### Module Organization

MIMIC-IV adopts a **relational structure** with data organized into three primary modules:

| Module | Description | Data Source |
|--------|-------------|-------------|
| **hosp** | Hospital-wide data | Hospital EHR system |
| **icu** | ICU-specific data | MetaVision clinical information system |
| **note** | Clinical notes | Deidentified free-text documentation |

### Key Identifiers

| Identifier | Description | Scope |
|------------|-------------|-------|
| `subject_id` | Unique patient identifier | All modules |
| `hadm_id` | Hospital admission identifier | Hospital stays |
| `stay_id` | ICU stay identifier (ICU module) | ICU stays |
| `stay_id` | ED stay identifier (ED module) | ED visits |

**Deidentification:**
- All identifiers are randomly assigned
- Dates shifted using patient-level offset
- Protected health information (PHI) removed

---

## Hosp Module

### Core Tables

| Table Name | Description | Key Fields |
|------------|-------------|------------|
| `patients` | Patient demographics and vital status | subject_id, gender, anchor_age, anchor_year, dod |
| `admissions` | Hospital admission records | hadm_id, subject_id, admittime, dischtime, admission_type, insurance |
| `transfers` | Patient location transfers | hadm_id, transfer_id, eventtype, careunit, intime, outtime |

### Clinical Data Tables

| Table Name | Description | Key Fields |
|------------|-------------|------------|
| `labevents` | Laboratory measurements | subject_id, hadm_id, itemid, charttime, value, valuenum |
| `microbiologyevents` | Microbiology cultures and sensitivities | subject_id, hadm_id, charttime, spec_type_desc, org_name |
| `prescriptions` | Medication prescriptions | subject_id, hadm_id, drug, dose_val_rx, route |
| `emar` | Electronic medication administration | subject_id, hadm_id, medication, event_txt, scheduletime |
| `diagnoses_icd` | ICD diagnosis codes | subject_id, hadm_id, icd_code, icd_version, seq_num |
| `procedures_icd` | ICD procedure codes | subject_id, hadm_id, icd_code, icd_version, seq_num |

### Administrative Tables

| Table Name | Description | Key Fields |
|------------|-------------|------------|
| `d_labitems` | Lab item dictionary | itemid, label, fluid, category |
| `d_icd_diagnoses` | ICD diagnosis code reference | icd_code, icd_version, long_title |
| `d_icd_procedures` | ICD procedure code reference | icd_code, icd_version, long_title |

**Schema Pattern:**
- All tables contain `subject_id` for patient linkage
- Dimension tables prefixed with `d_`
- Time-series data includes timestamp columns

---

## ICU Module

### Star Schema Design

The ICU module uses a **star schema** centered around two core tables:
- `icustays` - ICU stay information
- `d_items` - Item definitions for all charted data

### Core Tables

| Table Name | Description | Key Fields |
|------------|-------------|------------|
| `icustays` | ICU stay details | stay_id, subject_id, hadm_id, first_careunit, intime, outtime, los |
| `d_items` | Definition of charted items | itemid, label, abbreviation, linksto, category, unitname |

### Event Tables

All event tables follow a consistent structure with `stay_id` and `itemid` as key linking fields:

| Table Name | Description | Key Fields |
|------------|-------------|------------|
| `chartevents` | Vital signs, scores, assessments | stay_id, itemid, charttime, value, valuenum, valueuom |
| `inputevents` | IV fluids, medications, nutrition | stay_id, itemid, starttime, endtime, amount, rate |
| `outputevents` | Patient outputs (urine, drains, etc.) | stay_id, itemid, charttime, value |
| `procedureevents` | ICU procedures | stay_id, itemid, starttime, endtime, value |
| `datetimeevents` | Date/time based events | stay_id, itemid, charttime, value |
| `ingredientevents` | Ingredients of inputevents | stay_id, itemid, starttime, endtime, amount |

**Key Design Features:**
- Denormalized event tables for efficient querying
- All events link to `d_items` for item definitions
- All events link to `icustays` for patient context
- Timestamps allow temporal analysis

---

## MIMIC-IV-ED Emergency Department Module

### Overview

MIMIC-IV-ED is a **standalone linkable module** containing emergency department data from 2011-2019.

**Statistics:**
- **422,500** ED stays
- Linkable to MIMIC-IV via `subject_id`
- Linkable to MIMIC-CXR for chest X-ray images

### Schema Structure

MIMIC-IV-ED follows a **star-like structure** centered around the `edstays` table.

### Tables

| Table Name | Description | Key Fields | Row Count (approx) |
|------------|-------------|------------|-------------------|
| `edstays` | ED stay information | subject_id, stay_id, hadm_id, intime, outtime, arrival_transport | 422,500 |
| `diagnosis` | Discharge diagnoses (ICD-9/10) | subject_id, stay_id, icd_code, icd_version, seq_num | Variable (max 9 per stay) |
| `triage` | Triage assessments | subject_id, stay_id, temperature, heartrate, resprate, o2sat, sbp, dbp, pain, acuity | 422,500 |
| `vitalsign` | Vital signs during ED stay | subject_id, stay_id, charttime, temperature, heartrate, resprate, o2sat, sbp, dbp | Variable |
| `medrecon` | Medications prior to ED | subject_id, stay_id, charttime, name, gsn | Variable |
| `pyxis` | Medications dispensed in ED | subject_id, stay_id, charttime, name, gsn_rn | Variable |

### Diagnosis Table Details

**ICD Coding:**
- Both ICD-9 and ICD-10 codes available
- Maximum of **9 diagnosis codes** per stay
- `seq_num` indicates relevance (1 = highest, 9 = lowest)
- Codes assigned post-discharge for billing

**Example Query:**
```sql
SELECT
    e.subject_id,
    e.stay_id,
    d.seq_num,
    d.icd_code,
    d.icd_version,
    d.icd_title
FROM mimiciv_ed.edstays e
JOIN mimiciv_ed.diagnosis d ON e.stay_id = d.stay_id
WHERE d.icd_code LIKE 'A41%'  -- Sepsis codes
ORDER BY e.subject_id, d.seq_num;
```

### Linking to MIMIC-IV

**Key Linkage Points:**
- `subject_id`: Links to same patient across all MIMIC datasets
- ED stays appear in MIMIC-IV `transfers` table
- Can access:
  - Patient demographics from `patients` table
  - Lab values from `labevents` table
  - Medications from `prescriptions` table
  - Subsequent ICU data if patient admitted

**Example Cross-Module Query:**
```sql
-- Find ED patients who were admitted to ICU
SELECT
    ed.subject_id,
    ed.stay_id AS ed_stay_id,
    icu.stay_id AS icu_stay_id,
    ed.intime AS ed_arrival,
    icu.intime AS icu_admission,
    EXTRACT(EPOCH FROM (icu.intime - ed.intime))/3600 AS hours_ed_to_icu
FROM mimiciv_ed.edstays ed
JOIN mimiciv_icu.icustays icu ON ed.subject_id = icu.subject_id
WHERE icu.intime > ed.intime
    AND icu.intime < ed.outtime + INTERVAL '24 hours';
```

---

## Access Requirements

### PhysioNet Credentialing Process

MIMIC-IV is a **restricted-access resource** requiring credentialing. Access involves a **three-step process**:

#### Step 1: Submit Personal Details
- Create PhysioNet account
- Complete credentialing form
- Linking ORCID ID speeds up verification

#### Step 2: Complete Required Training
- **Course:** CITI "Data or Specimens Only Research"
- Focus on human subjects research ethics
- Upload training completion certificate

#### Step 3: Sign Data Use Agreement (DUA)
- **License:** PhysioNet Credentialed Health Data License 1.5.0
- **Agreement:** PhysioNet Credentialed Health Data Use Agreement 1.5.0
- Available in "Files" section after credentialing approval

### Existing MIMIC-III Users

If you have **existing MIMIC-III access:**
- Automatic approval after signing DUA
- No need to repeat training
- Simplified process

### Cloud Access (Recommended)

**Benefits:**
- Faster query performance
- No need to download large datasets
- Pre-configured database environments

**Platforms:**
- Google BigQuery
- AWS (Amazon Web Services)
- Must link PhysioNet account to cloud account

**BigQuery Access Example:**
```sql
-- Query structure for BigQuery
SELECT
    subject_id,
    hadm_id,
    admittime
FROM `physionet-data.mimiciv_hosp.admissions`
LIMIT 10;
```

---

## Sepsis Research Applications

### MIMIC-Sepsis Benchmark (2024)

A **curated sepsis cohort** derived from MIMIC-IV with standardized preprocessing:

**Statistics:**
- **35,239** ICU stays meeting Sepsis-3 criteria
- Time-aligned clinical features
- Standardized treatment data

**Features Included:**
- Vital signs
- Laboratory results
- Treatment interventions:
  - Vasopressors
  - Fluid administration
  - Mechanical ventilation
  - Antibiotic therapy

**Benchmark Tasks:**
1. Early mortality prediction
2. Length-of-stay estimation
3. Shock onset classification

**Key Advantages:**
- Transparent preprocessing pipeline
- Based on Sepsis-3 criteria (SOFA + infection)
- Structured imputation strategies
- Reproducible cohort definition

### Sepsis-3 Criteria

**Definition:**
- Suspected infection + SOFA score increase ≥2 points
- MIMIC-IV provides derived table based on Sepsis-3
- More clinically relevant than older definitions

### Notable Sepsis Studies Using MIMIC-IV

#### 1. Systemic Immune-Inflammation Index (SII)

**Study Cohort:** 16,007 sepsis patients

**Key Findings:**
- SII identified as prognostic biomarker
- Associated with mortality risk in critically ill sepsis patients
- Can aid risk stratification

**Reference:** [Association between admission systemic immune-inflammation index and mortality](https://link.springer.com/article/10.1007/s10238-023-01029-w)

#### 2. Blood Pressure Response Index (BPRI)

**Study Cohort:** 7,382 septic shock patients

**Key Findings:**
- Higher BPRI associated with reduced short-term mortality
- Each SD increase in BPRI led to:
  - 2.5% reduction in in-hospital mortality
  - 2.0% reduction in 28-day mortality
  - 1.8% reduction in 90-day mortality
- Potential prognostic tool for septic shock

**Reference:** [L-shaped association between BPRI and mortality](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0325110)

#### 3. Triglyceride-Glucose (TyG) Index

**Study Focus:** Insulin resistance marker in sepsis

**Key Findings:**
- Elevated TyG index associated with higher risk of:
  - In-hospital mortality
  - ICU mortality
- Novel biomarker for sepsis prognosis

**Reference:** [TyG index and mortality in sepsis](https://www.nature.com/articles/s41598-024-75050-8)

#### 4. Mean Arterial Pressure (MAP)

**Study Cohort:** 35,010 sepsis patients (2008-2019)

**Key Findings:**
- **U-shaped relationship** between MAP and mortality
- Optimal MAP range: **70-82 mmHg**
- Both high and low MAP increase 28-day mortality risk
- Clinical implication: Maintaining MAP in optimal range may improve outcomes

**Reference:** [MAP and mortality in sepsis - MIMIC-IV study](https://www.i-jmr.org/2025/1/e63291)

#### 5. Lymphocyte Counts

**Study Focus:** Immune response in sepsis

**Key Findings:**
- **U-shaped correlation** with hospital mortality
- Both lymphocytopenia and lymphocytosis affect survival
- Demonstrates dual functionality of lymphocytes in infection

**Reference:** [Lymphocyte count and sepsis mortality](https://intjem.biomedcentral.com/articles/10.1186/s12245-024-00682-6)

### Methodological Considerations

**Best Practices:**
1. Use **Sepsis-3 criteria** (SOFA + infection) for cohort definition
2. Leverage MIMIC-IV derived sepsis tables
3. Include comprehensive treatment data (antibiotics, fluids, vasopressors)
4. Account for temporal relationships in data
5. Consider time-series nature of clinical trajectories

**Common Limitations to Avoid:**
- Using outdated sepsis definitions (Sepsis-1, Sepsis-2)
- Ad-hoc data curation without standardization
- Ignoring antibiotic administration
- Focusing only on isolated aspects of care
- Not accounting for missing data patterns

---

## Common Research Tasks

### 1. Cohort Selection

**Sepsis Cohort Example:**
```sql
-- Select sepsis patients using Sepsis-3 criteria
-- Requires SOFA score ≥2 + suspected infection

WITH sofa_scores AS (
    SELECT
        stay_id,
        MAX(sofa_24hours) AS max_sofa
    FROM mimiciv_derived.sofa
    GROUP BY stay_id
),
suspected_infection AS (
    SELECT DISTINCT
        hadm_id,
        stay_id
    FROM mimiciv_hosp.microbiologyevents
    WHERE spec_type_desc IS NOT NULL
    UNION
    SELECT DISTINCT
        hadm_id,
        stay_id
    FROM mimiciv_icu.inputevents
    WHERE itemid IN (
        -- Antibiotic itemids
        225798, 225906, 225850, 225828
    )
)
SELECT
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.intime,
    i.outtime,
    s.max_sofa,
    a.hospital_expire_flag
FROM mimiciv_icu.icustays i
JOIN sofa_scores s ON i.stay_id = s.stay_id
JOIN suspected_infection si ON i.stay_id = si.stay_id
JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
WHERE s.max_sofa >= 2;
```

### 2. Feature Extraction

**Vital Signs (First 24 Hours):**
```sql
-- Extract vital signs from first 24 hours of ICU stay
SELECT
    ce.stay_id,
    ce.charttime,
    MAX(CASE WHEN di.label = 'Heart Rate' THEN ce.valuenum END) AS heart_rate,
    MAX(CASE WHEN di.label = 'Respiratory Rate' THEN ce.valuenum END) AS resp_rate,
    MAX(CASE WHEN di.label = 'O2 saturation pulseoxymetry' THEN ce.valuenum END) AS spo2,
    MAX(CASE WHEN di.label = 'Temperature Fahrenheit' THEN (ce.valuenum-32)*5/9 END) AS temperature_c,
    MAX(CASE WHEN di.label = 'Non Invasive Blood Pressure systolic' THEN ce.valuenum END) AS sbp,
    MAX(CASE WHEN di.label = 'Non Invasive Blood Pressure diastolic' THEN ce.valuenum END) AS dbp
FROM mimiciv_icu.chartevents ce
JOIN mimiciv_icu.d_items di ON ce.itemid = di.itemid
JOIN mimiciv_icu.icustays icu ON ce.stay_id = icu.stay_id
WHERE ce.charttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
    AND di.label IN (
        'Heart Rate',
        'Respiratory Rate',
        'O2 saturation pulseoxymetry',
        'Temperature Fahrenheit',
        'Non Invasive Blood Pressure systolic',
        'Non Invasive Blood Pressure diastolic'
    )
GROUP BY ce.stay_id, ce.charttime
ORDER BY ce.stay_id, ce.charttime;
```

**Laboratory Values:**
```sql
-- Extract key lab values for sepsis patients
SELECT
    le.subject_id,
    le.hadm_id,
    le.charttime,
    MAX(CASE WHEN di.label = 'Lactate' THEN le.valuenum END) AS lactate,
    MAX(CASE WHEN di.label = 'White Blood Cells' THEN le.valuenum END) AS wbc,
    MAX(CASE WHEN di.label = 'Platelet Count' THEN le.valuenum END) AS platelets,
    MAX(CASE WHEN di.label = 'Creatinine' THEN le.valuenum END) AS creatinine,
    MAX(CASE WHEN di.label = 'Bilirubin, Total' THEN le.valuenum END) AS bilirubin
FROM mimiciv_hosp.labevents le
JOIN mimiciv_hosp.d_labitems di ON le.itemid = di.itemid
WHERE di.label IN (
    'Lactate',
    'White Blood Cells',
    'Platelet Count',
    'Creatinine',
    'Bilirubin, Total'
)
GROUP BY le.subject_id, le.hadm_id, le.charttime
ORDER BY le.subject_id, le.charttime;
```

### 3. Outcome Extraction

**Mortality Outcomes:**
```sql
-- Extract multiple mortality outcomes
SELECT
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    a.hospital_expire_flag AS in_hospital_mortality,
    CASE
        WHEN p.dod IS NOT NULL
        AND p.dod <= a.dischtime + INTERVAL '28 days'
        THEN 1 ELSE 0
    END AS mortality_28day,
    CASE
        WHEN p.dod IS NOT NULL
        AND p.dod <= a.dischtime + INTERVAL '90 days'
        THEN 1 ELSE 0
    END AS mortality_90day,
    CASE
        WHEN p.dod IS NOT NULL
        AND p.dod <= a.dischtime + INTERVAL '1 year'
        THEN 1 ELSE 0
    END AS mortality_1year
FROM mimiciv_icu.icustays i
JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
JOIN mimiciv_hosp.patients p ON i.subject_id = p.subject_id;
```

### 4. Treatment Data

**Vasopressor Administration:**
```sql
-- Extract vasopressor use
SELECT
    ie.stay_id,
    ie.starttime,
    ie.endtime,
    ie.amount,
    ie.rate,
    ie.amountuom,
    ie.rateuom,
    di.label AS medication
FROM mimiciv_icu.inputevents ie
JOIN mimiciv_icu.d_items di ON ie.itemid = di.itemid
WHERE di.label IN (
    'Norepinephrine',
    'Epinephrine',
    'Dopamine',
    'Vasopressin',
    'Phenylephrine'
)
ORDER BY ie.stay_id, ie.starttime;
```

### 5. Clinical Scores

**SOFA Score (Sequential Organ Failure Assessment):**
```sql
-- Using derived SOFA table
SELECT
    stay_id,
    starttime,
    endtime,
    respiration_24hours,
    coagulation_24hours,
    liver_24hours,
    cardiovascular_24hours,
    cns_24hours,
    renal_24hours,
    sofa_24hours  -- Total SOFA score
FROM mimiciv_derived.sofa
WHERE sofa_24hours >= 2  -- Sepsis-3 threshold
ORDER BY stay_id, starttime;
```

---

## Code Examples

### Python: Loading MIMIC-IV Data

```python
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# Database connection (example with PostgreSQL)
engine = create_engine('postgresql://user:password@localhost:5432/mimic')

# Load admissions data
query = """
    SELECT
        subject_id,
        hadm_id,
        admittime,
        dischtime,
        admission_type,
        admission_location,
        discharge_location,
        insurance,
        ethnicity,
        hospital_expire_flag
    FROM mimiciv_hosp.admissions
    LIMIT 10000;
"""
admissions = pd.read_sql(query, engine)

# Load ICU stays
query_icu = """
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        first_careunit,
        last_careunit,
        intime,
        outtime,
        los
    FROM mimiciv_icu.icustays;
"""
icustays = pd.read_sql(query_icu, engine)

# Merge datasets
merged = pd.merge(admissions, icustays, on=['subject_id', 'hadm_id'], how='inner')
print(f"Combined dataset shape: {merged.shape}")
```

### Python: Feature Engineering for Sepsis Prediction

```python
import pandas as pd
import numpy as np
from datetime import timedelta

def extract_sepsis_features(stay_id, conn):
    """
    Extract features for sepsis prediction from MIMIC-IV

    Parameters:
    -----------
    stay_id : str
        ICU stay identifier
    conn : sqlalchemy.engine
        Database connection

    Returns:
    --------
    pd.DataFrame with engineered features
    """

    # Get ICU stay times
    stay_query = f"""
        SELECT intime, outtime
        FROM mimiciv_icu.icustays
        WHERE stay_id = '{stay_id}'
    """
    stay_info = pd.read_sql(stay_query, conn)
    intime = stay_info['intime'].iloc[0]

    # Define 24-hour window
    window_start = intime
    window_end = intime + timedelta(hours=24)

    # Extract vital signs
    vitals_query = f"""
        SELECT
            ce.charttime,
            di.label,
            ce.valuenum
        FROM mimiciv_icu.chartevents ce
        JOIN mimiciv_icu.d_items di ON ce.itemid = di.itemid
        WHERE ce.stay_id = '{stay_id}'
            AND ce.charttime BETWEEN '{window_start}' AND '{window_end}'
            AND di.label IN (
                'Heart Rate',
                'Respiratory Rate',
                'Temperature Fahrenheit',
                'O2 saturation pulseoxymetry'
            )
            AND ce.valuenum IS NOT NULL
    """
    vitals = pd.read_sql(vitals_query, conn)

    # Aggregate vital signs
    vital_features = {}
    for label in vitals['label'].unique():
        values = vitals[vitals['label'] == label]['valuenum']
        vital_features[f'{label}_mean'] = values.mean()
        vital_features[f'{label}_std'] = values.std()
        vital_features[f'{label}_min'] = values.min()
        vital_features[f'{label}_max'] = values.max()

    # Extract lab values
    labs_query = f"""
        SELECT
            le.charttime,
            di.label,
            le.valuenum
        FROM mimiciv_hosp.labevents le
        JOIN mimiciv_hosp.d_labitems di ON le.itemid = di.itemid
        JOIN mimiciv_icu.icustays icu ON le.subject_id = icu.subject_id
        WHERE icu.stay_id = '{stay_id}'
            AND le.charttime BETWEEN '{window_start}' AND '{window_end}'
            AND di.label IN (
                'Lactate',
                'White Blood Cells',
                'Creatinine',
                'Bilirubin, Total',
                'Platelet Count'
            )
            AND le.valuenum IS NOT NULL
    """
    labs = pd.read_sql(labs_query, conn)

    # Aggregate lab values (take first and last values)
    lab_features = {}
    for label in labs['label'].unique():
        values = labs[labs['label'] == label].sort_values('charttime')['valuenum']
        if len(values) > 0:
            lab_features[f'{label}_first'] = values.iloc[0]
            lab_features[f'{label}_last'] = values.iloc[-1]
            lab_features[f'{label}_max'] = values.max()

    # Extract SOFA score
    sofa_query = f"""
        SELECT
            sofa_24hours,
            respiration_24hours,
            coagulation_24hours,
            liver_24hours,
            cardiovascular_24hours,
            cns_24hours,
            renal_24hours
        FROM mimiciv_derived.sofa
        WHERE stay_id = '{stay_id}'
        ORDER BY starttime
        LIMIT 1
    """
    sofa = pd.read_sql(sofa_query, conn)
    sofa_features = sofa.iloc[0].to_dict() if len(sofa) > 0 else {}

    # Combine all features
    all_features = {
        'stay_id': stay_id,
        **vital_features,
        **lab_features,
        **sofa_features
    }

    return pd.DataFrame([all_features])

# Example usage
# features_df = extract_sepsis_features('30000001', engine)
```

### Python: Time-Series Data Processing

```python
import pandas as pd
import numpy as np

def create_time_series_dataset(stay_ids, conn, window_hours=24, step_hours=1):
    """
    Create time-series dataset with sliding windows

    Parameters:
    -----------
    stay_ids : list
        List of ICU stay identifiers
    conn : sqlalchemy.engine
        Database connection
    window_hours : int
        Length of observation window in hours
    step_hours : int
        Step size for sliding window in hours

    Returns:
    --------
    pd.DataFrame with time-series features
    """

    all_windows = []

    for stay_id in stay_ids:
        # Get stay information
        stay_query = f"""
            SELECT intime, outtime
            FROM mimiciv_icu.icustays
            WHERE stay_id = '{stay_id}'
        """
        stay_info = pd.read_sql(stay_query, conn)
        intime = pd.to_datetime(stay_info['intime'].iloc[0])
        outtime = pd.to_datetime(stay_info['outtime'].iloc[0])

        # Create sliding windows
        current_time = intime
        while current_time + pd.Timedelta(hours=window_hours) <= outtime:
            window_start = current_time
            window_end = current_time + pd.Timedelta(hours=window_hours)

            # Extract vitals for this window
            vitals_query = f"""
                SELECT
                    charttime,
                    MAX(CASE WHEN di.label = 'Heart Rate'
                        THEN ce.valuenum END) AS heart_rate,
                    MAX(CASE WHEN di.label = 'Respiratory Rate'
                        THEN ce.valuenum END) AS resp_rate,
                    MAX(CASE WHEN di.label = 'O2 saturation pulseoxymetry'
                        THEN ce.valuenum END) AS spo2
                FROM mimiciv_icu.chartevents ce
                JOIN mimiciv_icu.d_items di ON ce.itemid = di.itemid
                WHERE ce.stay_id = '{stay_id}'
                    AND ce.charttime BETWEEN '{window_start}' AND '{window_end}'
                    AND di.label IN ('Heart Rate', 'Respiratory Rate',
                                     'O2 saturation pulseoxymetry')
                GROUP BY charttime
                ORDER BY charttime
            """
            vitals = pd.read_sql(vitals_query, conn)

            # Aggregate window data
            window_features = {
                'stay_id': stay_id,
                'window_start': window_start,
                'window_end': window_end,
                'heart_rate_mean': vitals['heart_rate'].mean(),
                'heart_rate_std': vitals['heart_rate'].std(),
                'resp_rate_mean': vitals['resp_rate'].mean(),
                'spo2_min': vitals['spo2'].min()
            }

            all_windows.append(window_features)
            current_time += pd.Timedelta(hours=step_hours)

    return pd.DataFrame(all_windows)

# Example usage
# stay_ids = ['30000001', '30000002', '30000003']
# time_series_df = create_time_series_dataset(stay_ids, engine,
#                                              window_hours=6, step_hours=1)
```

### R: MIMIC-IV Data Analysis

```r
library(DBI)
library(RPostgres)
library(dplyr)
library(ggplot2)

# Connect to database
con <- dbConnect(
  Postgres(),
  dbname = "mimic",
  host = "localhost",
  port = 5432,
  user = "user",
  password = "password"
)

# Load sepsis cohort
sepsis_query <- "
  SELECT
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.intime,
    i.outtime,
    i.los,
    a.hospital_expire_flag,
    s.sofa_24hours
  FROM mimiciv_icu.icustays i
  JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
  JOIN mimiciv_derived.sofa s ON i.stay_id = s.stay_id
  WHERE s.sofa_24hours >= 2
"
sepsis_cohort <- dbGetQuery(con, sepsis_query)

# Calculate mortality rate
mortality_rate <- sepsis_cohort %>%
  summarize(
    total_patients = n(),
    deaths = sum(hospital_expire_flag),
    mortality_rate = mean(hospital_expire_flag) * 100
  )

print(paste("Sepsis cohort mortality rate:",
            round(mortality_rate$mortality_rate, 2), "%"))

# Analyze SOFA scores
sofa_analysis <- sepsis_cohort %>%
  group_by(sofa_category = cut(sofa_24hours,
                                breaks = c(0, 5, 10, 15, 24),
                                labels = c("2-5", "6-10", "11-15", "16+"))) %>%
  summarize(
    n = n(),
    mortality = mean(hospital_expire_flag) * 100,
    mean_los = mean(los)
  )

print(sofa_analysis)

# Visualize
ggplot(sofa_analysis, aes(x = sofa_category, y = mortality)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(
    title = "Hospital Mortality by SOFA Score Category",
    x = "SOFA Score Range",
    y = "Mortality Rate (%)"
  ) +
  theme_minimal()

# Disconnect
dbDisconnect(con)
```

### BigQuery: Cloud-Based Queries

```sql
-- BigQuery syntax for MIMIC-IV
-- Project: physionet-data

-- 1. Find septic shock patients with vasopressor use
WITH sepsis_patients AS (
  SELECT DISTINCT
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.intime,
    s.sofa_24hours
  FROM `physionet-data.mimiciv_icu.icustays` i
  JOIN `physionet-data.mimiciv_derived.sofa` s
    ON i.stay_id = s.stay_id
  WHERE s.sofa_24hours >= 2
),
vasopressor_use AS (
  SELECT DISTINCT
    ie.stay_id,
    di.label AS vasopressor,
    MIN(ie.starttime) AS first_vasopressor_time
  FROM `physionet-data.mimiciv_icu.inputevents` ie
  JOIN `physionet-data.mimiciv_icu.d_items` di
    ON ie.itemid = di.itemid
  WHERE di.label IN (
    'Norepinephrine',
    'Epinephrine',
    'Vasopressin',
    'Dopamine',
    'Phenylephrine'
  )
  GROUP BY ie.stay_id, di.label
)
SELECT
  sp.subject_id,
  sp.hadm_id,
  sp.stay_id,
  sp.sofa_24hours,
  vu.vasopressor,
  DATETIME_DIFF(vu.first_vasopressor_time, sp.intime, HOUR) AS hours_to_vasopressor
FROM sepsis_patients sp
JOIN vasopressor_use vu ON sp.stay_id = vu.stay_id
ORDER BY sp.subject_id, hours_to_vasopressor;

-- 2. Extract lactate trends for sepsis patients
SELECT
  le.subject_id,
  le.hadm_id,
  le.charttime,
  le.valuenum AS lactate,
  LAG(le.valuenum) OVER (
    PARTITION BY le.subject_id, le.hadm_id
    ORDER BY le.charttime
  ) AS previous_lactate,
  le.valuenum - LAG(le.valuenum) OVER (
    PARTITION BY le.subject_id, le.hadm_id
    ORDER BY le.charttime
  ) AS lactate_change
FROM `physionet-data.mimiciv_hosp.labevents` le
JOIN `physionet-data.mimiciv_hosp.d_labitems` di
  ON le.itemid = di.itemid
WHERE di.label = 'Lactate'
  AND le.valuenum IS NOT NULL
ORDER BY le.subject_id, le.charttime;
```

---

## Research Best Practices

### 1. Data Quality Checks

```sql
-- Check for missing values in critical fields
SELECT
    'chartevents' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN valuenum IS NULL THEN 1 ELSE 0 END) AS missing_valuenum,
    ROUND(100.0 * SUM(CASE WHEN valuenum IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_missing
FROM mimiciv_icu.chartevents
UNION ALL
SELECT
    'labevents' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN valuenum IS NULL THEN 1 ELSE 0 END) AS missing_valuenum,
    ROUND(100.0 * SUM(CASE WHEN valuenum IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_missing
FROM mimiciv_hosp.labevents;
```

### 2. Handling Missing Data

**Strategies:**
- **Forward fill:** Use last observation carried forward (LOCF)
- **Linear interpolation:** For time-series data
- **Mean/median imputation:** For static features
- **Indicator variables:** Create binary flag for missingness

**Example:**
```python
# Forward fill for vital signs
vitals_df = vitals_df.sort_values(['stay_id', 'charttime'])
vitals_df['heart_rate_filled'] = vitals_df.groupby('stay_id')['heart_rate'].fillna(method='ffill')

# Create missingness indicator
vitals_df['heart_rate_missing'] = vitals_df['heart_rate'].isna().astype(int)
```

### 3. Temporal Alignment

**Key Considerations:**
- Align features to common time grid
- Account for irregular sampling
- Define prediction windows clearly
- Avoid data leakage from future timepoints

**Example:**
```python
def align_to_grid(df, freq='1H'):
    """Align irregular time-series to regular grid"""
    df['charttime'] = pd.to_datetime(df['charttime'])
    df = df.set_index('charttime')
    df_resampled = df.resample(freq).mean()
    return df_resampled.reset_index()
```

### 4. Train-Test Splits

**Recommendations:**
- **Patient-level split:** Ensure no patient appears in both train and test
- **Temporal split:** Use chronological order to avoid look-ahead bias
- **Stratification:** Match outcome distributions across splits

**Example:**
```python
from sklearn.model_selection import train_test_split

# Patient-level split
unique_patients = df['subject_id'].unique()
train_patients, test_patients = train_test_split(
    unique_patients,
    test_size=0.2,
    random_state=42
)

train_df = df[df['subject_id'].isin(train_patients)]
test_df = df[df['subject_id'].isin(test_patients)]
```

---

## Additional Resources

### Official Documentation
- **MIMIC-IV Documentation:** https://mimic.mit.edu/docs/iv/
- **PhysioNet MIMIC-IV:** https://physionet.org/content/mimiciv/3.1/
- **MIMIC Code Repository:** https://github.com/MIT-LCP/mimic-code

### Tutorials and Workshops
- **MIMIC Workshop Materials:** https://github.com/MIT-LCP/mimic-workshop
- **BigQuery Tutorial:** https://mimic.mit.edu/docs/iii/tutorials/intro-to-mimic-iii-bq/
- **SQL Introduction:** https://github.com/MIT-LCP/mimic-code/blob/main/tutorials/sql-intro.md

### Research Papers
- **MIMIC-IV Technical Paper:** [MIMIC-IV, a freely accessible electronic health record dataset](https://www.nature.com/articles/s41597-022-01899-x)
- **MIMIC-Sepsis Benchmark:** [MIMIC-Sepsis: A Curated Benchmark](https://arxiv.org/abs/2510.24500)

### Community
- **MIMIC Forums:** https://github.com/MIT-LCP/mimic-code/discussions
- **Stack Overflow:** Tag `mimic` for questions

---

## References

### Dataset Documentation
1. [MIMIC-IV v3.1 - PhysioNet](https://physionet.org/content/mimiciv/3.1/)
2. [MIMIC-IV-ED v2.2 - PhysioNet](https://physionet.org/content/mimic-iv-ed/2.2/)
3. [MIMIC-IV Documentation](https://mimic.mit.edu/docs/iv/)
4. [Getting Started with MIMIC](https://mimic.mit.edu/docs/gettingstarted/)

### Access and Credentialing
5. [Requesting Access to MIMIC](https://mimic.mit.edu/iii/gettingstarted/)
6. [PhysioNet Credentialing Process](https://physionet.org/news/post/395)

### Scientific Publications
7. [MIMIC-IV: A freely accessible electronic health record dataset - Nature Scientific Data](https://www.nature.com/articles/s41597-022-01899-x)
8. [MIMIC-IV: A freely accessible electronic health record dataset - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9810617/)

### Sepsis Research
9. [MIMIC-Sepsis: A Curated Benchmark - arXiv](https://arxiv.org/abs/2510.24500)
10. [MIMIC-Sepsis: Full Paper - arXiv HTML](https://arxiv.org/html/2510.24500v1)
11. [SII and Mortality in Sepsis - Springer](https://link.springer.com/article/10.1007/s10238-023-01029-w)
12. [BPRI and Mortality in Septic Shock - PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0325110)
13. [TyG Index and Sepsis Mortality - Scientific Reports](https://www.nature.com/articles/s41598-024-75050-8)
14. [MAP and 28-Day Mortality in Sepsis - IJMR](https://www.i-jmr.org/2025/1/e63291)
15. [Lymphocyte Count and Sepsis Outcomes - IJEM](https://intjem.biomedcentral.com/articles/10.1186/s12245-024-00682-6)
16. [Concerns Regarding Sepsis ICD Codes - Journal of Translational Medicine](https://link.springer.com/article/10.1186/s12967-025-06612-1)

### Code Repositories
17. [MIMIC Code Repository - GitHub](https://github.com/MIT-LCP/mimic-code)
18. [MIMIC SQL Examples - GitHub](https://github.com/sarvarip/MIMIC-SQL)
19. [MIMIC Workshop - GitHub](https://github.com/MIT-LCP/mimic-workshop)
20. [MIMIC-IV BigQuery Documentation](https://github.com/MIT-LCP/mimic-iv-website/blob/master/content/about/bigquery.md)

### Additional Studies
21. [An Extensive Data Processing Pipeline for MIMIC-IV - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9854277/)
22. [Assessing Different Diagnoses in MIMIC-IV v2.2 and MIMIC-IV-ED](https://www.scientificarchives.com/article/assessing-different-diagnoses-in-mimic-iv-v2.2-and-mimic-iv-ed-datasets)

---

## Summary Table: MIMIC-IV at a Glance

| Category | Details |
|----------|---------|
| **Total Patients** | 364,627 unique individuals |
| **Hospitalizations** | 546,028 admissions |
| **ICU Stays** | 94,458 stays |
| **ED Visits** | 422,500 visits (2011-2019) |
| **Time Period** | 2008-2022 |
| **Institution** | Beth Israel Deaconess Medical Center, Boston, MA |
| **Modules** | hosp (hospital-wide), icu (ICU-specific), note (clinical notes), ed (emergency department) |
| **Total Tables** | 26+ tables across modules |
| **Access** | Restricted - requires PhysioNet credentialing + CITI training + DUA |
| **License** | PhysioNet Credentialed Health Data License 1.5.0 |
| **Cloud Platforms** | Google BigQuery, AWS |
| **Primary Use Cases** | Sepsis prediction, mortality forecasting, clinical decision support, treatment optimization |

---

**Document Created:** November 30, 2025
**Author:** Hybrid Reasoning Acute Care Research Team
**Version:** 1.0

For questions or updates, refer to the official MIMIC documentation at https://mimic.mit.edu/
