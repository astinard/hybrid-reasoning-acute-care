# LOINC for Laboratory Data in Knowledge Graphs

## Executive Summary

This document provides a comprehensive overview of LOINC (Logical Observation Identifiers Names and Codes) for representing laboratory data in knowledge graphs, with specific focus on acute care settings and sepsis workups. LOINC is the universal standard for identifying medical laboratory observations, first developed in 1994 by the Regenstrief Institute.

## 1. LOINC 6-Part Code Structure

### 1.1 Overview

Each LOINC test is represented by a formal six-part name and assigned a unique numeric code with a check digit. The fully specified name format is:

```
<component/analyte>:<kind of property>:<time aspect>:<system type>:<scale>:<method>
```

### 1.2 The Six Axes Explained

**1. Component/Analyte** - What is measured, evaluated, or observed
   - Examples: Glucose, Hemoglobin, Lactate, Creatinine, White blood cells
   - This is the primary substance or entity being measured
   - Most specific identifier of what the test measures

**2. Property (Kind of Property)** - The characteristic being measured
   - Mass Concentration (MCnc): mg/dL, g/L
   - Substance Concentration (SCnc): mmol/L, mEq/L
   - Catalytic Activity (CCnc): enzyme activity units
   - Arbitrary Concentration (ACnc): arbitrary units
   - Number Concentration (NCnc): cells/volume
   - Mass (Mass): absolute mass in grams
   - Volume (Vol): absolute volume
   - Presence/Identity (Prid): qualitative detection

**3. Time Aspect** - Timing of the measurement
   - Pt (Point in time): single measurement at specific time
   - 24H (24 hour): measurement over 24-hour period
   - 8H, 12H: other timed intervals
   - Continuous: ongoing monitoring
   - Random: unspecified timing

**4. System (Specimen Type)** - The sample source
   - Bld (Blood): whole blood
   - Ser (Serum): blood serum
   - Plas (Plasma): blood plasma
   - Ser/Plas: either serum or plasma
   - Urine: urine specimen
   - BldA (Arterial blood)
   - BldV (Venous blood)
   - CSF (Cerebrospinal fluid)
   - Sputum, Wound, etc.

**5. Scale** - The measurement scale
   - Qn (Quantitative): numeric result with units
   - Ord (Ordinal): ordered categories (negative, trace, 1+, 2+, 3+)
   - Nom (Nominal): named categories without order
   - Nar (Narrative): free text description
   - Doc (Document): attached document
   - Multi: multiple values

**6. Method (Optional)** - Measurement technique
   - Only included when method affects clinical interpretation
   - Examples: "by Automated count", "by High sensitivity method"
   - Distinguishes tests with different reference ranges or sensitivities
   - Examples: Immunoassay, Electrophoresis, Culture, PCR

### 1.3 Clinical Examples

**Example 1: White Blood Cell Count**
```
LOINC Code: 6690-2
Full Name: Leukocytes:NCnc:Pt:Bld:Qn:Automated count
Breakdown:
  - Component: Leukocytes (white blood cells)
  - Property: NCnc (Number Concentration)
  - Time: Pt (Point in time)
  - System: Bld (Blood)
  - Scale: Qn (Quantitative)
  - Method: Automated count
```

**Example 2: Serum Lactate**
```
LOINC Code: 2524-7
Full Name: Lactate:SCnc:Pt:Ser/Plas:Qn
Breakdown:
  - Component: Lactate
  - Property: SCnc (Substance Concentration in mmol/L)
  - Time: Pt (Point in time)
  - System: Ser/Plas (Serum or Plasma)
  - Scale: Qn (Quantitative)
  - Method: (unspecified - not clinically relevant)
```

**Example 3: High-Sensitivity CRP**
```
LOINC Code: 71426-1
Full Name: C reactive protein:MCnc:Pt:Bld:Qn:High sensitivity method
Breakdown:
  - Component: C reactive protein
  - Property: MCnc (Mass Concentration)
  - Time: Pt (Point in time)
  - System: Bld (Blood)
  - Scale: Qn (Quantitative)
  - Method: High sensitivity method (distinguishes from standard CRP)
```

### 1.4 Importance for Knowledge Graphs

The 6-part structure provides rich semantic information for knowledge graph representation:
- Each axis can be represented as a separate node property or relationship
- Enables precise matching and interoperability across systems
- Facilitates reasoning about test equivalencies and substitutions
- Supports inference based on component hierarchies and relationships

## 2. Sepsis Workup Panels

### 2.1 Complete Blood Count (CBC) Panel

**Primary CBC Panel Codes:**

| LOINC Code | Description | Use Case |
|------------|-------------|----------|
| 58410-2 | CBC panel - Blood by Automated count | Standard automated CBC |
| 57021-8 | CBC W Auto Differential panel - Blood | CBC with automated differential |
| 57782-5 | CBC W Ordered Manual Differential panel - Blood | CBC with manual differential |
| 69742-5 | CBC W Differential panel, method unspecified - Blood | Generic CBC with differential |
| 57022-6 | CBC W Reflex Manual Differential panel - Blood | Automated with reflex to manual |

**Key CBC Components for Sepsis:**

| Component | LOINC Code | Reference Range | Sepsis Significance |
|-----------|------------|-----------------|---------------------|
| White blood cells | 6690-2 | 4.5-11.0 K/uL | Leukocytosis (>12K) or leukopenia (<4K) |
| Neutrophils | 751-8 | 40-70% | Bandemia indicates left shift |
| Lymphocytes | 736-9 | 20-40% | Lymphopenia common in sepsis |
| Hemoglobin | 718-7 | 12-17 g/dL | May decrease with bleeding/hemolysis |
| Platelets | 777-3 | 150-400 K/uL | Thrombocytopenia (<100K) in severe sepsis |

**Clinical Example - Septic Patient CBC:**
```
Time: 2024-01-15 06:00
Patient: ICU-2024-001

LOINC 6690-2 (WBC): 18.5 K/uL [HIGH] - indicates leukocytosis
LOINC 751-8 (Neutrophils): 85% [HIGH] - neutrophil predominance
LOINC 26508-2 (Bands): 15% [HIGH] - left shift, immature forms
LOINC 736-9 (Lymphocytes): 8% [LOW] - relative lymphopenia
LOINC 777-3 (Platelets): 85 K/uL [LOW] - thrombocytopenia
LOINC 718-7 (Hemoglobin): 10.2 g/dL [LOW] - anemia

Interpretation: Leukocytosis with left shift and thrombocytopenia
consistent with severe infection/sepsis
```

### 2.2 Lactate Measurement

**Lactate LOINC Codes:**

| LOINC Code | Specification | Units | Clinical Context |
|------------|---------------|-------|------------------|
| 2524-7 | Lactate:SCnc:Pt:Ser/Plas:Qn | mmol/L | Most common, serum/plasma |
| 32693-4 | Lactate:SCnc:Pt:Bld:Qn | mmol/L | Whole blood (POC devices) |
| 14118-4 | Lactate:MCnc:Pt:Ser/Plas:Qn | mg/dL | Mass concentration |
| 19239-3 | Lactate:SCnc:Pt:BldC:Qn | mmol/L | Capillary blood |
| 32133-1 | Lactate:SCnc:Pt:PlasV:Qn | mmol/L | Venous plasma |
| 101656-7 | Lactate and pyruvate panel - Ser/Plas/Bld | Panel | Combined lactate/pyruvate |

**Key Distinctions:**
- SCnc (Substance Concentration) in mmol/L is most commonly used
- MCnc (Mass Concentration) in mg/dL less common
- Different specimen types (arterial vs venous vs capillary) have separate codes
- Lactate collected in fluoride or lithium tubes (plasma)
- ABG/VBG collections capture blood levels

**Clinical Significance:**
- Normal: <2.0 mmol/L
- Elevated: 2.0-4.0 mmol/L (tissue hypoperfusion)
- Severe: >4.0 mmol/L (septic shock, poor prognosis)
- Half-life: 20-30 minutes with adequate perfusion

**Serial Lactate Monitoring Example:**
```
Patient: ICU-2024-001
Diagnosis: Septic shock

T0 (Presentation): LOINC 2524-7 = 6.8 mmol/L [CRITICAL]
T1 (+3 hours):     LOINC 2524-7 = 4.2 mmol/L [IMPROVING]
T2 (+6 hours):     LOINC 2524-7 = 2.8 mmol/L [IMPROVING]
T3 (+12 hours):    LOINC 2524-7 = 1.9 mmol/L [NORMALIZED]

Lactate clearance >50% at 6 hours indicates adequate resuscitation
```

### 2.3 Blood Culture and Microbiology

**Blood Culture LOINC Codes:**

| LOINC Code | Description | Timeline |
|------------|-------------|----------|
| 90423-5 | Microorganism preliminary growth detection panel - Blood by Culture | 24-48 hours |
| 600-7 | Bacteria identified in Blood by Culture | Final identification |
| 92789-7 | Gram positive blood culture panel by NAA with probe detection | Rapid PCR-based |
| 71472-5 | Date of blood culture | Metadata |
| 635-3 | Bacteria identified in Blood by Aerobe culture | Aerobic bottle |
| 634-6 | Bacteria identified in Blood by Anaerobe culture | Anaerobic bottle |

**Rapid Diagnostic Panels:**
- Traditional culture: 24-72 hours
- MALDI-TOF identification: 18-24 hours (after growth)
- Multiplex PCR panels: 1-3 hours (direct from blood)
- FilmArray, BioFire, Verigene systems

### 2.4 Sepsis Biomarkers

**Procalcitonin (PCT):**

| LOINC Code | Description | Clinical Use |
|------------|-------------|--------------|
| 33959-8 | Procalcitonin:MCnc:Pt:Ser/Plas:Qn | Primary PCT test |
| LG15749-1 | Procalcitonin group | LOINC group for all PCT tests |

**PCT Interpretation:**
- <0.05 ng/mL: Normal, bacterial infection unlikely
- 0.05-0.5 ng/mL: Low risk
- 0.5-2.0 ng/mL: Moderate risk, possible sepsis
- 2.0-10.0 ng/mL: High risk, likely severe sepsis
- >10.0 ng/mL: Very high risk, septic shock likely

**PCT Kinetics:**
- Rises 6-12 hours after bacterial infection
- Peaks at 24-48 hours
- Half-life: 20-24 hours
- Decreases 50% per 24 hours with appropriate treatment
- More specific than CRP for bacterial vs viral infections

**C-Reactive Protein (CRP):**

| LOINC Code | Description | Use Case |
|------------|-------------|----------|
| 1988-5 | CRP:MCnc:Pt:Ser/Plas:Qn | Standard CRP |
| 71426-1 | CRP:MCnc:Pt:Bld:Qn:High sensitivity method | Point-of-care high-sensitivity |
| 30522-7 | CRP:MCnc:Pt:Ser/Plas:Qn:High sensitivity method | Lab-based high-sensitivity |
| 76486-0 | CRP:SCnc:Pt:Ser/Plas:Qn:High sensitivity method | Molar concentration hs-CRP |

**CRP Interpretation:**
- <10 mg/L: Normal
- 10-50 mg/L: Mild inflammation
- 50-200 mg/L: Moderate inflammation, possible bacterial infection
- >200 mg/L: Severe inflammation, likely bacterial sepsis

**CRP vs PCT:**
- CRP is non-specific (elevated in trauma, surgery, autoimmune)
- PCT more specific for bacterial infections
- PCT rises faster and clears faster than CRP
- Combined use improves diagnostic accuracy

### 2.5 Additional Sepsis Workup Labs

**Renal Function:**

| LOINC Code | Test | Significance |
|------------|------|--------------|
| 2160-0 | Creatinine:MCnc:Pt:Ser/Plas:Qn | Kidney function, AKI detection |
| 38483-4 | Creatinine:MCnc:Pt:Bld:Qn | Point-of-care creatinine |
| 3094-0 | BUN:MCnc:Pt:Ser/Plas:Qn | Blood urea nitrogen |
| 2161-8 | Creatinine:MCnc:Pt:Urine:Qn | Urine creatinine |

**Liver Function:**

| LOINC Code | Test | Significance |
|------------|------|--------------|
| 1975-2 | Bilirubin.total:MCnc:Pt:Ser/Plas:Qn | Liver dysfunction, cholestasis |
| 1742-6 | ALT:CCnc:Pt:Ser/Plas:Qn | Hepatocellular injury |
| 1920-8 | AST:CCnc:Pt:Ser/Plas:Qn | Hepatocellular injury |
| 6768-6 | Alkaline phosphatase:CCnc:Pt:Ser/Plas:Qn | Cholestatic pattern |

**Coagulation:**

| LOINC Code | Test | Significance |
|------------|------|--------------|
| 5902-2 | PT:Time:Pt:PPP:Qn | Prothrombin time, DIC screening |
| 3173-2 | aPTT:Time:Pt:PPP:Qn | Activated partial thromboplastin time |
| 3255-7 | Fibrinogen:MCnc:Pt:PPP:Qn | Consumptive coagulopathy |
| 48066-5 | D-dimer:MCnc:Pt:PPP:Qn | Fibrinolysis, thrombosis |

## 3. Temporal Patterns in Laboratory Results

### 3.1 Time Aspect Component

LOINC encodes temporal information in the third axis (Time Aspect):

| Time Code | Description | Example Use |
|-----------|-------------|-------------|
| Pt | Point in time | Single glucose measurement |
| 24H | 24-hour collection | 24-hour urine protein |
| 12H | 12-hour period | 12-hour creatinine clearance |
| 8H | 8-hour period | 8-hour urine collection |
| Random | Random timing | Random urine specimen |
| Continuous | Continuous monitoring | Continuous glucose monitoring |
| 1H, 2H, etc. | Specific intervals | Glucose tolerance test time points |

### 3.2 Temporal Knowledge Graphs for Laboratory Data

**Ontology Framework:**

The Clinical Time Ontology (CTO) extends OWL-Time to handle clinical temporal expressions:
- Fuzzy time: "early morning labs", "around noon"
- Cyclic time: "daily at 0600", "every 4 hours"
- Irregular time: "as needed", "prn"
- Negations: "not drawn today"
- Duration: "for 3 days", "until resolved"

**Allen's Interval Algebra for Lab Sequences:**

```
Lab Event Relationships:
- Before: Lab A completes before Lab B starts
- Meets: Lab A ends exactly when Lab B begins
- Overlaps: Lab A starts before and overlaps with Lab B
- During: Lab A occurs entirely during Lab B
- Starts: Lab A and Lab B start together
- Finishes: Lab A and Lab B finish together
- Equals: Lab A and Lab B are concurrent
```

**Example Temporal Pattern - Sepsis Bundle:**

```turtle
# RDF representation of sepsis bundle lab sequence
:SepsisBundleProtocol a :ClinicalProtocol ;
    :hasTemporalConstraint [
        a :TemporalConstraint ;
        :labTest :InitialLactate ;
        :loincCode "2524-7" ;
        :timing "T0" ;
        :timeWindow "within 1 hour of sepsis recognition"
    ] ;
    :hasTemporalConstraint [
        a :TemporalConstraint ;
        :labTest :BloodCultures ;
        :loincCode "90423-5" ;
        :timing "T0" ;
        :temporalRelation :before ;
        :relatedEvent :AntibioticAdministration ;
        :maxDelay "45 minutes"
    ] ;
    :hasTemporalConstraint [
        a :TemporalConstraint ;
        :labTest :RepeatLactate ;
        :loincCode "2524-7" ;
        :timing "T0 + 6 hours" ;
        :condition "if initial lactate > 2.0 mmol/L" ;
        :evaluateClearance true
    ] .
```

### 3.3 Time-Series Analysis Patterns

**Lactate Clearance Calculation:**

```python
# Temporal reasoning for lactate clearance
initial_lactate = {
    "timestamp": "2024-01-15T06:00:00Z",
    "loinc": "2524-7",
    "value": 6.8,
    "unit": "mmol/L"
}

followup_lactate = {
    "timestamp": "2024-01-15T12:00:00Z",
    "loinc": "2524-7",
    "value": 2.8,
    "unit": "mmol/L"
}

lactate_clearance = ((6.8 - 2.8) / 6.8) * 100 = 58.8%
time_delta = 6 hours

# Knowledge graph representation
:LactateClearanceEvent a :ClinicalMeasurement ;
    :hasInitialValue 6.8 ;
    :hasFollowupValue 2.8 ;
    :clearancePercentage 58.8 ;
    :timeInterval "PT6H" ;
    :clinicalSignificance :AdequateResuscitation .
```

**Creatinine Trend for AKI Detection:**

```
Temporal Pattern Recognition:
Baseline: LOINC 2160-0 = 0.9 mg/dL (T-72h)
Day 1:    LOINC 2160-0 = 1.2 mg/dL (T0)
Day 2:    LOINC 2160-0 = 1.8 mg/dL (T+24h)  [KDIGO Stage 1 AKI]
Day 3:    LOINC 2160-0 = 2.7 mg/dL (T+48h)  [KDIGO Stage 2 AKI]

Criteria: ≥0.3 mg/dL increase within 48 hours OR
          ≥1.5x baseline within 7 days
```

### 3.4 Irregular Sampling and Missing Data

**Challenge in Acute Care:**
- Labs drawn at irregular intervals (not evenly spaced)
- Missing values due to clinical decisions
- Varying temporal resolution (hourly vs daily)
- Different labs have different sampling frequencies

**Knowledge Graph Handling:**

```turtle
:LabObservation a :TimestampedEvent ;
    :loincCode "2524-7" ;
    :observedAt "2024-01-15T06:00:00Z"^^xsd:dateTime ;
    :value 6.8 ;
    :unit "mmol/L" ;
    :samplingRegimen :AsNeeded ;
    :clinicalContext :SepsisResuscitation .

:PreviousLabObservation a :TimestampedEvent ;
    :loincCode "2524-7" ;
    :observedAt "2024-01-15T00:00:00Z"^^xsd:dateTime ;
    :value 8.2 ;
    :timeGap "PT6H" ;
    :temporalRelation :before ;
    :nextObservation :LabObservation .
```

## 4. LOINC to SNOMED CT Mappings

### 4.1 Overview of Mapping Challenges

**Fundamental Differences:**

| Aspect | LOINC | SNOMED CT |
|--------|-------|-----------|
| Purpose | Laboratory observations (questions) | Clinical findings (answers) |
| Granularity | Very fine-grained (>90,000 lab codes) | Less granular for lab procedures |
| Structure | 6-part fully specified names | Concept hierarchy with relationships |
| Pre-coordination | Highly pre-coordinated | May require post-coordination |
| Focus | "What test was ordered?" | "What was found?" |

**Mapping Success Rates:**
- LOINC to SNOMED CT: 87% exact mapping for panel tests
- SNOMED CT to LOINC: 78.2% exact mapping
- 12 SNOMED mappings required post-coordination
- Some LOINC codes unmappable due to SNOMED underspecification

### 4.2 Collaboration and Standards

**Regenstrief-SNOMED International Collaboration:**
- Agreement signed July 2013
- Align representation of laboratory test attributes
- Reduce duplication between terminologies
- Provide common framework for LOINC and SNOMED CT
- Last collaboration package: July 2017 (not currently updated)

**SHIELD Initiative:**
Systemic Harmonization and Interoperability Enhancement for Laboratory Data
- Multi-stakeholder public-private partnership
- Harmonize laboratory coding using LOINC, SNOMED CT, and UCUM
- Focus on terminology standards integration

### 4.3 Complementary Use in HL7 Messages

**Division of Responsibilities:**

| Component | Terminology | HL7 Field |
|-----------|-------------|-----------|
| Test Question | LOINC | OBR-4, OBX-3 |
| Numeric Answer | UCUM units | OBX-5 (value) |
| Non-numeric Answer | SNOMED CT | OBX-5 (coded value) |
| Specimen Type | SNOMED CT | SPM-4 |
| Body Site | SNOMED CT | SPM-8 |

**Example HL7 v2 Message:**

```
OBR|1|12345|67890|600-7^Bacteria identified in Blood^LN
OBX|1|CE|600-7^Bacteria identified in Blood^LN||
    112283007^Escherichia coli^SCT||||||F
```

Where:
- OBR-4: LOINC 600-7 (what test - blood culture)
- OBX-5: SNOMED CT 112283007 (what found - E. coli)

### 4.4 LOINC-SNOMED Crosswalk Examples

**White Blood Cell Count:**

| LOINC Code | LOINC Name | SNOMED CT Code | SNOMED CT Name |
|------------|------------|----------------|----------------|
| 6690-2 | Leukocytes:NCnc:Pt:Bld:Qn | 767002 | White blood cell count procedure |
| 804-5 | Leukocytes:NCnc:Pt:Bld:Qn:Manual count | 767002 | White blood cell count procedure |

**Glucose:**

| LOINC Code | LOINC Name | SNOMED CT Code | SNOMED CT Name |
|------------|------------|----------------|----------------|
| 2345-7 | Glucose:MCnc:Pt:Ser/Plas:Qn | 33747003 | Glucose measurement |
| 2339-0 | Glucose:MCnc:Pt:Bld:Qn | 33747003 | Glucose measurement |

**Lactate:**

| LOINC Code | LOINC Name | SNOMED CT Code | SNOMED CT Name |
|------------|------------|----------------|----------------|
| 2524-7 | Lactate:SCnc:Pt:Ser/Plas:Qn | 271241007 | Lactate measurement |
| 32693-4 | Lactate:SCnc:Pt:Bld:Qn | 271241007 | Lactate measurement |

**Note:** Multiple LOINC codes often map to single SNOMED CT concept due to LOINC's finer granularity.

### 4.5 Post-Coordination Requirements

Some laboratory concepts require SNOMED CT post-coordination:

**Example: High-Sensitivity CRP**

```
LOINC: 71426-1 (C reactive protein:MCnc:Pt:Bld:Qn:High sensitivity method)

SNOMED CT post-coordinated expression:
  166842003 |C-reactive protein measurement| :
    246501002 |Technique| = 702873001 |High sensitivity technique| ,
    704319004 |Inherent location| = 119297000 |Blood specimen|
```

### 4.6 OHDSI Common Data Model Integration

**Crosswalk Creation:**

The OMOP CDM integrates both LOINC and SNOMED CT:

```sql
-- LOINC to SNOMED mapping in OMOP
SELECT
  c1.concept_code AS loinc_code,
  c1.concept_name AS loinc_name,
  cr.relationship_id,
  c2.concept_code AS snomed_code,
  c2.concept_name AS snomed_name
FROM concept c1
JOIN concept_relationship cr ON c1.concept_id = cr.concept_id_1
JOIN concept c2 ON cr.concept_id_2 = c2.concept_id
WHERE c1.vocabulary_id = 'LOINC'
  AND c2.vocabulary_id = 'SNOMED'
  AND cr.relationship_id = 'Maps to';
```

## 5. MIMIC-IV LOINC Distributions

### 5.1 LOINC Coverage in MIMIC-IV

**Version History:**

| Version | Total Lab Items | LOINC Mapped | Coverage | Notes |
|---------|----------------|--------------|----------|-------|
| MIMIC-III | 753 | 586 | 77.8% | Initial mapping effort |
| MIMIC-IV v0.4 | 1,625 | 267 | 16.4% | Incomplete mapping |
| MIMIC-IV v2.2 | 1,625+ | In progress | Variable | Community-driven |
| MIMIC-IV v3.1 | 1,625+ | In progress | Variable | October 2024 release |

**Mapping Challenges:**
- loinc_code column removed from v3.0+ (moved to MIMIC Code Repository)
- Errors found in original loinc_code values
- Community collaborative improvement process
- Mapping file: mimic_labitems_to_loinc.csv

### 5.2 MIMIC-III to MIMIC-IV Compatibility

**ItemID Consistency:**
- ItemIDs starting with "2xx" are same between MIMIC-III and MIMIC-IV
- MIMIC-IV contains additional itemIDs beyond MIMIC-III
- Can join MIMIC-III and IV tables to backfill LOINC codes
- Always perform sanity checks after cross-version mapping

**Recommended Approach:**
```sql
-- Backfill LOINC codes from MIMIC-III to MIMIC-IV
SELECT
  m4.itemid,
  m4.label AS mimic4_label,
  m3.loinc_code,
  m3.label AS mimic3_label
FROM mimiciv.d_labitems m4
LEFT JOIN mimiciii.d_labitems m3
  ON m4.itemid = m3.itemid
WHERE m3.loinc_code IS NOT NULL
  AND m4.itemid LIKE '2%';
```

### 5.3 Common Lab Tests in MIMIC-IV

**Top Laboratory Tests by Frequency:**

| ItemID | LOINC Code | Test Name | Frequency | % of Total |
|--------|------------|-----------|-----------|------------|
| 50912 | 2951-2 | Sodium | Very High | ~8.5% |
| 50902 | 2823-3 | Potassium | Very High | ~8.2% |
| 50931 | 2160-0 | Creatinine | Very High | ~8.0% |
| 50971 | 2345-7 | Glucose | Very High | ~7.8% |
| 51006 | 3094-0 | Blood Urea Nitrogen | Very High | ~7.5% |
| 51221 | 718-7 | Hematocrit | Very High | ~6.8% |
| 51222 | 718-7 | Hemoglobin | Very High | ~6.7% |
| 51265 | 777-3 | Platelet Count | Very High | ~6.5% |
| 51301 | 6690-2 | White Blood Cells | Very High | ~6.4% |
| 50868 | 1742-6 | Alanine Aminotransferase (ALT) | High | ~4.2% |

### 5.4 Sepsis-Relevant Labs in MIMIC-IV

**Critical Care Laboratory Panels:**

```python
# MIMIC-IV sepsis workup labs
sepsis_loinc_codes = {
    'WBC': '6690-2',
    'Neutrophils': '751-8',
    'Bands': '26508-2',
    'Lactate': '2524-7',
    'Procalcitonin': '33959-8',  # Less frequently available
    'CRP': '1988-5',              # Less frequently available
    'Platelets': '777-3',
    'Creatinine': '2160-0',
    'Bilirubin': '1975-2',
    'PT': '5902-2',
    'PTT': '3173-2'
}
```

**Data Processing Pipeline:**

The MIMIC-IV data processing pipeline provides:
- Mean frequency of lab codes across patients
- Percentage of missing values per lab code
- Temporal distribution of lab draws
- User can specify which lab codes to retain

### 5.5 LOINC Mapping Quality Issues

**Unmappable Tests (13% of unique tests):**

| Reason | Percentage | Example |
|--------|------------|---------|
| Unclear test meaning | 60% | Ambiguous abbreviations, local names |
| Not in LOINC table | 29% | Institution-specific tests, derived values |
| Multiple possible codes | 8% | Insufficient specificity to choose |
| Other | 3% | Various technical issues |

**Coverage of Test Results:**
- 87% of unique laboratory tests mapped to LOINC
- 94% of total number of test results mapped
- Most critical care tests well-represented in LOINC

### 5.6 FHIR Mapping for MIMIC-IV

**MIMIC-IV to FHIR R4 Transformation:**

GitHub project: srdc/mimic-iv-to-fhir

```json
{
  "resourceType": "Observation",
  "id": "mimic-lab-123456",
  "status": "final",
  "category": [{
    "coding": [{
      "system": "http://terminology.hl7.org/CodeSystem/observation-category",
      "code": "laboratory"
    }]
  }],
  "code": {
    "coding": [{
      "system": "http://loinc.org",
      "code": "2524-7",
      "display": "Lactate [Moles/volume] in Serum or Plasma"
    }]
  },
  "subject": {
    "reference": "Patient/mimic-patient-001"
  },
  "effectiveDateTime": "2024-01-15T06:00:00Z",
  "valueQuantity": {
    "value": 6.8,
    "unit": "mmol/L",
    "system": "http://unitsofmeasure.org",
    "code": "mmol/L"
  },
  "referenceRange": [{
    "low": {
      "value": 0.5,
      "unit": "mmol/L"
    },
    "high": {
      "value": 2.0,
      "unit": "mmol/L"
    }
  }]
}
```

### 5.7 Research Applications

**Common MIMIC-IV Laboratory Analysis Tasks:**

1. **Sepsis Prediction Models:**
   - Time-series of WBC, lactate, creatinine
   - SOFA score calculation from labs
   - Early warning systems

2. **Acute Kidney Injury Detection:**
   - Creatinine trajectory analysis
   - KDIGO criteria implementation
   - Temporal patterns in renal function

3. **Antibiotic Stewardship:**
   - Culture results and susceptibilities
   - PCT-guided antibiotic duration
   - Time to appropriate therapy

4. **Multi-modal Integration:**
   - Labs + vital signs + medications
   - Labs + clinical notes (NLP)
   - Labs + imaging + outcomes

## 6. Knowledge Graph Implementation Examples

### 6.1 RDF Schema for Laboratory Observations

```turtle
@prefix loinc: <http://loinc.org/rdf#> .
@prefix snomedct: <http://snomed.info/id/> .
@prefix fhir: <http://hl7.org/fhir/> .
@prefix ucum: <http://unitsofmeasure.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# Class definitions
:LabObservation a owl:Class ;
    rdfs:label "Laboratory Observation" ;
    rdfs:subClassOf fhir:Observation .

:LabPanel a owl:Class ;
    rdfs:label "Laboratory Panel" ;
    rdfs:subClassOf :LabObservation .

# Property definitions
:hasLOINCCode a owl:DatatypeProperty ;
    rdfs:domain :LabObservation ;
    rdfs:range xsd:string .

:hasComponent a owl:ObjectProperty ;
    rdfs:domain :LabPanel ;
    rdfs:range :LabObservation .

:observedValue a owl:DatatypeProperty ;
    rdfs:domain :LabObservation ;
    rdfs:range xsd:decimal .

:hasUnit a owl:ObjectProperty ;
    rdfs:domain :LabObservation ;
    rdfs:range ucum:Unit .

:observedAt a owl:DatatypeProperty ;
    rdfs:domain :LabObservation ;
    rdfs:range xsd:dateTime .
```

### 6.2 Sepsis Case Knowledge Graph

```turtle
# Patient instance
:Patient_ICU_001 a fhir:Patient ;
    :hasAdmission :Admission_20240115 .

:Admission_20240115 a :ICUAdmission ;
    :admissionTime "2024-01-15T04:30:00Z"^^xsd:dateTime ;
    :diagnosis snomedct:91302008 ;  # Sepsis
    :hasClinicalEvent :SepsisRecognition_001 .

# Sepsis recognition triggers lab orders
:SepsisRecognition_001 a :ClinicalEvent ;
    :eventTime "2024-01-15T05:45:00Z"^^xsd:dateTime ;
    :triggersProtocol :SepsisBundleProtocol ;
    :ordersLabPanel :CBCPanel_001 ;
    :ordersLabPanel :SepsisWorkup_001 .

# CBC Panel
:CBCPanel_001 a :LabPanel ;
    :hasLOINCCode "57021-8" ;
    :panelName "CBC W Auto Differential panel - Blood" ;
    :orderedAt "2024-01-15T05:45:00Z"^^xsd:dateTime ;
    :collectedAt "2024-01-15T06:00:00Z"^^xsd:dateTime ;
    :hasComponent :WBC_001, :Hemoglobin_001, :Platelets_001 .

:WBC_001 a :LabObservation ;
    :hasLOINCCode "6690-2" ;
    :testName "Leukocytes [#/volume] in Blood" ;
    :observedAt "2024-01-15T06:15:00Z"^^xsd:dateTime ;
    :observedValue 18.5 ;
    :hasUnit ucum:10*3/uL ;
    :interpretation :Abnormal ;
    :abnormalityType :High ;
    :referenceRangeLow 4.5 ;
    :referenceRangeHigh 11.0 .

:Lactate_001 a :LabObservation ;
    :hasLOINCCode "2524-7" ;
    :testName "Lactate [Moles/volume] in Serum or Plasma" ;
    :observedAt "2024-01-15T06:10:00Z"^^xsd:dateTime ;
    :observedValue 6.8 ;
    :hasUnit ucum:mmol/L ;
    :interpretation :Critical ;
    :clinicalSignificance :TissueHypoperfusion ;
    :triggersFollowup :Lactate_002 .

# Follow-up lactate (temporal relationship)
:Lactate_002 a :LabObservation ;
    :hasLOINCCode "2524-7" ;
    :observedAt "2024-01-15T12:00:00Z"^^xsd:dateTime ;
    :observedValue 2.8 ;
    :hasUnit ucum:mmol/L ;
    :temporalRelation :after ;
    :relatedObservation :Lactate_001 ;
    :timeDelta "PT6H"^^xsd:duration .

# Calculate lactate clearance
:LactateClearance_001 a :CalculatedMeasure ;
    :initialValue :Lactate_001 ;
    :followupValue :Lactate_002 ;
    :clearancePercentage 58.8 ;
    :interpretation :AdequateResuscitation .

# Blood cultures
:BloodCulture_001 a :LabObservation ;
    :hasLOINCCode "90423-5" ;
    :testName "Microorganism preliminary growth detection panel" ;
    :orderedAt "2024-01-15T05:45:00Z"^^xsd:dateTime ;
    :collectedAt "2024-01-15T05:50:00Z"^^xsd:dateTime ;
    :preliminaryResult [
        :resultTime "2024-01-16T18:00:00Z"^^xsd:dateTime ;
        :growthDetected true ;
        :gramStain :GramNegativeRods
    ] ;
    :finalResult [
        :resultTime "2024-01-17T10:00:00Z"^^xsd:dateTime ;
        :organism snomedct:112283007 ;  # Escherichia coli
        :organismName "Escherichia coli"
    ] .
```

## 7. Conclusion and Best Practices

### 7.1 Key Takeaways

1. **LOINC 6-Part Structure**: Provides rich semantic information essential for knowledge graph representation
2. **Sepsis Workup**: Core panels include CBC (58410-2), Lactate (2524-7), Blood Cultures (90423-5), PCT (33959-8)
3. **Temporal Modeling**: Critical for trend analysis, clearance calculations, and clinical decision support
4. **LOINC-SNOMED Mapping**: Complementary use - LOINC for questions, SNOMED for answers
5. **MIMIC-IV**: 87% of unique tests mapped, 94% of total results covered, ongoing community improvement

### 7.2 Recommendations for Knowledge Graph Design

**For Acute Care Applications:**
- Use LOINC codes as primary identifiers for all laboratory observations
- Maintain temporal relationships between repeated measurements
- Link to SNOMED CT for clinical findings and diagnoses
- Include reference ranges and interpretation flags
- Capture specimen collection and result times separately
- Support both point-in-time and time-series queries

**For Sepsis Monitoring:**
- Implement automatic lactate clearance calculations
- Track SOFA score components over time
- Link lab results to therapeutic interventions
- Support bundle compliance checking
- Enable early warning alerts based on lab trends

### 7.3 Future Directions

- Integration with LLMs for automated clinical reasoning
- Temporal knowledge graph construction from EHR data
- Multi-modal fusion of labs, vitals, notes, and imaging
- Standardization efforts continue with SHIELD initiative
- Expansion of MIMIC-IV LOINC mappings through community contributions

## References

- [LOINC Official Website](https://loinc.org/)
- [LOINC Knowledge Base](https://loinc.org/kb/faq/structure/)
- [SNOMED International LOINC Collaboration](https://loinc.org/collaboration/snomed-international/)
- [Issues in Mapping LOINC to SNOMED CT](https://pmc.ncbi.nlm.nih.gov/articles/PMC2655945/)
- [MIMIC-IV Database](https://physionet.org/content/mimiciv/3.1/)
- [MIMIC Code Repository](https://github.com/MIT-LCP/mimic-code)
- [Clinical Time Ontology](https://direct.mit.edu/dint/article/4/3/573/112548/)
- [Temporal Knowledge Graphs in Healthcare](https://arxiv.org/html/2502.21138)
- [Procalcitonin StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK539794/)
- [Comparative Study of LOINC and SNOMED CT](https://www.sciencedirect.com/science/article/abs/pii/S1386505625002722)
