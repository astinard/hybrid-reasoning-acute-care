# RxNorm Medication Knowledge Graph for Acute Care

## Executive Summary

This document provides a comprehensive overview of RxNorm as the foundational medication knowledge graph for emergency department and acute care systems. RxNorm, maintained by the National Library of Medicine (NLM), provides standardized nomenclature for clinical drugs and serves as the interoperability layer between disparate drug vocabularies used in healthcare systems.

**Key Applications for Hybrid Reasoning:**
- Medication ordering and clinical decision support
- Drug-drug interaction detection
- Medication reconciliation across care transitions
- Clinical documentation and semantic interoperability
- Prescription routing and dispensing systems

## 1. RxNorm Concept Types and Term Type Taxonomy (TTY)

### 1.1 Overview of RxNorm Architecture

RxNorm uses Semantic Attribute Base (SAB=RXNORM) term types (TTYs) to classify normalized names by concept type. The architecture is designed around clinical specificity, ranging from broad ingredient categories to precise dispensable formulations.

### 1.2 Generic Drug Categories

RxNorm defines **five primary categories** for generic drugs, ordered by increasing specificity:

#### 1.2.1 IN - Ingredient
**Definition:** The active ingredient(s) in a medication, without strength or dose form.

**Clinical Examples:**
- **RxCUI 1191** - Aspirin (acetylsalicylic acid)
- **RxCUI 5640** - Ibuprofen
- **RxCUI 7052** - Morphine
- **RxCUI 11289** - Warfarin
- **RxCUI 4337** - Ondansetron
- **RxCUI 2582** - Ceftriaxone

**Use Case:** Ingredient-level queries for allergy checking, therapeutic class assignment, and high-level medication reconciliation.

#### 1.2.2 PIN - Precise Ingredient
**Definition:** Ingredient that may or may not be clinically active (e.g., specific salt forms).

**Clinical Examples:**
- Warfarin Sodium (the sodium salt of warfarin)
- Morphine Sulfate (vs. Morphine Hydrochloride)
- Ceftriaxone Sodium

**Use Case:** Distinguishing between different salt forms that may have different bioavailability or clinical characteristics.

#### 1.2.3 SCDC - Semantic Clinical Drug Component
**Definition:** Ingredient plus strength, without dose form.

**Clinical Examples:**
- Aspirin 81 MG
- Morphine 10 MG
- Warfarin Sodium 5 MG
- Ondansetron 4 MG

**Use Case:** Representing partial prescribing information when dose form is not yet specified.

#### 1.2.4 SCDF - Semantic Clinical Drug Form
**Definition:** Ingredient plus dose form, without strength.

**Clinical Examples:**
- Aspirin Oral Tablet
- Morphine Injectable Solution
- Ondansetron Oral Tablet

**Use Case:** Formulary management and route-specific prescribing restrictions.

#### 1.2.5 SCD - Semantic Clinical Drug
**Definition:** Ingredient + Strength + Dose Form (the complete generic drug description).

**Clinical Examples:**
- **RxCUI 198013** - Naproxen 250 MG Oral Tablet
- **RxCUI 307667** - Acetaminophen 160 MG Oral Capsule
- **RxCUI 2264779** - Acetaminophen 325 MG Chewable Tablet
- **RxCUI 310965** - Ibuprofen 200 MG Oral Tablet
- **RxCUI 317297** - Aspirin 120 MG
- **RxCUI 855302** - Warfarin Sodium 2 MG Oral Tablet
- **RxCUI 855296** - Warfarin Sodium 10 MG Oral Tablet

**Use Case:** Complete medication ordering, prescription writing, electronic health record documentation. This is the **primary level** for clinical drug orders.

### 1.3 Branded Drug Categories

RxNorm defines **four parallel categories** for brand name drugs:

#### 1.3.1 BN - Brand Name
**Definition:** Proprietary name for a family of drug products containing specific active ingredients.

**Clinical Examples:**
- **Coumadin** (warfarin sodium)
- **Prozac** (fluoxetine)
- **Tylenol** (acetaminophen)
- **Zofran** (ondansetron)
- **Dilaudid** (hydromorphone)
- **Rocephin** (ceftriaxone)

**Use Case:** Patient communication, brand-specific formulary decisions, brand vs. generic substitution rules.

#### 1.3.2 SBDC - Semantic Branded Drug Component
**Definition:** Brand name plus strength, without dose form.

**Clinical Examples:**
- Coumadin 5 MG
- Zofran 4 MG
- Tylenol 325 MG

#### 1.3.3 SBDF - Semantic Branded Drug Form
**Definition:** Brand name plus dose form, without strength.

**Clinical Examples:**
- Coumadin Oral Tablet
- Zofran Orally Disintegrating Tablet
- Dilaudid Injectable Solution

#### 1.3.4 SBD - Semantic Branded Drug
**Definition:** Brand Name + Ingredient + Strength + Dose Form (complete branded drug description).

**Clinical Examples:**
- **RxCUI 855304** - Warfarin Sodium 2 MG Oral Tablet [Coumadin]
- **RxCUI 855300** - Warfarin Sodium 10 MG Oral Tablet [Jantoven]
- Ondansetron 4 MG Oral Tablet [Zofran]
- Morphine Sulfate 10 MG Injectable Solution [Duramorph]

**Use Case:** Brand-mandated prescriptions, branded drug ordering, brand-specific clinical protocols.

### 1.4 Package Types (Dispensable Products)

#### 1.4.1 GPCK - Generic Pack
**Definition:** Multipart generic drug packaging.

**Example:** Warfarin Sodium Starter Pack (contains multiple strengths for dose titration)

#### 1.4.2 BPCK - Branded Pack
**Definition:** Multipart branded drug packaging.

**Example:** Z-Pak (azithromycin dose pack)

**Use Case:** Both GPCK and BPCK term types represent dispensable products with NDC associations.

### 1.5 Clinical Ordering Best Practices

**Recommendation for EHR Systems:**
- **Primary Ordering Level:** SCD (Semantic Clinical Drug) or SBD (Semantic Branded Drug)
- **Minimum Required Specificity:** Ingredient + Strength + Dose Form
- **Allergy Checking:** Performed at IN (Ingredient) level
- **Formulary Management:** Typically at SCD level with brand substitution rules
- **Dispensing:** Maps to GPCK/BPCK with NDC associations

## 2. Drug-Drug Interaction (DDI) Ontologies

### 2.1 Overview of DDI Knowledge Representation

Drug-drug interactions represent a critical patient safety concern in acute care settings. Multiple ontologies and knowledge bases provide DDI information, with varying coverage, granularity, and clinical actionability.

### 2.2 DINTO - Drug-Drug Interaction Ontology

**Purpose:** Machine-readable ontology for describing and categorizing drug-drug interactions and their mechanisms.

**Key Components:**
- Mechanism-based interaction classification
- Pharmacokinetic vs. pharmacodynamic interactions
- Severity and clinical significance ratings
- Temporal relationships between drug administration

**Clinical Applications:**
- Automated DDI screening in CPOE systems
- Clinical decision support alerts
- Medication safety analytics
- Research on interaction patterns

### 2.3 Major DDI Knowledge Sources

#### 2.3.1 NDF-RT (National Drug File - Reference Terminology)
**Coverage:** Veterans Affairs drug terminology with hierarchical drug classes and interactions
**Integration:** Previously integrated with RxNorm; now replaced by MED-RT
**Overlap with DrugBank:** 24-30% for DDI pairs

#### 2.3.2 DrugBank
**Coverage:** Comprehensive drug database with extensive DDI information
**Structure:** Linked data representation
**Clinical Utility:** Widely used in academic and commercial applications

#### 2.3.3 CRESCENDDI Reference Set
**Size:** 10,286 positive controls and 4,544 negative controls
**Coverage:** 454 drugs, 179 adverse events
**Mapping:** RxNorm concepts with MedDRA adverse event terms
**Use Case:** Validation and benchmarking of DDI detection systems

### 2.4 DDI Interaction Types and Mechanisms

#### 2.4.1 Pharmacokinetic Interactions

**Absorption:**
- Chelation (e.g., fluoroquinolones + calcium/iron)
- pH alterations (e.g., proton pump inhibitors affecting drug solubility)

**Distribution:**
- Protein binding displacement (e.g., warfarin + highly protein-bound drugs)

**Metabolism:**
- CYP450 enzyme induction (e.g., rifampin inducing warfarin metabolism)
- CYP450 enzyme inhibition (e.g., fluconazole inhibiting CYP2C9)

**Excretion:**
- Renal tubular competition (e.g., probenecid + penicillin)

#### 2.4.2 Pharmacodynamic Interactions

**Synergism:**
- Two drugs producing greater combined effect
- Example: Fentanyl + Midazolam (respiratory depression)

**Antagonism:**
- Two drugs with opposing effects
- Example: Beta-agonist bronchodilators + beta-blockers

**Potentiation:**
- One drug enhancing the effect of another
- Example: Trimethoprim + Sulfamethoxazole

**Additive Effects:**
- Two drugs from same class
- Example: Multiple NSAIDs (GI bleeding risk)

### 2.5 Emergency Department DDI Critical Pairs

#### 2.5.1 High-Severity Anticoagulant Interactions

**Warfarin (RxCUI 11289) Interactions:**
- **+ NSAIDs** → Increased bleeding risk (GI hemorrhage)
- **+ Fluoroquinolones** → Enhanced anticoagulation (CYP2C9 inhibition)
- **+ Rifampin** → Decreased anticoagulation (CYP450 induction)
- **+ Amiodarone** → INR elevation (metabolism inhibition)

**DOAC Interactions:**
- **Apixaban + Strong CYP3A4/P-gp Inhibitors** (ketoconazole, ritonavir)
- **Rivaroxaban + Antiplatelet Agents** → Bleeding risk

#### 2.5.2 Opioid Interactions

**Morphine + CNS Depressants:**
- Benzodiazepines (Midazolam, Lorazepam)
- Alcohol
- Sedating antihistamines
- **Effect:** Respiratory depression, sedation, death

**Tramadol + Serotonergic Agents:**
- SSRIs, SNRIs
- MAO inhibitors
- **Effect:** Serotonin syndrome

#### 2.5.3 Cardiovascular Drug Interactions

**Beta-Blockers + Calcium Channel Blockers:**
- **Metoprolol + Diltiazem** → Bradycardia, heart block
- Monitor: Heart rate, blood pressure, ECG

**QT-Prolonging Agents:**
- **Ondansetron + Fluoroquinolones**
- **Azithromycin + Amiodarone**
- **Effect:** Torsades de pointes risk

#### 2.5.4 Antibiotic Interactions

**Fluoroquinolones + Antacids:**
- **Levofloxacin + Calcium/Magnesium/Aluminum**
- **Effect:** Reduced antibiotic absorption (chelation)
- **Mitigation:** Separate administration by 2-4 hours

**Linezolid + MAO Inhibitors:**
- **Effect:** Hypertensive crisis
- **Mechanism:** Linezolid has MAOI activity

### 2.6 Knowledge Graph Approaches to DDI Prediction

#### 2.6.1 Graph Embedding Methods

**KnowDDI Framework:**
- Uses graph neural networks on biomedical knowledge graphs
- Learns drug representations from neighborhood information
- Adaptive feature aggregation
- State-of-the-art prediction performance

**Predicting Rich DDI:**
- Large-scale drug knowledge graph from multiple sources
- Graph embedding + biomedical text embedding
- Link prediction for novel DDI discovery

#### 2.6.2 Linked Data and Semantic Web

**Approach:** RDF representation of EHR data linked to public DDI resources
**Benefits:**
- Integration of heterogeneous data sources
- Automated reasoning over drug properties
- Real-time DDI checking at point of care

**Example Query Pattern:**
```sparql
SELECT ?drug1 ?drug2 ?interaction ?severity
WHERE {
  ?patient :takingMedication ?drug1 .
  ?patient :takingMedication ?drug2 .
  ?drug1 :hasRxCUI ?rxcui1 .
  ?drug2 :hasRxCUI ?rxcui2 .
  ?interaction :involves ?rxcui1, ?rxcui2 .
  ?interaction :hasSeverity ?severity .
  FILTER (?severity = "major" || ?severity = "severe")
}
```

### 2.7 DDI Severity Classification

**Contraindicated:**
- Absolute prohibition of concurrent use
- Example: MAO inhibitors + SSRIs

**Major:**
- Potentially life-threatening or causing permanent damage
- Requires immediate intervention
- Example: Warfarin + NSAIDs (bleeding risk)

**Moderate:**
- May cause deterioration of condition
- May require intervention
- Example: Ciprofloxacin + Theophylline

**Minor:**
- Limited clinical significance
- May increase monitoring
- Example: Minor CYP450 interactions

## 3. Medication Timing and Administration Patterns

### 3.1 Medication Order Urgency Classification

#### 3.1.1 STAT Orders

**Definition:** One-time dose administered immediately without delay due to urgency.

**Timing Requirement:** Within 5-15 minutes (institution-specific)

**Emergency Department STAT Examples:**
- **Epinephrine 0.3 mg IM STAT** (anaphylaxis)
- **Naloxone 0.4 mg IV STAT** (opioid overdose)
- **Dextrose 50% 25g IV STAT** (severe hypoglycemia)
- **Labetalol 10 mg IV STAT** (hypertensive emergency)
- **Aspirin 325 mg PO STAT** (acute coronary syndrome)
- **Alteplase IV STAT** (acute ischemic stroke)

**Knowledge Graph Representation:**
```json
{
  "orderType": "STAT",
  "urgency": "immediate",
  "maxDelayMinutes": 5,
  "requiresDocumentation": {
    "indication": "required",
    "responseTime": "required",
    "administrationTime": "required"
  }
}
```

#### 3.1.2 NOW/First Dose Orders

**Definition:** First dose or loading dose administered promptly but not emergently.

**Timing Requirement:** Within 1-2 hours

**ED Examples:**
- **Ceftriaxone 2g IV NOW** (sepsis)
- **Vancomycin 15 mg/kg IV NOW** (suspected MRSA)
- **Heparin 80 units/kg IV bolus NOW** (PE/DVT)

#### 3.1.3 PRN (Pro Re Nata) Orders

**Definition:** Medication given "as needed" for specific indications with defined parameters.

**Critical Requirements:**
- **Specific indication** (must be documented in order)
- **Maximum frequency** (e.g., "q4-6h PRN")
- **Maximum dose in 24h**
- **Assessment before administration**
- **Hold parameters**

**ED PRN Order Examples:**

**Pain Management:**
```
Morphine Sulfate 2-4 mg IV q2h PRN severe pain (>7/10)
- Hold for: Respiratory rate < 12, sedation score > 2
- Max: 20 mg in 24h
- Indication: Only for acute pain, not chronic pain
```

```
Acetaminophen 650 mg PO q6h PRN mild-moderate pain (4-6/10)
- Max: 3000 mg in 24h
- Hold for: Hepatic dysfunction
```

**Antiemetic:**
```
Ondansetron 4 mg IV q6h PRN nausea/vomiting
- Hold for: QTc > 500 ms
- Max: 16 mg in 24h (per FDA guidance)
```

**Anxiety/Agitation:**
```
Lorazepam 0.5-1 mg PO/IV q4h PRN anxiety/agitation
- Hold for: Respiratory rate < 10, excessive sedation
- Max: 4 mg in 24h
- Assess: Sedation level before each dose
```

**Constipation:**
```
Docusate 100 mg PO BID PRN constipation
- Indication: No bowel movement in 48h
```

**Insomnia:**
```
Melatonin 3 mg PO qHS PRN insomnia
- Indication: Only for sleep, not anxiety
- Timing: At bedtime only
```

**Nursing Responsibilities for PRN:**
- Verify time of previous dose
- Confirm indication matches order
- Document assessment before administration
- Cannot offer PRN for different indication than ordered

### 3.2 Scheduled Medication Administration

#### 3.2.1 Time-Critical Scheduled Medications

**ISMP Definition:** Medications where early or delayed administration >30 minutes may cause harm or suboptimal therapy.

**Categories:**

**Critical/Time-Critical:**
- Anticoagulants (e.g., Heparin, Enoxaparin)
- Insulin (rapid-acting, scheduled)
- Antibiotics (first dose, time-dependent killing)
- Immunosuppressants
- Seizure medications
- Parkinson's medications

**Timing Window:** ±30 minutes of scheduled time

**ED Examples:**
- **Enoxaparin 1 mg/kg SQ q12h** (VTE treatment)
- **Insulin aspart per sliding scale AC + qHS**
- **Levetiracetam 500 mg PO q12h** (seizure prophylaxis)

#### 3.2.2 Non-Time-Critical Scheduled Medications

**ISMP Definition:** Medications administered on regular schedule but flexible timing acceptable.

**Timing Window:**
- **Daily medications:** ±2 hours
- **>Daily, ≤Weekly:** ±2 hours
- **BID, TID, QID:** ±1 hour (most institutions)

**ED Examples:**
- **Pantoprazole 40 mg PO daily** (stress ulcer prophylaxis)
- **Aspirin 81 mg PO daily** (antiplatelet therapy)
- **Atorvastatin 40 mg PO qHS** (cholesterol management)

#### 3.2.3 Standard Dosing Frequencies

**QD/Daily:** Once daily
- Example: Levothyroxine 50 mcg PO daily

**BID:** Twice daily (typically 08:00, 20:00)
- Example: Metoprolol 25 mg PO BID

**TID:** Three times daily (typically 08:00, 14:00, 20:00)
- Example: Amoxicillin 500 mg PO TID

**QID:** Four times daily (typically 06:00, 12:00, 18:00, 22:00)
- Example: Acetaminophen 650 mg PO QID

**Q4h, Q6h, Q8h, Q12h:** Fixed interval dosing
- Example: Vancomycin 1g IV q12h
- Example: Morphine PCA continuous infusion

**QHS:** At bedtime
- Example: Melatonin 3 mg PO qHS

**AC (ante cibum):** Before meals
- Example: Insulin lispro per sliding scale AC

**PC (post cibum):** After meals
- Example: Pancrelipase 1 capsule PO PC

### 3.3 CMS and Regulatory Guidelines

**CMS Updated Guidance (2012):**
- Hospitals have **flexibility** to establish timing policies
- Must identify medications requiring exact/precise timing
- One-size-fits-all 30-minute window is **precarious**
- Policies should consider:
  - Nature of prescribed medication
  - Specific clinical applications
  - Patient-specific needs

**Not Eligible for Scheduled Times:**
- STAT doses
- First-time/loading doses
- One-time doses
- Procedure-timed doses
- PRN medications

### 3.4 ED-Specific Timing Considerations

#### 3.4.1 Sepsis Bundle Timing

**3-Hour Bundle:**
- Blood cultures before antibiotics
- **Antibiotics within 1 hour of sepsis recognition** (ideally <45 min)
- Lactate measurement
- Fluid resuscitation (30 mL/kg crystalloid)

**Knowledge Graph:**
```json
{
  "condition": "sepsis",
  "timingRequirement": {
    "antibiotics": {
      "targetMinutes": 60,
      "measurementStart": "sepsis_recognition",
      "priority": "critical"
    },
    "bloodCultures": {
      "sequence": "before_antibiotics",
      "required": true
    }
  }
}
```

#### 3.4.2 Stroke Timing (tPA Administration)

**Alteplase for Acute Ischemic Stroke:**
- **Door-to-needle goal:** <60 minutes
- **Window:** Within 3-4.5 hours of symptom onset
- Time-critical: Every minute matters ("Time is Brain")

#### 3.4.3 ACS Medication Timing

**MONA Protocol (updated):**
- **Morphine:** PRN for chest pain
- **Oxygen:** Only if hypoxic (SpO2 <90%)
- **Nitroglycerin:** 0.4 mg SL q5min × 3 (if BP adequate)
- **Aspirin:** 162-325 mg STAT (chewed)

**Door-to-Balloon Time:** <90 minutes for STEMI

### 3.5 Medication Timing Ontology

**Temporal Properties:**
```json
{
  "medicationOrder": {
    "rxcui": "855302",
    "drugName": "Warfarin Sodium 2 MG Oral Tablet",
    "schedule": {
      "frequency": "daily",
      "timing": "17:00",
      "timingWindow": {
        "early": "-120min",
        "late": "+120min"
      },
      "timeClass": "non-time-critical",
      "dayOfWeek": ["all"],
      "duration": "ongoing"
    },
    "orderType": "scheduled",
    "firstDose": {
      "urgency": "routine",
      "targetTime": "next_scheduled_time"
    }
  }
}
```

## 4. RxNorm to NDC Mappings

### 4.1 Overview of NDC Structure

**National Drug Code (NDC):** Unique 10-11 digit identifier assigned by FDA to drug products.

**Format:** Labeler-Product-Package
- **Labeler (4-5 digits):** Manufacturer or distributor
- **Product (3-4 digits):** Strength, dosage form, formulation
- **Package (1-2 digits):** Package size and type

**Example NDC:** 00093-1045-01
- Labeler: 00093 (Teva Pharmaceuticals)
- Product: 1045 (Warfarin Sodium 2 mg tablet)
- Package: 01 (100-count bottle)

### 4.2 RxNorm-NDC Relationship

**Cardinality:**
- **One RxCUI → Multiple NDCs** (different manufacturers, package sizes)
- **One NDC → One RxCUI** (after NLM discontinued cascading)

**Term Types with NDC Associations:**
- SCD (Semantic Clinical Drug)
- SBD (Semantic Branded Drug)
- GPCK (Generic Pack)
- BPCK (Branded Pack)

**Example Mapping:**
```
RxCUI 855302 (Warfarin Sodium 2 MG Oral Tablet)
  ├─ NDC 00093-1045-01 (Teva, 100 count)
  ├─ NDC 00093-1045-10 (Teva, 1000 count)
  ├─ NDC 00378-4002-01 (Mylan, 100 count)
  └─ NDC 00591-3239-01 (Watson, 100 count)
```

### 4.3 Accessing NDC-RxCUI Mappings

#### 4.3.1 RxNorm Files (SQL)

**RXNSAT Table:**
- NDC values in Attribute Value (ATV) column
- Where Attribute Name (ATN) = 'NDC'
- SAB = 'RXNORM' for NLM-curated NDCs

**SQL Query Example:**
```sql
SELECT r.rxcui, r.str AS drug_name, s.atv AS ndc
FROM rxnconso r
JOIN rxnsat s ON r.rxcui = s.rxcui
WHERE s.atn = 'NDC'
  AND s.sab = 'RXNORM'
  AND r.tty IN ('SCD', 'SBD', 'GPCK', 'BPCK')
  AND r.str LIKE '%warfarin%'
ORDER BY r.rxcui, s.atv;
```

#### 4.3.2 RxNorm API

**getNDCs Function:**
- Returns active NDCs for a given RxCUI
- Only SAB=RXNORM curated associations
- REST API: `https://rxnav.nlm.nih.gov/REST/ndcs.json?rxcui=855302`

**Response Example:**
```json
{
  "ndcGroup": {
    "conceptName": "Warfarin Sodium 2 MG Oral Tablet",
    "conceptNdc": [
      "00093-1045-01",
      "00093-1045-10",
      "00378-4002-01",
      "00591-3239-01"
    ]
  }
}
```

**getNDCProperties Function:**
- Returns properties for specific NDC
- Includes packaging, labeler information
- REST API: `https://rxnav.nlm.nih.gov/REST/ndcproperties.json?id=00093-1045-01`

#### 4.3.3 Third-Party Tools

**NDCList.com:**
- RxNorm to NDC crosswalk
- Search by RxCUI, concept name, or term type
- Web-based interface

**GitHub ndc_map:**
- Maps NDCs to drug classifications (ATC)
- Queries RxNav API
- Programmatic access

**John Snow Labs NLP:**
- Pretrained model: rxnorm_ndc_mapper
- Maps RxNorm/RxNorm Extension to NDC
- Healthcare NLP pipeline integration

### 4.4 Clinical Use Cases for NDC Mapping

#### 4.4.1 Prescription Dispensing

**Workflow:**
1. Prescriber orders at RxCUI level (SCD/SBD)
2. EHR sends RxCUI to pharmacy system
3. Pharmacy maps to formulary-specific NDC
4. Dispense specific manufacturer product
5. Record NDC in medication administration record

**Example:**
- **Order:** Warfarin Sodium 2 MG Oral Tablet (RxCUI 855302)
- **Formulary Selection:** Teva generic (NDC 00093-1045-01)
- **Dispense:** 30-day supply
- **Bill:** Using NDC for insurance claims

#### 4.4.2 Medication Reconciliation

**Challenge:** EHR uses RxCUI, external records contain NDC

**Solution:**
```python
# Pseudocode for NDC reconciliation
def reconcile_medication(external_ndc):
    # Look up RxCUI from NDC
    rxcui = rxnorm_api.get_rxcui_from_ndc(external_ndc)

    # Normalize to SCD level
    scd_rxcui = rxnorm_api.get_related(rxcui, tty='SCD')

    # Check against current medications
    if scd_rxcui in patient.current_medications:
        return "duplicate"
    else:
        return "new_medication"
```

#### 4.4.3 Medication Shortages and Substitution

**Scenario:** NDC 00093-1045-01 (Teva Warfarin 2mg) on shortage

**Resolution:**
1. Identify all NDCs for RxCUI 855302
2. Check formulary for available alternatives
3. Substitute with NDC 00378-4002-01 (Mylan)
4. Update ordering system
5. Notify prescribers of substitution

#### 4.4.4 Billing and Reimbursement

**Insurance Claims:**
- **NDC required** for medication billing
- Maps to J-codes, HCPCS for procedure billing
- RxCUI provides semantic consistency
- NDC provides product-specific detail

**Example Mapping:**
- **RxCUI 1653662** - Ondansetron 4 MG/2 ML Injectable Solution
- **NDC 63323-0761-02** - Fresenius Kabi ondansetron 4mg/2mL vial
- **HCPCS J2405** - Ondansetron injection, per 1 mg

### 4.5 NDC Coverage and Limitations

**Coverage:**
- RxNorm provides "best or second-best coverage" of prescription NDCs
- Covers major manufacturers and distributors
- Regular updates from FDA and commercial sources

**Limitations:**
- Not all NDCs mapped (discontinued, new products)
- Package size variations not always complete
- Lag time for new drug approvals
- Historical NDCs may be inactive

**Active vs. Inactive NDCs:**
- RxNorm API returns only **active** NDCs
- Historical lookups require RXNSAT archive files
- NDC reuse by FDA (10-year period) creates ambiguity

## 5. Emergency Department Medication Patterns

### 5.1 Most Frequently Administered ED Medications (2006-2019)

#### 5.1.1 Top 10 ED Medications by Volume

**Rank 1: Ondansetron (RxCUI 4337)**
- **Volume:** 255.1 million ED visits
- **Drug Class:** 5-HT3 Receptor Antagonist (antiemetic)
- **Common Formulations:**
  - Ondansetron 4 mg IV (RxCUI 1653662)
  - Ondansetron 4 mg ODT (RxCUI 835695)
  - Ondansetron 8 mg Oral Tablet
- **Indications:** Nausea, vomiting, gastroenteritis
- **ED Dosing:** 4 mg IV/PO q6h PRN nausea
- **Safety Concern:** QTc prolongation (FDA limit 16 mg/24h)

**Rank 2: 0.9% Normal Saline (Sodium Chloride)**
- **Volume:** 251.3 million ED visits
- **RxCUI:** Multiple (concentration-dependent)
- **Indications:** Volume resuscitation, medication dilution, IV access maintenance
- **Typical Orders:**
  - NS 1000 mL IV bolus (hypovolemia)
  - NS 125 mL/hr IV continuous (maintenance)
  - NS 20 mL flush IV push

**Rank 3: Ibuprofen (RxCUI 5640)**
- **Volume:** 188.5 million ED visits
- **Drug Class:** NSAID
- **Common Formulations:**
  - Ibuprofen 600 mg PO (RxCUI 310965 for 200mg)
  - Ibuprofen 800 mg PO
- **Indications:** Mild-moderate pain, fever, inflammation
- **ED Dosing:** 400-800 mg PO q6-8h
- **Contraindications:** GI bleeding, renal dysfunction, aspirin allergy

**Rank 4: Acetaminophen (Tylenol)**
- **RxCUI 161** (ingredient)
- **Common Formulations:**
  - RxCUI 307667 - Acetaminophen 160 MG Oral Capsule
  - RxCUI 2264779 - Acetaminophen 325 MG Chewable Tablet
  - Acetaminophen 650 mg PO
  - Acetaminophen 1000 mg IV
- **Indications:** Mild-moderate pain, fever
- **ED Dosing:** 650-1000 mg PO/IV q6h
- **Max Dose:** 3000-4000 mg/24h (lower in liver disease)

**Rank 5: Morphine Sulfate (RxCUI 7052)**
- **Shortage Duration:** 3,202 days (longest)
- **Common Formulations:**
  - Morphine Sulfate 2 mg IV
  - Morphine Sulfate 4 mg IV
  - Morphine Sulfate 10 mg Injectable Solution
- **Indications:** Severe pain, acute coronary syndrome
- **ED Dosing:** 2-10 mg IV q2-4h PRN
- **Safety:** Respiratory depression, requires pulse oximetry

**Others in Top 10:**
- **Ketorolac** - NSAID for moderate-severe pain
- **Diphenhydramine** - Antihistamine (allergic reactions, dystonia)
- **Lorazepam** - Benzodiazepine (anxiety, seizures)
- **Ceftriaxone** - Third-generation cephalosporin (infections)
- **Pantoprazole** - Proton pump inhibitor (GI protection)

### 5.2 ED Medication Patterns by Clinical Condition

#### 5.2.1 Acute Coronary Syndrome (21.79% of ED admissions)

**MONA Protocol Medications:**

**Aspirin (RxCUI 1191):**
- Formulation: Aspirin 162-325 mg Chewable Tablet
- Dose: 325 mg PO STAT (chewed)
- Mechanism: Irreversible COX-1 inhibition, antiplatelet

**Nitroglycerin:**
- Formulation: Nitroglycerin 0.4 mg SL Tablet
- Dose: 0.4 mg SL q5min × 3 PRN chest pain
- Hold: SBP <90 mmHg, recent PDE5 inhibitor use

**Morphine Sulfate:**
- Dose: 2-4 mg IV q5-15min PRN pain
- Caution: May worsen outcomes in NSTEMI (use sparingly)

**Heparin/Anticoagulation:**
- Unfractionated Heparin: 60-80 units/kg IV bolus
- Enoxaparin: 1 mg/kg SQ q12h
- Fondaparinux: 2.5 mg SQ daily

**Dual Antiplatelet Therapy:**
- Aspirin 325 mg + Clopidogrel 600 mg loading
- Or Aspirin + Ticagrelor 180 mg loading

#### 5.2.2 Sepsis/Septic Shock

**Antimicrobial Regimens:**

**Ceftriaxone (RxCUI 2582):**
- **Frequency:** 32.69% of ED antimicrobials
- Formulation: Ceftriaxone 2 g IV
- Dose: 1-2 g IV q24h
- Spectrum: Broad gram-negative, some gram-positive
- Indications: Community-acquired pneumonia, UTI, meningitis

**Vancomycin:**
- Formulation: Vancomycin 1 g IV
- Dose: 15-20 mg/kg IV q8-12h (based on renal function)
- Spectrum: MRSA, resistant gram-positive
- Monitoring: Trough levels (goal 15-20 mcg/mL for severe infections)

**Piperacillin-Tazobactam:**
- Dose: 3.375-4.5 g IV q6-8h
- Spectrum: Extended gram-negative, anaerobes
- Indication: Hospital-acquired/healthcare-associated infections

**Metronidazole:**
- Dose: 500 mg IV q8h
- Spectrum: Anaerobes (C. difficile, intra-abdominal)

**Azithromycin:**
- Dose: 500 mg IV/PO daily
- Indication: Atypical pneumonia coverage
- DDI: QTc prolongation

**Supportive Medications:**

**Norepinephrine:**
- First-line vasopressor
- Dose: 0.05-2 mcg/kg/min IV (titrate to MAP ≥65)

**Lactated Ringer's/Normal Saline:**
- Initial: 30 mL/kg IV bolus
- Reassess after each liter

#### 5.2.3 Nausea/Vomiting/Gastroenteritis

**Ondansetron (86.53% of antiemetic use):**
- Dose: 4 mg IV/PO q6h PRN
- Advantages: Effective, minimal sedation
- Disadvantages: QTc prolongation, constipation

**Metoclopramide:**
- Dose: 10 mg IV/PO q6h PRN
- Mechanism: Dopamine antagonist, prokinetic
- Risk: Extrapyramidal symptoms (dystonia, akathisia)

**Prochlorperazine:**
- Dose: 5-10 mg IV/IM q6h PRN
- Risk: Sedation, dystonia

**Promethazine:**
- Dose: 12.5-25 mg IV/IM/PO q4-6h PRN
- Risk: Severe sedation, tissue necrosis if extravasates

#### 5.2.4 Pain Management

**Mild Pain (1-3/10):**
- Acetaminophen 650-1000 mg PO q6h
- Ibuprofen 400-600 mg PO q6h

**Moderate Pain (4-6/10):**
- Ketorolac 15-30 mg IV/IM (max 5 days)
- Tramadol 50-100 mg PO q6h PRN
- Acetaminophen 1000 mg IV

**Severe Pain (7-10/10):**
- Morphine 2-10 mg IV q2-4h
- Hydromorphone 0.5-2 mg IV q2-4h
- Fentanyl 25-100 mcg IV q1h PRN (rapid onset)

**Renal Colic:**
- Ketorolac 30 mg IV (preferred over opioids)
- Morphine 4-8 mg IV PRN if inadequate relief

**Migraine:**
- Metoclopramide 10 mg IV + Diphenhydramine 25 mg IV
- Prochlorperazine 10 mg IV
- NSAIDs (Ketorolac 30 mg IV)

#### 5.2.5 Seizures

**Benzodiazepines (First-line):**
- **Lorazepam** 4 mg IV (preferred, longer duration)
- **Midazolam** 10 mg IM/IV (alternative if no IV access)
- **Diazepam** 10 mg IV/rectal

**Second-line Anticonvulsants:**
- **Levetiracepam** 1000-3000 mg IV
- **Fosphenytoin** 20 mg PE/kg IV
- **Valproic Acid** 20-40 mg/kg IV

**Refractory Status Epilepticus:**
- Propofol infusion
- Midazolam infusion
- Pentobarbital

#### 5.2.6 Hypertensive Emergency

**Labetalol (Most Common Agent):**
- Dose: 10-20 mg IV q10-15min (max 300 mg)
- Or: 0.5-2 mg/min IV infusion
- Mechanism: Beta-blocker + alpha-blockade (vasodilation)
- Contraindications: Asthma, cocaine-related HTN, aortic dissection (use esmolol)

**Nicardipine:**
- Dose: 5 mg/hr IV, titrate by 2.5 mg/hr q5-15min
- Max: 15 mg/hr
- Preferred for: Stroke, eclampsia, renal disease

**Hydralazine:**
- Dose: 10-20 mg IV q4-6h
- Use: Eclampsia/preeclampsia
- Onset: 10-30 minutes (slower)

#### 5.2.7 Allergic Reactions/Anaphylaxis

**Epinephrine:**
- Dose: 0.3-0.5 mg IM (1:1000) q5-15min
- Location: Anterolateral thigh
- Severe anaphylaxis: 0.1 mg IV slow push (1:10,000)

**Diphenhydramine:**
- Dose: 25-50 mg IV/PO q6h
- Mechanism: H1 antihistamine

**Famotidine or Ranitidine:**
- Dose: 20 mg IV (famotidine)
- Mechanism: H2 blocker

**Methylprednisolone:**
- Dose: 125 mg IV
- Purpose: Prevent biphasic reaction

**Albuterol:**
- Dose: 2.5 mg nebulized q20min PRN bronchospasm

### 5.3 ED Medication Shortages Impact

**Critical Finding:** 60.4% of U.S. ED visits in 2019 involved medications experiencing shortages.

**Most Affected Classes:**
- **Opioid analgesics** (morphine: 3,202-day shortage)
- **Sedatives** (midazolam, lorazepam)
- **Antimicrobials** (ceftriaxone, piperacillin-tazobactam)
- **Vasopressors** (norepinephrine, epinephrine)
- **Emergency antidotes** (calcium gluconate)

**Clinical Impact:**
- Medication substitutions
- Delayed treatment
- Dosing errors
- Increased costs
- Rationing of critical medications

**Mitigation Strategies:**
- Multi-source procurement
- Therapeutic substitution protocols
- Real-time inventory monitoring
- Alternative dosing strategies

### 5.4 Medication Safety in Acute Care

**High-Alert Medications in ED:**
1. Anticoagulants (heparin, warfarin)
2. Insulin
3. Opioids
4. Sedatives (benzodiazepines, propofol)
5. Neuromuscular blockers
6. Concentrated electrolytes (KCl, CaCl2)

**Error Prevention Strategies:**
- Barcode medication administration
- RxNorm-based CPOE systems
- Automated DDI checking
- Smart pump drug libraries
- Independent double-checks for high-alert meds

## 6. Knowledge Graph Implementation Recommendations

### 6.1 Ontology Structure for ED Medication System

**Core Entities:**
- Medication (RxCUI as primary identifier)
- Patient
- Order
- Administration
- Allergy
- Drug-Drug Interaction
- Indication
- Contraindication

**Relationships:**
- hasIngredient (SCD → IN)
- hasBrandName (SBD → BN)
- mapsToNDC (SCD/SBD → NDC)
- interactsWith (RxCUI ↔ RxCUI)
- contraindicatedIn (RxCUI → Condition)
- indicatedFor (RxCUI → Condition)
- administeredTo (Order → Patient)

### 6.2 Example RDF Representation

```turtle
@prefix rxnorm: <http://purl.bioontology.org/ontology/RXNORM/> .
@prefix ndc: <http://example.org/ndc/> .
@prefix patient: <http://example.org/patient/> .

rxnorm:855302 a rxnorm:SemanticClinicalDrug ;
    rdfs:label "Warfarin Sodium 2 MG Oral Tablet" ;
    rxnorm:hasIngredient rxnorm:11289 ;
    rxnorm:hasStrength "2 MG" ;
    rxnorm:hasDoseForm "Oral Tablet" ;
    rxnorm:mapsToNDC ndc:00093-1045-01, ndc:00378-4002-01 ;
    rxnorm:interactsWith rxnorm:5640 ;
    rxnorm:interactionSeverity "major" ;
    rxnorm:contraindicatedIn <http://snomed.info/id/75544006> . # Active bleeding

patient:12345 a :Patient ;
    :hasAllergy rxnorm:11289 ;
    :takingMedication rxnorm:855302 .
```

## 7. Conclusion

RxNorm provides the foundational semantic infrastructure for medication knowledge graphs in acute care settings. Key takeaways for hybrid reasoning systems:

1. **Use SCD/SBD level** for clinical ordering (RxCUI with ingredient + strength + form)
2. **Leverage IN level** for allergy checking and therapeutic class assignments
3. **Map to NDC** for dispensing, billing, and product-specific tracking
4. **Integrate DDI ontologies** (DINTO, DrugBank, NDF-RT) for safety checking
5. **Encode timing semantics** (STAT, PRN, scheduled) for workflow automation
6. **Model ED-specific patterns** for common clinical scenarios

**Future Directions:**
- Real-time knowledge graph updates from FDA/NLM
- Machine learning for DDI prediction on incomplete graphs
- Temporal reasoning for medication timing optimization
- Integration with clinical pathways and order sets
- Personalized DDI risk scoring based on patient factors

## References

1. [RxNorm Technical Documentation](https://www.nlm.nih.gov/research/umls/rxnorm/docs/techdoc.html)
2. [RxNorm Appendix 5 - Term Types](https://www.nlm.nih.gov/research/umls/rxnorm/docs/appendix5.html)
3. [Drug-Drug Interaction Ontologies (ResearchGate)](https://www.researchgate.net/publication/286834043_An_ontology_for_drug-drug_interactions)
4. [ISMP Guidelines for Timely Administration](https://www.ismp.org/guidelines/timely-administration-scheduled-medications-acute)
5. [RxNorm to NDC Mapping (NLM)](https://www.nlm.nih.gov/research/umls/user_education/quick_tours/RxNorm/ndc_rxcui/NDC_RXCUI_DrugName.html)
6. [ED Medication Shortages Study (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8862149/)
7. [AAEM 50 Drugs Every Emergency Physician Should Know](https://www.aaemrsa.org/education/50-drugs/)
8. [KnowDDI: Accurate Drug-Drug Interaction Prediction (Nature)](https://www.nature.com/articles/s43856-024-00486-y)
9. [CRESCENDDI Reference Set (Nature Scientific Data)](https://www.nature.com/articles/s41597-022-01159-y)
10. [RxNorm Overview (NLM)](https://www.nlm.nih.gov/research/umls/rxnorm/overview.html)
