# SNOMED CT Analysis for Clinical Knowledge Graphs

## Executive Summary

SNOMED CT (Systematized Nomenclature of Medicine - Clinical Terms) is the world's most comprehensive clinical terminology system, containing over 357,000 health care concepts with unique meanings and formal logic-based definitions organized into hierarchies. This analysis explores SNOMED CT's structure, sepsis-related concepts, temporal modeling capabilities, Expression Constraint Language (ECL), and mapping challenges with ICD-10, with specific focus on applications in acute care knowledge graphs.

---

## 1. SNOMED CT Hierarchy: The 19 Top-Level Concepts

SNOMED CT organizes all clinical knowledge under a single root concept (138875005 |SNOMED CT Concept|) with 19 top-level hierarchies that branch into increasingly specific subtypes. Unlike ICD-10's mono-hierarchical structure, SNOMED CT is multi-hierarchical, allowing a single concept to exist in multiple branches simultaneously.

### 1.1 The Three Organizational Types

The 19 top-level hierarchies are organized into three main categories:

**Object Hierarchies:**
- **Clinical finding** (404684003) - Results of clinical observations, assessments, or judgments, including both normal and abnormal clinical states (e.g., asthma, headache, normal breath sounds). This hierarchy contains concepts used to represent diagnoses.
- **Procedure** (71388002) - Activities performed in healthcare provision, including invasive procedures, medication administration, imaging, education, therapies, and administrative procedures.
- **Body structure** (123037004) - Normal and abnormal anatomical structures including congenital anomalies and acquired body structures.
- **Organism** (410607006) - Living organisms including bacteria, viruses, fungi, parasites, and other microorganisms.
- **Substance** (105590001) - Chemical substances and materials relevant to healthcare.
- **Pharmaceutical/biologic product** (373873005) - Medicinal products, vaccines, and biological agents.
- **Physical object** (260787004) - Non-biological objects used in healthcare including medical devices, instruments, and materials.
- **Specimen** (123038009) - Samples obtained for examination, testing, or analysis.

**Value Hierarchies:**
- **Qualifier value** (362981000) - Values for SNOMED CT attributes that are not subtypes of other top-level concepts (e.g., severity modifiers, laterality values).
- **Observable entity** (363787002) - Entities that can be observed or measured (e.g., blood pressure, temperature, heart rate).
- **Physical force** (78621006) - Forces that can cause injury or be used therapeutically (e.g., radiation, heat, mechanical force).
- **Environment or geographical location** (308916002) - Environmental contexts and geographic locations relevant to healthcare.
- **Social context** (48176007) - Social circumstances, living situations, and social determinants of health.
- **Staging and scales** (370115009) - Standardized scales, scores, and staging systems used in clinical assessment.
- **Situation with explicit context** (243796009) - Clinical situations with explicit contextual information.

**Miscellaneous Hierarchies:**
- **Record artifact** (419891008) - Content created for documenting record events or states of affairs.
- **Event** (272379006) - Occurrences and happenings in healthcare contexts.
- **SNOMED CT Model Component** (900000000000441003) - Technical metadata supporting SNOMED CT releases.
- **Special concept** (370136006) - Specialized concepts for specific purposes.

### 1.2 Multi-Hierarchical Nature: A Key Differentiator

The multi-hierarchical structure allows concepts to have multiple parents, enabling more flexible and accurate representation of clinical knowledge. For example, "lung cancer" can be found in both the neoplasm hierarchy and the respiratory system hierarchy, whereas in ICD-10's mono-hierarchical structure, it would only appear in the neoplasm table.

### 1.3 Relationship Types: The IS-A Hierarchy

The primary relationship in SNOMED CT is `116680003 |is a|`, which organizes all concepts in a tree structure. This relationship enables:
- Logical subsumption reasoning
- Concept specialization and generalization
- Inheritance of defining characteristics
- Semantic query expansion

**Example:**
```
138875005 |SNOMED CT Concept|
  └─ 404684003 |Clinical finding|
      └─ 64572001 |Disease|
          └─ 87628006 |Bacterial infectious disease|
              └─ 91302008 |Sepsis|
                  └─ 10001005 |Bacterial sepsis|
```

---

## 2. Sepsis-Related Concepts with Actual SCTIDs

### 2.1 Core Sepsis Concepts

SNOMED CT provides a rich taxonomy for representing sepsis and related infectious conditions. The core sepsis concept and its relationships are:

| Concept | SCTID | Fully Specified Name | Description |
|---------|-------|---------------------|-------------|
| **Sepsis** | 91302008 | Sepsis (disorder) | Systemic inflammatory response to infection |
| **Bacterial sepsis** | 10001005 | Bacterial sepsis (disorder) | Sepsis caused by bacterial organisms |
| **Clinical sepsis** | 447931005 | Clinical sepsis (disorder) | Sepsis diagnosed clinically without microbiological confirmation |
| **Severe sepsis** | 76571007 | Sepsis with acute organ dysfunction (disorder) | Sepsis with evidence of organ dysfunction |
| **Septic shock** | 76571007 | Septic shock (disorder) | Severe sepsis with persistent hypotension |
| **Neonatal sepsis** | 75053002 | Neonatal sepsis (disorder) | Sepsis occurring in newborns |
| **Puerperal sepsis** | 398254007 | Puerperal sepsis (disorder) | Sepsis occurring postpartum |

### 2.2 Related Infectious Disease Concepts

| Concept | SCTID | Clinical Relevance |
|---------|-------|-------------------|
| **Bacteremia** | 5758002 | Presence of bacteria in blood without systemic inflammation |
| **Systemic inflammatory response syndrome** | 238150007 | SIRS - inflammatory response that may or may not be due to infection |
| **Multiple organ dysfunction syndrome** | 57653000 | MODS - progression of severe sepsis |
| **Fungemia** | 309095005 | Fungal organisms in bloodstream |
| **Viremia** | 49872002 | Viral organisms in bloodstream |

### 2.3 Organism-Specific Sepsis Concepts

SNOMED CT allows for highly specific representation of sepsis based on causative organisms:

| Concept | SCTID | Causative Agent |
|---------|-------|-----------------|
| **Staphylococcal sepsis** | 3434004 | Staphylococcus species |
| **Streptococcal sepsis** | 58750007 | Streptococcus species |
| **Escherichia coli sepsis** | 62260004 | E. coli |
| **Pseudomonal sepsis** | 58752004 | Pseudomonas species |
| **Candida sepsis** | 8745008 | Candida species (fungal) |

### 2.4 Anatomic Site-Specific Sepsis

| Concept | SCTID | Finding Site |
|---------|-------|--------------|
| **Intra-abdominal sepsis** | 449008003 | Abdominal cavity structure |
| **Pelvic sepsis** | 23879005 | Pelvic cavity structure |
| **Urinary tract infection with sepsis** | Combined coding | Urinary tract structure |

### 2.5 Sepsis Concept Relationships

The defining relationships for **91302008 |Sepsis|** include:

```
91302008 |Sepsis| :
  116676008 |Associated morphology| = 409774005 |Inflammatory morphology|
  370135005 |Pathological process| = 441862004 |Infectious process|
  246075003 |Causative agent| = 264395009 |Microorganism|
  363698007 |Finding site| = 122496007 |Entire blood|
```

### 2.6 Clinical Sepsis (447931005) - A Special Case

Clinical sepsis represents a diagnostically important concept where:
- Blood was not cultured OR no microorganism was isolated
- No apparent infection at another site
- Patient presents with fever, hypotension, or oliguria
- Appropriate antimicrobial therapy for sepsis was initiated

This concept enables proper documentation when microbiological confirmation is unavailable, which is common in acute care settings.

### 2.7 Sepsis Severity and Complications

| Concept | SCTID | Clinical Significance |
|---------|-------|----------------------|
| **Sepsis-induced hypotension** | 371039008 | Hemodynamic compromise |
| **Septic encephalopathy** | 37168009 | CNS involvement |
| **Sepsis-induced acute kidney injury** | 722096000 | Renal organ dysfunction |
| **Sepsis-induced acute respiratory distress syndrome** | 67782005 | Pulmonary organ dysfunction |
| **Sepsis-induced coagulopathy** | 438949009 | Hematologic dysfunction |

---

## 3. Temporal Concepts in SNOMED CT

### 3.1 The Clinical Course Attribute

SNOMED CT models temporal characteristics through the **246453008 |Clinical course|** attribute, which combines both onset and duration into a single characteristic. This design decision was made because:

1. Many conditions with acute (sudden) onsets also have acute (short-term) courses
2. Few conditions with chronic (long-term) durations require separation of rapidity of onset from duration
3. Combined clinical course has proven more reproducible and useful than separate onset and course attributes

### 3.2 Clinical Course Values

| Qualifier Value | SCTID | Definition |
|----------------|-------|------------|
| **Acute** | 424124008 | Sudden onset and/or short duration |
| **Chronic** | 90734009 | Gradual onset and/or long duration |
| **Acute on chronic** | 373933003 | Acute exacerbation of chronic condition |
| **Subacute** | 19939008 | Intermediate between acute and chronic |
| **Fulminant** | 48796009 | Extremely rapid progression with severe symptoms |
| **Recurrent** | 255227004 | Repeated episodes with intervening periods |
| **Persistent** | 255238004 | Continuing over extended period |
| **Episodic** | 255303004 | Occurring in discrete episodes |

### 3.3 The Ambiguity of "Acute"

The term "acute" in clinical terminology has multiple overlapping meanings:
- **Rapid onset** - The condition develops suddenly
- **Short duration** - The condition is expected to resolve quickly
- **High severity** - The condition is severe or critical

SNOMED CT addresses this by using context-specific concepts. For example:
- **Acute sepsis** emphasizes sudden onset and severe presentation
- **Chronic sepsis** would indicate prolonged infectious state (rare but possible)

### 3.4 Temporal Relationship Attributes

SNOMED CT provides several temporal relationship attributes:

| Attribute | SCTID | Description | Self-Grouped |
|-----------|-------|-------------|--------------|
| **After** | 255234002 | Clinical finding occurs after another event | Yes |
| **Before** | 288556008 | Clinical finding occurs before another event | Yes |
| **During** | 371881003 | Clinical finding occurs during another event | Yes |
| **Due to** | 42752001 | Causal temporal relationship | Yes |
| **Associated with** | 47429007 | General association without specific timing | Yes |
| **Temporally related to** | 726633004 | Unspecified temporal relationship | Yes |

**Self-grouped attributes** must be the only attribute in their relationship group; they cannot be combined with other attributes like finding site or causative agent.

### 3.5 Temporal Modeling Examples

**Example 1: Post-infectious complications**
```
240178002 |Acute glomerulonephritis following streptococcal infection| :
  255234002 |After| = 43878008 |Streptococcal infectious disease|
  363698007 |Finding site| = 68288005 |Glomerular structure|
  116676008 |Associated morphology| = 409774005 |Inflammatory morphology|
```

**Example 2: During-relationship**
```
698296003 |Nausea during pregnancy| :
  371881003 |During| = 289908002 |Pregnancy|
  363698007 |Finding site| = 69695003 |Stomach structure|
```

**Example 3: Acute on chronic condition**
```
195951007 |Acute on chronic renal failure| :
  246453008 |Clinical course| = 373933003 |Acute on chronic|
  363698007 |Finding site| = 64033007 |Kidney structure|
```

### 3.6 Limitations in Temporal Modeling

SNOMED CT's temporal capabilities have important limitations:

1. **No precise timestamps** - Cannot represent exact dates/times of onset or duration
2. **No explicit duration values** - Cannot specify "3 days" or "2 weeks"
3. **No ordering of multiple events** - Cannot represent complex temporal sequences
4. **Limited granularity** - Temporal relationships are qualitative, not quantitative

For precise temporal reasoning in acute care knowledge graphs, SNOMED CT concepts must be combined with:
- **Time-stamped observations** in the EHR data layer
- **Temporal ontologies** like OWL-Time or Allen's Interval Algebra
- **Event-based representations** with explicit start and end times

### 3.7 Temporal Patterns in Sepsis

For sepsis-related temporal modeling:

```
91302008 |Sepsis| :
  246453008 |Clinical course| = 424124008 |Acute|
  255234002 |After| = 40733004 |Infectious disease|
```

```
76571007 |Severe sepsis| :
  246453008 |Clinical course| = 48796009 |Fulminant|
  42752001 |Due to| = 91302008 |Sepsis|
```

---

## 4. SNOMED Expression Constraint Language (ECL)

### 4.1 Overview and Purpose

The SNOMED CT Expression Constraint Language (ECL) is a formal syntax for representing computable rules that define bounded sets of clinical meanings. ECL enables:

- **Value set definition** - Restricting valid values for EHR data elements
- **Intensional reference set definition** - Defining concept sets by rules rather than enumeration
- **Machine-processable queries** - Identifying matching expressions in clinical data
- **Concept model constraints** - Restricting attribute ranges in SNOMED CT definitions

ECL is part of the SNOMED CT Family of Languages and is essential for building clinical knowledge graphs that can reason over SNOMED CT concepts.

### 4.2 Core ECL Operators

#### 4.2.1 Descendant Operators

| Operator | Symbol | Description | Example |
|----------|--------|-------------|---------|
| **Descendant of** | `<` | All descendants excluding self | `< 91302008 \|Sepsis\|` |
| **Descendant or self** | `<<` | All descendants including self | `<< 91302008 \|Sepsis\|` |
| **Child of** | `<!` | Immediate children only | `<! 404684003 \|Clinical finding\|` |

#### 4.2.2 Ancestor Operators

| Operator | Symbol | Description | Example |
|----------|--------|-------------|---------|
| **Ancestor of** | `>` | All ancestors excluding self | `> 10001005 \|Bacterial sepsis\|` |
| **Ancestor or self** | `>>` | All ancestors including self | `>> 10001005 \|Bacterial sepsis\|` |
| **Parent of** | `>!` | Immediate parents only | `>! 91302008 \|Sepsis\|` |

#### 4.2.3 Refinement Operator

| Operator | Symbol | Description |
|----------|--------|-------------|
| **Refinement** | `:` | Constrains concepts by attribute values |

#### 4.2.4 Logical Operators

| Operator | Symbol | Description |
|----------|--------|-------------|
| **Conjunction (AND)** | `,` or `AND` | Both conditions must be true |
| **Disjunction (OR)** | `OR` | Either condition can be true |
| **Exclusion (MINUS)** | `MINUS` | Exclude matching concepts |

#### 4.2.5 Wildcard

| Operator | Symbol | Description |
|----------|--------|-------------|
| **Any concept** | `*` | Matches any concept |

### 4.3 Simple ECL Examples

**Example 1: All types of sepsis**
```ecl
<< 91302008 |Sepsis|
```
Returns: Sepsis and all its subtypes (bacterial sepsis, neonatal sepsis, etc.)

**Example 2: All direct subtypes of sepsis**
```ecl
<! 91302008 |Sepsis|
```
Returns: Only immediate children like bacterial sepsis, not deeper descendants

**Example 3: All clinical findings**
```ecl
<< 404684003 |Clinical finding|
```
Returns: All 100,000+ clinical finding concepts

**Example 4: All procedures on the kidney**
```ecl
<< 71388002 |Procedure| :
  405813007 |Procedure site - Direct| = << 64033007 |Kidney structure|
```
Returns: All procedures with kidney as the direct procedure site

### 4.4 Refinement Examples

**Example 5: Inflammatory conditions of the lung**
```ecl
<< 404684003 |Clinical finding| :
  363698007 |Finding site| = << 39057004 |Lung structure|,
  116676008 |Associated morphology| = << 409774005 |Inflammatory morphology|
```
Returns: Pneumonia, pleurisy, and other inflammatory lung conditions

**Example 6: Bacterial infections**
```ecl
<< 87628006 |Bacterial infectious disease| :
  246075003 |Causative agent| = << 409822003 |Bacteria|
```
Returns: All bacterial infections with bacteria as causative agent

**Example 7: Acute conditions**
```ecl
<< 404684003 |Clinical finding| :
  246453008 |Clinical course| = 424124008 |Acute|
```
Returns: All clinical findings with acute clinical course

### 4.5 Complex Sepsis-Related ECL Queries

**Example 8: All sepsis with specific causative organisms**
```ecl
<< 91302008 |Sepsis| :
  246075003 |Causative agent| = (
    << 3092008 |Staphylococcus|
    OR << 58800005 |Streptococcus|
    OR << 112283007 |Escherichia coli|
  )
```
Returns: Sepsis caused by Staph, Strep, or E. coli

**Example 9: Sepsis complications affecting specific organs**
```ecl
<< 404684003 |Clinical finding| :
  42752001 |Due to| = << 91302008 |Sepsis|,
  363698007 |Finding site| = (
    << 64033007 |Kidney structure|
    OR << 39057004 |Lung structure|
    OR << 12738006 |Brain structure|
  )
```
Returns: Sepsis-induced organ dysfunctions of kidney, lung, or brain

**Example 10: Infections with acute course excluding sepsis**
```ecl
<< 40733004 |Infectious disease| :
  246453008 |Clinical course| = 424124008 |Acute|
MINUS << 91302008 |Sepsis|
```
Returns: Acute infections excluding sepsis

### 4.6 Nested Refinements

**Example 11: Sepsis with organ dysfunction**
```ecl
<< 91302008 |Sepsis| :
  42752001 |Due to| = (
    << 40733004 |Infectious disease| :
      246075003 |Causative agent| = << 409822003 |Bacteria|
  )
```
Returns: Sepsis due to bacterial infections with nested causative agent constraint

### 4.7 Attribute Groups

**Example 12: Multiple morphologies at different sites**
```ecl
<< 404684003 |Clinical finding| :
  {
    363698007 |Finding site| = << 39057004 |Lung structure|,
    116676008 |Associated morphology| = << 56208002 |Ulcer|
  },
  {
    363698007 |Finding site| = << 122496007 |Blood|,
    116676008 |Associated morphology| = << 409774005 |Inflammatory morphology|
  }
```
Returns: Conditions with ulceration in lungs AND inflammation in blood

### 4.8 ECL for Value Set Definition in FHIR

ECL can be used directly in FHIR ValueSet resources:

```json
{
  "resourceType": "ValueSet",
  "url": "http://example.org/fhir/ValueSet/sepsis-conditions",
  "name": "SepsisConditions",
  "status": "active",
  "compose": {
    "include": [
      {
        "system": "http://snomed.info/sct",
        "filter": [
          {
            "property": "constraint",
            "op": "=",
            "value": "<< 91302008 |Sepsis|"
          }
        ]
      }
    ]
  }
}
```

### 4.9 ECL Performance Considerations

1. **Specificity** - More specific constraints execute faster
2. **Attribute indexing** - Some attributes may not be optimally indexed
3. **Wildcard usage** - Minimize use of `*` operator
4. **Descendant depth** - Shallow hierarchies query faster than deep ones

### 4.10 ECL Tools and Resources

- **Official Specification**: https://snomed.org/ecl
- **GitHub Repository**: https://github.com/IHTSDO/snomed-expression-constraint-language
- **ECL Quick Reference**: https://docs.snomed.org/snomed-ct-specifications/snomed-ct-expression-constraint-language/appendices/appendix-d-ecl-quick-reference
- **Australian ECL Examples**: https://audigitalhealth.github.io/ecl-examples/
- **Ontoserver ECL Reference**: https://ontoserver.csiro.au/shrimp/ecl_help.html

---

## 5. SNOMED CT to ICD-10 Mapping Challenges

### 5.1 Fundamental Structural Differences

#### 5.1.1 Hierarchical Structure

**SNOMED CT:**
- Multi-hierarchical (polyhierarchical) structure
- A single concept can have multiple parents
- Example: "Lung cancer" appears in both neoplasm and respiratory hierarchies
- Enables multiple classification perspectives simultaneously

**ICD-10:**
- Mono-hierarchical structure
- Each code has a single parent
- Example: "Lung cancer" appears only in the neoplasm chapter (C34.-)
- Forces a single classification perspective

**Impact:** Automated mapping must choose one ICD-10 classification path when SNOMED CT concept participates in multiple hierarchies, potentially losing important clinical context.

#### 5.1.2 Granularity and Specificity

**SNOMED CT:**
- Over 357,000 concepts with very high granularity
- Supports precoordinated and postcoordinated expressions
- Example: "Acute bacterial pneumonia of right lower lobe due to Streptococcus pneumoniae"

**ICD-10:**
- Approximately 14,000 codes (ICD-10-CM ~69,000)
- Less granular, primarily precoordinated
- Example: "J13 Pneumonia due to Streptococcus pneumoniae"

**Impact:** Many SNOMED CT concepts map to the same ICD-10 code (many-to-one mapping), resulting in loss of clinical detail. Conversely, some ICD-10 codes require contextual information to map accurately to SNOMED CT.

### 5.2 Mapping Cardinality Patterns

#### 5.2.1 One-to-Many Mappings

A single SNOMED CT concept may map to multiple ICD-10 codes depending on context.

**Example: 91302008 |Sepsis|**

Maps to multiple ICD-10-CM codes:
- A40.0 - Sepsis due to streptococcus, group A
- A40.1 - Sepsis due to streptococcus, group B
- A40.3 - Sepsis due to Streptococcus pneumoniae
- A40.8 - Other streptococcal sepsis
- A40.9 - Streptococcal sepsis, unspecified
- A41.0 - Sepsis due to Staphylococcus aureus
- A41.1 - Sepsis due to other specified staphylococcus
- A41.2 - Sepsis due to unspecified staphylococcus
- A41.3 - Sepsis due to Hemophilus influenzae
- A41.4 - Sepsis due to anaerobes
- A41.50 - Gram-negative sepsis, unspecified
- A41.51 - Sepsis due to Escherichia coli
- A41.52 - Sepsis due to Pseudomonas
- A41.53 - Sepsis due to Serratia
- A41.59 - Other Gram-negative sepsis
- A41.81 - Sepsis due to Enterococcus
- A41.89 - Other specified sepsis
- A41.9 - Sepsis, unspecified organism
- O85 - Puerperal sepsis
- P36.0 - Sepsis of newborn due to streptococcus, group B
- P36.1 - Sepsis of newborn due to other and unspecified streptococci
- P36.2 - Sepsis of newborn due to Staphylococcus aureus
- P36.3 - Sepsis of newborn due to other and unspecified staphylococci
- P36.4 - Sepsis of newborn due to Escherichia coli
- P36.5 - Sepsis of newborn due to anaerobes
- P36.8 - Other bacterial sepsis of newborn
- P36.9 - Bacterial sepsis of newborn, unspecified

**Challenge:** Requires additional clinical context (organism, patient age, pregnancy status) to select the appropriate ICD-10 code.

#### 5.2.2 Many-to-One Mappings

Multiple SNOMED CT concepts may map to a single ICD-10 code.

**Example: J18.9 (Pneumonia, unspecified organism)**

Maps from multiple SNOMED CT concepts:
- 233604007 |Pneumonia|
- 385093006 |Community acquired pneumonia|
- 53084003 |Bacterial pneumonia|
- 233607000 |Lobar pneumonia|
- Many others with unspecified organism

**Challenge:** Reverse mapping from ICD-10 to SNOMED CT is ambiguous and may require additional documentation review.

#### 5.2.3 No Direct Mapping

Some SNOMED CT concepts have no appropriate ICD-10 equivalent, and vice versa.

**SNOMED CT concepts without ICD-10 equivalents:**
- Normal findings (e.g., "Normal breath sounds")
- Certain procedure concepts
- Very specific postcoordinated expressions

**ICD-10 codes without SNOMED CT equivalents:**
- External cause codes (V, W, X, Y codes)
- Some administrative codes (Z codes)
- Certain morphology codes

### 5.3 Rule-Based Mapping

The NLM SNOMED CT to ICD-10-CM map uses "rule-based mapping" to handle complexity:

#### 5.3.1 Map Rules and Map Groups

**Map Rule:** A single possible target ICD-10 code with associated conditions
**Map Group:** A collection of related map rules for a single SNOMED CT concept

**Example Map Group for 91302008 |Sepsis|:**
```
Map Group 1:
  Rule 1: IF Causative agent = Streptococcus group A THEN A40.0
  Rule 2: IF Causative agent = Streptococcus group B THEN A40.1
  Rule 3: IF Causative agent = Streptococcus pneumoniae THEN A40.3
  ...
Map Group 2:
  Rule 1: IF Patient age < 28 days THEN P36.x (select specific code based on organism)
Map Group 3:
  Rule 1: IF Patient is pregnant/postpartum THEN O85
Default:
  Rule 1: IF no specific conditions met THEN A41.9
```

#### 5.3.2 Map Rule Attributes

Each map rule includes:
- **mapTarget** - The ICD-10 code
- **mapRule** - Conditions that must be satisfied
- **mapAdvice** - Guidance for code selection
- **mapPriority** - Order of rule evaluation
- **correlationId** - Identifies exact vs. approximate matches

### 5.4 Specific Mapping Challenges for Sepsis

#### 5.4.1 Organism Identification

**Challenge:** ICD-10 requires organism-specific codes, but SNOMED CT concept 91302008 |Sepsis| doesn't always include organism information.

**Solution Approaches:**
1. Query for causative agent relationship in SNOMED CT
2. Parse microbiology results from laboratory data
3. Use NLP to extract organism from clinical notes
4. Default to "unspecified organism" (A41.9) when organism unknown

#### 5.4.2 Age-Dependent Coding

**Challenge:** Neonatal sepsis (age <28 days) uses P36.x codes, while adult sepsis uses A40.x/A41.x codes.

**Solution:** Map rules must include patient age as contextual information.

#### 5.4.3 Pregnancy-Related Sepsis

**Challenge:** Puerperal sepsis (O85) applies only during pregnancy or within 6 weeks postpartum.

**Solution:** Map rules must check pregnancy status and timing relative to delivery.

#### 5.4.4 Severe Sepsis and Septic Shock

**Challenge:** ICD-10 requires combination coding for severe sepsis:
- Primary code: Underlying infection (A40.x/A41.x)
- Secondary code: R65.20 (Severe sepsis without septic shock) or R65.21 (Severe sepsis with septic shock)
- Additional codes: Specific organ dysfunctions

**SNOMED CT representation:**
```
76571007 |Severe sepsis| :
  42752001 |Due to| = 91302008 |Sepsis|,
  42752001 |Due to| = << 57653000 |Multiple organ dysfunction syndrome|
```

**Mapping complexity:** Requires decomposing SNOMED CT concept into multiple ICD-10 codes and ensuring proper sequencing.

### 5.5 Interoperability Efforts and Lessons Learned

#### 5.5.1 WHO-SNOMED International Pilot Project

A pilot project between SNOMED International and WHO to map SNOMED CT to ICD-11 Foundation revealed:

**Key Findings:**
1. **Tremendous effort required** - Manual mapping is labor-intensive and time-consuming
2. **Common challenges** - Structural differences, granularity mismatches, and semantic gaps
3. **Need for clear goals** - Mapping goals and use cases must be explicitly defined
4. **Resource requirements** - Adequate funding and expert personnel are essential
5. **Road map necessity** - Long-term planning and iterative improvement needed

**Recommendations:**
- Clarify goals and use cases for mapping
- Provide adequate resources for mapping teams
- Establish a clear road map for mapping development
- Reconsider incorporating SNOMED CT directly into ICD-11 Foundation
- Focus on interoperability rather than convergence

#### 5.5.2 Practical Implementation Failures

Research on NLM Map usage identified common failure patterns:

**Primary Failure Modes:**
1. **Omissions in problem list** - Conditions present but not documented in structured problem list (60% of failures)
2. **Suboptimal SNOMED CT mapping** - Incorrect or imprecise SNOMED CT concept selected for clinical term (30% of failures)
3. **True mapping gaps** - No appropriate ICD-10 code exists for SNOMED CT concept (10% of failures)

**Implications:**
- Improving problem list completeness is more important than refining maps
- Better clinical decision support for SNOMED CT concept selection is needed
- Some residual manual coding will always be necessary

### 5.6 Use Cases and Purposes

#### 5.6.1 Primary Use Case: Statistical and Reimbursement Coding

**Purpose:** Semi-automated generation of ICD-10 codes from SNOMED CT-encoded clinical data for:
- Healthcare reimbursement (billing)
- Public health reporting
- Disease surveillance
- Mortality and morbidity statistics
- Clinical registries

**Workflow:**
1. Clinician documents using SNOMED CT terms (or natural language mapped to SNOMED CT)
2. Mapping software suggests ICD-10 codes based on map rules
3. Coder reviews suggestions and selects appropriate codes based on additional context
4. Final ICD-10 codes are submitted for billing/reporting

#### 5.6.2 Benefits of SNOMED CT-to-ICD-10 Mapping

1. **Reduced duplicate coding effort** - Leverage clinical documentation for statistical coding
2. **Improved coding accuracy** - Algorithmic suggestions reduce human error
3. **Faster coding turnaround** - Semi-automated process is quicker than fully manual coding
4. **Better data quality** - Structured clinical data improves downstream code quality
5. **Enhanced semantic interoperability** - Enables cross-terminology queries and analysis

### 5.7 Recommendations for Knowledge Graph Implementation

#### 5.7.1 Dual Terminology Strategy

Maintain both SNOMED CT and ICD-10 in the knowledge graph:

```turtle
@prefix sct: <http://snomed.info/sct/> .
@prefix icd10: <http://hl7.org/fhir/sid/icd-10-cm/> .
@prefix map: <http://example.org/mapping/> .

# SNOMED CT concept
sct:91302008 a sct:ClinicalFinding ;
  rdfs:label "Sepsis"@en ;
  sct:isA sct:87628006 ; # Bacterial infectious disease
  sct:causativeAgent sct:264395009 . # Microorganism

# ICD-10 code
icd10:A41.9 a icd10:Code ;
  rdfs:label "Sepsis, unspecified organism"@en .

# Mapping relationship
map:map1 a map:Mapping ;
  map:source sct:91302008 ;
  map:target icd10:A41.9 ;
  map:correlation "SNOMED CT source concept is narrower than ICD-10 target code" ;
  map:mapRule "Use only when organism is not specified" .
```

#### 5.7.2 Contextual Information Capture

Ensure the knowledge graph captures contextual information needed for accurate mapping:

```turtle
:patient123_sepsis a sct:91302008 ; # Sepsis
  :causativeOrganism sct:112283007 ; # E. coli
  :patientAge "45"^^xsd:integer ;
  :pregnancyStatus "not pregnant" ;
  :observationDate "2025-01-15"^^xsd:date ;
  :appropriateICD10Code icd10:A41.51 . # Sepsis due to E. coli
```

#### 5.7.3 Mapping Quality Metadata

Track mapping confidence and source:

```turtle
map:sepsis_ecoli_mapping a map:Mapping ;
  map:source sct:91302008 ;
  map:target icd10:A41.51 ;
  map:mappingSource "NLM SNOMED CT to ICD-10-CM Map v2024" ;
  map:confidence "0.95"^^xsd:float ;
  map:requiresContext "causative organism = E. coli" ;
  map:lastReviewed "2024-09-30"^^xsd:date .
```

#### 5.7.4 Multi-Code Scenarios

Handle ICD-10 combination coding requirements:

```turtle
:patient456_severe_sepsis a sct:76571007 ; # Severe sepsis
  :requiresMultipleICD10Codes (
    icd10:A41.9  # Sepsis, unspecified organism (principal)
    icd10:R65.20 # Severe sepsis without septic shock
    icd10:N17.9  # Acute kidney failure, unspecified
  ) ;
  :codeSequencing "A41.9 must be principal diagnosis" .
```

---

## 6. Implementation Recommendations for Acute Care Knowledge Graphs

### 6.1 Sepsis Detection and Surveillance

**Use ECL to identify all sepsis cases:**
```ecl
<< 91302008 |Sepsis| OR
<< 76571007 |Severe sepsis| OR
(
  << 404684003 |Clinical finding| :
    42752001 |Due to| = << 91302008 |Sepsis|
)
```

**Capture temporal progression:**
```turtle
:sepsis_timeline a :ClinicalTimeline ;
  :initialInfection [
    :concept sct:233604007 ; # Pneumonia
    :onset "2025-01-10T08:00:00"^^xsd:dateTime
  ] ;
  :sepsisOnset [
    :concept sct:91302008 ; # Sepsis
    :onset "2025-01-11T14:30:00"^^xsd:dateTime ;
    :relationship [ :type sct:255234002 ; :target :initialInfection ] # After
  ] ;
  :organDysfunction [
    :concept sct:722096000 ; # Sepsis-induced AKI
    :onset "2025-01-12T02:15:00"^^xsd:dateTime ;
    :relationship [ :type sct:42752001 ; :target :sepsisOnset ] # Due to
  ] .
```

### 6.2 Integration with FHIR

**SNOMED CT in FHIR Condition resource:**
```json
{
  "resourceType": "Condition",
  "code": {
    "coding": [
      {
        "system": "http://snomed.info/sct",
        "code": "91302008",
        "display": "Sepsis"
      },
      {
        "system": "http://hl7.org/fhir/sid/icd-10-cm",
        "code": "A41.9",
        "display": "Sepsis, unspecified organism"
      }
    ]
  },
  "clinicalStatus": {
    "coding": [
      {
        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
        "code": "active"
      }
    ]
  },
  "verificationStatus": {
    "coding": [
      {
        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
        "code": "confirmed"
      }
    ]
  },
  "category": [
    {
      "coding": [
        {
          "system": "http://terminology.hl7.org/CodeSystem/condition-category",
          "code": "encounter-diagnosis"
        }
      ]
    }
  ],
  "onsetDateTime": "2025-01-11T14:30:00Z"
}
```

### 6.3 Reasoning Capabilities

Leverage SNOMED CT's formal semantics for:

1. **Subsumption reasoning** - Automatically infer that bacterial sepsis is a type of sepsis
2. **Classification** - Classify new postcoordinated expressions into the hierarchy
3. **Query expansion** - Expand queries to include all subtypes automatically
4. **Consistency checking** - Detect contradictory or redundant concept assignments

### 6.4 Quality Metrics

Track mapping quality in the knowledge graph:

```sparql
# Find all sepsis cases with missing ICD-10 codes
SELECT ?case ?snomedConcept
WHERE {
  ?case a :SepsisCase ;
        :snomedCTCode ?snomedConcept .
  ?snomedConcept sct:isA* sct:91302008 . # Descendant of Sepsis
  FILTER NOT EXISTS { ?case :icd10Code ?icd10 }
}
```

---

## 7. Conclusion

SNOMED CT provides a robust foundation for clinical knowledge graphs in acute care settings, particularly for sepsis surveillance and management. Its hierarchical structure, rich relationship model, and Expression Constraint Language enable sophisticated clinical reasoning and data retrieval. However, successful implementation requires:

1. **Understanding structural differences** between SNOMED CT and ICD-10
2. **Implementing rule-based mapping** with contextual information
3. **Capturing temporal relationships** through both SNOMED CT attributes and external temporal ontologies
4. **Maintaining dual terminology** representation in the knowledge graph
5. **Using ECL effectively** for value set definition and query formulation
6. **Ensuring data quality** through comprehensive documentation and mapping validation

By combining SNOMED CT's clinical expressiveness with careful attention to mapping challenges and temporal modeling limitations, acute care knowledge graphs can support advanced clinical decision support, quality measurement, and research applications.

---

## References and Resources

### SNOMED CT Hierarchy and Structure
- [SNOMED CT Concept Model](https://docs.snomed.org/snomed-ct-practical-guides/snomed-ct-starter-guide/6-snomed-ct-concept-model)
- [Top Level Hierarchies](https://ctcentric.com/docs/introduction/snomed-ct-concept-model/top-level-hierarchies/)
- [Root and Top-level Concepts](https://confluence.ihtsdotools.org/display/DOCEG/Root+and+Top-level+Concepts)

### Sepsis Concepts
- [SNOMED CT Browser - Sepsis](https://snomedbrowser.com/Codes/Details/91302008)
- [NCBO BioPortal - Sepsis](https://purl.bioontology.org/ontology/SNOMEDCT/91302008)
- [NCBO BioPortal - Clinical Sepsis](https://bioportal.bioontology.org/ontologies/SNOMEDCT?p=classes&conceptid=447931005)

### Temporal Concepts
- [Clinical Finding Defining Attributes](https://confluence.ihtsdotools.org/display/DOCEG/Clinical+Finding+Defining+Attributes)
- [Clinical Finding and Disorder](https://confluence.ihtsdotools.org/display/DOCEG/Clinical+Finding+and+Disorder)

### Expression Constraint Language
- [ECL Specification](https://docs.snomed.org/snomed-ct-specifications/snomed-ct-expression-constraint-language/introduction/1-introduction)
- [ECL Quick Reference](https://docs.snomed.org/snomed-ct-specifications/snomed-ct-expression-constraint-language/appendices/appendix-d-ecl-quick-reference)
- [ECL GitHub Repository](https://github.com/IHTSDO/snomed-expression-constraint-language)
- [Australian Digital Health ECL Examples](https://audigitalhealth.github.io/ecl-examples/)
- [Ontoserver ECL Reference](https://ontoserver.csiro.au/shrimp/ecl_help.html)

### SNOMED CT to ICD-10 Mapping
- [NLM SNOMED CT to ICD-10-CM Map](https://www.nlm.nih.gov/research/umls/mapping_projects/snomedct_to_icd10cm.html)
- [Promoting interoperability between SNOMED CT and ICD-11](https://pmc.ncbi.nlm.nih.gov/articles/PMC11258399/)
- [SNOMED CT Maps](https://www.snomed.org/maps)

### Relationship Types and Attributes
- [Relationship Group](https://confluence.ihtsdotools.org/display/DOCEG/Relationship+Group)
- [Attributes Used to Define SNOMED CT Concepts](https://ctcentric.com/docs/introduction/snomed-ct-concept-model/attributes-used-to-define-concepts/)
- [Using Defining Relationships](https://confluence.ihtsdotools.org/display/DOCANLYT/6.3+Using+Defining+Relationships)
