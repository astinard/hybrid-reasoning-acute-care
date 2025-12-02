# OHDSI OMOP Common Data Model (CDM) Research

## Executive Summary

The Observational Medical Outcomes Partnership (OMOP) Common Data Model (CDM) is an open community data standard designed to standardize the structure and content of observational data and enable efficient analyses that produce reliable evidence. Maintained by the Observational Health Data Sciences and Informatics (OHDSI) collaborative, the OMOP CDM enables large-scale network research across hundreds of healthcare databases worldwide.

**Current Version:** OMOP CDM v5.4 (Latest as of 2025)

## 1. Overview of OMOP CDM

### 1.1 What is OMOP CDM?

The OMOP CDM is a person-centric, relational database schema that standardizes healthcare data to enable observational research. It is simply a blueprint (schema) for how data should be structured and does not require any particular software implementation - any relational database will suffice (MySQL, Postgres, SQL Server, Oracle, etc.).

### 1.2 Key Design Principles

- **Person-Centric Model**: All clinical event tables are linked to the PERSON table, enabling longitudinal views of all healthcare-relevant events by person
- **Standardized Vocabularies**: Central component that allows organization and standardization of medical terms across clinical domains
- **Open Source**: Free to use and maintained by an international collaborative community
- **Database Agnostic**: Can be implemented on any relational database platform

### 1.3 OHDSI Organization

Founded in 2014, OHDSI (Observational Health Data Sciences and Informatics, pronounced "Odyssey") grew out of the successful OMOP initiative. Key statistics:

- **4,200+ collaborators** across 83 countries
- **810 million+ unique patient records** from around the world
- **100+ healthcare databases** from 20+ countries in the data network
- **1 billion+ patient records** collectively captured

## 2. OMOP CDM Schema Structure

### 2.1 Table Organization

The OMOP CDM v5.4 contains **39 tables** organized into **7 main categories**:

1. **Clinical Event Tables** (16 tables)
   - Person-centric tables containing clinical observations and events
   - Must be linked to the PERSON table

2. **Vocabulary Tables** (10 tables)
   - Contain standardized vocabularies and concept mappings
   - Enable semantic interoperability

3. **Metadata Tables** (2 tables)
   - Store metadata about the CDM instance

4. **Health System Data Tables** (4 tables)
   - Capture information about healthcare facilities and providers

5. **Health Economics Tables** (2 tables)
   - Store cost and payer information

6. **Standardized Derived Elements** (3 tables)
   - Contain derived data like eras and cohorts

7. **Results Schema Tables** (2 tables)
   - Only tables writable by end-users
   - Store analysis results

### 2.2 Core Clinical Event Tables

The following are the primary clinical event tables:

#### PERSON Table
Stores demographic details of patients:
- `person_id` (Primary Key)
- `gender_concept_id`
- `year_of_birth` (required)
- `month_of_birth`
- `day_of_birth`
- `birth_datetime`
- `race_concept_id`
- `ethnicity_concept_id`

#### VISIT_OCCURRENCE Table
Records patient interactions with healthcare system:
- `visit_occurrence_id` (Primary Key)
- `person_id` (Foreign Key to PERSON)
- `visit_concept_id` (Type of visit: inpatient, outpatient, ER, etc.)
- `visit_start_date`
- `visit_start_datetime`
- `visit_end_date`
- `visit_end_datetime`
- `visit_type_concept_id`
- `provider_id`
- `care_site_id`

#### CONDITION_OCCURRENCE Table
Captures patient conditions and diagnoses:
- `condition_occurrence_id` (Primary Key)
- `person_id` (Foreign Key to PERSON)
- `condition_concept_id` (Standard SNOMED concept)
- `condition_start_date`
- `condition_start_datetime`
- `condition_end_date`
- `condition_type_concept_id`
- `condition_status_concept_id`
- `condition_source_value` (Original source code)
- `condition_source_concept_id` (e.g., ICD-10 code)
- `visit_occurrence_id` (Foreign Key to VISIT_OCCURRENCE)

**Example:** Type 2 diabetes mellitus would be stored with:
- `condition_concept_id`: SNOMED concept for type 2 diabetes (standard)
- `condition_source_concept_id`: Original ICD-10 code E11 or SNOMED 44054006 from source

#### DRUG_EXPOSURE Table
Records drug utilization and medication administration:
- `drug_exposure_id` (Primary Key)
- `person_id` (Foreign Key to PERSON)
- `drug_concept_id` (Standard RxNorm concept)
- `drug_exposure_start_date`
- `drug_exposure_start_datetime`
- `drug_exposure_end_date`
- `drug_type_concept_id`
- `quantity`
- `days_supply`
- `sig` (Prescription directions)
- `route_concept_id`
- `dose_unit_concept_id`
- `drug_source_value`
- `drug_source_concept_id`
- `visit_occurrence_id` (Foreign Key to VISIT_OCCURRENCE)

**Standard Vocabulary:** RxNorm is the standard for drug exposures. Best practice is to store the lowest level RxNorm available and use vocabulary relationships for querying.

#### PROCEDURE_OCCURRENCE Table
Captures medical procedures:
- `procedure_occurrence_id` (Primary Key)
- `person_id` (Foreign Key to PERSON)
- `procedure_concept_id`
- `procedure_date`
- `procedure_datetime`
- `procedure_type_concept_id`
- `procedure_source_value`
- `procedure_source_concept_id`
- `visit_occurrence_id`

#### MEASUREMENT Table
Stores laboratory tests, vital signs, and other measurements:
- `measurement_id` (Primary Key)
- `person_id` (Foreign Key to PERSON)
- `measurement_concept_id` (Standard LOINC concept)
- `measurement_date`
- `measurement_datetime`
- `measurement_type_concept_id`
- `value_as_number`
- `value_as_concept_id`
- `unit_concept_id`
- `range_low`
- `range_high`
- `visit_occurrence_id`

#### OBSERVATION Table
Records clinical facts that don't fit other domains:
- `observation_id` (Primary Key)
- `person_id` (Foreign Key to PERSON)
- `observation_concept_id`
- `observation_date`
- `observation_datetime`
- `observation_type_concept_id`
- `value_as_number`
- `value_as_string`
- `value_as_concept_id`
- `visit_occurrence_id`

### 2.3 Additional Important Tables

#### SPECIMEN Table
Identifies biological samples from a person:
- `specimen_id`
- `person_id`
- `specimen_concept_id`
- `specimen_type_concept_id`
- `specimen_date`
- `specimen_datetime`
- `quantity`
- `unit_concept_id`

#### NOTE_NLP Table
Encodes output of Natural Language Processing on clinical notes:
- `note_nlp_id`
- `note_id`
- `section_concept_id`
- `snippet` (Extracted text)
- `offset` (Character position in note)
- `lexical_variant` (Actual text extracted)
- `note_nlp_concept_id` (Mapped standard concept)
- `nlp_system`
- `nlp_date`

#### FACT_RELATIONSHIP Table
Records relationships between facts in different tables:
- `domain_concept_id_1` (Source domain)
- `fact_id_1` (Source fact)
- `domain_concept_id_2` (Related domain)
- `fact_id_2` (Related fact)
- `relationship_concept_id` (Type of relationship)

**Use Cases:**
- Person relationships (parent-child)
- Care site hierarchies
- Indication relationships (drug-condition)

#### CONDITION_ERA and DRUG_ERA Tables
Derived tables that represent continuous periods:
- **CONDITION_ERA**: Derived from CONDITION_OCCURRENCE
- **DRUG_ERA**: Derived from DRUG_EXPOSURE
- **Era Definition**: Span of time when a patient is assumed to have a condition or exposure to an active ingredient

### 2.4 Key Vocabulary Tables

#### CONCEPT Table
Primary standardized representation of medical concepts:
- `concept_id` (Primary Key)
- `concept_name`
- `domain_id` (Clinical domain)
- `vocabulary_id` (Source vocabulary)
- `concept_class_id`
- `standard_concept` (Flag: 'S' for standard concepts)
- `concept_code`
- `valid_start_date`
- `valid_end_date`

#### VOCABULARY Table
List of vocabularies in the system:
- Common vocabularies: ICD-9, ICD-10, SNOMED, LOINC, RxNorm, CPT, HCPCS

#### DOMAIN Table
OMOP-defined domains for concepts:
- Examples: "Condition", "Drug", "Procedure", "Visit", "Device", "Specimen", "Measurement", "Observation"
- Each domain defines which CDM table the concept can be used in

#### CONCEPT_RELATIONSHIP Table
Defines relationships between concepts:
- `concept_id_1` (Source concept)
- `concept_id_2` (Related concept)
- `relationship_id` (e.g., "Maps to", "Is a", "Subsumes")
- Used for mapping source codes to standard concepts (relationship_id = 'Maps to')

#### SOURCE_TO_CONCEPT_MAP Table
Maintains local source codes not in standardized vocabularies:
- Recommended for ETL processes
- Maps local codes to standard concepts
- Note: For standard vocabularies, CONCEPT_RELATIONSHIP is preferred

## 3. Standardized Vocabularies

### 3.1 Overview

The OHDSI Standardized Vocabularies are a foundational component that enables standardization of methods, definitions, and results by defining the content of data. This enables true remote network research and analytics.

### 3.2 Key Concepts

**Concept**: A unique identifier for each fundamental unit of meaning used to express clinical information

**Standard Concept**: Concepts designated for normative expression of clinical entities within the OMOP CDM (marked with standard_concept = 'S')

**Domain Assignment**: Each concept is assigned to exactly one domain (stored in domain_id field)

### 3.3 Supported Vocabularies

The OMOP CDM leverages well-known standardized vocabularies:

- **ICD (International Classification of Disease)**: ICD-9, ICD-10
- **SNOMED-CT (Systematized Nomenclature of Medicine)**: Standard for conditions
- **LOINC (Logical Observation Identifiers Names and Codes)**: Standard for measurements
- **RxNorm**: Standard for drugs
- **CPT (Current Procedural Terminology)**: Procedures
- **HCPCS**: Healthcare procedures
- **NDC (National Drug Code)**: Drug products
- **ATC (Anatomical Therapeutic Chemical)**: Drug classification

### 3.4 Vocabulary Mapping Example

**Scenario**: Type 2 diabetes recorded differently at two organizations:
- **Organization A**: Uses SNOMED-CT code 44054006
- **Organization B**: Uses ICD-10 code E11

**OMOP Standardization**:
- Both mapped to the same standard SNOMED concept during ETL
- Standard concept stored in `condition_concept_id`
- Original source concept stored in `condition_source_concept_id`
- Enables cross-organization research and analysis

### 3.5 Vocabulary Benefits

- **Semantic Interoperability**: Enables meaning-based queries across heterogeneous data sources
- **Consolidated Format**: All vocabularies in common format - no need to handle multiple formats
- **Free to Use**: Available to community free of charge
- **Mandatory Reference**: Must be used for OMOP CDM instances
- **Regular Updates**: Maintained by OHDSI community

## 4. ETL (Extract, Transform, Load) Process

### 4.1 ETL Overview

To convert native/raw data to OMOP CDM, organizations must create an ETL process that:
1. Restructures data to match CDM schema
2. Maps source codes to standardized vocabularies
3. Maintains source values alongside standard concepts

### 4.2 ETL Best Practices - Four Major Steps

1. **Design**: Data experts and CDM experts collaborate on ETL design
2. **Code Mapping**: Medical knowledge experts create code mappings to standard vocabularies
3. **Implementation**: Technical experts implement the ETL code
4. **Quality Control**: All stakeholders validate data quality and mapping accuracy

### 4.3 Source to Standard Vocabulary Mapping

#### Mapping Process
- Source values mapped to Standard Concepts during ETL
- Both source and standard concepts stored in clinical event tables
- Source Concept represents the original code in source data
- Each Source Concept mapped to one or more Standard Concepts

#### Key Fields for Mapping
Every clinical event table includes:
- `[domain]_concept_id`: Standard concept (e.g., SNOMED for conditions)
- `[domain]_source_value`: Original text/code from source system
- `[domain]_source_concept_id`: Source vocabulary concept (e.g., ICD-10)

#### Handling Custom/Local Codes

For source codes not in OMOP Vocabulary:
- Create new CONCEPT records with `concept_id` starting at **2,000,000,000**
- This range distinguishes site-specific concepts from OMOP standard concepts
- Create CONCEPT_RELATIONSHIPS to map to standard terminologies
- Use tools like USAGI to facilitate mapping

### 4.4 OHDSI ETL Tools

#### White Rabbit
- Application for analyzing database structure and contents
- Preparation tool for designing ETL
- Scans source databases to understand structure

#### Rabbit-In-A-Hat
- Interactive design tool for ETL to OMOP CDM
- Visual mapping interface
- Generates ETL documentation

#### USAGI
- Application to create mappings between coding systems and standard concepts
- Helps populate SOURCE_TO_CONCEPT_MAP table
- Uses string matching and semantic similarity

#### Perseus
- Alternative tool for vocabulary mapping
- Helps with SOURCE_TO_CONCEPT_MAP table population

### 4.5 ETL Implementation Notes

- **Source Value Preservation**: Always maintain original source codes and values
- **Mapping Documentation**: Document all mapping decisions and assumptions
- **Data Quality**: Implement validation checks throughout ETL process
- **Iterative Process**: ETL is typically large undertaking requiring iteration

## 5. OHDSI Network Research

### 5.1 Mission and Approach

OHDSI's mission is to generate high-quality evidence through observational research using a distributed network approach.

### 5.2 Network Study Process

**OHDSI Network Study**: Research study run across multiple CDM instances at different institutions

**Key Components**:
- OMOP CDM standardization
- Standardized analytical tools
- Study packages with fully specified parameters
- Distributed execution

### 5.3 Distributed Network Model

#### How It Works
1. **Local Data Control**: Patient-level data never leaves individual institutions
2. **Study Protocol Distribution**: Research questions distributed as study protocols with analysis code
3. **Local Execution**: Each site runs analysis on their local CDM instance
4. **Aggregated Results**: Only summary statistics shared among collaborators
5. **Patient Privacy**: Full confidentiality maintained throughout

#### Key Advantages
- **Data Autonomy**: Each partner retains full control over patient-level data
- **Geographic Diversity**: Enables diverse patient cohorts across regions
- **Reproducibility**: Standardized methods ensure reproducible research
- **Scale**: Access to billion+ patient records without data sharing
- **Compliance**: Meets data privacy regulations (HIPAA, GDPR, etc.)

### 5.4 Network Statistics

- **100+ databases** participating in OHDSI network
- **20+ countries** represented
- **1 billion+ patient records** accessible for research
- **810 million+ unique patients** in network

### 5.5 Research Use Cases

OHDSI supports three primary analytical use cases:

#### 1. Clinical Characterization
- Disease natural history studies
- Treatment utilization patterns
- Quality improvement initiatives
- Population health management

#### 2. Population-Level Effect Estimation
- Causal inference methods
- Medical product safety surveillance
- Comparative effectiveness research
- Pharmacovigilance

#### 3. Patient-Level Prediction
- Machine learning algorithms
- Precision medicine applications
- Disease interception
- Risk stratification

### 5.6 OHDSI Analytical Tools

The OHDSI community has developed a robust library of open-source tools:
- Built on top of OMOP CDM
- Support all three use cases
- Freely available
- Regularly updated and maintained

## 6. Implementation Considerations

### 6.1 Database Platform Selection

The OMOP CDM is database-agnostic and supports:
- **PostgreSQL**
- **MySQL**
- **SQL Server**
- **Oracle**
- **BigQuery**
- **Redshift**
- **Snowflake**

DDL scripts available for all platforms in the OHDSI CommonDataModel GitHub repository.

### 6.2 Implementation Steps

1. **Select Database Platform**: Choose appropriate RDBMS
2. **Download DDL Scripts**: Get platform-specific scripts from OHDSI GitHub
3. **Create Database Schema**: Execute DDL to create empty CDM tables
4. **Load Vocabularies**: Download and load OHDSI standardized vocabularies
5. **Design ETL**: Map source data to CDM using ETL tools
6. **Execute ETL**: Transform and load source data
7. **Validate Data**: Run data quality checks
8. **Document Process**: Record all mapping decisions and assumptions

### 6.3 Data Quality Considerations

- **Completeness**: Ensure all required fields populated
- **Conformance**: Validate adherence to CDM specifications
- **Plausibility**: Check for logical inconsistencies
- **Vocabulary Mapping**: Verify accuracy of code mappings
- **Temporal Logic**: Validate date/time relationships

### 6.4 Governance and Compliance

- **Data Privacy**: Implement appropriate access controls
- **Regulatory Compliance**: Meet HIPAA, GDPR, and local regulations
- **IRB Approval**: Obtain necessary research ethics approvals
- **Data Use Agreements**: Establish clear terms for network participation
- **Audit Trails**: Maintain logs of data access and modifications

## 7. Example Implementations

### 7.1 Stanford STARR OMOP

Stanford Medicine Research Data Repository (STARR) has implemented OMOP CDM to enable observational medical outcomes research across Stanford Health Care data.

### 7.2 University of Florida OMOP

UF Health Integrated Data Repository uses OMOP CDM for clinical and translational science research.

### 7.3 UCSF CHIME OMOP

UCSF Center for Healthcare Improvement and Medical Effectiveness uses OMOP for population health research.

### 7.4 All of Us Research Program

NIH's All of Us Research Program uses OMOP CDM to standardize data from 1+ million participants, making it available for research.

## 8. Resources and References

### 8.1 Official Documentation

- **OMOP CDM Homepage**: [https://ohdsi.github.io/CommonDataModel/](https://ohdsi.github.io/CommonDataModel/)
- **OMOP CDM v5.4 Specification**: [https://ohdsi.github.io/CommonDataModel/cdm54.html](https://ohdsi.github.io/CommonDataModel/cdm54.html)
- **OHDSI Data Standardization**: [https://www.ohdsi.org/data-standardization/](https://www.ohdsi.org/data-standardization/)
- **The Book of OHDSI**: [https://ohdsi.github.io/TheBookOfOhdsi/](https://ohdsi.github.io/TheBookOfOhdsi/)

### 8.2 GitHub Resources

- **CommonDataModel Repository**: [https://github.com/OHDSI/CommonDataModel](https://github.com/OHDSI/CommonDataModel)
- DDL scripts for all database platforms
- Release notes and version history
- Community contributions

### 8.3 Key OHDSI Resources

- **Chapter 4 - The Common Data Model**: [https://ohdsi.github.io/TheBookOfOhdsi/CommonDataModel.html](https://ohdsi.github.io/TheBookOfOhdsi/CommonDataModel.html)
- **Chapter 5 - Standardized Vocabularies**: [https://ohdsi.github.io/TheBookOfOhdsi/StandardizedVocabularies.html](https://ohdsi.github.io/TheBookOfOhdsi/StandardizedVocabularies.html)
- **Chapter 6 - Extract Transform Load**: [https://ohdsi.github.io/TheBookOfOhdsi/ExtractTransformLoad.html](https://ohdsi.github.io/TheBookOfOhdsi/ExtractTransformLoad.html)
- **Chapter 20 - OHDSI Network Research**: [https://ohdsi.github.io/TheBookOfOhdsi/NetworkResearch.html](https://ohdsi.github.io/TheBookOfOhdsi/NetworkResearch.html)

### 8.4 Additional Learning Resources

- **OMOP CDM FAQ**: [https://ohdsi.github.io/CommonDataModel/faq.html](https://ohdsi.github.io/CommonDataModel/faq.html)
- **OMOP Vocabulary Wiki**: [https://www.ohdsi.org/web/wiki/doku.php?id=documentation:cdm:vocabulary](https://www.ohdsi.org/web/wiki/doku.php?id=documentation:cdm:vocabulary)
- **All of Us OMOP Basics**: [https://support.researchallofus.org/hc/en-us/articles/360039585391-Understanding-OMOP-Basics](https://support.researchallofus.org/hc/en-us/articles/360039585391-Understanding-OMOP-Basics)

### 8.5 Tools and Software

- **OHDSI Tools Homepage**: Available through OHDSI organization
- **White Rabbit**: Database scanning tool
- **Rabbit-In-A-Hat**: ETL design tool
- **USAGI**: Vocabulary mapping tool
- **Perseus**: Alternative mapping tool
- **ATLAS**: Web-based analytics platform
- **ACHILLES**: Data characterization tool

### 8.6 Academic Publications

- **Extending OMOP for Cancer Research**: [https://ascopubs.org/doi/full/10.1200/CCI.20.00079](https://ascopubs.org/doi/full/10.1200/CCI.20.00079)
- **OMOP CDM for Cancer Research Review**: [https://pmc.ncbi.nlm.nih.gov/articles/PMC11973147/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11973147/)
- **OMOP for Clinical Research Eligibility**: [https://pmc.ncbi.nlm.nih.gov/articles/PMC5893219/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5893219/)

## 9. Summary and Key Takeaways

### 9.1 Why OMOP CDM?

- **Standardization**: Enables consistent data structure across diverse healthcare databases
- **Interoperability**: Semantic standardization through common vocabularies
- **Scale**: Access to billion+ patient records through network research
- **Privacy**: Distributed model protects patient confidentiality
- **Open Source**: Free to use with strong community support
- **Research-Ready**: Purpose-built for observational health research

### 9.2 Critical Success Factors

1. **Executive Sponsorship**: Obtain organizational commitment
2. **Expert Collaboration**: Engage data, clinical, and technical experts
3. **Quality Mapping**: Invest in accurate vocabulary mapping
4. **Iterative Approach**: Plan for multiple ETL refinement cycles
5. **Community Engagement**: Participate in OHDSI community
6. **Documentation**: Maintain thorough documentation of all decisions

### 9.3 Common Use Cases for Healthcare Organizations

- **Population Health Management**: Characterize patient populations and disease patterns
- **Quality Improvement**: Track outcomes and quality metrics
- **Clinical Research**: Conduct observational studies with standardized methods
- **Pharmacovigilance**: Monitor drug safety signals
- **Comparative Effectiveness**: Evaluate treatment alternatives
- **Predictive Analytics**: Build risk models for precision medicine
- **Network Collaboration**: Participate in multi-site research studies

---

**Document Version**: 1.0
**Date**: 2025-11-30
**Based on**: OMOP CDM v5.4 specifications and OHDSI community resources
**Author**: Research compilation for hybrid-reasoning-acute-care project
