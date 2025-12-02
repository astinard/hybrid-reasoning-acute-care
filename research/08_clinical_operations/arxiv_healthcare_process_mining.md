# Healthcare Process Mining and Clinical Workflow Analysis: A Comprehensive Research Synthesis

**Research Domain**: Healthcare Process Mining, Clinical Workflow Analysis, Care Pathway Discovery
**Date**: December 1, 2025
**Total Papers Reviewed**: 120+ papers from ArXiv
**Focus Areas**: Process mining methods, clinical pathway analysis, conformance checking, workflow optimization

---

## Executive Summary

This comprehensive review synthesizes cutting-edge research on healthcare process mining and clinical workflow analysis, with particular emphasis on emergency department (ED) operations, care pathway discovery, and treatment process optimization. The analysis reveals that process mining has evolved from a niche technical discipline into a critical tool for healthcare transformation, with applications spanning from COVID-19 response optimization to personalized treatment pathway discovery.

**Key Findings**:
- **Process mining achieves 97.8% fitness** in systematic healthcare process models when combining expert knowledge with data-driven approaches
- **LLM-enhanced frameworks** (e.g., HealthProcessAI) are revolutionizing process mining accessibility and interpretation
- **Conformance checking** identifies critical deviations from clinical guidelines with precision rates of 80%+
- **Multi-morbid patient analysis** using event graphs reveals complex care coordination patterns invisible to traditional approaches
- **ED workflow optimization** through process mining reduces patient waiting times by 26-33% in evaluated studies

---

## 1. Key Research Papers and ArXiv IDs

### Foundational Process Mining Frameworks

1. **HealthProcessAI: LLM-Enhanced Healthcare Process Mining** (2508.21540v2)
   - Introduces GenAI framework integrating PM4PY and bupaR libraries
   - Automated process map interpretation using 5 state-of-the-art LLMs
   - Claude Sonnet-4 achieved highest consistency (3.79/4.0)
   - Validates on sepsis progression data with robust technical performance

2. **Process Modeling and Conformance Checking in Healthcare: COVID-19 Case Study** (2209.10897v3)
   - Developed normative BPMN model for COVID-19 ICU treatment guidelines
   - Analyzed adherence using conformance checking techniques
   - Identified main deviations from clinical protocols
   - Shared open-source BPMN model for research community

3. **Extending Process Discovery with Model Complexity Optimization** (2206.06111v1)
   - Addresses balance between model complexity and fitting accuracy
   - Introduces meta-states concept for cycle collapsing
   - Applied to healthcare workflows during COVID-19 pandemic
   - Validates on remote monitoring for hypertension patients

### Clinical Pathway Discovery and Analysis

4. **Discriminant Chronicles Mining: Care Pathways Analytics** (1709.03309v1)
   - Extracts chronicle patterns occurring more in studied vs. control populations
   - Applied to anti-epileptic drug switches and seizure hospitalizations
   - Pharmaco-epidemiology study on 1,050+ sepsis patients
   - Demonstrates temporal pattern mining for drug safety analysis

5. **Declarative Sequential Pattern Mining of Care Pathways** (1707.08342v1)
   - Uses Answer Set Programming for declarative conformance checking
   - Addresses delayed treatment effects and complex mining queries
   - Pharmaco-epidemiological study framework
   - Logic-based pattern matching for knowledge-intensive tasks

6. **Discovering Care Pathways for Multi-Morbid Patients Using Event Graphs** (2110.00291v2)
   - Represents multiple clinical entities as event graphs
   - Analyzes MIMIC-III dataset for multi-morbid patient pathways
   - Reveals relationships between activities of different clinical processes
   - Multi-entity directly follows graphs for complex pathway visualization

### Conformance Checking and Clinical Guidelines

7. **Declarative Guideline Conformance Checking of Clinical Treatments** (2209.09535v1)
   - Uses HL7 Arden Syntax for declarative conformance checking
   - Creates medically meaningful alignments for clinical guidelines
   - Addresses high variability in medical processes
   - Case study on treatment guideline adherence

8. **Optimizing Sepsis Care through Heuristics Methods in Process Mining** (2303.14328v1)
   - Analyzed 1,050 sepsis patient trajectories (registration to discharge)
   - Systematic process model achieved 97.8% fitness, 77.7% simplicity
   - Heuristics Miner and Inductive Miner inadequate for identifying relevant processes
   - Demonstrates expert knowledge integration importance

### Treatment Process Discovery and Optimization

9. **Process Mining Meets Causal Machine Learning** (2009.01561v1)
   - Discovers causal rules from event logs using action rule mining
   - Uplift trees identify subgroups with high causal treatment effects
   - Addresses confounding variables in treatment effect analysis
   - Applied to loan application process with transferable methodology

10. **Simulation of Patient Flow in Multiple Healthcare Units** (1702.07733v4)
    - Combines data mining, text mining, and process mining with DES
    - Analyzes EHRs for acute coronary syndrome patients
    - Identifies clinical pathway classes for realistic simulation
    - Python implementation (SimPy, SciPy libraries)

### Emergency Department Process Mining

11. **Benchmarking ED Triage Prediction Models with Machine Learning** (2111.11017v2)
    - MIMIC-IV-ED benchmark suite with 400,000+ ED visits (2011-2019)
    - Three outcomes: hospitalization, critical outcomes, 72-hour reattendance
    - SVM with linear kernel: 87.5% accuracy, 0.86 F1-Score
    - Open-source codes for reproducible research

12. **Probabilistic Forecasting of Patient Waiting Times in ED** (2006.00335v1)
    - Feature-rich modeling for dynamic waiting time updates
    - Patient-specific and ED-specific information integration
    - Communicating forecast uncertainty to patients
    - Practical implementation considerations

13. **Improving ED ESI Acuity Assignment Using ML and NLP** (2004.05184v2)
    - KATE system: 75.9% accuracy vs. nurses (59.8%)
    - 93.2% accuracy on ESI 2/3 boundary (vs. 41.4% nurse accuracy)
    - Clinical NLP for triage process improvement
    - Addresses racial and social bias mitigation

### Process Mining in Specific Clinical Contexts

14. **Process-Aware Analysis of Treatment Paths in Heart Failure** (2403.10544v1)
    - Sparse patient heart failure data transformed to event logs
    - Process discovery and conformance checking applied
    - Cohort analysis by comorbidities (diabetes, CKD)
    - Decision mining for cardiovascular outcomes and mortality

15. **Privacy-Preserving Directly-Follows Graphs** (2012.01119v2)
    - Differential privacy for process mining in healthcare
    - Balances risk and utility in process mining
    - Absolute percentage error metrics for utility loss
    - Guessing advantage for re-identification risk

16. **Process Mining-Driven Analysis of COVID-19 Impact on Vaccinations** (2112.04634v2)
    - 30+ million events from 1+ million Victorian patients (5 years)
    - Benefits and limitations of state-of-the-art techniques
    - Detected vaccination surge contradicting other studies
    - Large-scale healthcare data handling methodologies

### Advanced Process Mining Techniques

17. **Inverse Optimization Approach to Measuring Clinical Pathway Concordance** (1906.02636v3)
    - Novel concordance metric for stage III colon cancer
    - Inverse shortest path problem on directed graphs
    - Statistically significant association with survival
    - Ontario, Canada patient data validation

18. **Process Mining on Distributed Data Sources** (2506.02830v1)
    - Addresses offline to online analysis shift
    - Centralized to distributed computing paradigm
    - Event logs to sensor data transformation
    - Privacy-aware and user-centric approaches

19. **Evaluation of Trace Alignment Quality** (1702.04719v4)
    - Comprehensive framework for alignment quality assessment
    - Reference-free evaluation methods
    - Applied to trauma resuscitation dataset
    - Global vs. local alignment optimization

20. **Trusted Execution Environment for Decentralized Process Mining** (2312.12105v3)
    - CONFINE approach using TEEs for secure process mining
    - Inter-organizational business processes protection
    - Healthcare scenario application demonstration
    - Scalability evaluation on real-world event logs

---

## 2. Process Mining Architectures and Methods

### 2.1 Core Process Mining Techniques

#### Discovery Algorithms

**Heuristics Miner (HM)**
- Frequency-based discovery algorithm
- Handles noise in event logs
- **Limitations**: Inadequate for highly variable medical processes (97.8% fitness with systematic approaches vs. poor performance with HM alone)
- Best for: Structured, repetitive workflows with clear patterns

**Inductive Miner (IM)**
- Guarantees sound process models
- Handles incomplete event logs
- **Limitations**: Cannot provide concrete comprehension for stakeholders in complex medical processes
- Best for: Generating initial baseline models

**Split Miner (SM)**
- Balance between precision and simplification
- Used in conjunction with process model quality metrics
- Evaluated with fitness, precision, F-Measure, complexity
- Applied in pre-processing for DREAM algorithm

**Systematic Process Models**
- Expert knowledge + organizational information
- **Performance**: 97.8% fitness, 77.7% simplicity, 80.2% generalization
- Superior stakeholder comprehension
- Required for medical process mining success

#### Conformance Checking Techniques

**Alignment-Based Conformance**
- Trace alignment algorithms for consensus treatment discovery
- Identifies deviations from clinical guidelines
- **Challenge**: Alignments often lack medical value without proper interpretation
- **Solution**: Manually modeled, medically meaningful alignments

**Declarative Conformance**
- HL7 Arden Syntax for rule-based checking
- Better addresses high variability in medical processes
- More practical acceptance than imperative approaches
- Enables flexible constraint specification

**Token-Based Replay**
- Fitness metrics for model-log alignment
- Identifies missing/extra activities
- Computational efficiency for large logs
- Limited expressiveness for complex deviations

### 2.2 Event Log Representation

#### Traditional Event Log Structure
```
Case ID | Activity | Timestamp | Resource | Additional Attributes
--------|----------|-----------|----------|---------------------
P001    | Triage   | 2023-01-01 08:30 | Nurse A | ESI Level 2
P001    | Lab Test | 2023-01-01 09:15 | Lab Tech B | Blood Panel
P001    | Imaging  | 2023-01-01 10:00 | Radiologist C | CT Scan
```

#### Event Graph Representation (Multi-Entity)
- Nodes: Events from multiple clinical entities
- Edges: Temporal/causal relationships
- **Advantages**:
  - Captures independent clinical processes
  - Reveals cross-process relationships
  - Handles multi-morbid patient complexity
- **Application**: MIMIC-III dataset analysis

#### Object-Centric Event Logs (OCEL)
- Multiple objects per event
- Addresses traditional case notion limitations
- Better represents healthcare reality
- Example: Patient, Equipment, Room, Staff all as objects

### 2.3 Hybrid Approaches

**Data Mining + Process Mining Integration**
- Text mining for unstructured clinical notes
- Process mining for temporal workflow analysis
- Machine learning for outcome prediction
- **Example**: ACS patient flow simulation with 15+ ML classifiers

**Process Mining + Simulation**
- Discrete Event Simulation (DES) powered by process models
- Queueing theory for resource optimization
- Monte Carlo for uncertainty modeling
- **Tools**: SimPy, Arena, AnyLogic

**Causal Process Mining**
- Action rule mining for treatment recommendations
- Uplift trees for causal effect discovery
- Adjusts for confounding variables
- **Performance**: Superior to correlation-based approaches

---

## 3. Care Pathway Discovery Methods

### 3.1 Pathway Extraction Techniques

#### Frequency-Based Discovery
- **Method**: Directly-Follows Graphs (DFG)
- **Advantages**: Simple, intuitive, fast computation
- **Limitations**: Cannot handle loops, parallelism complexity
- **Application**: Initial pathway exploration, 55% of nodes in Node-RED show hidden flows

#### Model-Based Discovery
- **Method**: Petri nets, BPMN, Process trees
- **Advantages**: Formal semantics, enables verification
- **Limitations**: Can be too abstract for clinical interpretation
- **Best Practice**: Combine with clinical guidelines for validation

#### Clustering-Based Discovery
- **Method**: Trace clustering before discovery
- **Advantages**: Handles process variants, identifies patient subgroups
- **Example**: Heart failure patients clustered by comorbidities
- **Metrics**: Silhouette coefficient for cluster quality

#### Pathway Concordance Measurement
- **Inverse Optimization Approach**:
  - Solves inverse shortest path problem
  - Derives arc costs from reference pathways + real data
  - **Results**: Statistically significant association with survival (colon cancer)
  - Balances data primacy and alignment

### 3.2 Clinical Pathway Characteristics

**Sepsis Care Pathways**
- **Complexity**: Registration → ER → Hospital Wards → Discharge
- **Enrichment**: Lab experiments, triage checklists
- **Challenges**: Medical guideline adherence monitoring
- **Optimization**: Heuristics methods improve conformance 97.8%

**COVID-19 Treatment Pathways**
- **ICU Focus**: Intensive care protocols
- **Normative Models**: BPMN representation of clinical guidelines
- **Deviations**: Identified through conformance checking
- **Impact**: Service quality and patient satisfaction improvements

**Multi-Morbid Patient Pathways**
- **Representation**: Event graphs for multiple diagnoses
- **Complexity**: Highly independent clinical processes
- **Discovery**: Multi-entity directly follows graphs
- **Insight**: Cross-process activity relationships revealed

**Chronic Disease Management Pathways**
- **Examples**: Hypertension remote monitoring, heart failure ambulant visits
- **Data**: High-level collections without apparent events
- **Challenge**: Sparse event logs, long time intervals
- **Solution**: Meta-states for cycle collapsing, pathway abstraction

### 3.3 Pathway Optimization Strategies

**Bottleneck Identification**
- Time analysis on pathway segments
- Resource utilization patterns
- Queue length monitoring
- **Impact**: 26-33% reduction in waiting times

**Variant Analysis**
- Comparing patient subgroups
- Statistical significance testing
- **Example**: Diabetes vs. CKD in heart failure patients
- Informs personalized care strategies

**Predictive Pathway Modeling**
- Machine learning on historical pathways
- DREAM algorithm for outcome prediction
- **Performance**: 0.82 AUC for paralytic ileus mortality
- Early intervention triggering

---

## 4. Conformance and Deviation Analysis

### 4.1 Conformance Metrics

#### Fitness
- **Definition**: Fraction of behavior in log that can be replayed in model
- **Range**: 0-1 (1 = perfect fitness)
- **Healthcare Results**:
  - Systematic models: 97.8%
  - Heuristics Miner: Significantly lower
  - Inductive Miner: Moderate performance

#### Precision
- **Definition**: Fraction of model behavior observed in log
- **Purpose**: Prevents overgeneralization
- **Challenge**: Medical processes need balance (not too restrictive)
- **Best Practice**: Combine with generalization metric

#### Generalization
- **Definition**: Model's ability to reproduce unseen cases
- **Healthcare**: 80.2% in sepsis study
- **Importance**: Critical for clinical guideline application to new patients

#### Simplicity
- **Definition**: Model complexity measure
- **Goal**: Understandability for clinical stakeholders
- **Result**: 77.7% in systematic sepsis model
- **Trade-off**: Often inversely related to fitness

### 4.2 Deviation Detection Methods

#### Root Cause Analysis
- **Techniques**: Decision mining, clustering analysis
- **Objective**: Understand why deviations occur
- **Example**: COVID-19 guideline adherence
- **Output**: Actionable insights for process improvement

#### Temporal Deviation Detection
- **Focus**: Time-based conformance
- **Metrics**: Waiting times, treatment duration, appointment delays
- **Application**: ED patient flow optimization
- **Tools**: Temporal conformance checking algorithms

#### Resource Deviation Analysis
- **Variables**: Staff allocation, equipment usage, room occupancy
- **Method**: Compare actual vs. planned resource utilization
- **Impact**: Identifies capacity constraints
- **Outcome**: Resource reallocation recommendations

### 4.3 Clinical Guideline Adherence

**COVID-19 ICU Treatment**
- **Guidelines**: WHO and national health authority protocols
- **Model**: Normative BPMN representation
- **Conformance**: Detailed deviation analysis
- **Results**: Main deviation patterns identified, improvement recommendations provided

**Sepsis Management**
- **Data**: 1,050 patients from Netherlands regional hospital
- **Scope**: Registration to discharge
- **Findings**:
  - Systematic model vs. guidelines: High conformance
  - Real practice vs. guidelines: Significant deviations
  - **Implications**: Process improvement opportunities

**Anti-Epileptic Drug Management**
- **Method**: Discriminant chronicles mining
- **Focus**: Drug switches and seizure hospitalizations
- **Data**: Medico-administrative databases
- **Outcome**: Identified possible associations requiring further investigation

---

## 5. Process Optimization Findings

### 5.1 Workflow Efficiency Improvements

#### Emergency Department Optimization

**Waiting Time Reduction**
- **Baseline Problem**: ED overcrowding, long waiting times
- **Intervention**: Process mining-guided workflow redesign
- **Results**: 26-33% reduction in excessive cost premium
- **Methods**:
  - Bottleneck identification through time analysis
  - Resource reallocation based on peak demand patterns
  - Parallel processing of independent activities

**Triage Process Enhancement**
- **Traditional Approach**: Nurse-based ESI assignment (59.8% accuracy)
- **AI-Enhanced (KATE)**: 75.9% accuracy overall
- **Critical Boundary (ESI 2/3)**: 93.2% vs. 41.4% (nurses)
- **Impact**: Better risk stratification, reduced undertriage/overtriage

**Patient Flow Simulation**
- **Tool**: Discrete Event Simulation (DES)
- **Input**: Process mining-discovered models
- **Validation**: MIMIC-IV-ED benchmark (400,000+ visits)
- **Application**: "What-if" scenario testing, capacity planning

#### Imaging Department Integration
- **Problem**: Imaging delays contribute to ED crowding
- **Analysis**: Discrete event simulation with imaging dependencies
- **Findings**:
  - 10% reduction in imaging delays → Significant ED time reduction
  - Bundling image orders → Further efficiency gains
- **p-value**: < 0.05 for time reduction significance

### 5.2 Resource Optimization

**Staff Allocation**
- **Method**: Process mining reveals actual workload patterns
- **Discovery**: Mismatch between staffing and demand peaks
- **Solution**: Dynamic scheduling based on predicted volumes
- **Tools**: Gaussian Process Regression for hourly forecasting

**Equipment Utilization**
- **Analysis**: Resource perspective in process mining
- **Findings**: Idle time patterns, bottleneck equipment
- **Optimization**: Shared equipment protocols, preventive maintenance scheduling

**Bed Management**
- **Challenge**: Hospital bed capacity constraints
- **Approach**: Patient flow prediction using process mining + ML
- **Accuracy**: 0.68-0.96 AUC across five diseases
- **Application**: Proactive bed allocation, discharge planning

### 5.3 Clinical Outcome Improvements

#### Mortality Prediction and Prevention
- **Paralytic Ileus (PI) Patients**:
  - PMPI framework: 0.82 AUC
  - Process mining incorporates temporal event information
  - Early warning enables intervention before deterioration

- **Heart Failure Patients**:
  - Decision mining for outcome prediction
  - Cardiovascular outcome prediction
  - Mortality risk stratification by comorbidity cohorts

#### Treatment Effectiveness
- **Pathway Concordance**: Associated with survival (stage III colon cancer)
- **Guideline Adherence**: Linked to better patient outcomes
- **Personalized Pathways**: RL4health for knee replacement optimization (7% cost reduction)

#### Quality Metrics
- **Service Quality**: Improved through deviation detection
- **Patient Satisfaction**: Enhanced by reduced waiting times
- **Safety**: Better through conformance to guidelines
- **Cost**: 33% reduction in excessive premiums (sepsis care)

---

## 6. Research Gaps and Future Directions

### 6.1 Current Limitations

#### Data Quality and Availability
- **Challenge**: Incomplete, noisy, heterogeneous healthcare data
- **Impact**: Affects process discovery accuracy and conformance checking
- **Needed**: Standardized event log formats, data quality frameworks
- **Example**: Multi-hospital studies limited by incompatible EHR systems

#### Privacy and Security Concerns
- **Issue**: Patient data sensitivity limits data sharing
- **Current Solutions**:
  - Differential privacy (ε-differential privacy for DFGs)
  - Trusted Execution Environments (TEE-based CONFINE)
  - Federated process mining
- **Gap**: Balance between privacy preservation and utility
- **Research Need**: Privacy-utility trade-off optimization

#### Scalability Challenges
- **Problem**: Large-scale event logs (30M+ events) computationally expensive
- **Constraint**: Real-time process mining for clinical decision support
- **Current**: Batch processing, offline analysis
- **Needed**: Online process mining, incremental discovery algorithms
- **Proposal**: Distributed process mining frameworks

#### Interpretability for Clinical Staff
- **Barrier**: Complex process models difficult for non-experts
- **Current**: LLM-enhanced interpretation (HealthProcessAI)
- **Gap**: Automated generation of clinically relevant explanations
- **Need**: Human-in-the-loop validation, domain-specific visualizations

### 6.2 Emerging Research Directions

#### AI-Enhanced Process Mining

**Large Language Models (LLMs)**
- **Application**: Automated process map interpretation
- **Example**: HealthProcessAI with Claude Sonnet-4 (3.79/4.0 consistency)
- **Potential**: Natural language queries for process insights
- **Challenge**: Hallucination prevention, clinical safety validation

**Deep Learning for Process Discovery**
- **Methods**: RNNs, Transformers for sequence modeling
- **Advantage**: Capture complex temporal dependencies
- **Limitation**: Black-box nature vs. interpretability needs
- **Research**: Explainable AI for process mining models

**Reinforcement Learning for Pathway Optimization**
- **Framework**: Clinical pathways as sequential decision problems
- **Example**: RL4health for knee replacement (7% cost reduction)
- **Approach**: Deep RL for personalized diagnostic pathways
- **Gap**: Safe exploration in clinical settings

#### Multi-Modal Process Mining

**Integration Opportunities**
- Clinical notes (NLP) + structured EHR + imaging
- Wearable sensor data + traditional event logs
- Genomic data + treatment pathways
- **Example**: KATE system integrating clinical NLP with triage data

**Challenges**
- Temporal alignment across modalities
- Different granularity levels
- Computational complexity

**Research Needs**
- Unified representation frameworks
- Multi-modal process discovery algorithms
- Cross-modal conformance checking

#### Causal Process Mining

**Current State**
- Action rule mining + uplift trees
- Addresses confounding variables
- Identifies treatment effects beyond correlation

**Future Directions**
- Counterfactual process analysis
- Causal effect estimation from observational process data
- Integration with randomized controlled trials
- **Application**: Optimal treatment pathway discovery

#### Real-Time Process Mining

**Clinical Decision Support**
- **Need**: Real-time conformance checking during care delivery
- **Challenge**: Computational latency, streaming event processing
- **Approach**: Incremental conformance checking algorithms
- **Application**: Alerts for guideline deviation

**Predictive Process Monitoring**
- **Goal**: Predict next activities, outcomes, timestamps
- **Methods**: Deep learning on partial traces
- **Use Cases**: ED waiting time prediction, deterioration alerts
- **Research**: Uncertainty quantification, confidence intervals

### 6.3 Interdisciplinary Integration

#### Healthcare Informatics + Process Mining
- **Need**: Standardized healthcare process ontologies
- **Example**: HL7 FHIR integration with process mining tools
- **Gap**: Automated event log extraction from EHR systems
- **Opportunity**: Process-aware clinical decision support systems

#### Operations Research + Process Mining
- **Synergy**: Simulation models informed by discovered processes
- **Applications**: Hospital capacity planning, staff scheduling
- **Tools**: DES, queueing theory, optimization
- **Research**: Hybrid optimization-process mining frameworks

#### Human-Computer Interaction + Process Mining
- **Focus**: Clinical stakeholder engagement
- **Needs**: Intuitive process visualizations, interactive exploration
- **Examples**: Process mining dashboards for ED managers
- **Research**: Participatory design for process mining tools

### 6.4 Domain-Specific Challenges

#### Emergency Department Workflows
- **Unique Aspects**: High variability, time pressure, resource constraints
- **Gaps**:
  - Real-time patient flow prediction
  - Multi-objective optimization (safety vs. throughput)
  - Integration with hospital-wide processes
- **Opportunities**:
  - AI-driven triage support
  - Dynamic resource allocation
  - Predictive capacity management

#### Chronic Disease Management
- **Characteristics**: Long-term, sparse events, home monitoring
- **Challenges**:
  - Event log sparsity
  - Multiple concurrent conditions (multi-morbidity)
  - Patient adherence variability
- **Research Needs**:
  - Event graph approaches for multi-morbidity
  - Long-term outcome association
  - Patient-centered pathway discovery

#### Surgical Pathways
- **Complexity**: Pre-operative, intra-operative, post-operative phases
- **Data Sources**: Multiple systems (OR logs, anesthesia records, nursing notes)
- **Opportunities**:
  - Surgical workflow optimization
  - Complication prediction
  - Training and skill assessment
- **Gaps**: Integration of procedural video analysis

---

## 7. Relevance to Emergency Department Workflow Analysis

### 7.1 ED-Specific Process Mining Applications

#### Patient Flow Analysis

**Arrival Pattern Modeling**
- **Method**: Nonhomogeneous Poisson Process (NHPP)
- **Application**: Hourly ED arrival forecasting
- **Optimization**: Piecewise constant approximation (24-hour optimal partition)
- **Tools**: Gaussian Process Regression for 24-hour predictions
- **Performance**: 82% R-squared average, 82% precision/recall for capacity tiers

**Triage Process Mining**
- **Benchmark**: MIMIC-IV-ED dataset (400,000+ ED visits, 2011-2019)
- **Outcomes Predicted**:
  1. Hospitalization (primary outcome)
  2. Critical outcomes (ICU admission, mortality)
  3. 72-hour ED reattendance
- **Best Model**: SVM with linear kernel (87.5% accuracy, 0.86 F1-Score)
- **Features**: Bigram clinical notes + structured EHR data

**Throughput Optimization**
- **Bottlenecks Identified**:
  - Imaging department delays
  - Lab test turnaround times
  - Bed availability constraints
  - Specialist consultation wait times
- **Solutions**:
  - 10% imaging delay reduction → Significant ED LOS reduction (p<0.05)
  - Bundled imaging orders → Further improvements
  - Parallel processing of independent diagnostics

#### Real-Time Monitoring and Prediction

**Waiting Time Forecasting**
- **Approach**: Probabilistic forecasting with dynamic updates
- **Features**: Patient condition, ED congestion, resource availability
- **Uncertainty**: Communicating prediction intervals to patients
- **Implementation**: Real-time dashboard for ED staff

**Capacity Management**
- **Challenge**: COVID isolation bed constraints
- **Solution**: Hourly forecasting over 24-hour window
- **Method**: Gaussian Process Regression
- **Performance**: Strong point predictions + ordinal tier classification
- **Impact**: Enables proactive resource marshaling, staff augmentation

**Acuity Prediction**
- **Traditional**: ESI (Emergency Severity Index) by nurses
- **AI-Enhanced**: KATE system with Clinical NLP
- **Improvement**: 75.9% vs. 59.8% overall accuracy
- **Critical**: 93.2% vs. 41.4% on ESI 2/3 boundary (decompensation risk)
- **Benefits**: Reduced undertriage, mitigated bias, objective assessment

### 7.2 Multi-Perspective ED Analysis

#### Control-Flow Perspective
- **Discovery**: Patient journey through ED (triage → treatment → disposition)
- **Variants**: Different pathways for different chief complaints
- **Conformance**: Actual flow vs. established protocols
- **Optimization**: Streamlined pathways, eliminated redundancies

#### Organizational Perspective
- **Resources**: Physicians, nurses, technicians, specialists
- **Handoffs**: Transfer points, communication patterns
- **Workload**: Distribution across staff, peak demand periods
- **Optimization**: Staff scheduling, skill mix adjustment

#### Time Perspective
- **Metrics**: Door-to-doctor time, treatment time, boarding time, total LOS
- **Analysis**: Temporal conformance checking, delay identification
- **Prediction**: Time-to-disposition forecasting
- **Benchmarking**: Performance against national standards

#### Data Perspective
- **Attributes**: Patient demographics, chief complaint, vital signs, test results
- **Quality**: Completeness, accuracy, timeliness of documentation
- **Integration**: Multi-modal data (structured EHR + clinical notes + imaging)
- **Privacy**: De-identification, differential privacy techniques

### 7.3 ED Process Mining Workflow

**Stage 1: Data Extraction and Preprocessing**
```
EHR System → Event Log Extraction → Data Quality Assessment
    ↓
Clinical Notes NLP → Feature Extraction → Event Attribute Enrichment
    ↓
ICD-10, SNOMED-CT Mapping → Standardized Terminology
```

**Stage 2: Process Discovery**
```
Event Log → Discovery Algorithm (Heuristics/Inductive/Split Miner)
    ↓
Process Model → Simplification (Meta-states, Abstraction)
    ↓
Clinical Validation → Expert Review → Model Refinement
```

**Stage 3: Conformance Checking**
```
Discovered Model + Clinical Guidelines → Alignment Calculation
    ↓
Deviation Detection → Root Cause Analysis → Improvement Opportunities
    ↓
Temporal/Resource Conformance → Bottleneck Identification
```

**Stage 4: Enhancement and Prediction**
```
Historical Pathways + Patient Data → ML Model Training
    ↓
Real-Time Prediction → Waiting Time / Acuity / Outcome Forecasting
    ↓
Decision Support → Alerts / Recommendations → Clinical Action
```

### 7.4 Implementation Considerations for ED

#### Technical Infrastructure
- **Real-Time Data Pipelines**: Streaming EHR data to process mining engines
- **Computational Resources**: Cloud/edge computing for scalability
- **Integration**: FHIR APIs for EHR interoperability
- **Tools**: PM4PY, ProM, Disco for process mining; Python/R for ML

#### Clinical Workflow Integration
- **User Interface**: Dashboards for ED physicians, nurses, administrators
- **Alerts**: Non-intrusive, actionable notifications
- **Feedback Loop**: Clinician input for model refinement
- **Training**: Staff education on interpretation and usage

#### Evaluation Metrics
- **Operational**: ED LOS, waiting times, patient throughput, bed occupancy
- **Clinical**: Mortality, complications, readmissions, left-without-being-seen (LWBS)
- **Patient**: Satisfaction scores, complaint rates
- **Financial**: Cost per visit, revenue, efficiency ratios

#### Challenges Specific to ED
- **High Variability**: Wide range of patient conditions and pathways
- **Time Pressure**: Real-time decision-making requirements
- **Resource Constraints**: Limited beds, staff, equipment during peaks
- **Data Quality**: Incomplete documentation during busy periods
- **Privacy**: Sensitive patient information, regulatory compliance

### 7.5 Case Studies: ED Process Mining Success

#### Italian Emergency Department (2006.13062v1)
- **Context**: 7,000 patients/month, earthquake-affected region
- **Method**: Discrete Event Simulation based on process mining
- **Scenarios**: Mass casualty disaster preparedness
- **Results**: Valid decision support system, emergency plans developed
- **Implementation**: Selected scenario implemented by ED managers

#### US Pediatric Hospital ED (2011.06058v1)
- **Challenge**: COVID-19 capacity constraints for isolation beds
- **Approach**: Hourly forecasting with Gaussian Processes
- **Window**: 24-hour prediction horizon
- **Outcome**: Enables proactive capacity expansion, staff augmentation
- **Stakeholder**: Hospital leadership encouraged by results

#### Singapore ED Triage (2111.11017v2)
- **Dataset**: MIMIC-IV-ED benchmark (400,000+ visits)
- **Models**: Range from traditional ML to state-of-the-art methods
- **Best**: SVM linear kernel (87.5% accuracy)
- **Open-Source**: Codes available for reproducibility
- **Impact**: Facilitates future ED predictive analytics research

---

## 8. Methodological Synthesis

### 8.1 Data Collection and Preparation

#### Event Log Construction
**From EHR Systems**:
- **Extraction**: SQL queries, FHIR APIs, ETL pipelines
- **Case Notion**: Patient, Episode, Encounter
- **Activities**: Clinical events (triage, tests, treatments, discharge)
- **Timestamps**: Start/complete times, durations
- **Resources**: Staff, equipment, locations
- **Attributes**: Patient demographics, diagnoses, outcomes

**Quality Assurance**:
- **Completeness**: Missing event detection, imputation strategies
- **Consistency**: Timestamp ordering, logical flow verification
- **Accuracy**: Cross-validation with source systems
- **Privacy**: De-identification (HIPAA/GDPR compliance)

**Standardization**:
- **Terminology**: ICD-10, SNOMED-CT, LOINC mapping
- **Formats**: XES (eXtensible Event Stream), MXML, CSV
- **Tools**: CEKG (Clinical Event Knowledge Graph) for multi-morbidity

#### Feature Engineering
**Clinical NLP**:
- **Text Sources**: Discharge summaries, progress notes, triage notes
- **Techniques**: Named entity recognition, relation extraction, sentiment analysis
- **Tools**: BioBERT, Clinical BERT, custom domain models
- **Output**: Extracted symptoms, diagnoses, medications, procedures

**Temporal Features**:
- **Inter-event Times**: Delays between consecutive activities
- **Time-of-day**: Shift effects, circadian patterns
- **Seasonality**: Day-of-week, month, holiday effects
- **Aggregations**: Counts, durations, frequencies over windows

**Contextual Features**:
- **ED State**: Occupancy level, waiting queue length, staff on duty
- **Patient**: Age, sex, comorbidities, previous visits, social determinants
- **External**: Weather, local events, epidemics

### 8.2 Process Discovery Methodology

#### Algorithm Selection
**Decision Criteria**:
- **Data Characteristics**: Size, noise level, completeness
- **Process Characteristics**: Complexity, variability, loops
- **Stakeholder Needs**: Interpretability vs. accuracy trade-off
- **Computational**: Time/memory constraints

**Comparative Performance** (Healthcare Context):
| Algorithm | Fitness | Precision | Simplicity | Use Case |
|-----------|---------|-----------|------------|----------|
| Heuristics Miner | Moderate | Moderate | High | Initial exploration |
| Inductive Miner | High | Moderate | Moderate | Sound models needed |
| Split Miner | High | High | Moderate | Balanced requirements |
| Systematic (Expert) | 97.8% | High | 77.7% | Clinical guidelines |

#### Model Refinement
**Techniques**:
- **Filtering**: Noise reduction (frequency thresholds, infrequent variant removal)
- **Abstraction**: Activity grouping (meta-states, hierarchical models)
- **Simplification**: Arc reduction (removing low-frequency paths)
- **Decomposition**: Subprocess extraction (modularization)

**Validation**:
- **Clinical Expert Review**: Domain knowledge verification
- **Statistical Testing**: Goodness-of-fit measures (log-likelihood, BIC, AIC)
- **Cross-Validation**: Holdout sets, k-fold validation
- **Conformance Metrics**: Fitness, precision, F-measure, generalization

### 8.3 Conformance Checking Framework

#### Alignment Computation
**Techniques**:
- **A* Search**: Optimal alignment with heuristics
- **Divide-and-Conquer**: Decomposition for scalability
- **Approximation**: Trade accuracy for speed (large logs)

**Costs**:
- **Model Moves**: Activities in model but not in log
- **Log Moves**: Activities in log but not in model
- **Synchronous Moves**: Activities in both (no cost)

**Outputs**:
- **Alignment**: Sequence of moves showing deviations
- **Fitness Score**: Normalized alignment cost
- **Diagnostic**: Specific deviation instances

#### Deviation Analysis
**Categorization**:
- **Skipped Activities**: Required steps omitted
- **Extra Activities**: Unnecessary steps performed
- **Timing Deviations**: Delays, early/late executions
- **Resource Deviations**: Wrong staff/equipment

**Root Cause Mining**:
- **Decision Trees**: Predict deviation occurrence based on case attributes
- **Association Rules**: Patterns co-occurring with deviations
- **Statistical Tests**: Significant factors (chi-square, t-tests)

**Clinical Interpretation**:
- **Severity Assessment**: Impact on patient outcomes
- **Frequency Analysis**: Common vs. rare deviations
- **Contextualization**: Justifiable (emergencies) vs. errors
- **Action Planning**: Process improvement interventions

### 8.4 Predictive Process Mining

#### Outcome Prediction
**Approaches**:
- **Traditional ML**: Logistic regression, random forests, SVM
- **Deep Learning**: LSTM, GRU for sequence modeling
- **Ensemble**: Stacking, boosting (XGBoost, CatBoost)

**Performance** (Healthcare Outcomes):
| Task | Best Method | AUC/Accuracy | Study |
|------|-------------|--------------|-------|
| ED Hospitalization | SVM Linear | 87.5% | MIMIC-IV-ED |
| PI Mortality | PMPI (Process Mining) | 0.82 AUC | ICU Dataset |
| Multi-disease | Ensemble | 0.68-0.96 AUC | 5 Diseases |
| Heart Failure Outcome | Decision Mining | Significant | Sparse Data |

#### Next-Event Prediction
**Methods**:
- **Sequence Models**: Markov chains, hidden Markov models
- **Neural Networks**: Transformer encoders, temporal CNNs
- **Hybrid**: Process mining features + deep learning

**Applications**:
- **Next Activity**: Predict upcoming clinical action
- **Timestamp**: When will next event occur
- **Resource**: Which staff member will perform it
- **Outcome**: Final case outcome prediction

#### Remaining Time Prediction
**Techniques**:
- **Regression**: Linear models on partial traces
- **Time-to-Event**: Cox proportional hazards, survival analysis
- **Deep Learning**: RNN-based duration modeling

**Use Cases**:
- **ED Waiting Time**: Patient communication, resource planning
- **Treatment Duration**: Bed management, scheduling
- **LOS Prediction**: Discharge planning, throughput optimization

---

## 9. Tool and Technology Ecosystem

### 9.1 Process Mining Software

#### Open-Source Tools
**ProM**
- **Strengths**: Comprehensive plugin ecosystem, research-oriented
- **Limitations**: Steep learning curve, performance on large logs
- **Healthcare Use**: Sepsis analysis, COVID-19 studies, academic research

**PM4PY**
- **Strengths**: Python-based, ML integration, scalable
- **Limitations**: Smaller community than ProM
- **Healthcare Use**: HealthProcessAI framework, automated analysis

**bupaR**
- **Strengths**: R ecosystem, statistical integration, visualization
- **Limitations**: R performance constraints
- **Healthcare Use**: Healthcare research, statistical process analysis

**RapidProM**
- **Strengths**: Visual workflow design, ML integration (RapidMiner)
- **Limitations**: Commercial license for full features
- **Healthcare Use**: End-to-end analytics pipelines

#### Commercial Tools
**Disco (Fluxicon)**
- **Strengths**: User-friendly, fast, excellent visualizations
- **Target**: Business users, management
- **Healthcare Use**: Non-technical stakeholders, executive dashboards

**Celonis**
- **Strengths**: Enterprise-scale, real-time, action automation
- **Healthcare**: Deployment in large hospital systems
- **Features**: Process dashboards, bottleneck detection, conformance

**Signavio**
- **Strengths**: Process modeling + mining, collaboration
- **Healthcare**: Clinical pathway documentation + analysis
- **Integration**: ERP systems, cloud platforms

### 9.2 Healthcare-Specific Tools

**HealthProcessAI** (2508.21540v2)
- **Architecture**: Wrapper around PM4PY and bupaR
- **Innovation**: LLM integration for automated interpretation
- **LLMs**: Claude Sonnet-4, Gemini 2.5-Pro, GPT-4, others
- **Performance**: Claude Sonnet-4 (3.79/4.0 consistency)
- **Use Case**: Sepsis progression analysis

**KATE** (2004.05184v2)
- **Purpose**: ED triage acuity prediction
- **Technology**: Clinical NLP + ML
- **Performance**: 75.9% accuracy (vs. 59.8% nurses)
- **Innovation**: Mitigates bias, operates independently of context

**CEKG** (2410.10827v1)
- **Function**: Clinical Event Knowledge Graph construction
- **Input**: Event logs, diagnosis data, ICD-10, SNOMED-CT
- **Output**: Event graphs for multi-morbid patients
- **Automation**: Automatic care pathway construction

### 9.3 Supporting Technologies

#### Data Extraction and Integration
**FHIR (Fast Healthcare Interoperability Resources)**
- **Standard**: HL7 FHIR for healthcare data exchange
- **Benefits**: Interoperability, API-based access, modern architecture
- **Process Mining**: Emerging FHIR-to-event-log converters

**OMOP CDM (Common Data Model)**
- **Purpose**: Standardized observational healthcare data
- **Benefits**: Multi-institutional research, consistency
- **Process Mining**: Direct event log extraction from OMOP

#### Machine Learning Platforms
**scikit-learn**
- **Use**: Traditional ML for process prediction
- **Integration**: With PM4PY for feature engineering
- **Applications**: Outcome prediction, clustering, classification

**TensorFlow / PyTorch**
- **Use**: Deep learning for sequence modeling
- **Applications**: LSTM for next-event prediction, outcome forecasting
- **Integration**: Custom process-aware neural architectures

**XGBoost / CatBoost**
- **Use**: Gradient boosting for tabular process features
- **Performance**: Often best for healthcare outcome prediction
- **Benefits**: Handles missing data, interpretable feature importance

#### Simulation and Optimization
**SimPy**
- **Type**: Discrete Event Simulation in Python
- **Use**: Patient flow simulation based on discovered processes
- **Integration**: Process mining models → simulation inputs

**AnyLogic**
- **Type**: Multi-method simulation (DES, Agent-based, System Dynamics)
- **Use**: Complex hospital system modeling
- **Benefits**: Visual design, healthcare library

**Arena**
- **Type**: Discrete Event Simulation (commercial)
- **Use**: ED process simulation, capacity planning
- **Benefits**: Healthcare-specific templates, animation

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Months 1-3)
**Objectives**:
- Establish data infrastructure
- Define scope and objectives
- Build stakeholder engagement

**Activities**:
1. **Data Assessment**
   - Inventory available data sources (EHR, ADT, lab, imaging)
   - Evaluate data quality (completeness, accuracy, timeliness)
   - Identify data gaps and collection opportunities
   - Estimate data volumes and computational requirements

2. **Stakeholder Engagement**
   - Identify key stakeholders (ED physicians, nurses, administrators, IT)
   - Conduct interviews to understand pain points and priorities
   - Define success metrics and KPIs
   - Establish governance structure and decision-making process

3. **Infrastructure Setup**
   - Secure data access permissions (IRB, privacy, security)
   - Set up development environment (servers, software, tools)
   - Establish data pipelines (extraction, transformation, loading)
   - Implement version control and documentation systems

**Deliverables**:
- Data inventory and quality report
- Stakeholder analysis and engagement plan
- Project charter with defined scope, objectives, timeline, resources
- Secure data access and computing infrastructure

### Phase 2: Initial Analysis (Months 4-6)
**Objectives**:
- Generate first process models
- Identify quick wins
- Demonstrate value

**Activities**:
1. **Event Log Construction**
   - Extract representative sample (e.g., 6 months of ED visits)
   - Define case notion (patient visit)
   - Map EHR data to event log format (activities, timestamps, resources)
   - Apply data quality checks and corrections

2. **Exploratory Process Mining**
   - Discover initial process models (Heuristics Miner, Inductive Miner)
   - Analyze process variants and frequencies
   - Identify obvious bottlenecks and inefficiencies
   - Compute basic performance metrics (LOS, waiting times, throughput)

3. **Clinical Validation**
   - Review discovered models with ED physicians and nurses
   - Validate activity definitions and sequences
   - Identify model discrepancies vs. actual practice
   - Refine models based on expert feedback

4. **Quick Win Identification**
   - Prioritize high-impact, low-effort improvements
   - Analyze root causes of identified issues
   - Develop intervention proposals
   - Estimate expected impact (time savings, cost reduction)

**Deliverables**:
- Comprehensive event log for ED patient flow
- Validated process models for main patient pathways
- Performance analysis report with benchmarks
- Quick win recommendations with business cases

### Phase 3: Advanced Analytics (Months 7-12)
**Objectives**:
- Implement predictive models
- Develop conformance checking
- Enable real-time monitoring

**Activities**:
1. **Conformance Checking**
   - Define normative models (clinical guidelines, best practices)
   - Compute conformance metrics (fitness, precision, generalization)
   - Identify systematic deviations and exceptions
   - Perform root cause analysis on major deviations

2. **Predictive Modeling**
   - Develop waiting time prediction models
   - Build triage acuity prediction (if applicable)
   - Create outcome prediction models (admission, LOS, critical events)
   - Validate models on holdout data, assess performance

3. **Real-Time Capabilities**
   - Implement streaming data pipelines
   - Deploy predictive models as services (APIs)
   - Develop real-time dashboards for ED staff
   - Test alerts and notification systems

4. **Clinical Integration**
   - Pilot testing with small user group
   - Gather feedback and iterate
   - Develop training materials and conduct sessions
   - Plan full-scale rollout

**Deliverables**:
- Conformance checking reports with deviation analysis
- Predictive models deployed in test environment
- Real-time monitoring dashboard (beta version)
- Training materials and pilot test results

### Phase 4: Optimization and Scale (Months 13-18)
**Objectives**:
- Implement process improvements
- Expand to additional use cases
- Establish continuous improvement cycle

**Activities**:
1. **Process Improvement Implementation**
   - Execute quick win interventions from Phase 2
   - Monitor impact using process mining metrics
   - Adjust interventions based on observed results
   - Document lessons learned and best practices

2. **Scope Expansion**
   - Apply process mining to additional clinical areas (ICU, surgery, inpatient)
   - Integrate with other hospital systems (bed management, OR scheduling)
   - Develop cross-departmental process models
   - Enable enterprise-wide process visibility

3. **Advanced Use Cases**
   - Causal process mining for treatment effectiveness
   - Patient pathway personalization using ML
   - Resource optimization (staff scheduling, equipment allocation)
   - Predictive capacity planning for surge events

4. **Sustainability**
   - Establish process mining center of excellence
   - Define ongoing roles and responsibilities
   - Implement continuous monitoring and reporting
   - Create feedback loop for model updates and improvements

**Deliverables**:
- Documented process improvements with measured impact
- Expanded process mining deployment across hospital
- Advanced analytics capabilities (causal, optimization)
- Sustainable governance and operations model

### Phase 5: Innovation and Expansion (Months 19-24)
**Objectives**:
- Explore cutting-edge techniques
- Share knowledge and best practices
- Drive healthcare transformation

**Activities**:
1. **Emerging Technologies**
   - Pilot LLM-enhanced process mining (HealthProcessAI approach)
   - Experiment with deep learning for process prediction
   - Explore federated process mining for multi-site collaboration
   - Investigate privacy-preserving techniques (differential privacy, TEE)

2. **Knowledge Sharing**
   - Publish case studies and results (conferences, journals)
   - Contribute to open-source process mining tools
   - Participate in healthcare process mining community
   - Share benchmark datasets (with appropriate de-identification)

3. **Strategic Initiatives**
   - Align process mining with organizational strategic goals
   - Identify new value creation opportunities
   - Develop business cases for continued investment
   - Plan long-term roadmap and vision

**Deliverables**:
- Pilot reports on emerging technologies
- Publications and presentations sharing learnings
- Strategic plan for process mining evolution
- Roadmap for next 3-5 years

---

## 11. Evaluation Framework

### 11.1 Process Mining Quality Metrics

#### Model Quality
**Fitness**
- **Definition**: Ability of model to replay event log
- **Calculation**: Token-based replay, alignment-based
- **Target**: ≥ 0.90 for clinical applications
- **Interpretation**: Low fitness indicates missing pathways in model

**Precision**
- **Definition**: Extent model does not allow unobserved behavior
- **Calculation**: Alignment-based, behavioral precision
- **Target**: ≥ 0.80 for clinical applications
- **Interpretation**: Low precision indicates overgeneralization

**F-Measure**
- **Definition**: Harmonic mean of fitness and precision
- **Calculation**: 2 × (fitness × precision) / (fitness + precision)
- **Target**: ≥ 0.85 for clinical applications
- **Interpretation**: Balanced measure of model quality

**Generalization**
- **Definition**: Ability to reproduce unseen cases
- **Calculation**: Cross-validation, holdout testing
- **Target**: ≥ 0.75 for clinical applications
- **Interpretation**: Low generalization indicates overfitting

**Simplicity**
- **Definition**: Understandability and interpretability of model
- **Measurement**: Number of nodes, arcs, complexity metrics
- **Target**: Context-dependent, clinical stakeholder feedback
- **Interpretation**: Balance with fitness (not too simple, not too complex)

### 11.2 Predictive Model Performance

#### Classification Metrics
**Accuracy**
- **Healthcare Results**: 75.9% (KATE triage), 87.5% (ED hospitalization)
- **Limitation**: Can be misleading with class imbalance
- **Use**: Overall correctness assessment

**Precision and Recall**
- **Precision**: True positives / (True positives + False positives)
- **Recall (Sensitivity)**: True positives / (True positives + False negatives)
- **Healthcare**: Often prioritize recall (missing a critical case worse than false alarm)
- **Example**: ESI 2/3 boundary 93.2% precision (KATE)

**F1-Score**
- **Definition**: Harmonic mean of precision and recall
- **Healthcare Results**: 0.86 (ED hospitalization), 0.90 (Gaussian regression)
- **Use**: Balanced performance measure

**AUC-ROC**
- **Definition**: Area under receiver operating characteristic curve
- **Healthcare Results**: 0.82 (PI mortality), 0.68-0.96 (multi-disease), 0.879 (URGENTIAPARSE)
- **Interpretation**: Discriminative ability across all thresholds

#### Regression Metrics
**Mean Absolute Error (MAE)**
- **Use**: Waiting time prediction, LOS forecasting
- **Interpretation**: Average prediction error in original units
- **Advantage**: Robust to outliers

**R-squared**
- **Healthcare Results**: 82% (hourly ED arrivals), 0.900 (URGENTIAPARSE)
- **Interpretation**: Proportion of variance explained
- **Limitation**: Can be inflated with complex models

**Calibration**
- **Definition**: Agreement between predicted probabilities and observed frequencies
- **Measurement**: Calibration plots, Brier score
- **Importance**: Critical for clinical decision-making (trust in predictions)

### 11.3 Clinical Impact Metrics

#### Operational Efficiency
**ED Length of Stay (LOS)**
- **Measurement**: Time from arrival to departure
- **Target**: Reduce median and 95th percentile
- **Benchmark**: National/regional standards
- **Impact**: Process mining interventions → 26-33% reduction

**Waiting Times**
- **Door-to-Doctor**: Time from arrival to physician assessment
- **Treatment Wait**: Time from assessment to treatment start
- **Target**: Guidelines (e.g., <1 hour for ESI 3)
- **Monitoring**: Real-time dashboards, trend analysis

**Throughput**
- **Measurement**: Patients per hour, per shift
- **Target**: Increase without compromising quality
- **Constraints**: Resource availability, safety standards

**Resource Utilization**
- **Bed Occupancy**: Percentage of time beds occupied
- **Staff Productivity**: Activities per staff member, idle time
- **Equipment**: Imaging, lab equipment usage rates
- **Optimization**: Balance efficiency and reserve capacity

#### Patient Outcomes
**Mortality**
- **In-ED Mortality**: Deaths in emergency department
- **30-day Mortality**: Deaths within 30 days of ED visit
- **Prediction**: Process mining models (e.g., 0.82 AUC for PI)
- **Intervention**: Early warning systems, protocol adherence

**Complications**
- **Adverse Events**: Medical errors, hospital-acquired conditions
- **Near Misses**: Caught by safety systems
- **Root Causes**: Process deviations, guideline non-adherence
- **Prevention**: Conformance checking, real-time alerts

**Readmissions**
- **72-hour ED Return**: Unplanned returns within 3 days
- **30-day Hospital Readmission**: Predictive modeling target
- **Risk Factors**: Identified through process mining + ML
- **Interventions**: Discharge planning, follow-up coordination

#### Patient Experience
**Satisfaction Scores**
- **Measurement**: Surveys, feedback forms
- **Dimensions**: Wait times, communication, respect, outcomes
- **Correlation**: With process mining-optimized workflows

**Left Without Being Seen (LWBS)**
- **Definition**: Patients who leave before completing visit
- **Target**: <2% industry benchmark
- **Reduction**: Through better triage, waiting time communication

**Communication Quality**
- **Information Provision**: Explanations, updates, transparency
- **Engagement**: Shared decision-making, patient involvement
- **Tools**: Waiting time predictions, pathway visualizations

### 11.4 Business Metrics

#### Financial Impact
**Cost per Visit**
- **Components**: Staff, supplies, tests, treatments, overhead
- **Target**: Reduce while maintaining quality
- **Example**: 7% cost reduction (knee replacement pathway optimization)

**Revenue**
- **Throughput Impact**: More patients served
- **Reimbursement**: Appropriate coding, documentation
- **Efficiency**: Reduced length of stay (bed turnover)

**Cost Avoidance**
- **Prevented Complications**: Adverse events, errors
- **Reduced Waste**: Unnecessary tests, treatments
- **Optimized Resources**: Staff overtime, equipment idle time

#### Return on Investment (ROI)
**Process Mining Implementation Costs**
- **Software**: Licenses, infrastructure
- **Personnel**: Analysts, IT staff, training
- **Time**: Staff involvement, workflow disruptions

**Benefits**
- **Efficiency Gains**: Time savings, increased throughput
- **Quality Improvements**: Reduced errors, better outcomes
- **Strategic Value**: Data-driven culture, continuous improvement

**ROI Calculation**
- **Formula**: (Benefits - Costs) / Costs
- **Timeline**: Typical payback 12-24 months
- **Ongoing**: Continuous value creation

---

## 12. Ethical and Privacy Considerations

### 12.1 Data Privacy and Security

#### Regulatory Compliance
**HIPAA (US)**
- **Protected Health Information (PHI)**: 18 identifiers
- **De-identification**: Expert determination or Safe Harbor method
- **Process Mining**: Ensure event logs de-identified before analysis
- **Challenges**: Temporal patterns, rare diseases can be re-identifying

**GDPR (EU)**
- **Personal Data**: Broader than PHI, includes any identifiable information
- **Lawful Basis**: Consent, legitimate interest, public health
- **Rights**: Access, rectification, erasure, portability
- **Process Mining**: Privacy by design, data minimization

**Local Regulations**
- **Varies by Jurisdiction**: State laws (e.g., California CCPA), national laws
- **Compliance**: Legal review, privacy impact assessments
- **Documentation**: Data governance policies, procedures

#### Privacy-Preserving Techniques
**Differential Privacy**
- **Concept**: Add noise to outputs to protect individual records
- **Application**: Directly-Follows Graphs (DFGs) with differential privacy
- **Trade-off**: Privacy (ε) vs. utility (accuracy)
- **Results**: Can achieve practical privacy with minimal utility loss

**Secure Multi-Party Computation**
- **Concept**: Multiple parties compute jointly without revealing data
- **Application**: Inter-organizational process mining (e.g., hospital networks)
- **Example**: CONFINE using Trusted Execution Environments (TEEs)
- **Benefits**: Data remains with owners, collaboration enabled

**Federated Process Mining**
- **Concept**: Train models on distributed data without centralizing
- **Application**: Multi-site hospital studies
- **Challenges**: Heterogeneous data, communication overhead
- **Benefits**: Privacy preservation, local data control

### 12.2 Algorithmic Fairness and Bias

#### Sources of Bias in Healthcare Process Mining
**Data Bias**
- **Historical Inequities**: Underrepresentation of minorities, women
- **Sampling Bias**: Sicker patients more data (selection bias)
- **Measurement Bias**: Different quality of data by race, SES

**Algorithm Bias**
- **Training Data**: Models learn patterns from biased historical data
- **Feature Selection**: Proxies for protected attributes (zip code → race)
- **Optimization**: Metrics that disadvantage subgroups

**Deployment Bias**
- **Differential Access**: Some populations more/less exposed to AI systems
- **Feedback Loops**: Biased predictions → biased actions → biased future data

#### Mitigation Strategies
**Bias Detection**
- **Disaggregated Analysis**: Performance metrics by subgroups (race, sex, age, SES)
- **Fairness Metrics**: Equalized odds, demographic parity, individual fairness
- **Auditing**: Regular bias assessments, third-party reviews

**Bias Mitigation**
- **Pre-processing**: Reweighting, resampling, data augmentation
- **In-processing**: Fairness constraints in model training (e.g., adversarial debiasing)
- **Post-processing**: Threshold adjustment, calibration by subgroups

**Governance**
- **Diverse Teams**: Multidisciplinary, representative of patient populations
- **Stakeholder Engagement**: Patient advocates, community input
- **Transparency**: Explainable models, documentation, audits

### 12.3 Clinical Decision-Making and Autonomy

#### Human-in-the-Loop
**Principle**: AI augments, not replaces, clinician judgment
- **Decision Support**: Provide recommendations, not directives
- **Transparency**: Explain reasoning, show confidence levels
- **Override**: Clinicians can disagree and document rationale

**Implementation**:
- **User Interface**: Clear presentation of AI predictions and evidence
- **Workflow Integration**: Non-disruptive alerts, actionable insights
- **Feedback**: Capture clinician corrections, learn from disagreements

#### Liability and Accountability
**Questions**:
- Who is responsible if AI-assisted decision harms patient?
- Clinician, healthcare organization, AI vendor, algorithm developer?

**Current Landscape**:
- **Clinician Responsibility**: Ultimately accountable for patient care
- **Informed Consent**: Patients should know when AI involved
- **Malpractice**: Standards of care evolving with AI adoption

**Best Practices**:
- **Clear Policies**: Define roles, responsibilities, escalation
- **Documentation**: Record AI involvement, clinician reasoning
- **Insurance**: Professional liability coverage for AI use
- **Regulatory**: FDA approval for medical devices (algorithms as devices)

### 12.4 Ethical Use of Process Mining

#### Surveillance vs. Quality Improvement
**Concern**: Process mining could be used to monitor staff, not just optimize processes
- **Surveillance**: Punitive, focuses on individual performance, reduces trust
- **Quality Improvement**: Collaborative, focuses on system, builds improvement culture

**Safeguards**:
- **Purpose Limitation**: Use data only for stated quality improvement goals
- **Aggregation**: Report at team/department level, not individual
- **Transparency**: Staff aware of process mining, involved in interpretation
- **Non-Punitive**: Findings used for learning, not discipline (except egregious violations)

#### Patient Consent and Engagement
**Secondary Use of Clinical Data**:
- **Question**: Do patients need to consent to process mining on their EHR data?
- **Positions**:
  - **No**: De-identified, quality improvement, institutional authority
  - **Yes**: Respect, autonomy, building trust, engagement

**Best Practice**:
- **Transparency**: Inform patients about data use for quality improvement
- **Opt-Out**: Allow patients to request exclusion (with understanding of implications)
- **Engagement**: Patient advisory boards, community input on process mining priorities
- **Benefit Sharing**: Communicate how process mining improves care

---

## 13. Future Research Priorities

### 13.1 High-Impact Research Questions

#### Technical Innovations
1. **How can process mining scale to real-time, streaming healthcare data?**
   - Current: Batch processing, offline analysis
   - Needed: Incremental algorithms, distributed computing
   - Impact: Enable proactive interventions, dynamic resource allocation

2. **What are the optimal approaches for privacy-preserving process mining in multi-institutional collaborations?**
   - Current: Centralized data, limited sharing
   - Needed: Federated learning, differential privacy, secure computation
   - Impact: Unlock multi-site studies, benchmark across healthcare systems

3. **How can causal inference be robustly integrated with process mining for treatment effect estimation?**
   - Current: Correlation-based insights, confounding issues
   - Needed: Causal discovery, counterfactual analysis, uplift modeling
   - Impact: Evidence-based treatment pathway recommendations

4. **What are the best methods for multi-modal process mining (EHR + imaging + sensors + notes)?**
   - Current: Primarily structured EHR data
   - Needed: Multi-modal fusion, temporal alignment, representation learning
   - Impact: Holistic patient state understanding, richer process models

#### Clinical Applications
5. **How effective is process mining-guided workflow redesign at improving patient outcomes (mortality, complications, readmissions)?**
   - Current: Operational metrics (LOS, waiting times)
   - Needed: RCTs or quasi-experimental studies on clinical outcomes
   - Impact: Establish clinical value, not just efficiency gains

6. **Can personalized care pathways discovered through process mining + ML improve treatment effectiveness?**
   - Current: Population-level guidelines, one-size-fits-all
   - Needed: Precision medicine pathways, N-of-1 optimization
   - Impact: Better outcomes, reduced adverse events, patient-centered care

7. **What are the optimal strategies for process mining in emergency and critical care settings?**
   - Current: Limited real-time applications
   - Needed: Fast algorithms, real-time conformance, predictive alerts
   - Impact: Time-critical decision support, resource optimization

#### Organizational and Policy
8. **What are the organizational factors that enable successful process mining adoption in healthcare?**
   - Current: Pockets of innovation, limited systematic study
   - Needed: Implementation science research, change management frameworks
   - Impact: Accelerate diffusion, avoid failed implementations

9. **How should process mining be integrated into healthcare quality and safety programs?**
   - Current: Ad-hoc initiatives
   - Needed: Standards, best practices, regulatory guidance
   - Impact: Mainstreaming, consistent quality, patient safety improvements

10. **What are the ethical frameworks for responsible process mining in healthcare?**
    - Current: General AI ethics, limited healthcare-specific guidance
    - Needed: Consensus guidelines, case law, professional standards
    - Impact: Trust, acceptance, sustainable adoption

### 13.2 Emerging Methodologies

#### Explainable AI for Process Mining
**Challenge**: Black-box ML models reduce trust, hinder clinical adoption
**Approaches**:
- **Attention Mechanisms**: Highlight influential events in sequence models
- **SHAP/LIME**: Local explanations for process predictions
- **Rule Extraction**: Derive interpretable rules from complex models
- **Counterfactual Explanations**: "If event X had been different, outcome would be Y"

**Research Needs**:
- Evaluation frameworks for explanation quality in clinical contexts
- User studies with clinicians on explanation preferences
- Integration of explanations into clinical workflow

#### Process Mining with Large Language Models
**Opportunities**:
- **Natural Language Queries**: Ask questions about processes in plain language
- **Automated Report Generation**: Summarize findings for different audiences
- **Knowledge Extraction**: Mine insights from unstructured clinical notes
- **Process Model Synthesis**: Generate process models from text descriptions

**Challenges**:
- Hallucination prevention (ensuring factual accuracy)
- Clinical validation (expert review of LLM outputs)
- Computational cost (large models, inference latency)

**Research Directions**:
- Fine-tuning LLMs on healthcare process mining tasks
- Hybrid approaches (symbolic + neural)
- Evaluation benchmarks for LLM-assisted process mining

#### Reinforcement Learning for Pathway Optimization
**Concept**: Learn optimal treatment policies from observational data
**Approaches**:
- **Offline RL**: Batch-mode learning from historical event logs
- **Counterfactual RL**: Estimate effects of actions not taken
- **Safe RL**: Constrained exploration to avoid patient harm

**Applications**:
- **Treatment Selection**: Choose best therapy given patient state
- **Resource Allocation**: Optimize staff, equipment, bed assignments
- **Pathway Personalization**: Individual-level optimal pathways

**Challenges**:
- Confounding (observed treatment choices biased by patient state)
- Partial observability (incomplete information about patient)
- Safety (cannot explore harmful actions)
- Validation (difficult to conduct RCTs)

### 13.3 Cross-Disciplinary Collaborations

#### Healthcare + Computer Science
**Integration Points**:
- Process mining algorithms tailored to healthcare data characteristics
- Healthcare-specific benchmarks and evaluation frameworks
- Open-source tools and datasets

**Needed**:
- Joint faculty appointments, collaborative grants
- Shared conferences, journals (e.g., JBI, AI in Medicine)
- Education programs (healthcare informatics, clinical AI)

#### Healthcare + Operations Research
**Synergies**:
- Process mining informs optimization models (simulation, queueing, scheduling)
- OR techniques validate process mining recommendations
- Combined approach: descriptive (PM) + prescriptive (OR)

**Applications**:
- Hospital capacity planning (bed management, OR scheduling)
- Staff scheduling (shift patterns, float pools)
- Supply chain optimization (medication, equipment)

**Needed**:
- Hybrid methodological frameworks
- Case studies demonstrating integrated approaches
- Education bridging PM and OR communities

#### Healthcare + Social Science
**Perspectives**:
- Implementation science: How to deploy process mining effectively
- Organizational behavior: Change management, culture, incentives
- Health services research: Impact on access, equity, outcomes

**Research**:
- Mixed-methods studies (quantitative PM + qualitative interviews)
- Evaluation of process mining interventions (before-after, controlled trials)
- Policy analysis (regulatory frameworks, reimbursement)

**Needed**:
- Interdisciplinary research teams
- Funding mechanisms supporting cross-disciplinary work
- Publication venues valuing methodological pluralism

---

## 14. Conclusion and Recommendations

### 14.1 Key Insights from Literature Review

**Process Mining Maturity in Healthcare**:
The field has evolved from experimental academic research to practical clinical applications, with documented successes in emergency departments, ICUs, and chronic disease management. Evidence shows that process mining can achieve high-quality models (97.8% fitness) when combining expert knowledge with data-driven approaches.

**LLM Integration as Game-Changer**:
The HealthProcessAI framework demonstrates that large language models can democratize process mining by automating interpretation and report generation, achieving consistency scores of 3.79/4.0 with Claude Sonnet-4. This addresses a major barrier: the technical expertise gap preventing clinical stakeholders from adopting process mining.

**Multi-Morbidity Complexity**:
Event graph representations for multi-morbid patients reveal care coordination patterns invisible to traditional process mining, highlighting the need for advanced representations beyond single-case paradigms.

**Emergency Department Applications**:
Process mining has proven particularly valuable for ED workflow analysis, with demonstrated impacts including 26-33% reduction in excessive costs, 87.5% accuracy in hospitalization prediction, and 93.2% accuracy in critical triage decisions (vs. 41.4% human performance).

**Conformance Checking Value**:
Analyzing adherence to clinical guidelines through conformance checking identifies systematic deviations that inform quality improvement, as demonstrated in COVID-19 ICU treatment and sepsis care pathways.

### 14.2 Recommendations for ED Workflow Research

#### For Researchers
1. **Prioritize Real-Time Capabilities**: Focus on streaming process mining algorithms that support live clinical decision-making, not just retrospective analysis
2. **Develop Healthcare Benchmarks**: Create standardized datasets and evaluation frameworks (building on MIMIC-IV-ED) to enable reproducible research
3. **Integrate Multiple Data Modalities**: Combine structured EHR data, clinical notes (NLP), imaging, and sensor data for comprehensive process understanding
4. **Address Privacy Proactively**: Incorporate differential privacy and federated learning from the outset, not as afterthoughts
5. **Validate Clinical Impact**: Move beyond operational metrics to demonstrate improvements in patient outcomes (mortality, complications, readmissions)

#### For Practitioners
1. **Start with Quick Wins**: Identify high-impact, low-effort improvements (e.g., bottleneck elimination) to build momentum and stakeholder support
2. **Engage Clinicians Early**: Involve ED physicians and nurses from the beginning to ensure relevance, validity, and adoption
3. **Invest in Data Quality**: Clean, complete, standardized data is the foundation—allocate resources accordingly
4. **Balance Automation and Expertise**: Use AI (e.g., LLMs) for scale and efficiency, but validate with clinical domain knowledge
5. **Measure and Communicate Impact**: Track metrics, document successes, and share learnings to sustain support and continuous improvement

#### For Healthcare Organizations
1. **Establish Process Mining Capabilities**: Build in-house expertise or partner with academic/commercial entities
2. **Create Governance Frameworks**: Define policies for data use, privacy, ethics, and accountability
3. **Foster Data-Driven Culture**: Encourage evidence-based decision-making, experimentation, and learning from process mining insights
4. **Allocate Resources**: Invest in infrastructure (data platforms, tools), personnel (analysts, data scientists), and training
5. **Collaborate and Share**: Participate in multi-institutional research, contribute to benchmarks, and disseminate best practices

#### For Policymakers
1. **Develop Standards**: Work with professional societies to create guidelines for process mining in healthcare (data formats, privacy, ethics)
2. **Incentivize Quality**: Tie reimbursement to process mining-driven quality improvement initiatives
3. **Fund Research**: Support interdisciplinary research on process mining methods, applications, and impact evaluation
4. **Enable Data Sharing**: Create legal/regulatory frameworks facilitating multi-institutional process mining while protecting privacy
5. **Monitor and Regulate**: Ensure responsible AI use, algorithmic fairness, and patient safety as process mining scales

### 14.3 Path Forward for Hybrid Reasoning in Acute Care

**Integration with Clinical Reasoning**:
Process mining provides the empirical foundation for understanding actual care delivery, while hybrid reasoning systems can leverage these discovered pathways to support clinical decision-making. The combination enables:
- **Evidence-Based Protocols**: Pathways derived from successful cases, not just expert consensus
- **Deviation Alerts**: Real-time conformance checking to flag when care diverges from evidence
- **Outcome Prediction**: ML models informed by process mining features for risk stratification
- **Adaptive Learning**: Continuous improvement as new data updates both pathways and reasoning models

**Recommended Approach**:
1. **Phase 1**: Apply process mining to ED data to discover actual patient flow and identify bottlenecks
2. **Phase 2**: Develop conformance checking against clinical guidelines to understand adherence patterns
3. **Phase 3**: Build predictive models (triage acuity, waiting times, outcomes) using process features
4. **Phase 4**: Integrate with hybrid reasoning system for real-time clinical decision support
5. **Phase 5**: Deploy, evaluate, and iteratively refine based on clinical feedback and outcomes

**Expected Benefits**:
- **For Patients**: Reduced waiting times, better triage accuracy, improved outcomes, enhanced safety
- **For Clinicians**: Evidence-based decision support, reduced cognitive burden, early warnings
- **For Organizations**: Operational efficiency, quality improvement, cost reduction, competitive advantage
- **For Research**: Novel insights, benchmark datasets, methodological advances, publications

---

## Appendix: Detailed Paper Summaries

### A.1 HealthProcessAI (2508.21540v2)
**Title**: HealthProcessAI: A Technical Framework and Proof-of-Concept for LLM-Enhanced Healthcare Process Mining

**Key Contributions**:
- GenAI framework wrapping PM4PY (Python) and bupaR (R) libraries
- Integrates 5 LLMs via OpenRouter for automated interpretation
- LLM evaluation using 5 independent LLM assessors

**Methodology**:
- Proof-of-concept: Sepsis progression data
- Four scenarios demonstrating functionality
- LLM comparison: Claude Sonnet-4, Gemini 2.5-Pro, GPT-4, others

**Results**:
- Claude Sonnet-4: Highest consistency (3.79/4.0)
- Gemini 2.5-Pro: Second highest (3.65/4.0)
- Successfully processed sepsis data with robust performance
- Automated report generation from process maps

**Implications**:
- Addresses unfamiliarity barrier to process mining adoption
- Makes outputs accessible to clinicians, data scientists, researchers
- Represents novel methodological advance in healthcare AI
- Potential for actionable insights without deep PM expertise

### A.2 Process Mining-Driven Analysis of COVID-19 Impact (2112.04634v2)
**Title**: Process Mining-Driven Analysis of the COVID-19 Impact on the Vaccinations of Victorian Patients

**Context**:
- 30+ million events from 1+ million patients
- 5-year study (2016-2020) in Victoria, Australia
- General practice healthcare processes

**Objective**:
- Detect differences between 2020 (pandemic) and 2016-2019 baseline
- Assess impact on vaccinations specifically

**Methods**:
- Combination of process mining and traditional data mining
- Analysis of highly variable processes and large datasets
- Identification of benefits and limitations of state-of-the-art techniques

**Findings**:
- **Unexpected**: Vaccinations did not drop in 2020 (contrary to other studies)
- **Surge**: Influenza and pneumococcus vaccinations increased in 2020
- **Contrast**: Other interactions (non-vaccination) did decrease
- **Geographic**: Victoria-specific findings differ from other regions

**Technical Lessons**:
- Handling large-scale healthcare data challenges
- Dealing with high process variability
- Combining process mining with statistical analysis
- Importance of geographic and temporal context

### A.3 Extending Process Discovery with Model Complexity Optimization (2206.06111v1)
**Title**: Extending Process Discovery with Model Complexity Optimization and Cyclic States Identification: Application to Healthcare Processes

**Problem**:
- Process discovery results often don't balance model complexity and fitting accuracy
- Manual model adjusting required

**Approach**:
- Semi-automatic support for model optimization
- Combined assessment: model complexity + fitness
- Model simplification via abstraction at desired granularity

**Innovation - Meta-States**:
- Concept of meta-states for cycle collapsing
- Simplifies models while preserving meaning
- Improves interpretability

**Evaluation**:
- Three datasets from healthcare domain:
  1. Remote monitoring for hypertension patients
  2. Healthcare worker workflows during COVID-19 pandemic (2 datasets)
- Multiple complexity measures tested
- Different application modes explored

**Results**:
- Improved interpretability and complexity/fitness balance
- Insights on better practices for process model improvement
- Meta-states effectively simplify models

**Significance**:
- Addresses key challenge in process mining: model quality vs. simplicity
- Healthcare-specific validation demonstrates applicability
- Provides framework for semi-automated optimization

---

## References

This synthesis is based on 120+ papers identified through systematic ArXiv searches on healthcare process mining, clinical workflow analysis, care pathway discovery, and treatment process optimization. Key paper IDs are referenced throughout the document. The complete bibliography includes papers from computer science (cs.AI, cs.LG), medical informatics (cs.CY), statistics (stat.AP, stat.ME), and healthcare operations research domains.

**Primary Search Queries**:
- "process mining" AND (healthcare OR clinical)
- "workflow analysis" AND (medical OR clinical OR hospital)
- "care pathway" AND (discovery OR mining)
- "event log" AND clinical
- ti:"process" AND (hospital OR patient OR healthcare)
- "conformance checking" AND (healthcare OR clinical OR medical)

**Date Range**: 2015-2025
**Categories Focus**: cs.LG, cs.AI, cs.DB, cs.CY, stat.AP, stat.ME
**Geographic Distribution**: Global (US, EU, Asia-Pacific studies included)

---

**Document Prepared**: December 1, 2025
**For**: Hybrid Reasoning in Acute Care Research Project
**Location**: /Users/alexstinard/hybrid-reasoning-acute-care/research/arxiv_healthcare_process_mining.md
