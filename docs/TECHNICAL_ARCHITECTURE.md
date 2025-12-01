# TECHNICAL ARCHITECTURE AND PROTOTYPE
## Hybrid Reasoning for Acute Care: Implementation Roadmap

---

## 1. SYSTEM ARCHITECTURE

### 1.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CLINICIAN INTERFACE LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐   │
│  │ Risk Scores  │  │ Explanation  │  │ Timeline     │  │ What-If     │   │
│  │ & Alerts     │  │ Visualizer   │  │ View         │  │ Scenarios   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXPLANATION GENERATION MODULE                          │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │  • Reasoning Chain Extractor                                       │   │
│  │  • Counterfactual Generator (Path-based + GNN Explainer)          │   │
│  │  • Natural Language Renderer (Template + LLM)                     │   │
│  │  • Confidence Calibrator                                          │   │
│  └────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   NEURO-SYMBOLIC REASONING ENGINE                           │
│  ┌──────────────────────────┐  ┌──────────────────────────────────────┐  │
│  │   NEURAL COMPONENT       │  │   SYMBOLIC COMPONENT                 │  │
│  │  ┌────────────────────┐  │  │  ┌────────────────────────────────┐ │  │
│  │  │ Temporal GNN       │  │  │  │ Logical Neural Network (LNN)   │ │  │
│  │  │  - RGCN layers     │  │  │  │  - Clinical guideline rules    │ │  │
│  │  │  - Temporal attn   │  │  │  │  - Temporal logic constraints  │ │  │
│  │  │  - Multi-hop agg   │  │  │  │  - SNOMED hierarchy axioms     │ │  │
│  │  └────────────────────┘  │  │  └────────────────────────────────┘ │  │
│  │                          │  │                                      │  │
│  │  ┌────────────────────┐  │  │  ┌────────────────────────────────┐ │  │
│  │  │ Embedding Layer    │◄─┼──┼──► Soft Constraint Layer          │ │  │
│  │  │  - Node embeddings │  │  │  │  - Weighted rule satisfaction  │ │  │
│  │  │  - Edge embeddings │  │  │  │  - Penalty gradients           │ │  │
│  │  └────────────────────┘  │  │  └────────────────────────────────┘ │  │
│  └──────────────────────────┘  └──────────────────────────────────────┘  │
│                                                                             │
│  Joint Loss: L_total = α·L_prediction + β·L_constraint + γ·L_temporal     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   TEMPORAL KNOWLEDGE GRAPH CONSTRUCTION                     │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │  Graph Schema:                                                      │   │
│  │  • Nodes: {Patient, Event, Diagnosis, Medication, Lab, Vital}     │   │
│  │  • Edges: {PRECEDES, CAUSES, TREATS, INDICATES, CONTRADICTS}      │   │
│  │  • Temporal: [start_time, end_time, duration, interval_relation]  │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐    │
│  │ Entity       │  │ Relation     │  │ Temporal     │  │ Graph     │    │
│  │ Extraction   │→ │ Extraction   │→ │ Alignment    │→ │ Builder   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └───────────┘    │
│       │                  │                  │                 │           │
│       ▼                  ▼                  ▼                 ▼           │
│  BioClinical       Pattern Mining    Allen's Interval    Neo4j/DGL       │
│  BERT/PubMed       + Rule-based      Algebra Reasoner    Graph Store     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐    │
│  │ FHIR/HL7     │  │ MIMIC CSV    │  │ Clinical     │  │ DICOM     │    │
│  │ Parser       │  │ Loader       │  │ Notes NLP    │  │ Extractor │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └───────────┘    │
│         │                 │                 │                 │           │
│         └─────────────────┴─────────────────┴─────────────────┘           │
│                                    │                                       │
│                                    ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │             UNIFIED DATA MODEL (FHIR-based)                        │   │
│  │  • Observations (vitals, labs) + timestamps                        │   │
│  │  • Conditions (diagnoses) + onset/resolution times                 │   │
│  │  • MedicationAdministration + effective periods                    │   │
│  │  • Procedures + performed times                                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

                    Storage: PostgreSQL (raw) + Neo4j (graph)
                    Compute: PyTorch + DGL + CUDA
                    Deployment: Docker containers + FastAPI REST
```

### 1.2 Architecture Rationale

**Layered Design Principles:**
1. **Separation of Concerns**: Each layer has a single responsibility, enabling parallel development and testing
2. **Modularity**: Components can be swapped (e.g., Neo4j → DGL) without affecting downstream layers
3. **Explainability-by-Design**: Explanation module has direct access to reasoning engine internals
4. **Clinical Workflow Integration**: Interface layer designed for ED clinical decision points

**Key Design Decisions:**
- **Hybrid Graph Storage**: Neo4j for OLTP/queries, DGL for OLAP/training
- **Dual-Path Reasoning**: Neural for pattern recognition, symbolic for constraint satisfaction
- **Temporal-First**: Time is a first-class citizen, not a feature
- **FHIR Compliance**: Enables real-world deployment at partner hospitals

---

## 2. CORE DATA STRUCTURES

### 2.1 Temporal Knowledge Graph Schema

```python
# Node Types and Attributes
class NodeSchema:
    """Temporal KG node definitions"""

    PATIENT = {
        "id": str,              # Unique patient identifier
        "demographics": {
            "age": int,
            "sex": str,
            "ethnicity": str
        },
        "admission_time": datetime,
        "embeddings": np.ndarray  # Learned patient representation
    }

    EVENT = {
        "id": str,
        "type": Enum["vital", "lab", "medication", "procedure", "diagnosis"],
        "timestamp": datetime,
        "value": Union[float, str],
        "confidence": float,     # For extracted events
        "source": str,           # "structured" | "nlp" | "inferred"

        # Temporal extent (Allen's intervals)
        "start_time": datetime,
        "end_time": datetime,
        "duration": timedelta,

        # Clinical semantics
        "snomed_code": str,
        "severity": Enum["mild", "moderate", "severe", "critical"],
        "embeddings": np.ndarray
    }

    DIAGNOSIS = {
        "id": str,
        "icd10_code": str,
        "snomed_code": str,
        "description": str,
        "onset_time": datetime,
        "resolution_time": Optional[datetime],
        "certainty": Enum["confirmed", "suspected", "ruled_out"],
        "embeddings": np.ndarray
    }

    MEDICATION = {
        "id": str,
        "rxnorm_code": str,
        "name": str,
        "dose": str,
        "route": str,
        "start_time": datetime,
        "end_time": Optional[datetime],
        "indication": str,
        "embeddings": np.ndarray
    }

    LAB_RESULT = {
        "id": str,
        "loinc_code": str,
        "name": str,
        "value": float,
        "unit": str,
        "reference_range": Tuple[float, float],
        "is_abnormal": bool,
        "timestamp": datetime,
        "embeddings": np.ndarray
    }

    VITAL_SIGN = {
        "id": str,
        "type": Enum["hr", "bp", "rr", "temp", "spo2", "gcs"],
        "value": float,
        "timestamp": datetime,
        "is_critical": bool,
        "embeddings": np.ndarray
    }
```

```python
# Edge Types and Temporal Relations
class EdgeSchema:
    """Temporal KG edge definitions with Allen's interval relations"""

    # Temporal edges (derived from timestamps)
    PRECEDES = {
        "relation_type": "temporal",
        "source": str,  # Event ID
        "target": str,  # Event ID
        "time_gap": timedelta,
        "allen_relation": Enum[
            "before",        # e1 --- e2
            "meets",         # e1--- e2
            "overlaps",      # e1--[--e2--]
            "during",        # e1 [--e1--] e2
            "starts",        # [e1-- ]e2---
            "finishes",      # ---e1 --e1]e2
            "equals"         # [--e1==e2--]
        ],
        "weight": float  # For GNN message passing
    }

    # Causal edges (learned or rule-based)
    CAUSES = {
        "relation_type": "causal",
        "source": str,
        "target": str,
        "confidence": float,  # 0-1, from causal discovery or rules
        "evidence": List[str],  # Supporting event IDs
        "mechanism": str,  # e.g., "hypovolemia → hypotension"
    }

    # Treatment edges
    TREATS = {
        "relation_type": "treatment",
        "source": str,  # Medication/Procedure
        "target": str,  # Diagnosis/Symptom
        "effectiveness": Optional[float],  # If outcome known
        "guideline_supported": bool,
        "contraindications": List[str]
    }

    # Diagnostic edges
    INDICATES = {
        "relation_type": "diagnostic",
        "source": str,  # Lab/Vital/Symptom
        "target": str,  # Diagnosis
        "sensitivity": float,
        "specificity": float,
        "ppv": float,  # Positive predictive value
    }

    # Constraint violation edges
    CONTRADICTS = {
        "relation_type": "constraint",
        "source": str,
        "target": str,
        "violation_type": Enum[
            "drug_interaction",
            "temporal_impossibility",
            "guideline_violation"
        ],
        "severity": Enum["warning", "error", "critical"]
    }
```

### 2.2 Clinical Event Representation

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import numpy as np

@dataclass
class ClinicalEvent:
    """Unified representation for all clinical events"""

    # Core identifiers
    event_id: str
    patient_id: str
    event_type: str  # vital, lab, med, diagnosis, procedure

    # Temporal information
    start_time: datetime
    end_time: Optional[datetime] = None

    # Clinical coding
    codes: Dict[str, str] = None  # {"icd10": "A41.9", "snomed": "91302008"}

    # Event value
    value: Optional[float] = None  # For numerical measurements
    value_text: Optional[str] = None  # For categorical/text
    unit: Optional[str] = None

    # Metadata
    source: str = "structured"  # structured | nlp | inferred
    confidence: float = 1.0

    # Learned representations
    embedding: Optional[np.ndarray] = None

    def duration(self) -> Optional[timedelta]:
        """Calculate event duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return None

    def to_interval(self) -> 'AllenInterval':
        """Convert to Allen's interval representation"""
        return AllenInterval(
            start=self.start_time,
            end=self.end_time or self.start_time
        )
```

### 2.3 Allen's Interval Algebra Implementation

```python
from enum import Enum
from datetime import datetime

class AllenRelation(Enum):
    """13 possible relations between two intervals"""
    BEFORE = "before"          # X before Y: X----- Y-----
    AFTER = "after"            # X after Y: Y----- X-----
    MEETS = "meets"            # X meets Y: X----XY-----
    MET_BY = "met_by"          # X met-by Y: Y----YX-----
    OVERLAPS = "overlaps"      # X overlaps Y: X--[X--]Y--
    OVERLAPPED_BY = "overlapped_by"
    DURING = "during"          # X during Y: Y[--X--]Y
    CONTAINS = "contains"      # X contains Y: X[--Y--]X
    STARTS = "starts"          # X starts Y: [XY----]Y
    STARTED_BY = "started_by"
    FINISHES = "finishes"      # X finishes Y: Y[----YX]
    FINISHED_BY = "finished_by"
    EQUALS = "equals"          # X equals Y: [--XY--]

class AllenInterval:
    """Interval with Allen's temporal logic"""

    def __init__(self, start: datetime, end: datetime):
        assert start <= end, "Invalid interval"
        self.start = start
        self.end = end

    def relate_to(self, other: 'AllenInterval') -> AllenRelation:
        """Determine Allen relation to another interval"""

        if self.end < other.start:
            return AllenRelation.BEFORE
        elif self.start > other.end:
            return AllenRelation.AFTER
        elif self.end == other.start:
            return AllenRelation.MEETS
        elif self.start == other.end:
            return AllenRelation.MET_BY
        elif self.start == other.start and self.end == other.end:
            return AllenRelation.EQUALS
        elif self.start == other.start and self.end < other.end:
            return AllenRelation.STARTS
        elif self.start == other.start and self.end > other.end:
            return AllenRelation.STARTED_BY
        elif self.end == other.end and self.start > other.start:
            return AllenRelation.FINISHES
        elif self.end == other.end and self.start < other.start:
            return AllenRelation.FINISHED_BY
        elif self.start < other.start and self.end > other.end:
            return AllenRelation.CONTAINS
        elif self.start > other.start and self.end < other.end:
            return AllenRelation.DURING
        elif self.start < other.start < self.end < other.end:
            return AllenRelation.OVERLAPS
        else:  # other.start < self.start < other.end < self.end
            return AllenRelation.OVERLAPPED_BY

    def satisfies_constraint(
        self,
        other: 'AllenInterval',
        required_relation: AllenRelation
    ) -> bool:
        """Check if interval satisfies temporal constraint"""
        actual = self.relate_to(other)
        return actual == required_relation

class TemporalConstraint:
    """Clinical temporal constraint (e.g., "sepsis before septic shock")"""

    def __init__(
        self,
        event1_type: str,
        event2_type: str,
        relation: AllenRelation,
        time_bound: Optional[timedelta] = None  # Max time gap
    ):
        self.event1_type = event1_type
        self.event2_type = event2_type
        self.relation = relation
        self.time_bound = time_bound

    def is_satisfied(
        self,
        event1: ClinicalEvent,
        event2: ClinicalEvent
    ) -> bool:
        """Check if two events satisfy constraint"""
        interval1 = event1.to_interval()
        interval2 = event2.to_interval()

        # Check Allen relation
        if not interval1.satisfies_constraint(interval2, self.relation):
            return False

        # Check time bound if specified
        if self.time_bound:
            gap = abs((event2.start_time - event1.start_time))
            if gap > self.time_bound:
                return False

        return True
```

---

## 3. TECHNOLOGY STACK

### 3.1 Graph Database: Neo4j vs DGL vs PyG

| Component | Technology | Rationale | Alternatives Considered |
|-----------|------------|-----------|------------------------|
| **Graph Storage** | **Neo4j** | - Native temporal queries (APOC procedures)<br>- Cypher for complex pattern matching<br>- Production-ready for clinical deployment<br>- HIPAA compliance available | - PostgreSQL + pg_graph (limited temporal support)<br>- Amazon Neptune (vendor lock-in) |
| **Graph Learning** | **DGL (Deep Graph Library)** | - PyTorch integration<br>- Efficient heterogeneous graph support<br>- Temporal graph neural network modules<br>- Better documentation than PyG | - PyTorch Geometric (PyG): slightly faster but less mature temporal support<br>- Spektral (TensorFlow-based, smaller community) |
| **Hybrid Approach** | Neo4j (OLTP) + DGL (OLAP) | - Neo4j for real-time queries and storage<br>- DGL for batch training<br>- ETL pipeline between systems | Single system trade-off: either slow training (Neo4j) or no production readiness (DGL) |

**Implementation Strategy:**
```python
# Graph construction: Neo4j (persistent)
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687")

def create_temporal_edge(tx, source_id, target_id, relation_type, timestamp):
    tx.run("""
        MATCH (a:Event {id: $source_id})
        MATCH (b:Event {id: $target_id})
        CREATE (a)-[r:PRECEDES {
            timestamp: datetime($timestamp),
            relation: $relation_type
        }]->(b)
    """, source_id=source_id, target_id=target_id,
         relation_type=relation_type, timestamp=timestamp)

# Graph learning: DGL (training)
import dgl
import torch

def neo4j_to_dgl(neo4j_graph):
    """Convert Neo4j graph to DGL for training"""
    # Extract nodes and edges from Neo4j
    # Build DGL heterograph
    graph_data = {
        ('patient', 'has', 'event'): (patient_ids, event_ids),
        ('event', 'precedes', 'event'): (src_events, dst_events),
        # ... more edge types
    }
    g = dgl.heterograph(graph_data)
    return g
```

### 3.2 LNN Framework: IBM LNN vs Custom

| Option | Advantages | Disadvantages | Decision |
|--------|-----------|---------------|----------|
| **IBM LNN** | - Mature framework<br>- Well-documented<br>- Active research support<br>- Proven healthcare applications | - Limited temporal logic support<br>- Requires TensorFlow (not PyTorch)<br>- Less flexible for custom operators | **Extend IBM LNN** |
| **Custom Implementation** | - Full control over architecture<br>- Native PyTorch integration<br>- Optimized for temporal reasoning | - Reinventing the wheel<br>- Validation burden<br>- Maintenance overhead | Use for temporal extensions only |

**Hybrid Approach:**
- **Base**: IBM LNN for core logical operations (AND, OR, NOT, IMPLIES)
- **Extension**: Custom PyTorch modules for temporal operators (BEFORE, DURING, etc.)
- **Integration**: Wrapper layer to combine TensorFlow LNN with PyTorch GNN

```python
# IBM LNN for clinical rules
import lnn

# Define clinical guideline as LNN
model = lnn.Model()

# Predicates
has_fever = model.add_predicate('has_fever')
has_hypotension = model.add_predicate('has_hypotension')
has_tachycardia = model.add_predicate('has_tachycardia')
has_sepsis = model.add_predicate('has_sepsis')

# Rules (SIRS criteria simplified)
rule1 = model.add_formula(lnn.Implies(
    lnn.And(has_fever, has_tachycardia, has_hypotension),
    has_sepsis
))

# Custom temporal extension (PyTorch)
class TemporalLNNOperator(torch.nn.Module):
    """Temporal logic operators for LNN"""

    def __init__(self, operator_type='before'):
        super().__init__()
        self.operator_type = operator_type

    def forward(self, event1_time, event2_time, event1_conf, event2_conf):
        """
        Compute truth value of temporal relation
        Args:
            event1_time: (batch, 2) [start, end] timestamps
            event2_time: (batch, 2) [start, end] timestamps
            event1_conf: (batch,) confidence in event1
            event2_conf: (batch,) confidence in event2
        Returns:
            truth_value: (batch,) confidence in temporal relation
        """
        if self.operator_type == 'before':
            # event1.end < event2.start
            time_satisfied = (event1_time[:, 1] < event2_time[:, 0]).float()
        # ... other operators

        # Combine with event confidences (Łukasiewicz logic)
        truth_value = torch.min(time_satisfied,
                                torch.min(event1_conf, event2_conf))
        return truth_value
```

### 3.3 Temporal Reasoning: OWL-Time Integration

**OWL-Time Ontology** provides W3C-standard temporal representation:

```xml
<!-- Example: OWL-Time representation of sepsis onset -->
<owl:NamedIndividual rdf:about="#SepsisOnset_Patient123">
    <rdf:type rdf:resource="http://www.w3.org/2006/time#Instant"/>
    <time:inXSDDateTime rdf:datatype="xsd:dateTime">
        2025-03-15T14:32:00Z
    </time:inXSDDateTime>
</owl:NamedIndividual>

<owl:NamedIndividual rdf:about="#Sepsis_Patient123">
    <rdf:type rdf:resource="#SepsisEvent"/>
    <time:hasBeginning rdf:resource="#SepsisOnset_Patient123"/>
</owl:NamedIndividual>
```

**Integration Strategy:**
1. Store temporal annotations in OWL-Time format for interoperability
2. Convert to Allen intervals for reasoning
3. Use SWRL (Semantic Web Rule Language) for temporal rule encoding
4. Export results back to OWL-Time for clinical system integration

### 3.4 Infrastructure

```yaml
# Technology Stack Summary
Core Framework:
  - PyTorch: 2.1+
  - Python: 3.10+
  - CUDA: 12.1+ (for GPU acceleration)

Graph Processing:
  - Neo4j: 5.x (Community or Enterprise)
  - DGL: 1.1+
  - NetworkX: 3.x (for analysis)

Neuro-Symbolic:
  - IBM LNN: 1.0+
  - TensorFlow: 2.x (for LNN)
  - PyTorch-TensorFlow bridge

Clinical NLP:
  - spaCy: 3.x
  - scispaCy: 0.5+
  - BioClinicalBERT (HuggingFace)
  - UMLS integration

Data Processing:
  - Pandas: 2.x
  - NumPy: 1.24+
  - FHIR Client: 4.x
  - HL7apy: 1.3+

Experiment Tracking:
  - Weights & Biases (W&B)
  - MLflow
  - DVC (Data Version Control)

Deployment:
  - Docker: 20.x+
  - Kubernetes (optional, for scale)
  - FastAPI: 0.100+
  - Nginx (reverse proxy)

Testing & CI/CD:
  - pytest: 7.x
  - GitHub Actions
  - pre-commit hooks
```

**Compute Requirements:**

| Phase | GPU | RAM | Storage | Duration |
|-------|-----|-----|---------|----------|
| Data Preprocessing | Optional | 64 GB | 1 TB | 1 week |
| KG Construction | Optional | 128 GB | 2 TB | 2 weeks |
| Model Training | 4x A100 (40GB) | 256 GB | 500 GB | 4 weeks |
| Evaluation | 1x A100 | 64 GB | 100 GB | 1 week |
| Deployment (inference) | 1x T4 | 32 GB | 50 GB | Real-time |

**Cloud vs On-Prem:**
- **Development**: AWS/GCP with research credits (UCF has partnerships)
- **Training**: STOKES cluster at UCF (if available) or AWS p4d instances
- **Deployment**: On-premises at partner hospitals (HIPAA compliance)

---

## 4. KEY ALGORITHMS

### 4.1 Algorithm 1: Temporal KG Construction from EHR Events

```python
"""
Algorithm: MIMIC-to-TemporalKG Pipeline
Input: MIMIC-IV tables (admissions, chartevents, labevents, diagnoses_icd, etc.)
Output: Temporal knowledge graph G = (V, E, T)
"""

def construct_temporal_kg(patient_id: str, mimic_tables: Dict) -> TemporalKG:
    """
    Build temporal KG for a single patient stay

    Complexity: O(n log n) where n = number of events
    - O(n) for event extraction
    - O(n log n) for temporal sorting
    - O(n²) for edge construction (with pruning to O(n log n))
    """

    # Step 1: Extract and unify events
    events = []

    # 1a. Vital signs from chartevents
    for row in mimic_tables['chartevents']:
        if row['subject_id'] == patient_id:
            event = ClinicalEvent(
                event_id=f"vital_{row['charttime']}_{row['itemid']}",
                patient_id=patient_id,
                event_type="vital",
                start_time=row['charttime'],
                value=row['valuenum'],
                codes={"itemid": row['itemid']},
                source="structured"
            )
            events.append(event)

    # 1b. Lab results from labevents
    for row in mimic_tables['labevents']:
        if row['subject_id'] == patient_id:
            event = ClinicalEvent(
                event_id=f"lab_{row['charttime']}_{row['itemid']}",
                patient_id=patient_id,
                event_type="lab",
                start_time=row['charttime'],
                value=row['valuenum'],
                codes={"itemid": row['itemid']},
                source="structured"
            )
            events.append(event)

    # 1c. Diagnoses from diagnoses_icd (onset time = admission time)
    admission_time = mimic_tables['admissions'][patient_id]['admittime']
    for row in mimic_tables['diagnoses_icd']:
        if row['subject_id'] == patient_id:
            event = ClinicalEvent(
                event_id=f"dx_{row['icd_code']}",
                patient_id=patient_id,
                event_type="diagnosis",
                start_time=admission_time,  # Approximation
                codes={"icd10": row['icd_code']},
                source="structured"
            )
            events.append(event)

    # 1d. Medications from prescriptions/emar
    for row in mimic_tables['prescriptions']:
        if row['subject_id'] == patient_id:
            event = ClinicalEvent(
                event_id=f"med_{row['starttime']}_{row['drug']}",
                patient_id=patient_id,
                event_type="medication",
                start_time=row['starttime'],
                end_time=row['stoptime'],
                value_text=row['drug'],
                codes={"gsn": row['gsn']},
                source="structured"
            )
            events.append(event)

    # 1e. Extract events from clinical notes (NLP)
    for note in mimic_tables['noteevents']:
        if note['subject_id'] == patient_id:
            nlp_events = extract_events_from_text(
                text=note['text'],
                note_time=note['charttime'],
                patient_id=patient_id
            )
            events.extend(nlp_events)

    # Step 2: Temporal sorting and interval construction
    events.sort(key=lambda e: e.start_time)

    for event in events:
        if event.end_time is None:
            # Assign default duration based on event type
            if event.event_type == "vital":
                event.end_time = event.start_time + timedelta(hours=1)
            elif event.event_type == "medication":
                event.end_time = event.start_time + timedelta(days=1)
            # ... etc.

    # Step 3: Build graph nodes
    G = TemporalKG()

    # Add patient node
    patient_node = Node(
        node_id=patient_id,
        node_type="patient",
        attributes=mimic_tables['patients'][patient_id]
    )
    G.add_node(patient_node)

    # Add event nodes
    for event in events:
        event_node = Node(
            node_id=event.event_id,
            node_type=event.event_type,
            attributes={
                'timestamp': event.start_time,
                'value': event.value,
                'codes': event.codes
            }
        )
        G.add_node(event_node)

        # Connect to patient
        G.add_edge(patient_id, event.event_id, edge_type="HAS_EVENT")

    # Step 4: Build temporal edges (PRECEDES)
    for i, e1 in enumerate(events):
        # Only connect to next k events (temporal window pruning)
        for j in range(i+1, min(i+10, len(events))):
            e2 = events[j]

            # Compute Allen relation
            interval1 = e1.to_interval()
            interval2 = e2.to_interval()
            allen_relation = interval1.relate_to(interval2)

            # Add temporal edge
            G.add_edge(
                e1.event_id, e2.event_id,
                edge_type="PRECEDES",
                attributes={
                    'allen_relation': allen_relation.value,
                    'time_gap': (e2.start_time - e1.start_time).total_seconds(),
                    'weight': 1.0 / (1.0 + (e2.start_time - e1.start_time).total_seconds() / 3600)
                }
            )

    # Step 5: Add clinical knowledge edges (INDICATES, TREATS)
    clinical_rules = load_clinical_knowledge_base()

    for rule in clinical_rules:
        # Example: "High lactate INDICATES sepsis"
        if rule.type == "INDICATES":
            # Find matching events
            source_events = [e for e in events if matches_condition(e, rule.source)]
            target_events = [e for e in events if matches_condition(e, rule.target)]

            for se in source_events:
                for te in target_events:
                    # Check temporal constraint (e.g., lactate before sepsis diagnosis)
                    if check_temporal_constraint(se, te, rule.temporal_constraint):
                        G.add_edge(
                            se.event_id, te.event_id,
                            edge_type="INDICATES",
                            attributes={
                                'confidence': rule.confidence,
                                'source': 'guideline'
                            }
                        )

    # Step 6: Add inferred edges (causal discovery)
    # Use Granger causality or constraint-based methods
    inferred_edges = discover_causal_edges(events)
    for edge in inferred_edges:
        G.add_edge(
            edge.source, edge.target,
            edge_type="CAUSES",
            attributes={
                'confidence': edge.confidence,
                'source': 'inferred'
            }
        )

    return G


def extract_events_from_text(text: str, note_time: datetime, patient_id: str) -> List[ClinicalEvent]:
    """
    NLP-based event extraction from clinical notes
    Uses BioClinicalBERT for NER + scispaCy for relation extraction
    """
    import spacy
    nlp = spacy.load("en_core_sci_lg")

    doc = nlp(text)
    events = []

    # Extract entities
    for ent in doc.ents:
        if ent.label_ in ["PROBLEM", "TEST", "TREATMENT"]:
            event_type = {
                "PROBLEM": "diagnosis",
                "TEST": "lab",
                "TREATMENT": "medication"
            }[ent.label_]

            # Map to SNOMED (simplified - use UMLS in practice)
            snomed_code = map_to_snomed(ent.text)

            event = ClinicalEvent(
                event_id=f"nlp_{note_time}_{ent.start}",
                patient_id=patient_id,
                event_type=event_type,
                start_time=note_time,  # Approximation
                value_text=ent.text,
                codes={"snomed": snomed_code},
                source="nlp",
                confidence=0.8  # Lower confidence for NLP
            )
            events.append(event)

    return events
```

### 4.2 Algorithm 2: Neuro-Symbolic Inference with Clinical Constraints

```python
"""
Algorithm: Hybrid Neuro-Symbolic Sepsis Prediction
Input: Temporal KG G, clinical guidelines R, patient history H
Output: Risk score p ∈ [0,1], explanation E
"""

class HybridReasoningModel(torch.nn.Module):
    """
    Combines GNN (neural) with LNN (symbolic) for clinical prediction
    """

    def __init__(self, config):
        super().__init__()

        # Neural component: Temporal GNN
        self.gnn = TemporalGNN(
            in_dim=config.node_feat_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.embedding_dim,
            num_layers=config.num_gnn_layers,
            num_edge_types=len(EdgeSchema),
            temporal_encoding=config.temporal_encoding  # "sinusoidal" | "learnable"
        )

        # Symbolic component: Clinical constraint layer
        self.constraint_layer = ClinicalConstraintLayer(
            constraints=config.clinical_constraints,
            constraint_weight=config.constraint_weight
        )

        # Prediction head
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(config.embedding_dim, config.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(config.hidden_dim, 1),
            torch.nn.Sigmoid()
        )

        # Loss weights
        self.alpha = config.alpha  # Prediction loss weight
        self.beta = config.beta    # Constraint loss weight
        self.gamma = config.gamma  # Temporal consistency loss weight

    def forward(self, G, target_node_id, constraints):
        """
        Forward pass through hybrid reasoning

        Args:
            G: DGL graph
            target_node_id: Patient node to predict for
            constraints: List of TemporalConstraint objects

        Returns:
            risk_score: Prediction in [0,1]
            constraint_violations: Number of violated constraints
            embeddings: Node embeddings for explanation
        """

        # Step 1: Neural reasoning (GNN)
        node_embeddings = self.gnn(G)
        patient_embedding = node_embeddings[target_node_id]

        # Step 2: Extract risk score from neural path
        risk_score_neural = self.predictor(patient_embedding)

        # Step 3: Symbolic reasoning (constraint checking)
        constraint_scores = []
        constraint_violations = 0

        for constraint in constraints:
            # Example: "Sepsis requires (fever OR hypothermia) AND (tachycardia OR bradycardia)"
            # Evaluate constraint satisfaction
            sat_score = self.evaluate_constraint(G, node_embeddings, constraint)
            constraint_scores.append(sat_score)

            if sat_score < 0.5:  # Violated
                constraint_violations += 1

        # Step 4: Combine neural + symbolic
        avg_constraint_score = torch.stack(constraint_scores).mean()

        # Soft constraint: penalize predictions that violate clinical logic
        risk_score_final = risk_score_neural * avg_constraint_score

        return {
            'risk_score': risk_score_final,
            'risk_score_neural': risk_score_neural,
            'constraint_satisfaction': avg_constraint_score,
            'constraint_violations': constraint_violations,
            'embeddings': node_embeddings
        }

    def evaluate_constraint(self, G, embeddings, constraint):
        """
        Evaluate a clinical constraint using LNN operators

        Example constraint (SIRS criteria):
        has_sepsis :=
            (temp > 38 OR temp < 36) AND
            (hr > 90) AND
            (rr > 20 OR PaCO2 < 32) AND
            (wbc > 12000 OR wbc < 4000 OR bands > 10)
        """

        # Extract relevant node embeddings
        relevant_nodes = self.find_nodes_for_constraint(G, constraint)

        # Apply LNN operators
        if constraint.operator == "AND":
            scores = [self.check_predicate(G, embeddings, pred)
                     for pred in constraint.predicates]
            # Łukasiewicz t-norm: AND(a,b) = max(0, a + b - 1)
            result = torch.clamp(sum(scores) - len(scores) + 1, 0, 1)

        elif constraint.operator == "OR":
            scores = [self.check_predicate(G, embeddings, pred)
                     for pred in constraint.predicates]
            # Łukasiewicz t-conorm: OR(a,b) = min(1, a + b)
            result = torch.clamp(sum(scores), 0, 1)

        # ... more operators

        return result

    def compute_loss(self, outputs, targets, G, constraints):
        """
        Multi-objective loss combining prediction + constraints + temporal consistency
        """

        # L1: Standard prediction loss (BCE)
        L_prediction = F.binary_cross_entropy(
            outputs['risk_score'],
            targets
        )

        # L2: Constraint violation penalty
        # Penalize predictions that violate known clinical rules
        L_constraint = torch.clamp(
            1.0 - outputs['constraint_satisfaction'],
            0, 1
        )

        # L3: Temporal consistency loss
        # Embeddings of temporally close events should be similar
        L_temporal = self.temporal_consistency_loss(G, outputs['embeddings'])

        # Combined loss
        L_total = (
            self.alpha * L_prediction +
            self.beta * L_constraint +
            self.gamma * L_temporal
        )

        return L_total, {
            'loss_prediction': L_prediction.item(),
            'loss_constraint': L_constraint.item(),
            'loss_temporal': L_temporal.item()
        }

    def temporal_consistency_loss(self, G, embeddings):
        """
        Encourage embeddings of temporally adjacent events to be similar
        """
        temporal_edges = G.edges(etype='PRECEDES')
        src_embeds = embeddings[temporal_edges[0]]
        dst_embeds = embeddings[temporal_edges[1]]

        # Get time gaps
        edge_data = G.edges['PRECEDES'].data['time_gap']

        # Weight by temporal proximity (closer = higher weight)
        weights = torch.exp(-edge_data / 3600)  # Decay over hours

        # MSE weighted by proximity
        consistency_loss = (weights * F.mse_loss(
            src_embeds, dst_embeds, reduction='none'
        ).mean(dim=1)).mean()

        return consistency_loss


class TemporalGNN(torch.nn.Module):
    """
    Temporal Graph Neural Network for heterogeneous clinical graphs
    Uses R-GCN with temporal attention
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, num_edge_types, temporal_encoding):
        super().__init__()

        # Temporal encoding layer
        if temporal_encoding == "sinusoidal":
            self.temporal_encoder = SinusoidalTemporalEncoding(hidden_dim)
        else:
            self.temporal_encoder = LearnableTemporalEncoding(hidden_dim)

        # R-GCN layers (handles multiple edge types)
        self.layers = torch.nn.ModuleList([
            dgl.nn.RelGraphConv(
                in_dim if i == 0 else hidden_dim,
                hidden_dim,
                num_edge_types,
                activation=F.relu
            )
            for i in range(num_layers)
        ])

        # Temporal attention
        self.temporal_attention = TemporalAttention(hidden_dim)

        # Output projection
        self.output_layer = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, G):
        """
        Forward pass through temporal GNN

        Args:
            G: DGL heterograph with temporal annotations

        Returns:
            node_embeddings: (num_nodes, out_dim)
        """

        # Step 1: Encode temporal information
        timestamps = G.ndata['timestamp']
        temporal_embeds = self.temporal_encoder(timestamps)

        # Step 2: Initialize node features
        h = G.ndata['features'] + temporal_embeds

        # Step 3: Message passing with relation-specific transformations
        for layer in self.layers:
            h = layer(G, h, G.edata['edge_type'])

        # Step 4: Temporal attention pooling
        # Aggregate information with attention to recent events
        h = self.temporal_attention(h, timestamps)

        # Step 5: Output projection
        out = self.output_layer(h)

        return out


class SinusoidalTemporalEncoding(torch.nn.Module):
    """
    Sinusoidal positional encoding adapted for timestamps
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, timestamps):
        """
        Args:
            timestamps: (num_nodes,) Unix timestamps
        Returns:
            encoding: (num_nodes, d_model)
        """
        # Normalize timestamps to [0, 1] within stay
        t_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)

        # Sinusoidal encoding
        encoding = torch.zeros(len(timestamps), self.d_model)
        position = t_norm.unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) *
                             -(math.log(10000.0) / self.d_model))

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        return encoding
```

### 4.3 Algorithm 3: Explanation Chain Extraction

```python
"""
Algorithm: Generate Clinician-Interpretable Explanation
Input: Trained model M, patient graph G, prediction p
Output: Explanation chain E = (reasoning_path, counterfactuals, NL_summary)
"""

class ExplanationGenerator:
    """
    Generates multi-modal explanations for clinical predictions
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config

        # GNN Explainer for subgraph extraction
        self.gnn_explainer = GNNExplainer(model.gnn, epochs=200)

        # Counterfactual generator
        self.cf_generator = CounterfactualGenerator(model)

        # Natural language renderer
        self.nl_renderer = NLExplanationRenderer()

    def explain(self, G, patient_id, prediction):
        """
        Generate comprehensive explanation for a prediction

        Returns:
            explanation: {
                'reasoning_chain': List of (event, contribution) tuples,
                'counterfactuals': List of minimal changes that would flip prediction,
                'nl_summary': Natural language explanation,
                'confidence': Calibrated confidence score,
                'supporting_evidence': Clinical guidelines cited
            }
        """

        # Step 1: Extract important subgraph (GNN Explainer)
        important_subgraph = self.extract_reasoning_subgraph(G, patient_id)

        # Step 2: Build temporal reasoning chain
        reasoning_chain = self.build_temporal_chain(important_subgraph, patient_id)

        # Step 3: Generate counterfactuals
        counterfactuals = self.cf_generator.generate(G, patient_id, prediction)

        # Step 4: Extract constraint violations (if any)
        violated_constraints = self.check_constraints(G, patient_id)

        # Step 5: Render natural language
        nl_summary = self.nl_renderer.render(
            reasoning_chain,
            counterfactuals,
            prediction,
            violated_constraints
        )

        # Step 6: Compute calibrated confidence
        confidence = self.calibrate_confidence(prediction, violated_constraints)

        # Step 7: Cite supporting clinical guidelines
        supporting_evidence = self.find_supporting_guidelines(reasoning_chain)

        return {
            'reasoning_chain': reasoning_chain,
            'counterfactuals': counterfactuals,
            'nl_summary': nl_summary,
            'confidence': confidence,
            'supporting_evidence': supporting_evidence,
            'subgraph': important_subgraph
        }

    def extract_reasoning_subgraph(self, G, patient_id):
        """
        Use GNN Explainer to find most important nodes/edges
        """
        # Get node features and edge index
        node_features = G.ndata['features']

        # Run GNN Explainer
        node_mask, edge_mask = self.gnn_explainer.explain_node(
            patient_id,
            node_features,
            G
        )

        # Threshold to get binary mask
        important_nodes = (node_mask > 0.5).nonzero().squeeze()
        important_edges = (edge_mask > 0.5).nonzero().squeeze()

        # Extract subgraph
        subgraph = dgl.node_subgraph(G, important_nodes)

        return subgraph

    def build_temporal_chain(self, subgraph, patient_id):
        """
        Build chronological reasoning chain from important events

        Returns:
            chain: [
                {
                    'event': ClinicalEvent,
                    'timestamp': datetime,
                    'contribution': float,  # Importance score
                    'reason': str           # Why this event matters
                },
                ...
            ]
        """
        chain = []

        # Get all events in subgraph
        event_nodes = [n for n in subgraph.nodes() if subgraph.ndata['type'][n] == 'event']

        # Sort by timestamp
        event_nodes.sort(key=lambda n: subgraph.ndata['timestamp'][n])

        for node_id in event_nodes:
            event_data = subgraph.ndata[node_id]

            # Compute contribution using integrated gradients
            contribution = self.compute_contribution(subgraph, node_id, patient_id)

            # Determine reason (pattern matching against known patterns)
            reason = self.infer_reason(subgraph, node_id)

            chain.append({
                'event': self.node_to_event(event_data),
                'timestamp': event_data['timestamp'],
                'contribution': contribution,
                'reason': reason
            })

        return chain

    def compute_contribution(self, G, node_id, patient_id):
        """
        Integrated Gradients to compute feature importance
        """
        # Baseline: remove this node
        G_baseline = self.remove_node(G, node_id)

        # Prediction with node
        pred_with = self.model(G, patient_id, None)['risk_score']

        # Prediction without node
        pred_without = self.model(G_baseline, patient_id, None)['risk_score']

        # Contribution = difference
        contribution = abs(pred_with - pred_without)

        return contribution.item()


class CounterfactualGenerator:
    """
    Generate minimal counterfactual explanations
    "If lab value X was Y instead of Z, prediction would change"
    """

    def __init__(self, model):
        self.model = model

    def generate(self, G, patient_id, original_pred, num_cfs=3):
        """
        Find minimal changes to flip prediction

        Returns:
            counterfactuals: [
                {
                    'change': str,  # "Lactate: 4.2 → 1.5 mmol/L"
                    'new_prediction': float,
                    'plausibility': float  # Clinical plausibility score
                },
                ...
            ]
        """
        counterfactuals = []

        # Get all modifiable nodes (labs, vitals - not diagnoses)
        modifiable_nodes = self.get_modifiable_nodes(G)

        # Sort by gradient magnitude (most impactful first)
        G_copy = G.clone()
        G_copy.ndata['features'].requires_grad = True

        pred = self.model(G_copy, patient_id, None)['risk_score']
        pred.backward()

        gradients = G_copy.ndata['features'].grad

        # Sort nodes by gradient
        sorted_nodes = sorted(
            modifiable_nodes,
            key=lambda n: gradients[n].abs().sum(),
            reverse=True
        )

        # Try modifying top-k nodes
        for node_id in sorted_nodes[:10]:
            # Modify value
            original_value = G.ndata['features'][node_id].clone()

            # Try different perturbations
            for delta in [-0.5, -0.25, 0.25, 0.5]:  # Normalized scale
                G_cf = G.clone()
                G_cf.ndata['features'][node_id] += delta

                # Check new prediction
                new_pred = self.model(G_cf, patient_id, None)['risk_score']

                # Check if prediction flipped
                if (original_pred > 0.5 and new_pred < 0.5) or \
                   (original_pred < 0.5 and new_pred > 0.5):

                    # Compute clinical plausibility
                    plausibility = self.check_plausibility(
                        node_id,
                        original_value,
                        G_cf.ndata['features'][node_id]
                    )

                    if plausibility > 0.3:  # Filter implausible changes
                        counterfactuals.append({
                            'change': self.format_change(node_id, original_value,
                                                         G_cf.ndata['features'][node_id]),
                            'new_prediction': new_pred.item(),
                            'plausibility': plausibility
                        })

                        if len(counterfactuals) >= num_cfs:
                            break

            if len(counterfactuals) >= num_cfs:
                break

        return counterfactuals


class NLExplanationRenderer:
    """
    Convert structured explanation to natural language
    """

    def __init__(self):
        # Template-based for now; can upgrade to LLM later
        self.templates = {
            'high_risk': "Patient shows high sepsis risk ({confidence:.1%} confidence). Key factors: {factors}.",
            'low_risk': "Patient shows low sepsis risk ({confidence:.1%} confidence).",
            'factor': "{event_name} at {time} (contribution: {contribution:.1%})",
            'counterfactual': "If {change}, risk would be {new_risk}.",
            'constraint_violation': "Note: Prediction violates clinical guideline '{guideline}'."
        }

    def render(self, reasoning_chain, counterfactuals, prediction, violated_constraints):
        """
        Generate natural language summary

        Example output:
        '''
        Patient shows HIGH sepsis risk (78% confidence).

        Key contributing factors:
        1. Elevated lactate (4.2 mmol/L) at 14:30 - indicates tissue hypoperfusion
        2. Hypotension (SBP 85 mmHg) at 14:45 - suggests hemodynamic instability
        3. Tachycardia (HR 125 bpm) at 14:50 - compensatory response

        Counterfactual scenarios:
        - If lactate was 1.5 mmol/L (normal), risk would drop to 35%
        - If SBP was 110 mmHg, risk would drop to 42%

        Clinical guidelines cited:
        - Surviving Sepsis Campaign 2021: Lactate >2 mmol/L + hypotension = septic shock
        '''
        """

        # Header
        risk_level = "HIGH" if prediction > 0.5 else "LOW"
        summary = f"Patient shows {risk_level} sepsis risk ({prediction:.1%} confidence).\n\n"

        # Key factors
        if len(reasoning_chain) > 0:
            summary += "Key contributing factors:\n"
            for i, item in enumerate(reasoning_chain[:5], 1):  # Top 5
                summary += f"{i}. {item['event'].value_text or item['event'].value} "
                summary += f"at {item['timestamp'].strftime('%H:%M')} "
                summary += f"({item['contribution']:.1%} contribution) - {item['reason']}\n"

        # Counterfactuals
        if len(counterfactuals) > 0:
            summary += "\nCounterfactual scenarios:\n"
            for cf in counterfactuals:
                summary += f"- If {cf['change']}, risk would be {cf['new_prediction']:.1%}\n"

        # Constraint violations (warnings)
        if len(violated_constraints) > 0:
            summary += "\nNote: "
            for vc in violated_constraints:
                summary += f"Prediction may violate guideline '{vc['guideline']}'. "

        return summary
```

---

## 5. PROTOTYPE ROADMAP

### Month 1: Data Pipeline + Basic KG Construction

**Objectives:**
- [ ] Set up MIMIC-IV access (PhysioNet credentialing)
- [ ] Implement FHIR/MIMIC data loaders
- [ ] Build basic KG construction pipeline (structured data only)
- [ ] Set up Neo4j database and DGL environment

**Deliverables:**
```
hybrid-reasoning-acute-care/
├── data/
│   ├── mimic_loader.py          # Load MIMIC tables
│   ├── fhir_parser.py            # Parse FHIR resources
│   └── unified_model.py          # Convert to unified representation
├── kg/
│   ├── schema.py                 # Node/edge schemas
│   ├── builder.py                # KG construction logic
│   ├── temporal_reasoner.py      # Allen's interval algebra
│   └── neo4j_connector.py        # Neo4j interface
└── tests/
    ├── test_data_loader.py
    └── test_kg_construction.py
```

**Success Metrics:**
- Successfully construct KGs for 1000 MIMIC-IV patients
- Average KG size: 50-200 nodes, 100-500 edges per patient
- Query latency: <100ms for patient retrieval in Neo4j

**Risks:**
- PhysioNet access delays → Apply early, have UCF IT support
- Data quality issues → Extensive data validation scripts

---

### Month 2: Baseline Models on MIMIC

**Objectives:**
- [ ] Implement baseline models (LSTM, Transformer, standard GNN)
- [ ] Define evaluation metrics and protocols
- [ ] Establish benchmark results on MIMIC-IV-ED
- [ ] Set up experiment tracking (W&B)

**Deliverables:**
```
models/
├── baselines/
│   ├── lstm_baseline.py          # Sequential LSTM
│   ├── transformer_baseline.py   # Temporal Transformer
│   └── gnn_baseline.py           # Standard R-GCN
├── utils/
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # Evaluation metrics
│   └── data_splits.py            # Train/val/test splits
└── configs/
    └── baseline_config.yaml       # Hyperparameters
```

**Target Metrics (Sepsis Prediction on MIMIC-IV-ED):**
| Model | AUROC | AUPRC | Specificity@90% Sens | Early Prediction |
|-------|-------|-------|----------------------|------------------|
| LSTM Baseline | 0.75-0.80 | 0.30-0.40 | 0.40-0.50 | 0-3h before |
| Transformer | 0.78-0.83 | 0.35-0.45 | 0.45-0.55 | 0-4h before |
| R-GCN | 0.80-0.85 | 0.40-0.50 | 0.50-0.60 | 0-4h before |

**Success Criteria:**
- Replicate published MIMIC results (± 0.03 AUROC)
- Complete benchmarking paper draft

---

### Month 3: Temporal Encoding Integration

**Objectives:**
- [ ] Implement temporal KG enhancements (Allen's intervals)
- [ ] Add temporal attention mechanisms
- [ ] Compare temporal encoding strategies
- [ ] Integrate clinical NLP for note-based events

**Deliverables:**
```
models/
├── temporal/
│   ├── temporal_gnn.py           # Temporal-aware R-GCN
│   ├── temporal_attention.py     # Temporal attention layer
│   └── allen_encoder.py          # Allen relation encoding
├── nlp/
│   ├── event_extractor.py        # BioClinicalBERT NER
│   ├── relation_extractor.py     # scispaCy relations
│   └── umls_mapper.py            # Entity normalization
└── experiments/
    └── temporal_ablation.py       # Ablation study
```

**Target Improvements:**
| Model | AUROC | AUPRC | Early Prediction |
|-------|-------|-------|------------------|
| Temporal-GNN | 0.83-0.88 | 0.45-0.55 | 0-6h before |
| + NLP Events | 0.85-0.90 | 0.50-0.60 | 0-6h before |

**Success Criteria:**
- 3-5% AUROC improvement over baseline
- Demonstrate temporal encoding value in ablation study

---

### Month 4: LNN Constraint Layer

**Objectives:**
- [ ] Formalize clinical guidelines as LNN rules
- [ ] Implement neuro-symbolic integration
- [ ] Evaluate constraint satisfaction vs accuracy trade-off
- [ ] Clinician review of encoded guidelines

**Deliverables:**
```
models/
├── neuro_symbolic/
│   ├── lnn_layer.py              # LNN integration
│   ├── constraint_layer.py       # Soft constraint enforcement
│   └── hybrid_model.py           # Full neuro-symbolic model
├── knowledge/
│   ├── clinical_rules.yaml       # SIRS, qSOFA, etc.
│   ├── snomed_hierarchy.owl      # Medical ontology
│   └── rule_encoder.py           # Convert guidelines to LNN
└── evaluation/
    └── constraint_analysis.py     # Measure guideline adherence
```

**Clinical Guidelines to Encode:**
1. SIRS criteria (sepsis screening)
2. qSOFA score
3. Surviving Sepsis Campaign bundle elements
4. Drug interaction constraints
5. Temporal ordering constraints (e.g., "sepsis before septic shock")

**Success Criteria:**
- 95%+ guideline adherence (constraint satisfaction)
- <2% AUROC drop vs unconstrained model
- Zero critical safety violations (e.g., impossible temporal sequences)

---

### Months 5-6: Evaluation + Iteration

**Objectives:**
- [ ] Multi-site validation (eICU dataset)
- [ ] Clinician evaluation study (10-15 ED physicians at Orlando Health)
- [ ] Implement explanation interface
- [ ] Write first paper drafts (2-3 papers)
- [ ] Prepare open-source release

**Deliverables:**
```
evaluation/
├── clinician_study/
│   ├── study_protocol.pdf        # IRB-approved protocol
│   ├── explanation_interface.py  # Web UI for explanations
│   └── survey_analysis.py        # Analyze clinician feedback
├── external_validation/
│   ├── eicu_validation.py        # eICU results
│   └── generalization_study.py   # Cross-dataset analysis
└── papers/
    ├── temporal_kg_kdd/          # Paper 1: Temporal KG framework
    ├── neuro_symbolic_neurips/   # Paper 2: Hybrid reasoning
    └── benchmark_jamia/          # Paper 3: MIMIC benchmark
```

**Clinician Study Design:**
- **N = 15 ED physicians** (Orlando Health, AdventHealth)
- **30 patient cases** (15 sepsis, 15 non-sepsis)
- **Conditions**: Model explanations vs no explanations
- **Metrics**:
  - Diagnostic accuracy
  - Time to decision
  - Trust (5-point Likert)
  - Explanation helpfulness (5-point Likert)

**Target Outcomes:**
| Metric | Hypothesis |
|--------|-----------|
| Diagnostic Accuracy | +5-10% with explanations |
| Time to Decision | -15-20% with model |
| Trust Score | >4.0/5.0 |
| Explanation Helpfulness | >3.5/5.0 |

**Paper Submission Timeline:**
- **Month 5**: Submit to KDD (temporal KG paper)
- **Month 6**: Submit to AAAI/NeurIPS (neuro-symbolic paper)
- **Month 6**: Submit to JAMIA (benchmark paper)

---

## 6. OPEN SOURCE PLAN

### 6.1 GitHub Repository Structure

```
hybrid-reasoning-acute-care/
│
├── README.md                      # Project overview, installation
├── LICENSE                        # Apache 2.0 or MIT
├── CONTRIBUTING.md                # Contribution guidelines
├── CITATION.cff                   # Citation metadata
├── .github/
│   ├── workflows/
│   │   ├── tests.yml              # CI/CD: pytest on push
│   │   ├── docs.yml               # Auto-build docs
│   │   └── release.yml            # PyPI release automation
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
│
├── data/                          # Data loading and processing
│   ├── __init__.py
│   ├── mimic/
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   └── README.md              # MIMIC access instructions
│   ├── fhir/
│   │   └── parser.py
│   └── schemas/
│       └── clinical_event.py
│
├── kg/                            # Knowledge graph construction
│   ├── __init__.py
│   ├── builder.py
│   ├── schema.py
│   ├── temporal/
│   │   ├── allen_intervals.py
│   │   └── temporal_reasoner.py
│   └── backends/
│       ├── neo4j_backend.py
│       └── dgl_backend.py
│
├── models/                        # Neural and neuro-symbolic models
│   ├── __init__.py
│   ├── baselines/
│   │   ├── lstm.py
│   │   ├── transformer.py
│   │   └── gnn.py
│   ├── temporal_gnn/
│   │   ├── temporal_gnn.py
│   │   └── temporal_encoding.py
│   ├── neuro_symbolic/
│   │   ├── lnn_layer.py
│   │   ├── constraint_layer.py
│   │   └── hybrid_model.py
│   └── utils/
│       ├── train.py
│       └── evaluate.py
│
├── explanation/                   # Explanation generation
│   ├── __init__.py
│   ├── gnn_explainer.py
│   ├── counterfactuals.py
│   └── nl_renderer.py
│
├── knowledge/                     # Clinical knowledge bases
│   ├── rules/
│   │   ├── sirs_criteria.yaml
│   │   ├── qsofa.yaml
│   │   └── sepsis_bundle.yaml
│   └── ontologies/
│       └── snomed_subset.owl
│
├── benchmarks/                    # Benchmarking scripts
│   ├── mimic_sepsis/
│   │   ├── run_benchmark.py
│   │   └── results/
│   └── eicu_validation/
│       └── run_validation.py
│
├── experiments/                   # Experiment configs and scripts
│   ├── configs/
│   │   ├── baseline.yaml
│   │   ├── temporal_gnn.yaml
│   │   └── hybrid.yaml
│   └── scripts/
│       ├── train_baseline.sh
│       └── evaluate_all.sh
│
├── interface/                     # Clinician interface (web app)
│   ├── app.py                     # FastAPI backend
│   ├── frontend/                  # React frontend
│   └── static/
│
├── docs/                          # Documentation (Sphinx)
│   ├── source/
│   │   ├── index.rst
│   │   ├── quickstart.rst
│   │   ├── tutorials/
│   │   ├── api/
│   │   └── papers/
│   └── build/
│
├── tests/                         # Unit and integration tests
│   ├── test_data/
│   ├── test_kg/
│   ├── test_models/
│   └── test_explanation/
│
├── scripts/                       # Utility scripts
│   ├── download_mimic.sh
│   ├── setup_neo4j.sh
│   └── preprocess_all.py
│
├── notebooks/                     # Jupyter tutorials
│   ├── 01_data_exploration.ipynb
│   ├── 02_kg_construction.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_explanation_demo.ipynb
│
├── requirements/
│   ├── base.txt                   # Core dependencies
│   ├── dev.txt                    # Development dependencies
│   └── docs.txt                   # Documentation dependencies
│
├── setup.py                       # Package installation
├── pyproject.toml                 # Modern Python packaging
└── environment.yml                # Conda environment
```

### 6.2 Documentation Standards

**Documentation Levels:**

1. **README.md** (High-level overview)
   ```markdown
   # Hybrid Reasoning for Acute Care

   Official implementation of "Temporal Knowledge Graphs for Emergency Department Risk Stratification"

   ## Features
   - Temporal knowledge graph construction from MIMIC-IV
   - Neuro-symbolic reasoning with clinical constraints
   - Explainable predictions for clinicians

   ## Quick Start
   ```bash
   pip install hybrid-reasoning-acute-care
   python examples/train_mimic_sepsis.py
   ```

   ## Citation
   ```bibtex
   @inproceedings{yourname2026temporal,
     title={Temporal Knowledge Graphs for Acute Care},
     author={Your Name et al.},
     booktitle={KDD},
     year={2026}
   }
   ```
   ```

2. **API Documentation** (Sphinx + Read the Docs)
   - Auto-generated from docstrings
   - Interactive tutorials
   - Architecture diagrams

3. **Tutorials** (Jupyter Notebooks)
   - 01: Load MIMIC data and construct KG
   - 02: Train temporal GNN model
   - 03: Add clinical constraints
   - 04: Generate explanations
   - 05: Evaluate on custom data

4. **Paper Reproducibility**
   ```
   papers/
   ├── kdd2026_temporal_kg/
   │   ├── README.md                # Reproduction instructions
   │   ├── config.yaml              # Exact hyperparameters
   │   ├── run.sh                   # One-command reproduction
   │   └── results/                 # Expected outputs
   ```

### 6.3 Community Engagement Strategy

**Phase 1: Launch (Month 6)**
- [ ] Announce on Twitter/X, LinkedIn, Reddit (r/MachineLearning, r/datascience)
- [ ] Submit to Papers With Code
- [ ] Post on arXiv with code link
- [ ] Create demo video (5-min walkthrough)

**Phase 2: Growth (Months 7-12)**
- [ ] Present at NeurIPS/KDD (if accepted)
- [ ] Write blog post series (3-4 posts)
  - "Why Temporal Knowledge Graphs for Healthcare?"
  - "Neuro-Symbolic AI: Best of Both Worlds"
  - "Building Explainable Clinical AI"
- [ ] Engage with healthcare AI community (Stanford AIMI, MIT LCP)
- [ ] Respond to GitHub issues within 48h

**Phase 3: Ecosystem (Year 2+)**
- [ ] Organize workshop at AAAI/KDD
- [ ] Mentor open-source contributors (Google Summer of Code)
- [ ] Partner with EHR vendors for production deployment
- [ ] Expand to other clinical use cases (AKI, readmission, etc.)

**Success Metrics:**
| Timeframe | GitHub Stars | PyPI Downloads | Paper Citations |
|-----------|--------------|----------------|-----------------|
| 6 months | 50+ | 500+ | 5+ |
| 1 year | 200+ | 2000+ | 20+ |
| 2 years | 500+ | 5000+ | 50+ |

**Community Governance:**
- **Core maintainers**: 2-3 (PhD students + PI)
- **Code review**: All PRs require 1 approval
- **Release cycle**: Monthly patches, quarterly features
- **License**: Apache 2.0 (permissive, industry-friendly)

---

## 7. TECHNICAL RISKS AND MITIGATION

### 7.1 Implementation Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Neo4j performance bottleneck** | Medium | Medium | - Use Neo4j Enterprise with sharding<br>- Hybrid storage (Neo4j + DGL)<br>- Aggressive caching |
| **LNN-PyTorch integration complexity** | High | Medium | - Keep LNN layer small<br>- Use TF-PyTorch bridge<br>- Fallback: custom soft logic in PyTorch |
| **Temporal reasoning scalability** | Medium | High | - Pre-compute Allen relations<br>- Use sparse temporal graphs<br>- Approximate for distant events |
| **NLP event extraction errors** | High | Medium | - Ensemble multiple NER models<br>- Active learning for correction<br>- Confidence thresholding |
| **Hyperparameter tuning cost** | Medium | Medium | - Use Optuna for efficient search<br>- Limit search to key params<br>- Transfer from baselines |

### 7.2 Scientific Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Temporal encoding doesn't improve results** | Low | High | - Ablation studies to validate<br>- Multiple encoding strategies<br>- Focus on interpretability if accuracy flat |
| **Constraints hurt accuracy too much** | Medium | High | - Soft constraints (weighted penalty)<br>- Learn constraint weights<br>- Constraint-accuracy Pareto frontier |
| **Clinicians don't trust explanations** | Medium | High | - Iterative design with clinicians<br>- Multiple explanation modalities<br>- Extensive user study |
| **MIMIC results don't generalize to eICU** | Medium | High | - Multi-site training<br>- Domain adaptation techniques<br>- Identify transferable vs site-specific features |

### 7.3 Resource Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Insufficient GPU compute** | Low | Medium | - UCF STOKES cluster access<br>- AWS/GCP research credits<br>- Model distillation for efficiency |
| **MIMIC data access delays** | Low | High | - Apply for access immediately<br>- Use synthetic data for development<br>- Parallel work on other tasks |
| **Key personnel turnover** | Medium | High | - Thorough documentation<br>- Knowledge transfer sessions<br>- Recruit backup students |

---

## 8. EXPECTED TECHNICAL CONTRIBUTIONS

### 8.1 Novel Components

1. **Temporal KG Schema for Acute Care**
   - First formal schema for ED patient trajectories
   - Allen's interval algebra integration
   - Standardized benchmark for future work

2. **Neuro-Symbolic Clinical Reasoning**
   - Extension of LNN to temporal constraints
   - Hybrid loss function (prediction + constraint + temporal)
   - Differentiable guideline encoding

3. **Multi-Modal Explanation Framework**
   - Reasoning chains + counterfactuals + NL
   - Clinician-centered design
   - Validated with ED physicians

4. **Open-Source Infrastructure**
   - Production-ready implementation
   - Comprehensive benchmarks
   - Community-driven development

### 8.2 Expected Performance

**Target Metrics (MIMIC-IV-ED Sepsis Prediction):**

| Model | AUROC | AUPRC | Spec@90%Sens | Early Pred | Guideline Adherence |
|-------|-------|-------|--------------|------------|---------------------|
| Baseline (LSTM) | 0.78 | 0.35 | 0.45 | 3h | N/A |
| Temporal-GNN | 0.85 | 0.50 | 0.55 | 6h | N/A |
| **Hybrid (ours)** | **0.87** | **0.55** | **0.60** | **6h** | **95%+** |

**Key Differentiators:**
- **Accuracy**: Competitive with SOTA
- **Interpretability**: Vastly superior (reasoning chains, constraints)
- **Safety**: Guideline adherence prevents dangerous predictions
- **Generalizability**: Tested on multi-site data (eICU)

---

## CONCLUSION

This technical architecture demonstrates that **Hybrid Reasoning for Acute Care is eminently buildable** with current technology. The system leverages:

- **Mature frameworks**: PyTorch, DGL, Neo4j, IBM LNN
- **Public datasets**: MIMIC-IV, eICU (no data barriers)
- **Modular design**: Testable components, parallel development
- **Clear milestones**: 6-month roadmap to prototype
- **Open-source ethos**: Community validation and extension

**Why this will succeed:**
1. **Technical feasibility**: No fundamental research blockers, incremental improvements
2. **Clinical value**: Addresses real ED physician needs (explainability, trust)
3. **Scientific novelty**: Unique intersection of TKG + neuro-symbolic + clinical
4. **Reproducibility**: Public data, open code, clear documentation
5. **Team capability**: Interdisciplinary CS + Medicine collaboration at UCF

The 6-month prototype roadmap provides a concrete path from concept to working system, with clear success criteria at each milestone. This is not vaporware—it's a **research program ready for execution**.
