# Allen's Interval Algebra for Temporal Reasoning

## Overview

Allen's Interval Algebra (AIA) is a formal calculus for qualitative temporal reasoning introduced by James F. Allen in 1983. It provides a framework for representing time intervals and reasoning about their temporal relationships without relying on numerical time values. This algebra has become fundamental in artificial intelligence, temporal databases, planning systems, and increasingly in healthcare applications.

## Formal Foundation

### Mathematical Definition

Allen's interval algebra is based on the notion of relations between pairs of intervals. An interval **x** is represented as a tuple **⟨x⁻, x⁺⟩** of real numbers where **x⁻ < x⁺**, denoting the left and right endpoints of the interval respectively.

The logic Allen is formally defined as a tuple **(SymAllen, ForAllen, IntAllen, ⊨Allen)** where:
- **E** is an infinite set whose members are called events
- **R = {b, bi, m, mi, o, oi, s, si, d, di, f, fi, eq}** is the set of temporal relations disjoint from E
- Allen formulas are defined as: for all events e₁, e₂ ∈ E and all temporal relations r ∈ R, the formula is e₁ r e₂

### Key Properties

The algebra satisfies three fundamental properties:

1. **Distinct**: No pair of definite intervals can be related by more than one of the basic relationships
2. **Exhaustive**: Any pair of definite intervals is described by exactly one of the 13 relations
3. **Qualitative**: No numeric time spans are considered in the relations

Together with a converse operation, this turns Allen's interval algebra into a complete relation algebra consisting of 2¹³ = 8,192 possible unions of the basic relations.

## The 13 Basic Relations

The following table presents all 13 basic relations between two intervals X and Y:

| Relation | Symbol | Inverse | Symbol | Pictorial | Formal Definition | Example Expression |
|----------|--------|---------|--------|-----------|-------------------|-------------------|
| X before Y | < or b | X after Y | > or bi | XXX  YYY | X⁺ < Y⁻ | X < Y |
| X meets Y | m | X met-by Y | mi | XXXYYY | X⁺ = Y⁻ | X m Y |
| X overlaps Y | o | X overlapped-by Y | oi | XXX<br>&nbsp;&nbsp;YYY | X⁻ < Y⁻ < X⁺ < Y⁺ | X o Y |
| X starts Y | s | X started-by Y | si | XXX<br>YYYYY | X⁻ = Y⁻ ∧ X⁺ < Y⁺ | X s Y |
| X during Y | d | X contains Y | di | &nbsp;XXX<br>YYYYY | Y⁻ < X⁻ ∧ X⁺ < Y⁺ | X d Y |
| X finishes Y | f | X finished-by Y | fi | &nbsp;&nbsp;XXX<br>YYYYY | Y⁻ < X⁻ ∧ X⁺ = Y⁺ | X f Y |
| X equals Y | = or eq | X equals Y | = or eq | XXX<br>YYY | X⁻ = Y⁻ ∧ X⁺ = Y⁺ | X = Y |

### Detailed Relation Descriptions

#### 1. Before (b) / After (bi)
- **Definition**: X⁺ < Y⁻ (there is a gap between X and Y)
- **Interpretation**: Interval X ends before interval Y begins
- **Converse**: Y after X (Y bi X)

#### 2. Meets (m) / Met-by (mi)
- **Definition**: X⁺ = Y⁻ (endpoints touch)
- **Interpretation**: Interval X ends exactly when interval Y begins
- **Converse**: Y met-by X (Y mi X)

#### 3. Overlaps (o) / Overlapped-by (oi)
- **Definition**: X⁻ < Y⁻ < X⁺ < Y⁺
- **Interpretation**: Intervals overlap with X starting first
- **Converse**: Y overlapped-by X (Y oi X)

#### 4. Starts (s) / Started-by (si)
- **Definition**: X⁻ = Y⁻ ∧ X⁺ < Y⁺
- **Interpretation**: Both intervals start together, but X ends first
- **Converse**: Y started-by X (Y si X)

#### 5. During (d) / Contains (di)
- **Definition**: Y⁻ < X⁻ ∧ X⁺ < Y⁺
- **Interpretation**: Interval X is completely contained within interval Y
- **Converse**: Y contains X (Y di X)

#### 6. Finishes (f) / Finished-by (fi)
- **Definition**: Y⁻ < X⁻ ∧ X⁺ = Y⁺
- **Interpretation**: Both intervals end together, but Y starts first
- **Converse**: Y finished-by X (Y fi X)

#### 7. Equals (=, eq)
- **Definition**: X⁻ = Y⁻ ∧ X⁺ = Y⁺
- **Interpretation**: Both intervals have identical start and end points
- **Converse**: Self-converse (X = Y ↔ Y = X)

## Composition Table

The composition table enables transitive reasoning: if we know the relation between X and Y (R₁) and the relation between Y and Z (R₂), we can infer the possible relations between X and Z.

### Composition Operation

The composition of two IA relations R′ (between intervals I and K) and R″ (between intervals K and J) produces a new relation between intervals I and J, induced by R′ and R″.

**Example**: If I meets K (m) and K is during J (d), then I can be:
- overlaps J (o), or
- during J (d), or
- starts J (s)

Therefore: **m ⊗ d = {o, d, s}**

### Sample Composition Table Entries

| R₁ \ R₂ | < | m | o | s | d | f | = |
|---------|---|---|---|---|---|---|---|
| **<** | < | < | < | < | < | < | < |
| **m** | < | < | < | < | o,d,s | o,d,s | m |
| **o** | < | < | <,o,m | o | o,d,s | o,d,s | o |
| **d** | < | <,o,m,d,s | o,d,s | d | d | d,f,= | d |
| **f** | < | <,o,m,d,s | o,d,s | d,f,= | d,f,= | f | f |
| **=** | < | m | o | s | d | f | = |

**Full composition table**: The complete 13×13 composition table contains entries for all 169 basic relation pairs. Entries marked "full" contain all 13 relations; entries marked "con" contain {o, s, f, d, =, oi, si, fi, di}.

### Composition Properties

1. **Non-commutative**: R₁ ⊗ R₂ ≠ R₂ ⊗ R₁ in general
2. **Associative**: (R₁ ⊗ R₂) ⊗ R₃ = R₁ ⊗ (R₂ ⊗ R₃)
3. **Distributive over union**: R₁ ⊗ (R₂ ∪ R₃) = (R₁ ⊗ R₂) ∪ (R₁ ⊗ R₃)

### Calculating Arbitrary Compositions

For arbitrary relations R₁ and R₂ (which may be unions of basic relations):
**R₁ ⊗ R₂ = ⋃{rᵢ ⊗ rⱼ | rᵢ ∈ R₁, rⱼ ∈ R₂}**

where rᵢ and rⱼ are basic relations.

## Computational Complexity and Tractability

### NP-Completeness

The satisfiability problem for Allen's interval algebra is **NP-complete** when using the full set of 8,192 relations. This means determining whether a given set of temporal constraints is consistent requires exponential time in the worst case.

### Tractable Subclasses

Researchers have identified **18 maximal tractable subalgebras** of Allen's interval algebra. Every tractable subset of the full algebra is contained within at least one of these 18 subalgebras. All 18 are closed under:
- Composition (⊗)
- Converse (⁻¹)
- Intersection (∩)

### The H (Horn) Subalgebra

The most important tractable subalgebra is the **H (Horn) subalgebra**:
- It is the only one containing all 13 basic relations
- Inference can be performed using the **path consistency algorithm**
- Provides polynomial-time reasoning for a significant subset of temporal constraints
- Particularly useful for practical applications requiring complete expressiveness with efficient reasoning

### Path Consistency Algorithm

For tractable subclasses, the path consistency algorithm works as follows:

1. Start with a network of temporal constraints
2. For each triple of intervals (I, J, K):
   - Compute the composition: R(I,K) through J = R(I,J) ⊗ R(J,K)
   - Intersect with current R(I,K): R'(I,K) = R(I,K) ∩ [R(I,J) ⊗ R(J,K)]
   - Update R(I,K) with R'(I,K)
3. Repeat until no changes occur or inconsistency detected
4. If any relation becomes empty, the network is inconsistent

## Clinical and Healthcare Applications

### Temporal Reasoning in Healthcare

Allen's interval algebra has found significant applications in healthcare and clinical settings for:

1. **Electronic Health Records (EHR)**: Reasoning about the temporal ordering of clinical events, diagnoses, treatments, and outcomes
2. **Clinical Decision Support**: Inferring temporal relationships between symptoms, interventions, and patient responses
3. **Treatment Planning**: Scheduling and coordinating multiple concurrent therapies with temporal dependencies
4. **Medical Protocols**: Representing and verifying temporal constraints in clinical guidelines and care pathways
5. **Disease Progression Modeling**: Tracking and reasoning about the temporal evolution of disease states

### Clinical Examples

#### Example 1: Post-Operative Monitoring

Consider a post-surgical patient with three temporal intervals:

- **S**: Surgery interval (08:00-12:00)
- **A**: Anesthesia recovery (12:00-14:00)
- **P**: Pain medication administration (13:00-18:00)

**Relations**:
- S **meets** A (surgery ends exactly when recovery begins)
- A **during** P (recovery period is contained within pain medication coverage)
- S **before** end of P (surgery completes before medication ends)

**Composition reasoning**:
- Given: S m A and A d P
- Infer: S (m ⊗ d) P = {o, d, s} P
- Therefore: Surgery either overlaps with, is during, or starts the pain medication period

**Clinical significance**: This reasoning ensures pain management begins during or before recovery completion.

#### Example 2: Antibiotic Therapy and Infection

Consider intervals representing:

- **I**: Infection period (Day 1-7)
- **A₁**: First antibiotic course (Day 2-6)
- **A₂**: Second antibiotic course (Day 6-10)

**Relations**:
- I **started-by** A₁ is false (infection starts Day 1, antibiotic Day 2)
- A₁ **during** I (first course entirely within infection period)
- A₁ **meets** A₂ (sequential antibiotic courses with no gap)
- I **overlaps** A₂ (infection period overlaps second course)

**Composition reasoning**:
- Given: A₁ d I and I o A₂
- Infer: A₁ (d ⊗ o) A₂
- From composition table: d ⊗ o = {<, o, m, d, s}
- Combined with known A₁ m A₂, consistency verified

**Clinical significance**: Validates the treatment plan where sequential antibiotics span and extend beyond the infection period.

#### Example 3: Vital Signs Monitoring

Consider a patient in acute care:

- **H**: Hypotensive episode (10:15-10:45)
- **F**: IV fluid bolus administration (10:20-10:50)
- **N**: Return to normal blood pressure (10:50-11:00)

**Relations**:
- F **started-by** H is false; H **overlaps** F (hypotension detected, then fluids started)
- F **meets** N (fluid administration ends when normalization begins)
- H **overlaps** F (intervention starts during the problem)

**Composition reasoning**:
- Given: H o F and F m N
- Infer: H (o ⊗ m) N = {<, o, m, d, s} N
- Clinically: H < N (hypotensive episode before normalization)

**Clinical significance**: Establishes temporal causal chain: problem → intervention → resolution.

#### Example 4: Medication Interaction Window

Consider drug administration:

- **D₁**: Drug A administration (08:00-08:05)
- **E₁**: Drug A effective period (08:05-12:05)
- **D₂**: Drug B administration (10:00-10:05)
- **E₂**: Drug B effective period (10:05-14:05)

**Relations**:
- D₁ **meets** E₁ (drug takes effect immediately after administration)
- D₂ **meets** E₂
- D₁ **before** D₂
- E₁ **overlaps** E₂ (overlapping effective periods)

**Composition reasoning**:
- Given: D₁ m E₁ and E₁ o E₂ and E₂ mi D₂
- Infer: D₁ (m ⊗ o ⊗ mi) D₂
- Composition: (m ⊗ o) = {<, o, m, d, s}; then {<, o, m, d, s} ⊗ mi = full set
- Needs additional constraint: D₁ < D₂

**Clinical significance**: Identifies overlapping therapeutic windows for drug interaction checking.

#### Example 5: Sepsis Protocol Timeline

A patient with suspected sepsis has protocol-driven intervals:

- **T**: Triage assessment (00:00-00:15)
- **L**: Lactate measurement (00:15-00:20)
- **B**: Blood culture collection (00:10-00:25)
- **A**: Antibiotic administration (00:30-00:35)

**Relations**:
- T **meets** L (lactate drawn immediately after triage)
- T **overlaps** B (blood cultures overlap with triage)
- L **before** A (lactate before antibiotics)
- B **before** A (cultures before antibiotics - critical for protocol adherence)

**Protocol verification**:
- Sepsis-3 guidelines require antibiotics within 1 hour of recognition
- Cultures must precede antibiotics
- Relations verify correct temporal sequencing

**Clinical significance**: Validates adherence to time-sensitive sepsis protocols using qualitative temporal reasoning.

#### Example 6: ICU Weaning Protocol

Consider mechanical ventilation weaning:

- **V**: Full ventilator support (Day 1-3)
- **S**: Spontaneous breathing trial (Day 3, 2 hours)
- **W**: Weaning mode (Day 3-5)
- **E**: Extubation (Day 5)

**Relations**:
- V **finished-by** S (spontaneous trial at end of full support)
- S **meets** W (weaning begins immediately after successful trial)
- W **meets** E (extubation immediately after weaning completion)

**Composition reasoning**:
- Given: S f V and S m W
- Infer: V (fi ⊗ m) W = {<, o, m, d, s} W
- With additional knowledge V < W, we get V **meets** W

**Clinical significance**: Ensures proper temporal progression through ventilator weaning stages.

## Integration with Knowledge Graphs

### Temporal Knowledge Graphs (TKGs)

Allen's interval algebra serves as a foundation for temporal reasoning in knowledge graphs. Modern TKGs incorporate time information beyond simple timestamps by using interval relations to represent:

- **Event durations**: Start and end times of clinical events
- **State changes**: Transitions in patient conditions
- **Causal sequences**: Temporal ordering of interventions and outcomes
- **Concurrent processes**: Overlapping therapeutic interventions

### Representation Learning

Temporal knowledge graph representation learning aims to embed entities, relations, and temporal information into low-dimensional vector spaces while preserving Allen's algebraic properties:

- **Entity embeddings**: Patient, diagnosis, procedure, medication entities
- **Relation embeddings**: Clinical relationships (treats, causes, contraindicates)
- **Temporal embeddings**: Interval relations from Allen's algebra
- **Compositional reasoning**: Leveraging composition tables for link prediction

### Practical Frameworks

#### Graphiti

Graphiti is a modern framework for building temporally-aware knowledge graphs with:

- **Bi-temporal data model**: Tracks both event occurrence time and data ingestion time
- **Point-in-time queries**: Retrieve knowledge graph state at any historical moment
- **Incremental updates**: Add new temporal relations without full graph recomputation
- **Hybrid retrieval**: Combines semantic, keyword, and graph traversal methods

#### Implementation Approaches

For clinical applications, temporal knowledge graphs using Allen's algebra typically:

1. **Discrete time points**: Use versioned snapshots rather than continuous time for tractability
2. **Domain ontologies**: Leverage standardized medical ontologies (SNOMED CT, LOINC)
3. **Interval endpoints**: Explicitly represent interval start/end events as graph nodes
4. **Composition rules**: Encode Allen's composition table as graph reasoning rules
5. **Uncertainty handling**: Extend basic relations to handle uncertain temporal bounds

## OWL-Time Ontology Integration

### OWL-Time Standard

The W3C **OWL-Time** ontology provides a formal OWL-2 DL vocabulary for temporal concepts, directly implementing Allen's interval algebra:

**Namespace**: `http://www.w3.org/2006/time#`
**Prefix**: `time:`

### Core Temporal Entities

- **time:Instant**: A zero-duration temporal entity (point in time)
- **time:Interval**: A temporal entity with extent (duration)
- **time:TemporalEntity**: Abstract superclass of Instant and Interval
- **time:TemporalAggregate**: Collection of multiple temporal entities

### Allen Relations in OWL-Time

OWL-Time directly implements all 13 Allen relations as object properties:

#### General Temporal Relations
- **time:before**: General before relation
- **time:after**: General after relation

#### Interval-Specific Relations
- **time:intervalBefore** (corresponds to Allen's <)
- **time:intervalAfter** (corresponds to Allen's >)
- **time:intervalMeets** (corresponds to Allen's m)
- **time:intervalMetBy** (corresponds to Allen's mi)
- **time:intervalOverlaps** (corresponds to Allen's o)
- **time:intervalOverlappedBy** (corresponds to Allen's oi)
- **time:intervalStarts** (corresponds to Allen's s)
- **time:intervalStartedBy** (corresponds to Allen's si)
- **time:intervalDuring** (corresponds to Allen's d)
- **time:intervalContains** (corresponds to Allen's di)
- **time:intervalFinishes** (corresponds to Allen's f)
- **time:intervalFinishedBy** (corresponds to Allen's fi)
- **time:intervalEquals** (corresponds to Allen's =)

### OWL-Time Extensions

#### Entity Relations Extension

Adds four supplementary relations:
- **time:equals**: Equality of temporal entities
- **time:hasInside**: Generalized containment
- **time:disjoint**: Non-overlapping temporal entities
- **time:notDisjoint**: Overlapping or touching temporal entities

#### Temporal Aggregates Extension

Supports grouping of multiple temporal entities:
- **time:TemporalAggregate**: Composed of explicit sets of temporal entities
- Useful for representing recurrent events or discontinuous periods

### Clinical OWL-Time Example

```turtle
@prefix time: <http://www.w3.org/2006/time#> .
@prefix ex: <http://example.org/clinical#> .

# Define intervals
ex:Surgery a time:Interval ;
    time:hasBeginning ex:SurgeryStart ;
    time:hasEnd ex:SurgeryEnd .

ex:Recovery a time:Interval ;
    time:hasBeginning ex:RecoveryStart ;
    time:hasEnd ex:RecoveryEnd .

ex:Medication a time:Interval ;
    time:hasBeginning ex:MedStart ;
    time:hasEnd ex:MedEnd .

# Define Allen relations
ex:Surgery time:intervalMeets ex:Recovery .
ex:Recovery time:intervalDuring ex:Medication .

# Composed relation can be inferred
# Surgery (meets ⊗ during) Medication
# Results in: Surgery {overlaps, during, starts} Medication
```

### Integration with Clinical Ontologies

OWL-Time can be integrated with:

- **SNOMED CT**: Clinical terminology
- **LOINC**: Laboratory and clinical observations
- **HL7 FHIR**: Healthcare interoperability resources
- **Time-indexed EHR**: Temporal electronic health record modeling

## Implementation Considerations

### Software Libraries

**Java**:
- Interval tree implementations with Allen's algebra operations
- Jena framework for OWL-Time reasoning

**Python**:
- QSRlib: Qualitative spatial-temporal reasoning library
- RDFlib: OWL-Time ontology processing

**Prolog**:
- Native support for constraint logic programming
- Efficient implementation of composition table reasoning

**Answer Set Programming (ASP)**:
- ASP(DL): ASP extended with difference constraints
- Novel encoding for interval algebra satisfaction problems

### Performance Optimization

For large-scale clinical applications:

1. **Restrict to tractable subclass**: Use H (Horn) subalgebra when possible
2. **Path consistency**: Apply polynomial-time path consistency algorithm
3. **Indexing**: Use interval trees for efficient interval overlap queries
4. **Caching**: Precompute common composition table lookups
5. **Decomposition**: Break large temporal networks into smaller independent components

### Uncertainty and Fuzziness

Clinical temporal data often involves uncertainty. Extensions include:

**Fuzzy Allen Relations**:
- Assign membership degrees to relation memberships
- Represent vague temporal boundaries (e.g., "approximately during")
- Useful for patient-reported timelines with imprecise recall

**Probabilistic Extensions**:
- Assign probability distributions over possible relations
- Handle measurement uncertainty in event timestamps
- Support Bayesian inference over temporal networks

## Research Directions

### Current Applications

1. **Planning and Scheduling**: Project management, manufacturing, robotics
2. **Temporal Databases**: Query languages with temporal operators
3. **Healthcare AI**: Clinical decision support, protocol verification, outcome prediction
4. **Smart Environments**: Ambient assisted living for elderly care
5. **Natural Language Processing**: Temporal information extraction and reasoning

### Elderly Care Research

Ongoing research focuses on "synergy of qualitative spatio-temporal reasoning and smart environments for assisting the elderly at home":

- Combining Allen's temporal algebra with qualitative spatial reasoning
- Activity recognition using temporal patterns
- Fall detection and emergency response coordination
- Medication adherence monitoring with temporal constraints

### Future Challenges

1. **Scalability**: Reasoning over millions of temporal intervals in large EHR systems
2. **Real-time reasoning**: Continuous temporal inference for ICU monitoring
3. **Multi-modal integration**: Combining temporal, spatial, and causal reasoning
4. **Explainability**: Generating human-readable explanations of temporal inferences
5. **Standardization**: Wider adoption of OWL-Time in clinical information systems

## Summary

Allen's interval algebra provides a rigorous mathematical foundation for qualitative temporal reasoning with significant applications in healthcare. Its 13 basic relations, composition table, and tractable subclasses enable efficient reasoning about complex temporal constraints in clinical settings. Integration with knowledge graphs through OWL-Time and modern TKG frameworks makes Allen's algebra increasingly relevant for AI-driven clinical decision support, protocol verification, and patient timeline analysis.

The algebra's qualitative nature—reasoning without requiring precise numerical timestamps—aligns well with the realities of clinical documentation where exact times may be unknown or imprecise. As healthcare systems increasingly adopt knowledge graph technologies and temporal reasoning capabilities, Allen's interval algebra serves as a proven, well-understood formalism for representing and reasoning about temporal clinical data.

## References

### Primary Sources

- [Allen's interval algebra - Wikipedia](https://en.wikipedia.org/wiki/Allen's_interval_algebra)
- [Allen's Interval Algebra - UCI](https://ics.uci.edu/~alspaugh/cls/shr/allen.html)
- [Allen's Interval Algebra - QSRlib Documentation](https://qsrlib.readthedocs.io/en/latest/rsts/handwritten/qsrs/allen.html)
- [Knowledge Representation and Reasoning: Allen's temporal algebra - EMSE](https://www.emse.fr/~zimmermann/Teaching/KRR/allen.html)

### Academic Research

- [Reasoning about temporal relations: a maximal tractable subclass of Allen's interval algebra - ACM](https://dl.acm.org/doi/10.1145/200836.200848)
- [Reasoning about temporal relations: The tractable subalgebras of Allen's interval algebra - ACM](https://dl.acm.org/doi/10.1145/876638.876639)
- [Allen's Interval Algebra Makes the Difference - arXiv](https://arxiv.org/abs/1909.01128)
- [Allen's Interval Algebra Makes the Difference - Springer](https://link.springer.com/chapter/10.1007/978-3-030-46714-2_6)
- [Interval Algebra - ScienceDirect Topics](https://www.sciencedirect.com/topics/computer-science/interval-algebra)

### Temporal Knowledge Graphs

- [Temporal Reasoning Over Event Knowledge Graphs - University of Florida](https://dsr.cise.ufl.edu/wp-content/uploads/2018/01/TemporalReasoning_KBCOM.pdf)
- [Temporal knowledge graphs reasoning with iterative guidance by temporal logical rules - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0020025522013871)
- [Semantic Reasoning Technology on Temporal Knowledge Graph - Springer](https://link.springer.com/chapter/10.1007/978-3-031-20309-1_10)
- [Temporal Knowledge Graphs: Uncovering Hidden Patterns - Senzing](https://senzing.com/gph3-temporal-knowledge-graphs/)
- [Harnessing Temporal Dynamics: Advanced Reasoning using Temporal Knowledge Graphs - Medium](https://medium.com/@researchgraph/harnessing-temporal-dynamics-advanced-reasoning-using-temporal-knowledge-graphs-0da8483262f9)
- [Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs - PMLR](https://proceedings.mlr.press/v70/trivedi17a/trivedi17a.pdf)
- [A Survey on Temporal Knowledge Graph: Representation Learning and Applications - arXiv](https://arxiv.org/abs/2403.04782)

### OWL-Time Ontology

- [Time Ontology in OWL - W3C](https://www.w3.org/TR/owl-time/)
- [Extensions to the OWL-Time Ontology - entity relations - W3C](https://w3c.github.io/sdw/time-entity-relations/)
- [Extensions to the OWL-Time Ontology - temporal aggregates - W3C](https://www.w3.org/2021/sdw/time-aggregates/)
- [Time Ontology in OWL Standard - OGC](https://www.ogc.org/publications/standard/time-ontology-in-owl/)

### Tools and Frameworks

- [Graphiti: Build Real-Time Knowledge Graphs for AI Agents - GitHub](https://github.com/getzep/graphiti)
- [Graphiti: Knowledge Graph Memory for an Agentic World - Neo4j](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)
- [A Temporal Knowledge Graph Generation Dataset Supervised Distantly by Large Language Models - Nature](https://www.nature.com/articles/s41597-025-05062-0)

---

*Document created: 2025-11-30*
*Location: /Users/alexstinard/hybrid-reasoning-acute-care/research/allen_temporal_algebra.md*
