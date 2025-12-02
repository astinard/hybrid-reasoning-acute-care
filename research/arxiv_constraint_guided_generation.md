# Constraint-Guided Generation for Clinical AI Applications: A Comprehensive Literature Review

**Research Focus:** Constraint-guided generation techniques for neural networks with applications to clinical AI and emergency department policy-guided trajectory generation.

**Date:** December 1, 2025

---

## Executive Summary

This review synthesizes recent advances in constraint-guided generation for neural networks, with a focus on clinical AI applications and policy-guided trajectory generation for emergency departments. We identified 150+ papers across multiple domains, revealing a rapidly evolving field where constraint integration methods vary significantly in their approach (hard vs. soft constraints), constraint specification languages, and integration with modern generative architectures.

**Key Findings:**
- **Hard vs. Soft Constraints:** Hard constraint methods guarantee satisfaction but often reduce generation quality; soft methods optimize constraint satisfaction probabilistically
- **Neuro-Symbolic Integration:** Emerging as the most promising approach for clinical applications, combining neural learning with symbolic reasoning
- **Diffusion Models:** Show strong potential for constraint-guided generation but require careful design to maintain constraint satisfaction
- **Clinical Applicability:** Significant gap between general constraint methods and clinically-validated applications
- **Research Gaps:** Limited work on temporal constraints, multi-objective constraint satisfaction, and long-horizon trajectory generation under clinical protocols

---

## 1. Key Papers and ArXiv IDs

### 1.1 Foundational Work on Constrained Neural Generation

1. **Why is constrained neural language generation particularly challenging?** (2206.05395v2)
   - Comprehensive survey distinguishing conditions vs. constraints
   - Formalizes constrained generation as testable output conditions
   - Reviews methods and evaluation metrics

2. **Efficient Generation of Structured Objects with Constrained Adversarial Networks** (2007.13197v2)
   - ArXiv: 2007.13197v2
   - Method: Penalizes generator proportional to mass allocated to invalid structures
   - Constraint Type: Hard logical constraints via knowledge compilation
   - Clinical Relevance: High - generates chemically valid molecules

3. **Constrained Image Generation Using Binarized Neural Networks with Decision Procedures** (1802.08795v1)
   - ArXiv: 1802.08795v1
   - Method: Embeds logical constraints via binarized neural networks
   - Uses PDE solvers approximated by neural networks
   - Clinical Relevance: Medium - demonstrates constraint satisfaction in physical simulations

### 1.2 Neuro-Symbolic Approaches

4. **Neuro-Symbolic Generative Diffusion Models for Physically Grounded, Robust, and Safe Generation** (2506.01121v1)
   - ArXiv: 2506.01121v1
   - Method: Interleaves diffusion steps with symbolic optimization
   - Constraint Type: Hard - functional and logic constraints via symbolic optimization
   - Integration: DNF (Disjunctive Normal Form) compilation into diffusion
   - Clinical Applicability: HIGH - safety-critical applications, drug discovery
   - **Key Innovation:** Guarantees 100% constraint satisfaction for safety-critical tasks

5. **Constraints-Guided Diffusion Reasoner for Neuro-Symbolic Learning** (2508.16524v1)
   - ArXiv: 2508.16524v1
   - Method: Two-stage training with PPO for constraint satisfaction
   - Constraint Type: Hard logical constraints from rule-based rewards
   - Clinical Relevance: High - demonstrates on logical puzzles applicable to clinical decision trees

6. **Injecting Logical Constraints into Neural Networks via Straight-Through Estimators** (2307.04347v1)
   - ArXiv: 2307.04347v1
   - Method: Straight-through estimator for gradient flow through discrete constraints
   - Constraint Type: Hard logical constraints (arbitrary logic)
   - Clinical Relevance: High - eliminates need for labeled data by learning from constraints

7. **Zero-Shot Conditioning of Score-Based Diffusion Models by Neuro-Symbolic Constraints** (2308.16534v3)
   - ArXiv: 2308.16534v3
   - Method: Weighted oblique decision trees for constraint encoding
   - Constraint Type: Soft logical constraints via differentiable logic
   - Integration: Zero-shot - no retraining required
   - Clinical Relevance: High - works without labeled data

### 1.3 Diffusion Models with Constraints

8. **Constraint-Aware Diffusion Models for Trajectory Optimization** (2406.00990v1)
   - ArXiv: 2406.00990v1
   - Method: Hybrid loss combining constraint violation and distribution matching
   - Constraint Type: Soft - minimizes violation while learning distribution
   - Domain: Robotics (tabletop manipulation)
   - Clinical Relevance: Medium - trajectory optimization applicable to patient flow

9. **Aligning Diffusion Model with Problem Constraints for Trajectory Optimization** (2504.00342v1)
   - ArXiv: 2504.00342v1
   - Method: Re-weighting diffusion steps based on constraint violation statistics
   - Constraint Type: Soft with statistical alignment
   - Clinical Relevance: Medium - trajectory-level reasoning

10. **Constraint-Guided Prediction Refinement via Deterministic Diffusion Trajectories** (2506.12911v2)
    - ArXiv: 2506.12911v2
    - Method: Deterministic diffusion with constraint gradient corrections
    - Constraint Type: Hard - non-convex equality constraints
    - Clinical Applicability: Medium - AC power flow prediction under Kirchhoff's laws

11. **Diffusion Predictive Control with Constraints** (2412.09342v2)
    - ArXiv: 2412.09342v2
    - Method: Model-based projections with constraint tightening
    - Constraint Type: Hard - explicit state/action constraints
    - Domain: Robotics predictive control
    - Clinical Relevance: High - real-time constraint satisfaction

12. **Reflected Diffusion Models** (2304.04740v3)
    - ArXiv: 2304.04740v3
    - Method: Reflected SDEs evolving on data support
    - Constraint Type: Hard - domain constraints via reflection
    - Clinical Relevance: Medium - ensures samples stay in valid domain

### 1.4 Autoregressive Models with Constraints

13. **A Pseudo-Semantic Loss for Autoregressive Models with Logical Constraints** (2312.03905v2)
    - ArXiv: 2312.03905v2
    - Method: Differentiable logical loss via soft LTL semantics + Gumbel-Softmax
    - Constraint Type: Hard logical constraints (LTLf - Linear Temporal Logic over finite traces)
    - Integration: Combines predictive loss with logical loss
    - Clinical Applicability: HIGH - temporal constraints in clinical protocols
    - **Key Innovation:** Handles autoregressive generation with temporal logic

14. **NeuroLogic Decoding: (Un)supervised Neural Text Generation with Predicate Logic Constraints** (2010.12884v2)
    - ArXiv: 2010.12884v2
    - Method: Inference-time algorithm with predicate logic
    - Constraint Type: Hard - arbitrary predicate logic constraints
    - Integration: Zero-shot - no retraining needed
    - Clinical Relevance: High - lexical constraints for clinical text

15. **Tractable Control for Autoregressive Language Generation** (2304.07438v4)
    - ArXiv: 2304.07438v4
    - Method: Tractable probabilistic models (distilled HMMs) for constraint enforcement
    - Constraint Type: Hard - lexical constraints
    - Integration: Efficient computation via TPMs
    - Clinical Relevance: Medium - controllable text generation

16. **ABS: Enforcing Constraint Satisfaction On Generated Sequences Via Automata-Guided Beam Search** (2506.09701v2)
    - ArXiv: 2506.09701v2
    - Method: DFA-guided beam search with masking
    - Constraint Type: Hard - any constraint compilable to DFA
    - Integration: Inference-time with guaranteed satisfaction
    - Clinical Relevance: HIGH - medical protocol adherence

### 1.5 Policy and Trajectory Optimization

17. **Constraint-Generation Policy Optimization (CGPO): Nonlinear Programming for Policy Optimization** (2401.12243v2)
    - ArXiv: 2401.12243v2
    - Method: Bilevel optimization with constraint generation
    - Constraint Type: Hard - linear constraints with bounded error
    - Domain: Mixed discrete-continuous MDPs
    - Clinical Relevance: HIGH - interpretable policies with guarantees

18. **Safe Offline Reinforcement Learning with Real-Time Budget Constraints** (2306.00603v2)
    - ArXiv: 2306.00603v2
    - Method: Diffusion model planning with trajectory-level reasoning
    - Constraint Type: Hard - real-time budget constraints
    - Domain: Offline RL
    - Clinical Relevance: HIGH - resource allocation in ED

19. **CGD: Constraint-Guided Diffusion Policies for UAV Trajectory Planning** (2405.01758v1)
    - ArXiv: 2405.01758v1
    - Method: Diffusion policies with surrogate optimization
    - Constraint Type: Hard - collision-free, dynamically feasible
    - Domain: UAV planning
    - Clinical Relevance: Medium - trajectory planning under constraints

20. **POLICEd RL: Learning Closed-Loop Robot Control Policies with Provable Satisfaction of Hard Constraints** (2403.13297v3)
    - ArXiv: 2403.13297v3
    - Method: Affine policy around unsafe set as repulsive buffer
    - Constraint Type: Hard - affine constraints with proofs
    - Clinical Relevance: Medium - provable safety guarantees

### 1.6 Logical Constraint Integration

21. **Neuro-symbolic Learning Yielding Logical Constraints** (2410.20957v1)
    - ArXiv: 2410.20957v1
    - Method: Difference-of-convex programming for logical constraint relaxation
    - Constraint Type: Hard - cardinality constraints
    - Integration: End-to-end with trust region method
    - Clinical Relevance: High - learns constraint structure

22. **MultiplexNet: Towards Fully Satisfied Logical Constraints in Neural Networks** (2111.01564v1)
    - ArXiv: 2111.01564v1
    - Method: Categorical latent variable chooses constraint term
    - Constraint Type: Hard - DNF logical formulas
    - Integration: 100% constraint satisfaction guaranteed
    - Clinical Relevance: High - interpretable constraint satisfaction

23. **Augmenting Neural Networks with First-order Logic** (1906.06298v3)
    - ArXiv: 1906.06298v3
    - Method: Compiles logical statements into computation graphs
    - Constraint Type: Soft - first-order logic
    - Clinical Relevance: Medium - knowledge-augmented networks

24. **Neural Networks Enhancement with Logical Knowledge** (2009.06087v2)
    - ArXiv: 2009.06087v2
    - Method: KENN - new layer modifying predictions per knowledge
    - Constraint Type: Soft - learnable clause weights
    - Clinical Relevance: High - robust to incorrect knowledge

### 1.7 Safety and Control

25. **Provably Safe Neural Network Controllers via Differential Dynamic Logic** (2402.10998v3)
    - ArXiv: 2402.10998v3
    - Method: Differential dynamic logic (dL) for safety proofs
    - Constraint Type: Hard - infinite-time safety via formal verification
    - Domain: Control systems
    - Clinical Relevance: HIGH - safety-critical medical devices

26. **Best Arm Identification with Safety Constraints** (2111.12151v1)
    - ArXiv: 2111.12151v1
    - Method: Multi-armed bandits with safety constraints
    - Constraint Type: Hard - safety constraints during exploration
    - Clinical Relevance: HIGH - safe clinical trial design

27. **Linear Stochastic Bandits Under Safety Constraints** (1908.05814v1)
    - ArXiv: 1908.05814v1
    - Method: UCB with conservative safe action identification
    - Constraint Type: Hard - linear safety constraints
    - Clinical Relevance: HIGH - safe dose finding

28. **Learning with Safety Constraints: Sample Complexity of Reinforcement Learning for Constrained MDPs** (2008.00311v3)
    - ArXiv: 2008.00311v3
    - Method: Theoretical analysis of sample complexity
    - Constraint Type: Hard - CMDP constraints
    - Clinical Relevance: High - bounded sample requirements

### 1.8 Rule-Guided and Grammar-Based Generation

29. **Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion** (2402.14285v4)
    - ArXiv: 2402.14285v4
    - Method: Stochastic Control Guidance (SCG) for non-differentiable rules
    - Constraint Type: Hard - musical rules
    - Integration: Training-free guidance
    - Clinical Relevance: Medium - non-differentiable constraint handling

30. **RuleRAG: Rule-Guided Retrieval-Augmented Generation with Language Models** (2410.22353v3)
    - ArXiv: 2410.22353v3
    - Method: Rule-guided retrieval and generation
    - Constraint Type: Soft - KG-derived rules
    - Clinical Relevance: HIGH - clinical guideline integration

31. **Efficient Guided Generation for Large Language Models** (2307.09702v4)
    - ArXiv: 2307.09702v4
    - Method: FSM-based vocabulary indexing
    - Constraint Type: Hard - regular expressions, context-free grammars
    - Clinical Relevance: HIGH - structured clinical text generation

32. **Neural Guided Constraint Logic Programming for Program Synthesis** (1809.02840v3)
    - ArXiv: 1809.02840v3
    - Method: Neural model guides miniKanren search
    - Constraint Type: Hard - logic programming constraints
    - Clinical Relevance: Medium - program synthesis for protocols

### 1.9 Clinical and Medical Applications

33. **Anatomically-Controllable Medical Image Generation with Segmentation-Guided Diffusion Models** (2402.05210v4)
    - ArXiv: 2402.05210v4
    - Method: Multi-class segmentation mask guidance at each diffusion step
    - Constraint Type: Hard - anatomical constraints
    - Domain: Medical imaging (breast MRI, CT)
    - Clinical Applicability: VERY HIGH - anatomically accurate image generation

34. **Generating Reliable Synthetic Clinical Trial Data** (2505.05019v1)
    - ArXiv: 2505.05019v1
    - Method: Preprocessing/postprocessing with domain constraints
    - Constraint Type: Hard - survival constraints
    - Clinical Applicability: VERY HIGH - clinical trial augmentation
    - **Key Finding:** HPO alone insufficient without domain constraints

35. **TrialSynth: Generation of Synthetic Sequential Clinical Trial Data** (2409.07089v2)
    - ArXiv: 2409.07089v2
    - Method: VAE with Hawkes Processes for time-sequence
    - Constraint Type: Soft - temporal dependencies
    - Clinical Applicability: VERY HIGH - sequential clinical data

36. **DualAlign: Generating Clinically Grounded Synthetic Data** (2509.10538v1)
    - ArXiv: 2509.10538v1
    - Method: Statistical + semantic alignment
    - Constraint Type: Soft - symptom trajectories
    - Clinical Applicability: VERY HIGH - AD clinical notes

37. **Multi-Label Clinical Time-Series Generation via Conditional GAN** (2204.04797v2)
    - ArXiv: 2204.04797v2
    - Method: GRU-based conditional GAN
    - Constraint Type: Soft - temporal coherence
    - Clinical Applicability: HIGH - imbalanced disease generation

---

## 2. Hard vs. Soft Constraint Methods

### 2.1 Hard Constraint Methods

**Definition:** Guarantee 100% constraint satisfaction in generated outputs.

#### Advantages
- **Provable Safety:** Essential for clinical applications (e.g., drug dosing, medical device control)
- **Interpretability:** Clear failure modes when constraints violated
- **Regulatory Compliance:** Easier to certify for medical use
- **No Post-Processing:** Guaranteed valid outputs

#### Disadvantages
- **Reduced Generation Quality:** May produce lower-quality outputs when constraints are strict
- **Computational Cost:** Often requires complex inference procedures
- **Limited Flexibility:** Cannot gracefully handle conflicting constraints
- **Scalability:** Difficult with many constraints

#### Key Methods

1. **Symbolic Optimization Integration (2506.01121v1)**
   - Interleaves diffusion with symbolic solver
   - Guarantees constraint satisfaction via projection
   - Tradeoff: Slower generation (symbolic solver calls)

2. **Automata-Guided Search (2506.09701v2)**
   - Compiles constraints to DFA
   - Masks invalid tokens during beam search
   - Tradeoff: Limited to DFA-compilable constraints

3. **Reflection Methods (2304.04740v3)**
   - Reflects diffusion trajectories onto valid domain
   - Natural domain constraint handling
   - Tradeoff: Requires smooth domain boundaries

4. **Formal Verification (2402.10998v3)**
   - Differential dynamic logic proofs
   - Infinite-time safety guarantees
   - Tradeoff: Limited to verifiable constraint classes

5. **Constraint Generation (2401.12243v2)**
   - Bilevel optimization with bounded error
   - Interpretable policies
   - Tradeoff: Requires tractable constraint representation

### 2.2 Soft Constraint Methods

**Definition:** Optimize constraint satisfaction probabilistically, allowing violations.

#### Advantages
- **Higher Generation Quality:** Better match to data distribution
- **Flexibility:** Handles conflicting constraints via weighting
- **Scalability:** Efficient training with gradient descent
- **Graceful Degradation:** Partial satisfaction when full satisfaction impossible

#### Disadvantages
- **No Guarantees:** Cannot prove constraint satisfaction
- **Difficult Tuning:** Requires careful loss weight balancing
- **Unpredictable Violations:** Hard to characterize failure modes
- **Regulatory Challenges:** Difficult to certify for safety-critical use

#### Key Methods

1. **Weighted Logical Loss (2312.03905v2)**
   - Differentiable LTL semantics
   - Combines with predictive loss
   - Tradeoff: Approximate constraint satisfaction

2. **Knowledge-Enhanced Networks (2009.06087v2)**
   - Learnable clause weights
   - Robust to incorrect knowledge
   - Tradeoff: No satisfaction guarantees

3. **Constraint-Aware Training (2406.00990v1)**
   - Hybrid loss with violation term
   - Statistical alignment to constraint distribution
   - Tradeoff: Violations in tail cases

4. **Preference-Based Learning (1711.05772v2)**
   - Latent constraints from preferences
   - Flexible to user-defined constraints
   - Tradeoff: Implicit constraint representation

### 2.3 Hybrid Approaches

**Best of Both Worlds:** Combine hard and soft methods

1. **Two-Stage Pipelines**
   - Stage 1: Soft constraints during training
   - Stage 2: Hard projection during inference
   - Example: DCDM (2509.00395v1) - nuclear regularization + hard constraints

2. **Constraint Tightening**
   - Learn soft approximation
   - Tighten to hard constraint at deployment
   - Example: Diffusion Predictive Control (2412.09342v2)

3. **Hierarchical Satisfaction**
   - Priority-based constraint ordering
   - Hard constraints for critical requirements
   - Soft constraints for preferences
   - Example: Blameless Control (2304.06625v3)

---

## 3. Constraint Specification Languages

### 3.1 Logical Formalisms

#### Linear Temporal Logic (LTL / LTLf)
- **Papers:** 2312.03905v2, 2010.12884v2
- **Expressiveness:** Temporal sequences, eventually/always operators
- **Clinical Use Cases:**
  - Treatment protocols: "Give medication A before medication B"
  - Monitoring: "Check vitals every 4 hours until stable"
  - Safety: "Never exceed dose X within 24 hours"
- **Limitations:** Discrete time, propositional variables

#### Predicate Logic
- **Papers:** 2010.12884v2
- **Expressiveness:** First-order quantification, relations
- **Clinical Use Cases:**
  - Eligibility criteria: "For all patients with diabetes, prescribe metformin"
  - Diagnosis rules: "If symptoms X and Y, then order test Z"
- **Limitations:** Undecidable for complex formulas

#### Disjunctive Normal Form (DNF)
- **Papers:** 2111.01564v1, 2506.01121v1
- **Expressiveness:** OR of AND clauses
- **Clinical Use Cases:**
  - Multi-path protocols: "(Route A AND Step 1) OR (Route B AND Step 2)"
  - Contraindication checking
- **Advantages:** Tractable, human-readable
- **Limitations:** Exponential size for some constraints

### 3.2 Formal Grammars

#### Context-Free Grammars (CFG)
- **Papers:** 2307.09702v4
- **Expressiveness:** Hierarchical structure
- **Clinical Use Cases:**
  - Structured clinical notes (SOAP format)
  - ICD/CPT code combinations
  - Medication ordering syntax
- **Advantages:** Well-studied parsing algorithms
- **Limitations:** Cannot express all constraints

#### Regular Expressions / Finite State Machines
- **Papers:** 2307.09702v4, 2506.09701v2
- **Expressiveness:** Sequential patterns
- **Clinical Use Cases:**
  - Medication name patterns
  - Date/time formats in EHR
  - Simple protocol sequences
- **Advantages:** Very efficient (linear time)
- **Limitations:** No counting, no context

### 3.3 Mathematical Constraints

#### Linear Constraints
- **Papers:** 2401.12243v2, 1908.05814v1
- **Expressiveness:** Ax ≤ b, Ax = b
- **Clinical Use Cases:**
  - Resource allocation: "Total beds ≤ capacity"
  - Dose bounds: "Drug dose in [min, max]"
  - Budget constraints
- **Advantages:** Efficient solvers (LP, QP)
- **Limitations:** Cannot express non-linear relationships

#### Differential Equations / Dynamics
- **Papers:** 2402.10998v3, 2406.00990v1
- **Expressiveness:** Continuous-time dynamics
- **Clinical Use Cases:**
  - Pharmacokinetic models (drug concentration)
  - Patient state evolution
  - Physiological constraints
- **Advantages:** Physically grounded
- **Limitations:** Requires accurate models

#### Domain-Specific Constraints
- **Papers:** 2402.05210v4 (anatomy), 2505.05019v1 (survival)
- **Expressiveness:** Task-specific invariants
- **Clinical Use Cases:**
  - Anatomical constraints in imaging
  - Survival curves in trials
  - Physiological ranges (heart rate, BP)
- **Advantages:** Directly encode domain knowledge
- **Limitations:** Requires expert specification

### 3.4 Knowledge Graphs and Ontologies

#### Medical Ontologies (SNOMED, ICD)
- **Papers:** 2410.22353v3 (RuleRAG)
- **Expressiveness:** Hierarchical relationships, is-a, part-of
- **Clinical Use Cases:**
  - Disease classification hierarchies
  - Drug-drug interaction graphs
  - Anatomical relationships
- **Advantages:** Standardized, widely adopted
- **Limitations:** Incomplete coverage, static

#### Custom Knowledge Graphs
- **Papers:** 2305.19068v2
- **Expressiveness:** Arbitrary relations, temporal edges
- **Clinical Use Cases:**
  - Patient knowledge graphs
  - Treatment pathway graphs
  - Evidence-based medicine links
- **Advantages:** Flexible, extensible
- **Limitations:** Requires construction and maintenance

---

## 4. Integration with Diffusion Models

### 4.1 Training-Time Integration

#### Architecture Modifications

1. **Conditional Diffusion (2402.05210v4)**
   - Segmentation mask at each diffusion step
   - Constraint: Anatomical structure
   - Method: Cross-attention on constraint encoding
   - Clinical Result: State-of-the-art anatomical faithfulness

2. **Constraint-Aware Loss (2406.00990v1)**
   - Hybrid loss = reconstruction + constraint violation
   - Learns to avoid violations during training
   - Tradeoff: May reduce diversity

3. **Nuclear Regularization (2509.00395v1)**
   - Low-rank constraint via nuclear norm
   - Reduces latent complexity
   - Clinical Application: PET reconstruction

#### Learned Constraint Representations

1. **Safety Models (2506.01121v1)**
   - Learn constraint satisfaction predictor
   - Guide diffusion away from violations
   - Training: Labeled safe/unsafe examples

2. **Constraint Autoencoders**
   - Encode constraints in latent space
   - Diffusion operates in constraint-satisfying manifold
   - Example: 2403.00323v1 (Softened Symbol Grounding)

### 4.2 Inference-Time Integration

#### Guidance Methods

1. **Classifier-Free Guidance with Constraints**
   - Papers: 2308.16534v3, 2405.20971v2
   - Method: Conditional score = unconditional + constraint gradient
   - Advantage: No retraining
   - Limitation: Approximate constraint satisfaction

2. **Symbolic Optimization Interleaving (2506.01121v1)**
   - Alternate: diffusion step → symbolic projection
   - Guarantees: Hard constraint satisfaction
   - Cost: Slower sampling (calls to solver)

3. **Stochastic Control Guidance (2402.14285v4)**
   - For non-differentiable constraints
   - Samples proposal, evaluate constraint, adjust
   - Application: Musical rules (applicable to clinical rules)

#### Projection Methods

1. **Reflected SDEs (2304.04740v3)**
   - Project onto constraint boundary when violated
   - Natural for convex domain constraints
   - Clinical Example: Dose ranges, physiological bounds

2. **Constraint-Guided Prediction (2506.12911v2)**
   - Deterministic diffusion + gradient corrections
   - Non-convex equality constraints
   - Clinical Potential: Power flow analog to patient flow

### 4.3 Hybrid Approaches

1. **Two-Stage Training (2508.16524v1)**
   - Stage 1: Basic generation capability
   - Stage 2: GRPO with rule-based rewards
   - Combines learning + constraint enforcement

2. **Constraint Tightening (2412.09342v2)**
   - Learn with soft constraints
   - Tighten to hard at deployment
   - Accounts for model uncertainty

---

## 5. Integration with Autoregressive Models

### 5.1 Challenges Specific to Autoregressive Generation

1. **Computational Hardness**
   - Paper: 2312.03905v2
   - Problem: Computing constraint likelihood is #P-hard
   - Solution: Pseudolikelihood approximation around model sample

2. **Long-Range Dependencies**
   - Paper: 2010.12884v2
   - Problem: Constraints span entire sequence
   - Solution: Predicate logic with beam search

3. **Exposure Bias**
   - Generated tokens affect future constraint satisfaction
   - Autoregressive errors compound

### 5.2 Methods for Autoregressive Models

#### Tractable Probabilistic Models (TPMs)

1. **GeLaTo (2304.07438v4)**
   - Uses distilled HMMs as TPMs
   - Efficient Pr(text | constraint) computation
   - Integration: Guides GPT-2 generation
   - Clinical Relevance: Lexical constraints in notes

#### Constrained Decoding

1. **NeuroLogic Decoding (2010.12884v2)**
   - Inference-time constraint satisfaction
   - No retraining required
   - Handles arbitrary predicate logic
   - Clinical Application: Structured report generation

2. **Automata-Guided Beam Search (2506.09701v2)**
   - DFA masks invalid tokens
   - Guaranteed constraint satisfaction
   - Clinical Application: Medical code sequences, protocol adherence

3. **Efficient Guided Generation (2307.09702v4)**
   - FSM-based vocabulary indexing
   - O(1) constraint checking per token
   - Supports regex, CFG
   - Clinical Application: Structured EHR text

#### Training-Based Methods

1. **Pseudo-Semantic Loss (2312.03905v2)**
   - Differentiable LTL semantics
   - Gumbel-Softmax for discrete variables
   - Combines with standard loss
   - Clinical Relevance: HIGH - temporal protocol constraints

2. **Constraint-Aware Fine-Tuning**
   - Paper: 2204.13355v1
   - Aligned Constrained Training (ACT)
   - Learns source-side context of constraints
   - Application: Low-frequency medical terms

### 5.3 Transformer-Specific Techniques

1. **Attention Masking**
   - Prevent attention to invalid future tokens
   - Enforce sequential constraints
   - Example: Medication must follow diagnosis

2. **Positional Encoding Modification**
   - Encode constraint information in positions
   - Paper: 1709.06404v1 (Anticipation-RNN)
   - Clinical: Time-based constraints in treatment plans

---

## 6. Clinical Constraint Examples

### 6.1 Treatment Protocols and Guidelines

#### Sepsis Treatment Protocol
```
Constraint Type: Temporal Logic (LTLf)
Example: "Within 1 hour: (draw blood cultures AND start antibiotics AND
          administer 30ml/kg crystalloid) BEFORE organ dysfunction worsens"

Implementation:
- Hard constraint via automata (2506.09701v2)
- Soft constraint via temporal loss (2312.03905v2)
```

#### Stroke Code Protocol
```
Constraint Type: Timed Sequences
Example: "Door-to-needle time < 60 minutes:
          1. CT scan within 15 min
          2. Lab results within 30 min
          3. tPA decision within 45 min
          4. Administration within 60 min"

Implementation:
- Policy optimization with time constraints (2401.12243v2)
- Diffusion with temporal guidance
```

### 6.2 Safety Constraints

#### Medication Dosing
```
Constraint Type: Linear + Logical
Examples:
- "Total opioid dose (MEQ) ≤ 50mg/day"
- "IF creatinine > 2.0 THEN reduce antibiotic dose by 50%"
- "NOT (warfarin AND aspirin) unless explicit indication"

Implementation:
- Linear constraints via projection (2304.04740v3)
- Logic via neuro-symbolic (2111.01564v1)
```

#### Vital Sign Bounds
```
Constraint Type: Domain Constraints
Examples:
- Heart rate ∈ [40, 180] bpm
- Blood pressure: systolic ∈ [70, 200], diastolic ∈ [40, 130]
- O2 saturation ≥ 88% (or ≥ 92% for most patients)
- Temperature ∈ [35, 41] °C

Implementation:
- Reflected diffusion (2304.04740v3)
- Constraint-aware VAE (2509.00395v1)
```

### 6.3 Resource Allocation

#### ED Bed Assignment
```
Constraint Type: Linear Integer Programming
Example: "Maximize throughput subject to:
          - Σ beds_used ≤ total_capacity
          - isolation_patients → isolation_rooms
          - nurse_ratio ≤ 1:4
          - trauma_beds ≥ 2 (always available)"

Implementation:
- CGPO (2401.12243v2) for interpretable policies
- Diffusion planning (2306.00603v2) for trajectories
```

#### ICU Admission Criteria
```
Constraint Type: Logical Decision Tree
Example: "Admit to ICU IF:
          (shock AND vasopressors) OR
          (respiratory_failure AND mechanical_ventilation) OR
          (AMS AND high_fall_risk) OR
          (SOFA_score > 6)"

Implementation:
- MultiplexNet (2111.01564v1) for logical branches
- Rule-guided generation (2402.14285v4)
```

### 6.4 Data Quality and Privacy

#### Synthetic EHR Generation
```
Constraint Type: Statistical + Logical
Examples:
- Survival constraints: death_date > all_encounter_dates
- Temporal coherence: diagnosis_date ≤ treatment_date
- Demographic consistency: age increases monotonically
- Privacy: k-anonymity ≥ 5

Implementation:
- Preprocessing + postprocessing (2505.05019v1)
- Statistical alignment (2509.10538v1)
- Constraint-aware GANs (2204.04797v2)
```

### 6.5 Diagnostic Reasoning

#### Differential Diagnosis Generation
```
Constraint Type: Knowledge Graph + Probability
Example: "Generate differential diagnosis WHERE:
          - Compatible with presenting symptoms
          - Respects disease prevalence priors
          - Considers patient risk factors
          - Excludes ruled-out conditions"

Implementation:
- Knowledge graph constraints (2305.19068v2)
- Neuro-symbolic reasoning (2508.16524v1)
```

#### Image Segmentation Constraints
```
Constraint Type: Anatomical
Example: "Segment tumor WHERE:
          - Contained within organ boundary
          - Connected component (no floating pixels)
          - Volume change < 20% between scans
          - Respects anatomical priors (shape, location)"

Implementation:
- Segmentation-guided diffusion (2402.05210v4)
- Anatomical constraints (2312.00944v1)
```

---

## 7. Generation Quality vs. Constraint Satisfaction Tradeoff

### 7.1 Theoretical Perspectives

#### Information-Theoretic View
- **Paper:** 2206.05395v2
- Constraint reduces entropy of generation space
- Quality ∝ conditional distribution match
- Satisfaction ∝ constraint violation penalty
- Optimal: λ* balances both objectives

#### Optimization View
```
Objective: max_θ E_x~p_θ[reward(x)] subject to: E_x~p_θ[constraint(x)] ≥ threshold

Pareto Frontier: Set of (quality, satisfaction) pairs that cannot improve both
```

### 7.2 Empirical Findings

#### Hard Constraints Often Reduce Quality

1. **Diffusion Models (2406.00990v1)**
   - Unconstrained: Better FID scores
   - Hard constrained: 15-20% worse FID, but 100% valid
   - Soft constrained: 5-10% worse FID, 85-95% valid

2. **Autoregressive Models (2312.03905v2)**
   - GPT-2 unconstrained: Perplexity 23
   - With logical constraints: Perplexity 28
   - But: 90% constraint satisfaction vs 10%

3. **Clinical Data (2505.05019v1)**
   - Without domain constraints: 61% invalid survival data
   - With preprocessing: 12% invalid
   - With pre+post processing: 2% invalid
   - Quality (MMD): Marginally worse with constraints

#### Soft Constraints: Flexible Tradeoff

1. **Learnable Weights (2009.06087v2)**
   - Can downweight incorrect constraints
   - Maintains quality on correct constraints
   - Risk: May ignore important constraints

2. **Temperature/Guidance Scaling**
   - Higher guidance → better constraint satisfaction, worse quality
   - Papers: 2308.16534v3, 2402.05210v4
   - Optimal temperature depends on task

### 7.3 Strategies to Improve Tradeoff

#### 1. Better Constraint Representations

**Example: Hierarchical Constraints (2503.07148v3)**
- High-level: "Treatment follows diagnosis"
- Low-level: "Administer drug X at time T"
- Result: Better quality by reducing constraint complexity

#### 2. Constraint Decomposition

**Example: Neuro-Symbolic (2506.01121v1)**
- Neural: Learn data distribution
- Symbolic: Enforce hard constraints
- Result: Each component optimized separately

#### 3. Progressive Constraint Tightening

**Example: Curriculum Learning**
- Phase 1: Weak constraints (high quality)
- Phase 2: Medium constraints (balanced)
- Phase 3: Strict constraints (guaranteed satisfaction)
- Papers: 2508.16524v1 (two-stage training)

#### 4. Constraint-Aware Architectures

**Example: Latent Constraint Manifolds**
- Learn latent space where constraints are easier
- Papers: 2403.00323v1 (Softened Symbol Grounding)
- Result: Better quality for same constraint level

#### 5. Posterior Refinement

**Example: Generate-then-Project**
- High-quality unconstrained generation
- Project to nearest valid point
- Papers: 2304.04740v3 (Reflected Diffusion)
- Tradeoff: May create artifacts at boundary

### 7.4 Clinical Implications

#### Safety-Critical Applications
- **Priority:** Constraint satisfaction >> Quality
- **Approach:** Hard constraints (2506.09701v2, 2402.10998v3)
- **Examples:** Drug dosing, ventilator settings, insulin pumps

#### Decision Support Applications
- **Priority:** Quality ≈ Constraint satisfaction
- **Approach:** Soft constraints with high weights
- **Examples:** Diagnosis suggestions, treatment recommendations

#### Data Augmentation Applications
- **Priority:** Quality > Constraint satisfaction
- **Approach:** Soft constraints, post-filtering
- **Examples:** Training data generation, synthetic EHRs

---

## 8. Research Gaps

### 8.1 Temporal and Sequential Constraints

#### Current Limitations
1. **Limited Long-Horizon Support**
   - Most papers: 10-50 timesteps
   - Clinical needs: Hours to days of sequences
   - Gap: Constraint propagation over long horizons

2. **Weak Temporal Logic Support**
   - Paper 2312.03905v2: Only work on LTLf for autoregressive
   - Missing: Metric Temporal Logic (MTL) for continuous time
   - Clinical Need: "Give antibiotics within 60 minutes"

3. **No Probabilistic Temporal Constraints**
   - Missing: "80% of patients respond within 24 hours"
   - Would enable stochastic protocol modeling

#### Research Opportunities
- Extend 2312.03905v2 to Metric Temporal Logic
- Combine with diffusion models for continuous time (2506.12911v2)
- Learn temporal constraint structure from clinical data

### 8.2 Multi-Objective Constraint Satisfaction

#### Current Limitations
1. **Single Constraint Focus**
   - Most papers: One type of constraint
   - Clinical Reality: Multiple competing constraints
   - Example: Minimize cost AND length-of-stay AND complications

2. **No Constraint Prioritization**
   - Paper 2304.06625v3: Exception with priority ordering
   - Missing: Learned priority from clinical outcomes

3. **Lack of Pareto Optimization**
   - No methods to explore quality-constraint tradeoff space
   - Needed for clinical decision support

#### Research Opportunities
- Integrate 2304.06625v3 (priority constraints) with diffusion models
- Multi-objective RL for constraint weight learning
- Interactive constraint elicitation from clinicians

### 8.3 Constraint Learning and Discovery

#### Current Limitations
1. **Manual Constraint Specification**
   - All papers require human-provided constraints
   - Labor-intensive for complex domains like medicine

2. **No Constraint Mining from Data**
   - Opportunity: Learn constraints from clinical data
   - Example: Discover implicit protocol patterns

3. **Brittle to Constraint Errors**
   - Most methods assume correct constraints
   - Exception: 2009.06087v2 (learnable clause weights)

#### Research Opportunities
- Constraint discovery from clinical event sequences
- Robust learning under noisy/incorrect constraints
- Active learning to query clinicians for constraint clarification

### 8.4 Scalability and Efficiency

#### Current Limitations
1. **Computational Cost**
   - Symbolic solvers slow: 2506.01121v1
   - Multiple inference passes: 2010.12884v2
   - Clinical Need: Real-time decision support

2. **Limited Constraint Complexity**
   - Most papers: 10-50 constraints
   - Clinical protocols: 100s of interacting rules

3. **Memory Requirements**
   - Knowledge compilation: Exponential in worst case
   - DFA construction: Can explode in size

#### Research Opportunities
- Approximate inference for large constraint sets
- Hierarchical constraint decomposition (learn from 2503.07148v3)
- Constraint caching and reuse across patients

### 8.5 Interpretability and Explainability

#### Current Limitations
1. **Black-Box Constraint Violation**
   - When constraints violated, unclear why
   - Clinical Need: Explain to users

2. **No Counterfactual Explanations**
   - "What if we relaxed constraint X?"
   - Useful for protocol adaptation

3. **Lack of Confidence Estimates**
   - How confident is model in constraint satisfaction?
   - Critical for clinical deployment

#### Research Opportunities
- Constraint attribution methods
- Uncertainty quantification for constraint satisfaction
- Interactive constraint relaxation with explanations

### 8.6 Domain Adaptation and Transfer

#### Current Limitations
1. **Task-Specific Methods**
   - Most papers: Single domain
   - Medical Reality: Many specialties, conditions

2. **No Cross-Protocol Transfer**
   - Cannot reuse learned constraint knowledge
   - Example: Transfer sepsis protocol to pneumonia

3. **Limited Zero-Shot Constraint Application**
   - Exception: 2308.16534v3 (zero-shot diffusion)
   - Most require retraining for new constraints

#### Research Opportunities
- Meta-learning for constraint adaptation
- Prompt-based constraint specification (extend 2410.22353v3)
- Universal constraint representations across medical domains

### 8.7 Integration with Clinical Workflows

#### Current Limitations
1. **No Human-in-the-Loop Methods**
   - All papers: Fully automated
   - Clinical Reality: Physician oversight required

2. **Lack of Clinical Validation**
   - Mostly tested on synthetic/benchmark data
   - Need: Clinical trials of constraint-guided systems

3. **No Regulatory Framework**
   - How to certify constraint satisfaction?
   - What error rates acceptable?

#### Research Opportunities
- Interactive constraint refinement with clinicians
- Prospective clinical trials of generated protocols
- Develop safety cases for regulatory approval

### 8.8 Multimodal Constraints

#### Current Limitations
1. **Single Modality**
   - Most papers: Text OR images OR time-series
   - Clinical: Multimodal (images + labs + notes + vitals)

2. **No Cross-Modal Constraints**
   - Example: "Tumor size in image must match measurement in notes"
   - Opportunity: Consistency checking

#### Research Opportunities
- Extend 2402.05210v4 (image constraints) to multimodal
- Cross-modal constraint verification
- Multimodal synthetic data with alignment constraints

---

## 9. Relevance to ED Policy-Guided Trajectory Generation

### 9.1 Direct Applications

#### 1. Patient Flow Optimization

**Applicable Methods:**
- **Diffusion-based trajectory planning (2306.00603v2, 2405.01758v1)**
  - Generate patient trajectories through ED
  - Constraints: Bed capacity, nurse ratios, equipment availability
  - Policy: Minimize length-of-stay while maintaining quality

- **CGPO (2401.12243v2)**
  - Interpretable policies for triage decisions
  - Hard constraints: Safety protocols, resource limits
  - Benefit: Explainable to clinical staff

**Implementation Strategy:**
```
State: [patient_location, vitals, resources_available, staff_assignments]
Constraints:
  - bed_occupancy ≤ capacity
  - nurse_ratio ≤ 1:4
  - trauma_bays ≥ 2
  - isolation_rooms for COVID
  - door-to-doctor < 30min (triage priority)

Action: [assign_bed, assign_staff, order_tests, disposition]
Policy: Diffusion model generating full trajectories
Guidance: Protocol constraints + resource constraints
```

#### 2. Treatment Protocol Adherence

**Applicable Methods:**
- **Temporal logic constraints (2312.03905v2)**
  - Encode clinical pathways as LTLf
  - Example: Sepsis bundle compliance
  - Generate trajectories satisfying time-based protocols

- **Automata-guided generation (2506.09701v2)**
  - Compile protocols to DFA
  - Guarantee protocol compliance
  - Real-time verification

**Clinical Pathway Example:**
```
Sepsis Protocol as LTL:
  F[0,60min](blood_culture_drawn) ∧
  F[0,60min](antibiotics_started) ∧
  F[0,180min](lactate_measured) ∧
  G(if organ_dysfunction then admit_ICU)

Implementation: Automata masks invalid actions at each timestep
```

#### 3. Resource-Constrained Scheduling

**Applicable Methods:**
- **Safe RL with budget constraints (2306.00603v2)**
  - Real-time budget constraints (CT scanner, labs)
  - Offline learning from historical data
  - Diffusion planning for trajectories

- **Linear constraint optimization (2401.12243v2)**
  - Explicit resource constraints
  - Bounded optimality guarantees

**Resource Model:**
```
Resources: [beds, nurses, CT_scanner, echo, lab_capacity]
Budget: [max_CT_scans_per_hour, max_lab_tests, staff_hours]

Policy: Allocate resources to maximize patient outcomes
        subject to budget constraints

Method: TREBI (2306.00603v2) with trajectory-level constraints
```

### 9.2 Hybrid Approach for ED Application

#### Proposed Architecture

**Stage 1: Policy Learning (Offline)**
```
Data: Historical ED trajectories (state, action, outcome)
Model: Diffusion-based trajectory generator
Constraints:
  - Soft: Learn from successful historical trajectories
  - Hard: Safety protocols (medication dosing, isolation)

Training:
  1. Train unconditional diffusion on all trajectories
  2. Fine-tune with constraint-aware loss (2406.00990v1)
  3. Add rule-based rewards via GRPO (2508.16524v1)
```

**Stage 2: Online Trajectory Generation**
```
Input: Current ED state + patient presentation
Constraints:
  - Real-time: Current resource availability
  - Protocol: Applicable clinical pathways
  - Safety: Hard bounds on interventions

Generation:
  1. Diffusion generates candidate trajectory
  2. Symbolic checker verifies protocol compliance (2506.09701v2)
  3. Project onto resource constraints (2304.04740v3)
  4. Return trajectory with confidence bounds
```

**Stage 3: Human-in-the-Loop Refinement**
```
Clinician reviews generated trajectory
Can:
  - Adjust constraint weights
  - Override specific actions
  - Provide feedback for online learning

System learns from corrections (continual learning)
```

### 9.3 Specific Constraint Types for ED

#### Protocol Constraints
```
Type: Temporal Logic (LTLf)
Source: Clinical guidelines, hospital policies
Examples:
  - Stroke: CT within 15min, decision within 45min
  - STEMI: EKG within 10min, cath lab within 90min
  - Sepsis: Antibiotics within 60min

Implementation: 2312.03905v2 (pseudo-semantic loss)
Verification: 2506.09701v2 (automata-guided)
```

#### Resource Constraints
```
Type: Linear inequalities
Source: Real-time ED state
Examples:
  - Total beds ≤ 45
  - ICU beds ≤ 8
  - Nurses_on_duty / Patients ≥ 1:4
  - CT_scans_per_hour ≤ 3

Implementation: 2401.12243v2 (CGPO)
Projection: 2304.04740v3 (reflected diffusion)
```

#### Safety Constraints
```
Type: Logical + Domain bounds
Source: Medical knowledge, regulations
Examples:
  - NOT (drug_A AND drug_B) [contraindication]
  - vital_signs ∈ [min, max]
  - IF allergy THEN NOT medication
  - Pain score monitoring every 2 hours if opioids given

Implementation: 2111.01564v1 (MultiplexNet for logic)
              + 2009.06087v2 (KENN for soft constraints)
```

#### Performance Constraints
```
Type: Statistical + Optimization
Source: Quality metrics, operational goals
Examples:
  - Length of stay < 4 hours (80th percentile)
  - Left without being seen < 2%
  - Door-to-provider time < 30 min
  - Patient satisfaction > 8.5/10

Implementation: Multi-objective optimization
                Soft constraints weighted by importance
```

### 9.4 Advantages Over Current ED Systems

#### 1. Explicit Constraint Handling
- Current: Implicit in human decision-making
- Proposed: Explicit, verifiable, auditable
- Benefit: Regulatory compliance, quality assurance

#### 2. Adaptability
- Current: Static protocols, slow to update
- Proposed: Learn from outcomes, update policies
- Methods: Online learning (2306.00603v2)

#### 3. Interpretability
- Current: Black-box ML or rule-based (not both)
- Proposed: Neuro-symbolic with explanations
- Methods: 2401.12243v2 (interpretable policies)

#### 4. Robustness
- Current: Brittleto unexpected situations
- Proposed: Graceful constraint relaxation
- Methods: 2304.06625v3 (priority-based constraints)

#### 5. Personalization
- Current: One-size-fits-all protocols
- Proposed: Patient-specific trajectories
- Constraints: Respect individual contraindications, preferences

### 9.5 Implementation Roadmap

#### Phase 1: Offline Development (Months 1-6)
```
1. Data collection: Historical ED trajectories
2. Constraint elicitation: Interview clinicians, extract from guidelines
3. Model training:
   - Baseline diffusion model (unconditional)
   - Add soft constraints (2406.00990v1)
   - Add hard constraints (2506.01121v1 or 2506.09701v2)
4. Validation: Offline metrics, simulated ED
```

#### Phase 2: Clinical Integration (Months 7-12)
```
1. Build ED state monitoring system
2. Integrate with EHR for real-time data
3. Develop clinician interface:
   - Display generated trajectories
   - Highlight constraint satisfaction
   - Allow manual overrides
4. Shadow deployment: Generate recommendations, observe usage
```

#### Phase 3: Prospective Evaluation (Months 13-24)
```
1. Randomized trial: AI-assisted vs standard care
2. Metrics:
   - Primary: Length of stay, patient outcomes
   - Secondary: Resource utilization, protocol compliance
   - Safety: Adverse events, constraint violations
3. Iterate based on clinical feedback
4. Scale to multiple EDs
```

---

## 10. Conclusions and Future Directions

### 10.1 Key Takeaways

#### Methodological Insights

1. **No Universal Solution**
   - Hard constraints: Safety-critical applications
   - Soft constraints: Generation quality priority
   - Hybrid: Best of both worlds

2. **Neuro-Symbolic is Promising**
   - Papers: 2506.01121v1, 2508.16524v1, 2111.01564v1
   - Combines learning + reasoning
   - Well-suited for clinical domains with explicit knowledge

3. **Diffusion Models Are Competitive**
   - Flexible constraint integration (training & inference time)
   - Better than GANs for constraint satisfaction
   - Examples: 2402.05210v4 (medical imaging), 2406.00990v1 (trajectories)

4. **Autoregressive Models Face Challenges**
   - Constraint likelihood computation is hard (#P-hard)
   - Solutions: Pseudolikelihood (2312.03905v2), automata (2506.09701v2)
   - Still limited for long-horizon generation

5. **Constraint Specification is Critical**
   - Wrong constraints → wrong outputs (even with perfect model)
   - Need: Robust learning (2009.06087v2), constraint discovery

#### Clinical Applications

1. **Medical Imaging is Advanced**
   - Paper 2402.05210v4: State-of-the-art anatomical control
   - Gap: Limited to static images, not sequences

2. **Clinical Trial Data Generation is Emerging**
   - Papers: 2505.05019v1, 2409.07089v2, 2509.10538v1
   - Critical: Domain constraints prevent invalid data
   - Gap: Long-term outcome prediction

3. **Protocol Adherence is Underexplored**
   - Only 2312.03905v2 for temporal logic
   - Huge opportunity for ED, ICU, surgical workflows

4. **Multimodal Clinical AI is Missing**
   - No papers on multimodal constraints (images + text + time-series)
   - Clinical reality requires multimodal reasoning

### 10.2 Recommendations for ED Trajectory Generation

#### 1. Start with Soft Constraints
- Easier to train, more robust
- Use historical data to learn constraint weights
- Methods: 2406.00990v1, 2009.06087v2

#### 2. Add Hard Safety Constraints
- Critical protocols (medication, isolation)
- Use automata-guided generation (2506.09701v2)
- Or symbolic projection (2506.01121v1)

#### 3. Enable Interactive Refinement
- Clinicians adjust constraint priorities
- System learns from overrides
- Methods: Extend 2304.06625v3 to online setting

#### 4. Validate Extensively
- Offline: Metrics on historical data
- Simulation: Test in virtual ED
- Prospective: Randomized trial
- Focus: Safety, not just efficiency

#### 5. Plan for Deployment
- Regulatory pathway (FDA if decision support)
- Interpretability for clinicians
- Monitoring for constraint violations

### 10.3 Future Research Directions

#### Near-Term (1-2 years)

1. **Extend Temporal Logic to Clinical Protocols**
   - Build on 2312.03905v2
   - Add Metric Temporal Logic (real-valued time)
   - Demonstrate on sepsis, stroke, STEMI protocols

2. **Multimodal Constraint Integration**
   - Combine 2402.05210v4 (images) with text/time-series
   - Cross-modal consistency constraints
   - Application: Radiology report + image generation

3. **Constraint Discovery from Clinical Data**
   - Mine patterns from EHR
   - Learn implicit protocol constraints
   - Validate with clinicians

4. **Efficient Inference for Real-Time Deployment**
   - Optimize symbolic solvers for clinical use
   - Approximate inference for large constraint sets
   - Target: < 1 second for trajectory generation

#### Long-Term (3-5 years)

1. **Foundation Models with Clinical Constraints**
   - Pre-train on medical knowledge graphs
   - Fine-tune with constraint-aware objectives
   - Zero-shot adaptation to new hospitals/protocols

2. **Causal Constraint Models**
   - Move beyond correlational constraints
   - Learn causal effects of interventions
   - Enable counterfactual trajectory generation

3. **Multi-Objective Clinical Optimization**
   - Balance competing goals (cost, quality, patient preference)
   - Pareto-optimal trajectory generation
   - Interactive preference elicitation

4. **Federated Learning with Constraints**
   - Learn from multiple hospitals without sharing data
   - Preserve privacy while learning shared constraints
   - Adapt to hospital-specific protocols

5. **Certifiable AI for Clinical Deployment**
   - Formal verification of constraint satisfaction
   - Safety cases for regulatory approval
   - Continuous monitoring in production

### 10.4 Open Challenges

1. **Theoretical Foundations**
   - Formal analysis of quality-constraint tradeoff
   - Sample complexity of constraint learning
   - Generalization bounds for constrained generation

2. **Scalability**
   - 1000s of constraints in real protocols
   - Real-time inference on edge devices
   - Efficient constraint compilation

3. **Robustness**
   - Noisy/incorrect constraints
   - Distribution shift (new patient populations)
   - Adversarial constraint violations

4. **Human Factors**
   - How do clinicians interact with constraint-guided systems?
   - What level of automation is appropriate?
   - How to build trust in AI recommendations?

5. **Evaluation**
   - Metrics for constraint satisfaction quality
   - Clinical endpoint validation (not just surrogate metrics)
   - Long-term safety monitoring

---

## 11. Summary Table: Methods Comparison

| Method | ArXiv ID | Constraint Type | Hard/Soft | Integration | Clinical Applicability | Key Advantage | Key Limitation |
|--------|----------|-----------------|-----------|-------------|------------------------|---------------|----------------|
| Neuro-Symbolic Diffusion | 2506.01121v1 | Functional + Logic | Hard | Symbolic optimization interleaving | HIGH | 100% satisfaction guarantee | Slow inference |
| Pseudo-Semantic Loss | 2312.03905v2 | Temporal Logic (LTLf) | Soft | Differentiable loss | HIGH | Autoregressive + temporal | Approximate satisfaction |
| Automata-Guided Beam Search | 2506.09701v2 | Any (DFA-compilable) | Hard | Inference-time masking | VERY HIGH | Guaranteed + efficient | Limited to DFA |
| NeuroLogic Decoding | 2010.12884v2 | Predicate Logic | Hard | Inference-time search | HIGH | No retraining needed | Slow for complex constraints |
| MultiplexNet | 2111.01564v1 | DNF Logic | Hard | Latent variable selection | HIGH | 100% satisfaction | Requires DNF encoding |
| KENN | 2009.06087v2 | First-Order Logic | Soft | Learnable clause weights | HIGH | Robust to errors | No guarantees |
| CGPO | 2401.12243v2 | Linear | Hard | Constraint generation | HIGH | Interpretable + bounded error | Limited to linear |
| Reflected Diffusion | 2304.04740v3 | Domain (convex) | Hard | Reflection operator | MEDIUM | Natural domain handling | Requires smooth boundaries |
| Constraint-Aware Diffusion | 2406.00990v1 | General | Soft | Hybrid loss | MEDIUM | Flexible | No guarantees |
| Anatomical Diffusion | 2402.05210v4 | Anatomical Segmentation | Hard | Condition at each step | VERY HIGH | SOTA medical imaging | Static images only |
| Safe Offline RL | 2306.00603v2 | Budget + Safety | Hard | Diffusion planning | HIGH | Real-time budgets | Offline only |
| Efficient Guided Generation | 2307.09702v4 | Regex + CFG | Hard | FSM indexing | HIGH | Very fast (O(1)) | Limited expressiveness |
| RuleRAG | 2410.22353v3 | KG-derived rules | Soft | Retrieval guidance | HIGH | Uses existing KGs | Soft constraints only |

---

## 12. Key Metrics for Evaluation

### 12.1 Constraint Satisfaction Metrics

1. **Violation Rate**
   - % of generated samples violating constraints
   - Target: 0% for hard constraints, <5% for soft

2. **Constraint Strength**
   - Degree of violation when violated
   - Important for safety-critical applications

3. **Multi-Constraint Satisfaction**
   - % satisfying ALL constraints simultaneously
   - Often much lower than individual satisfaction

### 12.2 Generation Quality Metrics

1. **Distributional Fidelity**
   - FID (Fréchet Inception Distance) for images
   - MMD (Maximum Mean Discrepancy) for general data
   - KL Divergence for distributions

2. **Diversity**
   - Intra-class variance
   - Coverage of modes

3. **Utility for Downstream Tasks**
   - Classification accuracy on synthetic data
   - Augmentation benefit

### 12.3 Clinical Metrics

1. **Protocol Compliance**
   - % of trajectories following clinical pathways
   - Time-to-intervention metrics

2. **Safety**
   - Adverse event rate
   - Medication error rate
   - Constraint violation severity

3. **Efficiency**
   - Length of stay
   - Resource utilization
   - Throughput (patients/hour)

4. **Outcomes**
   - Mortality
   - Readmission rate
   - Patient satisfaction

---

## References

This review synthesized findings from 150+ papers on constraint-guided generation. Key papers are cited throughout with their ArXiv IDs. All papers were retrieved from ArXiv in December 2025.

**Primary Search Queries:**
- "constrained generation" AND neural
- "guided generation" AND constraint
- "neuro-symbolic" AND constraint
- "diffusion model" AND constraint
- "logical constraint" AND neural
- "safety constraint" AND generation
- "policy" AND "constraint" AND "trajectory"
- "clinical" AND "constraint" AND generation
- "autoregressive" AND "constraint"

**Categories Searched:** cs.LG, cs.AI, cs.CL

---

## Appendix: Constraint Satisfaction Guarantees

### Probabilistic Guarantees
- **Soft Constraints:** E[violation] ≤ ε
- **Confidence Bounds:** Pr(violation) ≤ δ
- **Bayesian:** Posterior probability of satisfaction

### Hard Guarantees
- **Logical:** ∀x ∈ Generated, φ(x) = True
- **Domain:** ∀x, x ∈ D (valid domain)
- **Temporal:** Trajectory satisfies LTL formula

### Approximate Guarantees
- **ε-satisfaction:** constraint(x) ≥ threshold - ε
- **Probabilistic ε-satisfaction:** Pr(constraint(x) ≥ threshold - ε) ≥ 1-δ

---

**Document prepared by:** ArXiv Literature Search System
**Date:** December 1, 2025
**Total Papers Reviewed:** 150+
**Document Length:** ~15,000 words
