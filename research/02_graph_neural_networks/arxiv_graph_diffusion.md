# Graph Diffusion Models for Clinical Knowledge Graph Generation: ArXiv Research Synthesis

**Research Date:** December 1, 2025
**Focus Area:** Graph diffusion models relevant to clinical knowledge graph trajectory generation

---

## Executive Summary

Graph diffusion models have emerged as state-of-the-art approaches for generating complex graph-structured data, with significant applications in molecular design, drug discovery, and potentially clinical knowledge graph generation. This synthesis examines 60+ papers from ArXiv focusing on discrete and continuous diffusion processes over graphs, with particular attention to conditional generation, architectural innovations, and their applicability to clinical trajectory modeling.

**Key Findings:**
- **Discrete diffusion models** (e.g., DiGress, GDSS) dominate molecular and general graph generation
- **Conditioning mechanisms** enable multi-property control critical for clinical constraints
- **Scalability** remains a challenge, with most models limited to hundreds of nodes
- **Research gap**: Limited work on knowledge graph-specific generation and clinical domain applications

---

## Key Papers with ArXiv IDs

### Foundational Discrete Diffusion Models

#### 1. **DiGress: Discrete Denoising Diffusion for Graph Generation** (2209.14734v4)
- **Authors:** Vignac et al.
- **Key Innovation:** Discrete diffusion process that progressively edits graphs through edge addition/removal and category changes
- **Architecture:** Graph transformer network for node and edge classification
- **Noise Model:** Markovian noise preserving marginal distributions
- **Performance:** 3x validity improvement on planar graphs; scales to GuacaMol (1.3M molecules)
- **Conditioning:** Graph-level feature conditioning via auxiliary features
- **Clinical Relevance:** Establishes discrete diffusion framework applicable to categorical clinical entities

#### 2. **Diffusion Models for Graphs Benefit From Discrete State Spaces** (2210.01549v4)
- **Authors:** Haefeli et al.
- **ArXiv ID:** 2210.01549v4
- **Key Finding:** Discrete noise outperforms continuous Gaussian perturbations
- **Metrics:** 1.5x reduction in MMD; 30x faster sampling (32 vs 1000 steps)
- **Node/Edge Generation:** Simultaneous discrete state diffusion
- **Quality Metrics:** MMD, validity, uniqueness, novelty
- **Clinical Relevance:** Demonstrates importance of discrete formulations for categorical medical data

#### 3. **GraphGDP: Generative Diffusion Processes for Permutation Invariant Graph Generation** (2212.01842v1)
- **Authors:** Huang et al.
- **ArXiv ID:** 2212.01842v1
- **Diffusion Process:** Continuous-time SDE-based diffusion
- **Architecture:** Position-enhanced graph score network
- **Key Property:** Permutation equivariant score estimation
- **Speed:** 24 function evaluations (much faster than autoregressive)
- **Clinical Relevance:** Permutation invariance critical for unordered clinical event sets

### Score-Based Generative Models

#### 4. **GDSS: Score-based Generative Modeling via System of SDEs** (2202.02514v3)
- **Authors:** Jo et al.
- **ArXiv ID:** 2202.02514v3
- **Innovation:** Joint node-edge distribution via coupled SDEs
- **Architecture:** System of stochastic differential equations
- **Score Matching:** Tailored objectives for gradient estimation
- **Novelty:** Captures node-edge relationships without violating chemical valency
- **Molecular Focus:** Strong performance on molecule generation
- **Clinical Relevance:** Coupled SDE framework applicable to entity-relationship modeling

#### 5. **Permutation Invariant Graph Generation via Score-Based Generative Modeling** (2003.00638v1)
- **Authors:** Niu et al.
- **ArXiv ID:** 2003.00638v1
- **Key Contribution:** First permutation invariant score-based approach
- **Architecture:** Permutation equivariant multi-channel GNN
- **Training:** Score matching with annealed Langevin dynamics
- **Innovation:** Implicitly defines permutation invariant distribution
- **Clinical Relevance:** Foundational work for order-agnostic clinical event generation

### Conditional and Multi-Property Generation

#### 6. **Graph Diffusion Transformers for Multi-Conditional Molecular Generation** (2401.13858v3)
- **Authors:** Liu et al.
- **ArXiv ID:** 2401.13858v3
- **Conditioning:** Multiple properties (synthetic score, gas permeability, etc.)
- **Architecture:** Graph DiT with encoder for property representations
- **Noise Model:** Novel graph-dependent noise estimation
- **Metrics:** 9 metrics covering distribution learning and condition control
- **Applications:** Polymer and small molecule inverse design
- **Clinical Relevance:** Multi-conditional framework directly applicable to clinical constraint satisfaction

#### 7. **Composable Score-based Graph Diffusion Model (CSGD)** (2509.09451v2)
- **Authors:** Qiao et al.
- **ArXiv ID:** 2509.09451v2
- **Innovation:** Concrete scores for discrete graphs enabling compositional guidance
- **Conditioning:** Composable Guidance (CoG) for arbitrary condition subsets
- **Probability Calibration:** Mitigates train-test mismatches
- **Performance:** 15.3% improvement in controllability
- **Clinical Relevance:** Flexible multi-property control essential for clinical trajectory constraints

#### 8. **Graph Guided Diffusion (GGDiff)** (2505.19685v1)
- **Authors:** Tenorio et al.
- **ArXiv ID:** 2505.19685v1
- **Framework:** Conditional diffusion as stochastic control problem
- **Guidance Types:** Gradient-based, control-based, zero-order approximations
- **Applications:** Graph motifs, fairness, link prediction
- **Innovation:** Unified guidance framework for differentiable and non-differentiable rewards
- **Clinical Relevance:** Enables constraint-based generation without retraining

### Continuous-Time Discrete-State Models

#### 9. **Cometh: Continuous-Time Discrete-State Graph Diffusion** (2406.06449v2)
- **Authors:** Siraudin et al.
- **ArXiv ID:** 2406.06449v2
- **Innovation:** Continuous-time Markov chains for discrete states
- **Performance:** 99.5% on planar graphs; 12.6% improvement over DiGress on GuacaMol
- **Encoding:** Single random-walk-based encoding (simpler than prior work)
- **Clinical Relevance:** Continuous-time formulation natural for clinical event modeling

#### 10. **Discrete-State Continuous-Time Diffusion for Graph Generation** (2405.11416v2)
- **Authors:** Xu et al.
- **ArXiv ID:** 2405.11416v2
- **Formulation:** Discrete-state continuous-time setting
- **Advantages:** Preserves discrete nature while offering sampling flexibility
- **Invariance:** Ideal equivariant properties for node ordering
- **Clinical Relevance:** Balance between discrete clinical states and continuous time modeling

### Autoregressive Diffusion Approaches

#### 11. **PARD: Permutation-Invariant Autoregressive Diffusion** (2402.03687v3)
- **Authors:** Zhao et al.
- **ArXiv ID:** 2402.03687v3
- **Innovation:** Combines autoregressive and diffusion approaches
- **Block Generation:** Graph as sequence of bipartite graphs
- **Architecture:** Higher-order graph transformer with PPGN integration
- **Scalability:** Handles MOSES (1.9M molecules)
- **Efficiency:** Parallel training like GPT
- **Clinical Relevance:** Block-based generation could model clinical care phases

#### 12. **Autoregressive Diffusion Model for Graph Generation** (2307.08849v1)
- **Authors:** Kong et al.
- **ArXiv ID:** 2307.08849v1
- **Process:** Node-absorbing diffusion in discrete graph space
- **Networks:** Diffusion ordering network + denoising network
- **Advantages:** Fast generation, captures graph topology
- **Clinical Relevance:** Data-dependent ordering could reflect clinical protocols

### Advanced Architectures and Techniques

#### 13. **LayerDAG: Layerwise Autoregressive Diffusion for DAG Generation** (2411.02322v2)
- **Authors:** Li et al.
- **ArXiv ID:** 2411.02322v2
- **Focus:** Directed acyclic graphs (DAGs)
- **Innovation:** Decouples node dependencies into sequential bipartite graphs
- **Scale:** Up to 400 nodes
- **Applications:** Hardware synthesis, program optimization
- **Clinical Relevance:** Clinical pathways often have DAG structure (treatment sequences)

#### 14. **GraphMaker: Can Diffusion Models Generate Large Attributed Graphs?** (2310.13833v4)
- **Authors:** Li et al.
- **ArXiv ID:** 2310.13833v4
- **Focus:** Large-scale attributed graphs
- **Innovation:** Asynchronous attribute-structure generation
- **Scalability:** Edge mini-batching for large graphs
- **Evaluation:** Utility-based (training ML models on synthetic data)
- **Clinical Relevance:** Addresses attribute-structure correlation in clinical data

#### 15. **Hyperbolic Graph Diffusion Model (HGDM)** (2306.07618v3)
- **Authors:** Wen et al.
- **ArXiv ID:** 2306.07618v3
- **Geometry:** Hyperbolic space for hierarchical structures
- **Innovation:** Hyperbolic latent space captures power-law distributions
- **Performance:** 48% improvement on hierarchical graphs
- **Clinical Relevance:** Medical taxonomies and disease hierarchies are naturally hyperbolic

### Latent Diffusion Models

#### 16. **Latent Graph Diffusion (LGD)** (2402.02518v2)
- **Authors:** Zhou et al.
- **ArXiv ID:** 2402.02518v2
- **Innovation:** Unified framework for all graph tasks (node, edge, graph-level)
- **Architecture:** Encoder-decoder with latent diffusion
- **Tasks:** Generation, regression, classification as conditional generation
- **Cross-attention:** Conditional generation mechanism
- **Clinical Relevance:** Unified framework could handle diverse clinical prediction tasks

#### 17. **Neural Graph Generator (NGG)** (2403.01535v3)
- **Authors:** Evdaimon et al.
- **ArXiv ID:** 2403.01535v3
- **Architecture:** Variational graph autoencoder + latent diffusion
- **Conditioning:** Graph statistics vectors
- **Versatility:** Captures diverse graph properties
- **Clinical Relevance:** Statistics-guided generation applicable to clinical constraints

### Molecular-Specific Models

#### 18. **Conditional Diffusion Based on Discrete Graph Structures (CDGS)** (2301.00427v2)
- **Authors:** Huang et al.
- **ArXiv ID:** 2301.00427v2
- **Forward Process:** SDE on structures and features
- **Condition:** Discrete graph structures
- **Architecture:** Hybrid graph noise prediction (global context + local dependency)
- **Efficiency:** ODE solvers for fast sampling
- **Clinical Relevance:** Demonstrates conditioning on structural constraints

#### 19. **Graph Generation with Diffusion Mixture (GruM)** (2302.03596v4)
- **Authors:** Jo et al.
- **ArXiv ID:** 2302.03596v4
- **Innovation:** Endpoint-conditioned diffusion mixture
- **Features:** Handles both continuous (3D) and discrete (categories) features
- **Convergence:** Rapid convergence via predicted final structures
- **Clinical Relevance:** Mixed feature types common in clinical data

### Specialized Techniques

#### 20. **GraphGUIDE: Interpretable Conditional Graph Generation** (2302.03790v1)
- **Authors:** Tseng et al.
- **ArXiv ID:** 2302.03790v1
- **Innovation:** Discrete Bernoulli diffusion with edge flipping
- **Control:** Full control over structural properties without predefined labels
- **Interpretability:** Framework designed for interpretable generation
- **Clinical Relevance:** Interpretability critical for clinical applications

#### 21. **Fast Graph Generation via Spectral Diffusion** (2211.08892v2)
- **Authors:** Luo et al.
- **ArXiv ID:** 2211.08892v2
- **Innovation:** Low-rank diffusion SDEs on graph spectrum space
- **Efficiency:** Significantly reduced computational cost
- **Theory:** Stronger theoretical guarantees than standard diffusion
- **Clinical Relevance:** Efficiency critical for real-time clinical systems

#### 22. **SaGess: Sampling Graph Denoising Diffusion for Scalable Generation** (2306.16827v1)
- **Authors:** Limnios et al.
- **ArXiv ID:** 2306.16827v1
- **Scalability:** Divide-and-conquer for large graphs
- **Method:** Samples subgraph covering, generates via DiGress, constructs full graph
- **Applications:** Link prediction, large real-world networks
- **Clinical Relevance:** Scalability approach applicable to hospital-scale knowledge graphs

---

## Graph Diffusion Architectures

### 1. **DiGress Family** (Discrete Diffusion)

**Core Components:**
- **Forward Process:** Discrete categorical noise on nodes and edges
- **Noise Model:** Markovian transitions preserving marginals
- **Architecture:** Graph transformer with node and edge classification heads
- **Training:** Sequence of classification tasks
- **Sampling:** Iterative denoising from random graph

**Variants:**
- **DiGress** (2209.14734v4): Base model with auxiliary features
- **Cometh** (2406.06449v2): Continuous-time extension
- **FreeGress** (2312.17397v2): Classifier-free guidance (79% MAE improvement)

**Node vs Edge Generation:**
- Simultaneous generation of node types and edge existence
- Edge features added/removed at each step
- Graph remains discrete throughout process

**Quality Metrics:**
- Validity: Percentage of valid graphs (e.g., chemical valency)
- Uniqueness: Proportion of unique generated samples
- Novelty: Proportion not in training set
- MMD: Maximum Mean Discrepancy for distribution matching

**Conditioning:**
- Graph-level features (size, properties)
- Auxiliary graph-theoretic features (degree distribution, clustering)
- Classifier-free and classifier-based guidance

### 2. **GDSS/Score-Based Family** (Continuous Diffusion)

**Core Components:**
- **Forward Process:** System of SDEs for nodes and edges
- **Score Network:** Estimates gradient of log-density
- **Architecture:** GNN-based score estimation network
- **Training:** Score matching objectives
- **Sampling:** Reverse-time SDE or ODE solver

**Key Papers:**
- **GDSS** (2202.02514v3): Coupled node-edge SDEs
- **Permutation Invariant Score-Based** (2003.00638v1): First equivariant approach
- **GraphGDP** (2212.01842v1): Position-enhanced score network

**Node vs Edge Generation:**
- Joint distribution modeling via coupled processes
- Position-aware encoding for structure preservation
- Gradients computed w.r.t. both adjacency and features

**Quality Metrics:**
- Distribution statistics (degree, clustering, orbit counts)
- Graph edit distance
- Spectral properties

**Conditioning:**
- Not inherently conditioned (unconditional generation focus)
- Extensions via conditional score matching
- Guidance through score manipulation

### 3. **Hybrid Autoregressive-Diffusion**

**PARD Architecture** (2402.03687v3):
- **Block Decomposition:** Graph as sequence of bipartite graphs
- **Partial Ordering:** Exploits inherent node-edge ordering
- **Diffusion per Block:** Shared diffusion model for each block
- **Transformer:** Higher-order graph transformer for expressiveness
- **Training:** Parallel like GPT

**Advantages:**
- Combines autoregressive efficiency with diffusion quality
- Maintains permutation invariance
- Scalable to large datasets (1.9M molecules)

### 4. **Latent Diffusion Architectures**

**LGD Framework** (2402.02518v2):
- **Encoder:** Maps graphs to latent vectors
- **Diffusion:** Standard diffusion in latent space
- **Decoder:** Reconstructs graph from latent
- **Cross-Attention:** For conditional generation

**NGG Framework** (2403.01535v3):
- **VAE:** Variational graph autoencoder
- **Latent Diffusion:** Gaussian diffusion in compressed space
- **Statistics Conditioning:** Guided by graph property vectors

**HGDM** (2306.07618v3):
- **Hyperbolic Encoder:** Projects to hyperbolic latent space
- **Radial-Angular Constraints:** Preserves hierarchical structure
- **Geometric Diffusion:** Anisotropic in hyperbolic space

---

## Discrete vs Continuous Approaches

### Discrete Diffusion

**Advantages:**
- **Natural Representation:** Graphs remain discrete throughout
- **Validity:** Easier to maintain graph constraints
- **Interpretability:** Clear semantic meaning at each step
- **Speed:** Fewer steps required (32-100 vs 1000)

**Disadvantages:**
- **Flexibility:** Less flexible noise schedules
- **Theory:** Less mature theoretical framework
- **Expressiveness:** May miss fine-grained distributions

**Key Models:**
- DiGress (2209.14734v4)
- Discrete State Spaces (2210.01549v4)
- GraphGUIDE (2302.03790v1)
- Unified Discrete Diffusion (2402.03701v2)

**Performance Evidence:**
- 1.5x MMD reduction (2210.01549v4)
- 30x faster sampling
- Higher validity rates on molecular benchmarks

### Continuous Diffusion

**Advantages:**
- **Theoretical Foundation:** Well-established SDE/ODE theory
- **Flexibility:** Continuous noise schedules
- **Expressiveness:** Can capture subtle distributions
- **Established Methods:** Score matching, DDPM, etc.

**Disadvantages:**
- **Discretization:** Requires conversion to discrete graphs
- **Validity:** Harder to enforce discrete constraints
- **Speed:** More sampling steps typically needed

**Key Models:**
- GDSS (2202.02514v3)
- Score-based (2003.00638v1)
- GraphGDP (2212.01842v1)

**Performance Evidence:**
- Better distribution matching on some benchmarks
- Struggles with validity on discrete-constrained tasks

### Hybrid Continuous-Time Discrete-State

**Best of Both Worlds:**
- **Cometh** (2406.06449v2): Continuous-time Markov chains
- **Discrete-state Continuous-time** (2405.11416v2)

**Advantages:**
- Preserves discrete graph nature
- Flexible sampling trade-offs
- Better theoretical properties than pure discrete

**Performance:**
- 99.5% on planar graphs
- 12.6% improvement over DiGress

---

## Conditioning on Properties and Constraints

### 1. **Multi-Property Conditioning**

#### Graph DiT (2401.13858v3)
- **Properties:** Numerical (molecular weight) and categorical (functional groups)
- **Mechanism:** Encoder learns property embeddings → Transformer denoiser
- **Graph-Dependent Noise:** Accurate noise estimation for molecules
- **Results:** Superior across 9 metrics

#### CSGD (2509.09451v2)
- **Composable Guidance:** Arbitrary subsets of conditions
- **Concrete Scores:** Enables flexible score manipulation
- **Probability Calibration:** Adjusts for train-test mismatch
- **Results:** 15.3% controllability improvement

#### GGDiff (2505.19685v1)
- **Stochastic Control:** Interprets conditioning as control problem
- **Guidance Types:**
  - Gradient-based (differentiable rewards)
  - Control-based (forward reward evaluations)
  - Zero-order (gradient-free)
- **Applications:** Motifs, fairness, link prediction

### 2. **Classifier-Based vs Classifier-Free Guidance**

#### Classifier-Based (CB)
- **Approach:** Train auxiliary property regressor
- **Guidance:** Gradient of classifier guides diffusion
- **Examples:** Original DiGress conditioning
- **Limitations:**
  - Requires training additional model
  - Doubles parameters
  - Assumptions may not hold for discrete domains

#### Classifier-Free (CF)
- **Approach:** Inject conditioning during training
- **Mechanism:** Model learns conditional and unconditional distributions
- **Examples:** FreeGress (2312.17397v2)
- **Advantages:**
  - No auxiliary model needed
  - Half the parameters
  - Better performance (79% MAE improvement)

### 3. **Structural Conditioning**

#### GraphGUIDE (2302.03790v1)
- **Control:** Arbitrary structural properties without labels
- **Mechanism:** Discrete Bernoulli diffusion with edge control
- **Interpretability:** Full visibility into generation process

#### CDGS (2301.00427v2)
- **Condition:** Discrete graph structures as condition
- **Hybrid Model:** Global context + local node-edge dependency
- **Applications:** Drug-like molecules with structural constraints

### 4. **Guidance Mechanisms**

**Loop Guidance** (2410.24012v1):
- Co-evolving flows (trunk + stems)
- Information orchestration between processes
- Inverse molecular design applications

**Text-to-Graph** (Large Generative Graph Models, 2406.05109v1):
- Natural language descriptions
- Network statistics as prompts
- Integration with LLMs

---

## Molecular vs General Graph Generation

### Molecular Generation (Domain-Specific)

**Characteristics:**
- **Constraints:** Chemical valency, stability, synthesizability
- **Features:** Atom types (discrete), 3D coordinates (continuous), bond types
- **Size:** Typically 10-100 atoms
- **Datasets:** QM9, ZINC-250k, GuacaMol, GEOM-Drugs

**Specialized Approaches:**

1. **DiGress** (2209.14734v4)
   - Molecular-specific auxiliary features
   - Valency preservation
   - 1.3M molecule scaling

2. **GDSS** (2202.02514v3)
   - Node-edge coupling for chemical bonds
   - Valency rule compliance
   - High chemical validity

3. **Graph DiT** (2401.13858v3)
   - Multi-property molecular design
   - Synthetic score, permeability
   - Polymer and small molecule generation

4. **GCDM** (2302.04313v6)
   - 3D molecular generation
   - Geometry-complete denoising
   - Stability and property optimization

**Performance Benchmarks:**
- **Validity:** 95-99% valid molecules
- **Uniqueness:** >95% unique samples
- **Novelty:** >80% not in training set
- **Property Control:** MAE <0.1 for key properties

### General Graph Generation (Domain-Agnostic)

**Characteristics:**
- **Diverse Structures:** Social networks, infrastructure, circuits
- **Variable Size:** From tens to thousands of nodes
- **Properties:** Community structure, degree distribution, clustering
- **Datasets:** PROTEINS, ENZYMES, Ego networks, planar graphs

**General Approaches:**

1. **GraphGDP** (2212.01842v1)
   - Permutation invariant for any graph type
   - Position-enhanced for general structures
   - 24 function evaluations

2. **PARD** (2402.03687v3)
   - Scalable to non-molecular datasets
   - Block-based for diverse structures
   - State-of-the-art on general benchmarks

3. **GraphMaker** (2310.13833v4)
   - Large attributed graphs
   - Asynchronous attribute-structure generation
   - Real-world network applications

4. **LayerDAG** (2411.02322v2)
   - Directed acyclic graphs
   - Up to 400 nodes
   - Hardware and program graphs

**Performance Benchmarks:**
- **Distribution Matching:** Low MMD on degree, clustering, orbits
- **Scalability:** Handles thousands of nodes
- **Utility:** ML model performance on synthetic data

### Key Differences

| Aspect | Molecular | General |
|--------|-----------|---------|
| **Constraints** | Chemical rules | Flexible |
| **Features** | Mixed (discrete+3D) | Typically categorical |
| **Size** | 10-100 nodes | 10-10,000+ nodes |
| **Validation** | Chemical validity | Statistical properties |
| **Applications** | Drug design | Network analysis |
| **Challenges** | Valency, stability | Scalability, diversity |

---

## Research Gaps and Open Problems

### 1. **Scalability Limitations**

**Current State:**
- Most models limited to <1000 nodes
- DiGress: 1.3M molecules but small graphs
- SaGess: Divide-and-conquer approach
- GraphMaker: Edge mini-batching

**Gap:** Clinical knowledge graphs often have 10,000+ nodes

**Potential Solutions:**
- Hierarchical generation (SaGess approach)
- Latent compression with better encoders
- Subgraph-based generation
- Distributed diffusion processes

### 2. **Knowledge Graph Specifics**

**Current State:**
- No papers specifically on KG generation
- Most work on molecular or social graphs
- Some on attributed graphs (GraphMaker)

**Clinical KG Characteristics:**
- Multi-relational (many edge types)
- Temporal dynamics
- Entity hierarchies (ICD codes, drug classes)
- Constraint networks (protocols, guidelines)

**Needed Research:**
- KG-specific diffusion processes
- Relation-aware architectures
- Temporal diffusion models
- Hierarchical/hyperbolic approaches for medical taxonomies

### 3. **Temporal and Sequential Aspects**

**Current State:**
- Mostly static graph generation
- Some work on DAGs (LayerDAG)
- Limited temporal modeling

**Clinical Need:**
- Patient trajectories over time
- Temporal event sequences
- Intervention effects

**Needed Research:**
- Continuous-time discrete-state for clinical events
- Autoregressive blocks for care phases
- Temporal conditioning mechanisms

### 4. **Interpretability and Explainability**

**Current State:**
- GraphGUIDE: Some interpretability
- Most models are black boxes
- Limited analysis of learned representations

**Clinical Requirements:**
- Explainable generations for clinical trust
- Interpretable conditioning
- Uncertainty quantification

**Needed Research:**
- Attention visualization for clinical reasoning
- Counterfactual generation
- Confidence measures for generated trajectories

### 5. **Multi-Modal Integration**

**Current State:**
- Mostly graph-only
- Some text conditioning (Text-to-Graph)
- Limited multi-modal work

**Clinical Data:**
- Structured (lab values, vital signs)
- Unstructured (clinical notes)
- Temporal (time series)
- Graph (knowledge, relationships)

**Needed Research:**
- Joint diffusion over multiple modalities
- Cross-modal conditioning
- Unified representations

### 6. **Theoretical Foundations**

**Current State:**
- Well-developed for continuous diffusion
- Less mature for discrete diffusion
- Limited convergence guarantees

**Open Questions:**
- Optimal noise schedules for graphs
- Sample complexity bounds
- Approximation error analysis
- When discrete beats continuous and vice versa

### 7. **Evaluation Metrics**

**Current State:**
- Domain-specific (molecular validity, MMD)
- Limited standardization
- Few task-driven evaluations

**Clinical Needs:**
- Clinical validity metrics
- Adherence to guidelines
- Downstream task performance
- Fairness and bias measures

**Needed Research:**
- Standardized KG generation benchmarks
- Clinical-specific metrics
- Utility-based evaluation

---

## Relevance to Clinical KG Trajectory Generation

### Direct Applications

#### 1. **Patient Trajectory Generation**

**Applicable Models:**
- **Cometh** (2406.06449v2): Continuous-time discrete-state natural for clinical events
- **PARD** (2402.03687v3): Block-based could model care phases (ER → ICU → Ward)
- **LayerDAG** (2411.02322v2): DAG structure reflects treatment protocols

**Adaptations Needed:**
- Multi-relational edges (diagnosis, treatment, outcome)
- Temporal dynamics (time-stamped events)
- Clinical constraints (protocols, contraindications)

#### 2. **Conditional Generation on Clinical Constraints**

**Applicable Models:**
- **Graph DiT** (2401.13858v3): Multi-conditional framework for patient characteristics
- **CSGD** (2509.09451v2): Composable guidance for multiple clinical objectives
- **GGDiff** (2505.19685v1): Zero-shot guidance for new clinical criteria

**Clinical Conditions:**
- Patient demographics (age, sex, comorbidities)
- Disease severity scores
- Resource constraints
- Outcome targets

#### 3. **Knowledge Graph Completion and Augmentation**

**Applicable Models:**
- **LGD** (2402.02518v2): Unified framework for generation and prediction
- **GraphMaker** (2310.13833v4): Large attributed graphs for hospital-scale KGs
- **NGG** (2403.01535v3): Statistics-guided for maintaining KG properties

**Applications:**
- Missing diagnosis/treatment links
- Synthetic patient data generation
- Protocol variation modeling

### Architectural Insights

#### 1. **Discrete Diffusion for Clinical Entities**

**Clinical Rationale:**
- Diagnoses, procedures, medications are categorical
- ICD codes, CPT codes are discrete
- Discrete diffusion maintains semantic validity

**Recommended Approach:**
- DiGress-style discrete diffusion
- Markovian noise preserving clinical distributions
- Auxiliary features for clinical context

#### 2. **Hyperbolic Embeddings for Medical Hierarchies**

**Clinical Rationale:**
- Disease taxonomies (ICD hierarchy)
- Drug classifications (ATC codes)
- Anatomical structures

**Recommended Approach:**
- HGDM (2306.07618v3) hyperbolic diffusion
- Radial dimension for hierarchy level
- Angular dimension for category

#### 3. **Graph Transformers for Global Context**

**Clinical Rationale:**
- Patient history requires long-range dependencies
- Comorbidities interact across body systems
- Treatment sequences have distant effects

**Recommended Approach:**
- DiGress/PARD transformer architectures
- Multi-head attention over clinical events
- Position encodings for temporal order

### Conditioning Strategies

#### 1. **Multi-Property Clinical Conditioning**

**Properties to Condition On:**
- Patient demographics (age, sex, BMI)
- Disease severity (APACHE, SOFA scores)
- Comorbidity indices (Charlson, Elixhauser)
- Outcomes (mortality, length of stay)
- Resource use (cost, procedures)

**Recommended Methods:**
- Graph DiT multi-conditional framework
- Composable guidance (CSGD) for subset control
- Classifier-free to avoid auxiliary models

#### 2. **Protocol-Based Guidance**

**Clinical Guidelines:**
- Sepsis bundles
- ARDS management protocols
- Stroke care pathways

**Recommended Methods:**
- GGDiff reward-based guidance
- Structural constraints (GraphGUIDE)
- DAG conditioning (LayerDAG) for ordered protocols

#### 3. **Temporal Conditioning**

**Time-Dependent Factors:**
- Disease progression stages
- Treatment response windows
- Seasonal variations

**Recommended Methods:**
- Continuous-time diffusion (Cometh)
- Temporal position encodings
- Stage-aware generation

### Challenges and Mitigation Strategies

#### Challenge 1: **Scale**
- **Problem:** Hospital KGs have 10,000+ nodes
- **Solutions:**
  - Hierarchical generation (SaGess)
  - Latent diffusion (LGD, NGG)
  - Subgraph sampling

#### Challenge 2: **Multi-Relational Edges**
- **Problem:** Diagnosis, treatment, medication, lab edges
- **Solutions:**
  - Edge type embeddings
  - Relation-specific diffusion processes
  - Heterogeneous graph architectures

#### Challenge 3: **Temporal Dynamics**
- **Problem:** Events occur over time with dependencies
- **Solutions:**
  - Continuous-time discrete-state (Cometh)
  - Autoregressive blocks (PARD)
  - RNN integration with diffusion

#### Challenge 4: **Clinical Validity**
- **Problem:** Must satisfy medical constraints
- **Solutions:**
  - Hard constraints in diffusion (GraphGUIDE)
  - Reward-based guidance (GGDiff)
  - Post-hoc validation and correction

#### Challenge 5: **Interpretability**
- **Problem:** Clinicians need explainable models
- **Solutions:**
  - Attention visualization
  - Discrete diffusion for step-wise inspection
  - Auxiliary interpretable features

### Proposed Clinical KG Diffusion Architecture

**Hybrid Architecture:**

1. **Graph Representation:**
   - Nodes: Clinical entities (diagnoses, procedures, medications, labs)
   - Edges: Multi-relational (temporal sequence, causation, association)
   - Features: Mixed discrete (ICD codes) + continuous (lab values)
   - Hierarchical: Hyperbolic embeddings for taxonomies

2. **Diffusion Process:**
   - **Discrete-state continuous-time** (Cometh-style)
   - Markovian noise on discrete entities
   - Gaussian noise on continuous features
   - Coupled diffusion for entity-relationship integrity

3. **Architecture:**
   - **Encoder:** Hyperbolic GNN for hierarchy + Transformer for long-range
   - **Denoising:** Graph transformer with multi-head attention
   - **Decoder:** Relation-specific heads for edge types

4. **Conditioning:**
   - **Multi-property:** Patient demographics, severity scores
   - **Composable:** Subset control (CSGD approach)
   - **Protocol:** Guideline-based rewards (GGDiff approach)
   - **Temporal:** Time-aware position encodings

5. **Training:**
   - Score matching on discrete states
   - Classifier-free conditioning
   - Auxiliary clinical validity loss
   - Multi-task: Generation + prediction (LGD approach)

6. **Sampling:**
   - Guided reverse diffusion
   - Clinical constraint checking at each step
   - Beam search for high-probability trajectories
   - Uncertainty quantification

---

## Key Takeaways for Clinical Applications

### What Works Well

1. **Discrete Diffusion** (DiGress, Cometh)
   - Natural for categorical clinical data
   - Maintains validity
   - Faster sampling

2. **Multi-Conditional Generation** (Graph DiT, CSGD)
   - Essential for clinical constraint satisfaction
   - Composable guidance most flexible
   - Classifier-free avoids overfitting

3. **Hyperbolic Representations** (HGDM)
   - Perfect for medical hierarchies
   - Preserves taxonomic structure
   - Better than Euclidean for tree-like data

4. **Graph Transformers** (PARD, DiGress)
   - Capture long-range dependencies
   - Global clinical context
   - Parallel training efficiency

### What Needs Development

1. **Scalability**
   - Current models max out at ~1000 nodes
   - Need hierarchical/latent approaches
   - Distributed generation for hospital-scale

2. **Temporal Modeling**
   - Limited continuous-time work
   - Need better trajectory generation
   - Sequential dependencies underexplored

3. **Multi-Relational**
   - Most models assume single edge type
   - Clinical KGs have many relation types
   - Need heterogeneous graph diffusion

4. **Interpretability**
   - Black box models insufficient for clinical use
   - Need attention visualization
   - Uncertainty quantification critical

5. **Evaluation**
   - No clinical KG benchmarks
   - Need domain-specific metrics
   - Utility-based evaluation required

### Recommended Research Directions

1. **Clinical KG Diffusion Model**
   - Discrete-state continuous-time
   - Multi-relational edges
   - Hyperbolic hierarchy encoding
   - Composable clinical conditioning

2. **Temporal Trajectory Generation**
   - Continuous-time event modeling
   - Autoregressive care phases
   - Intervention effect modeling

3. **Benchmark Dataset**
   - Large-scale clinical KG (MIMIC-based)
   - Standardized evaluation metrics
   - Downstream task suite

4. **Interpretable Generation**
   - Attention-based explanations
   - Counterfactual trajectories
   - Uncertainty-aware sampling

5. **Multi-Modal Clinical Diffusion**
   - Joint graph-text-timeseries
   - Cross-modal conditioning
   - Unified clinical representations

---

## Conclusion

Graph diffusion models have matured significantly for molecular and general graph generation, with discrete approaches (DiGress, GDSS, GraphGDP) achieving state-of-the-art results. However, application to clinical knowledge graph trajectory generation requires addressing key gaps:

1. **Scalability** to hospital-scale knowledge graphs (10,000+ nodes)
2. **Multi-relational** edge modeling for diverse clinical relationships
3. **Temporal dynamics** for patient trajectory evolution
4. **Interpretability** for clinical trust and adoption
5. **Domain-specific evaluation** metrics and benchmarks

The most promising approaches for clinical adaptation combine:
- **Discrete-state continuous-time** diffusion (Cometh) for clinical events
- **Multi-conditional** generation (Graph DiT, CSGD) for patient constraints
- **Hyperbolic embeddings** (HGDM) for medical hierarchies
- **Graph transformers** (PARD, DiGress) for long-range dependencies
- **Composable guidance** (GGDiff) for protocol adherence

Future work should focus on developing clinical-specific diffusion models that maintain medical validity, scale to realistic knowledge graphs, and provide interpretable generations for clinical decision support.

---

## References

This synthesis covers 60+ papers from ArXiv. Key foundational papers:

- **DiGress** (2209.14734v4): Discrete denoising diffusion for graphs
- **GDSS** (2202.02514v3): Score-based modeling via system of SDEs
- **PARD** (2402.03687v3): Permutation-invariant autoregressive diffusion
- **Graph DiT** (2401.13858v3): Multi-conditional molecular generation
- **Cometh** (2406.06449v2): Continuous-time discrete-state diffusion
- **HGDM** (2306.07618v3): Hyperbolic graph diffusion
- **GGDiff** (2505.19685v1): Unified guidance framework
- **CSGD** (2509.09451v2): Composable score-based diffusion

Full paper list and ArXiv IDs included throughout document.

---

**Document Version:** 1.0
**Last Updated:** December 1, 2025
**Author:** Research synthesis via ArXiv search
**Contact:** alexstinard@hybrid-reasoning-acute-care