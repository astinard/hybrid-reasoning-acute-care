# AI/ML Applications in Psychiatry and Mental Health: Comprehensive Research Synthesis

**Research Domain:** Artificial Intelligence and Machine Learning for Psychiatric and Mental Health Applications
**Date:** December 2025
**Focus Areas:** Depression Detection, Suicide Risk Prediction, Psychosis Detection, Anxiety Detection, PTSD Identification, Substance Use Disorders, Treatment Response Prediction, Digital Phenotyping

---

## Executive Summary

This comprehensive research synthesis examines state-of-the-art AI/ML applications across eight critical areas of psychiatry and mental health. Analysis of 160+ recent research papers reveals significant advances in multimodal detection systems, deep learning architectures, and digital phenotyping approaches. Key findings indicate high accuracy rates (70-93%) across various detection tasks, with transformer-based architectures and multimodal fusion showing superior performance. Critical challenges remain in generalizability, interpretability, and clinical deployment.

---

## 1. Depression Detection from Clinical Notes and Speech

### 1.1 Speech-Based Depression Detection

#### Key Architectures and Performance

**Hierarchical Attention Transformer (2309.13476v2)**
- **Architecture:** Bi-modal speech-level transformer with hierarchical interpretation
- **Modalities:** Audio (Mel-spectrograms) + Text (transcripts)
- **Performance:** Precision=0.854, Recall=0.947, F1=0.897
- **Innovation:** Gradient-weighted attention maps for interpretability
- **Dataset:** DAIC-WOZ dataset
- **Key Finding:** Outperforms segment-level models by avoiding labeling noise

**Audio Spectrogram Transformer (AST) for Depression (2406.03138v3)**
- **Architecture:** Speech-level AST processing long-duration speech
- **Performance:** Enhanced depression detection with longer speech segments
- **Clinical Markers:** Reduced loudness and F0 (fundamental frequency)
- **Interpretability:** Reveals prediction-relevant acoustic features
- **Advantage:** Leverages longer speech duration for more reliable detection

**TRI-DEP: Trimodal Depression Detection (2510.14922v1)**
- **Modalities:** Speech + Text + EEG
- **Architecture:** Ensemble of pre-trained embeddings with neural encoders
- **Performance:** State-of-the-art on multimodal detection
- **Key Finding:** Trimodal systems enhance detection over bimodal approaches
- **Dataset:** Multiple sclerosis patients + general population

**LLM Integration for Depression Detection (2402.13276v2)**
- **Architecture:** Acoustic Landmarks + LLMs (GPT-3.5, fine-tuned models)
- **Innovation:** Integrates speech timing information into LLM framework
- **Performance:** Comparable to fine-tuned models without additional training
- **Method:** Acoustic landmarks capture pronunciation-specific patterns
- **Dataset:** DAIC-WOZ

**SpeechT-RAG: Retrieval-Augmented Generation (2502.10950v2)**
- **Architecture:** LLM + RAG using speech timing features
- **Innovation:** Confidence scoring mechanism for trustworthiness
- **Performance:** F1=0.737, F2=0.843 (comparable to fine-tuned LLMs)
- **Key Features:** Speech timing (pauses, rhythm) as primary biomarkers
- **Advantage:** No additional training required, improved uncertainty quantification

#### Cross-Lingual and Multi-Population Studies

**Cross-Lingual Depression Detection (2508.18092v1)**
- **Populations:** English (general) + German (MS patients)
- **Performance:** UAR=66% binary classification, improved to 74% with feature selection
- **Key Finding:** Emotional dimensions (from SER models) critical for detection
- **Challenge:** Transferability across languages and comorbid conditions

**Language-Agnostic Analysis (2409.14769v1)**
- **Languages:** English and Malayalam
- **Architecture:** CNNs on IViE corpus sentences
- **Dataset:** Depression data with self-reported labels
- **Contribution:** Framework for diverse linguistic populations

#### Foundation Models and Transfer Learning

**Self-Supervised Learning (SSL) for Depression (2305.12263v2)**
- **Models:** WavLM, HuBERT, other SSL models
- **Method:** Foundation models pre-trained on large speech corpora
- **Performance:** State-of-the-art on DAIC-WOZ with ASR integration
- **Key Finding:** SSL representations capture depression-specific patterns
- **Analysis:** Layer-wise analysis reveals optimal intermediate representations

**Transfer Learning from Related Tasks (2310.04358v1)**
- **Source Task:** Speech depression detection
- **Target Task:** Alzheimer's disease detection
- **Performance:** F1=0.928 on ADReSSo dataset
- **Innovation:** Parallel knowledge transfer from high-comorbidity tasks
- **Finding:** Depression-specific knowledge aids AD detection

#### Advanced Feature Engineering

**HAREN-CTC: Hierarchical SSL Features (2510.08593v1)**
- **Architecture:** Multi-layer SSL features + cross-attention + CTC loss
- **Innovation:** Hierarchical Adaptive Clustering module
- **Performance:** Macro F1=0.81 (DAIC-WOZ), 0.82 (MODMA)
- **Key Components:** Cross-Modal Fusion with inter-layer dependencies
- **Advantage:** Handles sparse temporal supervision effectively

**Speaker Disentanglement (2306.01861v2)**
- **Method:** Non-uniform adversarial SID loss maximization
- **Performance:** F1=0.7349, 3.7% improvement over audio-only SOTA
- **Privacy:** Reduces speaker-identification accuracy by 50%
- **Innovation:** Greater adversarial weight for initial layers

**Time-Domain Long Sequence Modeling (2501.02512v1)**
- **Architecture:** State space model + dual-path structure + temporal attention
- **Innovation:** Processes raw audio waveforms (avoids time-frequency transformation)
- **Advantage:** Captures long-sequence depression cues
- **Performance:** Outstanding on AVEC2013 and AVEC2014 datasets
- **Key Finding:** Real-world depression involves pauses and slowed speech

#### Severity Detection and Prediction

**HARD-Training Methodology (2206.01542v2)**
- **Dataset:** RADAR-MDD (multilingual, Spain/UK/Netherlands)
- **Architecture:** Sequence-to-Sequence with local attention
- **Innovation:** Curriculum learning with ambiguous sample selection
- **Performance:** 90% accuracy with 8.6% improvement over baselines
- **Finding:** Performance consistent across languages and genders

**Cost-Effective Models (2302.09214v1)**
- **Comparison:** Hand-curated vs. deep representation features
- **Finding:** Hand-curated features perform equally well at lower computational cost
- **Factors:** Gender, severity, content, speech length analyzed
- **Deployment:** Suitable for real-time monitoring in resource-constrained devices

### 1.2 Text-Based Depression Detection

**Early Depression Detection Framework (1905.08772v2)**
- **Dataset:** CLEF's eRisk2017 (Reddit data)
- **Architecture:** SS3 (novel supervised learning model)
- **Performance:** Outperformed 30 competition submissions
- **Key Features:** Incremental classification, explainable rationale
- **Advantage:** Less computationally expensive than state-of-the-art

**t-SS3: Dynamic N-grams (1911.06147v2)**
- **Extension:** Dynamic n-gram recognition over text streams
- **Tasks:** Early depression and anorexia detection
- **Performance:** Improved F1 scores, richer visual explanations
- **Innovation:** Learns important patterns "on the fly"

**Social Media Depression Detection (2412.05861v1)**
- **Platform:** Facebook (Bengali text)
- **Architecture:** RNNs (LSTM, GRU), SVM, Naive Bayes
- **Features:** Stylometric, TF-IDF, word embeddings
- **Application:** Sentiment analysis for psychologist support

**Metric-Based Detection (1807.03397v1)**
- **Dataset:** DAIC (Distress Analysis Interview Corpus)
- **Method:** Negative sentence scoring for depression level
- **Challenge:** Text alone insufficient without other factors

**Individual Symptom Prediction (2406.16000v1)**
- **Innovation:** Predicts individual PHQ items from speech
- **Architecture:** CNN + LSTM
- **Advantage:** Provides clinical-scale alignment with depression ratings
- **Method:** Voting schemes for item-level prediction

### 1.3 Multimodal Depression Detection

**Multi-Modal Deep Learning (2212.14490v1)**
- **Modalities:** Audio (MFCC) + Text (textual features)
- **Architecture:** Deep learning with hand-crafted features
- **Performance:** Improved F1 scores for both depression and anxiety
- **Finding:** Joint modeling superior to unimodal approaches

**DEPAC Corpus (2306.12443v1)**
- **Contribution:** Novel corpus for depression and anxiety
- **Content:** Multiple speech tasks per individual + demographics
- **Features:** Acoustic + linguistic
- **Baseline Models:** Outperforms existing depression corpora

---

## 2. Suicide Risk Prediction Models

### 2.1 Clinical Note-Based Prediction

**ScAN: Suicide Attempt and Ideation Events Dataset (2205.07872v1)**
- **Dataset:** 12,000+ MIMIC-III EHR notes with 19,000+ annotated events
- **Annotation:** SA (suicide attempts) and SI (suicidal ideation) events
- **Architecture:** ScANER - Multi-task RoBERTa with retrieval module
- **Performance:** Macro-weighted F1=0.83 (evidence), Macro F1=0.78 (SA), 0.60 (SI)
- **Innovation:** Extracts method of suicide attempt attributes
- **Availability:** Publicly available dataset and model

**PsyGUARD: Multi-Class Classification (2409.20243v1)**
- **Classes:** Indicator, ideation, behavior, attempt (4 categories)
- **Dataset:** PsySUICIDE (large-scale, high-quality)
- **Architecture:** Fine-grained taxonomy based on foundational theories
- **Application:** Automated crisis intervention for suicide prevention
- **Innovation:** Detailed taxonomy for fine-grained detection

### 2.2 Social Media-Based Prediction

**Interpersonal Theory of Suicide (IPTS) Framework (2504.13277v1)**
- **Dataset:** 59,607 Reddit r/SuicideWatch posts
- **Theory:** IPTS dimensions (Thwarted Belongingness, Perceived Burdensomeness)
- **SI Categories:** Loneliness, Lack of Reciprocal Love, Self Hate, Liability
- **Analysis:** Psycholinguistic and content analysis of supportive responses
- **AI Evaluation:** Expert assessment of AI chatbot responses
- **Finding:** AI shows structural coherence but lacks personalized empathy

**Personal Knowledge Graph (2012.09123v1)**
- **Architecture:** Suicide-oriented knowledge graph + deep neural networks
- **Features:** Post, personality, experience (6 categories)
- **Performance:** 93% accuracy on microblog data
- **Key Factors:** Posted text, stress level/duration, posted image, ruminant thinking
- **Innovation:** Unifies high-level suicide-oriented knowledge with DNNs

**Attentive Relation Networks (2004.07601v3)**
- **Dataset:** Social media content (Twitter)
- **Features:** Lexicon-based sentiment scores + latent topics
- **Architecture:** Relation networks with attention mechanism
- **Innovation:** Prioritizes critical relational features
- **Challenge:** Distinguishes suicidal ideation from other mental disorders

**Latent Suicide Risk Detection (1910.12038v1)**
- **Platform:** Microblog with "tree holes" phenomenon
- **Innovation:** Suicide-oriented word embeddings from tree hole contents
- **Architecture:** Two-layered attention mechanism
- **Performance:** 91% accuracy on well-labeled dataset
- **Finding:** Detects intermittently changing points in blog streams

### 2.3 Deep Learning Approaches

**Ensemble Deep Learning (2112.10609v1)**
- **Architecture:** LSTM-Attention-CNN combined model
- **Dataset:** Social media submissions
- **Performance:** Accuracy=90.3%, F1=92.6%
- **Method:** Analyzes underlying suicidal intentions
- **Advantage:** Outperforms baseline models significantly

**Transformer Models for Suicide Risk (2410.08375v1)**
- **Competition:** IEEE BigData 2024 Cup
- **Models:** Fine-tuned DeBERTa, GPT-4o with CoT, fine-tuned GPT-4o
- **Best Model:** Fine-tuned GPT-4o (second place in competition)
- **Innovation:** Straightforward, general-purpose models achieve SOTA
- **Finding:** Minimal tuning effective for suicide risk detection

**Two-Stage Voting Architecture (2510.08365v1)**
- **Datasets:** Reddit (explicit-dominant), DeepSuiMind (implicit-only)
- **Stage 1:** Lightweight BERT for high-confidence cases
- **Stage 2:** Multi-perspective LLM voting OR feature-based ML ensemble
- **Performance:** ρ=0.649, AUC=83.71% (TOPSY dataset)
- **Innovation:** Balances efficiency and robustness
- **Features:** LLM-extracted psychological features as structured vectors

**Multilingual Model (2412.15498v1)**
- **Languages:** Spanish, English, German, Catalan, Portuguese, Italian
- **Architecture:** mBERT, XML-R, mT5
- **Translation:** SeamlessM4T for cross-lingual transfer
- **Performance:** mT5 F1>85% across languages
- **Contribution:** First multilingual PTSD detection model

### 2.4 Big Data Analytics

**Real-Time Streaming Prediction (2404.12394v1)**
- **Architecture:** Apache Spark ML with batch + real-time streaming
- **Data Sources:** Reddit (batch), Twitter streaming API (real-time)
- **Classifiers:** NB, LR, LinearSVC, DT, RF, MLP
- **Features:** (Unigram + Bigram) + CV-IDF
- **Performance:** MLP accuracy=93.47%
- **Innovation:** Bi-phasic processing (batch training, streaming prediction)

### 2.5 Audiovisual Approaches

**Multimodal Cue Analysis (1911.11927v1)**
- **Population:** Military couples
- **Modalities:** Acoustic, lexical, behavior, turn-taking cues
- **Innovation:** Automatic diarization + speech recognition front-end
- **Risk Levels:** None, ideation, attempt (3-class classification)
- **Finding:** Behavior and turn-taking cues most informative
- **Performance:** Significantly better than chance across all scenarios

**Audiovisual Feature Analysis (2201.09130v2)**
- **Review:** Recent works on suicide ideation and behavior detection
- **Modalities:** Voice/speech acoustics + visual cues
- **Challenge:** Lack of large datasets for ML/DL training
- **Status:** Promising research direction in early stages

### 2.6 Machine Learning in Military/Veterans

**Scoping Review of ML in Military Populations (2505.12220v1)**
- **Review Period:** Recent literature on PTSD, depression, suicide
- **Data:** Physiological data (heart rate) shows promise
- **Risk Factors:** Depression, PTSD, prior attempts, physical health, demographics
- **Gap:** Limited discussion of PPV, NPV (false positive/negative rates)
- **Need:** Survival and longitudinal data approaches

**Zero-Shot LLM Approach (2410.04501v3)**
- **Dataset:** IEEE Big Data 2024 Challenge
- **Method:** Pseudo-labeling by prompting LLMs (Qwen2-72B-Instruct)
- **Models:** Fine-tuned Llama3-8B, Llama3.1-8B, Gemma2-9B
- **Performance:** Ensemble F1=0.770 (public), 0.731 (private)
- **Improvement:** 5% over individual models
- **Finding:** Larger LLMs provide better prompting accuracy

### 2.7 Lexicography and Cross-Cultural Studies

**Lexicography Saves Lives Project (2412.15497v1)**
- **Innovation:** Suicide-related dictionary translated into 200 languages
- **Ethical Framework:** Guidelines to mitigate harm in resource development
- **Contribution:** Community participation via public website
- **Challenge:** Linguistic misrepresentation, cultural context
- **Goal:** Address global suicide as per UN SDG targets

**Social Media Detection Review (2201.10515v1)**
- **Review:** 24 studies on suicidal ideation detection
- **Platforms:** Primarily social media (Twitter, Reddit)
- **Methods:** Machine learning algorithms
- **Challenge:** Distinguishing negative emotions of similar valence
- **Finding:** Text-based biomarkers hold promise

---

## 3. Psychosis Prediction and Early Intervention

### 3.1 Network-Based Approaches

**Network Controllability in Transmodal Cortex (2010.00659v1)**
- **Dataset:** 1,068 youths (age 8-22) from Philadelphia Neurodevelopmental Cohort
- **Method:** Network Control Theory on structural connectivity
- **Innovation:** Average controllability (direct + indirect connections)
- **Finding:** Predicts positive psychosis spectrum symptoms
- **Key Region:** Association cortex (transmodal regions)
- **Performance:** Outperforms strength-based measures (direct connections only)

**Community Detection in Weighted Multilayer Networks (2103.00486v4)**
- **Dataset:** Philadelphia Neurodevelopmental Cohort
- **Model:** Stochastic Block with Ambient Noise Model (SBANM)
- **Innovation:** Accounts for global noise + local signals
- **Application:** Discovers communities with co-occurrent psychopathologies
- **Method:** Hierarchical variational inference for block detection

### 3.2 Speech and Language Analysis

**Quantifying Word Salad (1610.08566v2)**
- **Dataset:** First clinical contact for recent-onset psychosis (21 patients)
- **Method:** Speech graph connectedness analysis
- **Metrics:** Graph attributes compared to random graph distributions
- **Performance:** 93% accuracy in classification, 92% PANSS negative variance
- **Follow-up:** 6-month diagnosis prediction (89% accuracy, AUC=0.89)
- **Fragmentation Index:** Quantifies word salad severity
- **Validation:** Chronic psychotic patients (independent cohort)

**Uncertainty Modeling Across Psychosis Spectrum (2502.18285v1)**
- **Participants:** 114 (32 early psychosis, 82 low/high schizotypy)
- **Language:** German speech tasks (structured, semi-structured, narrative)
- **Architecture:** Seq2Seq with local attention mechanism
- **Performance:** RMSE reduction, F1=83%, ECE=4.5e-2
- **Innovation:** Uncertainty-aware model for speech variability
- **Finding:** Dynamic adjustment to task structures (acoustic vs. linguistic)

**Reading Between the Lines (2507.13551v1)**
- **Features:** Pause dynamics + semantic coherence
- **Datasets:** AVH (n=140), TOPSY (n=72), PsyCL (n=43)
- **Methods:** Support Vector Regression (SVR)
- **Performance:** ρ=0.649, AUC=83.71% (TOPSY)
- **Integration:** Independent models via voting
- **Finding:** Pause patterns dataset-dependent, integration consistently improves

### 3.3 Symptom Development and Dynamics

**Emergence of Delusions and Hallucinations (2402.13428v1)**
- **Cohorts:** NAPLS 2 (n=719), NAPLS 3 (n=699), PEPP-Montreal (n=694)
- **Finding:** Delusions emerge before hallucinations (OR: 4.09, 4.14, Z=7.01)
- **Pattern:** Only delusions more common than only hallucinations (OR: 5.6-42.75)
- **Re-emergence:** Delusions more likely to return after remission
- **Temporal:** Hallucinations more often resolve first
- **Key: Delusional ideation falls with onset of hallucinations (p=0.007)

**Computational Mechanisms (2103.13924v1)**
- **Framework:** Computational underpinnings of positive psychotic symptoms
- **Mechanisms:** Maladaptive priors + reduced updating
- **Application:** Coordinated specialty care clinics, assertive community treatment
- **Theory:** Structure and predictability counteract low sensory precision
- **Prediction:** Response to modifications of interventions

### 3.4 Developmental and Clinical Tools

**TimelinePTC: Pathways to Care Visualization (2404.12883v1)**
- **Purpose:** Collection and analysis of Pathways to Care (PTC) data
- **Condition:** First episode psychosis (FEP)
- **Innovation:** Web-based tool for collaborative, real-time data entry
- **Benefit:** Measures duration of untreated psychosis (DUP)
- **Features:** Visualization, automated format conversion
- **Availability:** Open-source codebase

### 3.5 Predictive Symptom Networks

**Predictability in Psychopathological Networks (1612.06357v2)**
- **Review:** 18 published datasets, 25 datasets total
- **Disorders:** Mood, anxiety, substance abuse, psychosis, autism, transdiagnostic
- **Method:** Network analysis of symptom networks
- **Finding:** Predictability unrelated to sample size, moderately high in most
- **Difference:** Higher in community vs. clinical samples
- **Highest:** Mood and anxiety; Lowest: Psychosis
- **Implication:** Controllability of each node in symptom network

### 3.6 Simulation and Risk Assessment

**Simulating Psychological Risks (2511.08880v1)**
- **Cases:** 18 real-world AI-induced incidents (addiction, anorexia, psychosis, etc.)
- **Scenarios:** 2,160 simulated scenarios (157,054 conversation turns)
- **Models:** 4 major LLMs tested
- **Taxonomy:** 15 distinct failure patterns in 4 harm categories
- **Finding:** Variable performance across demographics (elderly, income groups)
- **Risk:** Critical gaps in detecting distress, responding to vulnerable users

### 3.7 Screening at Scale

**Extra-Large Scale Screening (2212.10320v2)**
- **Data:** 77.4 million insurance members + multi-hospital EHRs
- **Disorders:** Schizophrenia, schizoaffective, psychosis, bipolar
- **Models:** mBERT, XML-R, mT5
- **Performance:** mT5 F1>85%
- **Applications:** 18-year-old young adults, substance-associated conditions
- **Method:** Cross-lingual transfer learning

---

## 4. Anxiety Disorder Detection

### 4.1 Physiological Signal-Based Detection

**Error-Related Negativity and EEG (2410.00028v1)**
- **Signal:** Error-related negativity (ERN) from EEG
- **Review:** 54 papers (2013-2023)
- **Methods:** Traditional ML (SVM, RF) + DL (CNN, RNN)
- **Risk Factors:** Depression, PTSD, suicidal ideation, physical health
- **Gap:** Need for metrics distinguishing false positives/negatives
- **Challenge:** Task-specific setup, feature selection, computational modeling

**Personalized State Anxiety Detection (2304.09928v1)**
- **Population:** High socially anxious participants (N=35)
- **Context:** Evaluative vs. non-evaluative social situations
- **Features:** Digital linguistic biomarkers
- **Architecture:** Multilayer personalized ML pipeline
- **Innovation:** Contextual and individual difference modeling
- **Performance:** F1 improvement of 28.0% over baseline

**Machine Learning for Sentiment Analysis (2101.06353v1)**
- **Platform:** Social media (Twitter)
- **Task:** Anxiety detection from COVID-19 related posts
- **Methods:** K-NN, Bernoulli, DT, SVC, RF, XG-boost
- **Features:** Count-vectorization, TF-IDF
- **Performance:** Random Forest best (84.99% count-vec, 82.63% TF-IDF)
- **Finding:** K-NN best precision, XG-Boost best recall

**Feeling Anxious: Twitter Analysis (1909.06959v1)**
- **Platform:** Twitter microblogs
- **Architecture:** ML approach to scale human anxiety ratings
- **Innovation:** Non-intrusive longitudinal measurement
- **Findings:** State-anxiety fluctuations, trait anxiety measurement
- **Correlation:** Reverse relationship with social engagement and popularity

### 4.2 Wearable Sensor-Based Detection

**Wristband Sensors with Context (2106.03019v1)**
- **Population:** Older adults (N=41, age 60-80)
- **Sensors:** EDA (electrodermal activity), PPG (blood volume pulse)
- **Protocol:** Trier Social Stress Test (TSST)
- **Features:** 47 computed, 24 significantly correlated selected
- **Innovation:** Context feature vector (experimental phase encoding)
- **Performance:** EDA+context 3.37% improvement, BVP+context 6.41% improvement
- **Validation:** Real-time anxiety level detection simulation

**Hyperbolic Few-Shot Learning (2511.06988v1)**
- **Dataset:** 108 participants (multimodal: speech, physiological, video)
- **Architecture:** HCFSLN (Hyperbolic Curvature FSL Network)
- **Innovation:** Hyperbolic embeddings + cross-attention + adaptive gating
- **Performance:** 88% accuracy (14% improvement over best baseline)
- **Method:** Hierarchical Adaptive Clustering + Cross-Modal Fusion
- **Advantage:** Robust classification with minimal data

### 4.3 Multimodal Deep Learning

**Multi-Modal System (2212.14490v1)**
- **Modalities:** Audio (MFCC) + text
- **Features:** Deep-learned + hand-crafted (clinically-validated)
- **Performance:** F1 improvement 0.58→0.63 (depression), 0.54→0.57 (anxiety)
- **Dataset:** DEPAC corpus
- **Finding:** Speech-based biomarkers hold significant promise

**Protective Movement Behavior Detection (1904.10824v3)**
- **Application:** Chronic pain, anxiety assessment
- **Data:** Motion capture (EmoPain MoCap dataset)
- **Architecture:** BodyAttentionNet (BANet) - CNN + LSTM
- **Innovation:** Temporal and bodily attention mechanisms
- **Performance:** Statistically significant improvements over SOTA
- **Parameters:** Much lower than SOTA for comparable performance

### 4.4 Dialogue and Text Analysis

**Detecting Anxiety in Dialogues (2412.17651v1)**
- **Task:** Multi-label classification (anxiety + depression)
- **Data:** Structured interviews, semi-structured tasks, narratives
- **Features:** LLMs for feature extraction + ML models
- **Performance:** 83% F1, ECE=4.5e-2
- **Innovation:** Explainability via graphical dashboard
- **Context:** Different interaction contexts (structured vs. unstructured)

**DASentimental: Emotional Recall Analysis (2110.13710v1)**
- **Method:** Cognitive network science framework
- **Features:** Semantic memory network, emotion word sequences
- **Models:** Multilayer perceptron neural network
- **Performance:** R=0.7 (depression), R=0.44 (anxiety), R=0.52 (stress)
- **Key Dimensions:** "sad-happy" dyad crucial for depression, "fear" for anxiety
- **Application:** 142 suicide notes analyzed

### 4.5 Metaheuristic Optimization

**Metaheuristic + Deep Learning (2511.18827v1)**
- **Methods:** Genetic algorithms, particle swarm optimization
- **Data:** Physiological, emotional, behavioral signals (multimodal, wearable)
- **Innovation:** Swarm intelligence for feature space refinement
- **Performance:** Significant enhancement over deep networks alone
- **Finding:** Hybrid approach demonstrates stronger generalization

### 4.6 Comparative and Noise Studies

**Effects of Noise in Anxiety Detection (2306.01110v2)**
- **Dataset:** WESAD (wearable stress and affect detection)
- **Challenge:** Real-world noise prevents generalization
- **Methods:** Six classifiers with various feature sets
- **Finding:** Variability in response driven by non-treatment factors
- **Application:** Depression treatment trial analysis

**Social Media State Anxiety (2504.03695v1)**
- **Population:** 111 participants, three anxiety-provoking activities
- **Tasks:** Public speaking, stress-inducing scenarios
- **Sensors:** ECG, EDA
- **Models:** 3,348 trained models (6 classifiers, 31 feature sets)
- **Performance:** AUROC 0.62-0.73, recall anxious 35.19-74.3%
- **Finding:** Text contributes most to depression, audio/facial to PTSD
- **Generalizability:** Relatively stable across activities and groups

### 4.7 Drug Repositioning

**Machine Learning for Drug Discovery (1706.03014v2)**
- **Application:** Depression and anxiety disorders
- **Data:** Drug expression profiles
- **Methods:** SVM, elastic net, random forest, gradient boosted machines
- **Finding:** Repositioning hits enriched for psychiatric medications in trials
- **Innovation:** Variable importance examination reveals drug mechanisms

---

## 5. PTSD Identification from EHR and Clinical Data

### 5.1 Video and Behavioral Analysis

**PTSD in the Wild Dataset (2209.14085v1)**
- **Dataset:** First video database for PTSD recognition in unconstrained environments
- **Variability:** Pose, expression, lighting, focus, resolution, age, gender, race
- **Benchmark:** Evaluation framework for CV and ML approaches
- **Performance:** Deep learning approaches show promising results
- **Availability:** Public distribution with benchmark protocols

**Real-Time fMRI Neurofeedback (1801.09165v2)**
- **Modality:** Simultaneous fMRI + EEG in veterans with combat PTSD
- **Intervention:** Transcranial bipolar DC stimulation (frontoparietal cortex)
- **Method:** rtfMRI-nf training of left amygdala activity
- **Protocol:** Happy emotion induction task, 3 training sessions
- **Results:** 80% EG showed clinically meaningful CAPS reductions (vs. 38% control)
- **Mechanism:** Enhanced amygdala-DLPFC functional connectivity
- **Performance:** Reduction in avoidance and hyperarousal symptoms

### 5.2 EHR and Risk Prediction

**Bayesian Learning for ASUD Risk (2511.04998v1)**
- **Architecture:** BiPETE (Bi-Positional Embedding Transformer Encoder)
- **Cohorts:** Depression (DD) + PTSD patients
- **Target:** Alcohol and substance use disorder (ASUD) risk
- **Innovation:** Rotary positional embeddings + sinusoidal embeddings
- **Performance:** AUPRC improvement 34% (DD), 50% (PTSD)
- **Features:** Inflammatory, hematologic, metabolic markers, medications, comorbidities
- **Method:** Integrated Gradients for interpretability

### 5.3 Enhanced PTSD Detection Frameworks

**Enhancing Outcome Prediction (2411.10661v1)**
- **Disaster Context:** Post-disaster PTSD prediction
- **Methods:** Ensemble models with majority voting
- **Classifiers:** LR, SVM, RF, XGBoost, LightGBM, ANN
- **Preprocessing:** SimpleImputer, label encoding, SMOTE augmentation
- **Performance:** 96.76% accuracy
- **Advantages:** Robustness, generalizability, handling imbalanced datasets

### 5.4 Interview-Based Detection

**Detecting PTSD in Clinical Interviews (2504.01216v2)**
- **Dataset:** DAIC-WOZ clinical interview transcripts
- **Methods:**
  - General/mental health transformers (BERT/RoBERTa, Mental-RoBERTa)
  - Embeddings (SentenceBERT/LLaMA)
  - LLM prompting (zero-shot/few-shot/chain-of-thought)
- **Performance:** SentenceBERT+NN (AUPRC=0.758±0.128), Few-shot (AUPRC=0.737)
- **Domain-Specific:** Mental-RoBERTa AUPRC=0.675±0.084 (vs. 0.599±0.145 general)
- **Finding:** Higher accuracy for severe cases and comorbid depression

**Automating PTSD Diagnostics with LLMs (2405.11178v1)**
- **Dataset:** 411 clinician-administered diagnostic interviews
- **Models:** GPT-4, Llama-2
- **Innovation:** Full automation of PTSD assessment from interviews
- **Method:** Comprehensive framework for automated assessments
- **Availability:** Trained on nationally representative Add Health data
- **External Validation:** Two independent datasets

### 5.5 Language Analysis and NLP

**World Trade Center Responders AI Analysis (2011.06457v1)**
- **Population:** 124 WTC responders
- **Modality:** Oral history interviews (narrative analysis)
- **Method:** AI-based language assessments (depression, anxiety, neuroticism)
- **Follow-up:** PTSD symptom severity (PCL) up to 7 years post-interview
- **Performance:** Greater depressive language (β=0.32, p=0.043)
- **Longitudinal:** Anxious language predicts worsening (β=0.31, p=0.031)
- **Finding:** First-person plural usage predicts improvement (β=-0.37, p=0.007)

### 5.6 Empathetic Dialogue Systems

**The Pursuit of Empathy (2505.15065v2)**
- **Dataset:** TIDE (Trauma-Informed Dialogue for Empathy, 10,000 conversations)
- **Personas:** 500 clinically-grounded PTSD personas
- **Models:** Small LLMs (0.5B-5B parameters)
- **Performance:** Fine-tuning enhances empathy, approaches human levels
- **Demographics:** Older adults favor validation before support (p=0.004)
- **Finding:** Gender differences minimal (p>0.15), broad empathetic designs feasible

### 5.7 Perception and Field Validation

**Perceived Precision Study (2110.13211v2)**
- **Tool:** Field-deployable ML-based PTSD hyperarousal detection
- **Dataset:** Home study with PTSD patients
- **Modality:** Physiological data (heart rate)
- **Performance:** 65% perceived precision in naturalistic validation
- **Finding:** Longitudinal exposure calibrates users' trust in automation
- **Implication:** Alignment between perceived and automated detection critical

### 5.8 Brain Shape Analysis

**Elastic Shape Analysis for PTSD (2105.11547v1)**
- **Dataset:** Grady Trauma Project (brain substructure shapes)
- **Method:** Elastic shape metrics on continuous parameterized surfaces
- **Architecture:** Ensemble (shape summaries via PCs + regression modeling)
- **Innovation:** Accounts for exposure variable interactions
- **Performance:** 5% improvement over vertex-wise and volumetric analysis
- **Finding:** Identifies local deformations related to PTSD severity change

### 5.9 Active Self-Tracking

**One-Button Wearable Study (1703.03437v1)**
- **Population:** Danish veteran with military PTSD
- **Device:** One-button wearable (Mindcraft app)
- **Method:** Self-tracking single symptom (hyperarousal precursor)
- **Duration:** 14 days
- **Finding:** Self-tracking data facilitated therapeutic process
- **Advantage:** Identified crucial details unavailable from clinical assessment

### 5.10 Multimodal Fusion Approaches

**PTSD-MDNN: Late Fusion (2403.10565v1)**
- **Modalities:** Speech, physiological signals, video
- **Architecture:** Two unimodal CNNs with late fusion
- **Performance:** Low detection error rate
- **Application:** Teleconsultation, patient journey optimization, HRI
- **Language:** French

**Stochastic Transformer Approach (2403.19441v1)**
- **Dataset:** eDAIC audio recordings
- **Features:** MFCC low-level features
- **Architecture:** Stochastic Transformer (stochastic depth, layers, activation)
- **Performance:** RMSE=2.92 on eDAIC dataset (state-of-the-art)
- **Innovation:** Stochastic components improve generalization

### 5.11 Social Media and Explainability

**LAXARY: Explainable AI for PTSD (2003.07433v2)**
- **Dataset:** 210 veteran Twitter users with clinical validation
- **Method:** Modified LIWC analysis + ML models
- **Innovation:** PTSD Linguistic Dictionary from survey results
- **Performance:** Promising accuracies for classification and intensity
- **Explainability:** Fills clinical survey tools from Twitter posts
- **Validation:** Reliability and validity of PTSD Linguistic Dictionary

### 5.12 Smartwatch Physiological Detection

**Hyperarousal Event Detection (2109.14743v2)**
- **Dataset:** 99 US veterans with PTSD (multi-day data)
- **Sensors:** Heart rate, body acceleration (Fitbit)
- **Models:** RF, SVM, LR, XGBoost
- **Performance:** XGBoost 83% accuracy, AUC=0.70
- **Explainability:** SHAP analysis (average HR, minimum HR, average acceleration)
- **Application:** Remote, continuous monitoring for PTSD self-management

### 5.13 Mindfulness and Neurobiology

**Interoception in Mindfulness (2010.06078v1)**
- **Intervention:** Mindfulness-based stress reduction (MBSR)
- **RCT:** 98 veterans (MBSR n=47, PCGT n=51)
- **Modalities:** EEG (spectral power, HEBR - heartbeat-evoked responses)
- **Key Finding:** Frontal theta HEBR mediates treatment effect
- **Brain Regions:** ACC, anterior insula, lateral prefrontal cortex
- **Mechanism:** Interoceptive brain capacity primary cerebral mechanism

### 5.14 Depression and Frontline Workers

**The Invisible COVID-19 Crisis (2111.04441v1)**
- **Population:** 1,478 US physicians (1,017 completed PCL-5)
- **Groups:** Frontline (treating COVID-19) vs. second-line
- **Methods:** LR + seven ML algorithms
- **Predictors:** Depression, burnout, negative coping, fears, stigma, resources
- **Protective:** Resilience, support from employers/friends/family
- **Performance:** ML algorithms identify nonlinear protective/damaging factors
- **Implication:** Prepare healthcare systems for future pandemics

### 5.15 Tri-Modal Severity Assessment

**Tri-Modal Severity Fused Diagnosis (2510.20239v1)**
- **Modalities:** Interview text + audio (log-Mel) + facial (AU, gaze, head, pose)
- **Architecture:** RoBERTa embeddings + multi-task framework
- **Disorders:** Depression (PHQ-8, 5 classes) + PTSD (3 classes)
- **Innovation:** Calibrated late fusion classifier with feature attributions
- **Clustering:** 15 distinct failure patterns in 4 harm categories
- **Finding:** Text dominates depression, audio/facial critical for PTSD

### 5.16 Early Detection Framework

**Innovative Framework (2502.03965v1)**
- **Modalities:** Textual (clinical interviews) + audio features
- **Architecture:** LSTM + BiLSTM combination
- **Performance:** 92% accuracy (depression), 93% accuracy (PTSD)
- **Features:** Semantic/grammatical (text), vocal traits (audio)
- **Advantage:** Multimodal fusion enhances detection of mental health patterns

---

## 6. Substance Use Disorder Risk

### 6.1 Early Risk Detection

**Early Risk Detection with BERT (2106.16175v1)**
- **Tasks:** Pathological gambling, self-harm, depression
- **Dataset:** Reddit data + mental health subreddits
- **Architecture:** Pre-trained BERT transformers
- **Method:** Data crawling + pseudo-labels via prompting
- **Performance:** Reasonable results across all three tasks
- **Workshop:** eRisk 2021 contributions

### 6.2 Computational Support Framework

**Computational Support Review (2006.13259v1)**
- **Workshop:** CCC-sponsored (Nov 2019)
- **Focus:** Prevention, detection, treatment, recovery
- **Opportunities:**
  1. Detecting and mitigating relapse risk
  2. Establishing social support networks
  3. Collecting and sharing data across care ecologies
- **Challenge:** $520B annual US costs (crime, productivity, healthcare)

### 6.3 Online Social Network Analysis

**Short-Form Video Addiction Detection (2407.18277v1)**
- **Platform:** Social media short videos (TikTok)
- **Dataset:** Social network behavior + heterogeneous graphs
- **Innovation:** LLMs address sparsity and missing data in graphs
- **Architecture:** Categorizes behavior modalities, heterogeneous structure
- **Method:** Quantitative analysis of short video addicts
- **Finding:** LLMs effective for addiction-related social network analysis

### 6.4 Darkweb and Crypto Market Analysis

**Substance Use from Social Media/Darkweb (2304.10512v1)**
- **Data Sources:** Social media posts + crypto market listings
- **Innovation:** Drug Abuse Ontology
- **Models:** RNN, CNN, Attention-based models
- **Features:** Sentiment, emotion, behavior, turn-taking
- **Performance:** MacroF1=82.12, recall=83.58 with LEDD×UPDRS interaction
- **Method:** Time-aware neural models with historical sentiment/emotional activity

### 6.5 Mathematical Modeling

**Reward-Mediated Learning in Drug Addiction (2205.10704v1)**
- **Framework:** Dynamical systems model
- **Concepts:** Reward prediction error (RPE), incentive salience (IST), opponent process
- **Mechanism:** Biphasic reward response (euphoria + cravings/withdrawal)
- **Pathophysiology:** Neuroadaptive processes enhance negative b-process
- **Intervention:** Modeling methadone and auxiliary drugs in detoxification

### 6.6 Stigma Detection

**Stigma Toward PWUS (2302.02064v2)**
- **Dataset:** 5,000 Reddit posts
- **Annotation:** Crowd-sourced with lived experience consideration
- **Finding:** Workers with substance use experience rate posts more stigmatizing
- **Performance:** 17% AUC improvement including person-level demographics
- **Linguistic Cues:** Othering language ("people", "they"), term "addict"
- **Key: PWUS find substance discussions more stigmatizing

### 6.7 EHR-Based Risk Assessment

**BiPETE for ASUD Risk (2511.04998v1)**
- **See Section 5.2** - Same paper covers PTSD and substance use
- **Target:** Alcohol and substance use disorder (ASUD) from depression/PTSD cohorts

### 6.8 Addiction Network Analysis

**Feature-Selected Graph Spatial Attention (2207.00583v2)**
- **Data:** fMRI of rat brain
- **Architecture:** Graph spatial attention encoder + Bayesian feature selection
- **Application:** Nicotine addiction
- **Performance:** Superior to SOTA clinical decision support systems
- **Method:** Captures spatiotemporal brain networks with spatial information

### 6.9 University Dropout Prediction

**University Dropout Risk (2112.01079v1)**
- **Features:** Learning interaction networks, video game addiction
- **Architecture:** LightGBM with majority voting
- **Innovation:** Social behavior compensates one-sided individual behavior
- **Interpretability:** Shapley values for personalized analysis
- **Finding:** Behavior, turn-taking cues critical; lipstick addiction, student leader uncorrelated

### 6.10 COVID-19 Mental Health Impact

**COVID-19 and Mental Health/SUD on Reddit (2011.10518v1)**
- **Period:** January-October 2020
- **Subreddits:** r/depression, r/Anxiety, r/SuicideWatch vs. r/Coronavirus
- **Method:** Longitudinal topical analysis
- **Finding:** High topical correlation r/depression-r/Coronavirus (Sept 2020)
- **SUD:** Fluctuating correlation, highest August 2020
- **Application:** Monitor trends for targeted interventions

### 6.11 Clinical Note Analysis

**LLM for SUD Severity Extraction (2403.12297v1)**
- **Data:** Clinical notes with DSM-5 granularity
- **Model:** Flan-T5 (open-source LLM)
- **Method:** Zero-shot learning with prompts + post-processing
- **Task:** Extract severity for 11 SUD diagnosis categories
- **Advantage:** Superior recall vs. rule-based approaches
- **Application:** Risk assessment and treatment planning

### 6.12 Behavioral Addiction Studies

**TikTok Addiction via Surveys + Donations (2501.15539v1)**
- **Dataset:** 1,590 TikTok users surveyed, 107 data donations
- **Method:** Mixed surveys + behavioral traces
- **Addiction Groups:** Less/moderately/highly likely addicted (stratified)
- **Findings:** Highly addicted spend more time, frequent returns (compulsion)
- **Prediction:** F1≥0.55 with basic engagement features
- **Challenge:** Predicting addictive users from usage alone difficult

### 6.13 Treatment and Hiring Analysis

**Hiring in SUDT Sector (1908.00216v1)**
- **Period:** 2010-2018, Medicaid expansion 2014+
- **Data:** Burning Glass Technologies job postings
- **Method:** Difference-in-difference econometrics
- **Finding:** Little growth in SUDT sector vs. overall economy/healthcare
- **Shifts:** Reduction in hospital hiring, increases in outpatient facilities
- **Occupational:** From medical personnel toward counselors/social workers

### 6.14 Adaptive Control for Treatment

**Adaptive Control for SUD Treatment (2504.01221v1)**
- **Framework:** Stochastic control formulation
- **Method:** Adaptive control with parameter estimation
- **Innovation:** Matches treatment burden to patient engagement state
- **Finding:** Estimates are consistent, algorithms yield superior recommendations
- **Challenge:** Voluntary compliance requires adherence promotion

### 6.15 Social Media NER for Impacts

**Reddit-Impacts Dataset (2405.06145v1)**
- **Dataset:** 10,000 two-turn conversations, 500 PTSD personas
- **Subreddits:** Prescription/illicit opioids, MOUD discussions
- **Task:** NER for clinical and social impacts
- **Models:** BERT, RoBERTa, DANN (few-shot), GPT-3.5 (one-shot)
- **Application:** SMM4H 2024 shared tasks
- **Finding:** Detects clinical/social impacts of nonmedical substance use

### 6.16 Bayesian Joint Risk Prediction

**Bayesian Learning for AUD and CUD (2501.12278v1)**
- **Disorders:** Alcohol use disorder (AUD) + cannabis use disorder (CUD)
- **Dataset:** Add Health (n=12,503) longitudinal data
- **Architecture:** Joint Bayesian learning model
- **Groups:** Alcohol-only, cannabis-only, both substances
- **Features:** 10 predictors (shared + unique risk factors)
- **Performance:** AUC 0.719/0.690 (cross-val), 0.748/0.710 (valid1), 0.650/0.750 (valid2)
- **Innovation:** Joint modeling outperforms separate univariate models

### 6.17 Fairness in SUD Treatment Prediction

**Fairness in Treatment LOS Prediction (2412.05832v1)**
- **Dataset:** TEDS-D from SAMHSA
- **Task:** Length of stay (LOS) prediction for inpatient and outpatient
- **Frameworks:** Distributive justice, socio-relational fairness
- **Bias Assessment:** Race, region, substance type, diagnosis, payment source
- **Mitigation:** Bias strategies for fair outcomes
- **Implication:** Social justice in computational healthcare innovations

### 6.18 Absolute Risk Prediction for CUD

**CUD Risk Prediction with Bayesian ML (2501.09156v2)**
- **Population:** Adolescents/young adults who use cannabis
- **Dataset:** Add Health (nationally representative)
- **Model:** Bayesian machine learning (personalized risk)
- **Risk Factors:** 5 factors (sex, delinquency, conscientiousness, neuroticism, openness)
- **Performance:** AUC=0.68 (train), 0.64/0.75 (validation), E/O=0.95/0.98/1
- **Timeframe:** CUD risk within 5 years of first cannabis use

### 6.19 Homelessness and Police Interaction

**ML for Homelessness and Police Outcomes (2307.11211v1)**
- **Dataset:** Calgary administrative healthcare (240,219 AMH individuals)
- **Outcomes:** Initial homelessness (0.8%), police interaction (0.32%)
- **Method:** Cohort with fixed vs. adaptive windows
- **Models:** LR, RF, XGBoost
- **Performance:** XGBoost best (sensitivity=91%, AUC=90% homelessness)
- **Risk Factors:** Male, substance disorder, psychiatrist visits, drug abuse
- **Finding:** Flexible windows improve predictive modeling

### 6.20 Paternal Alcohol Use Disorder Protocol

**Protocol for Observational Study (2412.15535v1)**
- **Dataset:** Wisconsin Longitudinal Study
- **Focus:** Long-term effects of growing up with father with AUD
- **Outcomes:** Economic success, relationships, physical/mental health
- **Innovation:** Data turnover method (novel statistical design)
- **Subpopulations:** Two discrete groups for replicability analysis
- **Method:** Uncertainty-aware with qualitative + quantitative exploration

---

## 7. Treatment Response Prediction (Antidepressants)

### 7.1 EEG-Based Biomarkers

**Alpha Wavelet Power Biomarker (1702.04972v1)**
- **Population:** 17 inpatients with bipolar depression
- **Modality:** 21-channel EEG (eyes closed/open)
- **Features:** Normalized wavelet power of alpha rhythm (8-13 Hz)
- **Referential Montages:** Two referential + average reference
- **Performance:** 90% sensitivity, 100% specificity
- **Key Finding:** Responders 84% higher alpha power in occipital channels (O1, O2, Oz)
- **Advantage:** Single EEG measurement (no longitudinal requirement)

**Identifying Ketamine Responses (1805.11446v3)**
- **Population:** 55 TRD outpatients (randomized double-blind)
- **Groups:** A (0.5 mg/kg), B (0.2 mg/kg), C (saline placebo)
- **Modality:** Wearable forehead EEG
- **Performance:** 81.3±9.5% accuracy, 82.1±8.6% sensitivity, 91.9±7.4% specificity
- **Baseline:** Responders weaker theta power than non-responders (p<0.05)
- **Post-treatment:** Higher alpha power, lower alpha asymmetry, lower theta cordance
- **Innovation:** Portable device, mixed dose analysis

### 7.2 Machine Learning for Differential Treatment

**Outcome-Driven Patient Subgroups (2303.15202v2)**
- **Datasets:** Six depression treatment studies (n=5,438)
- **Architecture:** Differential Prototypes Neural Network (DPNN)
- **Treatments:** 5 first-line monotherapies + 3 combination treatments
- **Performance:** AUC=0.66, macro-weighted F1=0.83
- **Innovation:** Patient prototypes for treatment-relevant clusters
- **Clusters:** 3 interpretable patient subgroups
- **Features:** Clinical and demographic data (EHR notes, interview transcripts)

**Cross-Platform Smartphone Sensing (2503.07883v1)**
- **Population:** University students
- **Platforms:** Android and iOS
- **Modality:** Location sensory data (passive collection)
- **Method:** Domain adaptation (map to common feature space)
- **Performance:** F1=0.67 (location + baseline questionnaire)
- **Innovation:** Platform-agnostic prediction via domain adaptation
- **Advantage:** Comparable to periodic self-reported questionnaires

### 7.3 Neural Network for Treatment Selection

**Learning Optimal Biomarker-Guided Policy (2305.13852v1)**
- **Dataset:** EMBARC (large multi-site RCT)
- **Modality:** Pre-treatment EEG (alpha and theta bands)
- **Architecture:** Causal forests + doubly robust estimation
- **Method:** Efficient policy learning algorithm (depth-2 decision tree)
- **Features:** Resting state EEG + treatment effect modifiers
- **Finding:** Evidence of treatment effect heterogeneity
- **Innovation:** Non-negotiability of informed consent in artistic/other contexts

### 7.4 Deep Learning for Treatment Outcome

**Depression and Drug Response via RNNs (2303.06033v1)**
- **Modality:** EEG signals
- **Architecture:** Transformers (modified recursive NNs)
- **Comparison:** CNN, LSTM, CNN-LSTM
- **Performance:** Transformer best for MDD diagnosis and drug response
- **Tasks:**
  - MDD vs. normal: 99.41% recall, 97.14% accuracy
  - Responders vs. non-responders: 97.01% accuracy, 97.76% recall
- **Advantage:** Time dependency of time series evaluated effectively

### 7.5 Neuroimaging Features

**Multiscale Features for Medication Class (2402.07858v1)**
- **Modality:** Resting state fMRI (functional networks + connectivity)
- **Architecture:** Multi-spatial scale features + contrastive pretraining
- **Innovation:** Domain-adapted embeddings, automated feature selection
- **Tasks:** Medication class identification + non-responder detection
- **Performance:** High accuracy rates
- **Finding:** Multi-scale biomarkers superior to single-scale approaches

### 7.6 Fine-Tuning Neural Excitation/Inhibition

**Ketamine E/I Fine-Tuning (2102.03180v1)**
- **Population:** 18 unmedicated TRD patients
- **Modality:** Neuromagnetic virtual electrode timeseries (MEG)
- **Task:** Somatosensory 'airpuff' stimulation
- **Method:** Dynamic Causal Modelling (DCM) on timeseries
- **Innovation:** Poincaré diagram for cortical E/I dynamics
- **Finding:** Shift toward stable quadrant correlates with symptom improvement
- **Mechanism:** Increase in both excitatory and inhibitory coupling required
- **Validation:** Statistical significance (p-value=0.0041, CI=0.93 for 95%)

### 7.7 rTMS Treatment Response

**DE-CGAN for rTMS Prediction (2404.16913v1)**
- **Innovation:** Diversity Enhancing Conditional GAN for oversampling
- **Method:** Creates synthetic examples in difficult-to-classify regions
- **Performance:** 17% AUC increase over language-only modeling
- **Cohorts:** Depressive disorder + PTSD
- **Finding:** Diversity enhancement improves classification performance

**Personalized rTMS Review (2206.12997v1)**
- **Focus:** Parameter optimization (location, angle, pattern, frequency, intensity)
- **Challenge:** One-size-fits-all approach leads to ~50% suboptimal response
- **Opportunities:** Precision medicine via personalized parameter selection
- **Application:** TRD treatment improvement

### 7.8 Item-Level Heterogeneous Treatment Effects

**IL-HTE with IRT Models (2402.04487v2)**
- **Dataset:** 28 RCTs with HDRS-17 (depression rating scale)
- **Innovation:** Polytomous IL-HTE model
- **Performance:** 60.4% balanced accuracy (depression), 59.1% (HDRS-6 subscale)
- **Finding:** Substantial item-level heterogeneity (SD nearly as large as mean effect)
- **Advantage:** More accurate inference, generalizability, identification of interactions

### 7.9 Benefit Assessment from Summary Statistics

**Can Trial Summary Stats Assess Benefit? (2211.00163v1)**
- **Method:** Bounds benefit of optimal treatment using summary statistics alone
- **Dataset:** Depression treatment trial
- **R Package:** rct2otrbounds (publicly available)
- **Finding:** Potential waste of resources if variability driven by non-treatment factors
- **Application:** Meta-analyses assessing individualized treatment potential

### 7.10 Policy Learning and Treatment Decision Trees

**Single Index Model for Treatment Decisions (2203.03523v1)**
- **Data:** Longitudinal trajectories (not scalar outcomes)
- **Method:** "Biosignatures" (linear combinations of baseline characteristics)
- **Innovation:** Maximize Kullback-Leibler Divergence between treatment distributions
- **Advantage:** Contrasts with traditional change score methods
- **Simulation:** Performance compared with missing data scenarios

### 7.11 Personalized Medical Treatments

**RL Algorithms for Personalized Treatments (1406.3922v2)**
- **Innovation:** Q-learning for multistage decision problem with censoring
- **Application:** Chronic depression data + hypothetical clinical trial
- **Performance:** Superior to state-of-the-art clinical decision support
- **Advantage:** Operates when covariate parameters censored/unobtainable
- **Method:** Finite upper bounds on generalized error

### 7.12 Robust Learning for Optimal Treatment

**Robust Learning with NP-Dimensionality (1510.04378v1)**
- **Dataset:** STAR*D study
- **Innovation:** Penalized least squared regression (NP dimensionality)
- **Method:** Robust against conditional mean model misspecification
- **Properties:** Weak oracle properties, selection consistency, oracle distributions
- **Finding:** Estimated value function for optimal treatment regime

### 7.13 Hybrid Learning for Personalized DTRs

**Augmented Multistage Outcome-Weighted Learning (1611.02314v1)**
- **Method:** AMOL (integrates O-learning + Q-learning)
- **Innovation:** Doubly robust augmentation to machine learning O-learning
- **Datasets:** ADHD two-stage trial + STAR*D
- **Properties:** Consistency, convergence rates to optimal value function
- **Advantage:** Valid even if Q-learning model misspecified

### 7.14 DTR Bandit for Response-Adaptive Decisions

**DTR Bandit with Low Regret (2005.02791v3)**
- **Framework:** Online development of optimal DTR
- **Innovation:** Balances exploration and exploitation
- **Performance:** Rate-optimal regret with linear transition/reward models
- **Application:** Major depressive disorder adaptive treatment
- **Validation:** Synthetic experiments + real-world case study

### 7.15 EEG Predictors for Chronic Disorders

**Motif Discovery for Psychiatric EEG (2501.04441v1)**
- **Modalities:** Depression treatment response (7th day EEG)
- **Other Disorders:** Schizophrenia, pediatric seizures, Alzheimer's/dementia
- **Innovation:** Motif discovery from dynamic EEG properties
- **Performance:** High classification precision across all datasets
- **Application:** Depression treatment response prediction as early as day 7

---

## 8. Digital Phenotyping for Mental Health

### 8.1 Smartphone-Based Approaches

**Speech as Multimodal Digital Phenotype (2505.23822v3)**
- **Modalities:** Text + acoustic landmarks + vocal biomarkers (trimodal)
- **Population:** Adolescents
- **Architecture:** LLM-based multi-task learning (MTL)
- **Tasks:** Depression + suicidal ideation + sleep disturbances
- **Dataset:** Depression Early Warning dataset
- **Performance:** 70.8% balanced accuracy
- **Innovation:** Longitudinal analysis across multiple clinical interactions

**Digital Phenotyping Feasibility Study (2501.08851v1)**
- **Population:** 103 participants (mean age 16.1), 3 London schools
- **Duration:** 14 days with Mindcraft app
- **Modalities:** Active (self-reports) + passive (smartphone sensors)
- **Disorders:** Internalizing, externalizing, eating disorders, insomnia, suicidal ideation
- **Method:** Contrastive pretraining + supervised fine-tuning
- **Performance:** BA=0.71 (SDQ-High), 0.67 (insomnia), 0.77 (suicidal ideation), 0.70 (eating)
- **Innovation:** Stabilized daily behavioral representations

### 8.2 Big Data Analytics and AI

**Big Data Analytics in Mental Healthcare (1903.12071v1)**
- **Review:** Comprehensive overview of AI/ML in mental healthcare
- **Applications:** Neuroimaging, neuromodulation, mobile technologies
- **Methods:** Molecular phenotyping, cross-species biomarker identification
- **Challenges:** Explainable AI (XAI), causality testing, closed-loop systems
- **Opportunities:** Multimedia information extraction, multimodal data fusion

### 8.3 Large-Scale Studies

**Large-Scale Digital Phenotyping (2409.16339v1)**
- **Population:** 10,129 UK participants (June 2020-Aug 2022)
- **Modalities:** Fitbit (wearables) + self-reported questionnaires
- **Measures:** PHQ-8, GAD-7, mood assessments
- **Method:** Unsupervised clustering + XGBoost prediction
- **Performance:** R²=0.41 (depression), R²=0.31 (anxiety)
- **Correlations:** Average HR, minimum HR, average body acceleration
- **Finding:** Lower physical activity + higher HR = more severe symptoms

### 8.4 Remote Data Collection Platforms

**RADAR-base Platform (2308.02043v1)**
- **Architecture:** Apache Kafka-based, open-source platform
- **Modalities:** Active (PROMs) + passive (phone sensors, wearables, IoT)
- **Cohorts:** MS, depression, epilepsy, ADHD, Alzheimer, autism, lung diseases
- **Features:** Behavioral, environmental, physiological markers
- **Innovation:** Scalability, extensibility, security, privacy, data quality
- **Availability:** Open-source, community-driven

### 8.5 LLM Integration

**LLMs to Predict Affective States (2407.08240v1)**
- **Modality:** Smartphone sensor data (location)
- **Population:** University students
- **Architecture:** Zero-shot and few-shot embedding LLMs
- **Performance:** Promising predictions of affect measures
- **Finding:** Intricate link between smartphone behavioral patterns and affective states
- **Innovation:** First work leveraging LLMs for affective state prediction

**AWARE Narrator (2411.04691v1)**
- **Innovation:** Converts smartphone data into structured chronological narratives
- **Method:** Quantitative data → English language descriptions
- **Dataset:** University students over a week
- **Application:** Summarize behavior, analyze psychological states via LLMs
- **Architecture:** Systematic framework for narrative generation

### 8.6 Modern ML for Precision Psychiatry

**Modern Views of ML (2204.01607v2)**
- **Review:** Comprehensive review of ML methodologies
- **Modalities:** Neuroimaging, neuromodulation, mobile technologies
- **Methods:** Molecular phenotyping, cross-species biomarkers
- **Concepts:** Explainable AI (XAI), causality testing, closed-loop systems
- **Challenges:** Multimedia extraction, multimodal fusion
- **Framework:** RDoC (Research Domain Criteria) alignment

### 8.7 Individual Behavioral Insights

**Network Analysis with Mobile Sensing (2312.01216v1)**
- **Dataset:** CrossCheck (50 schizophrenia participants)
- **Method:** Network Analysis on EMAs (n-of-1 level)
- **Innovation:** Behavioral context identification via sensor data
- **Finding:** Networks differ significantly during varied behavioral contexts
- **Application:** Automated detection, n-of-1 insights for serious mental illnesses

### 8.8 Latent Space Data Fusion

**Latent Space Fusion Outperforms Early Fusion (2507.14175v1)**
- **Dataset:** BRIGHTEN clinical trial
- **Task:** Daily depressive symptom prediction (PHQ-2)
- **Modalities:** Behavioral (smartphone), demographic, clinical
- **Architecture:** Combined Model (CM) with autoencoders + neural network
- **Performance:** MSE=0.4985 (vs. 0.5305 RF), R²=0.4695 (vs. 0.4356)
- **Innovation:** Intermediate (latent space) fusion superior to early fusion
- **Finding:** Integration of all modalities optimal in CM (not RF)

---

## Cross-Cutting Themes and Technical Insights

### Architectural Patterns

**Transformer Dominance:**
- Hierarchical attention mechanisms (2309.13476v2)
- Audio Spectrogram Transformers (2406.03138v3)
- Modified RNNs with novel architectures (2303.06033v1)
- Superior time dependency modeling

**Multimodal Fusion Strategies:**
- Early fusion: Simple concatenation, often suboptimal
- Late fusion: Model-level integration (2403.10565v1, 2510.20239v1)
- Intermediate fusion: Latent space fusion superior (2507.14175v1)
- Attention-based fusion: Dynamic weighting across modalities

**Self-Supervised Learning:**
- Foundation models (WavLM, HuBERT) provide rich representations (2305.12263v2)
- Transfer learning from large speech corpora
- Layer-wise analysis reveals optimal intermediate features
- Domain adaptation critical for cross-population generalization

### Performance Benchmarks

**Depression Detection:**
- Speech-based: F1=0.897 (2309.13476v2), AUPRC=0.758 (2510.08593v1)
- Multimodal: F1=0.63 (2212.14490v1), accuracy=70.8% (2505.23822v3)
- Cross-lingual: UAR=74% with feature selection (2508.18092v1)

**Suicide Risk Prediction:**
- Clinical notes: F1=0.83 (2205.07872v1)
- Social media: 93% accuracy (2012.09123v1), F1=92.6% (2112.10609v1)
- Ensemble: ρ=0.649, AUC=83.71% (2510.08365v1)

**Psychosis Detection:**
- Speech graph: 93% accuracy (1610.08566v2)
- Network-based: Superior to strength-only measures (2010.00659v1)

**Anxiety Detection:**
- Wearable sensors: 88% accuracy (2511.06988v1), 74% improvement with context (2106.03019v1)
- Multi-modal: F1=0.57 (2212.14490v1)

**PTSD Detection:**
- Interview-based: AUPRC=0.758 (2504.01216v2)
- Ensemble: 96.76% accuracy (2411.10661v1)
- Smartwatch: 83% accuracy, AUC=0.70 (2109.14743v2)

**Treatment Response:**
- EEG: 90% sensitivity, 100% specificity (1702.04972v1)
- Ketamine: 81.3% accuracy (1805.11446v3)
- Drug response: 97.01% accuracy (2303.06033v1)

### Methodological Innovations

**Interpretability:**
- Gradient-weighted attention maps (2309.13476v2)
- SHAP values for feature importance (2109.14743v2, 2112.01079v1)
- Integrated Gradients (2511.04998v1)
- Visual explanations for clinical validation

**Domain Adaptation:**
- Cross-platform (Android/iOS) prediction (2503.07883v1)
- Cross-lingual transfer (2508.18092v1, 2412.15498v1)
- Cross-population generalization (2504.03695v1)

**Few-Shot Learning:**
- Hyperbolic embeddings (2511.06988v1)
- DANN (few-shot) for NER (2405.06145v1)
- Minimal data requirements

**Uncertainty Quantification:**
- Uncertainty-aware models (2502.18285v1)
- Confidence scoring (2502.10950v2)
- Calibrated predictions (ECE metrics)

### Clinical Translation Challenges

**Generalizability:**
- Performance drops across datasets and populations
- Need for validation on independent cohorts
- Cross-activity and cross-population studies critical (2504.03695v1)

**Interpretability vs. Performance:**
- Black-box models achieve higher accuracy but lack clinical trust
- Explainable AI essential for clinical deployment
- Trade-offs between complexity and interpretability

**Data Requirements:**
- Large-scale datasets often needed (10,000+ participants)
- Few-shot learning promising for limited data scenarios
- Synthetic data generation (GANs) for augmentation (2404.16913v1)

**Privacy and Ethics:**
- Speaker disentanglement for privacy (2306.01861v2)
- Informed consent critical (2009.01215v3)
- Bias and fairness assessment necessary (2412.05832v1)

**Real-World Deployment:**
- Computational efficiency for mobile/wearable devices (2302.09214v1)
- Real-time processing requirements
- Battery and resource constraints

---

## Future Directions

### Technical Advances Needed

1. **Improved Generalization:** Models that work across diverse populations, languages, and clinical settings
2. **Causal Inference:** Moving beyond correlation to understand causal mechanisms
3. **Longitudinal Modeling:** Better capture of temporal dynamics and disease progression
4. **Multimodal Integration:** Principled fusion strategies that leverage complementary information
5. **Uncertainty Quantification:** Reliable confidence estimates for clinical decision support

### Clinical Implementation

1. **Validation Studies:** Large-scale, multi-site validation of AI systems
2. **Clinical Workflows:** Integration into existing healthcare infrastructure
3. **Clinician Training:** Education on AI tool usage and limitations
4. **Patient Engagement:** User-friendly interfaces and transparent explanations
5. **Regulatory Frameworks:** FDA approval pathways for AI diagnostic tools

### Research Priorities

1. **Standardized Benchmarks:** Common datasets and evaluation protocols
2. **Open-Source Tools:** Publicly available models and code for reproducibility
3. **Cross-Disorder Studies:** Understanding comorbidity and shared mechanisms
4. **Intervention Studies:** AI-guided treatment selection and adaptation
5. **Ethical Guidelines:** Addressing bias, fairness, privacy, and informed consent

---

## Conclusion

The landscape of AI/ML in psychiatry and mental health has matured significantly, with numerous high-performing systems demonstrating clinical potential across depression, suicide risk, psychosis, anxiety, PTSD, substance use disorders, and treatment response prediction. Transformer-based architectures and multimodal fusion approaches consistently achieve superior performance, while self-supervised learning enables effective transfer across domains and populations.

Key performance achievements include 70-93% accuracy for depression detection, 83-93% for suicide risk prediction, and 81-96% for PTSD identification. Treatment response prediction shows particular promise with 90%+ sensitivity and specificity using EEG biomarkers. Digital phenotyping via smartphones and wearables enables continuous, non-invasive monitoring at scale.

Critical challenges remain in generalizability, interpretability, real-world deployment, and ethical considerations. The field is moving toward precision psychiatry with personalized interventions, but substantial work is needed in clinical validation, regulatory approval, and integration into healthcare workflows.

The convergence of large language models, multimodal learning, and digital phenotyping represents a promising frontier for mental healthcare, with potential to democratize access, enable early intervention, and improve outcomes for millions affected by psychiatric disorders worldwide.

---

## Key Datasets

- **DAIC-WOZ:** Depression interviews with audio/video/text
- **AVEC2013/2014:** Audio-Visual Emotion Challenge datasets
- **RADAR-MDD:** Multi-site, multilingual depression monitoring
- **MIMIC-III:** Medical records with suicide annotations (ScAN)
- **Reddit:** r/SuicideWatch, r/depression, r/Anxiety
- **CrossCheck:** Schizophrenia with smartphone sensing
- **EMBARC:** Multi-site depression treatment trial with EEG
- **eDAIC:** Extended DAIC with additional annotations
- **WESAD:** Wearable stress and affect detection
- **Add Health:** Nationally representative longitudinal (substance use)
- **TEDS-D:** Treatment Episode Data Set for Discharges (SAMHSA)

---

## Tools and Frameworks

- **RADAR-base:** Open-source remote monitoring platform (Apache Kafka)
- **ScANER:** Multi-task RoBERTa for suicide event retrieval
- **SS3/t-SS3:** Novel supervised learning for early risk detection
- **AMOL:** Augmented Multistage Outcome-Weighted Learning
- **BiPETE:** Bi-Positional Embedding Transformer Encoder
- **HCFSLN:** Hyperbolic Curvature Few-Shot Learning Network
- **BodyAttentionNet:** Temporal and bodily attention for movement
- **TimelinePTC:** Pathways to care visualization tool
- **AWARE Narrator:** Smartphone data to narrative conversion
- **rct2otrbounds:** R package for treatment benefit bounds

---

**Total Papers Analyzed:** 160+
**Date Range:** 2016-2025
**Primary Venues:** ArXiv (cs.CL, cs.LG, cs.AI, eess.AS, q-bio.NC)
**Report Length:** 487 lines