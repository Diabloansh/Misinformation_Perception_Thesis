# Misinformation-Perception-Thesis

This repository contains all code used in my undergraduate thesis on misinformation, focusing on how **content features** and **temporal exposure** influence belief and spread.

The project combines:

* BERT classification
* Feature engineering 
* Data analysis and statistical modelling

---

## Overview

The work is structured into three main components:

### 1. Hierarchical BERT Classifier

A two-stage model that:

* Classifies headlines into **topics** (health vs technology)
* Then classifies them into **persuasion routes**:

  * Reasoning-based (central)
  * Affect-based (peripheral)
  * Neutral

This was used to test whether persuasion styles can be learned as discrete categories.

---

### 2. Feature-Based Representation

Instead of treating persuasion as labels, this approach models it as **continuous features**, including:

* Structural complexity (clause count, dependency depth)
* Emotional and stylistic signals (sentiment, punctuation)
* Neutral/report-like features (NER, reporting verbs)
* Semantic embeddings (SBERT)

This allows analysis of how persuasion operates in real-world data.

---

### 3. Data Analysis Pipeline

Includes:

* Statistical analysis (ANOVA, t-tests, mixed models)
* Believability experiments (temporal exposure effects)
* Popularity prediction using machine learning (XGBoost)

Key finding:
Content alone is not enough - **exposure and familiarity play a stronger role in belief formation**.

---

## Notes

* Some datasets (e.g. Twitter-based) may require hydration or API access
* LLM-generated data is used for controlled experiments
* Real-world datasets include FakeNewsNet 

---

## Author

Anagha Bhavsar
BSc Computer Science (with Psychology)
