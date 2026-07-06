# Misinformation-Perception-Thesis

## Overview

Through comprehensive data analysis and computational research, this project examines the mechanisms by which false or misleading information impacts audience credibility perception.

## Project Goals

- **Understand** cognitive mechanisms behind misinformation acceptance and believability judgments
- **Identify** patterns in how perception is shaped by false information using tweet and news data
- **Develop** predictive models for misinformation classification
- **Extract** feature-based representations to understand persuasion patterns in content
- **Provide** insights for better strategies

## Repository Structure

```
Misinformation-Perception-Thesis/
├── README.md                                      # This file
├── Experimental_Survey_Data_Analysis/             # Survey data analysis and exploration
│   ├── Synthetic_Data_Analysis.ipynb             # Analysis of synthetic experimental data
│   ├── TruthSeeker_Data_Analysis.ipynb           # Analysis of TruthSeeker dataset
│   └── Data/                                      # Survey datasets
├── Feature_Based_Representation/                  # Feature engineering and analysis
│   └── FakeNewsNet_advanced_persuasion_patterns_1.ipynb  # Persuasion pattern analysis
├── TruthSeeker_Believability_Analysis/            # Believability and credibility analysis
│   ├── truthseeker_tweet_pattern_analysis.ipynb  # Tweet pattern analysis for credibility
│   ├── classified_labels.csv                     # Classified credibility labels
│   └── manual_labels.json                        # Manual credibility annotations
└── MisinformationHeadlineClassifier/              # Advanced classification models
    ├── README.md                                 # Classifier-specific documentation
    ├── requirements.txt                          # Python dependencies
    ├── Binary_model/                             # Binary classification model
    ├── hierarchical_model/                       # Hierarchical classification model
    ├── main_route_classifier/                    # Main routing classifier
    └── supplementary_scripts/                    # Supporting utility scripts
```

## Technologies Used

- **Jupyter Notebook** - Interactive analysis, visualization, and experimentation
- **Python** - Core programming language for data processing and modeling

## Key Dependencies

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib & seaborn** - Data visualization
- **scikit-learn** - Machine learning and statistical analysis
- **nltk/spaCy** - Natural language processing
- Additional dependencies as specified in individual notebooks and `MisinformationHeadlineClassifier/requirements.txt`

## Project Components

### 1. **Experimental Survey Data Analysis**
Analyzes survey data from multiple sources including TruthSeeker and synthetic datasets. Explores how users perceive and judge the believability of information.

- `Synthetic_Data_Analysis.ipynb` - Analysis of experiment conducted on Synthetic data
- `TruthSeeker_Data_Analysis.ipynb` - Analysis of real-world TruthSeeker survey responses
- `Data/` - Directory containing survey datasets

### 2. **Feature-Based Representation**
Develops feature engineering approaches to understand persuasion patterns in misinformation.

- `FakeNewsNet_advanced_persuasion_patterns_1.ipynb` - Advanced feature extraction and persuasion pattern identification

### 3. **TruthSeeker Believability Analysis**
Analyzes patterns in how tweets and claims are perceived as believable or credible in real-world online setting.

- `truthseeker_tweet_pattern_analysis.ipynb` - Tweet-level credibility pattern analysis
- `classified_labels.csv` - Machine-classified credibility labels
- `manual_labels.json` - Manual credibility annotations

### 4. **Misinformation Headline Classifier**
Comprehensive classification system with multiple model architectures for misinformation detection.

- `Binary_model/` - Binary classification
- `hierarchical_model/` - Multi-level hierarchical classification
- `main_route_classifier/` - Main routing classifier for initial classification decisions
- `supplementary_scripts/` - Utility scripts for data preprocessing and evaluation
- See `MisinformationHeadlineClassifier/README.md` for detailed documentation

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip or conda package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Diabloansh/Misinformation_Perception_Thesis.git
cd Misinformation_Perception_Thesis
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies for analysis notebooks:**
```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn nltk spacy
```

4. **For the Misinformation Headline Classifier module:**
```bash
cd MisinformationHeadlineClassifier
pip install -r requirements.txt
```

5. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

## Usage Workflow

### Recommended Analysis Order:

1. **Start with Data Analysis:**
   - Begin with `Experimental_Survey_Data_Analysis/Synthetic_Data_Analysis.ipynb` for initial data exploration
   - Continue with `Experimental_Survey_Data_Analysis/TruthSeeker_Data_Analysis.ipynb` to understand real-world perception data

2. **Feature Engineering:**
   - Explore `Feature_Based_Representation/FakeNewsNet_advanced_persuasion_patterns_1.ipynb` to understand feature extraction

3. **Credibility Patterns:**
   - Review `TruthSeeker_Believability_Analysis/truthseeker_tweet_pattern_analysis.ipynb` for believability insights

4. **Classification Models:**
   - Examine `MisinformationHeadlineClassifier/` for advanced classification approaches
   - Refer to `MisinformationHeadlineClassifier/README.md` for model-specific details

## Data

The project utilizes multiple datasets:
- **TruthSeeker Survey Data** - Real-world credibility assessments from user surveys
- **Synthetic Experimental Data** - Generated data for controlled analysis
- **FakeNewsNet Dataset** - Misinformation news articles for pattern analysis
- **Manual Labels** - Human-annotated credibility classifications
- **Classified Labels** - Machine-classified credibility predictions

## Methodology

- **Research Focus:** Understanding how misinformation affects believability judgments
- **Data Collection:** Combination of survey-based and crowdsourced data
- **Analysis Approach:** Feature extraction, statistical analysis, and machine learning classification
- **Model Architecture:** Multiple classification approaches including binary, hierarchical, and routing classifiers
- **Evaluation:** Cross-validation and performance metrics for model assessment

## Author

**Ansh Madan** - [Diabloansh](https://github.com/Diabloansh)

**Last Updated:** May 2026
