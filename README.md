# Misinformation-Perception-Thesis

A research project investigating the relationship between misinformation exposure and individual perception patterns through data analysis, feature engineering, and classification modeling.

## Overview

This thesis explores how misinformation affects public perception and belief formation. Through comprehensive data analysis and computational research, this project examines the mechanisms by which false or misleading information spreads and impacts audience credibility perception.

## Project Goals

- **Analyze** the spread and impact of misinformation across different datasets and contexts
- **Understand** cognitive mechanisms behind misinformation acceptance and believability judgments
- **Identify** patterns in how perception is shaped by false information using tweet and news data
- **Develop** predictive models for misinformation classification
- **Extract** feature-based representations to understand persuasion patterns in misleading content
- **Provide** insights for better information literacy and fact-checking strategies

## Repository Structure

```
Misinformation-Perception-Thesis/
├── README.md                                      # This file
├── Classification_Model/                          # Machine learning classification models
│   ├── Final_Classification_Model.ipynb          # Final trained classification model
│   └── Experimental_Models/                       # Experimental model iterations
├── Experimental_Survey_Data_Analysis/             # Survey data analysis and exploration
│   ├── Synthetic_Data_Analysis.ipynb             # Analysis of synthetic experimental data
│   ├── TruthSeeker_Data_Analysis.ipynb           # Analysis of TruthSeeker dataset
│   └── Data/                                      # Survey datasets
├── Feature_Based_Representation/                  # Feature engineering and analysis
│   └── FakeNewsNet_advanced_persuasion_patterns_1.ipynb  # Persuasion pattern analysis
└── TruthSeeker_Believability_Analysis/            # Believability and credibility analysis
    └── truthseeker_tweet_pattern_analysis.ipynb  # Tweet pattern analysis for credibility
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
- Additional dependencies as specified in individual notebooks

## Project Components

### 1. **Experimental Survey Data Analysis**
Analyzes survey data from multiple sources including TruthSeeker and synthetic datasets. Explores how users perceive and judge the believability of information.

- `Synthetic_Data_Analysis.ipynb` - Experimental data exploration and statistical analysis
- `TruthSeeker_Data_Analysis.ipynb` - Analysis of real-world TruthSeeker survey responses

### 2. **Feature-Based Representation**
Develops feature engineering approaches to understand persuasion patterns in misinformation.

- `FakeNewsNet_advanced_persuasion_patterns_1.ipynb` - Advanced feature extraction and persuasion pattern identification

### 3. **Classification Model**
Builds and trains machine learning models to classify misinformation.

- `Final_Classification_Model.ipynb` - Final tuned classification model for misinformation detection

### 4. **Believability Analysis**
Analyzes patterns in how tweets and claims are perceived as believable or credible.

- `truthseeker_tweet_pattern_analysis.ipynb` - Tweet-level credibility pattern analysis

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip or conda package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/AnaghaBh/Misinformation-Perception-Thesis.git
cd Misinformation-Perception-Thesis
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn nltk spacy
```

4. **Launch Jupyter Notebook:**
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

4. **Classification:**
   - Examine `Classification_Model/Final_Classification_Model.ipynb` for the predictive model approach

## Data

The project utilizes multiple datasets:
- **TruthSeeker Survey Data** - Real-world credibility assessments from user surveys
- **Synthetic Experimental Data** - Generated data for controlled analysis
- **FakeNewsNet Dataset** - Misinformation news articles for pattern analysis

## Methodology

- **Research Focus:** Understanding how misinformation affects user perception and believability judgments
- **Data Collection:** Combination of survey-based and crowdsourced data
- **Analysis Approach:** Feature extraction, statistical analysis, and machine learning classification
- **Evaluation:** Cross-validation and performance metrics for model assessment

## Key Findings

[Research findings and insights will be detailed here as analysis progresses]

## Limitations

[Limitations and potential biases in the research will be documented]

## Future Work

- Expand classification models with deep learning approaches
- Incorporate temporal analysis of misinformation spread
- Develop real-time misinformation detection system
- Cross-cultural believability analysis

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to improve this research.

## License

[Add your chosen license - e.g., MIT, Apache 2.0, CC BY 4.0]

## Author

**Anagha Bhat** - [AnaghaBh](https://github.com/AnaghaBh)

## Acknowledgments

[Add acknowledgments for advisors, collaborators, funding sources, and datasets]

## Contact

For questions or inquiries about this research, please open an issue or contact the repository owner.

---

**Last Updated:** April 2026
