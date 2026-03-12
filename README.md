# Sentiment-Analysis-using-NLTK-and-Transformer 🤗

This project implements sentiment analysis on the Amazon Fine Food Reviews dataset (`568,454 reviews`) using two complementary approaches: `lexicon-based VADER (NLTK)` and `transformer-based RoBERTa (Hugging Face)`. The analysis covers `data ingestion`, `exploratory data analysis (EDA)`, `model evaluation`, and performance comparison on a balanced test set of 4,956 samples.

## Project Workflow
1. **Data Ingestion**: Loads Amazon Fine Food Reviews dataset and maps star ratings to sentiments (`1-2: Negative, 3: Neutral, 4-5: Positive`).
2. **EDA**: Analyzes score distribution, text length trends, and temporal patterns.
3. **VADER Analysis**: Rule-based sentiment scoring using Valence Aware Dictionary.
4. **RoBERTa Analysis**: Pretrained transformer model for contextual sentiment classification.
5. **Model Evaluation**: Classification reports, confusion matrices, and F1-score comparisons.

## Model Information

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Lexicon-based approach** using NLTK's pretrained sentiment lexicon
- Analyzes text through rule-based scoring of individual words and phrases
- Considers valence modifiers (intensifiers like "very", negations like "not")
- Computes compound score (-1 to +1) with thresholds: `≤-0.05=Negative, -0.05 to 0.05=Neutral, ≥0.05=Positive`
- Fast, interpretable, but struggles with sarcasm and context-dependent sentiment

### RoBERTa (Robustly optimized BERT approach)
- **Transformer-based pretrained model** from Hugging Face (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- Fine-tuned on 124M+ tweets for 124M parameters capturing contextual embeddings
- Processes entire sentences/paragraphs simultaneously using attention mechanisms
- Learns complex patterns including sarcasm, emojis, and domain-specific language
- Zero-shot inference capability with pipeline API for production deployment

## Model Performance Summary

| Model     | Accuracy | Macro F1 | Negative F1 | Neutral F1 | Positive F1 |
|-----------|----------|----------|-------------|------------|-------------|
| **VADER** | 80.0%    | 0.49     | 0.48        | 0.08       | 0.89        |
| **RoBERTa**| 85.0%  | 0.63     | 0.74        | 0.22       | 0.93        |

## Performance Analysis

### Classification Report Insights
- **RoBERTa** demonstrates superior performance (`85% accuracy vs 80%`) with balanced macro F1-score improvement (`0.63 vs 0.49`).
- Both models excel on **Positive** sentiment (F1 > 0.89) due to dataset imbalance (`77% positive samples`).
- **Neutral** detection remains challenging for both approaches (`VADER F1: 0.08, RoBERTa F1: 0.22`).
- **Negative** sentiment shows substantial improvement with RoBERTa (`F1: 0.74 vs 0.48`).

### Confusion Matrix Observations
- **High Positive Precision**: Both models correctly identify most positive reviews.
- **VADER Limitations**: Poor Neutral recall (5%) with most neutrals misclassified as Positive.
- **RoBERTa Improvements**: Better Negative recall (75% vs 41%) but continues Neutral-Positive confusion.
- **Dataset Imbalance Impact**: 3,819 Positive vs 746 Negative vs 391 Neutral samples influences model bias.

## Key Technical Features
- Preprocesses large-scale review dataset (568K+ samples)
- Implements balanced test set evaluation
- Generates comprehensive visualizations (score distribution, confusion matrices)
- Compares lexicon-based vs transformer-based approaches

## Usage Instructions
1. Run `sentiment-analysis-using-nltk-and-transformer.ipynb` in Python/Kaggle environment
2. Required libraries: NLTK, Transformers, scikit-learn, pandas, matplotlib, seaborn
3. Outputs include classification reports, confusion matrices, and performance visualizations
