# Sentiment Analyzer using BERT and Flask

This project presents a sentiment analysis system that utilizes a fine-tuned BERT model to classify tweets as either positive or negative. The system is deployed as a web application using the Flask framework.

## Features
- Leverages transfer learning through the use of a pre-trained BERT model (`bert-base-uncased`)
- Fine-tuned on the Sentiment140 dataset containing Twitter data
- Includes a simple Flask-based web interface for user interaction
- Provides performance evaluation through metrics and visualizations such as a confusion matrix and word cloud

## Getting Started

### 1. Clone the repository

### 2. Install required dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
Execute the training script (`train.py`) to fine-tune the model and save the output:

```bash
python train.py
```
Note: It is recommended to use Google Colab for training, due to the availability of GPU resources.

### 4. Launch the Web Application
```bash
python app.py
```

Then open your browser and navigate to: `http://127.0.0.1:5000`

## Project Structure

- `app.py`: Flask web application for sentiment analysis
- `train.py`: Script for preprocessing, model training, and saving
- `saved_model/`: Directory containing the fine-tuned model and tokenizer
- `requirements.txt`: List of required Python packages
- `README.md`: Project documentation and setup instructions

## Model Description

- Base Model: BERT (`bert-base-uncased`)
- Task: Binary sentiment classification
- Dataset: Sentiment140 (Twitter-based dataset)
- Output Labels: 0 = Negative, 1 = Positive

## Techniques Applied

- Transfer learning from pre-trained transformers
- Fine-tuning on domain-specific data (tweets)
- Deployment using Flask web framework
