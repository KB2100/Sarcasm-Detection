# Sarcasm-Detection
This repository contains a BERT-based sarcasm detection model for code-mixed (Hindi + English ) social media text. The project includes data preprocessing, model training, evaluation, and predictions. The model is fine-tuned using transformers, PyTorch, and Hugging Face.

# Download Model & Data
You can download the trained model and dataset from the following link:  
📥 [Download Model & Dataset](https://drive.google.com/drive/folders/1jtMeQXnuSpiLQCaf6uzAbeJbtyvXxka4?usp=drive_link)

# Table of Contents
Introduction
Installation
Training
Evaluation
Usage
Results
Future Improvements
Contributors


# Introduction
Sarcasm detection in code-mixed social media text is challenging due to informal writing, regional language mixing, and varying context. This project fine-tunes a multilingual BERT model to detect sarcasm effectively.

# Classes:
Sarcastic (YES)
Non-Sarcastic (NO)
Size:
Training samples: 6,495
Testing samples: 3,345
Preprocessing Steps:
Removing special characters, emojis, and stopwords
Tokenization using BERT multilingual tokenizer
Handling class imbalance using oversampling/undersampling

# Installation
To set up the project, install the required dependencies:

# Clone the repository
git clone https://github.com/KB2100/sarcasm-detection.git
cd sarcasm-detection

# Install dependencies
pip install -r requirements.txt

# Required Packages:
transformers
torch
pandas
sklearn

# Train the model 
python train.py
The model is fine-tuned using BERT-base-multilingual-cased.
Batch size: 16
Learning rate: 2e-5
Epochs: 3

# Evaluate the Model

# Make Predictions

# Save trained model


# Load model for future use
model.load_state_dict(torch.load("sarcasm_model.pth"))
model.to(device)
print("Model loaded successfully!")

# Future Improvements
Improve sarcasm detection in short texts.

Experiment with XLM-RoBERTa for better performance.

Add more regional language support.


