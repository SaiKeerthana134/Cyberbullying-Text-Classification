# Cyberbullying text classification using Deep Learning Models 

## Overview
To address the increasing surge in cyberbullying space, our project leverages advanced deep learning models to accurately detect and categorize online hate speech, aiming to provide a
safer environment for everyone to express their thoughts.
## Dataset
https://huggingface.co/datasets/Mitali05/sentiment-analysis-tweets-llama2-finetune 
## Github Link
https://github.com/SaiKeerthana134/DATA255_TEAM6
## Requirements

python version : `python-3.10.6`

Package requirements :`pip install -r requirements.txt`

## Deployment

### Steps to run the project:
* Download the zip file and unzip it, the folder structure is Code, Readme, project report
* Download data from the dataset link above and save it in the unzipped code folder, which contains all the .py files
* The first step is to run the `pip install -r requirements.txt` in your terminal giving the code folder path (your_path/code).
* To run the modeling and evaluation, run the main.py file (your_path/main.py)

## File Description 
**main.py** : when we run this file the data preprocessing , transformation, modeling and plots will excecute.
**Preprocess.py** : Removes Null values, Tokenization, Build the Vocabulary 
**SequencePadder.py** : Provides functionality to pad sequences to a uniform length for batch processing in models.
**TextCleaner.py** : Contains routines for preprocessing text data, such as removing special characters and lowercasing.
**TextDataset.py**:  Implements a custom PyTorch Dataset for loading and tokenizing text data.
**Word2VecTrainer.py**:  A utility for training Word2Vec embeddings from a given corpus of text.
**TrainAndEvaluate.py**:Houses the training loop and evaluation metrics for model performance assessment.
**RNNClassifier.py** Defines a recurrent neural network model for sequence classification tasks.
**LSTM.py**: Contains the implementation of a Long Short-Term Memory network for text classification.
**GRUClassifier.py**:Introduces a Gated Recurrent Unit classifier tailored for sequence learning.
**GRUClassifierVar.py**:Variant of the standard GRU classifier with modified parameters.
**EmbeddingMatrixCreator.py**:to create an embedding matrix that can be used in neural network models.
**BiLSTMClassifier.py**: a bidirectional LSTM model for improved context understanding in sequences.
**Bert.py**:Wraps the BERT model for fine-tuning on text classification tasks with transfer learning.

## Contributions

**Business Understanding, Data Collection**- Dharani, Gyana, Keerthana

**Data Preprocessing** - Dharani, Gyana, Keerthana

**Data Transformation** - Dharani, Gyana, Keerthana

**Modeling** - Dharani, Gyana, Keerthana

**Evaluation** - Dharani, Gyana, Keerthana

**programming, IDE, Github** - Dharani, Gyana, Keerthana

**PPT and Report** - Dharani, Gyana, Keerthana








