# LSTM for Time-Series Analysis in PyTorch

This repository provides a comprehensive framework for training and evaluating an LSTM model on time-series data using PyTorch. It is structured to facilitate easy data loading, training, testing, and feature importance evaluation across different time periods.

## Structure

- `dataloader`: Modules for loading data compatible with PyTorch models.
- `utils`: Includes dataset processing scripts and utility functions to assist in data manipulation and preparation.
- `evaluate`: Script for assessing the overall importance of input features to the model predictions.
- `evaluate_bytime`: Scripts for evaluating the importance of input features on a yearly basis, allowing for temporal analysis of feature significance.
- `model`: Contains the definition of the LSTM model architecture, detailing the layers and the flow of computation.
- `train`: Training script that takes the processed data and applies the LSTM model to learn from the data.
- `test`: Testing script that evaluates the performance of the trained model on a separate test dataset.

## Usage

To use this framework, you will need to ensure that your data is formatted correctly and placed within the appropriate directory. After preparing your data, you can train the model using the `train` script and subsequently test its performance with the `test` script.

For feature importance evaluation, run the `evaluate` script to obtain an overall importance score for each feature. If you wish to perform a temporal importance analysis, use `evaluate_bytime` to assess feature importance for each year in your dataset.

## Requirements

This codebase is written in Python and requires PyTorch. The necessary packages can be installed via `pip`:

# LDA Topic Extraction

This repository is dedicated to performing Latent Dirichlet Allocation (LDA) for topic extraction from text data. It includes both a script and a Jupyter Notebook for flexibility in how the LDA model is used and explored.

## Contents

- `lda.py`: A Python script that runs the LDA topic model algorithm on the provided dataset.
- `lda.ipynb`: A Jupyter Notebook version of the LDA topic model, which allows for interactive analysis and visualization of the topics extracted from the text data.
- `utils`: A collection of utility functions that support data preprocessing, model parameter tuning, and other tasks related to topic modeling.
- `test.txt`: A sample text file used for testing the LDA model implementation and demonstrating its capabilities.

## Getting Started

To begin using the LDA topic extraction tools in this repository, clone the repository to your local machine and ensure you have the necessary dependencies installed.

### Prerequisites

The following Python packages are required:

- numpy
- pandas
- scikit-learn
- gensim
- matplotlib (for visualization in the Jupyter Notebook)

# Select Valuable

The "Select Valuable" project focuses on extracting valuable information from datasets through preprocessing and classification models. This process aims to identify and harness the most impactful insights from your data.

## Contents

- `annotate sample`: This directory contains manually annotated data samples that serve as a benchmark for the model's performance and accuracy.

- `extract_valuable`: A script dedicated to extracting valuable information from the dataset. It filters and processes the raw data to distill the most significant elements.

- `pre_process_data`: This script performs various preprocessing operations on the data. Preprocessing is a crucial step that prepares raw data for further analysis and model training.

- `run_clean_vectorize`: This script takes preprocessed data and converts it into a vectorized format suitable for machine learning models. Vectorization is essential for transforming text data into a numerical format that algorithms can interpret.

- `train`: The training script where the classification model is trained with the preprocessed and vectorized data. This step involves learning from the data and adjusting the model's parameters accordingly.

- `test`: The testing script used to evaluate the trained model's performance on unseen data. This step is crucial for assessing the generalizability and effectiveness of the model.

- `utils`: A collection of utility functions that are used across the project. These functions provide common capabilities that facilitate data manipulation, feature extraction, and other necessary operations.

## Getting Started

To get started with the project, clone the repository and navigate to the project directory:

```sh
git clone https://github.com/your-username/select-valuable.git
cd select-valuable


You can install these packages using `pip`:

```sh
pip install numpy pandas scikit-learn nltk matplotlib

