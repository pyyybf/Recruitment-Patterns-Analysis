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

You can install these packages using `pip`:

```sh
pip install numpy pandas scikit-learn gensim matplotlib

