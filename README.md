# Recruitment Patterns Analysis

## Overview

In the wake of widespread layoffs affecting 72 companies, including major players like Meta, General Motors (GM), T-Mobile, Disney, and Google (Alphabet), the tech industry faces a critical employment crisis. Thirty-three of these companies have reported layoffs impacting over 10% of their workforce. This trend poses a formidable challenge for upcoming Master's in Analytics graduates seeking positions within the tech sector. The problem revolves around the need to understand discernible recruitment patterns across industries, specifically gleaned from SEC reports. 

Our project aims to investigate the factors driving workforce expansions or contractions in different sectors, using 10-K reports as primary data sources. By analyzing these reports, we aim to offer insights for job seekers, aid policy decisions for government agencies, and enhance the competitiveness of employment-focused platforms. 

The project's scope includes data collection, processing, and text mining of a vast amount of SEC report data. Limitations include indirect access to 10-K report text and the challenge of handling a substantial amount of text data (140 GB), necessitating meticulous labeling and analysis strategies. 

The project's success will be measured by the clarity of recruitment patterns identified across industries and the usefulness of insights provided for stakeholders in navigating the employment landscape.

## Getting Started

### 1. Fetch Data from SEC

Download data (2016q1–2023q2) from [Financial statement dataset from the SEC](https://www.sec.gov/dera/data/financial-statement-data-sets), and unzip them to the `data_original` folder. 

Move into `fetch_data` directory.

```shell
cd fetch_data
```

Customize your file path configuration in `utils/path.py`.

```python
data_original_dir = "./data_original"  # Data downloaded from website
html_dir = "./data_html"  # HTML file directory
url_dir = "./html_url"  # 10K url list directory
txt_dir = "./data_txt"  # TXT file directory
error_log_file = "./error_log.txt"  # Error log file
```

Fetch HTML files to the `data_html` folder, and generate the URL list of HTML pages in the `html_url` folder.
```shell
python fetch_html.py
```

Generate txt files with the text content of the HTML pages in the `data_txt` folder.

```shell
python generate_txt.py
```

### 2. Retrieve Recruitment Information and Split Train/Test Sets

Move into `retrieve_y` directory.

```shell
cd ../retrieve_y
```

Customize your file path configuration in `utils/paths.py`.

```python
output_dir = "./output"  # Directory for all output files
data_txt_dir = "./data_txt"  # Directory for raw txt files generated from HTML
data_cleaned_dir = "./data_cleaned"  # Directory for cleaned data
data_split_dir = "./data_split"  # Directory for split data (train_data/test_data)
```

Match recruitment info with regular expression into `number_match.csv`, record matched lines in `employee_lines.txt`. Then organize `number_match.csv` into `employee_num.json` to easily get numbers of employees by cik and year. Filter companies by average annual number of employees and availability of number of employees. Calculate change rates and generate `change_rate.csv` (each row records a company's rate of change for each year from 2017 to 2022).

```shell
python retrieve_y.py
```

Split rows in `change_rate.csv` into `train/test_change_rate.csv` as training and test sets. Split files in directory `data_cleaned` into directory `train_data` and `test_data`. Split the fiscal year data for the companies in a row into each company one row per year in `train/test_change_rate_unfilled.csv`.

```shell
python split_train_test.py
```

### 3. Select Valuable Sentences

The "Select Valuable" module focuses on extracting valuable information from datasets through preprocessing and classification models. This process aims to identify and harness the most impactful insights from your data. 

We manually labled 6000+ lines of text (Binary Classification: 1 = Valuable; 0 = Invaluable), train Random Forest to classify valuable/invaluable. Then filter the valuable lines in every document, and generate a smaller dataset.

This module based on 

```shell
cd ../Select_valuable
```

Customize your file path configuration in `utils/const/paths.py`.

#### Contents

`annotate sample`: This directory contains manually annotated data samples that serve as a benchmark for the model's performance and accuracy.

`extract_valuable.py`: A script dedicated to extracting valuable information from the dataset. It filters and processes the raw data to distill the most significant elements.

`pre_process_data.py`: This script performs various preprocessing operations on the data. Preprocessing is a crucial step that prepares raw data for further analysis and model training.

`run_clean_vectorize.py`: This script takes preprocessed data and converts it into a vectorized format suitable for machine learning models. Vectorization is essential for transforming text data into a numerical format that algorithms can interpret.

`train.py`: The training script where the classification model is trained with the preprocessed and vectorized data. This step involves learning from the data and adjusting the model's parameters accordingly.

`test.py`: The testing script used to evaluate the trained model's performance on unseen data. This step is crucial for assessing the generalizability and effectiveness of the model.

`utils`: A collection of utility functions that are used across the project. These functions provide common capabilities that facilitate data manipulation, feature extraction, and other necessary operations.

#### Statistics of Valuable Sentences Select

Accuracy of Random Forest: 89%

### 4. LDA

Discover 20 abstract topics in our documents collections with unsupervised learning. Each topic consists several words with a weight on each of it.

Move into `LDA` directory.

```shell
cd ../LDA
```

Customize your file path configuration in the main function in `lda.py` and `utils/const/paths.py`. 

Split and clean the lines in a single document. Then train the LDA model with this specific document.

```shell
python lda.py
```

Run `lda.ipynb` to train a LDA model with all documents in train set, and save the model for subsequent generation of scores on each topic for each document.

![image-20231126201958772](https://raw.githubusercontent.com/pyyybf/imgsbed/main/image-20231126201958772.png)

### 5. Vocabulary, TFIDF and Incidence Matrix

Move into `vocabulary_tfidf` directory.

```shell
cd vocabulary_tfidf
```

Customize your file path configuration in `utils/paths.py`. Modify the value of `base_dir` to generate training and test set data separately.

```python
base_dir = "./data_split/train_data"  # Directory for the base data (train/test)
# base_dir = "./data_split/test_data"
```

Extract vocabulary sets by year into `{year}_vocabulary_sets.json`.

```shell
python Get_Vocabulary_Sets.py
```

Combine vocabulary lists from previous years as `Whole_Vocabulary.json`, and record the number of documents in which each word occurs in `word_counts_total.json`.

```shell
python Get_Whole_Vocabulary.py
```

Generate TFIDF matrics in the `TFIDF` directory.

```shell
python TFIDF.py
```

Merge the y values with the TFIDF matrices and save the result in the `y_X` directory. Set y to a change rate value or a binary label (change rate larger than 0 => 1, no larger than 0 => 0) by setting the parameter `classification` of function `merge_X_y`.

```shell
python merge_y.py
```

### 6. Topics

Move into `topics` directory.

```shell
cd ../topics
```

Customize your file path configuration in `utils/paths.py`.

```python
lda_model_path = "./lda/trained_lda_model"  # Path of trained LDA model
id2word_path = "./lda/trained_lda_model.id2word"  # Path of id2word of trained LDA model
train_data_path = "./data/train_data"  # Directory for training files and change rates
test_data_path = "./data/test_data"  # Directory for test files and change rates
train_output_dir = "./topic_data/train_topic"  # Directory for training topic scores + y
test_output_dir = "./topic_data/test_topic"  # Directory for test topic scores + y
```

Use the trained LDA model and incidence matrices of a document to construct the score on each of the 20 topics for a single document, and merge the topic matrix with y. Save the results in `train/test_topics.csv`.

```shell
python calc_topic_score.py
```

Use the trained LDA model to calculate the scores of 20 topics for every Form 10-K (1.e. txt file), and split them by year. Save the results in directory `topic_data`.

```shell
python calc_topic_score_by_year.py
```

### 7. Models

#### 7.1 Baseline

##### 7.1.1 TFIDF Regression

This regression model use TFIDF as X, and use LASSO to predict the number value of change rate.

Move into `baseline` directory.

```shell
cd ../baseline
```

Customize your file path in `baseline_LASSO_train.py`. Then train Logistic Regression with TFIDF matrix and the number value of change rate.

```python
folder_path = 'path/to/your/csv/files'
y_train_file = 'train_y.csv'
model_file = 'multi_target_lasso_regression_model.joblib'
```

```shell
python baseline_LASSO_train.py
```

Test the model.

```shell
python baseline_LASSO_test.py
```

##### 7.1.2 TFIDF Classifier

This classification model use TFIDF as X, and use binary labels based on change rate as y.

Move into `Classifier_TFIDF` directory.

```shell
cd ../Classifier_TFIDF
```

Customize your file path configuration at the top of each python script before running.

```python
models = [
  'LogisticRegression',
  'KNeighborsClassifier',
  'SVC',
  'DecisionTreeclassifier',
  'RandomForestclassifier',
  'GradientBoostingClassifier',
  'NaiveBayes',
  'NeuralNetwork'
]
```

Get the best parameters with grid search. Train and save the classification models.

```shell
python Classifier_TFIDF_train.py
```

Use test set to evaluate these classifiers.

```
python Classifier_TFIDF_test.py
```

##### Statistics of Best Model

| Model Name | Hyper Parameters    | Accuracy | Precision | Recall | F1 Score |
| ---------- | ------------------- | -------- | --------- | ------ | -------- |
| SVC        | `C=1, gamma='auto'` | 0.6353   | 0.6353    | 1.0    | 0.7770   |

#### 7.2 TFIDF Time-Series Classifier

Set a hyper parameter $p$. Let $i$ be the number of subdivisions of $p$ based on the year, and use weighted average of the product of TFIDF for that year and $p^i$ as X for the classification model. Use the binary labels based on change rate as y.

$$
X=\sum_{i=0}^{6}(p^i\times TFIDF_i)\div\sum_{i=0}^{6}p^i
$$

$$
y=\begin{cases}  
1, & change\underline{}rate > 0 \\
0 & otherwise
\end{cases}
$$

Move into `tf_idf_time_series` directory.

```shell
cd ../tf_idf_time_series
```

Customize your file path configuration at the top of each python script before running.

Save list of cik (i.e. company) with change rate available every year in `intersection_of_cik_in_training_set.txt`.

```shell
python get_cik.py
```

Filter the rows in TFIDF csv file by the above cik list, and generate the final dataset for model.

```shell
python matching_cik.py
```

Train Logistic Regression with the new X matrix and binary labels based on change rate.

```shell
python process_file_train_LR_classifier_time_series.py
```

##### Statistics of Model on Test Set

| Model Name         | Hyper Parameters                            | Feature Number | Accuracy | Precision | Recall | F1 Score |
| ------------------ | ------------------------------------------- | -------------- | -------- | --------- | ------ | -------- |
| LogisticRegression | `solver='sag', penalty='l2', max_iter=1000` | 251893         | 0.3853   | 0.7368    | 0.0397 | 0.0754   |

#### 7.3 Topics Classifier

Move into `classifier_topics` directory.

```shell
cd ../classifier_topics
```

Customize your file path configuration in `utils/paths.py`.

```python
train_data_path = "./data/train_topics.csv"  # Training dataset
test_data_path = "./data/test_topics.csv"  # Test dataset
save_model_dir = "./checkpoints"  # Save directory for trained models
output_path = "./output.txt"  # Output file
```

```python
# Model list
classifier_names = [
    "LogisticRegression",
    "KNeighborsClassifier",
    "SVC",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "NaiveBayes",  # GaussianNB
    "NeuralNetworks",  # MLPClassifier
]
```

Train the classification models, get best parameneters with grid search, and then use test set to evaluate the classifier.

```shell
python train_model.py
```

Show the ROC curve for all classifiers except `SVC` in `roc.ipynb` to get better thresholds.

##### Statistics of Best Model

| Model Name | Hyper Parameters      | Accuracy | Precision | Recall | F1 Score |
| ---------- | --------------------- | -------- | --------- | ------ | -------- |
| SVC        | `C=100, gamma='auto'` | 0.5943   | 0.5931    | 0.9924 | 0.7425   |

#### 7.4 LSTM Topics Classifier

The LSTM (Long Short-Term Memory) model incorporated in this framework is adept at capturing long-term dependencies within time-series data. With its unique architecture of gates including forget gate, input gate, and output gate, it can effectively retain and manage information over extended sequences. This ability makes LSTM particularly suitable for complex time-series prediction tasks where understanding both the recent and long-term historical context is crucial.

##### LSTM Mathematical Model

At the core of the LSTM's ability to capture long-term dependencies are the following mathematical operations:

- **Forget Gate** $f_t$: Controls the extent to which a value remains in the cell state. 
  
$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

- **Input Gate** $i_t$ and **Candidate Memory** $\tilde{C}*t$: Decide the new information to be added to the cell state. 

$$
i_t = \sigma(W_i \cdot [h*{t-1}, x_t] + b_i)
$$

$$
\tilde{C}\*t = \tanh(W_C \cdot [h\*{t-1}, x_t]  + b_C)
$$

- **Cell State Update** $C_t$: Updates the old cell state into the new cell state. 
  
$$
C_t = f_t \ast C_{t-1} + i_t \ast \tilde{C}_t
$$

- **Output Gate** $o_t$ and **Hidden State** $h_t$: Control the output of the cell state to the rest of the network. 
  
$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$
  
$$
h_t = o_t \ast \tanh(C_t)
$$

Where $\sigma$ denotes the sigmoid function, $\tanh$ is the hyperbolic tangent function, and $\ast$ represents element-wise multiplication. These functions work together to regulate the information flow through the LSTM network over time.

##### Structure

- `dataloader`: Modules for loading and batching data in a format compatible with PyTorch models.
- `utils`: Utility functions and scripts for preprocessing and preparing datasets for training and evaluation.
- `evaluate`: A script to evaluate the overall importance of input features to the model's predictions.
- `evaluate_bytime`: Tools for assessing the importance of features on a yearly basis, providing insights into the temporal dynamics of feature significance.
- `model`: The LSTM model's architecture is defined here, including the configuration of its layers and computation flow.
- `train`: Execute this script to train the LSTM model with your preprocessed data.
- `test`: After training, this script is used to evaluate the LSTM model's performance on unseen test data.

##### Usage

1. Format your dataset according to the guidelines provided and place it in the designated directory.
2. Use the `train` script to initiate the training process with your prepared data.
3. Evaluate your trained model using the `test` script on a separate testing set.
4. For an analysis of feature importance, the `evaluate` script will rank features based on their influence on the model's output.
5. To understand how feature importance varies over time, run the `evaluate_bytime` script with temporal data.

Ensure that you have the latest version of `PyTorch` installed, along with any other dependencies.

##### Result Statistics

| Model Name | Accuracy | Precision | Recall | F1 Score |
| ---------- | -------- | --------- | ------ | -------- |
| LSTM       | 0.6621   | 0.6631    | 0.9534 | 0.7822   |

# test

## Code Contributors

<a href="https://github.com/pyyybf" style="border-radius: 50%"><img src="https://avatars.githubusercontent.com/u/52249010?v=4" alt="Yue Pan" width=40 style="border-radius: 50%"/></a>
<a href="https://github.com/ChentaoWEI" style="border-radius: 50%"><img src="https://avatars.githubusercontent.com/u/97670328?v=4" alt="Chentao Wei" width=40 style="border-radius: 50%"/></a>
<a href="https://github.com/alyciaqiu" style="border-radius: 50%"><img src="https://avatars.githubusercontent.com/u/129646186?v=4" alt="Yingyue Qiu" width=40 style="border-radius: 50%"/></a>
<a href="https://github.com/xizhu-lin" style="border-radius: 50%"><img src="https://avatars.githubusercontent.com/u/136132782?v=4" alt="Xizhu Lin" width=40 style="border-radius: 50%"/></a>
<a href="https://github.com/tyrzax" style="border-radius: 50%"><img src="https://avatars.githubusercontent.com/u/87009977?v=4" alt="Yunlong Li" width=40 style="border-radius: 50%"/></a>
