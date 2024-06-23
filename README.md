# Churn Prediction System

## Overview

This repository contains a project aimed at predicting customer churn in a telecom service unit. The project utilizes various machine learning algorithms to create a predictive model and is deployed using Streamlit for real-time use cases.

## Table of Contents
1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Development](#model-development)
6. [Deployment](#deployment)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## Project Description

Customer churn prediction is crucial for retaining valuable customers in the telecom industry. This project involves the following steps:
- Data acquisition and preprocessing
- Exploratory data analysis (EDA)
- Feature engineering
- Model training and evaluation
- Deployment using Streamlit


<img width="1502" alt="workflow_diagram" src="/Users/akb/Desktop/DS/BIA/Projects/Miniproject_2/workflow.png">

<img width="1502" alt="workflow_diagram" src="images/workflow.png">
<img width="1111" alt="deployment_image" src="images/deployment_SS.png">




## Dataset

The dataset used in this project is the Telco Customer Churn dataset from Kaggle. It contains information on customer demographics, account details, and subscription information.

- **Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7043 rows and 21 columns

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/churn-prediction-system.git
cd churn-prediction-system
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

To run the project locally, follow these steps:

1. **Data Preprocessing:**
   - Ensure the dataset is in the `data/` directory.
   - Run the data preprocessing script:
     ```bash
     python scripts/preprocess_data.py
     ```

2. **Model Training:**
   - Train the machine learning model:
     ```bash
     python scripts/train_model.py
     ```

3. **Run Streamlit App:**
   - Start the Streamlit app for real-time churn prediction:
     ```bash
     streamlit run app.py
     ```

## Model Development

### Data Understanding and Cleaning

- The dataset has numerical and categorical features.
- Handled missing values and inconsistencies.
- Performed univariate and bivariate analysis to understand feature importance.

### Feature Engineering

- Used techniques like SMOTE for handling class imbalance.
- Applied label encoding for categorical features.
- Selected features based on correlation analysis.

### Model Selection

Trained and evaluated various models:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- Decision Tree
- XGBoost

### Best Model

The Gradient Boosting model achieved the highest accuracy of 86%.

## Deployment

The model is deployed using Streamlit for real-time predictions. The deployment code is located in the `app.py` file. 

To deploy:
```bash
streamlit run app.py
```

<img width="1111" alt="deployment_image" src="/Users/akb/Desktop/DS/BIA/Projects/Miniproject_2/deployment_SS.png">


## Results

- **Gradient Boosting:** Accuracy of 86%
- **Random Forest:** Accuracy of 81%
- **Logistic Regression:** Accuracy of 80%

Evaluation metrics used include accuracy, ROC-AUC score, and confusion matrix.

## Contributing

Contributions are welcome! Please create a pull request or open an issue to discuss any changes.
