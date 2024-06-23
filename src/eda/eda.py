#This contains all the functions that is called on churn_data_processed.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek 
import missingno as msno


def load_data(data_path):
    # Load data from CSV or other format
    return pd.read_csv(data_path)

def info(data):
    info_data = data.info() 
    return info_data

def handle_total_charges(df):
  """
  Checks for strings in the 'TotalCharges' column and converts it to float32,
  replacing empty spaces with 0.

  Args:
      df (pandas.DataFrame): The DataFrame containing the 'TotalCharges' column.

  Returns:
      pandas.DataFrame: The DataFrame with the modified 'TotalCharges' column.
  """

  # Check if 'TotalCharges' is object type (might contain strings)
  if df['TotalCharges'].dtypes == 'object':
    print("There might be strings present in 'TotalCharges'. But it is now handled")
  else:
    print("There might not be strings in 'TotalCharges'.")

  # Replace empty spaces with 0 and convert to float32
  df['TotalCharges'] = df['TotalCharges'].replace(" ", 0).astype('float32')

  return df
 

def summarize_data(data):
    # Print descriptive statistics (mean, median, etc.)
    shape_data = data.shape 
    print(f"The Number of Rows and Columns are - {shape_data}")
    print(data.describe())
    duplicates = data.duplicated().sum()
    print("Checking for number of duplicates in this dataset -", duplicates)


def handle_missing_values(data):
    # Identify and handle missing values (e.g., imputation, removal)
    missing_value_column = data.isnull().sum()
    print(f"The following gives the count of missing value- {missing_value_column} ")
    # Visualize missing values as a matrix
    msno.matrix(data)


def visualize_data(data):
    # Create distribution plots (histograms, boxplots) for numerical features
    # sns.displot(data["column_name"])
    # plt.show()
    
    #Churn(Target Column)
    sns.countplot(data["Churn"], color="orange")
    plt.title("Target variable (Churn)")
    plt.show()


    #Churn Rate 
    yes_churn = data["Churn"].value_counts()[1]
    no_churn = data["Churn"].value_counts()[0]
    total_count = data["Churn"].count()
    churn_pct = (yes_churn / total_count) * 100
    no_churn_pct = (no_churn / total_count) * 100
    percentages = [churn_pct, no_churn_pct]
    # Create categories list
    categories = ["Churned", "Not Churned"] 
    # Create the bar graph
    plt.figure(figsize=(8, 6)) 
    plt.bar(categories, percentages, color=['red', 'green'])  
    plt.xlabel("Churn Status")
    plt.ylabel("Percentage (%)")
    plt.title("Customer Churn Rate")
    plt.show()


    #Number of Customers by their Tenure 
    sns.histplot(x="tenure", data=data, color="green", kde=True)
    plt.xlabel("Tenure in Months")
    plt.ylabel("Number of Customers")
    plt.title("Number of Customers by their Tenure")
    plt.show()

    #Tenure w.r.t Churn 
    sns.displot(data, x="tenure", hue="Churn", kind="kde", fill=True)
    plt.title("Tenure Distribution by Churn (Yes, No)")
    plt.show()


    #Distribution of Monthly Charges by Churn
    sns.displot(data, x="MonthlyCharges", hue="Churn", kind="kde", fill=True)
    plt.title("Distribution of Monthly Chargers by Churn")
    plt.show()


    #Senior Citizen (Feature2)
    value_count_senior = data["SeniorCitizen"].value_counts()[1]
    value_count_notsenior = data["SeniorCitizen"].value_counts()[0]
    plt.pie([value_count_senior,value_count_notsenior], labels=["Senior Citizen", "Not Senior Citizen"], autopct="%d%%")
    plt.title("Distribution of Customers by Senior Citizen Status")
    plt.show()

    #Senior Citzens and Churn Count
    sns.countplot(x="SeniorCitizen", hue='Churn', data=data)
    plt.show()


    #Partner and dependents (Feature3)
    fig, axis = plt.subplots(1, 2, figsize=(12,4))
    axis[0].set_title("Has partner")
    axis[1].set_title("Has dependents")
    axis_y = "percentage of customers"
    # Plot Partner column
    gp_partner = data.groupby('Partner')["Churn"].value_counts()/len(data)
    gp_partner = gp_partner.to_frame().rename({"Churn": axis_y}, axis=1).reset_index()
    percentage_of_customers = gp_partner.groupby('Partner')["Churn"].value_counts()/len(data)
    ax = sns.barplot(x='Partner', y= "count", hue='Churn', data=gp_partner, ax=axis[0])
    # Plot Dependents column
    gp_dep = data.groupby('Dependents')["Churn"].value_counts()/len(data)
    gp_dep = gp_dep.to_frame().rename({"Churn": axis_y}, axis=1).reset_index()
    ax = sns.barplot(x='Dependents', y= "count", hue='Churn', data=gp_dep, ax=axis[1])
    plt.ylabel("Percentage of Customers (%)")
    plt.show()


    #Phone Serivce 
    sns.countplot(data["PhoneService"])
    plt.show()

    #MultipleLines Service 
    sns.countplot(data["MultipleLines"])
    plt.show()

    #MultipleLines and Churn 
    axis_y = "percentage of customers"
    gp_mult = data.groupby('MultipleLines')["Churn"].value_counts()/len(data)
    gp_mult = gp_mult.to_frame().rename({"Churn": axis_y}, axis=1).reset_index()
    sns.barplot(x='MultipleLines', y= "count", hue='Churn', data=gp_mult)
    plt.ylabel("Percentage of Customers (%)")
    plt.show()


    #Internset Service 
    sns.countplot(data["InternetService"])
    plt.show()

    #Internset Service and Churn
    gp_internet = data.groupby('InternetService')["Churn"].value_counts()/len(data)
    gp_internet = gp_internet.to_frame().rename({"Churn": axis_y}, axis=1).reset_index()
    sns.barplot(x='InternetService', y= "count", hue='Churn', data=gp_internet)
    plt.ylabel("Percentage of Customers (%)")
    plt.show()

    #Number of Customer by Contract Type
    # sns.countplot(x="Contract", data=data, color="orange")
    # plt.ylabel("Number of Customers")
    # plt.title("Number of Customers by their Contract Type")
    # plt.show()

    #Churn by Contract Type  
    # Assuming "Contract" and "Churn" are in eda.df_data
    churn_contract = data[["Contract", "Churn"]]  # Select relevant columns
    # Create a crosstab for churn by contract type
    churn_by_contract = pd.crosstab(churn_contract["Contract"], churn_contract["Churn"])
    # Create a stacked bar chart
    churn_by_contract.plot(kind="bar", stacked=True)
    plt.title("Churn Distribution by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability (optional)
    plt.show()


#Onehot Enccoding Funcction
def encode_categorical(data):
    # One-hot encode categorical features
    data = pd.get_dummies(data, columns=["categorical_feature"])
    return data

#Label Encoding Function
def label_encode(data, column):
  """
  Encodes a categorical column in a DataFrame using label encoding.

  Args:
      data (pandas.DataFrame): The DataFrame containing the column to encode.
      column (str): The name of the column to encode.

  Returns:
      pandas.DataFrame: The DataFrame with the encoded column.
  """

  # Create a LabelEncoder object
  encoder = LabelEncoder()

  # Fit the encoder on the unique categories
  encoder.fit(data[column].unique())

  # Encode the column
  data[column] = encoder.transform(data[column])

  return data


#Mappping the encoding for reference 
def print_label_encoding_mapping(encoder):
  """
  Prints the mapping between categories and their encoded labels from a fitted LabelEncoder.

  Args:
      encoder (sklearn.preprocessing.LabelEncoder): The fitted LabelEncoder object.
  """

  # Access the mapping
  mapping = encoder.classes_

  # Print the mapping
  print(f"Category: Encoded Label")
  for category, label in zip(mapping, encoder.transform(mapping)):
    print(f"{category}: {label}")



#Oversampling Function 
def upsample(X,y):
   
   #Using the SMOTETomek library 
   smk = SMOTETomek(random_state=42)
   X,y =smk.fit_resample(X,y)
   return X,y


# def handle_outliers(data):
    # Identify and handle outliers (e.g., winsorization)


# def perform_eda(data_path):
#     data = load_data(data_path)
#     summarize_data(data)
#     visualize_data(data)
#     handle_missing_values(data)
#     handle_outliers(data)
#     data = encode_categorical(data)
#     return data