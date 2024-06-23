import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

# Utility functions for data preprocessing (if needed)

def load_data(data_path):
    # Load data from CSV or other format
    return pd.read_csv(data_path)



def preprocess_data(data):
    # Handle missing values, outliers, feature engineering

    # Scale numerical features
    import pandas as pd
    def scale_features(data):
        """
        Scales numerical features in a DataFrame using standard scaling (z-score normalization).

        Args:
            data (pandas.DataFrame): The DataFrame containing features to scale.

        Returns:
            pandas.DataFrame: The DataFrame with scaled numerical features.
        """

        # Get numerical columns (assuming you want to scale only numerical features)
        numerical_cols = data.select_dtypes(include=[np.number])  # Use numpy.number for numerical data types

        # Apply standard scaling to numerical columns
        scaler = StandardScaler()  # Create a StandardScaler object
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

        return data
    
    #Label Encoding
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

#Selecting X and Y
def select_features_target(data):
  """
  Selects features (X) and target variable (y) from a DataFrame.

  Args:
      data (pandas.DataFrame): The DataFrame containing features and target variable.

  Returns:
      tuple: A tuple containing the features (X) and target variable (y).
  """
  # Assuming the target variable column name is 'Churn' (modify if different)
  y = data['Churn']
#   X = data.drop('Churn', axis=1)    
#   X = data["tenure", "SeniorCitizen", "Partner", "Dependents", "MultipleLines", "Contract", "PaperlessBilling", "PaymentMethod","MonthlyCharges" ]  # Drop the target variable from features
  X = data[["tenure","InternetService","Contract","MonthlyCharges", "TotalCharges", "SeniorCitizen","Partner","Dependents","PaymentMethod","PaperlessBilling","MultipleLines","TechSupport"]]
  return X, y


#Splitting the data into Xtrain and Ytrain 
def split_data(X, y, test_size=0.22, random_state=42):
  """
  Splits data into training and testing sets.

  Args:
      X (pandas.DataFrame): The features DataFrame.
      y (pandas.Series): The target variable Series.
      test_size (float, optional): Proportion of data for the test set. Defaults to 0.2.
      random_state (int, optional): Seed for random splitting. Defaults to 42.

  Returns:
      tuple: A tuple containing the training and testing sets for features (X) and target variable (y).
  """
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
  return X_train, X_test, y_train, y_test



from sklearn.model_selection import GridSearchCV
def perform_grid_search(model_type, X_train, y_train, hyperparams, cv=5, scoring="accuracy", n_jobs=1):
  """
  Performs GridSearchCV for a given model type with specified parameters.

  **Important Note:** This function assumes you'll provide the model class constructor directly 
  as the 'model_type' argument. For example, to use SVC:

  grid_search = perform_grid_search(SVC, X_train, y_train, hyperparams, ...)

  Args:
      model_type (class): The model class constructor to use for GridSearchCV (e.g., SVC).
      X_train (pandas.DataFrame): Training features.
      y_train (pandas.Series): Training target variable.
      hyperparams (dict): Dictionary of hyperparameters for grid search.
      cv (int, optional): Number of folds for cross-validation. Defaults to 5.
      scoring (str, optional): Evaluation metric for scoring models. Defaults to "accuracy".
      n_jobs (int, optional): Number of CPU cores to use for parallelization. Defaults to -1 (all cores).

  Returns:
      GridSearchCV: The GridSearchCV object with the best model and search results.
  """

  # Create the model instance
  model = model_type()

  # Perform GridSearchCV
  grid_search = GridSearchCV(estimator=model, param_grid=hyperparams, cv=cv, scoring=scoring, n_jobs=n_jobs)
  grid_search.fit(X_train, y_train)

  # Return the GridSearchCV object
  return grid_search





#Model Training (Without Hyperprams)
def train_model_simple(data_path, model_type):
  """
  Trains a machine learning model based on the provided data and hyperparameters.

  Args:
      data_path (str): Path to the CSV file containing the data.
      model_type (str): The type of model to train ("LogisticRegression", "SVM", "RandomForest", or "GradientBoosting").
      hyperparams (dict, optional): Dictionary of hyperparameters for the model (if applicable). Defaults to None.

  Returns:
      tuple: A tuple containing the trained model, training features (X_train), training target variable (y_train), testing features (X_test), and testing target variable (y_test).
  """

  # Load the data
  data = load_data(data_path)

  # Separate features and target variable
  X, y = select_features_target(data)

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

  # Define and train the model based on model_type
  if model_type == "LogisticRegression":
      model = LogisticRegression()
  elif model_type == "SVM":
      model = SVC()  # Use hyperparams if provided
  elif model_type == "RandomForest":
      model = RandomForestClassifier()  # Use hyperparams if provided
  elif model_type == "GradientBoosting":
      model = GradientBoostingClassifier()  # Use hyperparams if provided
  elif model_type == "Xgboost":
      model = XGBClassifier() # Use hyperparams if provided
  else:
      raise ValueError(f"Invalid model type: {model_type}")  # Raise error for unsupported models

  # Train the model
  model.fit(X_train, y_train)

  # Return the trained model and split data
  return model, X_train, y_train, X_test, y_test


#Model Training with hyperparams
def train_model(data_path, model_type, hyperparams={}):
  """
  Trains a machine learning model based on the provided data and hyperparameters.

  Args:
      data_path (str): Path to the CSV file containing the data.
      model_type (str): The type of model to train ("LogisticRegression", "SVM", "RandomForest", or "GradientBoosting").
      hyperparams (dict, optional): Dictionary of hyperparameters for the model (if applicable). Defaults to None.

  Returns:
      tuple: A tuple containing the trained model, training features (X_train), training target variable (y_train), testing features (X_test), and testing target variable (y_test).
  """

  # Load the data
  data = load_data(data_path)

  # Separate features and target variable
  X, y = select_features_target(data)

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

  # Define and train the model based on model_type
  if model_type == "LogisticRegression":
      model = LogisticRegression()
  elif model_type == "DecisionTree":
      model = DecisionTreeClassifier(**hyperparams)
  elif model_type == "SVM":
      model = SVC(**hyperparams)  # Use hyperparams if provided
  elif model_type == "RandomForest":
      model = RandomForestClassifier(**hyperparams)  # Use hyperparams if provided
  elif model_type == "GradientBoosting":
      model = GradientBoostingClassifier(**hyperparams)  # Use hyperparams if provided
  else:
      raise ValueError(f"Invalid model type: {model_type}")  # Raise error for unsupported models

  # Train the model
  model.fit(X_train, y_train)

  # Return the trained model and split data
  return model, X_train, y_train, X_test, y_test


#Hyperparameter Tuning
def define_model_hyperparams(model_type):
  """
  Defines hyperparameters for a given model type.

  Args:
      model_type (str): The type of model for which to define hyperparameters.

  Returns:
      dict: A dictionary containing hyperparameters for the model.
  """

  if model_type == "LogisticRegression":
      # Example hyperparameters for LogisticRegression
      hyperparams = {
          "C": [0.01, 0.1, 1, 10],  # Regularization parameter
          "solver": ["liblinear", "lbfgs"]  # Solver algorithm
      }
  elif model_type == "SVM":
      # Example hyperparameters for SVM
      hyperparams = {
          "C": [0.1, 1, 10],  # Regularization parameter
          "kernel": ["linear", "rbf"]  # Kernel function
      }
  elif model_type == "RandomForest":
      # Example hyperparameters for RandomForestClassifier
      hyperparams = {
          "n_estimators": [100, 200, 500],  # Number of trees
          "max_depth": [3, 5, 8]  # Maximum depth of trees
      }
  elif model_type == "GradientBoosting":
      # Example hyperparameters for GradientBoostingClassifier
      hyperparams = {
          "learning_rate": [0.1, 0.01],  # Learning rate
          "n_estimators": [100, 200, 500]  # Number of boosting stages
      }
  else:
      raise ValueError(f"Unsupported model type: {model_type}")

  return hyperparams





from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#Metrics/evaluation
def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Trains a machine learning model, evaluates its performance

    Args:
        model (sklearn model): The machine learning model to train.
        X_train (array-like): The training features.
        X_test (array-like): The testing features.
        y_train (array-like): The training labels.
        y_test (array-like): The testing labels.

    Returns:
        tuple: A tuple containing:
            - accuracy (float): The accuracy score of the model.
            - report (str): The classification report for the model.
            - cm (array-like): The confusion matrix for the model.
    
    """

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.heatmap(cm, annot=True, fmt=".2f")
    plt.show()

    return accuracy, report, cm




#For deployment 
def save_model(model, filename):
    # Save the trained model using joblib
    import joblib
    joblib.dump(model, filename)

def load_model(filename):
    # Load the saved model using joblib
    import joblib
    return joblib.load(filename)





