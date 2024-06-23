#RandomForest 

#Importing the dataset 
from train_model import load_data

data = load_data("/Users/akb/Desktop/DS/BIA/Projects/Miniproject_2/data/churn_data_processed_modelling.csv")

#Selecting Target and Features 
from train_model import select_features_target 
X, y = select_features_target(data)

#Train Test Split 
from train_model import split_data
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.33, random_state=42)


#GridSearch CV
from train_model import perform_grid_search
from sklearn.ensemble import RandomForestClassifier
model_type = RandomForestClassifier

# hyperparams = {
#   "n_estimators": [100, 200, 500],
#   "max_depth": [3, 5, 8],
#   "min_samples_split": [2, 5, 10],
#   "min_samples_leaf": [1, 2, 4],
#   "max_features": ["sqrt", "log2", None]
# }

hyperparams = {
  "n_estimators": [100, 200],
  "max_depth": [3, 5],
  "min_samples_split": [10],
  "min_samples_leaf": [4],
  "max_features": ["sqrt"]
}

grid_search_result = perform_grid_search(model_type, X_train, y_train, hyperparams, cv=5, scoring="accuracy")

# Access the best model with the best hyperparameters
best_model = grid_search_result.best_estimator_

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search_result.best_params_)


best_hyperparms = {
    'max_depth': 5, 
    'max_features': 'sqrt', 
    'min_samples_leaf': 4, 
    'min_samples_split': 10, 
    'n_estimators': 200}

#Splitting Test andd Train + Model Training
from train_model import train_model
model, X_train, y_train, X_test, y_test = train_model(
    data_path="/Users/akb/Desktop/DS/BIA/Projects/Miniproject_2/data/churn_data_processed_modelling.csv",
    model_type="RandomForest", hyperparams = best_hyperparms
)

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

#Genrating Confusion Matrix 
import seaborn as sns
import matplotlib.pyplot as plt
confusion_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion_matrix, annot=True, fmt=".2f")
plt.show()

# Generate classification report
print(classification_report(y_test, y_pred))

#ROC Curve 
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Assuming binary classification, get probability of positive class
fpr, tpr, threshold = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess (AUC = 0.5)')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (TPR)')
plt.xlabel('False Positive Rate (FPR)')
plt.legend(loc='lower right')
plt.show()

# Save the trained model 
from train_model import save_model
save_model(model, "RF2_my_churn_model.pkl")  

# Later, to load the saved model:
from train_model import load_model
loaded_model = load_model("RF2_my_churn_model.pkl")

