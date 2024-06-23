#SVM

#Importing the dataset 
from train_model import load_data

data = load_data("/Users/akb/Desktop/DS/BIA/Projects/Miniproject_2/data/churn_data_processed_modelling.csv")

#Selecting Target and Features 
from train_model import select_features_target 
X, y = select_features_target(data)

# #Train Test Split 
# from train_model import split_data
# X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.33, random_state=42)

# #GridSearch CV
# from train_model import perform_grid_search
# from sklearn.svm import SVC
# model_type = SVC

# #Hyperparameters for SVM
# # hyperparams = {
# #     "C": [0.1, 1, 10],  # Regularization parameter
# #     "kernel": ["linear", "rbf"],  # Kernel function
# #     "gamma": [0.01, 0.1, 1]  # Gamma parameter for RBF kernel (adjust if using linear kernel)
# # }

# hyperparams = {
#     "C": [0.1],  # Regularization parameter (start with narrower range)
#     "kernel": ["linear"],  # Start with linear kernel first
# }


# result = perform_grid_search(model_type, X_train, y_train, hyperparams, cv=5, scoring="accuracy")

# best_hyperp = result.best_params_
model_params = {"C":0.1, "gamma":0.01, "kernel":"linear"}

#Model Training
from train_model import train_model
model, X_train, y_train, X_test, y_test = train_model(
    data_path="/Users/akb/Desktop/DS/BIA/Projects/Miniproject_2/data/churn_data_processed_modelling.csv",
    model_type="SVM", hyperparams=model_params)

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



# Save the trained model 
from train_model import save_model
save_model(model, "SVM2_my_churn_model.pkl")  

# Later, to load the saved model:
from train_model import load_model
loaded_model = load_model("SVM2_my_churn_model.pkl")

