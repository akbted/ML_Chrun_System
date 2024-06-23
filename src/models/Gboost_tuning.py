#Gradientboost Classifier 

#Importing the dataset 
from train_model import load_data

data = load_data("/Users/akb/Desktop/DS/BIA/Projects/Miniproject_2/data/churn_data_processed_modelling.csv")


#Selecting Target and Features 
from train_model import select_features_target 
X, y = select_features_target(data)

#Splitting Test andd Train + Model Training
from train_model import train_model
model, X_train, y_train, X_test, y_test = train_model(
    data_path="/Users/akb/Desktop/DS/BIA/Projects/Miniproject_2/data/churn_data_processed_modelling.csv",
    model_type="GradientBoosting"
)


# #Hyperparameter - Grid Search 
# from train_model import perform_grid_search
# from sklearn.ensemble import GradientBoostingClassifier
# hyperparams = {
#     "n_estimators": [50, 100, 200],
#     "learning_rate": [0.1, 0.3, 0.5],
#     "max_depth": [3, 5],
#     "min_samples_split": [2, 5],
#     "min_samples_leaf": [1, 3],
# } 

# model_type = GradientBoostingClassifier  # Pass the model class constructor
# grid_search = perform_grid_search(model_type, X_train, y_train, hyperparams, cv=5, scoring="accuracy", n_jobs=1)

# print(grid_search.best_params_)

#Training the model with best parameters again  
best_params  = {'learning_rate': 0.3,
 'max_depth': 5,
 'min_samples_leaf': 3,
 'min_samples_split': 5,
 'n_estimators': 200}

from train_model import train_model
model, X_train, y_train, X_test, y_test = train_model(
    data_path="/Users/akb/Desktop/DS/BIA/Projects/Miniproject_2/data/churn_data_processed_modelling.csv",
    model_type="GradientBoosting", hyperparams= best_params
)


# Metrics
from train_model import train_evaluate_model

accuracy, report, cm = train_evaluate_model(model, X_train, X_test, y_train, y_test)

print(f"The Accuracy of the Model = {accuracy}")
print(report)

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


#AUC Value
print("AUC - Model Performance:", roc_auc)





# Save the trained model 
from train_model import save_model
save_model(model, "Gboost_my_churn_model.pkl")  

# Later, to load the saved model:
from train_model import load_model
loaded_model = load_model("Gboost_my_churn_model.pkl")

