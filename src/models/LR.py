#Logistic Regression

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
    model_type="LogisticRegression"
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



# # Save the trained model 
# from train_model import save_model
# save_model(model, "LR_my_churn_model.pkl")  

# # Later, to load the saved model:
# from train_model import load_model
# loaded_model = load_model("LR_my_churn_model.pkl")

