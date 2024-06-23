#SVM

#Importing the dataset 
from train_model import load_data

data = load_data("/Users/akb/Desktop/DS/BIA/Projects/Miniproject_2/data/churn_data_processed_modelling.csv")

#Selecting Target and Features 
from train_model import select_features_target 
X, y = select_features_target(data)


#Model Training
from train_model import train_model_simple
model, X_train, y_train, X_test, y_test = train_model_simple(
    data_path="/Users/akb/Desktop/DS/BIA/Projects/Miniproject_2/data/churn_data_processed_modelling.csv",
    model_type="SVM")

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
save_model(model, "SVM1_my_churn_model.pkl")  

# Later, to load the saved model:
from train_model import load_model
loaded_model = load_model("SVM1_my_churn_model.pkl")

