#Loading the dataset 
from eda import load_data  
df_data = load_data("/Users/akb/Desktop/DS/BIA/Projects/Miniproject_2/data/Telco_Customer_Churn.csv")
print(df_data.head())

#Viewing the datatypes of each column 
from eda import info
info(df_data)

#Descriptive statistics
from eda import summarize_data
summarize_data(df_data)

# Handling the empty space in Total Charges
from eda import handle_total_charges
df_data = handle_total_charges(df_data)

#Checking if there are any missing values  
from eda import handle_missing_values
handle_missing_values(df_data)

#Checking if the target column has a balanced or no 
print(df_data.groupby('Churn').size()) #(imbalance in dataset)

#Visualisation 
from eda import visualize_data
visualize_data(df_data)


#Categorical Encoding and Saving the dataframe as a new csv "churn_data_processed"
from eda import label_encode
#Gender
label_encode(df_data, "gender")
#Partner 
label_encode(df_data, "Partner")
#Dependents
label_encode(df_data, "Dependents")
#PhoneService
label_encode(df_data, "PhoneService")
#Churn
label_encode(df_data, "Churn")

#Multiplelines
label_encode(df_data, "MultipleLines")
#InternetService
label_encode(df_data, "InternetService")
#OnlineSecurity
label_encode(df_data, "OnlineSecurity")
#DeviceProtection
label_encode(df_data, "DeviceProtection")
#TechSupport
label_encode(df_data, "TechSupport")
#StreamingMovies
label_encode(df_data, "StreamingMovies")
#Contract
label_encode(df_data, "Contract")
#PaymentMethod
label_encode(df_data, "PaymentMethod")
#PaperlessBilling
label_encode(df_data, "PaperlessBilling")
#OnlineBackup
label_encode(df_data, "OnlineBackup")
#StreamingTV
label_encode(df_data, "StreamingTV")

df_data = df_data.drop('customerID', axis=1) 
print(df_data.info())

#Correlation Matrix 
correlation = df_data.corr()
import seaborn as sns
import matplotlib.pyplot as plt 
plt.figure(figsize=(18,10))
sns.heatmap(correlation, annot=True)
plt.show()



# Save the processed DataFrame
df_data.to_csv("/Users/akb/Desktop/DS/BIA/Projects/Miniproject_2/data/churn_data_processed.csv", index=False)

#Oversampling to address the churn data imabalance 
from eda import upsample
X = df_data.drop("Churn", axis=1)
y = df_data["Churn"]

X_new, y_new = upsample(X,y)
print(y_new.value_counts()) #To verify that the data imbalance is addressed

#Combining the X_new, y_new into a new DataFrame
import pandas as pd
df_data = pd.concat([X_new, y_new], axis=1)



# Save the processed DataFrame
df_data.to_csv("/Users/akb/Desktop/DS/BIA/Projects/Miniproject_2/data/churn_data_processed_modelling.csv", index=False)

print("Data encoding and saving complete!")


