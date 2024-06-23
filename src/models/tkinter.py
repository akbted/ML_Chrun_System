import tkinter as tk
from tkinter import messagebox
from joblib import load

# Load the trained model
loaded_model = load("/Users/akb/Desktop/DS/BIA/Projects/Miniproject_2/Gboost_my_churn_model.pkl")

# Label encoding for categorical features
label_mapping = {
    'DSL': 0,
    'Fiber optic': 1,
    'No Service': 2,
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2,
    'Yes': 1,
    'No': 0,
    'No phone service': 2,
    'No internet service': 2,
    'Electronic check': 0,
    'Mailed check': 1,
    'Bank transfer (automatic)': 2,
    'Credit card (automatic)': 3,
    'Married': 1,
    'Single': 0,
    "Available": 1,
    "Not Available": 0,
}

def predict_churn():
    try:
        tenure = int(tenure_entry.get())
        InternetService = label_mapping[internet_service_var.get()]
        MultipleLines = label_mapping[multiple_lines_var.get()]
        Dependents = label_mapping[dependents_var.get()]
        Partner = label_mapping[partner_var.get()]
        PaperlessBilling = label_mapping[paperless_billing_var.get()]
        PaymentMethod = label_mapping[payment_method_var.get()]
        TechSupport = label_mapping[tech_support_var.get()]
        SeniorCitizen = label_mapping[senior_citizen_var.get()]
        Contract = label_mapping[contract_var.get()]
        MonthlyCharges = float(monthly_charges_entry.get())
        TotalCharges = float(total_charges_entry.get())

        prediction = loaded_model.predict([[tenure, InternetService, Contract, MonthlyCharges, TotalCharges, Dependents, PaperlessBilling, PaymentMethod, SeniorCitizen, Partner, TechSupport, MultipleLines]])

        if prediction[0] == 0:
            messagebox.showinfo("Prediction Result", "This customer is likely to stay.")
        else:
            messagebox.showerror("Prediction Result", "This customer is likely to churn.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create the main window
root = tk.Tk()
root.title("Customer Churn Prediction")

# Create and place the widgets
tk.Label(root, text="Tenure (Months)").grid(row=0, column=0)
tenure_entry = tk.Entry(root)
tenure_entry.grid(row=0, column=1)

internet_service_var = tk.StringVar()
tk.Label(root, text="Internet Service").grid(row=1, column=0)
tk.OptionMenu(root, internet_service_var, 'DSL', 'Fiber optic', 'No Service').grid(row=1, column=1)

multiple_lines_var = tk.StringVar()
tk.Label(root, text="Multiple Lines Status").grid(row=2, column=0)
tk.OptionMenu(root, multiple_lines_var, "Yes", "No", "No phone service").grid(row=2, column=1)

dependents_var = tk.StringVar()
tk.Label(root, text="Dependents").grid(row=3, column=0)
tk.Radiobutton(root, text="Yes", variable=dependents_var, value="Yes").grid(row=3, column=1)
tk.Radiobutton(root, text="No", variable=dependents_var, value="No").grid(row=3, column=2)

paperless_billing_var = tk.StringVar()
tk.Label(root, text="Paperless Billing").grid(row=4, column=0)
tk.Radiobutton(root, text="Yes", variable=paperless_billing_var, value="Yes").grid(row=4, column=1)
tk.Radiobutton(root, text="No", variable=paperless_billing_var, value="No").grid(row=4, column=2)

payment_method_var = tk.StringVar()
tk.Label(root, text="Payment Method").grid(row=5, column=0)
tk.OptionMenu(root, payment_method_var, "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)").grid(row=5, column=1)

contract_var = tk.StringVar()
tk.Label(root, text="Contract").grid(row=6, column=0)
tk.OptionMenu(root, contract_var, 'Month-to-month', 'One year', 'Two year').grid(row=6, column=1)

tech_support_var = tk.StringVar()
tk.Label(root, text="Tech Support Status").grid(row=7, column=0)
tk.OptionMenu(root, tech_support_var, "Available", "Not Available", "No internet service").grid(row=7, column=1)

senior_citizen_var = tk.StringVar()
tk.Label(root, text="Senior Citizen").grid(row=8, column=0)
tk.Radiobutton(root, text="Yes", variable=senior_citizen_var, value="Yes").grid(row=8, column=1)
tk.Radiobutton(root, text="No", variable=senior_citizen_var, value="No").grid(row=8, column=2)

partner_var = tk.StringVar()
tk.Label(root, text="Relationship Status").grid(row=9, column=0)
tk.Radiobutton(root, text="Married", variable=partner_var, value="Married").grid(row=9, column=1)
tk.Radiobutton(root, text="Single", variable=partner_var, value="Single").grid(row=9, column=2)

tk.Label(root, text="Monthly Charges (₹)").grid(row=10, column=0)
monthly_charges_entry = tk.Entry(root)
monthly_charges_entry.grid(row=10, column=1)

tk.Label(root, text="Total Charges (₹)").grid(row=11, column=0)
total_charges_entry = tk.Entry(root)
total_charges_entry.grid(row=11, column=1)

tk.Button(root, text="Predict", command=predict_churn).grid(row=12, column=1)

# Run the application
root.mainloop()
