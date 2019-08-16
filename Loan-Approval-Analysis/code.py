# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 

bank = pd.read_csv(path, delimiter=",")
categorical_var = bank.select_dtypes(include = "object")
print(categorical_var)
numerical_var = bank.select_dtypes(include = "number")
print(numerical_var)



# code starts here






# code ends here


# --------------
# code starts here
banks = bank.drop("Loan_ID", axis=1)
null = banks.isnull().sum()
print(null)
print("____")
bank_mode = banks.mode()
print(bank_mode)
print("_____")
banks = banks.fillna("bank_mode")
print(banks.isnull().sum())

#code ends here


# --------------
# Code starts here







avg_loan_amount = banks.pivot_table(index =['Gender','Married','Self_Employed'], values='LoanAmount' , aggfunc = "mean" )
print(avg_loan_amount)


# code ends here



# --------------
# code starts here





loan_approved_se=banks[(banks["Self_Employed"]=="Yes") & (banks["Loan_Status"]=="Y")].shape[0]
print(loan_approved_se)
loan_approved_nse=banks[(banks["Self_Employed"]=="No") & (banks["Loan_Status"]=="Y")].shape[0]

print(loan_approved_nse)
percentage_se = loan_approved_se*100/614
print(percentage_se)
percentage_nse= loan_approved_nse*100/614
print(percentage_nse)
# code ends here


# --------------
# code starts here

loan_term = banks["Loan_Amount_Term"].apply(lambda x: x/12)
print(loan_term)
print("_______________________")
big_loan_term = len(loan_term[loan_term>=25])
print(big_loan_term)


# code ends here


# --------------
# code starts here






loan_groupby = banks.groupby("Loan_Status")["ApplicantIncome","Credit_History"]
mean_values = loan_groupby.mean()
print(mean_values)




# code ends here


