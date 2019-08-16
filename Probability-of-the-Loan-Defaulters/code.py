# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)
p_a = len(df[df["fico"]>700])/len(df)
print(p_a)
p_b = len(df[df["purpose"]=="debt_consolidation"])/len(df)
print(p_b)
df1 = df[df["purpose"]=="debt_consolidation"]
p = len(df[(df["purpose"]=="debt_consolidation") & (df["fico"]>700)])/len(df)
print(p)
p_a_b = (p*p_b)/p_a
print(p_a_b)
p_b_a = (p*p_a)/p_b
print(p_b_a)
result = "p_a_b" =="p_a"
print(result)



# code ends here


# --------------
# code starts here





prob_lp = len(df[df["paid.back.loan"]=="Yes"])/len(df)
prob_cs = len(df[df["credit.policy"]=="Yes"])/len(df)
new_df = df[df["paid.back.loan"]=="Yes"]
p = len(df[(df["paid.back.loan"]=="Yes") & (df["credit.policy"]=="Yes")])/len(df)
prob_pd_cs = p/prob_lp
bayes = (prob_pd_cs*prob_lp)/prob_cs
print(bayes)




# code ends here


# --------------
# code starts here
df1= df[df["paid.back.loan"]=="No"]


df1.purpose.value_counts(normalize=True).plot(kind='bar')

# code ends here


# --------------
# code starts here
inst_median = df["installment"].median()
inst_mean = df.installment.mean()
plt.hist("installment",bins = 10)
plt.hist("log.annual.inc",bins = 10)


# code ends here


