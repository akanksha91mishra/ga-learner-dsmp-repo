# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here


# read the dataset
dataset = pd.read_csv(path)
# look at the first five columns
print(dataset.head())

# Check if there's any column which is not useful and remove it like the column id
dataset = dataset.drop(["Id"],1)

# check the statistical description
print(dataset.info())


# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols = dataset.columns

#number of attributes (exclude target)

#x-axis has target attribute to distinguish between classes
x = dataset["Cover_Type"]

#y-axis shows values of an attribute
y = dataset.drop(["Cover_Type"],1)
size = y.columns
#Plot violin for all attributes
for i in size:
    sns.violinplot(x=x,y=y[i])



# --------------
import numpy
upper_threshold = 0.5
lower_threshold = -0.5


# Code Starts Here
subset_train = dataset.iloc[:,0:10]
data_corr = subset_train.corr()
sns.heatmap(data_corr,annot=True)
correlation = list(data_corr.unstack().sort_values(kind="quicksort"))
corr_var_list = []
for i in correlation:
    if abs(i)>0.5 and i!=1:
        corr_var_list.append(i)
print(corr_var_list)
# Code ends here




# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)
X = dataset.drop(["Cover_Type"],1)
Y = dataset["Cover_Type"]
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size=0.2,random_state=0)
# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.

#Standardized
scaler = StandardScaler()

#Apply transform only for continuous data
X_train_temp = scaler.fit_transform(X_train.iloc[:,0:53])
X_test_temp = scaler.fit_transform(X_test.iloc[:,0:53])

#Concatenate scaled continuous data and categorical
X_train1 = numpy.concatenate((X_train_temp,X_train.iloc[:,52:]),axis=1)
X_test1 = numpy.concatenate((X_test_temp,X_test.iloc[:,52:]),axis=1)

scaled_features_train_df = pd.DataFrame(X_train1)
scaled_features_train_df.columns = X_train.columns
scaled_features_train_df.index = X_train.index
scaled_features_test_df = pd.DataFrame(X_test1)
scaled_features_test_df.columns = X_test.columns
scaled_features_test_df.index = X_test.index


# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:
skb = SelectPercentile(score_func=f_classif, percentile=20 )
predictors = skb.fit_transform(X_train1,Y_train)
scores = skb.scores_
Features = scaled_features_train_df.columns

dataframe = pd.DataFrame({"Features":Features,"scores":scores}).sort_values(ascending = False,by = "scores")
top_k_predictors = list(dataframe['Features'][:predictors.shape[1]]) 
print(top_k_predictors)


# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
clf = OneVsRestClassifier(LogisticRegression())
clf1 = OneVsRestClassifier(LogisticRegression())
model_fit_all_features = clf1.fit(X_train,Y_train)
predictions_all_features = model_fit_all_features.predict(X_test)
score_all_features = accuracy_score(Y_test,predictions_all_features)
print(score_all_features)
model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors],Y_train)
predictions_top_features = model_fit_top_features.predict(scaled_features_test_df[top_k_predictors])
score_top_features = accuracy_score(Y_test,predictions_top_features)
print(score_top_features)


