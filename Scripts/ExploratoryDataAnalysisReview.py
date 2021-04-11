
"""
Author: Alan Danque
Date:   20210323
Purpose:Preliminary Pandas Profiling for Telecom Churn Data

rows attributes
( , )
Rows:
Fields:
"""
import seaborn as sns
import os
import json
import pandas as pd
import glob
import time
start_time = time.time()
#from textblob import TextBlob
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandasql import sqldf
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from numpy.core.defchararray import find
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
from pandas_profiling import ProfileReport
import string
from io import StringIO
from html.parser import HTMLParser
import yaml
from pathlib import Path

start_time = time.time()
mypath = "C:/alan/DSC680/"
yaml_filename = "config_churn.yaml" # sys.argv[2]
base_dir = Path(mypath)
appdir = Path(os.path.dirname(base_dir))
backdir = Path(os.path.dirname(appdir))
config_path = base_dir.joinpath('Config')
ymlfile = config_path.joinpath(yaml_filename)
project_path = base_dir.joinpath('Project2Data')
data_path = project_path.joinpath('Orange')
data_file = data_path.joinpath('churn-bigml-80.csv')  #telecom_churn_data.csv')
results_path = base_dir.joinpath('Project2_Results')
results_path.mkdir(parents=True, exist_ok=True)
analytics_record = results_path.joinpath('generated_analytics_dataset.csv')
churnedrows = results_path.joinpath('churned.csv')

pandasEDA = results_path.joinpath('Propensity_Churn_PandasProfileReport_CHURNED_AND_NOT_output.html')
pandasEDA2 = results_path.joinpath('Propensity_Churn_PandasProfileReport_CHURNED_output.html')
pandasEDA3 = results_path.joinpath('Propensity_Churn_PandasProfileReport_Not_CHURNED_output.html')



# Read in the Telecom Churn Data
data_df = pd.read_csv(data_file)

# Create a unique identifier
data_df["Churned"] = data_df["Churn"].astype(int)
data_df["mobile_identifier"] =  data_df["State"].astype(str)+data_df["Account length"].astype(str)+data_df["Area code"].astype(str)+data_df["Total night charge"].astype(str)

data_df.rename(columns={'International plan': 'International_plan'}, inplace=True)
data_df["IP"]= data_df.International_plan.map(dict(Yes=1, No=0))
data_df.rename(columns={'Voice mail plan': 'Voice_mail_plan'}, inplace=True)
data_df["VMP"]= data_df.Voice_mail_plan.map(dict(Yes=1, No=0))


print(data_df.head())

churnedrows = results_path.joinpath('test.csv')
data_df.to_csv(churnedrows, index=False)

# check how many rows have all missing values and sum em all up
print("All null values:", data_df.isnull().all(axis=1).sum())

data_df_review = data_df.drop(['International_plan','Voice_mail_plan','State'],axis=1)

# PCA
from sklearn.model_selection import train_test_split


# Assign feature variable to X
X = data_df_review.drop(['Churned','mobile_identifier'],axis=1)
# Assign response variable to y
y = data_df_review['Churned']
y.head()


# Feature Standardisation
# 78
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

scaler = preprocessing.StandardScaler().fit(X)
XScale = scaler.transform(X)

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(XScale,y, train_size=0.7,test_size=0.3,random_state=100)
print(X_train.shape)




# Understand the data imbalance
y_train_imb = (y_train != 0).sum()/(y_train == 0).sum()
y_test_imb = (y_test != 0).sum()/(y_test == 0).sum()
print("Imbalance in Train Data:", y_train_imb)
print("Imbalance in Test Data:", y_test_imb)


count_class = pd.value_counts(data_df_review['Churned'], sort=True)
count_class.plot(kind='bar',rot = 0)
plt.title('Churn Distribution')
plt.xlabel('Churn')
img_file = results_path.joinpath('Churn_Distribution.png')
plt.savefig(img_file)
plt.show()




# Handle data imbalance by Performing SMOTE oversampling on the data set
### Other Sampling Techniques just for playing around
#from imblearn.combine import SMOTETomek
#from imblearn.under_sampling import NearMiss
#smk = SMOTETomek(random_state = 42)
#X_trainb,y_trainb = smk.fit_sample(X_train,y_train)
### Other Sampling Techniques just for playing around
#from imblearn.over_sampling import RandomOverSampler
#os = RandomOverSampler(sampling_strategy=1)
#X_trainb,y_trainb = os.fit_sample(X_train,y_train)



# issue: https://stackoverflow.com/questions/66364406/attributeerror-smote-object-has-no-attribute-fit-sample
from imblearn.over_sampling import SMOTE
#Using TensorFlow backend.
smt = SMOTE(random_state = 2)
X_trainb,y_trainb = smt.fit_resample(X_train,y_train)
print(X_trainb.shape)
print(y_trainb.shape)





#Importing the PCA module
from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized', random_state=42)
#Doing the PCA on the train data
pca.fit(X_trainb)
PCA(copy=True, iterated_power='auto', n_components=None, random_state=42,
    svd_solver='randomized', tol=0.0, whiten=False)
#91
print(pca.components_)

colnames = list(X.columns)
pcs_df = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'PC3':pca.components_[2],'Feature':colnames})
pcs_df.head(10)


fig = plt.figure(figsize = (5,5))
plt.scatter(pcs_df.PC1, pcs_df.PC2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
for i, txt in enumerate(pcs_df.Feature):
    plt.annotate(txt, (pcs_df.PC1[i],pcs_df.PC2[i]))
plt.tight_layout()
plt.title('Principal Component Analysis')
img_file = results_path.joinpath('Principal Component Scatter Plot.png')
plt.savefig(img_file)
plt.show()



#Making the screeplot - plotting the cumulative variance against the number of components
# fig = plt.figure(figsize = (12,9))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title('PCA Cumulative Explained Variance')
img_file = results_path.joinpath('PCA Cumulative Explained Variance.png')
plt.savefig(img_file)
plt.show()



# Looks like approx. 50 components are enough to describe 90% of the variance in the dataset
# We'll choose 50 components for our modeling
#Using incremental PCA for efficiency - saves a lot of time on larger datasets
from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components=16)
#Basis transformation - getting the data onto our PCs
df_train_pca = pca_final.fit_transform(X_trainb)
print(df_train_pca.shape)




#Creating correlation matrix for the principal components - we expect little to no correlation
#creating correlation matrix for the principal components
corrmat = np.corrcoef(df_train_pca.transpose())
#plotting the correlation matrix
plt.figure(figsize = (16,16))
sns.heatmap(corrmat,annot = True)
plt.title('PCA Heatmap')
img_file = results_path.joinpath('PCA Heatmap.png')
plt.savefig(img_file)
plt.show()





corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())
print("max corr:",corrmat_nodiag.max(), ", min corr: ", corrmat_nodiag.min(),)




#100
#Applying selected components to the test data - 45 components
telecom_test_pca = pca_final.transform(X_test)
print(telecom_test_pca.shape)

#Multiple Logistic Regression Models with the Principal Components
#Model 1 - Use the No. of Principal Components determined by PCA
#Training the model on the train data
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

learner_pca = LogisticRegression()
model_pca = learner_pca.fit(df_train_pca,y_trainb)
#Making prediction on the test data
pred_probs_test = model_pca.predict_proba(telecom_test_pca)[:,1]
"{:2.2}".format(metrics.roc_auc_score(y_test, pred_probs_test))
# Predict Results from PCA Model
ypred_pca = model_pca.predict(telecom_test_pca)
# Confusion matrix
confusion_PCA = metrics.confusion_matrix(y_test, ypred_pca)
print(confusion_PCA)

labels = ['1', '2', '3']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion_PCA)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()