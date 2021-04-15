
"""
Author: Alan Danque
Date:   20210415
Purpose:Principal Component Analysis and Predictions
"""
import imblearn
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
from sklearn import preprocessing
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
#use_label_encoder =False, eval_metric='logloss' - To suppress warnings.

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

states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

state_num = {
        'AK': '1',
        'AL': '2',
        'AR': '3',
        'AS': '4',
        'AZ': '5',
        'CA': '6',
        'CO': '7',
        'CT': '8',
        'DC': '9',
        'DE': '10',
        'FL': '11',
        'GA': '12',
        'GU': '13',
        'HI': '14',
        'IA': '15',
        'ID': '16',
        'IL': '17',
        'IN': '18',
        'KS': '19',
        'KY': '20',
        'LA': '21',
        'MA': '22',
        'MD': '23',
        'ME': '24',
        'MI': '25',
        'MN': '26',
        'MO': '27',
        'MP': '28',
        'MS': '29',
        'MT': '30',
        'NA': '31',
        'NC': '32',
        'ND': '33',
        'NE': '34',
        'NH': '35',
        'NJ': '36',
        'NM': '37',
        'NV': '38',
        'NY': '39',
        'OH': '40',
        'OK': '41',
        'OR': '42',
        'PA': '43',
        'PR': '44',
        'RI': '45',
        'SC': '46',
        'SD': '47',
        'TN': '48',
        'TX': '49',
        'UT': '50',
        'VA': '51',
        'VI': '52',
        'VT': '53',
        'WA': '54',
        'WI': '55',
        'WV': '56',
        'WY': '57'
}

# Read in the Telecom Churn Data
data_df = pd.read_csv(data_file)

print(data_df.isna().sum())
print(data_df.isnull().values.any())
"""
State                     0
Account length            0
Area code                 0
International plan        0
Voice mail plan           0
Number vmail messages     0
Total day minutes         0
Total day calls           0
Total day charge          0
Total eve minutes         0
Total eve calls           0
Total eve charge          0
Total night minutes       0
Total night calls         0
Total night charge        0
Total intl minutes        0
Total intl calls          0
Total intl charge         0
Customer service calls    0
Churn                     0
dtype: int64
False

# Since there are no nan, nulls or empty values
# Saved for if needed for use in another project
# replace whitespaces by null values
data_df.loc[df['FIELD'] == ' ', 'FIELD'] = np.nan
print(f"Cells filled with whitespace in 'FIELD' (After): ", len(df[df['FIELD'] == ' ']))

# replace null values by the median of 'FIELD'
FIELD_median = df['FIELD'].median()
data_df['FIELD'].fillna(FIELD, inplace=True)

# convert 'TotalCharges' from string to float
data_df['FIELD'] = df['FIELD'].astype(float)

"""

# Convert non-numeric to integer columns.
data_df["Churned"] = data_df["Churn"].astype(int)
data_df.rename(columns={'International plan': 'International_plan'}, inplace=True)
data_df["Internationalplan"]= data_df.International_plan.map(dict(Yes=1, No=0))
data_df.rename(columns={'Voice mail plan': 'Voice_mail_plan'}, inplace=True)
data_df["Voicemailplan"]= data_df.Voice_mail_plan.map(dict(Yes=1, No=0))
data_df["StateNum"]= data_df['State'].map(state_num)


# Drop non integer fields.
data_df.drop('International_plan', axis=1, inplace=True)
data_df.drop('Voice_mail_plan', axis=1, inplace=True)
data_df.drop('State', axis=1, inplace=True)
data_df.drop('Churn', axis=1, inplace=True)
data_df.drop('StateNum', axis=1, inplace=True)

# peek at the final analytic record
dataframe_review = results_path.joinpath('dataframe_review_analytic_record.csv')
data_df.to_csv(dataframe_review, index=False)


# Obtain Summary Statistics
df_desc = data_df.describe()
print(type(df_desc))
print(df_desc)
dataframe_desc = results_path.joinpath('dataframe_describe.csv')
df_desc.to_csv(dataframe_desc)


# Correlations

data_df.rename(columns={'Number vmail messages': 'VMail Msgs'}, inplace=True)
data_df.rename(columns={'Customer service calls': 'CustSvcCalls'}, inplace=True)

df_corr = data_df.corr()
print(df_corr)
dataframe_corr = results_path.joinpath('dataframe_corr.csv')
df_corr.to_csv(dataframe_corr)
sns.heatmap(df_corr, vmin=df_corr.values.min(), vmax=1, fmt='.1f', square=True, cmap="Blues", linewidths=0.1, annot=True, annot_kws={"size":8})
sns.set(font_scale=.3)
plt.title('Heatmap of Churn Dataset', fontsize = 12) # title with fontsize 20
#plt.xlabel(' ', fontsize = 15) # x-axis label with fontsize 15
#plt.ylabel(' ', fontsize = 15) # y-axis label with fontsize 15
img_file = results_path.joinpath('Churned_Correlation_Matrix.png')
plt.savefig(img_file)
plt.show()



# unique values for each column containing a categorical feature
def unique_values():
    cat_columns = np.unique(data_df.select_dtypes('object').columns)
    for i in cat_columns:
        print(i, data_df[i].unique())

print("unique_values()")
unique_values()

# list of binary variables, except 'Churn'
bin_var = [col for col in data_df.columns if len(data_df[col].unique()) == 2 and col != 'Churn']
print("bin_var")
print(bin_var)

# list of categorical variables
cat_var = [col for col in data_df.select_dtypes(['object']).columns.tolist() if col not in bin_var]
print("cat_var")
print(cat_var)


# apply Label Encoding for binaries
le = LabelEncoder()
for col in bin_var:
    data_df[col] = le.fit_transform(data_df[col])

# apply get_dummies for categorical
data_df = pd.get_dummies(data_df, columns=cat_var)

data_df.head()

# feature matrix
X = data_df.drop('Churned', axis=1)
# target vector
y = data_df['Churned']

X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

rus = RandomUnderSampler()
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)


#Importing the PCA module
#from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized', random_state=42)
#Doing the PCA on the train data
pca.fit(X_train_rus)
PCA(copy=True, iterated_power='auto', n_components=None, random_state=42,
    svd_solver='randomized', tol=0.0, whiten=False)
print("pca.components_")
print(pca.components_)

colnames = list(X.columns)
pcs_df = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'PC3':pca.components_[2],'Feature':colnames})
pcs_df.head(10)

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize = (5,5))
plt.scatter(pcs_df.PC1, pcs_df.PC2)
plt.xlabel('Principal Component 1', fontsize=10)
plt.ylabel('Principal Component 2', fontsize=10)
for i, txt in enumerate(pcs_df.Feature):
    plt.annotate(txt, (pcs_df.PC1[i],pcs_df.PC2[i]))
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.title('Principal Component Analysis', fontsize = 12)
img_file = results_path.joinpath('Principal_Component_Scatter_Plot.png')
plt.savefig(img_file)
plt.show()


#Making the screeplot - plotting the cumulative variance against the number of components
# fig = plt.figure(figsize = (12,9))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components', fontsize=10)
plt.ylabel('cumulative explained variance', fontsize=10)
plt.title('PCA Cumulative Explained Variance', fontsize = 12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
img_file = results_path.joinpath('PCA_Cumulative_Explained_Variance.png')
plt.savefig(img_file)
plt.show()


# Looks like approx. 50 components are enough to describe 90% of the variance in the dataset
# We'll choose 50 components for our modeling
#Using incremental PCA for efficiency - saves a lot of time on larger datasets
pca_final = IncrementalPCA(n_components=16)
df_train_pca = pca_final.fit_transform(X_train_rus)
print("df_train_pca.shape")
print(df_train_pca.shape)

#Creating correlation matrix for the principal components - I expect little to no correlation
corrmat = np.corrcoef(df_train_pca.transpose())
plt.figure(figsize = (16,16))
sns.set(font_scale=.8)
sns.heatmap(corrmat, vmin=df_corr.values.min(), vmax=1, fmt='.1f', square=True, cmap="Blues", linewidths=0.1, annot=True, annot_kws={"size":18})
plt.title('PCA Heatmap', fontsize = 16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
img_file = results_path.joinpath('PCA_Heatmap.png')
plt.savefig(img_file)
plt.show()


corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())
print("max corr:",corrmat_nodiag.max(), ", min corr: ", corrmat_nodiag.min(),)



# Ensemble model testing
svc = SVC()
lr = LogisticRegression(max_iter=200)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

model = []
cross_val = []
recall = []
for i in (svc, lr, xgb):
    model.append(i.__class__.__name__)
    cross_val.append(cross_validate(i, X_train_rus, y_train_rus, scoring='recall'))

for d in range(len(cross_val)):
    recall.append(cross_val[d]['test_score'].mean())

model_recall = pd.DataFrame
recall_df = pd.DataFrame(data=recall, index=model, columns=['Recall'])
dataframe_recall = results_path.joinpath('dataframe_model_recall.csv')
recall_df.to_csv(dataframe_recall, index=False)
print(recall_df)


# Support Vector Classifier
# parameters to be searched
param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

# find the best parameters
grid_search = GridSearchCV(svc, param_grid, scoring='recall')
grid_result = grid_search.fit(X_train_rus, y_train_rus)

print(f'Support Vector Classifier Best result: {grid_result.best_score_} for {grid_result.best_params_}')


# Logistic Regression
# parameters to be searched
param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
              'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
# find the best parameters
grid_search = GridSearchCV(lr, param_grid, scoring='recall')
grid_result = grid_search.fit(X_train_rus, y_train_rus)
print(f'Logistic Regression Best result: {grid_result.best_score_} for {grid_result.best_params_}')


# XGBoost
# parameter to be searched
param_grid = {'n_estimators': range(0, 1000, 25)}

# find the best parameter
grid_search = GridSearchCV(xgb, param_grid, scoring='recall')
grid_result = grid_search.fit(X_train_rus, y_train_rus)

print(f'XBGoost Best result: {grid_result.best_score_} for {grid_result.best_params_}')


# XGBoost
xgb = XGBClassifier(n_estimators=25, use_label_encoder=False, eval_metric='logloss')

# parameters to be searched
param_grid = {'max_depth': range(1, 8, 1),
              'min_child_weight': np.arange(0.0001, 0.5, 0.001)}

# find the best parameters
grid_search = GridSearchCV(xgb, param_grid, scoring='recall', n_jobs=-1)
grid_result = grid_search.fit(X_train_rus, y_train_rus)

print(f'XGBoost Best result: {grid_result.best_score_} for {grid_result.best_params_}')

# XGBoost
xgb = XGBClassifier(n_estimators=25, max_depth=1, min_child_weight=0.0001, use_label_encoder=False, eval_metric='logloss')

# parameter to be searched
param_grid = {'gama': np.arange(0.0, 20.0, 0.05)}

# find the best parameters
grid_search = GridSearchCV(xgb, param_grid, scoring='recall', n_jobs=-1)
grid_result = grid_search.fit(X_train_rus, y_train_rus)

print(f'XGBoost Best result: {grid_result.best_score_} for {grid_search.best_params_}')

# XGBoost
xgb = XGBClassifier(n_estimators=25, max_depth=1, min_child_weight=0.0001, gama=0.0, use_label_encoder=False, eval_metric='logloss')

# parameter to be searched
param_grid = {'learning_rate': [0.0001, 0.01, 0.1, 1]}

# find the best parameter
grid_search = GridSearchCV(xgb, param_grid, scoring='recall')
grid_result = grid_search.fit(X_train_rus, y_train_rus)

print(f'XGBoost Best result: {grid_search.best_score_} for {grid_search.best_params_}')

# final SVC model
svc = SVC(kernel='poly', C=1.0)
svc.fit(X_train_rus, y_train_rus)

# prediction
X_test_svc = scaler.transform(X_test)
y_pred_svc = svc.predict(X_test_svc)

# classification report
print("SVC Classification Report")
print(classification_report(y_test, y_pred_svc))

sns.set(font_scale=.8)
sns.set_context("poster", font_scale=.8)

# confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_svc, normalize='true'), annot=True, ax=ax)
ax.set_title('Support Vector Classifier Confusion Matrix')
ax.set_ylabel('Real Value')
ax.set_xlabel('Predicted Value')
img_file = results_path.joinpath('SVC_Confusion_Matrix.png')
plt.savefig(img_file)
plt.show()

print("SVC Confusion Matrix Complete Duration: --- %s seconds ---" % (time.time() - start_time))


# final Logistic Regression model
lr = LogisticRegression(solver='liblinear', max_iter=150, C=0.0001) #solver='lbfgs'
lr.fit(X_train_rus, y_train_rus)

# prediction
X_test_lr = scaler.transform(X_test)
y_pred_lr = lr.predict(X_test_lr)

# classification report
print("LR classification report")
print(classification_report(y_test, y_pred_lr))

# confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_lr, normalize='true'), annot=True, ax=ax)
ax.set_title('Logistic Regression Confusion Matrix')
ax.set_ylabel('Real Value')
ax.set_xlabel('Predicted Value')
img_file = results_path.joinpath('LogisticRegression_Confusion_Matrix.png')
plt.savefig(img_file)
plt.show()


print("Logistic Regression Confusion Matrix Complete Duration: --- %s seconds ---" % (time.time() - start_time))


# final XGBoost model
xgb = XGBClassifier(learning_rate=0.0001, n_estimators=25, max_depth=1, min_child_weight=0.0001, gamma=0, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train_rus, y_train_rus)

# XGBoost prediction
X_test_xgb = scaler.transform(X_test)
y_pred_xgb = xgb.predict(X_test_xgb)

# classification report
print("Final XGBoost Classification Report")
print(classification_report(y_test, y_pred_xgb))

# confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_xgb, normalize='true'), annot=True, ax=ax)
ax.set_title('XGBoost Confusion Matrix')
ax.set_ylabel('Real Value')
ax.set_xlabel('Predicted Value')
img_file = results_path.joinpath('Final_XGBoost_Confusion_Matrix.png')
plt.savefig(img_file)
plt.show()


print("Final XGBoost Complete Duration: --- %s seconds ---" % (time.time() - start_time))

"""
Final XGBoost Classification Report
              precision    recall  f1-score   support

           0       0.90      0.94      0.92       588
           1       0.37      0.25      0.30        79

    accuracy                           0.86       667
   macro avg       0.64      0.60      0.61       667
weighted avg       0.84      0.86      0.85       667

Final XGBoost Complete Duration: --- 464.31161236763 seconds ---
"""