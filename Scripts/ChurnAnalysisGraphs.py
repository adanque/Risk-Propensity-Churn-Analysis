
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
import matplotlib.cm as cm
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
data_df_review = data_df.drop(['International_plan','Voice_mail_plan','State'],axis=1)
count_class = pd.value_counts(data_df_review['Churned'], sort=True)
count_class.plot(kind='bar',rot = 0)
plt.title('Churn Distribution')
plt.xlabel('Churn')
img_file = results_path.joinpath('Churn_Distribution.png')
plt.savefig(img_file)
plt.show()


# Read in the Telecom Churn Data
data_df = pd.read_csv(data_file)
# I prefer to use SQL to prep my dataframe.
data_df["Churned"] = data_df["Churn"].astype(int)
print(data_df.head())
pysqldf = lambda t: sqldf(t, globals())
t = """
SELECT * FROM data_df a           
        ;"""
"""
388 rows of 2666 are churn rows

"""


cdf = pysqldf(t)

# Churn Rate per dataset
print(cdf['Churned'].value_counts())
print('\nTotal Churn Rate: {:.2%}'.format(cdf[cdf['Churned'] == 1].shape[0] / cdf.shape[0]))
print("We have an imbalanced dataset!")



df_grp = cdf.groupby(['International plan'])['Churned'].sum().reset_index()
ax = sns.barplot(x='International plan', y='Churned', data=df_grp)
plt.xlabel('International plan', fontsize=12)
plt.ylabel('Churned', fontsize=12)
plt.title('Churned Occurences by International plan')
plt.xticks(rotation = 90)
#plt.legend()
img_file = results_path.joinpath('Churned_BY_International_plan.png')
plt.savefig(img_file)
plt.show()

cdf = pysqldf(t)
df_grp = cdf.groupby(['Voice mail plan'])['Churned'].sum().reset_index()
ax = sns.barplot(x='Voice mail plan', y='Churned', data=df_grp)
plt.xlabel('Voice mail plan', fontsize=12)
plt.ylabel('Churned', fontsize=12)
plt.title('Churned Occurences by Voice mail plan')
plt.xticks(rotation = 90)
plt.legend()
img_file = results_path.joinpath('Churned_BY_Voice_mail_plan.png')
plt.savefig(img_file)
plt.show()
# More


# Create boxplots for outlier detection.
print("min")
print(cdf['Total day minutes'].min())
print("max")
print(cdf['Total day minutes'].max())
print("min")
print(cdf['Total day charge'].min())
print("max")
print(cdf['Total day charge'].max())

# draw a boxplot to check for outliers
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 7))
fig.suptitle("Boxplot of 'Total day minutes' and 'Total day charge'")
boxprops = whiskerprops = capprops = medianprops = dict(linewidth=1)
sns.boxplot(y=cdf['Total day minutes'], x= cdf['Churned'], data=cdf, orient='v', color='#488ab5', ax=ax[0],
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops)
ax[0].set_facecolor('#f5f5f5')
ax[0].set_yticks([0, 200, 400])

sns.boxplot(y=cdf['Total day charge'], x= cdf['Churned'], data=cdf, orient='v', color='#488ab5', ax=ax[1],
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops)
ax[1].set_facecolor('#f5f5f5')
ax[1].set_yticks([0, 40, 80])
plt.tight_layout(pad=4.0);
img_file = results_path.joinpath('Total_day_calls_and_charge_outlier_detection.png')
plt.savefig(img_file)
plt.show()


print("min")
print(cdf['Total eve minutes'].min())
print("max")
print(cdf['Total eve minutes'].max())
print("min")
print(cdf['Total eve charge'].min())
print("max")
print(cdf['Total eve charge'].max())

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 7))
fig.suptitle("Boxplot of 'Total eve minutes' and 'Total eve charge'")
boxprops = whiskerprops = capprops = medianprops = dict(linewidth=1)
sns.boxplot(y=cdf['Total eve minutes'], x= cdf['Churned'], data=cdf, orient='v', color='#488ab5', ax=ax[0],
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops)
ax[0].set_facecolor('#f5f5f5')
ax[0].set_yticks([0, 185, 370])

sns.boxplot(y=cdf['Total eve charge'], x= cdf['Churned'], data=cdf, orient='v', color='#488ab5', ax=ax[1],
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops)
ax[1].set_facecolor('#f5f5f5')
ax[1].set_yticks([0, 20, 40])
plt.tight_layout(pad=4.0);
img_file = results_path.joinpath('Total_eve_minutes_and_Total_eve_charge_outlier_detection.png')
plt.savefig(img_file)
plt.show()





print("min")
print(cdf['Total night minutes'].min())
print("max")
print(cdf['Total night minutes'].max())
print("min")
print(cdf['Total night charge'].min())
print("max")
print(cdf['Total night charge'].max())

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 7))
fig.suptitle("Boxplot of 'Total night minutes' and 'Total night charge'")
boxprops = whiskerprops = capprops = medianprops = dict(linewidth=1)
sns.boxplot(y=cdf['Total night minutes'], x= cdf['Churned'], data=cdf, orient='v', color='#488ab5', ax=ax[0],
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops)
ax[0].set_facecolor('#f5f5f5')
ax[0].set_yticks([0, 200, 400])

sns.boxplot(y=cdf['Total night charge'], x= cdf['Churned'], data=cdf, orient='v', color='#488ab5', ax=ax[1],
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops)
ax[1].set_facecolor('#f5f5f5')
ax[1].set_yticks([0, 10, 20])
plt.tight_layout(pad=4.0);
img_file = results_path.joinpath('Total_night_minutes_and_Total_night_charge_outlier_detection.png')
plt.savefig(img_file)
plt.show()



print("min")
print(cdf['Total intl minutes'].min())
print("max")
print(cdf['Total intl minutes'].max())
print("min")
print(cdf['Total intl charge'].min())
print("max")
print(cdf['Total intl charge'].max())

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 7))
fig.suptitle("Boxplot of 'Total intl minutes' and 'Total intl charge'")
boxprops = whiskerprops = capprops = medianprops = dict(linewidth=1)
sns.boxplot(y=cdf['Total intl minutes'], x= cdf['Churned'], data=cdf, orient='v', color='#488ab5', ax=ax[0],
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops)
            #,showfliers=False)
ax[0].set_facecolor('#f5f5f5')
ax[0].set_yticks([0, 15, 30])

sns.boxplot(y=cdf['Total intl charge'], x= cdf['Churned'], data=cdf, orient='v', color='#488ab5', ax=ax[1],
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops)
            #,showfliers=False)
ax[1].set_facecolor('#f5f5f5')
ax[1].set_yticks([0, 4, 8])
plt.tight_layout(pad=4.0);
img_file = results_path.joinpath('Total_intl_minutes_and_Total_intl_charge_outlier_detection.png')
plt.savefig(img_file)
plt.show()



#
df_grp = cdf.groupby(['State'])['Churned'].sum().reset_index()
fin_df_grp = df_grp.sort_values(by=['Churned'], ascending=False)
ax = sns.barplot(x='State', y='Churned', data=fin_df_grp)
plt.xlabel('State', fontsize=12)
plt.ylabel('Churned', fontsize=12)
plt.title('Churn By State')
plt.xticks(rotation = 90)
#plt.legend()
img_file = results_path.joinpath('Churned_BY_State.png')
plt.savefig(img_file)
plt.show()


cdf = pysqldf(t)
df_grp = cdf.groupby(['Customer service calls'])['Churned'].sum().reset_index()
ax = sns.barplot(x='Customer service calls', y='Churned', data=df_grp)
plt.xlabel('Customer service calls', fontsize=12)
plt.ylabel('Churned', fontsize=12)
plt.title('Churned Occurences by Customer service calls')
plt.xticks(rotation = 90)
#plt.legend()
img_file = results_path.joinpath('Churned_BY_Customer_service_calls.png')
plt.savefig(img_file)
plt.show()


cdf = pysqldf(t)
df_grp = cdf.groupby(['Area code'])['Churned'].sum().reset_index()
ax = sns.barplot(x='Area code', y='Churned', data=df_grp)
plt.xlabel('Area code', fontsize=12)
plt.ylabel('Churned', fontsize=12)
plt.title('Churned Occurences by Area code')
plt.xticks(rotation = 90)
#plt.legend()
img_file = results_path.joinpath('Churned_BY_Area_Code.png')
plt.savefig(img_file)
plt.show()

churned_df = lambda t: sqldf(t, globals())
t = """
SELECT [Total day minutes],	[Total day calls],	[Total day charge],	[Total eve minutes],	[Total eve calls],	[Total eve charge],	[Total night minutes],	[Total night calls],	[Total night charge],	[Total intl minutes],	[Total intl calls],	[Total intl charge]
 FROM cdf a where a.Churned = 1           
        ;"""
c_df = churned_df(t)

notchurned_df = lambda t: sqldf(t, globals())
t = """
SELECT [Total day minutes],	[Total day calls],	[Total day charge],	[Total eve minutes],	[Total eve calls],	[Total eve charge],	[Total night minutes],	[Total night calls],	[Total night charge],	[Total intl minutes],	[Total intl calls],	[Total intl charge] 
FROM cdf a where a.Churned = 0           
        ;"""
nc_df = notchurned_df(t)

print(nc_df.head())


for (columnName, columnData) in c_df.iteritems():
    metricslist = []
    x =[columnName + " Churned", c_df[columnName].min()]
    metricslist.append(x)

    x =[columnName+" Churned", c_df[columnName].mean()]
    metricslist.append(x)

    x =[columnName + " Churned", c_df[columnName].max()]
    metricslist.append(x)

    x =[columnName+" Not Churned", nc_df[columnName].min()]
    metricslist.append(x)

    x =[columnName+" Not Churned", nc_df[columnName].mean()]
    metricslist.append(x)

    x =[columnName+" Not Churned", nc_df[columnName].max()]
    metricslist.append(x)

    train = pd.DataFrame(data=metricslist, columns=['parent_category_name', 'deal_probability'])
    parent_categories = train['parent_category_name'].unique()
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = iter(cm.rainbow(np.linspace(0, 1, len(parent_categories))))
    for parent_category in parent_categories:
        ax.plot(range(len(train[train["parent_category_name"] == parent_category])),
                sorted(train[train["parent_category_name"] == parent_category].deal_probability.values),
                color=next(colors),
                marker="o",
                label=parent_category)
    plt.ylabel(columnName+' likelihood ', fontsize=12)
    plt.title('Distribution of likelihood '+columnName+ ' Min vs Mean vs Max ' )
    plt.legend(loc="best")
    img_file = results_path.joinpath('Distribution_of_likelihood_'+columnName.replace(' ','_')+'_Min_vs_Mean_vs_Max.png')
    plt.savefig(img_file)
    plt.show()


print("Complete Duration: --- %s seconds ---" % (time.time() - start_time))