# Risk-Propensity-Churn-Analysis

## _Analyzing Risk and the Propensity of Customer Churn._

<a href="https://www.linkedin.com/in/alandanque"> Author: Alan Danque </a>

<a href="https://adanque.github.io/">Click here to go back to Portfolio Website </a>

![A remote image](https://adanque.github.io/assets/img/ChurnRisk.jpg)

Abstract: 

Currently Under Construction - AD

For most companies that provide a service or build and sell tangible products, it is the cash flow of their business that ensures their success, survival, or demise. For many of these companies being able to obtain new customers or clients can be very costly and timely. It is a challenge for many companies to continually identify or create new target markets or take customers from their competition. The costs are contributed to advertising, marketing campaigns, the human power to evangelize their products, marketing products to purchase to entice new customers and the worst which is opportunity costs of their service not being used or their product not being sold and sitting on a shelf. Thus, it is wise for companies to work toward keeping their customers for a long as they can. A happy and loyal customer will continue to buy, use, rate their products well, and recommend their favorite products creating a winning passive revenue situation that invites new customers through a network of their own customers. However, when a customer leaves it not only loses out on the continual revenue from that customer - it also reduces that opportunity of passive recommendation marketing. This creates a problem or opportunity for businesses as it is better for them to identify customers who may leave. So that they could take vital and timely steps toward preventing customers from leaving to become a customer of the competition. This is the process of identifying churn or the process of identifying factors that lead customers that may leave. 


### Project Specific Questions
The goals of my project are to answer the following questions.

- Since subscriptions are normally monthly with metrics collected once a month, will it be possible to identify churn in as little as a couple months?
	- Answer: 

- What are the indicators that help identify dissatisfaction?
	- Answer: 

- What are these factors that lead to loyalty?
	- Answer: 

- Is there a way to identify dissatisfaction between monthly subscription payments?
	- Answer: 

- Where can this data be derived from?
	- Answer: 

- How can we identify how much churn affects the bottom line?
	- Answer: 

- Can churn be prevented?
	- Answer: 

- Are there indirect factors that lead to churn?
	- Answer: 

- Are there an early detection sign?
	- Answer: 

- Is there a way to show how much prevented churn has affected the bottom line of cash flow?
	- Answer: 



## Included Project Variables / Factors 
### Project Dataset:
- Type:		CSV
- Columns: 	20
- Rows:		2,667


 | Feature / Factors | Data Type | Definition | 
 | --------- | --------- | --------- | 
 | State | string | State Abbreviation | 
 | Account length | integer | Length of Account number | 
 | Area code | integer | Phone Area Code | 
 | International plan | string | Does the customer have an international calling plan? | 
 | Voice mail plan | string | Does the customer have a voice mail plan | 
 | Number vmail messages | integer | Number of voicemails | 
 | Total day minutes | double | Total call minutes during the day | 
 | Total day calls | integer | Total calls during the day | 
 | Total day charge | double | Total day usage charges | 
 | Total eve minutes | double | Total call minutes during the evening | 
 | Total eve calls | integer | Total calls during the evening | 
 | Total eve charge | double | Total evening usage charges | 
 | Total night minutes | double | Total call minutes during the night | 
 | Total night calls | integer | Total calls during the night | 
 | Total night charge | double | Total night usage charges | 
 | Total intl minutes | double | Total call minutes international | 
 | Total intl calls | integer | Total calls internationally | 
 | Total intl charge | double | Total international usage charges | 
 | Customer service calls | integer | Customer Service call counts for the period | 
 | Churn | string | Has the customer closed their account? | 





## Pythonic Libraries Used in this project
Package               Version
--------------------- ---------
- async-generator       1.10
- asyncio               3.4.3
- certifi               2020.12.5
- dask                  2021.3.1
- decorator             4.4.2
- defusedxml            0.7.1
- folium                0.12.1
- geojson               2.5.0
- geopy                 2.1.0
- gmaps                 0.9.0
- gmplot                1.4.1
- greenlet              1.0.0
- htmlmin               0.1.12
- html-parser           0.2
- ImageHash             4.2.0
- imgkit                1.1.0
- importlib-metadata    2.1.1
- joblib                1.0.1
- jsonschema            3.2.0
- MarkupSafe            1.1.1
- matplotlib            3.3.4
- mplleaflet            0.0.5
- networkx              2.5
- notebook              6.3.0
- numpy                 1.20.1
- packaging             20.9
- pandas                1.2.3
- pandas-profiling      2.11.0
- pandasql              0.7.3
- piianalyzer           0.1.0
- pygeohash             1.2.0
- pydot                 1.4.2
- pyodbc                4.0.30
- PyYAML                5.4.1
- requests              2.25.1
- scikit-learn          0.24.1
- scipy                 1.6.1
- seaborn               0.11.1
- Shapely               1.7.1
- simplejson            3.17.2
- sklearn               0.0
- SQLAlchemy            1.4.2
- threadpoolctl         2.1.0
- typing-extensions     3.7.4.3
- urllib3               1.26.4
- websockets            8.1

## Repo Folder Structure

└───Datasets

└───Scripts

└───Results

## Python Files 

| File Name  | Description |
| ------ | ------ |
| FName.py | Desc |

## Datasets
| File  | Description |
| ------ | ------ |
| churn-bigml-80.csv | Orange Telecom Dataset | 

## Results




### Prediction


### Metrics Evaluation

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churn_Distribution.png?raw=true)

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churned_BY_State.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churn_BY_State.png?raw=true)
Deciding which one I like better

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churned_BY_Area_Code.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churned_BY_Customer_service_calls.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churned_BY_International_plan.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churned_BY_Voice_mail_plan.png?raw=true)



# Propensity Distribution of Likeliness
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_day_calls_Min_vs_Mean_vs_Max.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_day_charge_Min_vs_Mean_vs_Max.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_day_minutes_Min_vs_Mean_vs_Max.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_eve_calls_Min_vs_Mean_vs_Max.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_eve_charge_Min_vs_Mean_vs_Max.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_eve_minutes_Min_vs_Mean_vs_Max.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_intl_calls_Min_vs_Mean_vs_Max.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_intl_charge_Min_vs_Mean_vs_Max.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_intl_minutes_Min_vs_Mean_vs_Max.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_night_calls_Min_vs_Mean_vs_Max.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_night_charge_Min_vs_Mean_vs_Max.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_night_minutes_Min_vs_Mean_vs_Max.png?raw=true)



![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churned_Correlation_Matrix.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churn_Matrix.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/No_Churn_Matrix.png?raw=true)


### Principal Component Analysis
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/PCA_Cumulative_Explained_Variance.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/PCA_Heatmap.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Principal_Component_Scatter_Plot.png?raw=true)


### Outlier Detection
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Total_day_calls_and_charge_outlier_detection.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Total_eve_minutes_and_Total_eve_charge_outlier_detection.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Total_intl_minutes_and_Total_intl_charge_outlier_detection.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Total_night_minutes_and_Total_night_charge_outlier_detection.png?raw=true)

### Confusion Matrices
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Confusion_Matrix.png?raw=true)


![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/SVC_Confusion_Matrix.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/XGBoost_Confusion_Matrix.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/LogisticRegression_Confusion_Matrix.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Final_XGBoost_Confusion_Matrix.png?raw=true)

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Models_Recall_Results.png?raw=true)

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/XG_Classification_Report_Results.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/LR_Classification_Report_Results.png?raw=true)
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/SVC_Classification_Report_Results.png?raw=true)

Model Accuracy and MAE results are looking pretty interesting!
To measure the accuracy and loss of my model, I am using a set of my predicted values minus the actual target values between my train and test data. Then taking the mean of the absolute value of each in the set of values to divide this number by my target test values and then multiply by 100 to generate a mean absolute percentage error.  I then subtract 100 minus the mean absolute percentage error to produce accuracy metrics.


### Visual Analyses 



### Variable Correlation Reviews



## Appendices







## Data Sources
| Source  | Description | URL |
| ------ | ------ | ------ |
| Kaggle | Orange Telecom Dataset | https://www.kaggle.com/mnassrib/telecom-churn-datasets | 

## References: 

Kundu, A. (October 2018). Machine Learning Powered Churn Analysis for Modern Day Business Leaders. Retrieved from: https://towardsdatascience.com/machine-learning-powered-churn-analysis-for-modern-day-business-leaders-ad2177e1cb0d

flo.tausend. (January 2019). Hands-on: Predict Customer Churn. Retrieved from: https://towardsdatascience.com/hands-on-predict-customer-churn-5c2a42806266

DataScience.com. (March 2017).The Challenges of Building a Predictive Churn Model. Retrieved from https://www.kdnuggets.com/2017/03/datascience-building-predictive-churn-model.html

Mahtab,M. Verma,S. Paton, D. (July 2020).Gaining a Deeper Understanding of Churn Using Data Science Workspace. Retrieved from https://medium.com/adobetech/gaining-a-deeper-understanding-of-churn-using-data-science-workspace-18a2190e0cf3

Samineni, P. (August 2020). Telecom Churn Prediction. Retrieved from https://medium.com/analytics-vidhya/telecom-churn-prediction-9ce72c24e961

Ahmad, A.K., Jafar, A. & Aljoumaa, K. Customer churn prediction in telecom using machine learning in big data platform. J Big Data 6, 28 (2019). https://doi.org/10.1186/s40537-019-0191-6

Lin, S. (July 2018). 5 data science models for predicting enterprise churn. Retrieved from https://www.reforge.com/brief/5-data-science-models-for-predicting-enterprise-churn#6Gd9Hstees3kSLTvzl_JsQ?utm_medium=social&utm_source=facebook&utm_content=brief

Athar, S. (September 2020). Customer Churn Prediction : End to End Machine Learning Case Study. Retrieved from https://medium.com/@sayedathar11/customer-churn-prediction-end-to-end-machine-learning-case-study-c14cdc4d2c92

Choudhary, B. (October 2020). Churn Prediction Using Machine Learning. Retrieved from https://medium.com/swlh/churn-prediction-using-machine-learning-25c856201884

Sree, (October 2020). A step-by-step approach to predict customer attrition using supervised machine learning algorithms in Python. Retrieved from https://towardsdatascience.com/predict-customer-churn-in-python-e8cd6d3aaa7

Zyabkina, T. (n.d.). Propensity to Churn Modeling and Its Use for Churn Reduction. Retrieved from https://zyabkina.com/propensity-to-churn-modeling-does-not-help-reduce-churn/





:bowtie: