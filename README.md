# Risk-Propensity-Churn-Analysis

## _Analyzing Risk and the Propensity of Customer Churn._

<a href="https://www.linkedin.com/in/alandanque"> Author: Alan Danque </a>

<a href="https://adanque.github.io/">Click here to go back to Portfolio Website </a>

<p align="center">
  <img width="460" height="400" src="https://adanque.github.io/assets/img/ChurnRisk.jpg">
</p>

## Abstract: 

For most companies that provide a service or build and sell tangible products, it is the cash flow of their business that ensures their success, survival, or demise. For many of these companies being able to obtain new customers or clients can be very costly and timely. It is a challenge for many companies to continually identify or create new target markets or take customers from their competition. The costs are contributed to advertising, marketing campaigns, the human power to evangelize their products, marketing products to purchase to entice new customers and the worst which is opportunity costs of their service not being used or their product not being sold and sitting on a shelf. Thus, it is wise for companies to work toward keeping their customers for a long as they can. A happy and loyal customer will continue to buy, use, rate their products well, and recommend their favorite products creating a winning passive revenue situation that invites new customers through a network of their own customers. However, when a customer leaves it not only loses out on the continual revenue from that customer - it also reduces that opportunity of passive recommendation marketing. This creates a problem or opportunity for businesses as it is better for them to identify customers who may leave. So that they could take vital and timely steps toward preventing customers from leaving to become a customer of the competition. This is the process of identifying churn or the process of identifying factors that lead customers that may leave. 

<p align="center">
  <img width="460" height="400" src="https://adanque.github.io/assets/img/Churn460.jpg">
</p>

### Project Specific Questions
The goals of my project are to answer the following questions.

- Since subscriptions are normally monthly with metrics collected once a month, will it be possible to identify churn in as little as a couple months?
	- Answer: Yes, I found that it is still possible to build a prediction model using a snapshot of data.

- What are the indicators that help identify dissatisfaction?
	- Answer: Excessive Day, Eve and Night Usage Charge amounts and if the customer has an international plan.

- What are these factors that lead to loyalty?
	- Answer: Lower charges relating to usage.

- Is there a way to identify dissatisfaction between monthly subscription payments?
	- Answer: The relation of charges to usage.

- Where can this data be derived from?
	- Answer: From the customer usage minutes and the related charges. In conjunction with their plan options.

- How can we identify how much churn affects the bottom line?
	- Answer: Would need financials data describing revenue and costs to answer this question.

- Can churn be prevented?
	- Answer: Yes, by possibly allowing a rate discount program using a loyalty-based system.

- Are there indirect factors that lead to churn?
	- Answer: When the phone usage occurs.  

- Are there an early detection sign?
	- Answer: Yes, if usage monitoring and alerting were available – then it may be the customer alert function that would help prevent the customer from churning.

- Is there a way to show how much prevented churn has affected the bottom line of cash flow?
	- Answer: : I believe it is possible. However, this dataset does not include the information needed to answer this question.


##  Project Variables / Factors 
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


##  Methods used
1.	Generate Pandas Profiling report for all rows in dataset.
2.	Filter dataset on churn label for Churn = True and rerun pandas profiling report.
3.	Filter dataset on churn label for Churn = False and rerun pandas profiling report.
4.	Create visualizations of interactions between variables ie boxplots, bar graphs, line graphs, correlation matrices, scatter plots. 
5.	Document observations between variable correlations between the churn and not churn.
6.	Evaluate algorithms and predictive models using F1 score, accuracy, and confusion matrices.


## Pythonic Libraries Used in this project
Package               Version
--------------------- ---------
- imbalanced-learn      0.8.0
- imblearn              0.0
- xgboost               1.3.3
- sklearn               0.0
- pandas                1.2.3
- numpy                 1.20.1
- matplotlib            3.3.4
- pandas-profiling      2.11.0
- pandasql              0.7.3
- pyodbc                4.0.30
- PyYAML                5.4.1
- SQLAlchemy            1.4.2
- seaborn               0.11.1

## Repo Folder Structure

└───Datasets

└───Scripts

└───Results

## Python Files 

| File Name  | Description |
| ------ | ------ |
| ExploratoryDataAnalysisReview.py | Data Reviews, Exploratory Analysis and Data Wrangling |
| ChurnAnalysisGraphs.py | Generates all graphs and plots as displayed in my project |
| PCA_Ensemble_Model_Testing.py | Correlation Matrix, Principal Component Analysis, Model testing and Evaluations |

## Datasets
| File  | Description |
| ------ | ------ |
| churn-bigml-80.csv | Orange Telecom Dataset | 

## Analyses

### Metrics Evaluation and Observations

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churn_Distribution.png?raw=true)
Our churn dataset is imbalanced. With 14% churn.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churned_BY_State.png?raw=true)
Here we can see the significant difference between churn by state.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churned_BY_Area_Code.png?raw=true)
The data set contains mostly California area codes for San Jose (408), San Francisco (415) and Oakland (510). With San Fracisco having the most customer churn.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churned_BY_Customer_service_calls.png?raw=true)
This chart indicates that the number of customer service calls is likely not a contributing factor to customer churn. As those with 0 and 1 service calls have the most churned customers.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churned_BY_International_plan.png?raw=true)
This plot above displays that less than 40% of customers with an international plan have churned.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churned_BY_Voice_mail_plan.png?raw=true)
This visualization displays a low amount of churn based on voicemail. Indicating that their voicemail plan is less likely a reason for churning.

### Outlier Detection
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Total_day_calls_and_charge_outlier_detection.png?raw=true)
The above box plot displays outliers for those who have not churned. However, does not have any outliers for those who do churn. Possibly indicating that usage is normal before they churn.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Total_eve_minutes_and_Total_eve_charge_outlier_detection.png?raw=true)
The above box plot is interesting as those who do churn display as outliers outside the max areas. Indicating that evening rates and excessive usage may be a factor of churn.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Total_intl_minutes_and_Total_intl_charge_outlier_detection.png?raw=true)
The above box plot appears to be much like the evening box plot as the outliers above the max mark for those who churn indicate it may be contributing factor to churn. 

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Total_night_minutes_and_Total_night_charge_outlier_detection.png?raw=true)
The above appears to be again similiar to the evening and international max value churn outliers. Indicating that these customers may have been unhappy by the related rates for usage for nightly rates.

# Propensity Distribution of Likeliness
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_day_calls_Min_vs_Mean_vs_Max.png?raw=true)
The above graph shows that day calls are not likely a factor to those who churned.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_day_charge_Min_vs_Mean_vs_Max.png?raw=true)
The above graph does display that the charges for calls may be likely a contributing factor to churn.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_day_minutes_Min_vs_Mean_vs_Max.png?raw=true)
The above graph supports the day charges as relates to the number of minutes contributing to the charges. Therefore a likely factor to churn.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Distribution_of_likelihood_Total_eve_calls_Min_vs_Mean_vs_Max.png?raw=true)
The above is an interesting propensity graph as it appears those who churn may have incurred excessive charges earlier in the period and therefore churned as a cause of it.

### Variable Correlation Reviews
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churned_Correlation_Matrix.png?raw=true)

## Filtered Correlation Reviews
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Churn_Matrix.png?raw=true)
The above correlation matrics shows an interesting set of correlations between variables that may help with identifying trends for customers before they churn.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/No_Churn_Matrix.png?raw=true)
The above correlation matrix shows +

### Principal Component Analysis
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/PCA_Cumulative_Explained_Variance.png?raw=true)
Here we found that 10 components explains 90% of our variance of 17 components.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/PCA_Heatmap.png?raw=true)
We can see above there is no correlation between our components.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Principal_Component_Scatter_Plot.png?raw=true)
Here we can see tight angles between PCs from coordinates 0,0 indicating close positive correlations. 
These include:
- Total intl call plans and charges. 
- Total day minutes and day charges.
- Total eve minutes and eve charges.
- Total night minutes and night charges.


### Prediction
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Models_Recall_Results.png?raw=true)
Recall testing to estimate positive predictions using Cross validation. 

### Note: The following uses GridSearchCV to itereate through a list of parameter options for each of the three algorithms to optimize the prediction results.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/XG_Classification_Report_Results.png?raw=true)
Using XGBoost, we received high precision, recall and F1 Score. And an accuracy of 84%
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/Final_XGBoost_Confusion_Matrix.png?raw=true)
The above confusion matrix supports our findings on it's ability to predict non churn. However I believe since this is a classification between two classes ie Churn not Churn we can deduce from those who are predicted to not churn to identify possible churn.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/LR_Classification_Report_Results.png?raw=true)
Using Logistic Regression, we received high precision, ok recall and an ok F1 Score. However the accuracy is at 61%
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/LogisticRegression_Confusion_Matrix.png?raw=true)
The above confusion matrix supports the finding for Logistic Regression. However I did find that the score does appear to improve when I increase the max iterations to the model.

![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/SVC_Classification_Report_Results.png?raw=true)
Using Support Vector Classification, we received high precision, ok recall and an ok F1 Score. However the accuracy was also at 61%
![A remote image](https://github.com/adanque/Risk-Propensity-Churn-Analysis/blob/main/Results/SVC_Confusion_Matrix.png?raw=true)
the above confusion matrix supports the estimates displayed earlier on its ability to predict those who do not churn. 

### Conclusion: 
- XGBoost in my case is the best algorithm to base my model to predict churn.
- The predictive variables to watch: If the customer has an international plan, high costs per Total Day Charge, Total Eve Charge and Total Night Charge.
- These factors appear to indicate that when charges are excessively high for their usage, the customer may likely churn to seek lower rates with another provider.

### Recommendation: 
- To reduce and prevent customer churn, I would recommend that the telecom vendor institutes a loyalty system that discounts high amounts of charges when the charges suddenly occur either on first instance or for several instances per month. 



## Appendices
### Data Sources
| Source  | Description | URL |
| ------ | ------ | ------ |
| Kaggle | Orange Telecom Dataset | https://www.kaggle.com/mnassrib/telecom-churn-datasets | 


### References: 

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