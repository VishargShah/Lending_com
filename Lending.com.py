# -*- coding: utf-8 -*-
"""
Created on Wed May 31 22:54:36 2017

@author: Visharg Shah
"""

################-----Lending.com-----###############

#Importing various libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the file
loans = pd.read_csv('loan.csv')

#Lets see various datatype
loans.info()

#Observing top 5 rows to get idea about data
loans.head()

#Checking out the title of column in data
loans.columns

#Observing presence of null values in different columns
plt.figure(figsize=(20,10))
sns.heatmap(loans.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Removing columns containing lot of na values and id,member_id as well as it is of no use
loans.drop(['id','member_id','open_acc_6m','open_il_6m','open_il_12m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m','max_bal_bc','all_util','open_il_24m',
            'inq_fi','total_cu_tl','inq_last_12m','annual_inc_joint','dti_joint','verification_status_joint','desc','mths_since_last_record','mths_since_last_major_derog' ],axis=1,inplace=True)

#Lets gain some statistical information
loans.describe()

#Filling null values with mean
loans['mths_since_last_delinq'] = loans['mths_since_last_delinq'].fillna(value=loans['mths_since_last_delinq'].mean())

loans['tot_cur_bal'] = loans['tot_cur_bal'].fillna(value=loans['tot_cur_bal'].mean()) 

loans['tot_coll_amt'] = loans['tot_coll_amt'].fillna(value=loans['tot_coll_amt'].mean()) 

loans['total_rev_hi_lim'] = loans['total_rev_hi_lim'].fillna(value=loans['total_rev_hi_lim'].mean()) 

loans['annual_inc'] = loans['annual_inc'].fillna(value=loans['annual_inc'].mean())

loans['total_pymnt'] = loans['total_pymnt'].fillna(value=loans['total_pymnt'].mean())

loans['delinq_2yrs'] = loans['delinq_2yrs'].fillna(value=loans['delinq_2yrs'].mean())

loans['inq_last_6mths'] = loans['inq_last_6mths'].fillna(value=loans['inq_last_6mths'].mean())

loans['pub_rec'] = loans['pub_rec'].fillna(value=loans['pub_rec'].mean())

loans['revol_util'] = loans['revol_util'].fillna(value=loans['revol_util'].mean())

#Again checking the info of data
loans.info()

#####---EDA---#####

##Lets observe if there any more na values with heatmap.
plt.figure(figsize=(20,10))
sns.heatmap(loans.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Observing different values in our target variable.
loans['loan_status'].unique()

##Current cant be taken as bad loan or indicator nor it can be considered as fully paid hence we are not going to consider it here

##Setting up Loans Status
loans['not_fully_paid'] = loans['loan_status'].map({'Current': 2 ,'Fully Paid': 0, 'Charged Off':1, 'Late(31-120 days)':1, 'In Grace Period': 1, 'Late(16-30 days)': 1, 'Does not meet the credit policy. Status:Fully Paid' : 0, 'Default': 1, 'Does not meet the credit policy. Status:Charged Off' : 1})

loans = loans[loans['not_fully_paid'] != 2]

loans["not_fully_paid"] = loans["not_fully_paid"].apply(lambda not_fully_paid : 0 if not_fully_paid == 0 else 1)

#Plotting countplot to see paid vs non paid counts.
sns.set_style('whitegrid')
sns.countplot(x='not_fully_paid',data=loans,palette='RdBu_r')

#Reason for taking loans?
#Creating countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid.
plt.figure(figsize=(20,20),dpi = 3)
sns.countplot(x='purpose',hue='not_fully_paid', data=loans,palette='Set1')
plt.show

##Hence most of the people are taking loan due to debt_consolidation and also more than 1,20,000 people are paying while around 45,000 not fully paid.Ratio for credit card is also almost look similar in terms of fully paid and not fully paid when compared to debt_consolidation.

#Plotting countplot of emp_length column with hue of not_fully_paid to see who are more likely to pay
plt.figure(figsize=(20,20))

sns.set_style('whitegrid')
sns.countplot(x='emp_length',hue='not_fully_paid',data=loans,palette='RdBu_r')
plt.show

#Value Count
loans['emp_length'].value_counts()

#Converting emp_length to number completely
loans['emp_length'] = loans['emp_length'].str.replace('+','')
loans['emp_length'] = loans['emp_length'].str.replace('<','')
loans['emp_length'] = loans['emp_length'].str.replace('years','')
loans['emp_length'] = loans['emp_length'].str.replace('year','')
loans['emp_length'] = loans['emp_length'].str.replace('n/a','0')

loans['emp_length'] = loans['emp_length'].map(float)

##We can see people with more than 10 years of experience are given loan more frequently which is quite obvious as there chances of paying loan back increases.But the peak of 1 years is higher than 6 years which means there is also other factors like home_ownership.So lets explore that.

#Importing plotly for more interactive plot
##For Jupyter nb
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print(__version__) # requires version >= 1.9.0

import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()

loans['home_ownership'].iplot(kind='hist',bins=25)

#Plotting barplot between int_rate and home_ownership
sns.barplot(x='int_rate',y='home_ownership',data= loans)

##Here we can see mortgage will have lower int_rate as compare to own and rent as there is some security so risk is less and hence int_rate is less. While of rent and none have highest int_rate as they are less secure situation.

#Lets see correlation between different variables using heatmap
plt.figure(figsize=(20,20))

sns.heatmap(loans.corr(),annot = True)
plt.show

##Hence we will remove feature which are highly correlated to each other to remove multicollinearity and choose independent feature which are correlated to target variable and important in our sense

final_data = [['annual_inc','emp_length','installment','loan_amnt','purpose','grade','int_rate','dti','revol_bal','total_pymnt','revol_util','last_pymnt_amnt',
'inq_last_6mths','delinq_2yrs','pub_rec','home_ownership','not_fully_paid']]

Final_data = pd.DataFrame(loans,columns= final_data)

#We need to transform categorical value of above into dummy variables so sklearn will be able to understand them.
cat_feats = ['purpose','grade','home_ownership']
Final_data = pd.get_dummies(Final_data,columns=cat_feats,drop_first=True)

#Train Test Split
from sklearn.cross_validation import train_test_split

X = Final_data.drop('not_fully_paid',axis=1)
y = Final_data['not_fully_paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

#Fitting Decision Tree
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

#Prediction and evaluation
predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))

##I am getting 98.5% accuracy.Just to be sure about our model.Lets try predicting with logistic regression and see if there is any drastic change in accuracy.

#Logistic Regression
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))

#Almost getting the same result around 98.5% no drastic change.


