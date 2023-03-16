#!/usr/bin/env python
# coding: utf-8

# Import all the necessary packages

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler , OrdinalEncoder, LabelEncoder 
from sklearn.model_selection import train_test_split , RandomizedSearchCV , cross_val_score , StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score , confusion_matrix


# Read the data

# In[2]:


adult = pd.read_csv(r'C:\Users\QA-1\Downloads\adult.data.csv' , header=None , na_values=' ?')


# In[3]:


col = ['age', 'workclass','fnlwgt','education' ,'education-num','marital-status','occupation' ,'relationship' ,'race' ,'sex' ,'capital-gain','capital-loss','hours-per-week','native-country', 'Income']


# In[4]:


adult.columns = col


# In[5]:


adult.head()


# In[6]:


adult.tail(10)


# In[7]:


adult.info()


# Find null values

# In[8]:


adult.isnull().sum()


# In[9]:


adult.shape


# In[10]:



adult.workclass.value_counts()


# In[11]:


adult.workclass.fillna('No_Info', inplace=True)


# In[12]:


adult.occupation.value_counts()


# In[13]:


adult[(adult.occupation.isnull()) & (adult.workclass=='No_Info')]


# In[14]:


adult.occupation.fillna('Unemployed',inplace=True)


# In[15]:


adult['native-country'].value_counts()


# In[16]:


adult[adult['native-country'].isnull()]


# In[17]:


adult['native-country'].fillna('other',inplace=True)


# In[18]:


adult.isnull().sum()


# In[19]:


adult.head()


# Replace the - by _ in all columns

# In[20]:



adult.columns = list(map(lambda x : x.replace('-','_') ,adult.columns))


# In[21]:



adult.head()


# In[22]:


adult.workclass.unique()


# In[23]:


adult.workclass = adult.workclass.apply(lambda x : x.replace('-','_'))


# In[24]:


adult.workclass.unique()


# Convert the target variable to numerical

# In[25]:


adult.Income.value_counts()


# In[26]:


adult.Income.value_counts() / len(adult)


# In[27]:


adult[adult.Income == ' <=50K']


# In[28]:


# the data is imbalanced 
adult.Income = adult.Income.apply(lambda x : 0 if x ==' <=50K' else 1)


# In[29]:


sns.countplot(adult.Income)


# In[30]:


adult.Income.value_counts()


# In[31]:


adult.age.max()


# In[32]:


adult.age.min()  


# In[33]:


adult = pd.get_dummies(data= adult, columns=['workclass'] , drop_first=True)


# In[34]:


adult.head()


# In[35]:


adult.education.unique()


# In[36]:


val = [[' Preschool',1] ,[' 1st-4th' ,2],[ ' 5th-6th' ,3] , [' 7th-8th' ,4] ,[ ' 9th',5] , [' 10th',6] ,[' 11th' ,7],
       [ ' 12th',8],[ ' Some-college',9] , [' Bachelors',10],
      [' HS-grad' ,11] ,[ ' Masters',12] ,[ ' Doctorate' ,13] , [' Prof-school' ,14] ,[' Assoc-acdm',15], [' Assoc-voc' ,16]]


# In[37]:



Ord_enc = OrdinalEncoder()


# In[38]:


adult.education = Ord_enc.fit_transform(np.array(adult.education).reshape(-1,1))


# In[39]:



adult.head()


# In[40]:


adult.marital_status.unique()


# 
# Remove the space from each value of categorical column

# In[41]:


adult.marital_status = adult.marital_status.str.replace(' ',"")


# In[42]:


adult.marital_status.unique()


# In[43]:


adult.marital_status = adult.marital_status.str.replace('-', '_')


# In[44]:



adult = pd.get_dummies(data = adult , columns=['marital_status'], drop_first=True)


# In[45]:


adult.occupation = Ord_enc.fit_transform(np.array(adult.occupation).reshape(-1,1))


# In[46]:



adult.relationship.unique()


# In[47]:



adult.relationship = adult.relationship.str.replace('-', '_')


# In[48]:


le = LabelEncoder()
adult.relationship = le.fit_transform(np.array(adult.relationship).reshape(-1,1))
adult.race = le.fit_transform(np.array(adult.race).reshape(-1,1))


# In[49]:


adult.race.unique()


# In[50]:


adult = pd.get_dummies(adult , columns=['sex'], drop_first=True)


# In[51]:


adult.head()


# In[52]:


adult.native_country.unique()


# In[53]:


adult.native_country = le.fit_transform(np.ravel(adult.native_country))


# In[54]:


adult.native_country.value_counts()


# In[55]:


adult.head(10)


# In[56]:


sns.boxplot(adult.fnlwgt)


# In[57]:


sns.boxplot(adult.capital_gain)


# In[58]:


adult.hours_per_week.max()


# In[59]:


adult.hours_per_week.min()


# # Apply min max scalar to fnlwgt and capital gain column

# In[60]:


min_trans = MinMaxScaler(feature_range=(0,100))


# In[61]:


for col in ['fnlwgt' ,'capital_gain']:
    adult[col] = min_trans.fit_transform(np.array(adult[col]).reshape(-1,1))


# In[62]:



adult.head()


# # perform feature selection

# In[63]:


plt.figure(figsize=(25,18))
sns.heatmap(adult.corr() , annot=True)


# In[64]:



# find the feature importance using Random Forest model


# In[65]:


RFC = RandomForestClassifier(n_estimators=200)


# In[66]:


RFC.fit(adult.drop('Income',axis=1) , adult.Income )


# In[67]:


RFC.feature_importances_


# In[68]:


feature_score = pd.DataFrame(RFC.feature_importances_ , index= adult.drop('Income',axis=1).columns )


# In[69]:


feature_score.columns = ['score']


# In[70]:


feature_score.sort_values(by='score', ascending=False)


# In[71]:


x = adult.drop('Income' , axis=1)
y = adult.Income


# In[72]:


mut_info = mutual_info_classif(x,y)


# In[73]:


pd.DataFrame(mut_info , index=x.columns).sort_values(0,ascending=False)


# In[74]:


from sklearn.feature_selection import SelectKBest


# # Model Building

# In[75]:


LR = LogisticRegression(class_weight= {0 : 3, 1 : 7})


# In[76]:


x= adult.drop('Income' , axis=1)
y= adult.Income


# In[77]:


x_train , x_test , y_train ,y_test = train_test_split(x , y, test_size=0.2, stratify=y , random_state= 86252)


# In[78]:


cross_val_score(LR ,x ,y ,cv=20,scoring= 'accuracy',n_jobs=-1)


# In[79]:


LR.fit(x_train,y_train)


# In[80]:


LR.predict(x_test)


# In[81]:


accuracy_score(y_test , LR.predict(x_test))


# In[82]:


confusion_matrix(y_test , LR.predict(x_test))


# In[83]:


data = LR.predict_proba(x_test)
data[0]


# In[84]:


output = []
for i in range(len(data)):
    if data[i][0] > 0.3:
        output.append(0)
    else:
        output.append(1)


# In[85]:


output


# In[86]:


confusion_matrix(y_test , output)


# # Random forest

# In[87]:


RFC = RandomForestClassifier(n_estimators=200, max_depth= 8)


# In[88]:


RFC.fit(x_train,y_train)


# In[89]:


RFC.predict(x_test)


# In[90]:


accuracy_score(y_test ,RFC.predict(x_test))


# In[91]:


confusion_matrix(y_test ,RFC.predict(x_test))


# In[92]:


param = {'n_estimators' : [100,200,400,500,700,1000],
        'max_depth' : [5,8,10,12,15,20],
        'min_samples_split' : [2,3,4,5],
         'min_samples_leaf' : [1,2,3,4,5,6],
        'criterion' : ['gini' , 'entropy'],
        'class_weight' : ['balanced' , {0:3,1:7},{0:2,1:8}]}


# In[93]:


rnd_cv = RandomizedSearchCV(RFC , param , cv=10 , scoring='accuracy' ,n_iter=5 , n_jobs=-1)


# In[95]:


rnd_cv.fit(x,y)


# In[96]:


rnd_cv.best_params_


# In[97]:


rnd_cv.best_estimator_


# In[98]:


rnd_cv.best_score_


# In[99]:


confusion_matrix(y_test ,rnd_cv.predict(x_test))


# In[100]:


pred = rnd_cv.predict(x_test)
f1_score(y_test ,y_pred = np.array(pred))


# # use randomized search cv to get best model out of different models

# In[101]:


clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier()
clf_3 = RandomForestClassifier()
clf_4 = AdaBoostClassifier()


# In[102]:


from sklearn.pipeline import Pipeline


# In[103]:


pipe = Pipeline([('classifier' , clf_1)])


# In[128]:


# for Logistic regression
param1 = {}
param1['classifier'] = [clf_1]
param1['classifier__penalty'] = ['l1','l2']
param1['classifier__solver'] = ['lbfgs', 'liblinear']
param1['classifier__class_weight'] = [{0:3,1:7},{0:1,1:9},{0:2,1:8}]

# for Decision Tree
param2 = {}
param2['classifier'] = [clf_2]
param2['classifier__max_depth'] = [3,5,7,8,10,12]
param2['classifier__min_samples_split'] = [2,4,5,7,8,10]
param2['classifier__criterion'] = ['gini', 'entropy']
param2['classifier__class_weight'] = [{0:3,1:7},{0:1,1:9},{0:2,1:8}]

# for Random forest
param3 = {}
param3['classifier'] = [clf_3]
param3['classifier__max_depth'] = [3,5,7,8,10,12]
param3['classifier__n_estimators'] = [100,200,300,400,500,700]
param3['classifier__min_samples_split'] = [2,4,5,7,8,10]
param3['classifier__criterion'] = ['gini', 'entropy']
param3['classifier__class_weight'] = [{0:3,1:7},{0:1,1:9},{0:2,1:8}]
# for Adaboost
param4 = {}
param4['classifier'] = [clf_4]
param4['classifier__n_estimators'] = [50,100,150,200,300,350,400,]
param4['classifier__learting_rate'] = [0.5,0.8,1,1.5,2,3]


# In[129]:


param = [param1, param2 ,param3 , param4]


# In[130]:


rand_cv = RandomizedSearchCV(pipe , param , cv=10, scoring='accuracy' ,n_iter=5 , n_jobs=-1, return_train_score=True)


# In[131]:


rand_cv.fit(x,y)


# In[132]:


rand_cv.best_estimator_


# In[133]:


rand_cv.best_score_


# In[134]:


rand_cv.best_params_


# In[135]:


rand_cv.scorer_


# In[136]:


adult[adult.columns[7:]].head(10)


# In[137]:


len([45,6.9,6,7,5,0,4,3.5,1,67,10,1,0,0,0,0,0,0,0,1,0,0,1,1,0,1])


# In[138]:


rand_cv.predict([[45,6.9,6,7,5,0,4,3.5,1,67,10,1,0,0,0,0,0,0,0,1,0,0,1,1,0,1]])


# In[139]:


import pickle


# # save a model using Pickle

# In[140]:


with open('mini_project_2.pkl' , 'wb+') as f:
    pickle.dump(rand_cv,f)


# # Read a pickle file

# In[141]:


# 1st Approach
pd.read_pickle('mini_project_2.pkl')


# In[142]:


# 2nd Approach
with open('mini_project_2.pkl' , 'rb') as f:
    model = pickle.load(f)


# In[143]:


model.best_estimator_

