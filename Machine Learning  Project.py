#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import KFold


# In[2]:


df=pd.read_csv('Cancer.csv')
df


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)
len(df)


# In[9]:


df.diagnosis.unique()


# In[10]:


df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df['diagnosis']


# In[11]:


df.head()


# In[12]:


df.describe()


# In[13]:


plt.hist(df['diagnosis'])
plt.title('Diagnosis (M=1 , B=0)')
plt.show()


# In[14]:


features_mean = list(df.columns[1:11])
features_mean


# In[15]:


dfM=df[df['diagnosis'] ==1]
dfB=df[df['diagnosis'] ==0]


# In[16]:


import matplotlib.pyplot as plt


# In[17]:


features_mean = list(df.columns[2:12])
plt.figure(figsize=(8, 10))
for i, feature in enumerate(features_mean, 1):
    plt.subplot(5, 2, i)
    plt.hist(dfM[feature], bins=50, alpha=0.5, label='M',color='r', density=True)
    plt.hist(dfB[feature], bins=50, alpha=0.5, label='B', color='g', density=True)
    plt.legend(loc='upper right')
    plt.title(feature)
    plt.tight_layout()


# In[27]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

traindf, testdf = train_test_split(df, test_size = 0.3)

def classification_model(model, data, predictors, outcome, n_folds=5,random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[outcome],
                                                        test_size=0.2, random_state= 42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.3%}")
    
    cross_val_scores = cross_val_score(model, data[predictors], data[outcome], cv=n_folds)
    mean_cross_val_score = cross_val_scores.mean()
    print(f"Cross-Validation Score: {mean_cross_val_score:.3%}")
    return model

predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean',
                 'concave points_mean']
outcome_var='diagnosis'
model=LogisticRegression()
classification_model(model,traindf,predictor_var,outcome_var)
predictor_var = ['radius_mean']
model=LogisticRegression()
classification_model(model,traindf,predictor_var,outcome_var)


# In[19]:


predictor_vars = ['radius_mean', 'perimeter_mean','area_mean',
                  'compactness_mean', 'concave points_mean']
outcome_var = 'diagnosis'
model = LinearRegression()
classification_model(model,traindf,predictor_var,outcome_var)
predictor_var = ['radius_mean']
model=LinearRegression()
classification_model(model,traindf,predictor_var,outcome_var)


# In[20]:


from sklearn.tree import DecisionTreeClassifier

predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean',
                 'concave points_mean']
model = DecisionTreeClassifier()
classification_model(model,traindf,predictor_var,outcome_var)
predictor_var = ['radius_mean']
model = DecisionTreeClassifier()
classification_model(model,traindf,predictor_var,outcome_var)


# In[21]:


predictor_var = ['radius_mean', 'perimeter_mean', 'area_mean', 
                 'compactness_mean', 'concave points_mean']
outcome_var = 'diagnosis'
model = KNeighborsClassifier()
classification_model(model, traindf, predictor_var, 
outcome_var)
predictor_var = ['radius_mean']
model = KNeighborsClassifier()
classification_model(model, traindf, predictor_var, 
outcome_var)


# In[22]:


from sklearn.ensemble import RandomForestClassifier

predictor_var = features_mean
model = RandomForestClassifier(n_estimators=100,min_samples_split=25, 
                               max_depth=7, max_features=2)
classification_model(model,traindf,predictor_var,outcome_var)
featimp = pd.Series(model.feature_importances_, 
index=predictor_var).sort_values(ascending=False)
print(featimp)


# In[23]:


predictor_var = ['concave points_mean','area_mean','radius_mean','perimeter_mean',
                 'concavity_mean',]
model = RandomForestClassifier(n_estimators=100, min_samples_split=25,
                               max_depth=7, max_features=2)
classification_model(model,traindf,predictor_var,outcome_var)


# In[24]:


predictor_var = ['radius_mean']
model = RandomForestClassifier(n_estimators=100)
classification_model(model,traindf,predictor_var,outcome_var)
predictor_var = features_mean
model = RandomForestClassifier(n_estimators=100,min_samples_split=25,max_depth=7,
                               max_features=2)
classification_model(model,testdf,predictor_var,outcome_var)

