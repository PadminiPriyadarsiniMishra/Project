#!/usr/bin/env python
# coding: utf-8

# In[118]:


import pandas as pd


# In[119]:


df = pd.read_csv("Desktop/Ad Click Data.csv")


# In[120]:


df.head(30)


# In[121]:


df.tail()


# In[122]:


df.info()


# In[123]:


df.describe()


# In[124]:


df.drop('Ad Topic Line',axis=1,inplace=True)
df.drop('City',axis=1,inplace=True)
df.drop('Country',axis=1,inplace=True)
df.drop('Timestamp',axis=1,inplace=True)


# In[125]:


df.head(10)


# In[126]:


y=df['Clicked on Ad']


# In[127]:


df.drop('Clicked on Ad',axis=1,inplace=True)


# In[128]:


x=df


# In[129]:


from sklearn.model_selection import train_test_split


# In[130]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.30)


# In[131]:


from sklearn.linear_model import LogisticRegression


# In[132]:


lr=LogisticRegression()


# In[133]:


lr.fit(x_train,y_train)


# In[134]:


lr.score(x_test,y_test)


# In[135]:


from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(x_train,y_train)


# In[136]:


clf.score(x_test,y_test)


# In[137]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)


# In[138]:


dtc.score(x_test,y_test)


# In[139]:


import matplotlib.pyplot as plt


# In[140]:


df.hist(figsize=(10,11))
plt.show()


# In[141]:


import seaborn as sns
plt.figure(figsize=(20,20))
glue = sns.load_dataset("glue").pivot("Model", "Task", "Score")
sns.heatmap(glue,cmap="crest",annot=True, fmt=".1f")
plt.show()


# In[142]:


fig = plt.figure(figsize = (6,2))
sns.countplot(data = df , y = 'Male')
print(df['Male'].value_counts())


# In[ ]:




