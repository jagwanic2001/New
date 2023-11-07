#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[5]:


t_d = pd.read_csv("C://Users//user//OneDrive//Documents//train.csv")


# In[6]:


t_d.head()


# In[7]:


t_d.shape


# In[8]:


t_d.info()


# In[9]:


t_d.isnull().sum()


# In[10]:


t_d=t_d.drop(columns='Cabin',axis=1)


# In[11]:


t_d['Age'].fillna(t_d['Age'].mean(),inplace = True)


# In[12]:


t_d.isnull().sum()


# In[13]:


print(t_d['Embarked'].mode())


# In[14]:


t_d['Embarked'].fillna(t_d['Embarked'].mode()[0],inplace = True)


# In[15]:


t_d.isnull().sum()


# In[15]:


t_d.describe()


# In[16]:


sns.countplot('Survived',data = t_d)


# In[17]:


sns.countplot('Sex',data = t_d)


# In[18]:


sns.countplot('Sex',hue = 'Survived',data = t_d)


# In[19]:


sns.countplot('Pclass',data = t_d)


# In[20]:


sns.countplot('Pclass',hue='Survived',data = t_d)


# In[21]:


sns.countplot('Embarked',hue = 'Survived',data = t_d)


# In[22]:


t_d.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace = True)


# In[23]:


t_d.head()


# In[24]:


X = t_d.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis = 1)
Y = t_d['Survived']


# In[25]:


print(X)


# In[26]:


print(Y)


# In[27]:


X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size = 0.2,random_state=2)


# In[28]:


print(X_train)


# In[ ]:





# In[29]:


print(Y_train)


# In[30]:


print(X_test)


# In[31]:


model = LogisticRegression()


# In[32]:


model.fit(X_test,Y_test)


# In[33]:


X_test_p=model.predict(X_test)


# In[34]:


print(X_test_p)


# In[35]:


X_train_p = model.predict(X_train)


# In[36]:


print(X_train_p)


# In[ ]:


training_data = accuracy_score(Y_test,X_test_p)
print(training_data)


# In[29]:


t_d = pd.read_csv("C://Users//user//OneDrive//Documents//train.csv")


# In[30]:


t_d.head()


# In[ ]:





# In[51]:


t_d.head()


# In[56]:


t_d1= t_d.replace({'Survived':{10,20}},inplace = True)


# In[69]:


t_d.head() 


# In[70]:


t_d['Survived'].replace({'male':10,'female':20},inplace = True)


# In[94]:


t_d['Survived'].replace({1:3,0:4},inplace = True)


# In[95]:


t_d.head()


# In[77]:


type('Age')


# In[80]:


t_d.head()


# In[33]:


t_d['Survived'].replace({'D':0,'S':1},inplace = True)


# In[34]:


t_d.head()


# In[35]:





# In[115]:


df=pd.DataFrame(t_d)


# In[150]:


t_d['Survived'].value_counts()


# In[36]:


t_d['Sex'].value_counts()


# In[37]:


ax = sns.countplot('Sex',hue="Survived",data = t_d)
for i in ax.containers:
   ax.bar_label(i,)


# In[163]:


t_d['Sex'].value_counts()


# In[164]:


t_d['Survived'].value_counts()


# In[173]:


data = [31.871345,68.128655]
keys = ['MALES','FEMALES']
plt.pie(data,labels = keys,colors = sns.color_palette('dark'),autopct = '%.0f%%')
plt.title('Survived in titanic')
plt.legend(title = "survived")
plt.show()


# In[ ]:




