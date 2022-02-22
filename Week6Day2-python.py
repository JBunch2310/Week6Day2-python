#!/usr/bin/env python
# coding: utf-8

# In[13]:


print (a)


# In[15]:


a = 'My Test'


# In[ ]:


#For this program, we're going to test out some of the capabilitier of jupter notebooks and python 


# In[2]:


print ("Hello World")


# In[12]:


#this is a list
myList = [1, 2, 'Test']
print(myList)


# In[11]:


a = 10
a = 'Test'
b = 5

print(a)


# In[8]:


student = {'name' : 'Jane Doe', 'grade': 100}
print(student['name'])


# In[16]:


#Train a model


# In[22]:


fruits = ['Orange', 'apple', 'kiwi']
fruits [-3]


# In[23]:


x = 10
if x > 5:
    print('X is larger')
else:
    print("x is smaller")


# In[24]:


def myFunction(x):
    return x**2

print(myFunction(10))


# In[57]:


import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np


# In[63]:


iris = datasets.load_iris()
print(iris)
X = iris.data[:, :2]
y = iris.target


# In[59]:


plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")


# In[61]:


df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target
print (df)


# In[65]:


# Splitting the dataset into the Training set and Test set
X = df.iloc[:, [0,1,2, 3]].values
y = df.iloc[:, 4].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[68]:


print(y_train)


# In[67]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto')
classifier.fit(X_train, y_train)


# In[70]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Predict probabilities
probs_y=classifier.predict_proba(X_test)
### Print results 
probs_y = np.round(probs_y, 2)


# In[71]:


print(probs)


# In[72]:


res = "{:<10} | {:<10} | {:<10} | {:<13} | {:<5}".format("y_test", "y_pred", "Setosa(%)", "versicolor(%)", "virginica(%)\n")
res += "-"*65+"\n"
res += "\n".join("{:<10} | {:<10} | {:<10} | {:<13} | {:<10}".format(x, y, a, b, c) for x, y, a, b, c in zip(y_test, y_pred, probs_y[:,0], probs_y[:,1], probs_y[:,2]))
res += "\n"+"-"*65+"\n"
print(res)


# In[76]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[77]:


# Plot confusion matrix
import seaborn as sns
import pandas as pd
# confusion matrix sns heatmap 
ax = plt.axes()
df_cm = cm
sns.heatmap(df_cm, annot=True, annot_kws={"size": 30}, fmt='d',cmap="Blues", ax = ax )
ax.set_title('Confusion Matrix')
plt.show()


# In[ ]:




