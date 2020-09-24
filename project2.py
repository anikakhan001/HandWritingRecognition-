#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


dataset = pd.read_csv('E://mnist_test/mnist_test.csv')
dataset.head()


# In[2]:


dataset['label'].unique()


# In[3]:


from sklearn.datasets import load_digits
digits = load_digits()


# In[4]:


import numpy as np 
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
 plt.subplot(1, 5, index + 1)
 plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)


# In[5]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)


# In[6]:


from sklearn.linear_model import LogisticRegression


# In[7]:


logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)


# In[8]:


logisticRegr.predict(x_test[0].reshape(1,-1))


# In[9]:


logisticRegr.predict(x_test[0:10])


# In[10]:


predictions = logisticRegr.predict(x_test)


# In[11]:


score = logisticRegr.score(x_test, y_test)
print(score)


# In[12]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
mnist


# In[13]:


print(mnist.data.shape)
# These are the labels
print(mnist.target.shape)


# In[14]:


from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split(
 mnist.data, mnist.target, test_size=1/7.0, random_state=0)


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
 plt.subplot(1, 5, index + 1)
 plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)


# In[16]:


from sklearn.linear_model import LogisticRegression


# In[17]:


# default solver is incredibly slow thats why we change it
logisticRegr = LogisticRegression(solver = 'lbfgs')


# In[18]:


logisticRegr.fit(train_img, train_lbl)


# In[19]:


logisticRegr.predict(test_img[0].reshape(1,-1))


# In[20]:


logisticRegr.predict(test_img[0:10])


# In[21]:


predictions = logisticRegr.predict(test_img)


# In[22]:


accuracy = logisticRegr.score(test_img, test_lbl)
print(accuracy)


# In[23]:


import numpy as np 
import matplotlib.pyplot as plt
index = 0
misclassifiedIndexes = []
for label, predict in zip(test_lbl, predictions):
 if label != predict: 
  misclassifiedIndexes.append(index)
  index +=1


# In[27]:


plt.figure(figsize=(20,4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
 plt.subplot(1, 5, plotIndex + 1)
 plt.imshow(np.reshape(test_img[badIndex], (28,28)), cmap=plt.cm.gray)
 plt.title('Predicted: {}, Actual: {}'.format(predictions[badIndex], test_lbl[badIndex]), fontsize = 15)


# In[ ]:




