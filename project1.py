#!/usr/bin/env python
# coding: utf-8

# # Import The Dataset

# In[16]:


import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


# # Code for images having the same dimensions and properties. 

# In[17]:


transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])


# # Download the data sets and load them to DataLoader

# In[18]:


trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)


# # Shape the images and the labels

# In[19]:


dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)


# # Display one image from the training set

# In[20]:


plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');


# # Show how the dataset looks like.

# In[21]:


figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')


# # Build the neural network

# In[22]:


input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)


# In[23]:


criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss


# # Adjusting Weights

# In[24]:


print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)


# # neural network loop over the training set and updates the weights.
# Number of times we iterate over the training set, we will be seeing a gradual decrease in training loss.

# In[25]:


optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)


# # Testing & Evaluation
# to see how the model works.

# In[26]:


def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.cpu().data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    
    images, labels = next(iter(valloader))

img = images[0].view(1, 784)
with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)


# # calculate the accuracy.
# Result

# In[27]:


correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)

    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))


# # Saving The Model
#  we can load it and use it directly without further training.

# In[28]:


torch.save(model, './my_mnist_model.pt')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


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


# In[ ]:


dataset['label'].unique()


# In[ ]:


from sklearn.datasets import load_digits
digits = load_digits()


# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
 plt.subplot(1, 5, index + 1)
 plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)


# In[ ]:


logisticRegr.predict(x_test[0].reshape(1,-1))


# In[ ]:


logisticRegr.predict(x_test[0:10])


# In[ ]:


predictions = logisticRegr.predict(x_test)


# In[ ]:


score = logisticRegr.score(x_test, y_test)
print(score)


# In[ ]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
mnist


# In[ ]:


print(mnist.data.shape)
# These are the labels
print(mnist.target.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split(
 mnist.data, mnist.target, test_size=1/7.0, random_state=0)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
 plt.subplot(1, 5, index + 1)
 plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


# default solver is incredibly slow thats why we change it
logisticRegr = LogisticRegression(solver = 'lbfgs')


# In[ ]:


logisticRegr.fit(train_img, train_lbl)


# In[ ]:


# Returns a NumPy Array
# Predict for One Observation (image)
logisticRegr.predict(test_img[0].reshape(1,-1))


# In[ ]:


logisticRegr.predict(test_img[0:10])


# In[ ]:


predictions = logisticRegr.predict(test_img)


# In[31]:


score = logisticRegr.score(test_img, test_lbl)
print(accuracy:)


# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt
index = 0
misclassifiedIndexes = []
for label, predict in zip(test_lbl, predictions):
 if label != predict: 
  misclassifiedIndexes.append(index)
  index +=1


# In[ ]:


plt.figure(figsize=(20,4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
 plt.subplot(1, 5, plotIndex + 1)
 plt.imshow(np.reshape(test_img[badIndex], (28,28)), cmap=plt.cm.gray)
 plt.title(‘Predicted: {}, Actual: {}’.format(predictions[badIndex], test_lbl[badIndex]), fontsize = 15)


# In[ ]:





# In[ ]:




