#!/usr/bin/env python
# coding: utf-8

# <!-- ######## This is a comment, visible only in the source editor  ######## -->
# <h1 style="color: #ff9900; text-align: center;"><em><strong>The sparks foundation</strong></em></h1>
# <p style="text-align: center;"><strong>Intern: Data science and business analytics</strong></p>
# <hr />
# <h2 style="color: #ff2a00; text-align: center;"><strong>Name : Akash. S</strong></h2>
# <h3 style="text-align: center;"><strong>Task 6: Prediction using Decision tree algorithm</strong></h3>
# <h3 style="text-align: left;">Queries :</h3>
# <div class="lm-Widget p-Widget jp-InputArea jp-Cell-inputArea">
# <div class="lm-Widget p-Widget jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
# <ul>
# <li>Create a decision tree and visualize it graphically</li>
# <li>If we feed any new data it should be able to predict the class accordingly</li>
# </ul>
# <h3>Dataset : <a href="https://bit.ly/3kXTdox" target="https://bit.ly/3kXTdox" rel="opener">https://bit.ly/3kXTdox</a></h3>
# <p></p>
# </div>
# </div>

# ## <center> Implementation of Decision Tree Algorithm using Sklearn </center>

# ### Importing the dataset

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns',None)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('https://bit.ly/3kXTdox')
print('The shape of the dataset : ',data.shape)
data.head()

Observation : So the dataset contains 150 rows and 6 columns including the dependent and the independent features
# ### Exploratory Data analysis

# #### Describing the data

# In[3]:


data.describe()


# #### Checking for missing values

# In[4]:


for i in data.columns:
    print(f'{i} has {data[i].isnull().sum()} missing values')

Observation : Hence, the features has no missing values
# #### Correlation between features

# In[5]:


data.corr()


# In[6]:


sns.heatmap(data.corr(),annot=True)
plt.show()

Observation : The correlation status between the features are viewed and it is visible that negative correlation exists majorily
# #### Detection of outliers

# In[7]:


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
data['SepalLengthCm'].plot.box()
plt.subplot(2,2,2)
data['SepalWidthCm'].plot.box()
plt.subplot(2,2,3)
data['PetalLengthCm'].plot.box()
plt.subplot(2,2,4)
data['PetalWidthCm'].plot.box()
plt.show()

Observation : As per the box plot there are some outliers present in the 'Sepal Width Cm' feature where the maximum value of the feature is 4.0 and the minimum value of the feature is 2.0 which is an outlier and the values greater than 4.0 are considered to be outliers
# #### Plotting the distribution 

# In[8]:


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.distplot(data['SepalLengthCm'])
plt.subplot(2,2,2)
sns.distplot(data['SepalWidthCm'])
plt.subplot(2,2,3)
sns.distplot(data['PetalLengthCm'])
plt.subplot(2,2,4)
sns.distplot(data['PetalWidthCm'])
plt.show()

Observation : The 'SepalWidthCm' feature is somehow normally distributed than the others
# ### Feature engineering

# #### Removing the feature 'Id'

# In[9]:


data.drop('Id',axis=1,inplace=True)


# In[10]:


data.head()


# #### Removing the outliers in 'SepalWidthCm'

# In[11]:


upper_boundary = data.SepalWidthCm.mean() + 3*data.SepalWidthCm.std()
lower_boundary = data.SepalWidthCm.mean() - 3*data.SepalWidthCm.std()
print(f'Consider the data greater than {lower_boundary} and lesser than {upper_boundary}')


# In[12]:


upper_outlier = data[data['SepalWidthCm'] > upper_boundary].index
data.drop(upper_outlier,inplace = True)
lower_outlier = data[data['SepalWidthCm'] < lower_boundary].index
data.drop(lower_outlier,inplace = True)
data.shape


# ### Data splitting

# In[13]:


X = data.drop(['Species'],axis = 1)
y = data[['Species']]


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
print(f'The shape of training data is : {x_train.shape,y_train.shape}')
print(f'The shape of testing data is : {x_test.shape,y_test.shape}')


# ### Model creation

# In[15]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
print('Model has been created and trained')


# ### Representing the model

# #### 1) Using text representation

# In[16]:


print(tree.export_text(model))


# #### 2) Using graphical representation

# In[17]:


fig = plt.figure(figsize=(10,10))
_ = tree.plot_tree(model, 
                   feature_names=data.columns[:-1],  
                   class_names=data.Species.unique(),rounded=True,filled=True,fontsize=14)
plt.savefig('tree.png',format='png',bbox_inches = "tight")


# ### Testing the model

# In[18]:


y_pred = model.predict(x_test)


# ### Evaluation metrics of the model

# In[19]:


from sklearn.metrics import accuracy_score
result = accuracy_score(y_test,y_pred)
print('The accuracy of the decision tree model : ',result)

