#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# **reading csv file**

# In[2]:


df=pd.read_csv("diabetes.csv")


# **dataset values**

# In[3]:


df.head()


# In[4]:


df.tail()


# **data types**

# In[5]:


df.dtypes


# In[6]:


import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=['BMI','DiabetesPedigreeFunction'])
df=encoder.fit_transform(df)


# In[7]:


df.columns


# **checking null values**

# In[8]:


df.isnull().sum()


# In[9]:


df.isnull().sum().sum()


# **statistics of data**

# In[10]:


import pandas as pd
summary_stats = df.describe()
print(summary_stats)


# In[11]:


mean = df.mean()
median = df.median()
std = df.std()
min_val = df.min()
max_val = df.max()
quartiles = df.quantile([0.25, 0.50, 0.75])

print("Mean:")
print(mean)
print("\nMedian:")
print(median)
print("\nStandard Deviation:")
print(std)
print("\nMinimum:")
print(min_val)
print("\nMaximum:")
print(max_val)
print("\nQuartiles:")
print(quartiles)


# In[12]:


df.columns


# **Data Visualization**

# In[13]:


df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()


# In[14]:


plt.figure(figsize=(10, 8))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.show()


# In[15]:


plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Glucose', y='BMI')
plt.show()


# In[16]:


plt.figure(figsize=(6, 6))
sns.countplot(data=df, x='Outcome')
plt.show()


# **Class Imbalance**

# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt

class_counts = df['Outcome'].value_counts()

plt.figure(figsize=(6, 6))
sns.countplot(data=df, x='Outcome')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Distribution of Outcome')
plt.show()

print(class_counts)


# **Feature Correlations**

# In[18]:


numerical_features = df.select_dtypes(include='number')

correlation_matrix = numerical_features.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True,cmap='coolwarm',square=True)
plt.title('Correlation Matrix')
plt.show()


# **Outiler Detection**
# 

# In[19]:


numericals=df.select_dtypes(include='number')


# In[20]:


z_scores = (numericals-numericals.mean())/numericals.std()
zt=3


# In[21]:


outliers=(z_scores > zt)|( z_scores < -zt)
plt.figure(figsize=(10,8))
sns.boxplot(data=numericals)
plt.xticks(rotation=45)
plt.title('box plot of numerical features')


# In[22]:


print(outliers.sum())


# In[23]:


df.shape


# **KDE plot **

# In[24]:


num_columns = len(df.columns.drop('Outcome'))
num_rows = (num_columns + 1) // 2

plt.figure(figsize=(10, 5*num_rows))
for i, column in enumerate(df.columns.drop('Outcome')):
    plt.subplot(num_rows, 2, i+1)
    sns.kdeplot(data=df[column], fill=True)
    plt.title(f'KDE plot of {column}')

plt.tight_layout()
plt.show()


# **Splitting the Dataset into training set and test set**

# In[25]:


from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[26]:


x=df.drop(['Outcome'],1)
y=df['Outcome']


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=1)


# **Scikit-Learn Pipeline**

# In[28]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# In[29]:


pipeline_lr = Pipeline([('scalar1', StandardScaler()), ('lr_classifier', LogisticRegression())])
pipeline_knn = Pipeline([('scalar2', StandardScaler()), ('knn_classifier', KNeighborsClassifier())])
pipeline_dc = Pipeline([('scalar3', StandardScaler()), ('dc_classifier', DecisionTreeClassifier())])
pipeline_svc = Pipeline([('scalar4',StandardScaler()), ('svc_classifier', SVC())])
pipeline_rf = Pipeline([('scalar5', StandardScaler()), ('rf_classifier', RandomForestClassifier())])
pipeline_gb = Pipeline([('scalar6', StandardScaler()), ('gb_classifier', GradientBoostingClassifier())])


# In[30]:


pipelines=[pipeline_lr,pipeline_knn,pipeline_dc,pipeline_svc,pipeline_rf,pipeline_gb]


# In[31]:


for pipe in pipelines:
    pipe.fit(x_train, y_train)


# In[32]:


pipe_dict = {0:'LR',1:'knn',2:'dc',3:'svc',4:'rf',5:'gb'}


# In[33]:


pipe_dict


# In[34]:


for i,model in enumerate(pipelines):
    print('{} test accuracy:{}'.format(pipe_dict[i],model.score(x_test,y_test)*100))


# In[35]:


model=RandomForestClassifier(max_depth=5)


# In[36]:


new_data=pd.DataFrame({
    'Pregnancies':7,
    'Glucose':150.0,
    'BloodPressure':70.0,
    'SkinThickness':35.0,
    'Insulin':79.98,
       'BMI':34.0, 
    'DiabetesPedigreeFunction':0.643, 
    'Age':60,
},index=[0])


# In[37]:


model.fit(x_train,y_train)


# In[38]:


y_pred=model.predict(new_data)


# In[39]:


if y_pred[0] == 0:
    print('non-diabetic')
else:
    print('diabetic')
    


# **saving model using joblib**

# In[40]:


import joblib


# In[41]:


joblib.dump(model,'model_diabetics')


# In[42]:


model1=joblib.load('model_diabetics')


# In[43]:


model1.predict(new_data)

