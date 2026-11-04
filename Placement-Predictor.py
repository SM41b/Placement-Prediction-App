#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries for data processing, visualization, model training, evaluation, and saving
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


# Loading the original placement dataset and keeping a copy for PostgreSQL
df = pd.read_csv('placement.csv')
raw_data = df.copy()


# In[3]:


# Displaying the first five rows of the placement dataset.
df.head()


# In[4]:


# Displaying the number of rows and columns in the dataset.
df.shape


# In[5]:


# Displaying the dataset's structure, column names, data types, and non-null value counts.
df.info()


# In[6]:


# Checking for missing values in each column
df.isnull().sum()


# In[7]:


# Removing the unnecessary identifier column from the dataset
df = df.iloc[:, 1:]


# In[8]:


# Displaying the first five rows of the updated dataset.
df.head()


# In[9]:


# Checking the distribution of placed and not-placed students
print(df['placement'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='placement')
plt.title('Placement Distribution')
plt.show()


# In[10]:


# Importing Matplotlib's Pyplot module for data visualization.
import matplotlib.pyplot as plt


# In[11]:


# Visualizing the relationship between CGPA and IQ, with colors representing placement status.
plt.scatter(df['cgpa'], df['iq'], c=df['placement'])


# In[12]:


# Visualizing the distributions of CGPA and IQ
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.histplot(df['cgpa'], kde=True, ax=axes[0])
axes[0].set_title('CGPA Distribution')

sns.histplot(df['iq'], kde=True, ax=axes[1])
axes[1].set_title('IQ Distribution')

plt.tight_layout()
plt.show()


# In[13]:


# Separating independent features and dependent target
X = df[['cgpa', 'iq']]
y = df['placement']


# In[14]:


X


# In[15]:


y


# In[16]:


# Checking the data type of the target variable.
type(y)


# In[17]:


# Checking the data type of the independent features variable.
type(X)


# In[18]:


# Displaying the dimensions of the target variable.
y.shape


# In[19]:


# Displaying the dimensions of the independent features.
X.shape


# In[20]:


# Importing train_test_split to divide the dataset into training and testing sets.
from sklearn.model_selection import train_test_split


# In[21]:


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[22]:


X_train


# In[23]:


y_train


# In[24]:


X_test


# In[25]:


y_test


# In[26]:


# Creating a pipeline for feature scaling and logistic regression
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])


# In[27]:


# Training the logistic regression pipeline
clf.fit(X_train, y_train)


# In[28]:


# Predict student placement outcomes using the trained Logistic Regression model.
y_pred = clf.predict(X_test)


# In[29]:


y_test


# In[30]:


# Import accuracy_score to evaluate the model's prediction accuracy.
from sklearn.metrics import accuracy_score


# In[31]:


# Evaluating the model's accuracy on testing data
print("Testing Accuracy:", accuracy_score(y_test, y_pred))


# In[32]:


# Evaluating precision, recall, and F1-score for each placement class
print(classification_report(y_test, y_pred))


# In[33]:


# Displaying correct and incorrect placement predictions
print(confusion_matrix(y_test, y_pred))


# In[34]:


# In[35]:


# Import the function used to visualize the classifier's decision boundaries.
from mlxtend.plotting import plot_decision_regions


# In[36]:


# Visualizing the decision boundaries of the logistic regression pipeline
plot_decision_regions(X_train.to_numpy(), y_train.to_numpy(), clf=clf, legend=2)
plt.xlabel('CGPA')
plt.ylabel('IQ')
plt.title('Student Placement Decision Boundary')
plt.show()


# In[37]:


# Saving the complete machine learning pipeline for future predictions
joblib.dump(clf, 'placement_model.pkl')


# In[38]:


# Predicting placement status for a new student using CGPA and IQ
input_data = pd.DataFrame([[7.5, 115]], columns=['cgpa', 'iq'])
prediction = clf.predict(input_data)

if prediction[0] == 1:
    print("The student is likely to be placed.")
else:
    print("The student is likely not to be placed.")


