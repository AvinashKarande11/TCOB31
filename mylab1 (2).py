#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd 
import numpy as np
df={'RN':[1,2,3,4,5],'SName':['a','b','c','d','e']}


# In[11]:


d1=pd.DataFrame(df)


# In[26]:


d2=pd.DataFrame({'RN':[1,2,3,4,5],'Marks':[11,22,32,42,54]})


# In[27]:


d3=pd.merge(d1,d2, on='RN')
d3


# In[33]:


d4=pd.DataFrame({'RN':[1,2,3,4,5],'SName':['a','b','c','d','e'],'Marks':[11,22,32,42,54]})
d4
                 


# In[39]:


d5=d4.pivot(index='RN',columns='SName',values='Marks')
d5


# In[52]:


df1=pd.DataFrame({'RN':[1,2,3,4,np.NaN],'SName':['a','b','c',np.NaN,'e'],'Marks':[11,22,32,np.NaN,54]})
df1


# In[55]:


d6=df1.bfill(axis=0)
d6


# In[57]:


df2=pd.read_csv("/home/ubuntu/Downloads/Demo Data DSBDA.csv")
df2


# In[60]:


d=df2.groupby('GENDER')
d.first()


# In[61]:


d.count()


# In[62]:


print(d.get_group('M'))


# In[64]:


df['']-corr(df['DELD'])


# In[66]:


dd=df2.describe()
dd


# In[68]:


df2 = df2.drop_duplicates()
df2


# In[75]:


# Group by GENDER
grouped = df2.groupby('GENDER')

# Get all rows for females
females = grouped.get_group('F')

# Get all rows for males
males = grouped.get_group('M')
print("Females:")
print(females)
print("males:")
print(males)


# In[6]:


d = df.filter(items=['DM'])


# In[7]:


d


# In[12]:



df = pd.read_excel('/home/ubuntu/Downloads/Demo Data DSBDA.xlsx')


# In[14]:


pip install openpyxl




# In[15]:


import pandas as pd

# Load the Excel file into a DataFrame
df = pd.read_excel('/home/ubuntu/Downloads/Demo Data DSBDA.xlsx')

# Display the first few rows of the DataFrame
print(df.head())


# In[16]:


conda to install openpyxl


# In[18]:


"%pip install"


# In[19]:


import pandas as pd

# Load the Excel file into a DataFrame
df = pd.read_excel('/home/ubuntu/Downloads/Demo Data DSBDA.xlsx')

# Display the first few rows of the DataFrame
print(df.head())


# In[20]:


d = df.filter(items=['DM'])


# In[21]:


d


# In[22]:


get_ipython().run_line_magic('pip', 'install openpyxl')


# In[23]:


import pandas as pd

# Load the Excel file into a DataFrame
df = pd.read_excel('/home/ubuntu/Downloads/Demo Data DSBDA.xlsx')

# Display the first few rows of the DataFrame
print(df.head())


# In[24]:


e = df.filter(items=['DM'])


# In[25]:


e


# In[32]:


r['DM'] = pd.to_numeric(r['DM'], errors='coerce')


# In[35]:


r = r.dropna(subset=['DM'])
r


# In[36]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 7))
plt.boxplot(r['DM'])
plt.show()


# In[40]:


plt.hist(r['DM'],bins=5)


# In[41]:


import pandas as pd

# Define column names
col_names = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Species']

# Load the dataset from the UCI repository with column names
csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = pd.read_csv(csv_url, names=col_names)

# Display the first few rows of the dataset
print(iris.head())


# In[42]:


# Check the data types of the columns
print(iris.dtypes)


# In[43]:


# First 5 rows of the dataset
print(iris.head())


# In[44]:


# Last 5 rows of the dataset
print(iris.tail())


# In[45]:


# Get the index of the dataset
print(iris.index)


# In[46]:


# Get the column labels of the dataset
print(iris.columns)


# In[47]:


# Get the shape (number of rows and columns) of the dataset
print(iris.shape)


# In[48]:


# Get the data types of the columns
print(iris.dtypes)


# In[49]:


# Get the column names as an array
print(iris.columns.values)


# In[50]:


# Get descriptive statistics of the dataset
print(iris.describe(include='all'))


# In[51]:


# Access the 'Sepal_Length' column
print(iris['Sepal_Length'])


# In[88]:


# Sort columns in descending order
print(iris.sort_index(axis=1, ascending=False))


# In[89]:


# Sort the dataset by 'Sepal_Length'
print(iris.sort_values(by="Sepal_Length"))


# In[54]:


# Select the 5th row
print(iris.iloc[5])


# In[55]:


# Select the first 3 rows
print(iris[0:3])


# In[56]:


# Select 'Sepal_Length' and 'Sepal_Width' columns
print(iris.loc[:, ["Sepal_Length", "Sepal_Width"]])


# In[57]:


# Select the first 3 rows of all columns
print(iris.iloc[:3, :])


# In[58]:


# Select the first 3 columns of all rows
print(iris.iloc[:, :3])


# In[59]:


# Select the first 3 rows and the first 2 columns
print(iris.iloc[:3, :2])


# In[83]:


iris.isnull()


# In[80]:


# Count missing values across each column
print(iris.isnull().sum())

# Count total missing values in the entire DataFrame
print(iris.isnull().sum().sum())


# In[81]:


iris.isnull().any()


# In[82]:


iris.isnull().sum(axis=1)


# In[ ]:




