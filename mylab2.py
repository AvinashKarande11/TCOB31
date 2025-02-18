#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[36]:


df = pd.read_csv("Downloads/StudentsPerformance.csv")
df


# In[24]:


series = pd.isnull(df["Math_Score"])
series


# In[29]:


print(df.isnull())


# In[30]:


series = pd.isnull(df["Math_Score"])
print(df[series])
#Check for missing values in a specific column 


# In[31]:


df.notnull()


# In[38]:


df_filled = df.fillna(0)
print(df_filled)


# In[41]:


# Fill missing values in 'Math_Score' with the mean of the column
df['Math_Score'] = df['Math_Score'].fillna(df['Math_Score'].mean())

# Print the DataFrame to see the changes
print(df)


# In[42]:


df['Math_Score'] = df['Math_Score'].fillna(df['Math_Score'].median())
print(df)


# In[45]:


df['Math_Score'] = df['Math_Score'].fillna(df['Math_Score'].mode()[0])
print(df)


# In[47]:


df['Math_Score'] = df['Math_Score'].fillna(df['Math_Score'].min())
print(df)


# In[49]:


df['Math_Score'] = df['Math_Score'].fillna(df['Math_Score'].max())
print(df)


# In[51]:


df_cleaned = df.dropna()
print(df_cleaned)
 # Drops rows where any column has NaN


# In[13]:


df.dropna(axis=1)  # Drops columns where any value is NaN


# In[16]:


df.dropna(axis=0, how='any', inplace=True)  # Drops rows with any NaN value and updates the DataFrame
df


# In[52]:


import numpy as Reading_Scorenp


# In[55]:


sorted_rscore = sorted(df['Reading_Score'])


# In[56]:


q1 = np.percentile(sorted_rscore, 25)
q3 = np.percentile(sorted_rscore, 75)
print(q1, q3)


# In[3]:


import pandas as pd
import numpy as np


# In[6]:


df.boxplot(column=['Math_Score', 'Reading_Score', 'Writing_Score' , 'Placement_Score' ])


# In[18]:


print(np.where(df['Math_Score'] > 80))  # Adjusted threshold
print(np.where(df['Reading_Score'] < 70))  # Adjusted threshold
print(np.where(df['Writing_Score'] < 80))  # Adjusted threshold




# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[22]:


fig, ax = plt.subplots(figsize=(18,10))
ax.scatter(df['Placement_Score'], df['Placement_Offer_Count'])
plt.show()


# In[24]:


print(np.where((df['Placement_Score'] < 50) & (df['Placement_Offer_Count'] > 1)))
print(np.where((df['Placement_Score'] > 85) & (df['Placement_Offer_Count'] < 3)))


# In[33]:


#Z-scores are a way to measure how many standard deviations a data point is away from the mean of the dataset
from scipy import stats
z_scores = stats.zscore(df['Reading_Score'])


# In[35]:


outliers = np.where(z_scores > 1.5)  # Or use < 1.5 for lower-end outliers
print(outliers)


# In[36]:


outlier_indices = outliers[0]  # Get indices of outliers
outlier_values = df['Math_Score'].iloc[outlier_indices]  # Get values of the outliers
print(outlier_values)
Reading_Score


# In[37]:


import pandas as pd
import numpy as np


# In[43]:


import matplotlib.pyplot as plt

# Plot the histogram of the 'Math_Score' variable before removing outliers
df['Math_Score'].plot(kindReading_Score='hist', bins=20, edgecolor='black')
plt.title('Histogram of Math_Score before removing outliers')
plt.xlabel('Math Score')
plt.ylabel('Frequency')
plt.show()


# In[44]:


# Apply log transformation to 'Math_Score' column
df['log_math'] = np.log10(Reading_Scoredf['Math_Score'])

# Plot the histogram after log transformation
df['log_math'].plot(kind='hist', bins=20, edgecolor='black')
plt.title('Histogram of Math_Score after Log Transformation')
plt.xlabel('Log of Math Score')
plt.ylabel('Frequency')
plt.show()


# In[46]:


df.hist(figsize=(10,10),bins=20)
plt.show()


# In[47]:


# Step 1: Calculate the 1st Quartile (Q1) and 3rd Quartile (Q3)
Q1 = df['Math_Score'].quantile(0.25)
Q3 = df['Math_Score'].quantile(0.75)

# Step 2: Compute the IQR (Interquartile Range)
IQR = Q3 - Q1

# Step 3: Determine the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f'Q1 (25th percentile): {Q1}')
print(f'Q3 (75th percentile): {Q3}')
print(f'IQR: {IQR}')
print(f'Lower Bound for Outliers: {lower_bound}')
print(f'Upper Bound for Outliers: {upper_bound}')

# Step 4: Identify the outliers
outliers = df[(df['Math_Score'] < lower_bound) | (df['Math_Score'] > upper_bound)]
print("Outliers detected using IQR:")
print(outliers)

