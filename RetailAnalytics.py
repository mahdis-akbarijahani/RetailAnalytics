#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import calendar
from sklearn.compose import make_column_selector
import warnings
#warnings.filterwarnings('ignore')


# In[60]:


df = pd.read_csv('file.csv')


# In[61]:


df.head()


# In[62]:


# Assuming df is your DataFrame
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# If you want to modify the DataFrame in-place, you can use:
# df.drop('Unnamed: 0', axis=1, inplace=True)


# In[42]:


df.head()


# In[63]:


# Assuming df is your DataFrame
df_sorted_by_index = df.sort_index()

# If you want to modify the original DataFrame in-place, you can use:
# df.sort_index(inplace=True)
df.head()


# In[64]:


print(df.shape)


# In[65]:


df['Avg_Price']


# In[66]:


df.columns


# In[67]:


# Describe the DataFrame

df.describe()


# In[68]:


# Check the data types
data_types = df.dtypes

# Display the data types
print(data_types)


# In[69]:


# Count the missing values column-wise
missing_values_count = df.isnull().sum()

# Display the count of missing values for each column
print(missing_values_count)


# In[70]:


# Dropped nan values as the number (11) is insignificant compared to sample size(7046)
df.dropna(inplace=True)
df.isna().sum()


# In[71]:


df = pd.DataFrame(df)

duplicate_rows = df[df.duplicated()]

# Display the duplicate rows
print("Duplicate Rows:")
print(duplicate_rows)


# In[72]:


df.nunique()


# In[73]:


df['Total Prices']=df.Avg_Price+df.Delivery_Charges
df['Total_Spend']=df['Offline_Spend']+df['Online_Spend']
new=df[['Offline_Spend','Online_Spend','Month','Total_Spend']].groupby('Month').sum()

mon=list(calendar.month_name)[1:]


# In[74]:


df['Location'].unique()


# In[75]:


sb.histplot(df['Location'], color='blue')
plt.ylabel('Frequency')
plt.xlabel('Cities')
plt.xticks(rotation=65)
plt.title('Location Frequencies')

# Show the plot
plt.show()
plt.savefig("Tenure_Months.png")


# In[76]:


plt.figure(figsize=(25,15))
sb.boxplot(x='Product_Category',y='Avg_Price',data=df)
plt.title("product purchased according to genderwise")
plt.xlabel("Product_Category")
plt.ylabel("Avg_Price")
plt.show()
plt.savefig("Tenure_Months.png")


# In[77]:


import plotly.express as px

# Assuming df is your DataFrame
avg_price_by_category = df.groupby('Product_Category')['Avg_Price'].mean().reset_index()

# Create a line plot with Plotly Express
fig = px.line(avg_price_by_category, x='Product_Category', y='Avg_Price',
              title='Monthly Average Price Shopping',
              labels={'Product_Category': 'Product Category', 'Avg_Price': 'Average Price'},
              line_shape='linear')

fig.update_layout(xaxis_tickangle=-45)
fig.show()
plt.savefig("Tenure_Months.png")


# In[78]:


import seaborn as sb
import matplotlib.pyplot as plt

# Assuming df is your DataFrame and 'Location', 'Avg_Price', and 'Month' are column names
sb.lineplot(x='Location', y='Avg_Price', data=df, palette='viridis')
plt.xticks(rotation=60)
plt.title("Average Price Over Time by Location")
plt.show()
plt.savefig("Tenure_Months.png")


# In[79]:


ax = (df['Gender'].value_counts()*100.0 /len(df))\
.plot.pie(autopct='%.1f%%', labels = ['F', 'M'],figsize =(5,5), fontsize = 12 )
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('Gender',fontsize = 12)
ax.set_title('% of Gender', fontsize = 12)
plt.savefig("Gender.png") #code for saving pic


# In[80]:


# Assuming your dataset has 'TransactionDate' and 'AveragePrice' columns
# Make sure the 'TransactionDate' column is in datetime format
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], format='%Y-%m-%d')

# Sort the DataFrame by 'TransactionDate'
df = df.sort_values(by='Transaction_Date')

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(df['Transaction_Date'], df['Avg_Price'], color='blue')
plt.xlabel('Transaction_Date')
plt.ylabel('Avg_Price')
plt.title('Average Price Over Time')
plt.xticks(rotation=45)
plt.tight_layout()


plt.show()
plt.savefig("Tenure_Months.png")


# In[ ]:





# In[21]:


male=df[['Total Prices','Gender']].groupby('Gender').get_group('M')['Total Prices'].sum()
female=df[['Total Prices','Gender']].groupby('Gender').get_group('F')['Total Prices'].sum()
sizes=[round(male),round(female)]
labels=['Males','Females']
plt.pie(sizes,autopct='%1.1f%%',labels=[f'{label}\nPrice :{size}' for label, size in zip(labels, sizes)],shadow=True,explode=(0.1,0),colors=['blue','orange'])
plt.title('Price Spend')
plt.savefig("Tenure_Months.png")


# In[22]:


sb.lineplot(y=new.Total_Spend,x=new.index,color='blue')
plt.xticks(new.index,mon,rotation=60)
plt.title("Total Spend With Month")


# In[23]:


delivery = df[['Avg_Price', 'Month']].groupby('Month').sum()

plt.figure(figsize=(10, 6))
plt.bar(delivery.index, delivery['Avg_Price'], color='blue')
plt.xticks(range(12), mon, rotation=60)
plt.title("Avg price per Month")
plt.xlabel('Month')
plt.ylabel('Avg price')

plt.show()


# In[24]:


sns.set(rc={'figure.figsize':(14,10)})
sns.distplot(df['Tenure_Months'], kde = False, color ='blue', bins = 72, hist=True)
plt.savefig("Tenure_Months.png") #code for saving pic


# In[25]:


df['Tenure_Months'].nunique()


# In[26]:


fig=plt.figure()
axis=fig.add_axes([1,1,1,1])
nd1=df.Coupon_Status
sb.countplot(data=nd1,x=nd1,ax=axis)

for i in axis.patches:
    axis.annotate(i.get_height(),(i.get_x(),i.get_height()),va='bottom',ha='left')
plt.ylabel('Frequency')
plt.title('Coupon_status')
plt.savefig("Tenure_Months.png")


# In[37]:


# Create a pipeline for regression with one-hot encoding, normalization, and regression
# Splitting train and test data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor


# Splitting data into features (X) and target variable (y)
X = df.drop('Avg_Price', axis=1)
y = df['Avg_Price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

# Creating a pipeline for preprocessing and regression
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), ['Month', 'Total Prices', 'Tenure_Months']),
        ('cat', OneHotEncoder(drop='first'), ['Gender', 'Coupon_Status'])
    ])
#OneHotEncoder
#drop='first'
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor())])

# Fitting the pipeline on the training data
pipeline.fit(X_train, y_train)

# Evaluating the pipeline on the testing data
score = pipeline.score(X_test, y_test)
print(f"R-squared score on the testing data: {score:.2f}")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)



# In[30]:


X_train


# In[31]:


y_train


# In[33]:


numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()


# In[34]:


correlation_matrix


# In[35]:


correlation_matrix["Avg_Price"]


# In[40]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you already have the correlation matrix
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

# Selecting only the correlation values related to 'Avg_Price'
correlation_with_avg_price = correlation_matrix["Avg_Price"]

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Draw the heatmap using seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

# Show the plot
plt.title('Correlation Matrix')
plt.show()

# Display correlation values related to 'Avg_Price'
print("Correlation with Avg_Price:\n", correlation_with_avg_price)

