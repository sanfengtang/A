#!/usr/bin/env python
# coding: utf-8

# In[342]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import warnings
import pymongo
import datetime
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # for displaying labels normally
plt.rcParams['axes.unicode_minus'] = False  # Used to display negative signs normally
pd.set_option('display.max_rows', 100,'display.max_columns', 100,"display.max_colwidth",1000,'display.width',1000)


# In[343]:


data = pd.read_excel(r'C:\Users\Administrator\Desktop\2023\7002\data v1\cars_info.xlsx')


# In[344]:


# Compute the proportion of missing values for each feature
missing_ratio = data.isnull().sum() / len(data)

# Sort by the proportion of missing values from high to low
missing_ratio_sorted = missing_ratio.sort_values(ascending=False)

# Take out the top 10 features with missing value ratio
missing_ratio_top10 = missing_ratio_sorted[:10]

# draw bar chart
plt.barh(range(len(missing_ratio_top10)), missing_ratio_top10)
plt.yticks(range(len(missing_ratio_top10)), missing_ratio_top10.index)
plt. xlabel("Missing Ratio")
plt.ylabel("Feature")
plt.title("Top 10 Features with Highest Missing Ratio")
plt. show()


# In[345]:


# Compute the proportion of missing values for each feature
missing_ratio = data.isnull().sum() / len(data)

# Sort by the proportion of missing values from high to low
missing_ratio_sorted = missing_ratio.sort_values(ascending=False)

# Take out the top 10 features with missing value ratio
missing_ratio_top10 = missing_ratio_sorted[:10]

# Create a DataFrame to show the missing value ratio for each feature
missing_df = pd.DataFrame({'Feature': missing_ratio_top10.index, 'Missing Ratio': missing_ratio_top10.values})

# Print the DataFrame
print(missing_df)


# In[346]:


# Compute number of records, number of features and distribution of variable types
num_records = data. shape[0]
num_features = data. shape[1]
type_distribution = data.dtypes.value_counts()

# Compute missing values
missing_values = data.isna().sum()

# Compute statistical summary information
summary = data. describe()

# output result
print("Number of records:", num_records)
print("Number of features:", num_features)
print("variable type distribution:\n", type_distribution)
print("Missing values:\n", missing_values)
print("Statistic summary information:\n", summary)


# In[347]:


for c in data.columns:
    if data[c].isna().sum() > 80000:
        data.drop([c], axis=1, inplace=True)


# In[348]:


for c in data.columns:
    if data[c].isin(["无"]).sum() > 60000:
        data.drop([c], axis=1, inplace=True)


# In[349]:


#Because the data itself contains separate columns of length, width and height, the "length * width * height (mm)" column is deleted
data.drop(['L*W*H(mm)'], axis=1, inplace=True)


# In[350]:


data.head()


# In[351]:


for c in data.columns:
    if data[c].isin(["标配"]).sum() > 60000:
        print(c, data[c].isin(["标配"]).sum())
        data.drop([c], axis=1, inplace=True)


# In[352]:


data.shape


# In[353]:


data.dropna(axis=0,subset = ["selling price", "Displacement (L)"], inplace=True)


# In[354]:


data.shape


# In[355]:


# This column contains a large number of range values, and there is a new car price, delete it
data.drop(["Manufacturer's new car guide price"], axis=1, inplace=True)


# In[356]:


for c in data.columns:
    print(c+":    "+str(data[c].dtypes))
    print(data[c].value_counts())


# In[357]:


data.info()


# In[358]:


data.describe()


# In[359]:


int(data['passenger/person'].mean())


# In[360]:


data['transfer record'].fillna(0, inplace=True)
data['passenger/person'].fillna(int(data['passenger/person'].mean()), inplace=True)


# In[361]:


data.describe()


# In[362]:


data.to_excel("precess1.xlsx", index=False)


# In[363]:


data = pd.read_excel("precess1.xlsx", na_values=np.nan)


# In[364]:


data.shape


# In[365]:


data.head()


# In[366]:


data.columns


# In[367]:


numerical_col = ['selling price', 'new car price', 'driven distance', 'Displacement (L)', 'Maximum speed(km/h)', 'Official 0-100km/h acceleration (s)', "Ministry of Industry and Information Technology's comprehensive fuel consumption (L/100km)", 'Length (mm)', 'Width (mm)', 'height (mm)', 'Wheelbase (mm)', 'Front track(mm)', 'Rear track(mm)', 'Number of doors', 'Fuel tank capacity (L)', 'curb weight(kg)', 'Minimum ground clearance (mm)', 'Displacement (mL)', 'Number of cylinders (pieces)', 'Number of valves per cylinder (pieces)', 'compression ratio', 'Maximum horsepower (Ps)', 'Maximum power (kW)', 'Maximum torque (N·m)', 'transfer record', 'passenger/person']


# In[368]:


numerical_df = data[numerical_col]


# In[369]:


numerical_df.head()


# # 非数值型数据替换为np.nan

# In[370]:


for c in numerical_col[5:]:
    numerical_df[c] = numerical_df[c].replace("无", np.nan).replace("false", np.nan).replace("未知", np.nan)


# In[371]:


numerical_df['Displacement (L)'] = numerical_df['Displacement (L)'].replace("无", np.nan).replace("未知", np.nan)
numerical_df['Maximum speed(km/h)'] = numerical_df['Maximum speed(km/h)'].replace("无", np.nan).replace("未知", np.nan)


# In[372]:


import numpy as np

numerical_df = numerical_df.replace('false', np.nan)


# In[373]:


numerical_df.describe()


# In[374]:


numerical_df.isna().sum()


# In[375]:


numerical_df.head()


# In[376]:


numerical_df.shape


# In[377]:


for c in numerical_col:
    print(c, set(numerical_df[c]))


# In[378]:


missing_ratio = data[['Ministry of Industry and Information Technology\'s comprehensive fuel consumption (L/100km)', 'Length (mm)', 'Width (mm)', 'height (mm)', 'Wheelbase (mm)', 'Front track(mm)', 'Rear track(mm)', 'Fuel tank capacity (L)', 'curb weight(kg)', 'Minimum ground clearance (mm)', 'Displacement (mL)', 'compression ratio', 'Maximum horsepower (Ps)', 'Maximum power (kW)', 'Maximum torque (N·m)']].isnull().sum() / len(data)
missing_ratio_sorted = missing_ratio.sort_values(ascending=False)
print(missing_ratio_sorted)


# In[379]:


# Select the columns of interest
columns_of_interest = ['Ministry of Industry and Information Technology\'s comprehensive fuel consumption (L/100km)',
                       'Length (mm)', 'Width (mm)', 'height (mm)', 'Wheelbase (mm)',
                       'Front track(mm)', 'Rear track(mm)', 'Fuel tank capacity (L)',
                       'curb weight(kg)', 'Minimum ground clearance (mm)', 'Displacement (mL)',
                       'compression ratio', 'Maximum horsepower (Ps)', 'Maximum power (kW)',
                       'Maximum torque (N·m)']

# Filter the missing_ratio by columns of interest
missing_ratio = missing_ratio[columns_of_interest]

# Sort by the proportion of missing values from high to low
missing_ratio_sorted = missing_ratio.sort_values(ascending=False)

# draw bar chart
plt.barh(range(len(missing_ratio_sorted)), missing_ratio_sorted)
plt.yticks(range(len(missing_ratio_sorted)), missing_ratio_sorted.index)
plt.xlabel("Missing Ratio")
plt.ylabel("Feature")
plt.title("Features with Missing Ratio")
plt.show()


# In[380]:


# Save the processed data to a CSV file
numerical_df.to_csv('processed_data.csv', index=False)


# # 空值填充

# In[384]:


# 统计每个特征缺失值的数量
null_counts = data[['Official 0-100km/h acceleration (s)',
                  "Ministry of Industry and Information Technology's comprehensive fuel consumption (L/100km)",
                  'Length (mm)',
                  'Width (mm)',
                  'height (mm)',
                  'Wheelbase (mm)',
                  'Front track(mm)',
                  'Rear track(mm)',
                  'Fuel tank capacity (L)',
                  'curb weight(kg)',
                  'Minimum ground clearance (mm)',
                  'Displacement (mL)',
                  'compression ratio',
                  'Maximum horsepower (Ps)',
                  'Maximum power (kW)',
                  'Maximum torque (N·m)']].isnull().sum()

# 绘制条形图
plt.bar(null_counts.index, null_counts.values)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Number of missing values')
plt.title('Missing Values by Feature')
plt.show()

# 绘制饼图
plt.pie(null_counts.values, labels=null_counts.index, autopct='%1.1f%%')
plt.title('Missing Values by Feature')
plt.show()


# In[385]:


# fill missing values with mean for some columns
mean_fill_col = ['Official 0-100km/h acceleration (s)',
                 "Ministry of Industry and Information Technology's comprehensive fuel consumption (L/100km)",
                 'Length (mm)',
                 'Width (mm)',
                 'height (mm)',
                 'Wheelbase (mm)',
                 'Front track(mm)',
                 'Rear track(mm)',
                 'Fuel tank capacity (L)',
                 'curb weight(kg)',
                 'Minimum ground clearance (mm)',
                 'Displacement (mL)',
                 'compression ratio',
                 'Maximum horsepower (Ps)',
                 'Maximum power (kW)',
                 'Maximum torque (N·m)'
                 ]

# fill missing values with mode for some columns
many_fill_col = ['Number of doors',
                 'Number of cylinders (pieces)',
                 'Number of valves per cylinder (pieces)'] # Most are 4


# In[386]:


numerical_df = numerical_df.astype(float)


# In[387]:


numerical_df.head()


# In[388]:


for c in mean_fill_col:
    numerical_df[c].fillna(numerical_df[c].mean(), inplace=True)
    
for c in many_fill_col:
    numerical_df[c].fillna(4, inplace=True)


# In[389]:


data[ numerical_col ].head()


# In[390]:


data[ numerical_col ] = numerical_df


# In[391]:


data[ numerical_col ].head()


# In[272]:


# 将处理后的数据保存到CSV文件中
numerical_df.to_csv('processed_data.csv', index=False)


# # 处理 ['座位数', '行李厢容积(L)', '最大功率转速(rpm)', '最大扭矩转速(rpm)']

# In[393]:


def pickNum(df, c):
    if '-' in df[c]:
        num_list = df[c].split('-')
        return num_list[0]
    elif '―' in df[c]:
        num_list = df[c].split('―')
        return num_list[0]
    elif '～' in df[c]:
        num_list = df[c].split('～')
        return num_list[0]
    elif '/' in df[c]:
        num_list = df[c].split('/')
        return num_list[0]
    else:
        return df[c]


# In[394]:


# def pickNum1(df):
# if '-' in df['number of seats']:
# num_list = df['number of seats'].split('-')
# return num_list[0]
# elif '/' in df['number of seats']:
# num_list = df['number of seats'].split('/')
# return num_list[0]
# else:
# return df['number of seats']
    
# def pickNum2(df):
# if '-' in df['luggage volume (L)']:
# num_list = df['luggage compartment volume (L)'].split('-')
# return num_list[0]
# elif '/' in df['luggage compartment volume (L)']:
# num_list = df['luggage compartment volume (L)'].split('/')
# return num_list[0]
# else:
# return df['luggage compartment volume (L)']
    
# def pickNum3(df):
# if '-' in df['maximum power speed (rpm)']:
# num_list = df['maximum power speed (rpm)'].split('-')
# return num_list[0]
# elif '/' in df['maximum power speed (rpm)']:
# num_list = df['maximum power speed (rpm)'].split('/')
# return num_list[0]
# else:
# return df['maximum power speed (rpm)']
    
# def pickNum4(df):
# if '-' in df['Maximum torque speed (rpm)']:
# num_list = df['maximum torque speed (rpm)'].split('-')
# return num_list[0]
# elif '/' in df['Maximum torque speed (rpm)']:
# num_list = df['maximum torque speed (rpm)'].split('/')
# return num_list[0]
# else:
# return df['maximum torque speed (rpm)']


# In[400]:


pickNum_col = ['number of seats',
               'Luggage compartment volume (L)',
               'Maximum power speed(rpm)',
               'Maximum torque speed(rpm)'
              ]


# In[401]:


data[pickNum_col] = data[pickNum_col].astype(str)


# In[402]:


for c in pickNum_col:
    data[c] = data.apply(lambda x:pickNum(x, c), axis=1)


# In[403]:


# data['number of seats'] = data.apply(lambda x:pickNum1(x), axis=1)
# data['luggage compartment volume (L)'] = data.apply(lambda x:pickNum2(x), axis=1)
# data['maximum power speed (rpm)'] = data.apply(lambda x:pickNum3(x), axis=1)
# data['maximum torque speed (rpm)'] = data.apply(lambda x:pickNum4(x), axis=1)


# In[404]:


data[pickNum_col].head()


# In[405]:


for c in pickNum_col:
    data[c] = data[c].replace("无", np.nan).replace("false", np.nan).replace("未知", np.nan)


# In[406]:


data['number of seats'].value_counts()


# In[407]:


data[pickNum_col] = data[pickNum_col].astype(float)


# In[408]:


data['number of seats'].fillna(5, inplace=True)


# In[409]:


for c in pickNum_col[1:]:
    data[c].fillna(data[c].mean(), inplace=True)


# In[410]:


len(data.dtypes[data.dtypes == float])


# # 处理日期型数据

# In[414]:


date_col = ['Business insurance expiration date', 'registration date', 'Date of manufacture', 'Vehicle Tax Expiry Date']


# In[415]:


data[date_col].head()


# In[422]:


data['data acquisition date'] = '2022-11-25'


# In[423]:


date_col.append('data acquisition date')


# In[424]:


data[date_col].head()


# In[429]:


def calDate(df, c):
     if pd.isnull(df['factory date']):
         return np.nan
     else:
         d1=datetime.datetime.strptime('2022-11-25',"%Y-%m-%d")
         d2=datetime.datetime.strptime(df[c],"%Y-%m-%d")
         diff_days=d1-d2
# print(diff_days)
         return diff_days.days


# In[430]:


for c in date_col[:-1]:
    data[c] = data[c].replace("--", np.nan)
data2 = data


# In[431]:


for c in date_col[:-1]:
     data[c+'difference (day)'] = data.apply(lambda x:calDate(x, c), axis=1)


# In[ ]:


data.head()


# In[ ]:


new_date_col = ['Commercial insurance expiration date difference (days)', 'Traffic compulsory insurance expiration date difference (days)', 'Registration date difference (days)', 'Production date difference (days)', 'Vehicle tax expiration date difference (days) )']


# In[ ]:


for c in new_date_col:
    data[c].fillna(data[c].mean(), inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.drop(date_col, axis=1, inplace=True)


# In[ ]:


data.describe()


# In[ ]:


numerical_col_new = list(data.describe().columns)


# In[ ]:


non_numerical_col = []
for c in data.columns:
    if c not in numerical_col_new:
        non_numerical_col.append(c)
len(non_numerical_col)


# In[ ]:


non_numerical_col


# In[ ]:


len(numerical_col_new)


# In[ ]:


for c in non_numerical_col:
#     print(c+":    "+str(data[c].dtypes))
    print(data[c].value_counts())


# # 处理0-1型数据

# In[ ]:




