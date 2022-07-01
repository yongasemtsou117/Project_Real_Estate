#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('C:/Users/blaise/OneDrive/Desktop/Data Science cours/Bengaluru_House_Data.xls')


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.shape


# In[6]:


df.groupby('area_type')['area_type'].agg('count')


# In[ ]:





# In[7]:


df=df.drop(['area_type','society','balcony','availability'],axis='columns')


# In[8]:


df.head()


# In[9]:


df.isnull().sum()


# In[10]:


df=df.dropna()
df.isnull().sum()


# In[11]:


df.isnull().sum()


# In[12]:


df['size'].unique()


# In[13]:


df['bhk']=df['size'].apply(lambda x: int(x.split(' ')[0]))


# In[14]:


df.head()


# In[15]:


df[df['bhk']>20]


# In[16]:


df.total_sqft.unique()


# In[17]:


#Find the total_sqft value represented in range 
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[18]:


df[~df['total_sqft'].apply(is_float)].head(25)


# In[19]:


#Handle this range value
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return(float(tokens[0])+ float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[20]:


convert_sqft_to_num('5.31Acres')


# In[21]:


df['total_sqft']=df['total_sqft'].apply(convert_sqft_to_num)


# In[22]:


df.iloc[30]


# In[23]:


df5=df.copy()


# In[24]:


#Created a new columns price_per_sqft
df5['price_per_sqft']=df5['price']*100000/df['total_sqft']


# In[25]:


df5.head()


# In[26]:


#Dimensional problem it's a high dimensional for Dummies values
len(df5.location.unique())


# In[27]:


df5.location=df5.location.apply(lambda x: x.strip())


# In[28]:


location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[29]:


len(location_stats[location_stats<=10])


# In[30]:


location_stats_less_than_10=location_stats[location_stats<=10]
location_stats_less_than_10


# In[ ]:





# In[31]:


len(df5.location.unique())


# In[32]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique()) 


# In[33]:


#Deal with Outliers by using standard deviation


# 

# In[34]:


df5.head()


# In[35]:


df5[df5['total_sqft']/df5['bath']<300].head()
#We notice there are 1020.0 total_sqft for just 6 Bedroom and 600.0 total_sqft for 8 Bedroom is unusual. 
#Means we need to remove this data because is an Outliers


# In[36]:


df5.shape


# In[37]:


#Use the filter to filter this outliers
df6=df5[~(df5['total_sqft']/df5['bath']<300)]
df6


# In[38]:


df6.shape


# In[39]:


#We move a lot of Outliers but maybe there some Outliers in data we have to use standard deviation technique  


# In[40]:


df6.price_per_sqft.describe()


# In[41]:


#Max=176470.588235 and Min=267.829813 are the extreme values we going to build a function to remove this extreme values
def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7=remove_pps_outliers(df6)
df7.shape


# In[42]:


#visualization
def plot_scatter_chart(df,location):
    bhk2=df[(df.location == location)&(df.bhk==2)]
    bhk3=df[(df.location==location)&(df.bhk==3)]
    plt.figure(figsize=(15,10))
    plt.scatter(bhk2.total_sqft, bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price,marker='+',color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Hebbal")
    


# In[43]:


len(df6[df6.location=='Hebbal'])


# In[44]:


#We should also remove proprieties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apratment(with same square ft area). What we will do is for a given location, we will build a dictionnary of stats per bhk, i.e.

{
    '1':{
        'mean':4000,
        'std':2000,
        'count':34
    },
    '2':{
        'mean':4300,
        'std':2300,
        'count':22
    },
}

#Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment


# In[45]:


def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats ={}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8=remove_bhk_outliers(df7)
df8.shape


# In[46]:


plot_scatter_chart(df8,"Hebbal")


# In[47]:


import matplotlib
plt.figure(figsize=(20,10))
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel('Price Per Square Feet')
plt.ylabel('Count')


# In[48]:


df8.bath.unique()


# In[49]:


df8[df8.bath>10]


# In[50]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel('Number of bathrooms')
plt.ylabel('Count')


# In[51]:


#Remove outlier "any apartment have bathroom great than 2 bedrooms"
df8[df8.bath>df8.bhk+2]


# In[52]:


#We remove that outliers
df9=df8[df8.bath<df8.bhk+2]
df9.shape


# In[53]:


#Start to create machine learning Model
df10=df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)


# In[54]:


#the only character variable "location " need to be transform in Dummies variable
dummies=pd.get_dummies(df10.location)


# In[55]:


#In teh rule of dummies variable you have to drop 1 column to avoid repetition
df11=pd.concat([df10,dummies.drop('other',axis='columns')], axis='columns')
df11.head()


# In[56]:


df12=df11.drop('location',axis='columns')


# In[57]:


df12


# In[58]:


df12.shape


# In[59]:


X=df12.drop('price',axis='columns')
X.head()


# In[60]:


y=df12.price
y.head()


# In[61]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=10)


# In[62]:


from sklearn.linear_model import LinearRegression
lr_clf=LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[63]:


#We want to find the best parameter for the model, in that we use K-fold Validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(),X,y, cv=cv)


# In[64]:


#Try other regression technique
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos ={
        'linear_regression':{
            'model':LinearRegression(),
            'params':{
                'normalize':[True,False]
            }
        },
        'lasso':{
            'model':Lasso(),
            'params':{
                'alpha':[1,2],
                'selection':['random','cyclic']
            }
        },
        'decision_tree':{
            'model':DecisionTreeRegressor(),
            'params':{
                'criterion': ['mse','friedman_mse'],
                'splitter':['best','random']
            }
        }
    }
    scores =[]
    cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs= GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score':gs.best_score_,
            'best_params':gs.best_params_
        })
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])
find_best_model_using_gridsearchcv(X,y)


# In[ ]:


#Make a prediction
def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]
    
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index >=0:
        x[loc_index]=1
        
    return lr_clf.predict([x])[0]


# In[ ]:


np.where(X.columns=='5th Block Hbr Layout')[0][0]


# In[ ]:


predict_price('1st Phase JP Nagar',1000,2,2)


# In[ ]:


predict_price('1st Phase JP Nagar',1000,2,4)


# In[ ]:


predict_price('Indira Nagar',1000,2,2)


# In[ ]:


#Exporting Model In the Website


# In[ ]:


import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# In[ ]:


import json
columns ={
    'data_columns':[col.lower()for col in X.columns]
}
with open('columns.json','w') as f:
    f.write(json.dumps(columns))


# In[ ]:


#Next step is to write python Interface

