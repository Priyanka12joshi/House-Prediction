#!/usr/bin/env python
# coding: utf-8

# librairies importing
# 

# In[2]:


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[3]:


data = pd.read_csv("housing.csv")


# In[4]:


data.dropna(inplace=True)


# In[5]:


data.head()


# In[7]:


data.describe()


# In[8]:


numerical_features = ['bedrooms','bathrooms','area']
target = 'price'


# In[48]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
 # categorical variable
categorical_columns = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']
#label encode these colmns
label_encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])
# seprerate the features and target avriable
numerical_fetaures = ['area','bathrooms','bedrooms','stories','parking']+categorical_columns
target = 'price'


# In[49]:


#split data
x= data[numerical_features]
y = data[target]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2,random_state = 42)
#standardise the numerical features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[50]:


#  model random forest 
model = RandomForestRegressor(n_estimators = 200,random_state = 42)
model.fit(x_train, y_train)


# In[51]:


model


# In[52]:


#evaluate the model 
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f"Mean Squared Error:{mse}")
print(f"R^2 Score : {r2}")


# # XGBoost and hyperparameter tunning

# In[5]:


pip install xgboost 


# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ModuleNotFoundError: No module named 'xgboost


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# In[8]:


data = pd.read_csv("housing.csv")


# In[9]:


data.head(),data.info()


# In[10]:


data.dropna(inplace = True)


# In[14]:


categorical_columns = data.select_dtypes(include = ['object']).columns
data = pd.get_dummies(data,columns = categorical_columns,drop_first = True)


# In[16]:


#fetaure and atrget
target = 'price'
numerical_features = [col for col in data.columns if col != target]
x = data[numerical_features]
y = data[target]


# In[17]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state = 42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[18]:


xgb_model = XGBRegressor(objective ='reg:squarederror',random_state = 42)


# In[21]:


param_grid = {
    'n_estimators': [100,200,300],
    'learning_rate':[0.01,0.1,0.2],
    'max_depth':[3,5,7],
    'subsample':[0.7,0.8,0.9]
}
grid_search = GridSearchCV(estimator = xgb_model,param_grid = param_grid,scoring = 'neg_mean_squared_error',cv = 3,verbose = 1)
grid_search.fit(x_train,y_train)


# In[25]:


best_xgb_model = grid_search.best_estimator_
print(f"Best Parameters: { grid_search.best_params_}")
y_pred = best_xgb_model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f"Mean square Error:{mse}")
print(f"R^2 Score:{r2}")
features_importances = pd.Series(best_xgb_model.feature_importances_,index = numerical_features).sort_values(ascending = False)
print("Feature Importances:")
print(features_importances)


# In[ ]:




