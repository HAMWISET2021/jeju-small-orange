#!/usr/bin/env python
# coding: utf-8

# WISET 감귤팀 / 김효정 유새하 이관구 이상협 홍형근
# 
# 기상청 날씨데이터를 이용한 제주노지감귤 비상품 출하량 예측

# # 사용할 패키지 import

# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# In[3]:


import numpy as np
import pandas as pd


# # 비상품량 업로드

# In[ ]:





# In[1]:


from google.colab import files
uploaded = files.upload()


# In[22]:


#2020~1994 가공품량
y = pd.read_excel('./비상품량.xlsx').iloc[:,-2]


# 2020~1994 생산량
#y=[515778, 491149, 467600, 440254, 466817, 519243, 573442, 554007,558942, 500106, 480565, 655046, 520350, 677770, 568920, 600511, 536668, 597373, 739266, 600140, 518731, 593188, 510644, 659121, 416557, 508445, 548945]
y


# #날씨 엑셀파일 업로드 및 전처리

# In[4]:


for year in range(1994,2022):
  str = "weather_row{0} = pd.read_excel('./weather_raw/{0}weather.xlsx')".format(year)
  exec(str)
'''

weather_row2010 = pd.read_excel('2010weather.xlsx')
weather_row2011 = pd.read_excel('2011weather.xlsx')
weather_row2012 = pd.read_excel('2012weather.xlsx')
weather_row2013 = pd.read_excel('2013weather.xlsx')
weather_row2014 = pd.read_excel('2014weather.xlsx')
weather_row2015 = pd.read_excel('2015weather.xlsx')
weather_row2016 = pd.read_excel('2016weather.xlsx')
weather_row2017 = pd.read_excel('2017weather.xlsx')
weather_row2018 = pd.read_excel('2018weather.xlsx')
weather_row2019 = pd.read_excel('2019weather.xlsx')
weather_row2020 = pd.read_excel('2020weather.xlsx')
weather_row2021 = pd.read_excel('2021weather.xlsx')
'''
weather_row2010.head(5)


# In[5]:


#필요없는 행, 열 제거
#서귀포 지점에 해당하는 데이터만, 변수값은 임의로 기온, 강수량, 풍속, 습도, 현지기압, 일조, 지면온도 으로 설정하여 추출함.
for year in range(1994,2022):
  str1 = "weather_row{0}_extract = weather_row{1}[weather_row{0}['지점']==189]".format(year,year)
  str2 = "weather_row{0}_extract = weather_row{1}_extract.iloc[:,[2,3,5,7,11,15,19,31]]".format(year,year)
  exec(str1)
  exec(str2)
  



# In[ ]:


weather_row1994_extract


# In[6]:


# 5,6,7,8,9월 데이터만 가져옴.
for year in range(1994,2022):
  str_temp = "weather_MayToSep_{0}_temp = weather_row{0}_extract['{0}-04-30 23:00:00'<weather_row{0}_extract['일시']]".format(year)
  str = "weather_MayToSep_{0} = weather_MayToSep_{0}_temp[weather_row{0}_extract['일시']<'{0}-10-01 01:00:00']".format(year)
  exec(str_temp)
  exec(str)
  


# In[7]:


weather_MayToSep_1996


# In[8]:


#nan 을 0으로 바꾸기

for year in range(1994,2022):
  


  
  str = "weather_MayToSep_{0}.loc[np.isnan(weather_MayToSep_{0}['강수량(mm)']), '강수량(mm)'] = 0".format(year)
  exec(str)
  str = "weather_MayToSep_{0}.loc[np.isnan(weather_MayToSep_{0}['일조(hr)']), '일조(hr)'] = 0".format(year)
  exec(str)
  str = "weather_MayToSep_{0}.loc[np.isnan(weather_MayToSep_{0}['기온(°C)']), '기온(°C)'] = 0".format(year)
  exec(str)
  str = "weather_MayToSep_{0}.loc[np.isnan(weather_MayToSep_{0}['풍속(m/s)']), '풍속(m/s)'] = 0".format(year)
  exec(str)
  str = "weather_MayToSep_{0}.loc[np.isnan(weather_MayToSep_{0}['습도(%)']), '습도(%)'] = 0".format(year)
  exec(str)
  str = "weather_MayToSep_{0}.loc[np.isnan(weather_MayToSep_{0}['현지기압(hPa)']), '현지기압(hPa)'] = 0".format(year)
  exec(str)
  str = "weather_MayToSep_{0}.loc[np.isnan(weather_MayToSep_{0}['지면온도(°C)']), '지면온도(°C)'] = 0".format(year)
  exec(str)
  


# In[9]:



for year in range(1994,2022):
  str = "weather_MayToSep_{0} = weather_MayToSep_{0}.reset_index(drop=True)".format(year)
  exec(str)

  str = "weather_MayToSep_{0} = weather_MayToSep_{0}.set_index('일시')".format(year)
  exec(str)


# In[10]:


weather_MayToSep_2001


# In[11]:


from datetime import datetime

for year in range(1994,2022):
  str = "weatherAverage{0} = weather_MayToSep_{0}.resample(rule='M').mean()".format(year)
  exec(str)
  str = "weatherAverage{0} = weatherAverage{0}.iloc[:-1,:]".format(year)
  exec(str)


# In[60]:


weatherAverage2001


# In[13]:


weatherAverage2010.shape


# In[14]:


#5개월치 날씨 DataFrame 안의 모든 데이터를 1차원 리스트로 변환.


for year in range(1994,2022):
  #빈 리스트 형성
  makelist_str = 'x_list_{0} = list()'.format(year)
  exec(makelist_str)
  for j in range(0,weatherAverage2010.shape[1]):
    str = 'x_list_{0}.extend(list(weatherAverage{0}.iloc[:,j]))'.format(year)
    exec(str)
  



# In[15]:


x_list_2001


# In[ ]:





# In[16]:


#연도별 1차원 리스트들을 합하여 2차원 리스트로 변환.
x_list = list()
for year in range(2020,1993,-1):
  str = 'x_list.append(x_list_{0})'.format(year)
  exec(str)
len(x_list)


# In[43]:


x_array = np.array(x_list)


# In[45]:


#길이체크
len(x_array[0])


# # 모델 실행 및 결과 확인

# In[ ]:


y_test


# In[58]:


class SingleLayer :
  def __init__(self, learning_rate=0.1, l1=0, l2=0) :
    self.w = None
    self.b = None
    self.losses = []
    self.val_losses = []
    self.lr = learning_rate
    self.l1 = l1
    self.l2 = l2
    

  def fit(self, x, y, epochs=100, x_val=None, y_val=None) :
    self.w = np.ones(x.shape[1])
    self.b = 0
    for i in range(epochs) :
      loss = 0
      indexes = np.random.permutation(np.arange(len(x)))
      for i in indexes :
        z = self.forpass(x[i])
        a = self.activation(z)
        err = -(y[i] - a)
        w_grad, b_grad = self.backdrop(x[i], err)
        w_grad += self.l1 * np.sign(self.w) + self.l2*self.w
        self.w -= self.lr * w_grad
        self.b -= b_grad
        a = np.clip(a, 1e-10, 1-1e-10)
        loss += -(y[i]*np.log(a) + (1-y[i])*np.log(1-a))
      self.losses.append(loss/len(y) + self.get_loss())
      self.update_val_loss(x_val, y_val)

  def get_loss(self) :
    return self.l1 * np.sum(np.abs(self.w)) + self.l2/2 * np.sum(self.w**2)
  def forpass (self, x):
    y_hat = x*self.w + self.b
    return y_hat
  def backdrop(self,x,err) :
    m = len(x)
    w2_grad = np.dot(self.a1.T,err) / m
    b2_grad = np.sum(err) / m
    err_to_hidden = np.dot(err,self.w2.T) * self.a1 * (1-self.a1)
    w1_grad = np.dot(x.T, err_to_hidden) / m
    b1_grad = np.sum(err_to_hidden) / m
    return w1_grad, b1_grad, w2_grad, b2_grad
  
  def activation(self, z) :
    f = 1 / (1+np.exp(-z))
    return f
  def update_val_loss(self, x_val, y_val) :
    if x_val is None :
      return
    val_loss = 0
    for i in range(len(x_val)) :
      z = self.forpass(x_val[i])
      a = self.activation(z)
      a = np.clip(a, 1e-10, 1-1e-10)
      val_loss += -(y_val[i]*np.log(a)+(1-y_val[i])*np.log(1-a))
    self.val_losses.append(val_loss/len(y_val) + self.get_loss())


# In[61]:


get_ipython().run_line_magic('pinfo', 'LinearRegression')


# In[62]:


#선형회귀
X_train, X_test, y_train, y_test = train_test_split(x_array, y, test_size = 0.03)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg = LinearRegression()
reg.fit(X_train_scaled,y_train)

y_train_hat = reg.predict(X_train_scaled)
y_test_hat = reg.predict(X_test_scaled)
print('RMSE')
print('train: ',mean_squared_error(y_train,y_train_hat)*0.5,'test: ',mean_squared_error(y_test,y_test_hat)*0.5)
print('R_Squared')
print('train: ',r2_score(y_train,y_train_hat)*0.5,'test: ',r2_score(y_test,y_test_hat)*0.5)


# In[59]:


import matplotlib.pyplot as plt
l1_list = [0.0001, 0.001, 0.01]

for l1 in l1_list :
  lyr = SingleLayer(l1=l1)
  lyr.fit(X_train, y_train, x_val = X_test, y_val = y_test)

  plt.ylim(0,0.3)
  plt.plot(lyr.losses)
  plt.plot(lyr.val_losses)
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train_loss', 'val_loss'])
  plt.show()

  plt.ylim(-4,4)
  plt.plot(lyr.w, 'bo')
  plt.title('Weight(l1={})'.format(l1))
  plt.ylabel('value')
  plt.xlabel('weight')
  plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


#릿지
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.001)
ridge.fit(X_train_scaled,y_train)

y_train_hat = ridge.predict(X_train_scaled)
y_test_hat = ridge.predict(X_test_scaled)
print('RMSE')
print('train: ',mean_squared_error(y_train,y_train_hat)*0.5,'test: ',mean_squared_error(y_test,y_test_hat)*0.5)
print('R_Squared')
print('train: ',r2_score(y_train,y_train_hat)*0.5,'test: ',r2_score(y_test,y_test_hat)*0.5)


# In[ ]:


#라쏘
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1)
lasso.fit(X_train_scaled,y_train)

y_train_hat = lasso.predict(X_train_scaled)
y_test_hat = lasso.predict(X_test_scaled)
print('RMSE')
print('train: ',mean_squared_error(y_train,y_train_hat)*0.5,'test: ',mean_squared_error(y_test,y_test_hat)*0.5)
print('R_Squared')
print('train: ',r2_score(y_train,y_train_hat)*0.5,'test: ',r2_score(y_test,y_test_hat)*0.5)


# In[ ]:


#뉴럴 네트워크
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor()
mlp.fit(X_train_scaled,y_train)

y_train_hat = mlp.predict(X_train_scaled)
y_test_hat = mlp.predict(X_test_scaled)
print('RMSE')
print('train: ',mean_squared_error(y_train,y_train_hat)*0.5,'test: ',mean_squared_error(y_test,y_test_hat)*0.5)
print('R_Squared')
print('train: ',r2_score(y_train,y_train_hat)*0.5,'test: ',r2_score(y_test,y_test_hat)*0.5)


# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


# In[ ]:


pipe = Pipeline([('preprocessing',None),('regressor',LinearRegression())])
hyperparam_grid = [
                   {'regressor': [LinearRegression(),Ridge(),Lasso()],'preprocessing':[StandardScaler(),MinMaxScaler(),None]},
                   {'regressor': [KNeighborsRegressor()],'preprocessing':[StandardScaler(),MinMaxScaler(),None],
                    'regressor__n_neighbors':[1,2,3,4,5],'regressor__weights':['uniform','distance'],'regressor__p':[1,2,3]},
                   {'regressor': [RandomForestRegressor()],'preprocessing':[StandardScaler(),MinMaxScaler(),None],
                    'regressor__max_features':['auto','sqrt','log2']},
                   {'regressor': [SVR()],'preprocessing':[StandardScaler(),MinMaxScaler(),None],
                    'regressor__epsilon':[0,0.1,0.01,0.001],'regressor__gamma':[0.001,0.01,0.1],'regressor__C':[1,10,100]},
                   {'regressor': [MLPRegressor()],'preprocessing':[StandardScaler(),MinMaxScaler(),None]}
]
kfold = KFold(n_splits=27)
grid = GridSearchCV(pipe, hyperparam_grid,scoring='r2',cv=kfold)
grid.fit(x_list,y)


# In[ ]:


print(grid.best_params_)
print(grid.best_estimator_)

