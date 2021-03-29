import pandas as pd
#from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
import numpy as np
import os

#sample = pd.read_csv("data/sample.csv")
sample = open("data/sample.csv")
#sample = open(os.path.join(os.getcwd(),"data/sample.csv"))

df = pd.read_csv("data/train.csv")


df_ft = df.copy()
print(list(df_ft))
#df_ft = df_ft.drop('Id',inplace=False,axis=1)
df_ft.drop('Id',inplace=True,axis=1)

df_ft['x6'] = (df_ft['x1'].values)**2
df_ft['x7'] = (df_ft['x2'].values)**2
df_ft['x8'] = (df_ft['x3'].values)**2
df_ft['x9'] = (df_ft['x4'].values)**2
df_ft['x10'] = (df_ft['x5'].values)**2

df_ft['x11'] = np.exp(df_ft['x1'].values)
df_ft['x12'] = np.exp(df_ft['x2'].values)
df_ft['x13'] = np.exp(df_ft['x3'].values)
df_ft['x14'] = np.exp(df_ft['x4'].values)
df_ft['x15'] = np.exp(df_ft['x5'].values)

df_ft['x16'] = np.cos(df_ft['x1'].values)
df_ft['x17'] = np.cos(df_ft['x2'].values)
df_ft['x18'] = np.cos(df_ft['x3'].values)
df_ft['x19'] = np.cos(df_ft['x4'].values)
df_ft['x20'] = np.cos(df_ft['x5'].values)

df_ft['x21'] = 1.0

#finished defining feature transformations
print(list(df_ft))
df_np = df_ft.to_numpy()
y = df_np[:,0]
X = df_np[:,1:]

#linreg = LinearRegression(fit_intercept=True).fit(X,y)
linreg = LinearRegression(fit_intercept=False).fit(X,y)
print(linreg.coef_)
print(linreg.intercept_)
print(len(linreg.coef_))

result = (linreg.coef_).copy()
y_pred = linreg.predict(X)
rmse = mean_squared_error(y,y_pred,squared=False)
print(rmse)

result_str = ""
for i in range(len(result)):
    result_str+=str(result[i])
    result_str+="\n"

submission = open("data/sub.csv","w")
submission.write(result_str)
submission.close()




