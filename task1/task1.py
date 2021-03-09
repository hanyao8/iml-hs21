import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
import numpy as np
import os

#sample = pd.read_csv("data/sample.csv")
sample = open("data/sample.csv")
#sample = open(os.path.join(os.getcwd(),"data/sample.csv"))

df = pd.read_csv("data/train.csv")
df_np = df.to_numpy()
y = df_np[:,0]
X = df_np[:,1:]

ridge0 = Ridge().fit(X,y)
y_pred = ridge0.predict(X)
mse0 = mean_squared_error(y,y_pred,squared=False)

alphas = [0.1,1,10,100,200]
result = np.zeros(len(alphas))
for i in range(len(alphas)):
    ridge_cv = Ridge(alpha=alphas[i])
    cv_results = cross_validate(ridge_cv,X,y,cv=10,
            scoring=["neg_root_mean_squared_error"])
    rmse = -1.0*cv_results['test_neg_root_mean_squared_error']
    rmse_cv_mean = np.mean(rmse)
    result[i] = rmse_cv_mean

result_str = ""
for i in range(len(result)):
    result_str+=str(result[i])
    result_str+="\n"

submission = open("data/sub.csv","w")
submission.write(result_str)
submission.close()
