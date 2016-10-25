import pandas as pd
import sklearn as sc
import numpy as np
import matplotlib as mp
from scipy.stats import skew
import csv
import math
from sklearn import linear_model
import xgboost as xgb

from sklearn.linear_model import Ridge,RidgeCV,ElasticNet,LassoCV,LassoLarsCV
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor as rfr

data_train=[]
data_test=[]
data_final=[]
price=[]

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, data_train, price, scoring="mean_squared_error", cv = 5))
    return(rmse)
test=pd.read_csv('../../data/test.csv')   
train=pd.read_csv('../../data/train.csv')

# print train.head()

# preprocess(test);
# preprocess(train);

data_final = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

#dropping columns
#data_final.drop('SaleType',axis=1)
data_final.drop('MiscVal',axis=1)
data_final.drop('MiscFeature',axis=1)
#data_final.drop('PoolArea',axis=1)
#data_final.drop('PoolQC',axis=1)
data_final.drop('3SsnPorch',axis=1)
data_final.drop('ScreenPorch',axis=1)
data_final.drop('EnclosedPorch',axis=1)
data_final.drop('KitchenAbvGr',axis=1)
data_final.drop('BsmtHalfBath',axis=1)

#data_final[skewed_feats] = np.log1p(data_final[skewed_feats])

'''
data_final['Age'] = data_final['YrSold'] - data_final['YearBuilt']
data_final['AgeRemod'] = data_final['YrSold'] - data_final['YearRemodAdd']
data_final['Baths'] = data_final['FullBath'] + data_final['HalfBath']
data_final['BsmtBaths'] = data_final['BsmtFullBath'] + data_final['BsmtHalfBath']
data_final['OverallQual_Square']=data_final['OverallQual']*data_final['OverallQual']
data_final['OverallQual_3']=data_final['OverallQual']*data_final['OverallQual']*data_final['OverallQual']
data_final['OverallQual_exp']=np.exp(data_final['OverallQual'])
data_final['GrLivArea_Square']=data_final['GrLivArea']*data_final['GrLivArea']
data_final['GrLivArea_3']=data_final['GrLivArea']*data_final['GrLivArea']*data_final['GrLivArea']
data_final['GrLivArea_exp']=np.exp(data_final['GrLivArea'])
data_final['GrLivArea_log']=np.log(data_final['GrLivArea'])
data_final['TotalBsmtSF_/GrLivArea']=data_final['TotalBsmtSF']/data_final['GrLivArea']
data_final['OverallCond_sqrt']=np.sqrt(data_final['OverallCond'])
data_final['OverallCond_square']=data_final['OverallCond']*data_final['OverallCond']
data_final['LotArea_sqrt']=np.sqrt(data_final['LotArea'])
data_final['1stFlrSF_sqrt']=np.sqrt(data_final['1stFlrSF'])
del data_final['1stFlrSF']
data_final['TotRmsAbvGrd_sqrt']=np.sqrt(data_final['TotRmsAbvGrd'])
del data_final['TotRmsAbvGrd']
'''




#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = data_final.dtypes[data_final.dtypes != "object"].index
#categorical_feats= data_final.dtypes[data_final.dtypes == "object"].index

# print numeric_feats
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.5]
skewed_feats = skewed_feats.index
# numeric_feats=numeric_feats.index

#print skewed_feats

data_final[skewed_feats] = np.log1p(data_final[skewed_feats])

#print 1,data_final.shape

data_final = pd.get_dummies(data_final)

#filling NA's with the mean of the column:
data_final = data_final.fillna(data_final[:train.shape[0]].mean())

#creating matrices for sklearn:
data_train = data_final[:train.shape[0]]
data_test = data_final[train.shape[0]:]

price = train.SalePrice


rf_model=rfr(n_estimators=100)
rf_model.fit(data_train,price)

rmse1=np.sqrt(-cross_val_score(rf_model,data_train,price,scoring="mean_squared_error",cv=5))

#print "Root Mean Square Error"

#print rmse1.mean()

lasso_model = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(data_train, price)

rmse2=rmse_cv(lasso_model)

#print rmse2.mean()

dtrain = xgb.DMatrix(data_train, label = price)
dtest = xgb.DMatrix(data_test)

params = {"max_depth":2, "eta":0.09}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

#model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()



xgb_model = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.07) #the params were tuned using xgb.cv
xgb_model.fit(data_train, price)



rf_preds = np.expm1(rf_model.predict(data_test))
lasso_preds = np.expm1(lasso_model.predict(data_test))
xgb_preds=np.expm1(xgb_model.predict(data_test))

final_result=0.7*lasso_preds+0.2*xgb_preds+0.1*rf_preds

solution = pd.DataFrame({"id":test.Id, "SalePrice":final_result})
solution.to_csv("kaggle_sol.csv", index = False)
