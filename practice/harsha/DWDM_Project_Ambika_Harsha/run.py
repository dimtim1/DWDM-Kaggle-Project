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
from sklearn.ensemble import RandomForestRegressor


test=pd.read_csv('data/test.csv')   
train=pd.read_csv('data/train.csv')


def preprocess(train,test):

    all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                          test.loc[:,'MSSubClass':'SaleCondition']))

    #dropping columns
    all_data.drop('MiscVal',axis=1)
    all_data.drop('MiscFeature',axis=1)
    all_data.drop('3SsnPorch',axis=1)
    all_data.drop('ScreenPorch',axis=1)
    all_data.drop('EnclosedPorch',axis=1)
    all_data.drop('KitchenAbvGr',axis=1)
    all_data.drop('BsmtHalfBath',axis=1)


    #log transform the target:
    train["SalePrice"] = np.log1p(train["SalePrice"])

    #log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    #categorical_feats= all_data.dtypes[all_data.dtypes == "object"].index

    # print numeric_feats
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.5]
    skewed_feats = skewed_feats.index
    # numeric_feats=numeric_feats.index

    #print skewed_feats

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    #print 1,all_data.shape
    all_data = pd.get_dummies(all_data)

    #filling NA's with the mean of the column:
    all_data = all_data.fillna(all_data[:train.shape[0]].mean())

    #creating matrices for sklearn:
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]

    y = train.SalePrice

    return X_train,X_test,y


def getModels(train,y):

    model_rf = RandomForestRegressor(n_estimators=100)
    model_rf.fit(X_train,y)

    model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)

    dtrain = xgb.DMatrix(X_train, label = y)
    dtest = xgb.DMatrix(X_test)

    model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.07) #the params were tuned using xgb.cv
    model_xgb.fit(X_train, y)

    return (model_rf,model_lasso,model_xgb)



def predict(X_test,model):
    return np.expm1(model.predict(X_test))


def evaluate(model1, model2,model3, X_training, y_training, X_valid, y_valid):
    model1.fit(X_training,  y_training)
    model2.fit(X_training,  y_training)
    model3.fit(X_training,  y_training)
    preds1 = model1.predict(X_valid)
    preds2 = model2.predict(X_valid)
    preds3 = model2.predict(X_valid)
    preds = 0.7*preds1 + 0.2*preds2 + 0.1*preds3
    rmse = math.sqrt(sum((preds-y_valid)*(preds-y_valid)/len(preds)))
    return rmse


def KFold():

    meanRes=0
    print 'K    RMSE'
    for i in range(0, 10):
        X_training = pd.concat([X_train[0:i*146], X_train[(i+1)*146:1460]])
        X_valid = X_train[i*146:(i+1)*146]
        y_training = pd.concat([y[0:i*146], y[(i+1)*146:1460]])
        y_valid = y[i*146:(i+1)*146]

        res = evaluate(model_rf,model_lasso,model_xgb, X_training, y_training, X_valid, y_valid)
        meanRes+=res
        print i+1,res
    return(meanRes/10)


X_train,X_test,y = preprocess(train,test)
model_rf,model_lasso,model_xgb = getModels(X_train,y)

rf_preds = predict(X_test,model_rf)
lasso_preds = predict(X_test,model_lasso)
xgb_preds = predict(X_test,model_xgb)

mean = KFold()
print 'Average RMSE=',mean
preds=0.7*lasso_preds+0.2*xgb_preds+0.1*rf_preds

solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
sol2 = solution[['id','SalePrice']]
sol2.to_csv("xgb_lasso_rf.csv", index = False)
