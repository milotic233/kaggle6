import pandas as pd
df = pd.read_csv("train.csv")
strs = df['target'].value_counts()
value_map = dict((v, i) for i,v in enumerate(strs.index))
value_map = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3,'Class_5': 4,'Class_6': 5,'Class_7': 6,'Class_8': 7,'Class_9': 8}
df = df.replace({'target':value_map})
df = df.drop(columns=['id'])
x_train = df.iloc[:, :-1]
y_train = df['target']
df = pd.read_csv("test.csv")
# df = df.drop(columns=['id'])
x_test = df.iloc[:, 1:] # keep the id column for output
from sklearn.ensemble import RandomForestClassifier
from deepforest import CascadeForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
params={
    'max_depth':3,
    'num_class':9,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss'
}
dtrain = xgb.DMatrix(x_train,label=y_train)
X_test=xgb.DMatrix(x_test)
model=xgb.train(params,dtrain,100)
y_pred = model.predict(X_test)
proba = pd.DataFrame(y_pred)
proba=proba.values
output = pd.DataFrame({'id': x_test.index, 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3]})
output.to_csv('m_submission.csv', index=False)
df = pd.read_csv("test.csv")
output = pd.DataFrame({'id': df['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5':proba[:,4], 'Class_6':proba[:,5], 'Class_7':proba[:,6], 'Class_8':proba[:,7], 'Class_9':proba[:,8]})
output.to_csv('m_submission.csv', index=False)