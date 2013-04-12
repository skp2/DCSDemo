import pandas as pd
import numpy as np
import pylab as pl
import matplotlib as mpl
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

def get_data():
  train_file = '/dev/DCS/data/Train.csv'
  validate_file = '/dev/DCS/data/Validate.csv'
  train_data = pd.read_csv(train_file)
  validate_data = pd.read_csv(validate_file)
  return (train_data,validate_data)

def do_estimation(models,train,test):
  mae_gaps=[]
  mae_amts=[]
  for m in models:
    m.fit(train.ix[:, [2,3,4,5] ] ,train['Target'])
    preds = m.predict(test.ix[:, [2,3,4,5]])
    mae_amt = mean_absolute_error(preds,test['Target'])
    mae_amts.append(mae_amt)

    m.fit(train.ix[:, [1,3,4,5] ] ,train['DaysSinceLast'])
    preds = m.predict(test.ix[:, [1,3,4,5]])
    mae_gap = mean_absolute_error(preds,test['DaysSinceLast'])
    mae_gaps.append(mae_gap)
  mae_gaps.append(mean_absolute_error(test['DaysSinceLast2'],test['DaysSinceLast']))
  mae_amts.append(mean_absolute_error(test['LastPayment'],test['Target']))
  return (mae_gaps,mae_amts)


models = [ Ridge(alpha=0.1),GradientBoostingRegressor(),RandomForestRegressor()]
model_names = ['Ridge Regression','Gradient Boosted Tree','Random Forest', 'Benchmark']

train,test = get_data()

mae_gaps,mae_amts = do_estimation(models,train,test)

fig1 = pl.figure('Payment Amount MAE')
ax1 = pl.subplot(111)
ax1.bar(range(len(model_names)),mae_amts,width=0.5)
ax1.set_xticks(np.arange(len(model_names))+0.25)
ax1.set_xticklabels(model_names)
ax1.set_title('Payment Amount MAE')
fig1.savefig('MAE_Payment_Amount.png')
  

fig2 = pl.figure('Payment Gap MAE')
ax2 = pl.subplot(111)
ax2.set_title('Payment Gap MAE')
ax2.bar(range(len(model_names)),mae_gaps,width=0.5)
ax2.set_xticks(np.arange(len(model_names))+0.25)
ax2.set_xticklabels(model_names)
fig2.savefig('MAE_Payment_Gap.png')
