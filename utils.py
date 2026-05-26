import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tqdm.autonotebook import tqdm
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns
import os 
import pickle 
import statistics

from methods import ACI, ACI_smooth, OGD, OGD_smooth, SF_OGD, decay_OGD, decay_OGD_smooth, ECI, ECI_full, PID_log, PID_log_half_smooth, PID_log_half_smooth_bis, PID_log_full_smooth

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.kernel_ridge import KernelRidge

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.feature_selection import VarianceThreshold

from scipy.stats import loguniform, randint

#different training methods

#the indexation of the dataset starts at 1 then if you want to start the training from the beginning put start_train=1
#then if end_train=start_test-1 there is no step between train and test
#start_test-end_train=ahead
def train(X, Y, start_train, end_train, start_test, end_test, basemodel, **kwargs):
    # Initialization
    assert start_train>0, 'indexation of the dataset starts at 1'

    n = len(Y)
    test_size = end_test - start_test + 1
    train_size = end_train-start_train + 1
    assert test_size+train_size<=n

    y_pred = np.empty(test_size)
    y_test = np.empty(test_size)

    assert basemodel in ['RF','LR'], 'basemodel must be RF or LR.'
    if basemodel == 'RF':
        try:
            n_estimators = kwargs['n_estimators']
            min_samples_leaf = kwargs['min_samples_leaf']
            max_features = kwargs['max_features']
        except:
            raise ValueError("Arguments n_estimators, min_samples_leaf and max_features must be passed")
    print('Forecasting...')
    for t in tqdm(range(test_size)):
        x_train_t = X[start_train-1+t:(end_train+t),]
        x_test_t = X[(start_test+t),].reshape(1, -1)
        y_train_t = Y[start_train-1+t:(end_train+t)]
        y_test_t = Y[(start_test-1+t)] #true value
        if basemodel == 'RF':
            reg = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,
                                        random_state=1)
        elif basemodel == 'LR':
            reg = LinearRegression()
        reg.fit(x_train_t, y_train_t)
        y_pred_t = reg.predict(x_test_t)
        y_test[t]=y_test_t
        y_pred[t]=y_pred_t
    print('Forecasting finished') 
    return y_test, y_pred

regressors={              
                    "RF": (RandomForestRegressor, {"random_state": 2025}), #loss=MSE
                    "SVR": (SVR, {},),
                    "XGB": (XGBRegressor, {"random_state": 2025}), #loss=RMSE             
                    "KR": (KernelRidge, {}),
                    "LGBM": (LGBMRegressor, {"verbose":-1,"random_state": 2025})          
                    }
dict_params_search={                        
                "RF": {"scaling":[MinMaxScaler(),StandardScaler()],'model__n_estimators': randint(200,1000),'model__max_depth' : randint(4,12)},
                "SVR": {"scaling":[MinMaxScaler(),StandardScaler()],'model__kernel':['rbf','linear','sigmoid'], "model__gamma": loguniform(1e-4,10), "model__C": loguniform(5e-2,5e4)},
                "XGB": {"scaling":[MinMaxScaler(),StandardScaler()],"model__n_estimators": randint(200,1000), "model__max_depth":randint(4,12), "model__learning_rate": loguniform(1e-3, 1e-1)},                
                "KR": {"scaling":[MinMaxScaler(),StandardScaler()],'model__kernel':['rbf','linear','sigmoid'], "model__alpha": loguniform(1e-5,10), "model__gamma": loguniform(1e-4,10)},
                "LGBM": {"scaling":[MinMaxScaler(),StandardScaler()],"model__n_estimators": randint(200,1000), "model__max_depth":randint(4,12), "model__learning_rate": loguniform(1e-3, 1e-1)}          
                }

def train_opt(X, Y, start_train, end_train, start_test, end_test, basemodel, sliding, cv_search, n_iter_search,retrain_every_n_steps):
    assert start_train>0, 'indexation of the dataset starts at 1'
    ahead = start_test-end_train
    assert ahead>=1, "test must be after train" 

    n = len(Y)
    test_size = end_test - start_test + 1
    assert test_size+end_train - start_train + 1<=n

    y_pred = np.empty(test_size)
    y_test = np.empty(test_size)

    assert basemodel in ['RF','SVR', 'XGB', 'KR', 'LGBM'], 'basemodel must be RF, SVR, XGB, KR or LGBM'
    model,model_params=regressors[basemodel]
    #ML predictions
    print('Forecasting...')
    for t in tqdm(range(test_size)):
        if sliding==True:
            x_train_t = X[start_train-1+t:(end_train-1+t),]
            x_test_t = X[(start_test-1+t),].reshape(1, -1)
            y_train_t = Y[start_train-1+t:(end_train-1+t)] 
        else:
            x_train_t = X[start_train-1:(end_train-1+t),]
            x_test_t = X[(start_test-1+t),].reshape(1, -1) #mandatory for the model to predict (need an array (n_samples,n_features) and not 1D)
            y_train_t = Y[start_train-1:(end_train-1+t)]
        y_test_t = Y[(start_test-1+t)]
        if t%retrain_every_n_steps==0:
            pip= Pipeline([('variance_0',VarianceThreshold(threshold=0)),('scaling','passthrough'),("feature_selection","passthrough"),("model",model(**model_params))])
            hyperparams_reg=dict_params_search[basemodel]
            tscv = TimeSeriesSplit(n_splits=cv_search)
            full_model = RandomizedSearchCV(pip, hyperparams_reg, n_iter=n_iter_search, n_jobs = -1, cv=tscv)
            full_model.fit(x_train_t, y_train_t)
        y_pred_t = full_model.predict(x_test_t)
        y_test[t]=y_test_t
        y_pred[t]=y_pred_t
    print('Forecasting finished')
    return y_test, y_pred

def train_calibrate_aci(X, Y, train_size, basemodel, params_basemodel):
    # Initialization
    n = len(Y)
    test_size = n - train_size
    idx = np.array(range(train_size))
    n_half = int(np.floor(train_size/2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half] #première moitié pour train et deuxième moitié pour calibrer
    y_pred = np.empty(test_size)
    res_cal = np.empty(test_size)
    y_test = np.empty(test_size)
    mean_reg = True
    assert basemodel in ['RF','LR'], 'basemodel must be RF or LR.'
    if basemodel == 'RF':
        if mean_reg:
            n_estimators = params_basemodel['n_estimators']
            min_samples_leaf = params_basemodel['min_samples_leaf']
            max_features = params_basemodel['max_features']
    print('Forecasting...')
    for t in tqdm(range(test_size)):
        x_train_t = X[t:(train_size+t),]
        x_test_t = X[(train_size+t),].reshape(1, -1)
        y_train_t = Y[t:(train_size+t)]
        y_test_t = Y[(train_size+t)]
        if mean_reg:
            if basemodel == 'RF':
                reg = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,
                                            random_state=1)
            elif basemodel == 'LR':
                reg = LinearRegression()
            reg.fit(x_train_t[idx_train,:], y_train_t[idx_train])
            y_pred_t = reg.predict(x_test_t)
            y_pred_cal_t = reg.predict(x_train_t[idx_cal,:])
            res_cal_t = np.abs(y_train_t[idx_cal]-y_pred_cal_t)
        y_test[t]=y_test_t
        y_pred[t]=y_pred_t
        res_cal[t]=res_cal_t
    print('Forecasting finished') 
    return y_test, y_pred, res_cal

#sliding=slide the training set when moving forward
def train_withoutX(Y, start_train, end_train, start_test, end_test, basemodel, sliding,**kwargs):
    # Initialization
    assert start_train>0, 'indexation of the dataset starts at 1'

    ahead = start_test-end_train
    assert ahead>=1, "test must be after train" 

    n = len(Y)
    test_size = end_test - start_test + 1
    assert test_size+end_train - start_train + 1<=n

    y_pred = np.empty(test_size)
    y_test = np.empty(test_size)

    assert basemodel in ['AR','Theta'], 'basemodel must be AR or Theta'
    if basemodel=='Theta':
        period_regressor=kwargs['period_regressor']
    print('Forecasting...')
    for t in tqdm(range(test_size)):
        if sliding==True:
            y_train_t = Y[start_train-1+t:(end_train+t)] 
            train_size = end_train - start_train + 1
        else:
            y_train_t = Y[start_train-1:(end_train+t)]
            train_size = end_train + t - start_train + 1
        y_test_t = Y[(start_test-1+t)]
        if basemodel == 'AR':
            model = AutoReg(y_train_t,lags=3).fit()
            y_pred_t = model.predict(start=train_size-1+ahead,end=train_size-1+ahead) #train_size-1 because according to the documentation AutoReg uses a 0-indexation
            y_pred_t = y_pred_t[0]
        # elif basemodel == 'Prophet':
        #     model = Prophet()
        #     model.fit(y_train_t)
        #     y_pred_t = model.predict(start=train_size-1+ahead,end=train_size-1+ahead) 
        #     y_pred_t = y_pred_t[0]
        elif basemodel == 'Theta':
            model = ThetaModel(y_train_t,period=period_regressor).fit()
            y_pred_t = model.forecast(theta=2).iloc[0]
        y_test[t]=y_test_t
        y_pred[t]=y_pred_t
    print('Forecasting finished') 
    
    return y_test, y_pred

#load dataset

def load_dataset(name):
    if name == "daily-climate":
        df = pd.read_csv('./datasets/daily-climate.csv')
        df.rename({'date': 'timestamp', 'meantemp': 'Y'}, axis=1, inplace=True)
        df.drop("Unnamed: 0", axis=1,inplace=True)
        df.set_index('timestamp', inplace=True)
    if name == "GOOGL":
        df = pd.read_csv('./datasets/djia.csv')
        df.rename({'Date': 'timestamp', 'Open': 'Y'}, axis=1, inplace=True)
        df=df[df['Name']=='GOOGL']
        df.drop('Name', axis=1, inplace=True)
        df.set_index('timestamp', inplace=True)
    if name == "AMZN":
        df = pd.read_csv('./datasets/djia.csv')
        df.rename({'Date': 'timestamp', 'Open': 'Y'}, axis=1, inplace=True)
        df=df[df['Name']=='AMZN']
        df.drop('Name', axis=1, inplace=True)
        df.set_index('timestamp', inplace=True)
    if name == "MSFT":
        df = pd.read_csv('./datasets/djia.csv')
        df.rename({'Date': 'timestamp', 'Open': 'Y'}, axis=1, inplace=True)
        df=df[df['Name']=='MSFT']
        df.drop('Name', axis=1, inplace=True)
        df.set_index('timestamp', inplace=True)
    df['Y'] = df['Y'].astype(float)
    df.interpolate(inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df[[col for col in df.columns if col != 'Y'] + ['Y']]
    return df

def load_player(df,player):
    players=df["Code anonyme"].unique() 
    assert player in players
    filter_player=df["Code anonyme"]==player
    df_player = df.loc[filter_player].copy()
    df_player.dropna(inplace=True)
    df_player.rename({'Date': 'timestamp', 'FC_moyenne': 'Y'}, axis=1, inplace=True)
    df_player.drop(columns=['FC_moyenne_perc','HRZ1','HRZ2','HRZ3','HRZ4','HRZ5','HRZ6','Saison','Code anonyme','Effectif','debut_seq_format','fin_seq_format','Timestamp_debut','Timestamp_fin','meteo_source'],inplace=True) ##remove str, time stamp and HR information
    df_player.set_index('timestamp', inplace=True)
    df_player.index = pd.to_datetime(df_player.index)
    df_player= df_player[[col for col in df_player.columns if col != 'Y'] + ['Y']]
    return df_player

# relevance-aware functions

def quadratic_asym(a,b,alpha):
    # a and b are the coefficient with which you will divide q to choose the points where f=0 and f=1 
    # ie choose x and x' such as x=-q/a and x'=q/b where f(x)=0 and f(x')=1
    def quadratic(s,q):
        x=s-q
        l=-q/a
        r=q/b
        if x>=0:
            quad=((1-alpha)/(r**2))*(x**2)+alpha
            value=min(quad,1)
        else:
            quad=((-alpha)/(l**2))*(x**2)+alpha
            value=max(quad,0)
        return value
    return quadratic

def quadratic_symleft(a,alpha):
    def quadratic(s,q):
        x=s-q
        l=-q/a
        if x>=0:
            quad=(alpha/(l**2))*(x**2)+alpha
            value=min(quad,1)
        else:
            quad=((-alpha)/(l**2))*(x**2)+alpha
            value=max(quad,0)
        return value
    return quadratic

def quadratic_symright(b,alpha):
    def quadratic(s,q):
        x=s-q
        r=q/b
        if x>=0:
            quad=((1-alpha)/(r**2))*(x**2)+alpha
            value=min(quad,1)
        else:
            quad=((alpha-1)/(r**2))*(x**2)+alpha
            value=max(quad,0)
        return value
    return quadratic

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#mean as an argument to be general in the file "methods" (our function needs it)
def dev_sigmoid_c(c):
    def dev_sigmoid(s,q, mean):
        x=s-q
        dev_sig=c*sigmoid(c*x)*(1-sigmoid(c*x)) #use of form sig(1-sig) for numeric computation
        return dev_sig
    return dev_sigmoid

def f_w_v(w,v,alpha):
    def f(s, q, mean):
        x=s-q
        a=v/mean
        b=np.array([-np.log((1-alpha)/alpha) for _ in range(w.shape[0])])
        return np.sum(w[:, None] * sigmoid(a[:, None] * x + b[:, None]), axis=0)    
    return f

def dev_f_w_v(w,v,alpha):
    def dev_sum(s, q, mean):
        x=s-q
        a=v/mean
        b=np.array([-np.log((1-alpha)/alpha) for _ in range(w.shape[0])])
        z=a[:, None] * x + b[:, None]
        return np.sum(w[:, None] * a[:, None] * sigmoid(z) * (1-sigmoid(z)), axis=0) #use of form sig(1-sig) for numeric computation
    return dev_sum

# exe train+OCP

def exe(dataset, start_train, end_train, start_test, end_test, basemodel, name_method, alpha, is_X,sliding=True,**kwargs):
    train_size=end_train-start_train+1
    #methods = {"ACI":ACI, "OGD":OGD, "SF_OGD":SF_OGD, "decay_OGD":decay_OGD, "ECI":ECI, "PID_log":PID_log, "PID_log_half_smooth":PID_log_half_smooth, "PID_log_full_smooth":PID_log_full_smooth}
    value_X = 'with_X' if is_X else 'without_X'
    value_slide = 'slide' if sliding else 'no_slide'
    result_path = f'./results/forecasts/{dataset}/{value_X}/{basemodel}/'
    config_name = f'{start_train}_{end_train}_{start_test}_{end_test}_{value_slide}'
    os.makedirs(result_path, exist_ok=True)
    file_name = result_path+config_name+'.pkl'
    #TODO add saving ACI if ACI (depends on the parameter of the RF)
    if name_method=="ACI" or name_method=="ACI_smooth":
        y_test_aci, y_pred_aci, res_cal_aci = train_calibrate_aci(X, Y, train_size, basemodel, **kwargs)
    else:
        try:
            with open(file_name, 'rb') as handle:
                forecasts = pickle.load(handle)
        except:
            data=load_dataset(dataset) 
            Y=data['Y']
            Y_array=Y.to_numpy()
            if is_X:
                name_X=data.columns[:-1]
                X=data[name_X]
                X_array=X.to_numpy()
                y_test, y_pred= train(X_array, Y_array, start_train, end_train, start_test, end_test, basemodel, **kwargs)
                forecasts={'y_test':y_test,'y_pred':y_pred}
            else: 
                y_test, y_pred = train_withoutX(Y_array, start_train, end_train, start_test, end_test, basemodel, sliding, **kwargs)
                forecasts={'y_test':y_test,'y_pred':y_pred}
            with open(file_name, 'wb') as handle:
                pickle.dump(forecasts,handle)
        y_test, y_pred = forecasts['y_test'], forecasts['y_pred']

    if name_method=="ACI":
        y_lowers, y_uppers, err, _=ACI(y_test_aci, y_pred_aci, res_cal_aci, alpha, kwargs['gamma']) 
    elif name_method=="OGD":
        y_lowers, y_uppers, err=OGD(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'])
    elif name_method=="OGD_smooth":
        y_lowers, y_uppers, err, smooth_err=OGD_smooth(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'],kwargs['f'])
    elif name_method=="SF_OGD":
        y_lowers, y_uppers, err=SF_OGD(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'])
    elif name_method=="decay_OGD":
        y_lowers, y_uppers, err=decay_OGD(y_test, y_pred, alpha, kwargs['q1'],kwargs['eps'])
    elif name_method=="decay_OGD_smooth":
        y_lowers, y_uppers, err, smooth_err=decay_OGD_smooth(y_test, y_pred, alpha, kwargs['q1'],kwargs['eps'],kwargs['f'])
    elif name_method=="ECI":
        y_lowers, y_uppers, err=ECI(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'], kwargs['f_dev'],proportional_lr=False)
    elif name_method=="ECI_full":
        y_lowers, y_uppers, err=ECI_full(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'], kwargs['f'],kwargs['f_dev'],proportional_lr=False)
    elif name_method=="PID_log":
        y_lowers, y_uppers, err=PID_log(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'], kwargs['Csat'], kwargs['KI'],proportional_lr=False)
    elif name_method=="PID_log_half_smooth":
        y_lowers, y_uppers, err, smooth_err=PID_log_half_smooth(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'], kwargs['Csat'], kwargs['KI'], kwargs['f'],proportional_lr=False)
    elif name_method=="PID_log_half_smooth_bis":
        y_lowers, y_uppers, err, smooth_err=PID_log_half_smooth_bis(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'], kwargs['Csat'], kwargs['KI'], kwargs['f'],proportional_lr=False)
    elif name_method=="PID_log_full_smooth": 
        y_lowers, y_uppers, err, smooth_err =PID_log_full_smooth(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'], kwargs['Csat'], kwargs['KI'], kwargs['f'],proportional_lr=False)
    else:
        raise ValueError("Unknown method")
    #TODO compute y_lowers and y_uppers thanks to q and interval function (depending on asymmetric or not)
    try:
        result={"y_lowers":y_lowers, "y_uppers":y_uppers, "err":err, 'smooth_err':smooth_err}
    except:
        result={"y_lowers":y_lowers, "y_uppers":y_uppers, "err":err}
    return result

def exe_player(df_player, name_player, start_train, end_train, start_test, end_test, basemodel,name_method, alpha,cv_search, cv_method, n_iter_search, with_acute_load,retrain_every_n_steps,sliding=False,**kwargs):
    value_slide = 'slide' if sliding else 'no_slide'
    value_load = 'with_acute_load' if with_acute_load else 'without_acute_load'
    value_cv_method = "folds" if cv_method=="KFolds" else ''
    result_path = f'./results/forecasts/HR/{name_player}/{basemodel}/'
    config_name = f'{start_train}_{end_train}_{start_test}_{end_test}_{value_load}_{value_slide}_{cv_search}{value_cv_method}_{n_iter_search}_{retrain_every_n_steps}'
    os.makedirs(result_path, exist_ok=True)
    file_name = result_path+config_name+'.pkl'
    try:
        with open(file_name, 'rb') as handle:
            forecasts = pickle.load(handle)
    except:
        if with_acute_load==False: #if we don't want to keep acute load information
            df_player_updated=df_player.drop(columns=["acute_DT","acute_HSR","acute_SPR","acute_ACC","acute_DEC","acute_HR5","acute_HR6"]) #remove features about acute load
        else:
            df_player_updated=df_player
        Y=df_player_updated['Y']
        Y_array=Y.to_numpy()
        name_X=df_player_updated.columns[:-1]
        X=df_player_updated[name_X]
        X_array=X.to_numpy()
        y_test, y_pred= train_opt(X_array, Y_array, start_train, end_train, start_test, end_test, basemodel, sliding, cv_search, n_iter_search,retrain_every_n_steps)
        forecasts={'y_test':y_test,'y_pred':y_pred}
        with open(file_name, 'wb') as handle:
            pickle.dump(forecasts,handle)
    y_test, y_pred = forecasts['y_test'], forecasts['y_pred']
    if name_method=="OGD":
        y_lowers, y_uppers, err=OGD(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'])
    elif name_method=="OGD_smooth":
        y_lowers, y_uppers, err, smooth_err=OGD_smooth(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'],kwargs['f'])
    elif name_method=="SF_OGD":
        y_lowers, y_uppers, err=SF_OGD(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'])
    elif name_method=="decay_OGD":
        y_lowers, y_uppers, err=decay_OGD(y_test, y_pred, alpha, kwargs['q1'],kwargs['eps'])
    elif name_method=="decay_OGD_smooth":
        y_lowers, y_uppers, err, smooth_err=decay_OGD_smooth(y_test, y_pred, alpha, kwargs['q1'],kwargs['eps'],kwargs['f'])
    elif name_method=="ECI":
        y_lowers, y_uppers, err=ECI(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'], kwargs['f_dev'],proportional_lr=False)
    elif name_method=="ECI_full":
        y_lowers, y_uppers, err=ECI_full(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'], kwargs['f'],kwargs['f_dev'],proportional_lr=False)
    elif name_method=="PID_log":
        y_lowers, y_uppers, err=PID_log(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'], kwargs['Csat'], kwargs['KI'],proportional_lr=False)
    elif name_method=="PID_log_half_smooth":
        y_lowers, y_uppers, err, smooth_err=PID_log_half_smooth(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'], kwargs['Csat'], kwargs['KI'], kwargs['f'],proportional_lr=False)
    elif name_method=="PID_log_half_smooth_bis":
        y_lowers, y_uppers, err, smooth_err=PID_log_half_smooth_bis(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'], kwargs['Csat'], kwargs['KI'], kwargs['f'],proportional_lr=False)
    elif name_method=="PID_log_full_smooth": 
        y_lowers, y_uppers, err, smooth_err =PID_log_full_smooth(y_test, y_pred, alpha, kwargs['q1'], kwargs['lr'], kwargs['Csat'], kwargs['KI'], kwargs['f'],proportional_lr=False)
    else:
        raise ValueError("Unknown method")
    #TODO compute y_lowers and y_uppers thanks to q and interval function (depending on asymmetric or not)
    try:
        result={"y_lowers":y_lowers, "y_uppers":y_uppers, "y_test":y_test, "y_pred":y_pred, "err":err, 'smooth_err':smooth_err}
    except:
        result={"y_lowers":y_lowers, "y_uppers":y_uppers, "y_test":y_test, "y_pred":y_pred, "err":err}
    return result

#PLOT

def plot_exp(filename,results,name_methods,method_to_title):
    color_plot = {'coverage': '#377eb8','size': '#4daf4a'}
    fig, axs = plt.subplots(2,len(name_methods), figsize=(6,6))
    widths=[(results[method]['y_uppers']-results[method]['y_lowers']) for method in results.keys()]
    min_width = np.ceil(np.stack(widths).min()) if np.stack(widths).min()>=0 else np.floor(np.stack(widths).min()) 
    max_width = np.ceil(np.stack(widths).max()) if np.stack(widths).max()>=0 else np.floor(np.stack(widths).max()) 
    for i,method in enumerate(results.keys()):
        cov=1-results[method]['err']
        cov_moving_avg=[cov[:j].sum()/len(cov[:j]) for j in range(1,len(cov)+1)]
        width=results[method]['y_uppers']-results[method]['y_lowers']
        axs[0,i].scatter(range(len(width)),width,c=color_plot['size'],marker='.',s=5)
        axs[1,i].scatter(range(len(cov_moving_avg)),cov_moving_avg,c=color_plot['coverage'],marker='.',s=5)
        axs[1,i].axhline(0.9,linestyle='--',color='black',linewidth=1)
        axs[0,i].set_ylim(min_width*0.9, max_width*1.1)
        axs[1,i].set_ylim(-0.05,1.05)
        axs[0,i].set_title(f'{method_to_title[method]}')
    axs[0,0].set_ylabel('Interval width')
    axs[1,0].set_ylabel('Coverage')
    fig.supxlabel('Time (sessions)')
    #fig.legend(loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(f'results/images/{filename}.svg',bbox_inches='tight')
    print('Figure saved')
    plt.show()

def means_exp(result,burnin=None,only_correct_intervals=False):
    cov=1-result['err']
    width=result['y_uppers']-result['y_lowers']
    if burnin:
        cov=cov[burnin-1:]
        width=width[burnin-1:]
    if only_correct_intervals:
        mask = (cov == 1)
        width = width[mask]
    cov_mean=cov.mean()
    width_mean=width.sum()/len(width)
    width_median=statistics.median(width)
    return cov_mean,width_mean, width_median

def plot_evolution(result, filename=None):
    y_test = result['y_test']
    y_lower = result['y_lowers']
    y_upper = result['y_uppers']
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Ground truth', color="black", linestyle="-",linewidth=1.5)
    plt.plot(y_lower, label='Upper and lower bounds', color="#4fa8c8", linestyle='--', linewidth=1)
    plt.plot(y_upper, color='#4fa8c8', linestyle='--', linewidth=1)
    plt.fill_between(range(len(y_test)), y_lower, y_upper, color='#4fa8c8', alpha=0.2, label='Prediction interval')
    plt.xlabel('Time (sessions)')
    plt.ylabel('Heart Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if filename:
        plt.savefig(f'results/images/{filename}.svg', bbox_inches='tight')
        print('Figure saved')
    plt.show()

def plot_interval_widths_by_player(rows,filename,methods=None):
    df=pd.DataFrame(rows)
    if methods:
        df_update=df[df['method'].isin(methods)]
    else:
        df_update=df
    palette={
    "PID":"#1f77b4",              
    "r-aPID":"#ff7f0e",           
    "PID Equation (8)":"#8a3bd5", 
    "PID Equation (9)":"#f550b3",
    "r-aECI":"#2ca02c",           
    "ECI":"#d62728"              
    }
    markers={
    "PID":"o",        
    "r-aPID":"X",     
    "PID Equation (8)":"s",
    "PID Equation (9)":"^",
    "r-aECI":"v",
    "ECI":"P"
    }
    plt.figure(figsize=(12,6))
    sns.scatterplot(data=df_update,x="player",y="width_median",hue="method",style="method",s=140,palette=palette,markers=markers)
    sns.lineplot(data=df_update,x="player",y="width_median",units="player",estimator=None,color="gray",linewidth=1.5,alpha=0.7)
    plt.ylabel("Median interval width")
    plt.xlabel("")
    plt.xticks(rotation=90)
    plt.grid(True,alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f'results/images/{filename}.svg', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # import random
    # data=load_dataset("daily-climate")
    # y=[i for i in range(100)]
    #print(train_withoutX(y, 1, 80, 83, 100, "AR"))
    # test= [np.array([i-1,1-i]) for i in range(5)]
    # print(np.stack(test,axis=0)[:,1],test[1])
    # import pandas as pd
    # df = pd.read_csv('./datasets/djia.csv')
    # print(set(df['Name']))
    print(load_player('MHSC-45'))
