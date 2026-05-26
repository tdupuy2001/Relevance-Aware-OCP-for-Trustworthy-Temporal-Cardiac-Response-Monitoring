import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tqdm.autonotebook import tqdm
import os
from statsmodels.tsa.forecasting.theta import ThetaModel

#TODO add a SCORE FUNCTION as a parameter (to be abble to change the score (cf. l. 50 base_test))
#if done the output must be q and the interval will be compute after (for non symmetric interval the method is compute two times)

def ACI(y_test, y_pred, res_cal, alpha, gamma):
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    tab_alpha_t = np.full(test_size, alpha,dtype=float)
    err = np.empty(test_size,dtype=float)
    print('Starting ACI')
    for t in tqdm(range(test_size)):
        alpha_t = tab_alpha_t[t]
        # Original ACP
        if(1-alpha_t <= 0):
            y_lower_t, y_upper_t = 0, 0
            err_t = 1
        elif(1-alpha_t >= 1):
            y_lower_t, y_upper_t = -np.inf, np.inf
            err_t = 0
        else:
            q = np.quantile(res_cal[t],1-alpha_t)
            y_lower_t, y_upper_t = y_pred-q, y_pred+q
            err_t = 1-float((y_lower_t <= y_test) & (y_test <= y_upper_t))
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        alpha_t = alpha_t + gamma*(alpha-err_t)
        err[t] = err_t
        if t < (test_size-1):
            tab_alpha_t[t+1] = alpha_t
    print('End')
    return y_lowers, y_uppers, err, tab_alpha_t

def ACI_smooth(y_test, y_pred, res_cal, alpha, gamma, f):
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    tab_alpha_t = np.full(test_size, alpha,dtype=float)
    err = np.empty(test_size,dtype=float)
    smooth_err = np.empty(test_size,dtype=float)
    scores=np.abs(y_test-y_pred)
    print('Starting ACI')
    for t in tqdm(range(test_size)):
        alpha_t = tab_alpha_t[t]
        # Original ACP
        if(1-alpha_t <= 0):
            y_lower_t, y_upper_t = 0, 0
            err_t = 1
        elif(1-alpha_t >= 1):
            y_lower_t, y_upper_t = -np.inf, np.inf
            err_t = 0
        else:
            q = np.quantile(res_cal[t],1-alpha_t)
            y_lower_t, y_upper_t = y_pred-q, y_pred+q
            err_t = 1-float((y_lower_t <= y_test) & (y_test <= y_upper_t))
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        smooth_err[t]= f(scores[t],q[t]) if q[t]!=0 else err_t
        alpha_t = alpha_t + gamma*(alpha-smooth_err[t])
        err[t] = err_t
        if t < (test_size-1):
            tab_alpha_t[t+1] = alpha_t
    print('End')
    return y_lowers, y_uppers, err, tab_alpha_t

def OGD(y_test, y_pred, alpha, q1, lr):
    # Initialization
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    print('Starting OGD')
    for t in tqdm(range(test_size)):
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t] #symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) # same as err=float(q[i] > scores[t])
        err[t] = err_t
        if t < test_size - 1:
            q[t+1] = q[t] + lr * (err_t-alpha) 
    print('End')
    return y_lowers, y_uppers, err

def OGD_smooth(y_test, y_pred, alpha, q1, lr, f,t_burnin=100):
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    smooth_err = np.empty(test_size,dtype=float)
    scores=np.abs(y_test-y_pred)
    print('Starting OGD smooth') 
    for t in tqdm(range(test_size)):
        t_lr = t
        t_lr_min = max(t_lr - t_burnin, 0)
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t] #symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) # same as err=float(q[i] > scores[t])
        err[t] = err_t
        smooth_err[t]= f(scores[t],q[t],np.abs(np.mean(scores[t_lr_min:t]-q[t_lr_min:t]))) if t>0 else err_t
        #smooth_err[t]= f(scores[t],q[t],np.abs(np.mean(scores[t_lr_min:t+1]))) if t>0 else err_t
        if t < test_size - 1:
            q[t+1] = max(q[t] + lr*(smooth_err[t]-alpha),0)
    print('End')
    return y_lowers, y_uppers, err, smooth_err

def SF_OGD(y_test, y_pred, alpha, q1, lr):
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    print('Starting SF OGD')
    for t in tqdm(range(test_size)):
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t] #symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) # same as err=float(q[i] > scores[t])
        err[t] = err_t
        all_grad_square = (alpha-err[:t]) ** 2
        lr_norm = lr / np.sqrt(np.sum(all_grad_square))
        if t < test_size - 1:
            q[t+1] = q[t] + lr_norm * (err_t-alpha)
    print('End')
    return y_lowers, y_uppers, err


def decay_OGD(y_test, y_pred, alpha, q1, eps):
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    print('Starting decay OGD')
    for t in tqdm(range(test_size)):
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t] #symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) # same as err=float(q[i] > scores[t])
        err[t] = err_t
        lr_t = (t+1)**(-0.5-eps) 
        if t < test_size - 1:
            q[t+1] = q[t] + lr_t * (err_t-alpha)
    print('End')
    return y_lowers, y_uppers, err

def decay_OGD_smooth(y_test, y_pred, alpha, q1, eps, f, t_burnin=100):
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    smooth_err = np.empty(test_size,dtype=float)
    scores=np.abs(y_test-y_pred)
    print('Starting decay OGD smooth') 
    for t in tqdm(range(test_size)):
        t_lr = t
        t_lr_min = max(t_lr - t_burnin, 0)
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t] #symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) # same as err=float(q[i] > scores[t])
        err[t] = err_t
        smooth_err[t]= f(scores[t],q[t],np.abs(np.mean(scores[t_lr_min:t]-q[t_lr_min:t]))) if t>0 else err_t
        lr_t = (t+1)**(-0.5-eps) 
        if t < test_size - 1:
            q[t+1] = max(q[t] + lr_t * (smooth_err[t]-alpha),0)
    print('End')
    return y_lowers, y_uppers, err, smooth_err

def ECI(y_test, y_pred, alpha, q1, lr, f_dev,t_burnin=100, proportional_lr=False):
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    scores=np.abs(y_test-y_pred)
    #print('Starting ECI') 
    for t in range(test_size):
        t_lr = t
        t_lr_min = max(t_lr - t_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t] #symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) # same as err=float(q[i] < scores[t])
        err[t] = err_t 
        mean = np.abs(np.mean(scores[t_lr_min:t_lr]-q[t_lr_min:t_lr])) if t>0 else 1 #TODO
        #mean = np.abs(np.mean(scores[t_lr_min:t_lr+1])) if t>0 else 1
        if t < test_size - 1:
            q[t+1] = max(q[t] + lr_t *(err_t-alpha+(scores[t]-q[t])*f_dev(scores[t],q[t],mean)),0)  
    #print('End')
    return y_lowers, y_uppers, err

def ECI_full(y_test, y_pred, alpha, q1, lr, f, f_dev,t_burnin=100, proportional_lr=False):
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    smooth_err = np.empty(test_size,dtype=float)
    scores=np.abs(y_test-y_pred)
    print('Starting ECI') 
    for t in tqdm(range(test_size)):
        t_lr = t
        t_lr_min = max(t_lr - t_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t] #symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) # same as err=float(q[i] < scores[t])
        err[t] = err_t
        smooth_err[t]= f(scores[t],q[t],np.abs(np.mean(scores[t_lr_min:t]-q[t_lr_min:t]))) if t>0 else err_t 
        mean = np.abs(np.mean(scores[t_lr_min:t_lr]-q[t_lr_min:t_lr])) if t>0 else 1
        if t < test_size - 1:
            q[t+1] = max(q[t]+lr_t *(smooth_err[t]-alpha+(scores[t]-q[t])*f_dev(scores[t],q[t],mean)),0)  
    print('End')
    return y_lowers, y_uppers, err


def mytan(x):
    if x >= np.pi/2:
        return np.infty
    elif x <= -np.pi/2:
        return -np.infty
    else:
        return np.tan(x)

def saturation_fn_log(x, t, Csat, KI):
    if KI == 0:
        return 0
    tan_out = mytan(x * np.log(t)/(Csat * t))
    out = KI * tan_out
    return  out

def saturation_fn_tanh(x, t, Csat, KI):
    if KI == 0:
        return 0.0
    z = x * np.log(t) / (Csat * t)
    return KI * np.tanh(z)

def PID_log(y_test, y_pred, alpha, q1, lr, Csat, KI, period_scorecaster=5, t_burnin=100, proportional_lr=True, is_scorecast=False):
    # Initialization
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    scores=np.abs(y_test-y_pred)
    scorecasts = np.empty(test_size,dtype=float)
    #print('Starting PID log')
    for t in range(test_size):
        t_lr = t
        t_lr_min = max(t_lr - t_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t] #symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) 
        err[t] = err_t
        x = np.sum(err[:(t+1)]-alpha)
        integrator = saturation_fn_log(x, t+1, Csat, KI)
        # Update the next quantile
        if t < test_size - 1:
            if is_scorecast and t>t_burnin:
                curr_scores = np.nan_to_num(scores[:(t+1)]) 
                model = ThetaModel(curr_scores.astype(float),period=period_scorecaster).fit()
                scorecasts[t] = model.forecast(theta=2).iloc[0] #TODO modify if ahead=!1
                q[t+1]=max(scorecasts[t]+integrator+lr_t*(err_t-alpha),0)
            else:
                q[t+1]=max(q[t]+integrator+lr_t*(err_t-alpha),0)
    #print('End')
    return y_lowers, y_uppers, err

def PID_log_half_smooth(y_test, y_pred, alpha, q1, lr, Csat, KI, f, period_scorecaster=5, t_burnin=100, proportional_lr=True, is_scorecast=False):
    # Initialization
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    smooth_err = np.empty(test_size,dtype=float)
    scores=np.abs(y_test-y_pred)
    scorecasts = np.empty(test_size,dtype=float)
    #print('Starting PID log half smooth')
    for t in range(test_size):
        t_lr = t
        t_lr_min = max(t_lr - t_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t]#symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) 
        err[t] = err_t
        smooth_err[t]= f(scores[t],q[t],np.abs(np.mean(scores[t_lr_min:t]-q[t_lr_min:t]))) if t>0 else err_t
        #smooth_err[t]= f(scores[t],q[t],np.abs(np.mean(scores[t_lr_min:t+1]))) if t>0 else err_t
        x = np.sum(smooth_err[:(t+1)]-alpha)
        integrator = saturation_fn_log(x, t+1, Csat, KI)
        # Update the next quantile
        if t < test_size - 1:
            if is_scorecast and t>t_burnin:
                curr_scores = np.nan_to_num(scores[:(t+1)]) 
                model = ThetaModel(curr_scores.astype(float),period=period_scorecaster).fit()
                scorecasts[t] = model.forecast(theta=2).iloc[0] #TODO modify if ahead=!1
                q[t+1]=max(scorecasts[t]+integrator+lr_t*(smooth_err[t]-alpha),0)
            else:
                q[t+1]=max(q[t]+integrator+lr_t*(err_t-alpha),0)
    #print('End')
    return y_lowers, y_uppers, err, smooth_err

def PID_log_half_smooth_bis(y_test, y_pred, alpha, q1, lr, Csat, KI, f, period_scorecaster=5, t_burnin=100, proportional_lr=True, is_scorecast=False):
    # Initialization
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    smooth_err = np.empty(test_size,dtype=float)
    scores=np.abs(y_test-y_pred)
    scorecasts = np.empty(test_size,dtype=float)
    #print('Starting PID log half smooth bis')
    for t in range(test_size):
        t_lr = t
        t_lr_min = max(t_lr - t_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t] #symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) 
        err[t] = err_t
        smooth_err[t]= f(scores[t],q[t],np.abs(np.mean(scores[t_lr_min:t]-q[t_lr_min:t]))) if t>0 else err_t
        #smooth_err[t]= f(scores[t],q[t],np.abs(np.mean(scores[t_lr_min:t+1]))) if t>0 else err_t #TODO
        # if np.mean(scores[t_lr_min:t]-q[t_lr_min:t])<0:
        #     smooth_err[t]=1-smooth_err[t]
        x = np.sum(err[:(t+1)]-alpha)
        integrator = saturation_fn_log(x, t+1, Csat, KI)
        # Update the next quantile
        if t < test_size - 1:
            if is_scorecast and t>t_burnin:
                curr_scores = np.nan_to_num(scores[:(t+1)]) 
                model = ThetaModel(curr_scores.astype(float),period=period_scorecaster).fit()
                scorecasts[t] = model.forecast(theta=2).iloc[0] #TODO modify if ahead=!1 #model.forecat() forecasts the next time step
                q[t+1]=max(scorecasts[t]+integrator+lr_t*(smooth_err[t]-alpha),0)
            else:
                q[t+1]=max(q[t]+integrator+lr_t*(smooth_err[t]-alpha),0)
    #print('End')
    return y_lowers, y_uppers, err, smooth_err

def PID_log_full_smooth(y_test, y_pred, alpha, q1, lr, Csat, KI, f, period_scorecaster=5, t_burnin=100, proportional_lr=True, is_scorecast=False):
    # Initialization
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    smooth_err = np.empty(test_size,dtype=float)
    scores=np.abs(y_test-y_pred)
    scorecasts = np.empty(test_size,dtype=float)
    #print('Starting PID log full smooth')
    for t in range(test_size):
        t_lr = t
        t_lr_min = max(t_lr - t_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t] #symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) 
        err[t] = err_t
        smooth_err[t]= f(scores[t],q[t],np.abs(np.mean(scores[t_lr_min:t]-q[t_lr_min:t]))) if t>0 else err_t
        #smooth_err[t]= f(scores[t],q[t],np.abs(np.mean(scores[t_lr_min:t+1]))) if t>0 else err_t
        x = np.sum(smooth_err[:(t+1)]-alpha)
        integrator = saturation_fn_log(x, t+1, Csat, KI)
        # Update the next quantile
        if t < test_size - 1:
            if is_scorecast and t>t_burnin:
                curr_scores = np.nan_to_num(scores[:(t+1)]) 
                model = ThetaModel(curr_scores.astype(float),period=period_scorecaster).fit()
                scorecasts[t] = model.forecast(theta=2).iloc[0] #TODO modify if ahead=!1
                q[t+1]=max(scorecasts[t]+integrator+lr_t*(smooth_err[t]-alpha),0)
            else:
                q[t+1]=max(q[t]+integrator+lr_t*(smooth_err[t]-alpha),0)
    #print('End')
    return y_lowers, y_uppers, err, smooth_err


# def PID_log_half_smooth(y_test, y_pred, y_max_train, y_min_train, alpha, q1, lr, Csat, KI, f, t_burnin=100, proportional_lr=True, scorecasts=False):
#     # Initialization
#     test_size = y_test.shape[0]
#     y_lowers = np.empty(test_size,dtype=float)
#     y_uppers = np.empty(test_size,dtype=float)
#     q = np.full(test_size,q1,dtype=float)
#     err = np.empty(test_size,dtype=float)
#     smooth_err = np.empty(test_size,dtype=float)
#     scores=np.abs(y_test-y_pred)
#     print('Starting PID log half smooth')
#     for t in tqdm(range(test_size)):
#         t_lr = t
#         t_lr_min = max(t_lr - t_burnin, 0)
#         lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
#         y_lower_t = y_pred[t]-q[t]
#         y_upper_t = y_pred[t]+q[t] #symmetric interval
#         y_lowers[t] = float(y_lower_t)
#         y_uppers[t] = float(y_upper_t)
#         err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) 
#         err[t] = err_t
#         smooth_err[t]= f(scores[t],q[t], y_max_train[t], y_min_train[t]) if q[t]!=0 else err_t
#         x = np.sum(smooth_err[:(t+1)]-alpha)
#         integrator = saturation_fn_log(x, t+1, Csat, KI)
#         # Update the next quantile
#         if t < test_size - 1:
#             if scorecasts:
#                 #TODO add the scorecaster here
#                 q[t+1]=scorecasts[t+1]+integrator+lr_t*(err_t-alpha)
#             else:
#                 q[t+1]=q[t]+integrator+lr_t*(err_t-alpha)
#     print('End')
#     return y_lowers, y_uppers, err, smooth_err

# def PID_log_full_smooth(y_test, y_pred, y_max_train, y_min_train, alpha, q1, lr, Csat, KI, f, t_burnin=100, proportional_lr=True, scorecasts=False):
#     # Initialization
#     test_size = y_test.shape[0]
#     y_lowers = np.empty(test_size,dtype=float)
#     y_uppers = np.empty(test_size,dtype=float)
#     q = np.full(test_size,q1,dtype=float)
#     err = np.empty(test_size,dtype=float)
#     smooth_err = np.empty(test_size,dtype=float)
#     scores=np.abs(y_test-y_pred)
#     print('Starting PID log full smooth')
#     for t in tqdm(range(test_size)):
#         t_lr = t
#         t_lr_min = max(t_lr - t_burnin, 0)
#         lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
#         y_lower_t = min(y_pred[t]-q[t],y_pred[t]+q[t]) #TODO change this
#         y_upper_t = max(y_pred[t]+q[t],y_pred[t]-q[t]) #symmetric interval
#         y_lowers[t] = float(y_lower_t)
#         y_uppers[t] = float(y_upper_t)
#         err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) 
#         err[t] = err_t
#         smooth_err[t]= f(scores[t],q[t], y_max_train[t], y_min_train[t]) if q[t]!=0 else err_t
#         x = np.sum(smooth_err[:(t+1)]-alpha)
#         integrator = saturation_fn_log(x, t+1, Csat, KI)
#         # Update the next quantile
#         if t < test_size - 1:
#             if scorecasts:
#                 #TODO add the scorecaster here
#                 q[t+1]=scorecasts[t+1]+integrator+lr_t*(smooth_err[t]-alpha)
#             else:
#                 q[t+1]=q[t]+integrator+lr_t*(smooth_err[t]-alpha)
#     print('End')
#     return y_lowers, y_uppers, err, smooth_err

