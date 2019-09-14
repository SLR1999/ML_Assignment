import numpy as np
import matplotlib.pyplot as plt

def logistic_func(w, X): 
    ''' 
    logistic(sigmoid) function 
    '''
    return 1.0/(1 + np.exp(-np.dot(X, w.T))) 
  
  
def log_gradient(w, X, y): 
    ''' 
    logistic gradient function 
    '''
    first_calc = logistic_func(w, X) - y.reshape(X.shape[0], -1) 
    final_calc = np.dot(first_calc.T, X) 
    return final_calc 
  
  
def cost_func(w, X, y): 
    ''' 
    cost function, J 
    '''
    log_func_v = logistic_func(w, X) 
    y = np.squeeze(y) 
    step1 = y * np.log(log_func_v) 
    step2 = (1 - y) * np.log(1 - log_func_v) 
    final = -step1 - step2 
    return np.mean(final) 
  
  
def grad_desc(X, y, w, alpha=.01, converge_change=.001): 
    ''' 
    gradient descent function 
    '''
    cost = cost_func(w, X, y) 
    change_cost = 1
    num_iter = 1
      
    while(change_cost > converge_change): 
        old_cost = cost 
        w = w - (alpha * log_gradient(w, X, y)) 
        cost = cost_func(w, X, y) 
        change_cost = old_cost - cost 
        num_iter += 1
      
    return w, num_iter  
  
  
def pred_values(w, X): 
    ''' 
    function to predict labels 
    '''
    pred_prob = logistic_func(w, X) 
    pred_value = np.where(pred_prob >= .5, 1, 0) 
    return np.squeeze(pred_value) 
  
  
def plot_reg(X, y, w): 
    ''' 
    function to plot decision boundary 
    '''
    # labelled observations 
    x_0 = X[np.where(y == 0.0)] 
    x_1 = X[np.where(y == 1.0)] 
      
    # plotting points with diff color for diff label 
    plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='b', label='y = 0') 
    plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1') 
      
    # plotting decision boundary 
    x1 = np.arange(0, 1, 0.1) 
    x2 = -(w[0,0] + w[0,1]*x1)/w[0,2] 
    plt.plot(x1, x2, c='k', label='reg line') 
  
    plt.xlabel('x1') 
    plt.ylabel('x2') 
    plt.legend() 
    plt.show() 
