import numpy as np
import matplotlib.pyplot as pl
from matplotlib import patches
from scipy.optimize import curve_fit, root, fsolve
from scipy.signal import argrelextrema
from numpy import dot, outer

def main():
    
    # Set parameters
    n_minimax = 28                     # Number of minimax points
    R_minimax = 10**10                 # Range of the minimax approximation
    n_x       = 5000                   # total number of points on the x-axis for optimization
    eps_diff  = 10**(-10)

    xdata = 10**(np.logspace(0,np.log(np.log10(R_minimax)),n_x))/10

    alphas_betas = np.loadtxt("alpha_beta_of_N_"+str(n_minimax)+"_L2")
    alphas_betas_E = np.append(alphas_betas,1)

    sort_indices = np.argsort(alphas_betas_E[0:n_minimax])

    fig1, (axis1) = pl.subplots(1,1)
    axis1.set_xlim((0.8,R_minimax))
    axis1.semilogx(xdata,eta_plotting(xdata,alphas_betas_E))

    pl.show()

def eta(x, *params):
    return 1/(2*x) - (np.exp(-outer(x,params[0:np.size(params)//2]))).dot(params[np.size(params)//2:])

def eta_plotting(x, *params):
    params_1d = np.transpose(params)[:,0]
    return 1/(2*x) - (np.exp(-outer(x,params_1d[0:np.size(params)//2]))).dot(params_1d[np.size(params)//2:(np.size(params)//2)*2])

if __name__ == "__main__":
    main()

