import numpy as np
import matplotlib.pyplot as pl
from matplotlib import patches
from scipy.optimize import curve_fit, root, fsolve
from scipy.signal import argrelextrema
from numpy import dot, outer

def main():
    
    # Set parameters
    n_minimax = 28                     # Number of minimax points
    R_minimax = 10**12                 # Range of the minimax approximation
    n_x       = 2000                   # total number of points on the x-axis for optimization
    n_opt = 2

    xdata = 10**(np.logspace(0,np.log(np.log10(R_minimax)),n_x))/10
    ydata = np.zeros(n_x)

    alphas_betas_read = np.loadtxt("alpha_beta_of_N_"+str(n_minimax))

    alphas_betas_init = np.append(alphas_betas_read[0:n_opt] , alphas_betas_read[n_minimax:n_minimax+n_opt] )

    print("alphas_betas_init", alphas_betas_init)
    print("np.shape(alphas_betas_init)", np.shape(alphas_betas_init))

    alphas_betas_L2_opt, alphas_betas_conv = curve_fit(eta, xdata, ydata, p0=alphas_betas_init)

    sort_indices = np.argsort(alphas_betas_L2_opt[0:n_opt])

    alphas_betas_L2_opt = np.append(np.append(alphas_betas_L2_opt[sort_indices],alphas_betas_read[n_opt:n_minimax]),np.append(alphas_betas_L2_opt[sort_indices+n_opt], alphas_betas_read[n_minimax+n_opt:] ) )
    np.savetxt("alpha_beta_of_N_"+str(n_minimax)+"_L2",alphas_betas_L2_opt )

    fig1, (axis1) = pl.subplots(1,1)
    axis1.set_xlim((0.8,R_minimax))
    axis1.semilogx(xdata,eta_plotting(xdata,alphas_betas_L2_opt))

    pl.show()

def eta(x, *params):
    n_opt = np.size(params)//2
    n_minimax = 28
    alphas_betas_init = np.loadtxt("alpha_beta_of_N_"+str(n_minimax))
    return 1/x - (np.exp(-outer(x,np.append(params[0:n_opt], alphas_betas_init[n_opt:n_minimax])))).dot(np.append(params[n_opt:],alphas_betas_init[n_minimax+n_opt:]))

def eta_plotting(x, *params):
    params_1d = np.transpose(params)[:,0]
    return 1/x - (np.exp(-outer(x,params_1d[0:np.size(params)//2]))).dot(params_1d[np.size(params)//2:(np.size(params)//2)*2])

if __name__ == "__main__":
    main()

