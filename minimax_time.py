import numpy as np
import matplotlib.pyplot as pl
from matplotlib import patches
from scipy.optimize import curve_fit, root, fsolve
from scipy.signal import argrelextrema
from numpy import dot, outer

def main():
    
    # Set parameters
    n_minimax = 10                    # Number of minimax points
    R_minimax = 10**8                 # Range of the minimax approximation
    n_x       = 1000                   # total number of points on the x-axis for optimization
    eps_diff  = 10**(-5)

    # initialization: 
    ###############################################################################################
    alphas_betas = np.zeros(2*n_minimax)

    xdata = 10**(np.logspace(0,np.log(np.log10(R_minimax)),n_x))
    ydata = np.zeros(n_x)

    alphas_betas_init = np.logspace(-3,-2,2*n_minimax)
    alphas_betas_L2_opt, alphas_betas_conv = curve_fit(eta, xdata, ydata, p0=alphas_betas_init)
    alphas_betas_E = np.append(alphas_betas_L2_opt,1)

    E_old = alphas_betas_E[-1]*2
    i = 0
    while (alphas_betas_E[-1]/E_old < 1-eps_diff or alphas_betas_E[-1] > E_old):

        E_old = alphas_betas_E[-1]
        extrema_x = np.append(xdata[0], xdata[argrelextrema(eta_plotting(xdata,alphas_betas_E[0:np.size(alphas_betas_E)-1]), np.greater)[0]])
        extrema_x = np.append(extrema_x, xdata[argrelextrema(eta_plotting(xdata,alphas_betas_E[0:np.size(alphas_betas_E)-1]), np.less)[0]])
        alphas_betas_E[np.size(alphas_betas_E)-1] = np.average(np.abs(eta_plotting(extrema_x,alphas_betas_L2_opt)))

        alphas_betas_E = fsolve(eta_for_alphas_betas_E_update, x0=alphas_betas_E, args=extrema_x)
        i += 1
        print("iteration =", i, "E=",  alphas_betas_E[-1])

    fig1, (axis1) = pl.subplots(1,1)
    axis1.set_xlim((0.8,R_minimax))
    axis1.semilogx(xdata,eta_plotting(xdata,alphas_betas_L2_opt))
    axis1.semilogx(xdata,eta_plotting(xdata,alphas_betas_E))
    axis1.semilogx(extrema_x, eta_plotting(extrema_x,alphas_betas_L2_opt), "x")
    axis1.semilogx([0.8,R_minimax], [alphas_betas_E[-1],alphas_betas_E[-1]])
    axis1.semilogx([0.8,R_minimax], [-alphas_betas_E[-1],-alphas_betas_E[-1]])

    pl.show()

def eta(x, *params):
    return 1/(2*x) - (np.exp(-outer(x,params[0:np.size(params)//2]))).dot(params[np.size(params)//2:])

def eta_plotting(x, *params):
    params_1d = np.transpose(params)[:,0]
    return 1/(2*x) - (np.exp(-outer(x,params_1d[0:np.size(params)//2]))).dot(params_1d[np.size(params)//2:(np.size(params)//2)*2])

def eta_for_alphas_betas_E_update(x, *params):
    params_1d = np.transpose(params)[:,0]
    size_params = np.size(params_1d)
    size_x = np.size(x)
    E = np.empty(size_x)
    E[0:size_x//2+1] = x[size_params-1]
    E[size_x//2+1:] = -x[size_params-1]

    return 1/(2*params_1d) - (np.exp(-outer(params_1d,x[0:np.size(x)//2]))).dot(x[np.size(x)//2:np.size(x)-1]) - E

if __name__ == "__main__":
    main()

