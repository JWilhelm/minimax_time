import numpy as np
import matplotlib.pyplot as pl
from matplotlib import patches
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from numpy import dot, outer

def main():
    
    # Set parameters
    n_minimax = 10                    # Number of minimax points
    R_minimax = 10**8                 # Range of the minimax approximation
    n_x       = 500                   # total number of points on the x-axis for optimization

    # initialization: 
    ###############################################################################################
    alphas_betas = np.zeros(2*n_minimax)

    xdata = np.logspace(0,np.log10(R_minimax),n_x)
    ydata = np.zeros(n_x)

    alphas_betas_init = np.logspace(-3,-2,2*n_minimax)

    alphas_betas_L2_opt, alphas_betas_conv = curve_fit(eta, xdata, ydata, p0=alphas_betas_init)

    i = 0
    while i < 1:

        max_x = xdata[argrelextrema(eta_plotting(xdata,alphas_betas_L2_opt), np.greater)[0]]
        min_x = xdata[argrelextrema(eta_plotting(xdata,alphas_betas_L2_opt), np.less)[0]]

        if eta_plotting(xdata[0],alphas_betas_L2_opt) > 0:
            max_x = np.append(xdata[0], max_x)
        else:
            min_x = np.append(xdata[0], min_x)

        i += 1


    print ("maxm =", max_x)
    print ("eta_plotting(max_x,alphas_betas_L2_opt) =", eta_plotting(max_x,alphas_betas_L2_opt))
    print ("minm =", min_x)
    print ("eta_plotting(min_x,alphas_betas_L2_opt) =", eta_plotting(min_x,alphas_betas_L2_opt))


    fig1, (axis1) = pl.subplots(1,1)
    axis1.set_xlim((1,R_minimax))
    axis1.semilogx(xdata,eta_plotting(xdata,alphas_betas_L2_opt))
    axis1.semilogx(max_x, eta_plotting(max_x,alphas_betas_L2_opt), "x")
    axis1.semilogx(min_x, eta_plotting(min_x,alphas_betas_L2_opt), "x")

    pl.show()



def eta(x, *params):
    return 1/(2*x) - (np.exp(-outer(x,params[0:np.size(params)//2]))).dot(params[np.size(params)//2:])

def eta_plotting(x, *params):
    params_1d = np.transpose(params)[:,0]
    return 1/(2*x) - (np.exp(-outer(x,params_1d[0:np.size(params)//2]))).dot(params_1d[np.size(params)//2:])

if __name__ == "__main__":
    main()

