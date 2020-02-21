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

    print("alphas_betas_init =", alphas_betas_init)
    print("shape alphas_betas_init =", np.shape(alphas_betas_init))

    alphas_betas_L2_opt, alphas_betas_conv = curve_fit(eta, xdata, ydata, p0=alphas_betas_init)

    i = 0
    while i < 10:

#        maxm = argrelextrema(y, np.greater)
#        minm = argrelextrema(y, np.less)
        i += 1



    print("alphas_betas_L2_opt =", alphas_betas_L2_opt)
    print("shape alphas_betas_L2_opt =", np.shape(alphas_betas_L2_opt))


    fig1, (axis1) = pl.subplots(1,1)
    axis1.set_xlim((1,R_minimax))
    axis1.set_ylim((10e-15,1))
    axis1.loglog(xdata,np.abs(eta(xdata,alphas_betas_L2_opt)))

    pl.show()



def eta(x, *params):
    print("np.shape(params)",np.shape(params))
#    print("np.size(params)",np.size(params))
#    print("np.size(params)//2",np.size(params)//2)
#    print("np.shape =", np.shape(outer(x,params[0:np.size(params)//2])))
#    print("params[0:np.size(params)//2] =", params[0:np.size(params)//2])

    return 1/(2*x) - (np.exp(-outer(x,params[0:np.size(params)//2]))).dot(params[np.size(params)//2:])

if __name__ == "__main__":
    main()

