import numpy as np
import matplotlib.pyplot as pl
from matplotlib import patches
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

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
    while i < 10:

#        maxm = argrelextrema(y, np.greater)
#        minm = argrelextrema(y, np.less)
        i += 1



    print("alphas_betas_L2_opt =", alphas_betas_L2_opt)

    fig1, (axis1) = pl.subplots(1,1)
    axis1.set_xlim((1,R_minimax))
    axis1.set_ylim((10e-15,1))
    axis1.loglog(xdata,np.abs(eta(xdata,alphas_betas_L2_opt)))
#        alphas_betas_L2_opt[0],
#        alphas_betas_L2_opt[1],
#        alphas_betas_L2_opt[2],
#        alphas_betas_L2_opt[3],
#        alphas_betas_L2_opt[4],
#        alphas_betas_L2_opt[5],
#        alphas_betas_L2_opt[6],
#        alphas_betas_L2_opt[7],
#        alphas_betas_L2_opt[8],
#        alphas_betas_L2_opt[9],
#        alphas_betas_L2_opt[10],
#        alphas_betas_L2_opt[11],
#        alphas_betas_L2_opt[12],
#        alphas_betas_L2_opt[13],
#        alphas_betas_L2_opt[14],
#        alphas_betas_L2_opt[15],
#        alphas_betas_L2_opt[16],
#        alphas_betas_L2_opt[17],
#        alphas_betas_L2_opt[18],
#        alphas_betas_L2_opt[19],
#        alphas_betas_L2_opt[20],
#        alphas_betas_L2_opt[21],
#        alphas_betas_L2_opt[22],
#        alphas_betas_L2_opt[23],
#        alphas_betas_L2_opt[24],
#        alphas_betas_L2_opt[25],
#        alphas_betas_L2_opt[26],
#        alphas_betas_L2_opt[27],
#        alphas_betas_L2_opt[28],
#        alphas_betas_L2_opt[29],
#        )))

    pl.show()



def eta(x, *params):
    return 1/(2*x) - np.sum(params[np.size(params)//2:]*np.exp(-x*params[0:np.size(params)//2]))
#   - beta_1*np.exp(-x*alpha_1) \
#   - beta_2*np.exp(-x*alpha_2) \
#   - beta_3*np.exp(-x*alpha_3) \
#   - beta_4*np.exp(-x*alpha_4) \
#   - beta_5*np.exp(-x*alpha_5) \
#   - beta_6*np.exp(-x*alpha_6) \
#   - beta_7*np.exp(-x*alpha_7) \
#   - beta_8*np.exp(-x*alpha_8) \
#   - beta_9*np.exp(-x*alpha_9) \
#   - beta_10*np.exp(-x*alpha_10) \
#   - beta_11*np.exp(-x*alpha_11) \ 
#   - beta_12*np.exp(-x*alpha_12) \
#   - beta_13*np.exp(-x*alpha_13) \
#   - beta_14*np.exp(-x*alpha_14) \
#   - beta_15*np.exp(-x*alpha_15) \

if __name__ == "__main__":
    main()

