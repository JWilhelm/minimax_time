import numpy as np
import matplotlib.pyplot as pl
from matplotlib import patches
from scipy.optimize import curve_fit

def main():
    
    # Set parameters
    n_minimax = 20                    # Number of minimax points
    R_minimax = 10**8                 # Range of the minimax approximation
    n_x       = 500                   # total number of points on the x-axis for optimization

    # initialization: 
    ###############################################################################################
    i_values_Chebyshev = np.linspace(0,n_minimax+1)*np.pi/(n_minimax+1)

    print("i_values_Chebyshev", i_values_Chebyshev)

    x_extrema = np.array((R_minimax+1)/2 - (R_minimax-1)/2*np.cos(i_values_Chebyshev))

    alphas = np.zeros(n_minimax)

    betas = np.zeros(n_minimax)

    xdata = np.logspace(0,np.log10(R_minimax),n_x)
    ydata = np.zeros(n_x)

    print("xdata", xdata)

    popt, pcov = curve_fit(eta, xdata, ydata)


    print("x_extrema", x_extrema)


def eta(x, alphas, betas):
   return 1/(2*x) - np.dot(betas, np.exp(-x*alphas))


if __name__ == "__main__":
    main()

