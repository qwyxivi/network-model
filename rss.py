import numpy as np
from scipy.stats import norm as normal_dist
from scipy.optimize import curve_fit

def unicorn(low, high, N):
    return np.random.random(size= N)*(high-low)+low

def RSS_integrate(fxn, samples=1000, estimator_block = 40, low=-1.0, high=1.0, depth=0):
    """returns integral, and variance of error"""
    print("  "*depth,low, high, samples)
    if samples<=1.5*estimator_block:
        values = [fxn(x) for x in unicorn(low, high, samples+2)]
        return np.mean(values)*(high-low), (np.std(values, ddof=1)/np.sqrt(samples+2)*(high-low))**2
    LL, LH = low, (low+high)/2
    RL, RH = (low+high)/2, high
    left = [fxn(x) for x in unicorn(LL, LH, estimator_block//2)]
    right = [fxn(x) for x in unicorn(RL, RH, estimator_block//2)]
    remainder = samples - estimator_block
    sleft = np.std(left, ddof=1)
    sright = np.std(right, ddof=1)
    slval = np.mean(left)*(LH-LL)
    srval = np.mean(right)*(RH-RL)
    slerr = (sleft/np.sqrt(estimator_block/2)*(LH-LL))**2
    srerr = (sright/np.sqrt(estimator_block/2)*(RH-RL))**2
    delta = (sleft+sright)/8
    allocleft = int(round(remainder*(sleft+delta)/(sleft+sright+2*delta)))
    allocright = int(round(remainder*(sright+delta)/(sleft+sright+2*delta)))
    Lval, Lerr = RSS_integrate(fxn, samples=allocleft, estimator_block = estimator_block, low=LL, high=LH, depth=depth+1)
    Rval, Rerr = RSS_integrate(fxn, samples=allocright, estimator_block = estimator_block, low=RL, high=RH, depth=depth+1)
    L = Lval*(slerr/(slerr+Lerr))+slval*(Lerr/(slerr+Lerr))
    LE = slerr*Lerr/(slerr+Lerr)
    R = Rval*(srerr/(srerr+Rerr))+srval*(Rerr/(srerr+Rerr))
    RE = srerr*Rerr/(srerr+Rerr)
    print("  "*depth,"<-", L+R)
    return L+R, LE+RE

def peak(x, width, height):
    return height*np.exp(-(x/width)**2)

def peak_fit(ansatz, rand_f, scale=0.5, samples=40, debug=False):
    datax = np.linspace(-scale, scale, samples+1)
    temp = [rand_f(xi) for xi in datax]
    datay = np.array([np.real(tempi[0]) for tempi in temp])
    datayerr = np.array([np.real(tempi[2]) for tempi in temp])
    if debug: print(datayerr)
    popc, pcov = curve_fit(ansatz, datax, datay, sigma=datayerr, absolute_sigma=True, p0=[scale/2, 1.0])
    x = np.linspace(1.1*min(datax)-0.1*max(datax), 1.1*max(datax)-0.1*min(datax), 101)
    y = ansatz(x, *popc)
    return popc, pcov, x, y, datax, datay, datayerr
