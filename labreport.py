import matplotlib.pyplot as plt
import scipy.optimize
import pandas as pd
import numpy as np
import os

__plotTemplate = ""
plotTemplatePath = "C:/Users/david/OneDrive/Studium/Praktika/Jupyter/plotTemplate.tex"
dataDir = "output/data/"
plotDir = "output/plots/"
curveOverhead = 0.2

def __loadPlotTemplate():
    global __plotTemplate
    if not __plotTemplate:
        with open(plotTemplatePath,'r') as file:
            __plotTemplate = file.read()
    return __plotTemplate

def __WriteLatexPlot(data,fit,texName):
    template = __loadPlotTemplate()
    xmode = "linear"
    ymode = "linear"
    template = template.replace("XMODE",xmode)
    template = template.replace("YMODE",ymode)
    template = template.replace("DATAPLOT",f"{texName}Plot")
    template = template.replace("DATAFIT",f"{texName}Fit")

    os.makedirs(dataDir, exist_ok=True)
    data.to_csv(dataDir + texName + "Plot.tex",index = False,sep=' ')
    fit.to_csv(dataDir + texName + "Fit.tex",index = False,sep=' ')

    os.makedirs(plotDir, exist_ok=True)
    file = open(plotDir + texName + ".tex",'w')
    file.write(template)
    file.close()

def funcFit(f,x,y,texName="",verbose= True,p0 = None, regionMin = None, regionMax = None):
    '''
    Fits to f(x,p1,p2,...)\n

    p0 = Array-like initial parameters, all 1 if p0=None
    texName = Name of Latex files outputted to /output/data/ and /output/plots if not empty

    Return optimal (p1, p2,...), standard deviation (d1, d2,...)
    '''

    (popt,pcov)=scipy.optimize.curve_fit(f,x[regionMin:regionMax],y[regionMin:regionMax],p0)
    psdev = np.sqrt(np.diag(pcov))

    minx = min(x[regionMin:regionMax])
    maxx = max(x[regionMin:regionMax])
    rangex = maxx-minx
    xlow = minx-rangex*0.2
    
    xhigh = maxx+rangex*0.2

    data = pd.DataFrame({'X':x,'Y':y})
    fit = pd.DataFrame({'X':np.linspace(xlow,xhigh,200)})
    fit['Y'] = f(fit.X,*popt)

    if verbose:
        for n,(p,sd) in enumerate(zip(popt,psdev)):
            print(f"p{n}:\t{p}\t±{sd}")
        plt.figure()
        plt.scatter(data.X,data.Y)
        plt.plot(fit.X,fit.Y)
        plt.show()

    if texName:
            __WriteLatexPlot(data,fit,texName)
    return popt,psdev

def linFit(x,y,texName = "",verbose=True):

        '''
        Fit to m*x+t\n

        texName = Name of Latex files outputted to /output/data/ and /output/plots if not empty

        Return m, t, standard_deviation_m, standard_deviation_t
        '''

        ((m,t),((var_m,_),(_,var_t))) = np.polyfit(x,y,1,cov = True)
        sd_m = np.sqrt(var_m)
        sd_t = np.sqrt(var_t)
        minx = min(x)
        maxx = max(x)
        rangex = maxx-minx
        xlow = minx-rangex*0.2
        xhigh = maxx+rangex*0.2
        data = pd.DataFrame({'X':x,'Y':y})
        fit = pd.DataFrame({'X':[xlow,xhigh],'Y':[xlow*m+t,xhigh*m+t]})

        if verbose:
            print(f"m*x+t: {m} * x + {t}")
            print(f"Standard Deviation m: {sd_m}")
            print(f"Standard Deviation t: {sd_t}")
            plt.figure()
            plt.scatter(data.X,data.Y)
            plt.plot(fit.X,fit.Y)
            plt.show()

        if texName:
            __WriteLatexPlot(data,fit,texName)
        return m,sd_m,t,sd_t