"""
Modelling Saccharomyces cerevisiae for the production of fermented beverages

Paul A. Valle, Yolocuauhtli Salazar, Luis N. Coria, Nicolas O. Soto-Cruz, Jesus B. Paez-Lerma

Postgraduate Program in Engineering Sciences, BioMath Research Group, Tecnologico Nacional de Mexico/IT Tijuana, Blvd. Alberto Limon Padilla s/n, Tijuana 22454, Mexico
Postgraduate Program in Engineering, Tecnologico Nacional de Mexico/IT Durango, Blvd. Felipe Pescador 1830 Ote., Durango 34080, Mexico
Departament Chemical and Biochemical Engineering, Tecnologico Nacional de Mexico/IT Durango, Blvd. Felipe Pescador 1830 Ote., Durango 34080, Mexico
"""
import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import scipy.stats
import statistics as st
import warnings; warnings.filterwarnings("ignore")
#%% Functions
def plotdataraw(t,x,y,z):
    #plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size':11})
    fig = plt.figure(figsize = (10,4))
        
    ax1 = fig.add_subplot(1,3,1) 
    ax1.plot(t,x[:,0],linestyle = '-', lw = 0.5, marker = 'o', ms = 3, color = [0.12,0.41,0.55], label = '$x_1(t)$')
    ax1.plot(t,x[:,1],linestyle = '-', lw = 0.5, marker = 'o', ms = 3, color = [0.25,0.51,0.47], label = '$x_2(t)$')
    ax1.plot(t,x[:,2],linestyle = '-', lw = 0.5, marker = 'o', ms = 3, color = [0.91,0.66,0.07], label = '$x_3(t)$')   
    ax1.set_xlabel('$t$ $[h]$')
    ax1.set_ylabel('$x(t)$ $[g/L]$')
    ax1.set_xlim([-2,75])
    xticks = np.arange(0,73,8)
    ax1.set_xticks(xticks)
    ax1.set_ylim([-1,15])
    yticks = np.arange(-1,16,1)
    ax1.set_yticks(yticks)
    ax1.legend(bbox_to_anchor = (1, 1), fontsize = 10, title="$Glucose$", frameon = True)
    
    ax2 = fig.add_subplot(1,3,2) 
    ax2.plot(t,y[:,0],linestyle = '-', lw = 0.5, marker = 'o', ms = 3, color = [0.12,0.41,0.55], label = '$y_1(t)$')
    ax2.plot(t,y[:,1],linestyle = '-', lw = 0.5, marker = 'o', ms = 3, color = [0.25,0.51,0.47], label = '$y_2(t)$')
    ax2.plot(t,y[:,2],linestyle = '-', lw = 0.5, marker = 'o', ms = 3, color = [0.91,0.66,0.07], label = '$y_3(t)$')   
    ax2.set_xlabel('$t$ $[h]$')
    ax2.set_ylabel('$y(t)$ $[g/L]$')
    ax2.set_xlim([-2,75])
    xticks = np.arange(0,73,8)
    ax2.set_xticks(xticks)
    ax2.set_ylim([0,110])
    yticks = np.arange(0,111,10)
    ax2.set_yticks(yticks)
    ax2.legend(bbox_to_anchor = (1, 1), fontsize = 10, title="$Fructose$", frameon = True)
    
    ax3 = fig.add_subplot(1,3,3) 
    ax3.plot(t,z[:,0],linestyle = '-', lw = 0.5, marker = 'o', ms = 3, color = [0.12,0.41,0.55], label = '$z_1(t)$')
    ax3.plot(t,z[:,1],linestyle = '-', lw = 0.5, marker = 'o', ms = 3, color = [0.25,0.51,0.47], label = '$z_2(t)$')
    ax3.plot(t,z[:,2],linestyle = '-', lw = 0.5, marker = 'o', ms = 3, color = [0.91,0.66,0.07], label = '$z_3(t)$')   
    ax3.set_xlabel('$t$ $[h]$')
    ax3.set_ylabel('$z(t)$ $[g/L]$')
    ax3.set_xlim([-2,75])
    xticks = np.arange(0,73,8)
    ax3.set_xticks(xticks)
    ax3.set_ylim([0,65])
    yticks = np.arange(0,66,5)
    ax3.set_yticks(yticks)
    ax3.legend(bbox_to_anchor = (1, 0.35), fontsize = 10, title="$Ethanol$", frameon = True)
    
    fig.tight_layout()
    fig.savefig('python_dataraw.png', dpi = 600)
    fig.savefig('python_dataraw.pdf')

def plotdata(t,x,y,z):
    #plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size':11})
    fig = plt.figure(figsize = (10,4))
        
    ax1 = fig.add_subplot(1,3,1) 
    ax1.plot(t,x,linestyle = 'None', marker = 'x', ms = 3, color = [0.3,0.5,0.2], label = '$x(t)$') 
    ax1.set_xlabel('$t$ $[h]$')
    ax1.set_ylabel('$x(t)$ $[g/L]$')
    ax1.set_xlim([-2,75])
    xticks = np.arange(0,73,8)
    ax1.set_xticks(xticks)
    ax1.set_ylim([-1,15])
    yticks = np.arange(-1,16,1)
    ax1.set_yticks(yticks)
    ax1.legend(bbox_to_anchor = (1, 1), fontsize = 10, title="$Glucose$", frameon = True)
    
    ax2 = fig.add_subplot(1,3,2) 
    ax2.plot(t,y,linestyle = 'None', marker = 'x', ms = 3, color = [0.5,0.1,0.05], label = '$y(t)$') 
    ax2.set_xlabel('$t$ $[h]$')
    ax2.set_ylabel('$y(t)$ $[g/L]$')
    ax2.set_xlim([-2,75])
    xticks = np.arange(0,73,8)
    ax2.set_xticks(xticks)
    ax2.set_ylim([0,110])
    yticks = np.arange(0,111,10)
    ax2.set_yticks(yticks)
    ax2.legend(bbox_to_anchor = (1, 1), fontsize = 10, title="$Fructose$", frameon = True)
    
    ax3 = fig.add_subplot(1,3,3) 
    ax3.plot(t,z,linestyle = 'None', marker = 'x', ms = 3, color = [0,0.25,0.4], label = '$z(t)$')
    ax3.set_xlabel('$t$ $[h]$')
    ax3.set_ylabel('$z(t)$ $[g/L]$')
    ax3.set_xlim([-2,75])
    xticks = np.arange(0,73,8)
    ax3.set_xticks(xticks)
    ax3.set_ylim([0,65])
    yticks = np.arange(0,66,5)
    ax3.set_yticks(yticks)
    ax3.legend(bbox_to_anchor = (1, 0.2), fontsize = 10, title="$Ethanol$", frameon = True)
    
    fig.tight_layout()
    fig.savefig('python_data.png', dpi = 600)
    fig.savefig('python_data.pdf')

def plotfit(t,xo,yo,zo,xa,ya,za):
    #plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size':11})
    fig = plt.figure(figsize = (10,4))
        
    ax1 = fig.add_subplot(1,3,1)
    ax1.axhline(y = 0, color = 'k', linestyle = '--', lw = 0.75)
    ax1.plot(t,xo,linestyle = 'None', marker = 'x', ms = 3, color = [0.3,0.5,0.2], label = '$x_o(t)$')
    ax1.plot(t,xa,linestyle = '-', lw = 1, color = [0.5,0.5,0.5], label = '$x_a(t)$') 
    ax1.set_xlabel('$t$ $[h]$')
    ax1.set_ylabel('$x(t)$ $[g/L]$')
    ax1.set_xlim([-2,75])
    xticks = np.arange(0,73,8)
    ax1.set_xticks(xticks)
    ax1.set_ylim([-1,15])
    yticks = np.arange(-1,16,1)
    ax1.set_yticks(yticks)
    ax1.legend(bbox_to_anchor = (1, 1), fontsize = 10, title="$Glucose$", frameon = True)
    
    ax2 = fig.add_subplot(1,3,2)
    ax2.axhline(y = 0, color = 'k', linestyle = '--', lw = 0.75)
    ax2.plot(t,yo,linestyle = 'None', marker = 'x', ms = 3, color = [0.5,0.1,0.05], label = '$y_o(t)$')
    ax2.plot(t,ya,linestyle = '-', lw = 1, color = [0.5,0.5,0.5], label = '$y_a(t)$')
    ax2.set_xlabel('$t$ $[h]$')
    ax2.set_ylabel('$y(t)$ $[g/L]$')
    ax2.set_xlim([-2,75])
    xticks = np.arange(0,73,8)
    ax2.set_xticks(xticks)
    ax2.set_ylim([-10,110])
    yticks = np.arange(-10,111,10)
    ax2.set_yticks(yticks)
    ax2.legend(bbox_to_anchor = (1, 1), fontsize = 10, title="$Fructose$", frameon = True)
    
    ax3 = fig.add_subplot(1,3,3) 
    ax3.plot(t,zo,linestyle = 'None', marker = 'x', ms = 3, color = [0,0.25,0.4], label = '$z_o(t)$')
    ax3.plot(t,za,linestyle = '-', lw = 1, color = [0.5,0.5,0.5], label = '$z_a(t)$') 
    ax3.set_xlabel('$t$ $[h]$')
    ax3.set_ylabel('$z(t)$ $[g/L]$')
    ax3.set_xlim([-2,75])
    xticks = np.arange(0,73,8)
    ax3.set_xticks(xticks)
    ax3.set_ylim([0,65])
    yticks = np.arange(0,66,5)
    ax3.set_yticks(yticks)
    ax3.legend(bbox_to_anchor = (1, 0.25), fontsize = 10, title="$Ethanol$", frameon = True)
    
    fig.tight_layout()
    fig.savefig('python_results.png', dpi = 600)
    fig.savefig('python_results.pdf')

def variant(to,fo,rho0,rho):
    x0,y0,z0 = xo[0],yo[0],zo[0]
    p1,p2,p3 = rho[0],rho[1],rho[2]
    
    def model(t,p4,p5,p6):
        dt = 1E-3
        n = round(max(t)/dt)
        time = np.linspace(0,max(t),n+1)
        x = np.zeros(n+1); x[0] = x0
        y = np.zeros(n+1); y[0] = y0
        z = np.zeros(n+1); z[0] = z0
        
        def f(x,y,z):
            dx = - p4*x*z - p1*x;
            dy = - p5*y*z - p2*y;
            dz = + p6*(x + y)*z - p3*z;
            return dx,dy,dz
        
        for i in range(0,n):
            fx,fy,fz = f(x[i],y[i],z[i])
            xn = x[i] + fx*dt
            yn = y[i] + fy*dt
            zn = z[i] + fz*dt
            fxn,fyn,fzn = f(xn,yn,zn)
            x[i+1] = x[i] + (fx + fxn)*dt/2
            y[i+1] = y[i] + (fy + fyn)*dt/2
            z[i+1] = z[i] + (fz + fzn)*dt/2
        
        xi,yi,zi = np.zeros(len(t)),np.zeros(len(t)),np.zeros(len(t))  
        for i in range(0,len(t)):
            k = abs(time-t[i]) < 1E-4
            xi[i] = x[k]
            yi[i] = y[k]
            zi[i] = z[k]
        
        fi = list(xi) + list(yi) + list(zi)
        return fi
    
    npar = len(rho0)
    low = np.ones(npar)*m.inf*(-1)
    sup = np.ones(npar)*m.inf

    Estimate,cov = curve_fit(model, to, fo, rho0, bounds = (low,sup))

    fa = model(to,Estimate[0],Estimate[1],Estimate[2])
    return fa,Estimate,cov

def biostatistics(Estimate,cov,fo,fa,xo,yo,zo,xa,ya,za):
    alpha = 0.05
    dof = len(fo) - len(Estimate)
    tval = scipy.stats.t.ppf(q = 1-alpha/2, df = dof)
    SE = np.diag(cov)**(0.5)
    pvalue = 2*scipy.stats.t.sf(np.abs(Estimate/SE), dof)
    MoE = SE*tval
    CI95 = np.zeros([len(Estimate),2])
    for i in range(0,len(Estimate)):
        CI95[i,0] = Estimate[i]-MoE[i]
        CI95[i,1] = Estimate[i]+MoE[i] 
   
    print('\nSample size (n): ',len(fo))
    print('Parameters to be estimated (pars): ',len(Estimate))
    print('Degrees of freedom (dof): ',dof)
    print('alpha: ', alpha)
    print('t-Student value: ',tval)
    print('\nFixed parameters values')
    print('rho1: ',rho1)
    print('rho2: ',rho2)
    print('rho3: ',rho3,'\n')
   
    Parameter = ['rho4','rho5','rho6']
    df = pd.DataFrame(list(zip(Parameter,Estimate,SE,MoE,CI95,pvalue)),
                      columns = ['Parameter','Estimate','SE','MoE','CI95','pValue'])
    print(df.to_string(index = False))
    
    def rsquared(po,pa):
        n = len(po)
        p = len(Estimate)
        SSE,SST = 0,0
        for i in range(0,n):
            SSE += (po[i] - pa[i])**2
            SST += (po[i] - st.mean(po))**2
            R2 = 1 - SSE/SST
        
        R2adj = 1 - ((n - 1)/(n - p))*SSE/SST
        return R2,R2adj
    
    print('\nR-squared [adjusted]: ',rsquared(fo,fa)[1])
    print('\nR-squared results for each variable')
    print('R-squared [x(t)]: ',rsquared(xo,xa)[0])
    print('R-squared [y(t)]: ',rsquared(yo,ya)[0])
    print('R-squared [z(t)]: ',rsquared(zo,za)[0])  
#%% Extracting the data
data = pd.read_csv('data.csv', header = None)

To = data.iloc[:,0]; to = To.to_numpy()

X1 = data.iloc[:,4]; x1 = X1.to_numpy() 
X2 = data.iloc[:,5]; x2 = X2.to_numpy()
X3 = data.iloc[:,6]; x3 = X3.to_numpy()
xo = np.mean([x1,x2,x3], axis = 0); xo = gaussian_filter(xo, sigma = 0.62)

Y1 = data.iloc[:,7]; y1 = Y1.to_numpy() 
Y2 = data.iloc[:,8]; y2 = Y2.to_numpy()
Y3 = data.iloc[:,9]; y3 = Y3.to_numpy()
yo = np.mean([y1,y2,y3], axis = 0); yo = gaussian_filter(yo, sigma = 0.62)

Z1 = data.iloc[:,1]; z1 = Z1.to_numpy() 
Z2 = data.iloc[:,2]; z2 = Z2.to_numpy()
Z3 = data.iloc[:,3]; z3 = Z3.to_numpy() 
zo = np.mean([z1,z2,z3], axis = 0); zo = gaussian_filter(zo, sigma = 0.62)

fo = np.concatenate((xo,yo,zo))
#%% Plotting the data
xr = np.stack((x1,x2,x3), axis = 1)
yr = np.stack((y1,y2,y3), axis = 1)
zr = np.stack((z1,z2,z3), axis = 1)
plotdataraw(to,xr,yr,zr)
plotdata(to,xo,yo,zo)
#%% Fitting the model, Biostatistics and In silico experimentation
rho0 = [0.001,0.001,0.001]
rho1,rho2 = np.log(2)/840960,np.log(2)/1680
rho3 = max([rho1,rho2])*1.1
rho = (rho1,rho2,rho3)

fa,Estimate,cov = variant(to,fo,rho0,rho)

sys = np.array_split(fa,3) 
xa,ya,za = sys[0],sys[1],sys[2]

biostatistics(Estimate,cov,fo,fa,xo,yo,zo,xa,ya,za)

plotfit(to,xo,yo,zo,xa,ya,za)