#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:13:43 2018

@author: cemma
"""

# Plotting functions for Neural network
# Plotting the pv data
def plot_pv_data_v2(X):
    import matplotlib.pyplot as plt
    plt.subplot(231)
    plt.plot(X[:,3])
    plt.ylabel('L1S amp [kV]')

    plt.subplot(232)
    plt.plot(X[:,4])
    plt.ylabel('L1S phase [deg]')    

    plt.subplot(233)
    plt.plot(X[:,7])
    plt.ylabel('BC1 current [kA]')    

    plt.subplot(234)
    plt.plot(X[:,5])
    plt.ylabel('L1X amp [kV]')    

    plt.subplot(235)
    plt.plot(X[:,6])
    plt.ylabel('L1X phase [deg]')    

    plt.subplot(236)
    plt.plot(X[:,20]*10**-3)
    plt.ylabel('BC2 current [kA]')
    
    plt.tight_layout()
    plt.show()

# Plotting the predicted vs actual current profiles    
def plot_pred_vs_actual_v2(tztrain,Iz_scaled,predict_Iz):
    import matplotlib.pyplot as plt
    import numpy as np
    f,axarr = plt.subplots(2,3)
    for i in range(6):        
        ns = int(Iz_scaled[:,0].shape+Iz_scaled[:,0].shape*(0.5*(2*np.random.rand(1,1))-1));                
        curr_integral = np.trapz(Iz_scaled[ns,:],x=tztrain[ns,:]*1e-15);
        conv_factor = 180e-12/curr_integral*1e-3;
        if i<3:
            axarr[0,i].plot(tztrain[ns,:],Iz_scaled[ns,:]*conv_factor,label='Actual');
            axarr[0,i].plot(tztrain[ns,:],predict_Iz[ns,:]*conv_factor,'r--',label='Predicted');

        else:
            axarr[1,i-3].plot(tztrain[ns,:],Iz_scaled[ns,:]*conv_factor,label='Actual');
            axarr[1,i-3].plot(tztrain[ns,:],predict_Iz[ns,:]*conv_factor,'r--',label='Predicted');
 
        for ax in axarr.flat:
            ax.set(xlabel='t [fs]', ylabel='Current [kA]')
    f.tight_layout()            
    plt.show()
    
def plot_lps_vs_prediction_contour(lps,predicted_lps,X,Y):
    import matplotlib.pyplot as plt
    import numpy as np    
    ns = int(lps.shape[2]+lps.shape[2]*(0.5*(2*np.random.rand(1,1))-1));        
    fig, (ax, ax2) = plt.subplots(1,2)
    ax.contourf(X,Y,lps[:,:,ns],cmap = plt.cm.viridis)
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Energy Deviation [MeV]')
    ax2.contourf(X,Y,predicted_lps[:,:,ns],cmap = plt.cm.viridis)
    ax2.set_xlabel('Time [fs]')
    plt.show()
    
def plot_lps_vs_prediction_v2(lps,predicted_lps,x,y):
    import matplotlib.pyplot as plt
    import numpy as np    
    ns = int(lps.shape[2]+lps.shape[2]*(0.5*(2*np.random.rand(1,1))-1));        
    fig, (ax, ax2) = plt.subplots(1,2)
    ax.imshow(lps[:,:,ns],extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]),interpolation = 'none')
    ax.set_aspect(3)
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Energy Deviation [MeV]')
    ax2.imshow(predicted_lps[:,:,ns],extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]))
    ax2.set_xlabel('Time [fs]')
    ax2.set_aspect(3)
    plt.show()
    
def plot_lps_and_current(lps,x,y,tztrain,Iz_scaled):
    import matplotlib.pyplot as plt
    import numpy as np    
    ns = int(lps.shape[2]+lps.shape[2]*(0.5*(2*np.random.rand(1,1))-1));        
    curr_integral = np.trapz(Iz_scaled[ns,:],x=tztrain[ns,:]*1e-15);
    conv_factor = 180e-12/curr_integral*1e-3;
    fig, (ax, ax2) = plt.subplots(1,2,figsize=(10,3))
    ax.imshow(lps[:,:,ns],extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]),aspect = "auto")
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Energy Deviation [MeV]')
    ax2.plot(tztrain[ns,:],Iz_scaled[ns,:]*conv_factor)
    ax2.set_xlabel('Time [fs]')
    ax2.set_ylabel('Current [kA]')
    ax2.set_aspect("auto")
    plt.show()
    
def plot_lps_and_current_w_prediction(lps,predicted_lps,x,y,tztrain,Iz_scaled,predict_Iz):
    import matplotlib.pyplot as plt
    import numpy as np    
    ns = int(lps.shape[2]+lps.shape[2]*(0.5*(2*np.random.rand(1,1))-1));        
    curr_integral = np.trapz(Iz_scaled[ns,:],x=tztrain[ns,:]*1e-15);
    conv_factor = 180e-12/curr_integral*1e-3;
    fig, (ax, ax2, ax3) = plt.subplots(1,3,figsize=(10,3))
    ax.imshow(lps[:,:,ns],extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]),aspect = "auto")
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Energy Deviation [MeV]')
    ax3.plot(tztrain[ns,:],Iz_scaled[ns,:]*conv_factor,label='XTCAV')
    ax3.plot(tztrain[ns,:],predict_Iz[ns,:]*conv_factor,label='Predicted')
    ax3.set_xlabel('Time [fs]')
    ax3.set_ylabel('Current [kA]')
    ax2.set_aspect("auto")
    ax2.imshow(predicted_lps[:,:,ns],extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]),aspect = "auto")
    ax2.set_xlabel('Time [fs]')
    ax2.set_aspect("auto")
    plt.show()