#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:57:31 2018

@author: emanuel
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import skfmm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline,RegularGridInterpolator
#from scipy.integrate import solve_ivp
import datetime
try:
    import pyekfmm
except:
    pass

# velocity field v has to be regular of shape (ny,nx), (lats,lons)
#scikit-fmm only works for regular Cartesian grids, but grid cells may have
# a different (uniform) length in each dimension
def test3d():
    #%% Test 3D
    x = np.arange(-500,600,10,dtype=float)
    y = np.arange(0,1000,10,dtype=float)
    z = np.hstack((np.arange(10),np.arange(10,22,2),[22.,25.,28.,32.,36.,40.,
                                                     45.,50.,55.,60.,65.,70.,
                                                     80.,90.,100.,120.,140.,
                                                     160.,180.,200.,230.,260,
                                                     300.,350.,400,450.,500]))
    z = np.arange(0,500,10)
    X,Y,Z = np.meshgrid(x,y,z)
    source = (-410,112,420)
    receiver1 = (333,840,0)
    receiver2 = (-105,45,0)
    receivers = [receiver1,receiver2]
    
    vprofile = np.ones_like(z)*2.
    vprofile[z>5.] = 3.5
    vprofile[z>15] = 3.8
    vprofile[z>25] = 4.4
    vprofile[z>220] = 4.8
    v = np.reshape(np.tile(vprofile,len(x)*len(y)),X.shape)
    
    #v[:] = 4.0
    #dists = np.sqrt((X-source[0])**2+(Y-source[1])**2+(Z-source[2])**2)
    #times = dists/4.0
    
    ttime = calculate_ttime_field_3D(X,Y,Z,v,source,refine_source_grid=False)
    #ttime = calculate_ttime_field_3D(X,Y,Z,v,source)
    
    fig = plt.figure()
    for i in range(9):
        ax = fig.add_subplot(3,3,i+1)
        zi = i*int(len(z)/9)
        depth = z[zi]
        ax.contour(X[:,:,0],Y[:,:,0],ttime[:,:,zi],
                   levels=np.linspace(0,np.max(ttime),50))
        ax.set_title("%.1f km" %depth)
    plt.show()
    
    paths = shoot_ray_3D(x, y, z, ttime, source, receivers)
    
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    for path in paths:
        ax1.plot(path[:,0],path[:,1],label='path')
        ax2.plot(path[:,0],path[:,2])
        ax3.plot(path[:,1],path[:,2])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim([np.min(x),np.max(x)])
    ax1.set_ylim([np.min(y),np.max(y)])
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right')
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    ax2.set_xlim([np.min(x),np.max(x)])
    ax2.set_ylim([np.min(z),np.max(z)])
    ax2.set_aspect('equal')
    ax3.set_xlabel('y')
    ax3.set_ylabel('z')
    ax3.set_xlim([np.min(y),np.max(y)])
    ax3.set_ylim([np.min(z),np.max(z)])
    ax3.set_aspect('equal')
    plt.show()
    
    #%% 3D sensitivity kernel
    
    ttimefield_src = calculate_ttime_field_3D(X,Y,Z,v,source,refine_source_grid=True)
    ttimefield_rcv = calculate_ttime_field_3D(X,Y,Z,v,receiver1,refine_source_grid=False)
    
    # dominant frequency
    f0 = 1./5
    kernel,ttime_rcv = empirical_sensitivity_kernel3d(X, Y, source, receiver1, ttimefield_src, ttimefield_rcv, f0, Z=Z)
    
    vbound = 0.5*np.max(np.abs(kernel))
    fig = plt.figure(figsize=(12,10))
    fig.suptitle("3D sensitivity kernel test")
    for i in range(9):
        ax = fig.add_subplot(3,3,i+1)
        zi = i*int(len(z)/9)
        depth = z[zi]
        ax.pcolormesh(X[:,:,0],Y[:,:,0],kernel[:,:,zi],
                      vmin=-vbound,vmax=vbound,cmap=plt.cm.seismic,
                      shading='nearest')
        ax.set_title("depth = %.1f km" %depth)
    plt.show()
    
    #%%

def main():
    print("test run")
    X, Y = np.meshgrid(np.linspace(-2000,2000,301), np.linspace(0,4000,401))
    dx = X[0][1]-X[0][0]
    dy = Y[:,0][1]-Y[:,0][0]
    source = (-1500,2500)
    receivers = [(1497,1940),[1402,1205]]
    #receivers = [np.array([1497,1950]),np.array([1200,1500])]
    v = np.ones_like(X)*3.5
    v[Y<1300] = 4.5
    v[(abs(X)<300)*(Y>1000)*(Y<3100)] *= 0.01
    v[(X>1000)*(X<1200)*(Y>2400)*(Y<2900)] = 5.5
    wavelength = 50
    v_smooth = smooth_velocity_field(v,dx,dy,wavelength)

    t0 = datetime.datetime.now()
    for j in range(10):    
        #xnew,ynew,ttimefield = calculate_ttime_field(X[0],Y[:,0],
        #                            v_smooth,source,interpolate=True,
        #                            refine_source_grid=True)
        xnew = X[0]
        ynew = Y[:,0]
        ttimefield = calculate_ttime_field_samegrid(X,Y,v_smooth,source,
                                                    refine_source_grid=True)
        path_list = shoot_ray(xnew,ynew,ttimefield,source,receivers)
    print("10 runs take",(datetime.datetime.now()-t0).total_seconds(),"s")

    X,Y = np.meshgrid(xnew,ynew)
    fig = plt.figure()
    plt.pcolormesh(X,Y,v_smooth,vmin=-1.0,vmax=6.0,cmap=plt.cm.gray,shading='nearest')
    plt.contour(X,Y,ttimefield,levels=np.linspace(0,1300,50))
    plt.plot(X,Y,'k.',markersize=0.1)
    plt.plot(source[0],source[1],'ro')
    for receiver in receivers:
        plt.plot(receiver[0],receiver[1],'bo')
    for path in path_list:
        plt.plot(path[:,0],path[:,1],'k',linewidth=3)
    plt.colorbar(label='travel time [s]')
    plt.xlabel('xdistance [km]')
    plt.ylabel('ydistance [km]')
    plt.text(0,2500,'v=0.01 km/s',rotation=90,color='white')
    plt.gca().set_aspect('equal')
    plt.show()
    #plt.close(fig)
    
    print("test 2")
    v = np.ones_like(X)*3.5
    v[200:220,80:100] = 4.8
    v[120:140,200:220] = 2.5
    v[240:290,150:170] = 2.5
    v_smooth = smooth_velocity_field(v,dx,dy,500)

    if False:
        Xfine,Yfine = np.meshgrid(np.linspace(X[0,0],X[0,-1],len(X[0])*4),
                                  np.linspace(Y[0,0],Y[-1,0],len(Y[:,0])*4))
        vfu = RegularGridInterpolator((X[0],Y[:,0]),v_smooth.T)
        v_smooth = vfu((Xfine,Yfine))
    else:
        Xfine = X
        Yfine = Y
    
    ttimefield_src = calculate_ttime_field_samegrid(Xfine,Yfine,v_smooth,source,
                                                    refine_source_grid=True)
    ttimefield_rcv = calculate_ttime_field_samegrid(Xfine,Yfine,v_smooth,receivers[0],
                                                    refine_source_grid=True)
    frequency = 1./30
    kernel,ttime_direct = empirical_sensitivity_kernel(Xfine,Yfine,source,receivers[0],
                                          ttimefield_src,ttimefield_rcv,
                                          frequency,synthetic=True)
    
    kernel_analytical = analytical_sensitivity_kernel(Xfine,Yfine,source,receivers[0],
                                                      np.mean(v_smooth),frequency)
    
    dist = np.sqrt(np.sum((np.array(source)-np.array(receivers[0]))**2))
    ttime0 = dist / np.mean(v_smooth)
    ttime1 = dist / (np.mean(v_smooth)+0.01)
    #ttime0 + np.sum(kernel_analytical*0.01)
    path = shoot_ray(Xfine[0],Yfine[:,0],ttimefield_src,source,receivers[0])[0]
    
    fig = plt.figure(figsize=(14,12))
    plt.subplot(221)
    plt.title("Travel time field")
    cbar = plt.pcolormesh(Xfine,Yfine,v_smooth,cmap=plt.cm.gray,shading='nearest')
    plt.contour(Xfine,Yfine,ttimefield_src,levels=50)
    plt.plot(path[:,0],path[:,1],'k--',linewidth=0.4)
    plt.plot(source[0],source[1],'ro')
    plt.plot(receivers[0][0],receivers[0][1],'bo')
    plt.gca().set_aspect('equal')
    plt.colorbar(cbar,label='velocity',shrink=0.5)
    plt.subplot(222)
    plt.title("empirical kernel")
    plt.contourf(Xfine,Yfine,kernel,cmap=plt.cm.seismic,
                 #vmin=-np.max(np.abs(kernel)),vmax=np.max(np.abs(kernel)),levels=40)
                 levels=np.linspace(-0.5*np.max(np.abs(kernel)),
                                     0.5*np.max(np.abs(kernel)),40),extend='both')
    plt.plot(path[:,0],path[:,1],'k--',linewidth=0.4)
    plt.plot(source[0],source[1],'ro')
    plt.plot(receivers[0][0],receivers[0][1],'bo')
    plt.gca().set_aspect('equal')
    plt.colorbar(shrink=0.5)
    plt.subplot(223)
    plt.title("analytical kernel (const velocity)")
    plt.contourf(Xfine,Yfine,kernel_analytical,cmap=plt.cm.seismic,
                 #vmin=-np.max(np.abs(kernel)),vmax=np.max(np.abs(kernel)),levels=40)
                 levels=np.linspace(-0.5*np.max(np.abs(kernel)),
                                     0.5*np.max(np.abs(kernel)),40),extend='both')
    plt.plot(source[0],source[1],'ro')
    plt.plot(receivers[0][0],receivers[0][1],'bo')
    plt.gca().set_aspect('equal')
    plt.colorbar(shrink=0.5)
    plt.subplot(224)
    plt.title("hybrid kernel")
    plt.contourf(Xfine,Yfine,0.5*(kernel+kernel_analytical),cmap=plt.cm.seismic,
                 #vmin=-np.max(np.abs(kernel)),vmax=np.max(np.abs(kernel)),levels=40)
                 levels=np.linspace(-0.5*np.max(np.abs(kernel)),
                                     0.5*np.max(np.abs(kernel)),40),extend='both')
    plt.plot(source[0],source[1],'ro')
    plt.plot(receivers[0][0],receivers[0][1],'bo')
    plt.gca().set_aspect('equal')
    plt.colorbar(shrink=0.5)
    plt.show()
    #plt.close(fig)
    
    firstorbit = first_kernel_orbit(Xfine, Yfine, source, receivers[0], 
                                    ttimefield_src, ttimefield_rcv, frequency,
                                    phaseshift=np.pi/4.)
    
    # rcvdist = np.sqrt((receivers[0][0]-X)**2 + (receivers[0][1]-Y)**2)
    # ttime_rcv = ttimefield_src.flatten()[rcvdist.argmin()]
    # w = 2*np.pi*frequency
    # testkernel = 1.-np.abs(w*(ttime_rcv - ttimefield_rcv - ttimefield_src))/np.pi
    # testkernel[testkernel<0.] = 0.
    
    fig = plt.figure(figsize=(15,7))
    plt.subplot(121)
    plt.contourf(Xfine,Yfine,kernel,cmap=plt.cm.seismic,
                 #vmin=-np.max(np.abs(kernel)),vmax=np.max(np.abs(kernel)),levels=40)
                 levels=np.linspace(-0.5*np.max(np.abs(kernel)),
                                     0.5*np.max(np.abs(kernel)),40),extend='both')
    plt.plot(source[0],source[1],'ro')
    plt.plot(receivers[0][0],receivers[0][1],'bo')
    plt.gca().set_aspect('equal')
    plt.colorbar(shrink=0.5)
    plt.title("Empirical Kernel")
    plt.subplot(122)
    plt.contourf(Xfine,Yfine,firstorbit,cmap=plt.cm.seismic,
                 #vmin=-np.max(np.abs(kernel)),vmax=np.max(np.abs(kernel)),levels=40)
                 levels=np.linspace(-0.5*np.max(np.abs(kernel)),
                                     0.5*np.max(np.abs(kernel)),40),extend='both')
    plt.plot(source[0],source[1],'ro')
    plt.plot(receivers[0][0],receivers[0][1],'bo')
    plt.gca().set_aspect('equal')
    plt.colorbar(shrink=0.5)
    plt.title("First Orbit of Kernel")  
    plt.show()
    #plt.close(fig)
    
    print("test3")
    receiver = np.array(source)+[2500.0,0.]
    v = np.ones_like(Xfine)*3.5
    v[(Xfine>-500)*(Xfine<-200)*(Yfine>2300)*(Yfine<2700)] = 2.0
    v_smooth = smooth_velocity_field(v,dx,dy,500)
    fat_ray,_,path = fat_ray_kernel(Xfine, Yfine, source, receiver,v_smooth, 1./5)
    gs = GridSpec(2,2,height_ratios=(1,3),wspace=0.2)
    fig = plt.figure(figsize=(15,9))
    ax3 = fig.add_subplot(gs[1,1])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1,0])

    ax1.plot(Xfine[0],np.sum(fat_ray,axis=0))
    ax1.set_title("Summed Kernel")
    ax1.set_xlim(np.min(Xfine),np.max(Xfine))
    ax2.set_title("Velocity field")
    cbar = ax2.pcolormesh(Xfine,Yfine,v_smooth,cmap=plt.cm.gray,shading='nearest')
    #ax2.contour(Xfine,Yfine,ttimefield_src,levels=50)
    ax2.plot(path[:,0],path[:,1],'k--',linewidth=0.4)
    ax2.plot(source[0],source[1],'ro')
    ax2.plot(receiver[0],receiver[1],'bo')
    ax2.set_aspect('equal')
    cax2 = ax2.inset_axes([0.8,0.05,0.02,0.3])
    plt.colorbar(cbar,cax=cax2,label='velocity',shrink=0.5)
    cbar = ax3.contourf(Xfine,Yfine,fat_ray,cmap=plt.cm.magma_r,
                 #vmin=-np.max(np.abs(kernel)),vmax=np.max(np.abs(kernel)),levels=40)
                 levels=np.linspace(0,np.max(fat_ray)/1,40))#,vmax=1.)#np.mean(firstorbit))
    ax3.plot(source[0],source[1],'ro')
    ax3.plot(receiver[0],receiver[1],'bo')
    ax3.set_aspect('equal')
    cax3 = ax3.inset_axes([0.8,0.05,0.02,0.3])
    plt.colorbar(cbar,cax=cax3,shrink=0.5)
    ax3.set_title("Fat Ray Kernel")  

    # same x size of line plot (3 is from the height_ratio)
    asp = np.diff(ax1.get_xlim())[0] / (3*np.diff(ax1.get_ylim())[0])
    ax1.set_aspect(asp)
    plt.show() 
    #%%
    
 
# smoothing is not used in the actual ttime field calculation below
def smooth_velocity_field(v,dx,dy,wavelength):
    # 0.5: standard deviation sigma relates to one side of the gaussian bell
    # 1./3: the entire wavelength should fit in 3 standard deviations
    sigmax = 0.5*wavelength/dx*1./3
    sigmay = 0.5*wavelength/dy*1./3
    v_smooth = gaussian_filter(v,sigma=[sigmay,sigmax],truncate=3.0,mode='nearest')
    return v_smooth
   

def calculate_ttime_field(x,y,v,source,interpolate=True,refine_source_grid=True,pts_refine=5):
        
    if source[0] > np.max(x) or source[0] < np.min(x) or source[1] > np.max(y) or source[1] < np.min(y):
        raise Exception("Source must be within the coordinate limits!")
   
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    
    xsource = source[0]
    ysource = source[1]
    sourceidx = (np.argmin(np.abs(y-ysource)),np.argmin(np.abs(x-xsource)))
    
    # create a new x and y axis because the source point normally doesn't coincide with the gridpoint locations
    xshift = x[sourceidx[1]]-source[0]
    yshift = y[sourceidx[0]]-source[1]
    xnew = x-xshift
    ynew = y-yshift
    
    Xnew,Ynew = np.meshgrid(xnew,ynew)
    
    # with the new axis, the resulting traveltime field is equivalent to an interpolated
    # travel time field on a shifted grid, using a nearest neighbour interpolation.
    # This means that the source location is correct, with respect to the receiver.
    # However, there will be an error in the velocity field (which is small for a smooth velocity field)
    # otherwise: use a fast spline interpolation
    if interpolate or refine_source_grid:
        #v_func = RegularGridInterpolator((x,y),v.T,method='linear',bounds_error=False,fill_value=None)
        #v = v_func((Xnew,Ynew))
        # Linear interpolation is more exact on very coarse grids...
        v_func = RectBivariateSpline(x,y,v.T)
        v = (v_func(xnew,ynew)).T

    # creating a finer grid around the source and doing the traveltime calculations
    # on this fine grid before expanding to the entire grid
    # gives slightly better results
    create_testplots = False
    if refine_source_grid:
        fine_factor = 20
        xfine = np.linspace(xsource-pts_refine*dx,xsource+pts_refine*dx,fine_factor*2*pts_refine+1)
        yfine = np.linspace(ysource-pts_refine*dy,ysource+pts_refine*dy,fine_factor*2*pts_refine+1)
        
        # limit the fine subaxis to the valid range
        xfine = xfine[np.abs(xfine-np.min(xnew)).argmin():np.abs(xfine-np.max(xnew)).argmin()+1]
        yfine = yfine[np.abs(yfine-np.min(ynew)).argmin():np.abs(yfine-np.max(ynew)).argmin()+1]
                
        Xfine,Yfine = np.meshgrid(xfine,yfine)
        # we start with a perfectly circular traveltimefield which is centered
        # around the source at a distance of 1/4th of the minimum sampling
        # distance.
        radius = np.min((dx,dy))/4.
        phi_fine = np.sqrt((Xfine-xsource)**2+(Yfine-ysource)**2)
        v_fine = (v_func(xfine,yfine)).T # v_func(tuple(np.meshgrid(xfine,yfine))) # (v_func(xfine,yfine)).T
        ttime_fine = skfmm.travel_time(phi_fine-radius, v_fine, dx=[dy/fine_factor,dx/fine_factor])
        #ttime_inner = np.max(ttime_fine[phi_fine<radius])
        ttime_fine += radius/np.mean(v_fine[phi_fine<radius])
        #ttime_fine[phi_fine<radius] = phi_fine[phi_fine<radius]/np.mean(v_fine[phi_fine<radius])
        
        if create_testplots:# for testing       
            plt.figure()
            plt.pcolormesh(Xnew,Ynew,v,cmap=plt.cm.seismic_r,shading='nearest')
            plt.pcolormesh(xfine,yfine,v_fine,shading='nearest',cmap=plt.cm.seismic_r)
            cont = plt.contour(xfine,yfine,ttime_fine,levels=np.linspace(0,np.max(ttime_fine),90))
            #plt.contour(xfine,yfine,ttime_fine1,levels=np.linspace(0,np.max(ttime_fine),90),linestyles='dashed')
            plt.plot(source[0],source[1],'rv')
            plt.gca().set_aspect('equal')
            plt.colorbar(cont)
            plt.show()
        
        # from the traveltimes on the fine grid we have to find an iso-
        # velocity contour that can serve as input for the larger grid
        # this iso-velocity contour is normally smoother compared to the one
        # we would get from a calculation without the grid refinement                
        phi_coarse = np.ones_like(Xnew)*np.max(ttime_fine)
        yidx0 = np.max([0,sourceidx[0]-pts_refine])
        yidx1 = np.min([sourceidx[0]+pts_refine+1,len(ynew)])
        xidx0 = np.max([0,sourceidx[1]-pts_refine])
        xidx1 = np.min([sourceidx[1]+pts_refine+1,len(xnew)])
        phi_coarse[yidx0:yidx1, xidx0:xidx1] = ttime_fine[::fine_factor,::fine_factor]
        
        if create_testplots:
            plt.figure()
            plt.pcolormesh(Xnew,Ynew,v,shading='nearest',cmap=plt.cm.seismic_r)
            levels=np.linspace(0,np.max(ttime_fine),30)
            cont=plt.contour(Xnew,Ynew,phi_coarse,linestyles='dashed',levels=levels)
            plt.contour(Xfine,Yfine,ttime_fine,linestyles='solid',levels=levels)
            plt.colorbar(cont)
            plt.gca().set_aspect('equal')
            plt.show()
        
        # minimum traveltime to the border of the fine-grid region
        borderxmax = sourceidx[1]+pts_refine if sourceidx[1]+pts_refine < len(xnew) else 0
        borderymax = sourceidx[0]+pts_refine if sourceidx[0]+pts_refine < len(ynew) else 0
        minborder_ttime = np.min((phi_coarse[sourceidx[0], sourceidx[1]-pts_refine],
                                  phi_coarse[sourceidx[0], borderxmax],
                                  phi_coarse[sourceidx[0]-pts_refine, sourceidx[1]],
                                  phi_coarse[borderymax, sourceidx[1]]))
        phi_coarse -= minborder_ttime
        
        if create_testplots:
            plt.figure()
            plt.pcolormesh(Xnew,Ynew,v,shading='nearest',cmap=plt.cm.seismic_r)
            cont=plt.contour(Xnew,Ynew,phi_coarse,linestyles='dashed',levels=[-1,0,1])
            plt.contour(Xfine,Yfine,ttime_fine,linestyles='solid',
                        levels=[minborder_ttime-1,minborder_ttime,minborder_ttime+1])
            plt.colorbar(cont)
            plt.gca().set_aspect('equal')
            plt.show()
        
        try:
            ttime_field = skfmm.travel_time(phi_coarse, v, dx=[dy,dx])
        except:
            print("had to reduce order")
            ttime_field = skfmm.travel_time(phi_coarse, v,order=1, dx=[dy,dx])
        ttime_field += minborder_ttime
        ttime_field[yidx0:yidx1, xidx0:xidx1] = ttime_fine[::fine_factor,::fine_factor]
                   
    else:
        phi = (Xnew-xsource)**2 + (Ynew-ysource)**2# - (dx/1000.)**2
        phi[sourceidx[0],sourceidx[1]] *= -1
        try:
            ttime_field = skfmm.travel_time(phi, v, dx=[dy,dx])
            #ttime_field1 = skfmm.travel_time(phi, v,order=1, dx=[dy,dx]) 
            #print("max error:",np.max(np.abs(ttime_field-ttime_field1)))
        except:
            #print("had to reduce order")
            ttime_field = skfmm.travel_time(phi, v,order=1, dx=[dy,dx])

    if create_testplots: # for testing
        plt.figure()
        ax = plt.gca()
        plt.pcolormesh(Xnew,Ynew,v,shading='nearest',cmap=plt.cm.seismic_r)
        levels = np.linspace(0,np.max(ttime_field),80)
        plt.contour(xnew,ynew,ttime_field,levels=levels,label='after grid refinement')
        #plt.legend(loc='upper right')
        #plt.contour(xnew,ynew,ttime_field2,linestyles='dashed',levels=levels,label='before grid refinement')
        #plt.contour(xnew,ynew,np.sqrt(phi)/3.8,linestyles='dotted',levels=80,label='homogeneous velocity contours')
        ax.set_aspect('equal')
        plt.show()
        
        
    return xnew,ynew,ttime_field


# this is the same as above, but the input and output grids are identical
# the ttime field is less exact but the calculations are faster and the further
# use is simpler.
def calculate_ttime_field_samegrid(X,Y,v,source,refine_source_grid=True,pts_refine=5):
        
    xsource = source[0]
    ysource = source[1]
    x = X[0]
    y = Y[:,0]  
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    
    if xsource > np.max(x) or xsource < np.min(x) or ysource > np.max(y) or ysource < np.min(y):
        raise Exception("Source must be within the coordinate limits!")
    
    if refine_source_grid:
        v_func = RectBivariateSpline(x,y,v.T)

    # creating a finer grid around the source and doing the traveltime calculations
    # on this fine grid before expanding to the entire grid
    # gives slightly better results
    if refine_source_grid:
        fine_factor = 20
        xfine = np.linspace(xsource-pts_refine*dx,xsource+pts_refine*dx,fine_factor*2*pts_refine+1)
        yfine = np.linspace(ysource-pts_refine*dy,ysource+pts_refine*dy,fine_factor*2*pts_refine+1)
                
        # update 21.05.2021: it is now okay if the source is at the border,
        # results should still be correct.
        #if np.min(xfine)<np.min(X) or np.min(yfine)<np.min(Y) or np.max(xfine)>np.max(X) or np.max(yfine)>np.max(Y):
        #    raise Exception("WARNING: the source coordinate is too close to the border of the domain, this may cause errors when trying to refine the grid!")
        
        Xfine,Yfine = np.meshgrid(xfine,yfine)
        # we start with a perfectly circular traveltimefield which is centered
        # around the source at a distance of 1/4th of the minimum sampling
        # distance.
        radius = np.min((dx,dy))/4.
        phi_fine = np.sqrt((Xfine-xsource)**2+(Yfine-ysource)**2)
        v_fine = (v_func(xfine,yfine)).T # v_func(tuple(np.meshgrid(xfine,yfine))) # (v_func(xfine,yfine)).T
        ttime_fine = skfmm.travel_time(phi_fine-radius, v_fine, dx=[dy/fine_factor,dx/fine_factor])
        #ttime_inner = np.max(ttime_fine[phi_fine<radius])
        ttime_fine += radius/np.mean(v_fine[phi_fine<radius])
        # this last line is not really necessary:
        ttime_fine[phi_fine<radius] = phi_fine[phi_fine<radius]/np.mean(v_fine[phi_fine<radius])
        
        # # for testing
        # plt.figure()
        # ax = plt.gca()
        # plt.pcolormesh(X,Y,v,shading='nearest',cmap=plt.cm.seismic_r)
        # plt.pcolormesh(Xfine,Yfine,v_fine,shading='nearest',cmap=plt.cm.seismic_r)
        # plt.plot(X.flatten(),Y.flatten(),'k.')
        # plt.contour(xfine,yfine,ttime_fine,levels=np.linspace(0,50,80))
        # ttime_fine2 = skfmm.travel_time(phi_fine, v_fine, dx=[dy/fine_factor,dx/fine_factor])
        # plt.contour(xfine,yfine,ttime_fine2,linestyles='dashed',levels=np.linspace(0,50,80))
        # plt.plot(xsource,ysource,'rv')
        # ax.set_aspect('equal')
        # plt.show()
        
        # from the traveltimes on the fine grid we have to find an iso-
        # velocity contour that can serve as input for the larger grid
        # this iso-velocity contour is normally smoother compared to the one
        # we would get from a calculation without the grid refinement                
        ttime_fine_func = RegularGridInterpolator((xfine,yfine),ttime_fine.T,
                                                  method='nearest',bounds_error=False)
        phi_coarse = ttime_fine_func((X,Y))
        # find the largest closed contour [TOO SLOW]
        # plt.ioff()
        # levels=np.linspace(0,np.max(phi_fine)/np.mean(v_fine),20)
        # cset = plt.contour(xfine, yfine, ttime_fine,levels=levels)
        # # Iterate over all the contour segments and find the largest
        # stop = False
        # for i, segs in enumerate(cset.allsegs[::-1]):
        #     for j, seg in enumerate(segs):
        #         # First make sure it's closed
        #         if (seg[0]-seg[-1]).any():
        #             continue
        #         else:
        #             stop = True
        #             break
        #     if stop:
        #         contour_ttime = levels[::-1][i]
        #         break

        # minimum traveltime to the border of the fine-grid region
        sourceidx = np.array([np.argmin(np.abs(y-ysource)),np.argmin(np.abs(x-xsource))])
        # ignore the contour levels that are too close to the border of the domain
        # 0.9 as safety margin
        contour_ttime = 0.9*np.min((np.nanmax(phi_coarse[sourceidx[0]:,sourceidx[1]] if len(y)-sourceidx[0] > pts_refine else np.max(ttime_fine)),
                                    np.nanmax(phi_coarse[:sourceidx[0],sourceidx[1]] if sourceidx[0] >= pts_refine else np.max(ttime_fine)),
                                    np.nanmax(phi_coarse[sourceidx[0],sourceidx[1]:] if len(x)-sourceidx[1] > pts_refine else np.max(ttime_fine)),
                                    np.nanmax(phi_coarse[sourceidx[0],:sourceidx[1]] if sourceidx[1] >= pts_refine else np.max(ttime_fine))))
        phi_coarse -= contour_ttime
        phi_coarse[np.isnan(phi_coarse)] = np.nanmax(phi_coarse)
        
        # # for testing
        # plt.figure()
        # ax = plt.gca()
        # #phi_coarse[np.isnan(phi_coarse)] = np.nanmax(phi_coarse)
        # cbar = plt.tricontourf(X.flatten(),Y.flatten(),phi_coarse.flatten(),
        #                         cmap=plt.cm.seismic,levels=np.arange(-50,51,1))
        # plt.plot(X.flatten(),Y.flatten(),'k.')
        # plt.contour(xfine,yfine,ttime_fine,levels=np.linspace(0,50,50))
        # plt.contour(x,y,phi_coarse,linestyles='dashed',levels=np.linspace(0,50,50))
        # plt.plot(xsource,ysource,'rv')
        # ax.set_aspect('equal')
        # plt.colorbar(cbar)
        # plt.show()
        
        try:
            ttime_field = skfmm.travel_time(phi_coarse, v, dx=[dy,dx])
        except:
            print("had to reduce order")
            ttime_field = skfmm.travel_time(phi_coarse, v,order=1, dx=[dy,dx])
        ttime_field += contour_ttime
        ttime_field[phi_coarse<=0] = phi_coarse[phi_coarse<=0]+contour_ttime
        
    else:
        phi = np.sqrt((X-xsource)**2 + (Y-ysource)**2)
        radius = 1.5*np.max((dx,dy))
        try:
            ttime_field = skfmm.travel_time(phi-radius, v, dx=[dy,dx])
        except:
            ttime_field = skfmm.travel_time(phi-radius, v, order=1, dx=[dy,dx])
        ttime_field += radius/np.mean(v[phi<radius])
        ttime_field[phi<radius] = phi[phi<radius]/np.mean(v[phi<radius])

    # # for testing
    # plt.figure()
    # ax = plt.gca()
    # plt.plot(X.flatten(),Y.flatten(),'k.')
    # plt.contour(x,y,ttime_field,levels=np.linspace(0,np.max(ttime_field),80),label='with grid refinement')
    # phi = (X-xsource)**2 + (Y-ysource)**2# - (dx/1000.)**2
    # phi[sourceidx[0],sourceidx[1]] *= -1
    # ttime_field2 = skfmm.travel_time(phi, v, dx=[dy,dx])
    # plt.contour(x,y,ttime_field2,linestyles='dashed',
    #             levels=np.linspace(0,np.max(ttime_field),80),label='without grid refinement')
    # #plt.contour(xnew,ynew,np.sqrt(phi)/3.8,linestyles='dotted',levels=80,label='homogeneous velocity contours')
    # plt.plot(xsource,ysource,'rv')
    # ax.set_aspect('equal')
    # plt.legend(loc='upper right')
    # plt.colorbar()
    # plt.show()
        
#    if extended_grid:
#        return xnew,ynew,ttime_field#[2:-2,2:-2]
#    else:
    return ttime_field

# this includes azimuthal anisotropy
def calculate_ttime_field_aniso(x,y,v,psi2,psi2amp,source):
    """

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    psi2 : TYPE
        in radians, zero is east (x direction), pi/2 is north (y direction).
    psi2amp : TYPE
        in relative units (dVs/Vs).
    source : TYPE
        DESCRIPTION.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    time (numpy array of same shape as input arrays)

    """
        
    xsource = source[0]
    ysource = source[1]
    if xsource > np.max(x) or xsource < np.min(x) or ysource > np.max(y) or ysource < np.min(y):
        raise Exception("Source must be within the coordinate limits!")      
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    if dx!=dy:
        print(f"Warning: dx and dy should be the same, otherwise pyekfmm may not work correctly ({dx},{dy}).")
    
    velx = (v*(1+psi2amp*np.cos(2*psi2))).transpose().flatten(order='F').astype('float32')
    vely = (v*(1+psi2amp*np.cos(2*(psi2-np.pi/2.)))).transpose().flatten(order='F').astype('float32')
    eta = (np.ones_like(v)*0).flatten().astype('float32')
    t=pyekfmm.eikonalvti(velx,vely,eta,xyz=np.array([xsource,0,ysource]),
                         ax=[x.min(),dx,len(x)],ay=[0,0.05,1],az=[y.min(),dy,len(y)],order=1)
    time=t.reshape(len(x),len(y),order='F').transpose()
    
    return time


def calculate_ttime_field_3D(X,Y,Z,V,source,refine_source_grid=True,pts_refine=5):
    
    xsource = source[0]
    ysource = source[1]
    zsource = source[2]
    x = X[0,:,0]
    y = Y[:,0,0]
    z = Z[0,0,:]
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    if (dx.max()-dx.min() > dx.min()/1000.) or (dy.max()-dy.min() > dy.min()/1000.) or (dz.max()-dz.min() > dz.min()/1000.):
        raise Exception("Grid must be regular in all 3 axes (x,y,z).")
    
    dx = dx[0]
    dy = dy[0]
    dz = dz[0]
    
    if (xsource > np.max(x) or xsource < np.min(x) or ysource > np.max(y) or
        ysource < np.min(y) or zsource > np.max(z) or zsource < np.min(z) ):
        raise Exception("Source must be within the coordinate limits!")
        
    sourceidx = np.array([np.argmin(np.abs(y-ysource)),
                          np.argmin(np.abs(x-xsource)),
                          np.argmin(np.abs(z-zsource))])
    
    if refine_source_grid:
        v_func = RegularGridInterpolator((y,x,z),V,bounds_error=False,fill_value=None)

    # creating a finer grid around the source and doing the traveltime calculations
    # on this fine grid before expanding to the entire grid
    # gives slightly better results
    if refine_source_grid:
        fine_factor = 10
        xfine = np.linspace(xsource-pts_refine*dx,xsource+pts_refine*dx,fine_factor*2*pts_refine+1)
        yfine = np.linspace(ysource-pts_refine*dy,ysource+pts_refine*dy,fine_factor*2*pts_refine+1)
        zfine = np.linspace(zsource-pts_refine*dz,zsource+pts_refine*dz,fine_factor*2*pts_refine+1)

        Xfine,Yfine,Zfine = np.meshgrid(xfine,yfine,zfine)
        # we start with a perfectly circular traveltimefield which is centered
        # around the source at a distance of 1/4th of the minimum sampling
        # distance.
        radius = np.min((dx,dy))/4.
        phi_fine = np.sqrt((Xfine-xsource)**2+(Yfine-ysource)**2+(Zfine-zsource)**2)
        v_fine = v_func((Yfine,Xfine,Zfine))
        ttime_fine = skfmm.travel_time(phi_fine-radius, v_fine, 
                                       dx=[dy/fine_factor,dx/fine_factor,dz/fine_factor])
        ttime_fine += radius/np.mean(v_fine[phi_fine<radius])
        # this last line is not really necessary:
        ttime_fine[phi_fine<radius] = phi_fine[phi_fine<radius]/np.mean(v_fine[phi_fine<radius])
        
        # from the traveltimes on the fine grid we have to find an iso-
        # velocity contour that can serve as input for the larger grid
        # this iso-velocity contour is normally smoother compared to the one
        # we would get from a calculation without the grid refinement                
        ttime_fine_func = RegularGridInterpolator((yfine,xfine,zfine),ttime_fine,
                                                  method='nearest',bounds_error=False)
        phi_coarse = ttime_fine_func((Y,X,Z))

        # phi_coarse contains the traveltimes from the fine grid interpolated
        # on the coarse grid. By subtracting the contour_ttime, we have a new
        # zero contour. If contour_ttime is too small, the benefit from refining
        # the source grid is lost (zero contour is again very small and
        # undersampled by the rough grid). It must, however, also not be too
        # large, to assure that we can find the contour level in all 3 dimensions
        # The function below does this check in all 6 directions away from the source.
        # ignore the contour levels that are too close to the border of the domain
        # 0.9 as safety margin
        contour_ttime = 0.9*np.min((np.nanmax(phi_coarse[sourceidx[0]:,sourceidx[1],sourceidx[2]] if len(y)-sourceidx[0] > pts_refine else np.max(ttime_fine)),
                                    np.nanmax(phi_coarse[:sourceidx[0],sourceidx[1],sourceidx[2]] if sourceidx[0] >= pts_refine else np.max(ttime_fine)),
                                    np.nanmax(phi_coarse[sourceidx[0],sourceidx[1]:,sourceidx[2]] if len(x)-sourceidx[1] > pts_refine else np.max(ttime_fine)),
                                    np.nanmax(phi_coarse[sourceidx[0],:sourceidx[1],sourceidx[2]] if sourceidx[1] >= pts_refine else np.max(ttime_fine)),
                                    np.nanmax(phi_coarse[sourceidx[0],sourceidx[1],sourceidx[2]:] if len(z)-sourceidx[2] > pts_refine else np.max(ttime_fine)),
                                    np.nanmax(phi_coarse[sourceidx[0],sourceidx[1],:sourceidx[2]] if sourceidx[2] >= pts_refine else np.max(ttime_fine))))
        phi_coarse -= contour_ttime
        phi_coarse[np.isnan(phi_coarse)] = np.nanmax(phi_coarse)
        
        try:
            ttime_field = skfmm.travel_time(phi_coarse, V, dx=[dy,dx,dz])
        except:
            print("had to reduce order")
            ttime_field = skfmm.travel_time(phi_coarse, V,order=1, dx=[dy,dx,dz])
        ttime_field += contour_ttime
        ttime_field[phi_coarse<=0] = phi_coarse[phi_coarse<=0]+contour_ttime
        
    else:
        phi = np.sqrt((X-xsource)**2 + (Y-ysource)**2 + (Z-zsource)**2)
        #phi[sourceidx[0],sourceidx[1],sourceidx[2]] *= -1
        radius = 2.5*np.max((dx,dy,dz))
        try:
            ttime_field = skfmm.travel_time(phi-radius, V, dx=[dy,dx,dz])
        except:
            ttime_field = skfmm.travel_time(phi-radius, V, order=1, dx=[dy,dx,dz])
        ttime_field += radius/np.mean(V[phi<radius])
        ttime_field[phi<radius] = phi[phi<radius]/np.mean(V[phi<radius])

    return ttime_field    


# same as calculate_ttime_field_3D but using typical geographic coordinates
# extra slow because of the additional transformations and interpolations
def calculate_ttime_field_geographic(LAT,LON,DEPTH,V,source,refine_source_grid=True,pts_refine=5,dx=None,dy=None,dz=None,receiver=None):
        
    lat = np.unique(LAT)
    lon = np.unique(LON)
    depth = np.unique(DEPTH)
    
    # cartesian coordinate system in x,y,z units but irregular spaced
    Xsph,Ysph,Zsph = geographic2cartesian(LAT,LON,DEPTH)
    
    # the grid must be regular, x,y,z are not necessarily horizontal 
    # and vertical directions anymore, depending where on the sphere we are...
    if dz is None:
        dz = np.min(np.diff(depth))
    if dy is None:
        dy = np.min(np.diff(lat))*111.19
    if dx is None:
        dx = dy
    x = np.arange(Xsph.min(),Xsph.max()+dx,dx)
    y = np.arange(Ysph.min(),Ysph.max()+dy,dy)
    z = np.arange(Zsph.min(),Zsph.max()+dz,dz)
    Xcart,Ycart,Zcart = np.meshgrid(x,y,z)
    
    # interpolate velocity field to the regular cartesian coordinate system
    LATtrans,LONtrans,Ztrans = cartesian2geographic(Xcart,Ycart,Zcart)
    func = RegularGridInterpolator((lat,lon,depth),V,
                                   bounds_error=False,fill_value=1e-3)
    Vcart = func((LATtrans,LONtrans,Ztrans))
    
    # get the traveltime field with the FMM
    source_cart = geographic2cartesian(source[0],source[1],source[2])
    ttime_field_cart = calculate_ttime_field_3D(
        Xcart, Ycart, Zcart, Vcart, source_cart, 
        refine_source_grid=refine_source_grid,pts_refine=pts_refine)
    
    # interpolate back onto geographic coordinate system
    func = RegularGridInterpolator((y,x,z),ttime_field_cart)
    ttime_field = func((Ysph,Xsph,Zsph))
    
    if receiver is not None:
        receiver_cart = geographic2cartesian(receiver[0],receiver[1],receiver[2])
        path_cart = shoot_ray_3D(x, y, z, ttime_field_cart, source_cart, receiver_cart)[0]
        path_geo = cartesian2geographic(path_cart[:,0],path_cart[:,1],path_cart[:,2])
        return ttime_field,path_geo,path_cart
    else:
        return ttime_field



def shoot_ray(x,y,ttimefield,source,receivers,stepsize=0.33):
    """

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    ttimefield : TYPE
        DESCRIPTION.
    source : TYPE
        DESCRIPTION.
    receivers : TYPE
        DESCRIPTION.
    stepsize : float, optional
        Size of the ray tracing step, relative w.r.t. the average grid sampling distance. The default is 0.33, meaning 1/3rd of a cell size.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # ray will be shot from the receiver to the source
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    step_dist = np.sqrt(dx**2+dy**2)*stepsize

    grad = np.gradient(ttimefield)
    spl_x = RectBivariateSpline(x,y,grad[1].T/dx)
    spl_y = RectBivariateSpline(x,y,grad[0].T/dy)
    def descent(t,xy):
        return [-spl_x.ev(xy[0],xy[1]),-spl_y.ev(xy[0],xy[1])]
     
    """
    # for testing - interesting artefacts in the reconstructed velocity field
    # I am not sure where they come from. Maybe because of the use of carte-
    # sian coordinates for a problem that is better described in polar coords?
    ttimefu = RectBivariateSpline(x,y,ttimefield.T)
    expected_ttimes = ttimefu.ev(np.array(receivers)[:,0],np.array(receivers)[:,1])
    velfield = 1./np.sqrt((grad[0]/dy)**2+(grad[1]/dx)**2)
    
    plt.figure()
    plt.pcolormesh(x,y,velfield,vmin=3.5,vmax=4.1)
    plt.contour(x,y,ttimefield,levels=80)
    plt.plot(receivers[0][0],receivers[0][1],'rv')
    plt.plot(path_list[0][:,0],path_list[0][:,1],'.')
    plt.show()
    """

    if len(np.shape(receivers)) == 1:
        receivers = [receivers]    

    path_list = []
    for receiver in receivers:
        if receiver[0]>np.max(x) or receiver[0]<np.min(x) or receiver[1]>np.max(y) or receiver[1]<np.min(y):
            print("Warning: receiver location outside map boundary")
            path = np.array([[np.nan,np.nan]])
            path_list.append(path)
            continue
        step_inc = 1.
        path = [[receiver[0],receiver[1]]]
        total_dist = np.sqrt((receiver[1]-source[1])**2 + (receiver[0]-source[0])**2)
        N = int(total_dist/step_dist*3)
        if N>100000:
            print("Ray tracing warning: path is sampled with more than 100k steps, consider using a larger stepsize")
        for i in range(N):
            x0,y0 = path[-1]
            if np.isinf(x0):
                raise Exception("ray tracing error")
            dist = np.sqrt((x0-source[0])**2 + (y0-source[1])**2)
            gradx,grady = descent(1,[x0,y0])
            step_distance = np.sqrt(gradx**2+grady**2)
            step_inc = step_dist/step_distance
            if dist/(step_distance*step_inc) <= 1.:
                break

            path.append([x0+gradx*step_inc,y0+grady*step_inc])
            
            # in vincinity of the source, the traveltimefield is not very exact, just make a straight line jump
            if np.abs(x0-source[0])<dx/0.5 and np.abs(y0-source[1])<dy/0.5:
                break
        else:
            print("ERROR: ray tracing is not converging towards the source!")
            print("source:",source,"receiver:",receiver)
            raise Exception()
        path.append([source[0],source[1]])
        path_list.append(np.array(path)[::-1])
    return path_list
    #ALTERNATIVE setting it up as initial value problem. is slower
#    def event(t,xy):
#        if np.abs(source[0]-xy[0])<dx/2. and np.abs(source[1]-xy[1])<dy/2.:
#            return 0
#        else:
#            return (source[0]-xy[0])**2+(source[1]-xy[1])**2
#    event.terminal = True
#    sol = solve_ivp(descent,[0,5000],[receiver[0],receiver[1]],events=event,dense_output=True)
#    return sol.y.T
  
def shoot_ray_3D(x,y,z,ttimefield,source,receivers,stepsize=0.5):
    """
    stepsize : float, optional
        Size of the ray tracing step. The steps along the path will be such that
        they are never larger than stepsize*(dx,dy,dz) along any of the three
        caresian axes.

    """
    def running_mean(x, N):
        if N>len(x):
            print("Warning: Length of the array is shorter than the number of samples to apply the running mean. Setting N=%d." %len(x))
            N=len(x)
        if N<=1:
            return x
        if N%2 == 0:
            N+=1
        idx0 = int((N-1)/2)
        runmean = np.zeros(len(x))
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        runmean[idx0:-idx0] = (cumsum[N:] - cumsum[:-N]) / N
        for i in range(idx0):
            runmean[i] = np.mean(x[:2*i+1])
            runmean[-i-1] = np.mean(x[-2*i-1:])
        return runmean
    
    global ray_outside_map
    ray_outside_map = False
    def descent(yxz):
        global ray_outside_map
        try:
            return -xgrad(yxz),-ygrad(yxz),-zgrad(yxz)
        except: # if outside the map boundary
            yp,xp,zp = yxz # vector pointing in source direction (ray is traced from receiver to source)
            if not ray_outside_map:
                print("Warning! Ray is traced along the map boundary. This may cause problems.")
                ray_outside_map = True
                print(yxz)
            return -np.array([xp-source[0],yp-source[1],zp-source[2]])
            # xax = np.linspace(xp-dx,xp+dx,3)
            # yax = np.linspace(yp-dy,yp+dy,3)
            # zax = np.linspace(zp-dz,zp+dz,3)
            # X,Y,Z = np.meshgrid(xax,yax,zax)
            # dists = np.sqrt((X-source[0])**2+(Y-source[1])**2+(Z-source[2])**2)
            # yg,xg,zg = np.gradient(dists/3.0)
            # return -xg[1,1,1],-yg[1,1,1],-zg[1,1,1]
    
    # ray will be shot from the receiver to the source
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    dz = z[1]-z[0]

    grad = np.gradient(ttimefield,dy,dx,dz)
    xgrad = RegularGridInterpolator((y,x,z),grad[1],method='linear')
    ygrad = RegularGridInterpolator((y,x,z),grad[0],method='linear')
    zgrad = RegularGridInterpolator((y,x,z),grad[2],method='linear')
    
    if len(np.shape(receivers)) == 1:
        receivers = [receivers]    

    path_list = []
    for receiver in receivers:
        if (receiver[0]>np.max(x) or receiver[0]<np.min(x) or 
            receiver[1]>np.max(y) or receiver[1]<np.min(y) or 
            receiver[2]>np.max(z) or receiver[2]<np.min(z) ):
            print("Warning: receiver location outside map boundary")
            path = np.array([[np.nan,np.nan,np.nan]])
            path_list.append(path)
            continue
        step_inc = 1.
        path = [[receiver[0],receiver[1],receiver[2]]]
        total_dist = np.sqrt((receiver[1]-source[1])**2 + (receiver[0]-source[0])**2 + (receiver[2]-source[2])**2)
        # calculate the maximum number of steps that are needed to get from source to receiver times 3
        N = int(total_dist/(np.min([dx,dy,dz])*stepsize))*3
        if N>100000:
            print("Ray tracing warning: path is sampled with more than 100k steps, consider using a larger stepsize")
        for i in range(N):
            x0,y0,z0 = path[-1]
            if np.isinf(x0):
                raise Exception("ray tracing error")
            dist = np.sqrt((x0-source[0])**2 + (y0-source[1])**2 + (z0-source[2])**2)
            gradx,grady,gradz = descent((y0,x0,z0))
            # distance of the ray tracing step if following the gradient
            step_distance = np.sqrt(gradx**2+grady**2+gradz**2)
            # step_inc gives the maximum allowed increment so that no step
            # is larger than stepsize*gridspacing(in x,y,z)
            step_inc = np.min([dx/np.abs(gradx),dy/np.abs(grady),dz/np.abs(gradz)])*stepsize
            if dist/(step_distance*step_inc) <= 1.:
                break

            path.append([x0+gradx*step_inc,y0+grady*step_inc,z0+gradz*step_inc])
            
            # in vincinity of the source, the traveltimefield is not very exact, just make a straight line jump
            if np.abs(x0-source[0])<dx and np.abs(y0-source[1])<dy and np.abs(z0-source[2])<dz:
                break
        else:
            print("ERROR: ray tracing is not converging towards the source!")
            print("source:",source,"receiver:",receiver)
            raise Exception()
        path.append([source[0],source[1],source[2]])
        path = np.array(path)[::-1]
        path = np.column_stack((running_mean(path[:,0],5),
                                running_mean(path[:,1],5),
                                running_mean(path[:,2],5)))
        path_list.append(path)
    return path_list

    """
    zidx = 20
    test = path_list[1]#np.vstack(path)#path_list[0]
    plt.figure()
    plt.plot(X.flatten(),Y.flatten(),'.',color='lightgrey')
    plt.contour(X[:,:,0],Y[:,:,0],ttimefield[:,:,zidx],
                   levels=np.linspace(0,np.max(ttime),50))
    plt.quiver(X[:,:,0],Y[:,:,0],grad[1][:,:,zidx],grad[0][:,:,zidx],units='xy')
    plt.plot(test[:,0],test[:,1])
    plt.plot(source[0],source[1],'o')
    plt.plot(receiver[0],receiver[1],'o')
    plt.gca().set_aspect('equal')
    plt.show()   
    """



def analytical_sensitivity_kernel(X,Y,source,receiver,c,f0,gaussfilt=2,
                                  maxorbits=5):
    """
    

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    source : TYPE
        DESCRIPTION.
    receiver : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    f0 : TYPE
        DESCRIPTION.
    gaussfilt : TYPE, optional
        DESCRIPTION. The default is 2.
    maxorbits : Integer, kernel will be truncated after 'maxorbits' orbits.
        DESCRIPTION. The default is 5.

    Returns
    -------
    kernel : TYPE
        DESCRIPTION.

    """
    
    if maxorbits<1:
        raise Exception("maxorbits has to be at least 1")
    
    spatial_sampling = np.max([X[0,1]-X[0,0],Y[1,0]-Y[0,0]])
    
    fmax = c/(2*spatial_sampling)
    
    # for a constant velocity model as in Lin and Ritzwoller 2010
    df = np.sqrt(np.log(0.01)*(-f0**2/gaussfilt**2))
    if f0+df > fmax:
        print("Warning! grid spacing is too large for the given frequency. This may cause aliasing.")
    
    frequency = np.linspace(np.max([f0-df,0.0001]),f0+df,100)
    w = 2*np.pi*frequency
    dw = w[1]-w[0]
    
    source = np.array(source)
    receiver = np.array(receiver)
    
    srcdist = np.sqrt((source[0]-X)**2 + (source[1]-Y)**2)
    rcvdist = np.sqrt((receiver[0]-X)**2 + (receiver[1]-Y)**2)
    mindist = 0.5*np.sqrt(np.diff(X[0,:2])+np.diff(Y[:2,0]))
    srcdist[srcdist<mindist] = mindist
    rcvdist[rcvdist<mindist] = mindist
    pairdist = np.sqrt(np.sum((source - receiver)**2))
    
    k = w / c
    
    # check that the grid spacing is small enough for the maximum frequency
    # grid spacing needs to be smaller than two times the period in space
    if np.max([X[0,1]-X[0,0],Y[1,0]-Y[0,0]]) > np.pi/np.max(k):
        #print("Warning: grid spacing is too large, kernel may not be accurate!")
        gspace = np.pi/np.max(k)
        print("Required grid spacing would be:",gspace,"   currently:",np.max([X[0,1]-X[0,0],Y[1,0]-Y[0,0]]))
    
    # region of interest (max 5 orbits = 9*pi/2)
    # 1st orbit: pi/2, every following orbit: pi
    orbit_phase = np.pi/2. + (maxorbits-1) * np.pi
    idx = (np.abs(2*np.pi*f0/c*(pairdist - rcvdist - srcdist)+np.pi/4) <= orbit_phase)
    
    g = np.exp(-(gaussfilt*2*np.pi*(frequency-f0)/(2*np.pi*f0))**2)
    
    # instead maybe analytical integration?
    # int_0_inf(sqrt(w)*cos(w)*exp(w**2))dw -> integration by parts and then Fresnel integral?
    kernel = np.zeros_like(X)
    for i in range(len(frequency)):
        instantaneous_kernel = ((-2*w[i] / (pairdist * c)) * 
            np.sqrt(pairdist/(8*np.pi*k[i]*srcdist[idx]*rcvdist[idx])) *
            np.cos(k[i]*(pairdist - rcvdist[idx] - srcdist[idx]) + np.pi/4.))
        if np.isnan(instantaneous_kernel).any():
            break
        kernel[idx] += g[i]**2 * instantaneous_kernel * dw
    kernel /= np.sum(g**2*dw)
    # kernelsum = np.sum(np.abs(kernel))
    # remove all very small values
    kernel[np.abs(kernel)<0.01*np.max(np.abs(kernel))] = 0.
    # # cleanup kernel (in some cases, the frequency axis is not sampled with
    # # enough steps, so that some sensitivity remains in remote areas)
    # joint_dist = srcdist + rcvdist
    # sumkernelweights0 = np.max(np.abs(kernel))
    # for isocontour in np.linspace(np.min(joint_dist),np.max(joint_dist),50)[1:]:
    #     sumkernelweights = np.sum(np.abs(kernel[joint_dist<isocontour]))
    #     if sumkernelweights/sumkernelweights0 < 1.03:
    #         kernel[joint_dist>isocontour] = 0.0
    #         break
    #     sumkernelweights0 = sumkernelweights
    # kernel *= kernelsum/np.sum(np.abs(kernel))
        
    """ 
    k = 2*np.pi*f0 / c    
    kernel1 = ((-2*2*np.pi*f0 / (pairdist * c) ) * 
              np.sqrt(pairdist/(8*np.pi*k*srcdist*rcvdist)) *
              np.cos(k*(pairdist-rcvdist-srcdist) + np.pi/4.))
    
    plt.figure()
    plt.subplot(121)
    plt.contourf(X,Y,kernel1,cmap=plt.cm.seismic,
                 levels=np.linspace(-15e-6,15e-6,40),extend='both')
    plt.plot(source[0],source[1],'ro')
    plt.plot(receiver[0],receiver[1],'bo')
    plt.gca().set_aspect('equal')
    plt.colorbar(shrink=0.5)
    plt.subplot(122)
    plt.contourf(X,Y,kernel,cmap=plt.cm.seismic,
                 levels=np.linspace(-15e-6,15e-6,40),extend='both')
    plt.plot(source[0],source[1],'ro')
    plt.plot(receiver[0],receiver[1],'bo')
    plt.gca().set_aspect('equal')
    plt.colorbar(shrink=0.5)
    plt.show()
    """
    
    return kernel 
        
        

def empirical_sensitivity_kernel(X,Y,source,receiver,ttimefield_src,
                                 ttimefield_rcv,f0,synthetic=False,
                                 maxorbits=5):
    
    #%%
    #f0 = 1./5.
    if maxorbits<1:
        raise Exception("maxorbits has to be at least 1")
        
    # filter range
    # we assume a gaussian filter exactly as in Lin and Ritzoller 2010
    # we are only interested in the frequency range around f0 where the filter
    # weights are larger 0.01
    df = np.sqrt(np.log(0.01)*(-f0**2/2**2))
    
    frequency = np.linspace(np.max([f0-df,0.0001]),f0+df,100)
    w = 2*np.pi*frequency
    dw = w[1]-w[0]
    
    source = np.array(source)
    receiver = np.array(receiver)
    
    srcdist = np.sqrt((source[0]-X)**2 + (source[1]-Y)**2)
    rcvdist = np.sqrt((receiver[0]-X)**2 + (receiver[1]-Y)**2)
    
    mindist = 0.5*np.sqrt(np.diff(X[0,:2])+np.diff(Y[:2,0]))
    srcdist[srcdist<mindist] = mindist
    rcvdist[rcvdist<mindist] = mindist
    
    ttime_rcv = ttimefield_src.flatten()[rcvdist.argmin()] # ttime from src to rcv
    
    pairdist = np.sqrt(np.sum((source - receiver)**2))

    c = pairdist / (ttime_rcv - np.pi/(4*w))
    k = w / c
    
    frequency = frequency[k>0]
    c = c[k>0]
    w = w[k>0]
    k = k[k>0]
    
    # The kernel calculations assumes a far field approximation (see eq. 10abcd
    # of Lin and Ritzwoller 2010). If the traveltimes are synthetic times,
    # the phase shift is different (synthetic ttimes are just dist/c not
    # dist/c + pi/(4w)).
    if synthetic:
        phaseshift = np.pi/4.
    else:
        phaseshift = np.pi/2.

    # check that the grid spacing is small enough for the maximum frequency
    # grid spacing needs to be smaller than two times the period in space
    if np.max([X[0,1]-X[0,0],Y[1,0]-Y[0,0]]) > np.pi/np.max(k):
        print("Warning: grid spacing is too large, kernel may not be accurate!")
        gspace = np.pi/np.max(k)
        print("Required grid spacing would be:",gspace,"   currently:",
              np.max([X[0,1]-X[0,0],Y[1,0]-Y[0,0]]))

    # region of interest (max 5 orbits = 9*pi/2)
    # 1st orbit: pi/2, every following orbit: pi
    orbit_phase = np.pi/2. + (maxorbits-1) * np.pi
    idx = (np.abs(2*np.pi*f0*(ttime_rcv - ttimefield_rcv - ttimefield_src)
                  +phaseshift) <= orbit_phase)
      
    g = np.exp(-(2*2*np.pi*(frequency-f0)/(2*np.pi*f0))**2)
    
    kernel = np.zeros_like(X)
    for i in range(len(frequency)):
        instantaneous_kernel = ((-2*w[i] / (pairdist * c[i])) * 
            np.sqrt(pairdist/(8*np.pi*k[i]*srcdist[idx]*rcvdist[idx])) *
            np.cos(w[i]*(ttime_rcv - ttimefield_rcv[idx] - 
                         ttimefield_src[idx]) + phaseshift))
        if np.isnan(instantaneous_kernel).any():
            break
        kernel[idx] += g[i]**2 * instantaneous_kernel * dw
    kernel /= np.sum(g**2*dw)
    # kernelsum = np.sum(np.abs(kernel))
    # remove all very small values
    kernel[np.abs(kernel)<0.01*np.max(np.abs(kernel))] = 0.
    # # cleanup kernel (in some cases, the frequency axis is not sampled with
    # # enough steps, so that some sensitivity remains in remote areas)
    # joint_ttime = ttimefield_rcv + ttimefield_src
    # sumkernelweights0 = np.max(np.abs(kernel))
    # for isocontour in np.linspace(np.min(joint_ttime),np.max(joint_ttime),50)[1:]:
    #     sumkernelweights = np.sum(np.abs(kernel[joint_ttime<isocontour]))
    #     if sumkernelweights/sumkernelweights0 < 1.03:
    #         kernel[joint_ttime>isocontour] = 0.0
    #         break
    #     sumkernelweights0 = sumkernelweights
    # kernel *= kernelsum/np.sum(np.abs(kernel))

    """
    c = pairdist / (ttime_rcv - np.pi/(4*2*np.pi*f0))
    k = 2*np.pi*f0 / c    
    kernel1 = ((-2*2*np.pi*f0 / (pairdist * c)) * 
              np.sqrt(pairdist/(8*np.pi*k*srcdist*rcvdist)) *
              np.cos(2*np.pi*f0*(ttime_rcv - ttimefield_rcv - ttimefield_src) + np.pi/2.))
    
    #kernel1 = (2*np.pi*f0*(ttime_rcv - ttimefield_rcv - ttimefield_src))/np.pi
    idx = ((2*np.pi*f0*(ttime_rcv - ttimefield_rcv - ttimefield_src))/np.pi >= 0.)
    #idx = (kernel1>=0)#*(kernel1<0.5)
    kernel1[~idx] = 0.
    levels = np.linspace(-np.max(np.abs(kernel))*0.9,np.max(np.abs(kernel))*0.9,40)
    plt.figure()
    plt.subplot(121)
    plt.contourf(X,Y,kernel1,cmap=plt.cm.seismic,#levels=np.linspace(-2,2,50))
                 levels=levels,extend='both')
    plt.plot(source[0],source[1],'ro')
    plt.plot(receiver[0],receiver[1],'bo')
    plt.gca().set_aspect('equal')
    plt.colorbar(shrink=0.5)
    plt.subplot(122)
    plt.contourf(X,Y,kernel,cmap=plt.cm.seismic,
                 levels=levels,extend='both')
    plt.plot(source[0],source[1],'ro')
    plt.plot(receiver[0],receiver[1],'bo')
    plt.gca().set_aspect('equal')
    plt.colorbar(shrink=0.5)
    plt.show()
    """
    #%%
    return kernel,ttime_rcv
    

def empirical_sensitivity_kernel3d(X,Y,source,receiver,ttimefield_src,
                                 ttimefield_rcv,f0,Z=None,synthetic=False,
                                 maxorbits=5):
    
    #%%
    #f0 = 1./5.
    if maxorbits<1:
        raise Exception("maxorbits has to be at least 1")
        
    # filter range
    # we assume a gaussian filter exactly as in Lin and Ritzoller 2010
    # we are only interested in the frequency range around f0 where the filter
    # weights are larger 0.01
    df = np.sqrt(np.log(0.01)*(-f0**2/2**2))
    
    frequency = np.linspace(np.max([f0-df,0.0001]),f0+df,100)
    w = 2*np.pi*frequency
    dw = w[1]-w[0]
    
    source = np.array(source)
    receiver = np.array(receiver)
    
    if Z is None:
        srcdist = np.sqrt((source[0]-X)**2 + (source[1]-Y)**2)
        rcvdist = np.sqrt((receiver[0]-X)**2 + (receiver[1]-Y)**2)
        mindist = 0.5*np.sqrt(np.diff(X[0,:2])+np.diff(Y[:2,0]))
    else:
        srcdist = np.sqrt((source[0]-X)**2 + (source[1]-Y)**2 + (source[2]-Z)**2)
        rcvdist = np.sqrt((receiver[0]-X)**2 + (receiver[1]-Y)**2 + (receiver[2]-Z)**2)
        mindist = 0.5*np.sqrt(np.diff(X[0,:2,0])+np.diff(Y[:2,0,0])+np.diff(Z[0,0,:2]))

    srcdist[srcdist<mindist] = mindist
    rcvdist[rcvdist<mindist] = mindist
    
    ttime_rcv = ttimefield_src.flatten()[rcvdist.argmin()] # ttime from src to rcv
    
    pairdist = np.sqrt(np.sum((source - receiver)**2))

    c = pairdist / (ttime_rcv - np.pi/(4*w))
    k = w / c
    
    frequency = frequency[k>0]
    c = c[k>0]
    w = w[k>0]
    k = k[k>0]
    
    # The kernel calculations assumes a far field approximation (see eq. 10abcd
    # of Lin and Ritzwoller 2010). If the traveltimes are synthetic times,
    # the phase shift is different (synthetic ttimes are just dist/c not
    # dist/c + pi/(4w)).
    if synthetic:
        phaseshift = np.pi/4.
    else:
        phaseshift = np.pi/2.

    # check that the grid spacing is small enough for the maximum frequency
    # grid spacing needs to be smaller than two times the period in space
    if np.max([X[0,1,0]-X[0,0,0],Y[1,0,0]-Y[0,0,0],Z[0,0,1]-Z[0,0,0]]) > np.pi/np.max(k):
        print("Warning: grid spacing is too large, kernel may not be accurate!")
        gspace = np.pi/np.max(k)
        print("Required grid spacing would be:",gspace,"   currently:",
              np.max([X[0,1]-X[0,0],Y[1,0]-Y[0,0]]))

    # region of interest (max 5 orbits = 9*pi/2)
    # 1st orbit: pi/2, every following orbit: pi
    orbit_phase = np.pi/2. + (maxorbits-1) * np.pi
    idx = (np.abs(2*np.pi*f0*(ttime_rcv - ttimefield_rcv - ttimefield_src)
                  +phaseshift) <= orbit_phase)
      
    g = np.exp(-(2*2*np.pi*(frequency-f0)/(2*np.pi*f0))**2)
    
    kernel = np.zeros_like(X)
    for i in range(len(frequency)):
        instantaneous_kernel = ((-2*w[i] / (pairdist * c[i])) * 
            np.sqrt(pairdist/(8*np.pi*k[i]*srcdist[idx]*rcvdist[idx])) *
            np.cos(w[i]*(ttime_rcv - ttimefield_rcv[idx] - 
                         ttimefield_src[idx]) + phaseshift))
        if np.isnan(instantaneous_kernel).any():
            break
        kernel[idx] += g[i]**2 * instantaneous_kernel * dw
    kernel /= np.sum(g**2*dw)
    # kernelsum = np.sum(np.abs(kernel))
    # remove all very small values
    kernel[np.abs(kernel)<0.01*np.max(np.abs(kernel))] = 0.
    # # cleanup kernel (in some cases, the frequency axis is not sampled with
    # # enough steps, so that some sensitivity remains in remote areas)
    # joint_ttime = ttimefield_rcv + ttimefield_src
    # sumkernelweights0 = np.max(np.abs(kernel))
    # for isocontour in np.linspace(np.min(joint_ttime),np.max(joint_ttime),50)[1:]:
    #     sumkernelweights = np.sum(np.abs(kernel[joint_ttime<isocontour]))
    #     if sumkernelweights/sumkernelweights0 < 1.03:
    #         kernel[joint_ttime>isocontour] = 0.0
    #         break
    #     sumkernelweights0 = sumkernelweights
    # kernel *= kernelsum/np.sum(np.abs(kernel))

    """
    c = pairdist / (ttime_rcv - np.pi/(4*2*np.pi*f0))
    k = 2*np.pi*f0 / c    
    kernel1 = ((-2*2*np.pi*f0 / (pairdist * c)) * 
              np.sqrt(pairdist/(8*np.pi*k*srcdist*rcvdist)) *
              np.cos(2*np.pi*f0*(ttime_rcv - ttimefield_rcv - ttimefield_src) + np.pi/2.))
    
    #kernel1 = (2*np.pi*f0*(ttime_rcv - ttimefield_rcv - ttimefield_src))/np.pi
    idx = ((2*np.pi*f0*(ttime_rcv - ttimefield_rcv - ttimefield_src))/np.pi >= 0.)
    #idx = (kernel1>=0)#*(kernel1<0.5)
    kernel1[~idx] = 0.
    levels = np.linspace(-np.max(np.abs(kernel))*0.9,np.max(np.abs(kernel))*0.9,40)
    plt.figure()
    plt.subplot(121)
    plt.contourf(X,Y,kernel1,cmap=plt.cm.seismic,#levels=np.linspace(-2,2,50))
                 levels=levels,extend='both')
    plt.plot(source[0],source[1],'ro')
    plt.plot(receiver[0],receiver[1],'bo')
    plt.gca().set_aspect('equal')
    plt.colorbar(shrink=0.5)
    plt.subplot(122)
    plt.contourf(X,Y,kernel,cmap=plt.cm.seismic,
                 levels=levels,extend='both')
    plt.plot(source[0],source[1],'ro')
    plt.plot(receiver[0],receiver[1],'bo')
    plt.gca().set_aspect('equal')
    plt.colorbar(shrink=0.5)
    plt.show()
    """
    #%%
    return kernel,ttime_rcv    
 
def first_kernel_orbit(X,Y,source,receiver,ttimefield_src,ttimefield_rcv,f0,
                       phaseshift=np.pi/2.):
    """ first orbit of empirical kernel: phaseshift = np.pi/2.
    first orbit of analytical (synthetic ) kernel: phaseshift = np.pi/4.
    fat ray kernel: phaseshift = 0 (experimental)
    """
    source = np.array(source)
    receiver = np.array(receiver)

    srcdist = np.sqrt((source[0]-X)**2 + (source[1]-Y)**2)
    rcvdist = np.sqrt((receiver[0]-X)**2 + (receiver[1]-Y)**2)
    
    ttime_rcv = ttimefield_src.flatten()[rcvdist.argmin()] # ttime from src to rcv
    pairdist = np.sqrt(np.sum((source - receiver)**2))
    
    # mindist to avoid division by small numbers in the second term in kernel
    # the second term controls the amplitude. There can be numerical instabilities
    # because of the finite size of the gridcells
    # 5 is just an experimental value without much meaning
    mindist = 5*pairdist
    # alternatively set the srcdist and rcvdist values that are too small to
    # a min threshold
    #mindist = 0.5*np.sqrt(np.diff(X[0,:2])**2+np.diff(Y[:2,0])**2)
    #srcdist[srcdist<mindist] = mindist
    #rcvdist[rcvdist<mindist] = mindist
    
    w = 2*np.pi*f0
    c = pairdist / (ttime_rcv - np.pi/(4*w))
    k = w / c
    kernel = ((-2*w / (pairdist * c)) * 
              np.sqrt(pairdist/(8*np.pi*k*srcdist*rcvdist+mindist)) *
              np.cos(w*(ttime_rcv - ttimefield_rcv - ttimefield_src) + phaseshift))    
    
    #kernel = (w*(ttime_rcv - ttimefield_rcv - ttimefield_src))/np.pi
    idx = (np.abs(w*(ttime_rcv - ttimefield_rcv - ttimefield_src)+phaseshift) <= np.pi/2.)
    #idx = (kernel1>=0)#*(kernel1<0.5)
    kernel[~idx] = 0.
    
    # levels = np.linspace(-np.max(np.abs(kernel))*0.9,np.max(np.abs(kernel))*0.9,40)
    # plt.figure(figsize=(15,9))
    # plt.subplot(121)
    # plt.plot(X.flatten(),Y.flatten(),'k.',ms=1,)
    # plt.contourf(X,Y,kernel,cmap=plt.cm.seismic,#levels=50,)#levels=np.linspace(-2,2,50))
    #               levels=levels,extend='both')
    # plt.plot(source[0],source[1],'ro')
    # plt.plot(receiver[0],receiver[1],'bo')
    # plt.gca().set_aspect('equal')
    # plt.colorbar(shrink=0.5)
    # plt.show()
    
    return kernel
    

def fat_ray_kernel(X,Y,source,receiver,vel,f0):
    
    # source = np.array(source)
    # receiver = np.array(receiver)
    
    dx = np.diff(X[0,:2])[0]
    dy = np.diff(Y[:2,0])[0]
    wavelength = np.min(vel)*1./f0
    if wavelength < np.min([dx,dy]):
        raise Exception("grid size too small, fat ray kernels will be unstable."+
                        " Consider using a ray approximation.")
    
    if np.std(vel)==0.: # constant velocity field
        ttimefield_src = (np.sqrt((source[0]-X)**2 + (source[1]-Y)**2))/vel
        ttimefield_rcv = (np.sqrt((receiver[0]-X)**2 + (receiver[1]-Y)**2))/vel
        path = np.vstack((source,receiver))
    else:
        ttimefield_src = calculate_ttime_field_samegrid(
            X,Y,vel,source,refine_source_grid=True)
        ttimefield_rcv = calculate_ttime_field_samegrid(
             X,Y,vel,receiver,refine_source_grid=True)
        path = shoot_ray(X[0],Y[:,0],ttimefield_src,source,receiver)[0]
        
    kernel = first_kernel_orbit(X,Y,source,receiver,ttimefield_src,
                                ttimefield_rcv,f0,phaseshift=0.)
    
    distance = np.sum(np.sqrt(np.sum(np.diff(path,axis=0)**2,axis=1)))
        
    #if (kernel > 0.).any():
    #    raise Exception()
    kernel = np.abs(kernel)
    #kernel[kernel>0.] = 1. # WARNING: THIS IS PROBABLY NOT GOOD!!!!
    kernel = gaussian_filter(kernel,0.5) # avoid instabilities
    # threshold to avoid too much weight at the station locations (need to find a better option)
    # instead of thershold, using mindist in first_kernel_orbit function
    #thresh = 5*np.mean(kernel[kernel>0.])
    #kernel[kernel>thresh] = thresh
    # remove very small elements so that the number of nonzero values in the matrices is small
    kernel[kernel<0.01*np.max(kernel)] = 0.
    # normalize kernel to distance
    kernel = kernel/np.sum(kernel)*distance
    
    angle = np.arctan((receiver[1]-source[1])/(receiver[0]-source[0]+1e-16))
    PHI = np.ones_like(kernel)*angle
    PHI[kernel==0.] = 0.

    # gradient_field = np.gradient(ttimefield_src)    
    # gradient_field[1] /= dx # x gradient
    # gradient_field[0] /= dy
    # PHI = np.arctan(gradient_field[0]/(gradient_field[1]+1e-16))    
    # PHI[kernel==0.] = 0.
    
    # plt.figure()
    # plt.subplot(212)
    # plt.pcolormesh(X,Y,kernel)
    # plt.plot(source[0],source[1],'o')
    # plt.subplot(211)
    # plt.plot(X[0],np.sum(kernel,axis=0))
    # plt.show()
    # # normalize kernel
    # mindist = np.sqrt(np.diff(X[0,:2])+np.diff(Y[:2,0]))
    # ttime_bins = np.arange(0,np.max(ttimefield_src),15*mindist/np.min(vel))
    # ttime_bins = np.reshape(np.repeat(ttime_bins,2)[1:-1],(len(ttime_bins)-1,2))
    # #ttime_mid = ttime_bins[:-1] + np.diff(ttime_bins)/2.
    # idx = np.where(kernel>0.)
    # for t1,t2 in ttime_bins:
    #     subidx = (ttimefield_src[idx]>=t1)*(ttimefield_src[idx]<t2)
    #     print(np.sum(subidx))
    #     subidx = (idx[0][subidx],idx[1][subidx])
    #     kernel[subidx] /= np.sum(kernel[subidx])
    
    return kernel, PHI, path
    
    
    

# def get_gaussian_kernel(x,y,ttimefield,source,receiver,wavelength,stepsize=0.33):

#     # ray will be shot from the receiver to the source
#     dx = x[1]-x[0]
#     dy = y[1]-y[0]
#     step_dist = np.sqrt(dx**2+dy**2)*stepsize

#     grad = np.gradient(ttimefield)
#     spl_x = RectBivariateSpline(x,y,grad[1].T)
#     spl_y = RectBivariateSpline(x,y,grad[0].T*dx/dy)
#     def descent(t,xy):
#         return [-spl_x.ev(xy[0],xy[1]),-spl_y.ev(xy[0],xy[1])]   
    
#     if np.shape(receiver) != np.shape((1,2)):
#         raise Exception("receiver shape not correct")  

#     if receiver[0]>np.max(x) or receiver[0]<np.min(x) or receiver[1]>np.max(y) or receiver[1]<np.min(y):
#         print("Warning: receiver location outside map boundary")
#         return ()
    
#     step_inc = 1.
#     path = [[receiver[0],receiver[1]]]
#     total_dist = np.sqrt((receiver[1]-source[1])**2 + (receiver[0]-source[0])**2)
#     N = int(total_dist/step_dist*3)
#     if N>100000:
#         print("Ray tracing warning: path is sampled with more than 100k steps, consider using a larger stepsize")
#     for i in range(N):
#         x0,y0 = path[-1]
#         if np.isinf(x0):
#             raise Exception("ray tracing error")
#         dist = np.sqrt((x0-source[0])**2 + (y0-source[1])**2)
#         gradx,grady = descent(1,[x0,y0])
#         step_distance = np.sqrt(gradx**2+grady**2)
#         step_inc = step_dist/step_distance
#         if dist/(step_distance*step_inc) <= 1.:
#             break

#         path.append([x0+gradx*step_inc,y0+grady*step_inc])
        
#         # in vincinity of the source, the traveltimefield is not very exact, just make a straight line jump
#         if np.abs(x0-source[0])<dx/1.5 and np.abs(y0-source[1])<dy/1.5:
#             break
#     else:
#         print("ERROR: ray tracing is not converging towards the source!")
#         print("source:",source,"receiver:",receiver)
#         raise Exception()
#     path.append([source[0],source[1]])
#     path = np.array(path)[::-1]
    

#     azim = np.arctan2(receiver[1]-source[1],receiver[0]-source[0])
#     azim = np.linspace(azim-0.02,azim+0.01,100)
#     step_dists = np.random.uniform(step_dist/10,step_dist,100)
#     testrays = np.column_stack((step_dists*np.cos(azim),step_dists*np.sin(azim)))
#     raylist = []
#     for ray in testrays:
#         step_inc = 1.
#         path2 = [[source[0],source[1]],[source[0]+ray[0],source[1]+ray[1]]]
#         total_dist = np.sqrt((receiver[1]-source[1])**2 + (receiver[0]-source[0])**2)
#         N = int(total_dist/step_dist*3)
#         if N>100000:
#             print("Ray tracing warning: path is sampled with more than 100k steps, consider using a larger stepsize")
#         for i in range(N):
#             x0,y0 = path2[-1]
#             if np.isinf(x0):
#                 raise Exception("ray tracing error")
#             dist = np.sqrt((x0-receiver[0])**2 + (y0-receiver[1])**2)
#             if i==0:
#                 previous_dist = dist
#             gradx,grady = descent(1,[x0,y0])
#             step_distance = np.sqrt(gradx**2+grady**2)
#             step_inc = step_dist/step_distance
#             if dist/(step_distance*step_inc) <= 1.:
#                 break
    
#             path2.append([x0-gradx*step_inc,y0-grady*step_inc])
            
#             # in vincinity of the source, the traveltimefield is not very exact, just make a straight line jump
#             if np.abs(x0-receiver[0])<3*dx and np.abs(y0-receiver[1])<3*dy:
#                 break
            
#             if previous_dist < dist:
#                 break
            
#             previous_dist = dist
            
#         else:
#             print("ERROR: ray tracing is not converging!")
#             print("source:",source,"receiver:",receiver)
#             raise Exception()
#         path2.append([receiver[0],receiver[1]])
#         path2 = np.array(path2)[::-1]
#         if len(path2) > 3:
#             raylist.append(path2)
    
    
#     plt.figure()
#     plt.pcolormesh(x,y,ttimefield,shading='nearest')
#     plt.plot(path[:,0],path[:,1])
#     for ray in raylist:
#         plt.plot(ray[:,0],ray[:,1],'.')
#     plt.plot(source[0],source[1],'yv')
#     plt.plot(receiver[0],receiver[1],'rv')
#     plt.show()    
    
    
#     kernel = np.zeros_like(ttimefield)
#     weights = np.zeros_like(kernel)
#     weights[np.abs(receiver[1]-y).argmin(),np.abs(receiver[0]-x).argmin()] = 1.
#     weights = gaussian_filter(weights,sigma=(wavelength/3./dy,wavelength/3./dx),
#                               mode='constant',truncate=3.0) # sigma=(ysig,xsig)
#     weights /= np.max(weights)

#     #path_list = []
#     ind = np.where(weights>0.1)
#     receivers = np.column_stack((x[ind[1]],y[ind[0]],weights[ind]))
        
#     path_list = []
#     for xrec,yrec,weight in receivers:
    
#         step_inc = 1.
#         path = [[xrec,yrec]]
#         total_dist = np.sqrt((yrec-source[1])**2 + (xrec-source[0])**2)
#         N = int(total_dist/step_dist*3)
#         if N>100000:
#             print("Ray tracing warning: path is sampled with more than 100k steps, consider using a larger stepsize")
#         for i in range(N):
#             x0,y0 = path[-1]
#             if np.isinf(x0):
#                 raise Exception("ray tracing error")
#             dist = np.sqrt((x0-source[0])**2 + (y0-source[1])**2)
#             gradx,grady = descent(1,[x0,y0])
#             step_distance = np.sqrt(gradx**2+grady**2)
#             step_inc = step_dist/step_distance
#             if dist/(step_distance*step_inc) <= 1.:
#                 break
    
#             path.append([x0+gradx*step_inc,y0+grady*step_inc])
            
#             # in vincinity of the source, the traveltimefield is not very exact, just make a straight line jump
#             if np.abs(x0-source[0])<dx/1.5 and np.abs(y0-source[1])<dy/1.5:
#                 break
#         else:
#             print("ERROR: ray tracing is not converging towards the source!")
#             print("source:",source,"receiver:",receiver)
#             raise Exception()
#         path.append([source[0],source[1]])
        
#         path_list.append(np.array(path)[::-1])
    
#     plt.figure()
#     plt.pcolormesh(x,y,weights,shading='nearest')
#     for path in path_list:
#         plt.plot(path[:,0],path[:,1],'magenta')
#     plt.show()
    
#     return path_list
    


def geographic2cartesian(lat,lon,depth):
    # lat,lon,depth -> x,y,z
    
    x = (6371.-depth)*np.cos(np.deg2rad(lon))*np.cos(np.deg2rad(lat))
    y = (6371.-depth)*np.sin(np.deg2rad(lon))*np.cos(np.deg2rad(lat))
    z = (6371.-depth)*np.sin(np.deg2rad(lat))
    
    return (x,y,z)

def cartesian2geographic(x,y,z):
    # x,y,z -> lat,lon,depth
    
    r = np.sqrt(x**2+y**2+z**2)
    lon = np.rad2deg(np.arctan2(y,x))
    lat = np.rad2deg(np.arcsin(z/r))
    depth = 6371.-r
    
    return (lat,lon,depth)

def rotate_spherical_coordinates(lat,lon,theta,phi):
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    lat_rot = np.arcsin(np.cos(theta_rad)*np.sin(lat_rad) - 
                        np.cos(lon_rad)*np.sin(theta_rad)*np.cos(lat_rad))
    lon_rot = np.arctan2(np.sin(lon_rad), np.tan(lat_rad)*np.sin(theta_rad) + 
                         np.cos(lon_rad)*np.cos(theta_rad)) - phi_rad
    return np.rad2deg(lat_rot),np.rad2deg(lon_rot)
    



if __name__ == "__main__":
    main()
