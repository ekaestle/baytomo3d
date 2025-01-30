#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:08:47 2022

@author: emanuel
"""


import numpy as np
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
import pyproj, glob, os, gzip, pickle, time
#from shapely.geometry import MultiLineString, LineString
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix
from scipy.interpolate import interp1d
#from scipy.interpolate import interp1d
from obspy.core.event import read_events
from obspy.clients.fdsn.client import Client
import cartopy.crs as ccrs
import cartopy.feature as cf
from . import FMM
#from numba import jit

"""
CLASS
"""

class BodywaveData(object):
    
    def __init__(self,sources=None,receivers=None,piercepoints=None,
                 paths=None,ttimes_base=None,residuals_obs=None,
                 model_bottom_depth=None,event_idx=[],projection=None):
        
        self.type = 'bodywaves'
        
        if sources is None:
            self.isactive = False
        else:
            self.isactive = True
            args = locals()
            for arg in args:
                if args[arg] is None:
                    raise Exception("Not all values in BodywaveData are defined (%s)." %arg)
        self.sources = sources
        self.receivers = receivers
        self.piercepoints = piercepoints
        self.paths = paths
        self.ttimes_base = ttimes_base
        self.data = residuals_obs
        self.event_idx = event_idx
        for dataset in self.event_idx:
            if (np.any(np.diff(self.event_idx[dataset])<0) or
                np.any(np.diff(self.event_idx[dataset])>1)):
                raise Exception("event idx array has to be sorted!")
        self.M = {} # matrix for mean calculation, needed to demean the synthetic travel times
        for dataset in self.event_idx:
            ievents = np.unique(self.event_idx[dataset])
            self.M[dataset] = lil_matrix((len(ievents),len(self.data[dataset])))
            for ievent in ievents:
                idx = ievent==self.event_idx[dataset]
                self.M[dataset][ievent,idx] = 1./np.sum(idx)
            self.M[dataset] = self.M[dataset].tocsr()
        self.projection = projection
        self.A = None
        self.model_bottom_depth = model_bottom_depth
        
        if self.isactive:
            self.minx = 1e99
            self.maxx = -1e99
            self.miny = 1e99
            self.maxy = -1e99
            for dataset in paths:
                for path in paths[dataset]:
                    self.minx = np.min([np.min(path[:,0]),self.minx])
                    self.maxx = np.max([np.max(path[:,0]),self.maxx])
                    self.miny = np.min([np.min(path[:,1]),self.miny])
                    self.maxy = np.max([np.max(path[:,1]),self.maxy])
                
            self.datastd = {}
            for dataset in residuals_obs:
                self.datastd[dataset] = np.ones(len(self.data[dataset]))*0.1
              
    #         self.set_collection_datastd()
            
    # def set_collection_datastd(self):
    #     if self.isactive:
    #         self.collection_datastd = {}
    #         for dataset in self.data:
    #             self.collection_datastd[dataset] = []
                
    # def append_collection_datastd(self):
    #     if self.isactive:
    #         for dataset in self.datastd:
    #             self.collection_datastd[dataset].append(np.mean(self.datastd[dataset]))

"""
FUNCTIONS
"""

def get_iasp91(modeldepths=None):
    
    # IASP91 model
    
    depths = np.array([
        0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   10,
        11,   12,   13,   14,   15,   16,   17,   18,   19,   20,   20,
        21,   22,   23,   24,   25,   26,   27,   28,   29,   30,   31,
        32,   33,   34,   35,   35,   40,   45,   50,   60,   70,   80,
        90,  100,  110,  120,  120,  130,  140,  150,  160,  170,  180,
        190,  200,  210,  210,  220,  230,  240,  250,  260,  270,  280,
        290,  300,  310,  320,  330,  340,  350,  360,  370,  380,  390,
        400,  410,  410,  420,  430,  440,  450,  460,  470,  480,  490,
        500,  510,  520,  530,  540,  550,  560,  570,  580,  590,  600,
        610,  620,  630,  640,  650,  660,  660,  670,  680,  690,  700,
        710,  720,  730,  740,  750,  760,  760,  770,  780,  790,  800,
        900, 1000, 1100, 1200, 1300, 1400, 1500, 2000, 2500, 2700, 2740,
        2740, 2750, 2800, 2850, 2889, 2889, 2900, 3000, 3100, 3200, 3300,
        3400, 3500, 4000, 4500, 5153, 5153, 5500, 6000, 6371])
    
    vp = np.array([ 
        5.8   ,  5.8   ,  5.8   ,  5.8   ,  5.8   ,  5.8   ,  5.8   ,
        5.8   ,  5.8   ,  5.8   ,  5.8   ,  5.8   ,  5.8   ,  5.8   ,
        5.8   ,  5.8   ,  5.8   ,  5.8   ,  5.8   ,  5.8   ,  5.8   ,
        6.5   ,  6.5   ,  6.5   ,  6.5   ,  6.5   ,  6.5   ,  6.5   ,
        6.5   ,  6.5   ,  6.5   ,  6.5   ,  6.5   ,  6.5   ,  6.5   ,
        6.5   ,  6.5   ,  8.04  ,  8.0406,  8.0412,  8.0418,  8.0429,
        8.0441,  8.0453,  8.0465,  8.0476,  8.0488,  8.05  ,  8.05  ,
        8.0778,  8.1056,  8.1333,  8.1611,  8.1889,  8.2167,  8.2444,
        8.2722,  8.3   ,  8.3   ,  8.3365,  8.373 ,  8.4095,  8.446 ,
        8.4825,  8.519 ,  8.5555,  8.592 ,  8.6285,  8.665 ,  8.7015,
        8.738 ,  8.7745,  8.811 ,  8.8475,  8.884 ,  8.9205,  8.957 ,
        8.9935,  9.03  ,  9.36  ,  9.3936,  9.4272,  9.4608,  9.4944,
        9.528 ,  9.5616,  9.5952,  9.6288,  9.6624,  9.696 ,  9.7296,
        9.7632,  9.7968,  9.8304,  9.864 ,  9.8976,  9.9312,  9.9648,
        9.9984, 10.032 , 10.0656, 10.0992, 10.1328, 10.1664, 10.2   ,
       10.79  , 10.8166, 10.8432, 10.8697, 10.8963, 10.9229, 10.9495,
       10.9761, 11.0026, 11.0292, 11.0558, 11.0558, 11.0738, 11.0917,
       11.1095, 11.1272, 11.2997, 11.464 , 11.6208, 11.7707, 11.9142,
       12.0521, 12.1849, 12.7944, 13.3697, 13.6076, 13.6564, 13.6564,
       13.6587, 13.6703, 13.6818, 13.6908,  8.0088,  8.028 ,  8.1995,
        8.3642,  8.5222,  8.6735,  8.818 ,  8.9558,  9.5437,  9.9633,
       10.2578, 11.0914, 11.1644, 11.227 , 11.2409]) 
    
    vs = np.array([
        3.36  , 3.36  , 3.36  , 3.36  , 3.36  , 3.36  , 3.36  , 3.36  ,
        3.36  , 3.36  , 3.36  , 3.36  , 3.36  , 3.36  , 3.36  , 3.36  ,
        3.36  , 3.36  , 3.36  , 3.36  , 3.36  , 3.75  , 3.75  , 3.75  ,
        3.75  , 3.75  , 3.75  , 3.75  , 3.75  , 3.75  , 3.75  , 3.75  ,
        3.75  , 3.75  , 3.75  , 3.75  , 3.75  , 4.47  , 4.4718, 4.4735,
        4.4753, 4.4788, 4.4824, 4.4859, 4.4894, 4.4929, 4.4965, 4.5   ,
        4.5   , 4.502 , 4.504 , 4.506 , 4.508 , 4.51  , 4.512 , 4.514 ,
        4.516 , 4.518 , 4.522 , 4.5394, 4.5568, 4.5742, 4.5916, 4.609 ,
        4.6264, 4.6438, 4.6612, 4.6786, 4.696 , 4.7134, 4.7308, 4.7482,
        4.7656, 4.783 , 4.8004, 4.8178, 4.8352, 4.8526, 4.87  , 5.07  ,
        5.0912, 5.1124, 5.1336, 5.1548, 5.176 , 5.1972, 5.2184, 5.2396,
        5.2608, 5.282 , 5.3032, 5.3244, 5.3456, 5.3668, 5.388 , 5.4092,
        5.4304, 5.4516, 5.4728, 5.494 , 5.5152, 5.5364, 5.5576, 5.5788,
        5.6   , 5.95  , 5.9759, 6.0019, 6.0278, 6.0538, 6.0797, 6.1057,
        6.1316, 6.1576, 6.1835, 6.2095, 6.2095, 6.2172, 6.2249, 6.2326,
        6.2402, 6.3138, 6.3833, 6.4489, 6.511 , 6.57  , 6.626 , 6.6796,
        6.921 , 7.1484, 7.2445, 7.2645, 7.2645, 7.267 , 7.2794, 7.2918,
        7.3015, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
        0.    , 0.    , 0.    , 0.    , 3.4385, 3.5   , 3.5528, 3.5645])
    
    if modeldepths is None:
        return(depths,vp,vs)
    else:
        vpfunc = interp1d(depths,vp)
        vsfunc = interp1d(depths,vs)
        return (vpfunc(modeldepths),vsfunc(modeldepths))
    
    
    
    

def trace_direct_ray(source,receiver,phase,
                     model_bottom_depth=None):
    
    model = TauPyModel(model="iasp91")
    
    if source[0]==receiver[0] and source[1]==receiver[1]:
        raise Exception("Source and receiver position are identical.")
        
    srclon,srclat,srcdepth = source
    rcvlon,rcvlat,rcvdepth = receiver
    
    dist_deg = locations2degrees(srclat,srclon,rcvlat,rcvlon)
    path = model.get_ray_paths(srcdepth,dist_deg,phase_list=(phase),
                               receiver_depth_in_km=rcvdepth)
    if len(path)==0:
        return None
    path = path[0].path
    
    # this part was only necessary for the old version of obspy.taup
    # in the new version, it's possible to define the receiver_depth_in_km
    # if rcvdepth>0:
    #     idxrcv = np.where(path['depth']>rcvdepth)[0][-1]
    #     func = interp1d(path['depth'][idxrcv:],path['time'][idxrcv:])
    #     arr_time = func(rcvdepth)
    #     func = interp1d(path['depth'],path['dist'])
    #     arr_dist = func(rcvdepth)
    #     path = path[:idxrcv+2]
    #     path[-1] = (path[-1][0],arr_time,arr_dist,rcvdepth)
    
    if model_bottom_depth is not None:
        if (path['depth']<model_bottom_depth).all():
            raise Exception("ray does not arrive from below!")
            
        eps = 1e-8 # otherwise the function below may not work if model_bottom_
        #depth coincides exactly with a layer discontinuity in the model (like 660km)
        # get the point along the ray path where it pierces the model bottom depth
        pierceidx = np.where((path['depth']-model_bottom_depth+eps)[:-1]*
                             (path['depth']-model_bottom_depth+eps)[1:]<0)[0][-1]
        dist_pierce = np.interp(model_bottom_depth,
                                path['depth'][pierceidx:pierceidx+2][::-1],
                                path['dist'][pierceidx:pierceidx+2][::-1])
        ttime_pierce = np.interp(model_bottom_depth,
                                 path['depth'][pierceidx:pierceidx+2][::-1],
                                 path['time'][pierceidx:pierceidx+2][::-1])
        # extract the part of the path that runs inside our model box
        path_box = path[pierceidx:]
        path_box[0] = (0,ttime_pierce,dist_pierce,model_bottom_depth)
        path = path_box
    
    # get the great circle latitudes and longitudes along the ray path
    g = pyproj.Geod(ellps='WGS84')
    lonlats = g.npts(srclon,srclat,rcvlon,rcvlat,98)
    lonlats = np.vstack(((srclon,srclat),lonlats,(rcvlon,rcvlat)))
    # interpolate the lats and lons to the points along path_box
    lons = np.interp(path['dist'],
                     np.linspace(0,path['dist'][-1],100),
                     lonlats[:,0])
    lats = np.interp(path['dist'],
                     np.linspace(0,path['dist'][-1],100),
                     lonlats[:,1])    
    path = np.column_stack((lons,lats,path['depth'],path['time']))
            
    return path
    


def compare_paths(dist,phase):
    from scipy.interpolate import interp1d
    
    refmod = np.loadtxt("../../Literatur/alpine_models/IASP91.csv",delimiter=',')
    func = interp1d(refmod[:,1],refmod[:,2],bounds_error=False,fill_value=1.5)
    
    dist = 67.
    
    x = np.arange(-500,6000,2)
    y = np.arange(2000,6500,2)
    X,Y = np.meshgrid(x,y)
    V = func(np.sqrt(X**2+Y**2))
    
    srcx = 0.
    srcy = 6371.
    rcvx = 6371.*np.cos(np.deg2rad(90-dist))
    rcvy = 6371.*np.sin(np.deg2rad(90-dist))
    
    ttimefield = FMM.calculate_ttime_field_samegrid(X, Y, V, source=(srcx,srcy))
    eikonalray = FMM.shoot_ray(x, y, ttimefield, (srcx,srcy), (rcvx,rcvy))[0]
    
    distkm = dist*111.19
    xflat = np.arange(-100,distkm+100,25)
    zflat = np.arange(0,2000,10)
    ztrans,_ = flat_earth_transform(zflat, 1.0)
    #ztrans = zflat ###############
    zregtrans = np.arange(0,np.max(ztrans),10)
    Xtrans,Ztrans = np.meshgrid(xflat,zregtrans)
    zinv,_ = flat_earth_transform(zregtrans, 1.0,inverse=True)
    #zinv = zregtrans ####################
    _,vtrans = flat_earth_transform(zinv, func(6371.-zinv))
    #vtrans = func(6371.-zinv) #################
    Vtrans = np.reshape(np.repeat(vtrans,len(xflat)),Xtrans.shape)
    sourceflat = (0,0)
    receiverflat = (distkm,0)
    
    ttimeflat = FMM.calculate_ttime_field_samegrid(Xtrans,Ztrans,Vtrans,sourceflat)
    flatraytrans = FMM.shoot_ray(Xtrans[0], Ztrans[:,0], ttimeflat, sourceflat, receiverflat)[0]
    zray,_ = flat_earth_transform(flatraytrans[:,1], 1.0,inverse=True)
    #zray = flatraytrans[:,1] ################
    pathdeg = np.interp(flatraytrans[:,0],np.linspace(0,distkm,100),np.linspace(0,dist,100))
    xflatpath = (6371.-zray)*np.cos(np.pi/2.-np.deg2rad(pathdeg))
    yflatpath = (6371.-zray)*np.sin(np.pi/2.-np.deg2rad(pathdeg))
    
    plt.figure()
    plt.pcolormesh(Xtrans,Ztrans,Vtrans)
    plt.contour(Xtrans,Ztrans,ttimeflat,levels=50)
    plt.plot(sourceflat[0],sourceflat[1],'o')
    plt.plot(receiverflat[0],receiverflat[1],'o')
    plt.plot(flatraytrans[:,0],flatraytrans[:,1])
    plt.gca().set_aspect('equal')
    plt.ylim(np.max(Ztrans),0)
    plt.show()
    
    # check that it works also in 3D
    X3D,Y3D,Z3D = np.meshgrid(Xtrans[0],np.linspace(-50,50,21),Ztrans[:,0])
    V3D = np.reshape(np.tile(vtrans,len(xflat)*21),X3D.shape)
    source3d = (0,0,0)
    receiver3d = (distkm,0,0)
    ttime3d = FMM.calculate_ttime_field_3D(X3D, Y3D, Z3D, V3D, source3d,
                                           refine_source_grid=False)
    path3d = FMM.shoot_ray_3D(X3D[0,:,0], Y3D[:,0,0], Z3D[0,0,:], ttime3d, source3d, receiver3d)[0]
    
    source3d_2 = (flatraytrans[900][0],0,flatraytrans[900][1])
    ttime3d_2 = FMM.calculate_ttime_field_3D(X3D, Y3D, Z3D, V3D, receiver3d,
                                           refine_source_grid=False)
    path3d_2 = FMM.shoot_ray_3D(X3D[0,:,0], Y3D[:,0,0], Z3D[0,0,:], ttime3d_2, receiver3d, source3d_2)[0]
    
    path3d_3 = FMM.shoot_ray(X3D[0,:,0], Z3D[0,0,:], ttime3d[10,:,:].T, source3d, receiver3d)[0]
    
    
    plt.figure()
    plt.pcolormesh(Xtrans,Ztrans,Vtrans)
    plt.contour(Xtrans,Ztrans,ttime3d[10,:,:].T,levels=50)
    plt.contour(Xtrans,Ztrans,ttimeflat,levels=50)
    plt.plot(sourceflat[0],sourceflat[1],'o')
    plt.plot(receiverflat[0],receiverflat[1],'o')
    plt.plot(flatraytrans[:,0],flatraytrans[:,1])
    plt.plot(path3d[:,0],path3d[:,2])
    #plt.contour(Xtrans,Ztrans,ttime3d_2[10,:,:].T,levels=50)
    plt.plot(source3d_2[0],source3d_2[2])
    plt.plot(path3d_2[:,0],path3d_2[:,2],'--')
    plt.plot(path3d_3[:,0],path3d_3[:,1],'--')
    plt.gca().set_aspect('equal')
    plt.ylim(np.max(Ztrans),0)
    plt.show()    
    
    model = TauPyModel(model="iasp91")
    path = model.get_ray_paths(0.,dist,phase_list=("P"))[0]
    taupray = path.path
    xpath = (6371.-taupray['depth'])*np.cos(np.pi/2-taupray['dist'])
    ypath = (6371.-taupray['depth'])*np.sin(np.pi/2.-taupray['dist'])
    
    plt.figure()
    #plt.pcolormesh(X,Y,V)
    plt.contour(X,Y,ttimefield,levels=np.linspace(0,2000,101))
    plt.plot(srcx,srcy,'o')
    plt.plot(rcvx,rcvy,'o')
    plt.plot(eikonalray[:,0],eikonalray[:,1])
    plt.plot(xflatpath,yflatpath,'--')
    plt.plot(xpath,ypath,linestyle='dotted')
    plt.gca().set_aspect('equal')
    plt.show()
    
    
    

# g = pyproj.Geod(ellps='WGS84')
# source = (0,0,50)
# receiver = (12,46,0)

# path = trace_direct_ray(source,receiver,"P",model_bottom_depth=700)


# fig = plt.figure()
# ax = plt.axes(projection="3d")
# u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
# xs = 6371*np.cos(u)*np.sin(v)
# ys = 6371*np.sin(u)*np.sin(v)
# zs = 6371*np.cos(v)
# ax.plot_wireframe(xs, ys, zs, color="red")
# ax.plot(x,y,z)
# plt.show()

def calculate_eikonal_paths(X,Y,Z,V,sources,receivers,verbose=False,Nprocs=1,Nsources=None):
    
    t0 = time.time()
    paths = []
    if verbose:
        print("Calculating Eikonal rays in 3D with the Fast Marching Method.")
        if Nsources is None:
            Nsources = len(sources)
    for i in range(len(sources)):
        if verbose:
            if i%1000 == 0:
                calc_time = time.time()-t0
                print("  Eikonal path 3D: %d/%d (elapsed time: %ds)" %(i*Nprocs,Nsources,calc_time))            
                    
        if i==0 or (sources[i]!=sources[i-1]).any():
            ttime_field = FMM.calculate_ttime_field_3D(X,Y,Z,V,sources[i],
                                                       refine_source_grid=False)
        # returns a list
        path = FMM.shoot_ray_3D(X[0,:,0],Y[:,0,0],Z[0,0,:],ttime_field,
                                sources[i],receivers[i],stepsize=0.66)
        paths.append(path[0])

    if verbose:
        calc_time = time.time()-t0
        print("  Eikonal path 3D: %d/%d (elapsed time: %ds)" %(i*Nprocs+1,Nsources,calc_time)) 

    return paths
 

def flat_earth_transform(depths,velocities,radius=6371.,inverse=False):
    # Müller, 1985
    if inverse:
        v_trans = velocities / (radius/(radius-depths))
        z_trans = radius * (1. - 1./np.exp(depths/radius))
    else:
        v_trans = velocities * radius/(radius-depths)
        z_trans = radius * np.log(radius/(radius-depths))
    
    return z_trans,v_trans


# alternative to the flat earth transform, we can also place a cartesian grid
# on top of a spherical earth and do the calculations in this new grid.
# this approach seems to be slower, however, and the grid is larger, because
# we will have empty cells above the surface and to left and right of the model
# region. The results are somewhat similar, not exactly the same, however...

# def cartesian2spherical_grid(x,y,z,V,projection_str,dx,dy,dz):
#     # maybe add a shift so that the central latitude is at the north pole?
#     # in that case there would be less empty gridcells 
    
#     v_func = RegularGridInterpolator((y,x,z),V,
#                                      bounds_error=False,fill_value=1e-3)
    
#     X,Y,Z = np.meshgrid(x,y,z)
    
#     xyz_sph = cartesian2spherical(np.column_stack((
#         X.flatten(),Y.flatten(),Z.flatten())), projection_str)
    
#     xreg = np.arange(np.floor(np.min(xyz_sph[:,0])),np.ceil(np.max(xyz_sph[:,0]))+dx,dx)
#     yreg = np.arange(np.floor(np.min(xyz_sph[:,1])),np.ceil(np.max(xyz_sph[:,1]))+dy,dy)
#     zreg = np.arange(np.floor(np.min(xyz_sph[:,2])),np.ceil(np.max(xyz_sph[:,2]))+dz,dz)
#     Xr,Yr,Zr = np.meshgrid(xreg,yreg,zreg)
    
#     xyz_cart = spherical2cartesian(np.column_stack((
#         Xr.flatten(),Yr.flatten(),Zr.flatten())), projection_str)
    
#     Vr = np.reshape(v_func(xyz_cart[:,[1,0,2]]),Xr.shape)

#     return Xr,Yr,Zr,Vr

#     # plt.figure()
#     # plt.contourf(Y[:,30,:],Z[:,30,:],V[:,30,:])
#     # plt.colorbar()
#     # plt.show()

#     # plt.figure()
#     # plt.contourf(zreg,yreg,Vr[:,6,:])
#     # plt.colorbar()
#     # plt.show()


# def cartesian2spherical(xyz,projection_str):
    
#     p = pyproj.Proj(projection_str)
#     lon,lat = p(xyz[:,0]*1000,xyz[:,1]*1000,inverse=True)
    
#     xsph = (6371.-xyz[:,2])*np.cos(np.deg2rad(lon))*np.cos(np.deg2rad(lat))
#     ysph = (6371.-xyz[:,2])*np.sin(np.deg2rad(lon))*np.cos(np.deg2rad(lat))
#     zsph = (6371.-xyz[:,2])*np.sin(np.deg2rad(lat))
    
#     return np.column_stack((xsph,ysph,zsph))

# def spherical2cartesian(xyz,projection_str):
    
#     p = pyproj.Proj(projection_str)
#     r = np.sqrt(np.sum(xyz**2,axis=1))
#     lon = np.rad2deg(np.arctan2(xyz[:,1],xyz[:,0]))
#     lat = np.rad2deg(np.arcsin(xyz[:,2]/r))
#     depth = 6371.-r
    
#     x,y = p(lon,lat)
#     return np.column_stack((x/1000.,y/1000.,depth))
    
#%%
def create_A_matrix(gridpoints,paths,verbose=False):
       
    if verbose:
        print("filling body-wave matrix...")
    
    x = np.unique(gridpoints[:,0])
    y = np.unique(gridpoints[:,1])
    z = np.unique(gridpoints[:,2])
        
    # gridlines (lines between gridpoints, delimiting cells) 
    dx = np.diff(x)
    xgrid = x[:-1]+dx/2.
    dz = np.diff(z)
    zgrid = z[:-1]+dz/2.
    dy = np.diff(y)
    ygrid = y[:-1]+dy/2.    

    tree = KDTree(gridpoints)
        
    # it can cause problems if the start or end points of the path are 
    # exactly on the grid lines. Points are then moved by this threshold
    thresh = np.min(np.hstack((dx,dy,dz)))/1000.
    
    A = {}
    
    for dataset in paths:
    
        # initializing sparse matrix
        A[dataset] = lil_matrix((len(paths[dataset]),len(x)*len(y)*len(z)),dtype='float32')

        for i in range(len(paths[dataset])):
            
            xpath = paths[dataset][i][:,0]
            ypath = paths[dataset][i][:,1]
            zpath = paths[dataset][i][:,2]
            
            if ((np.min(x)>xpath).any() or (np.max(x)<xpath).any() or
                (np.min(y)>ypath).any() or (np.max(y)<ypath).any() or
                (np.min(z)>zpath).any() or (np.max(z)<zpath).any()):
                raise Exception("path lies outside of the grid region")

            # x-points directly on one of the xgridlines:
            idx_x = np.where(np.abs(xgrid[:, None] - xpath[None, :]) < thresh)[1]
            xpath[idx_x] += thresh
            # y-points directly on one of the ygridlines:
            idx_y = np.where(np.abs(ygrid[:, None] - ypath[None, :]) < thresh)[1]
            ypath[idx_y] += thresh 
            # z-points directly on one of the ygridlines:
            idx_z = np.where(np.abs(zgrid[:, None] - zpath[None, :]) < thresh)[1]
            zpath[idx_z] += thresh 
            
            t = np.arange(len(xpath))
            
            # tx,ty,tz give the distance relative to the total path length t
            # where the path is crossing a gridline
            idx_grid, idx_t = np.where((xgrid[:, None] - xpath[None, :-1]) * 
                                       (xgrid[:, None] - xpath[None, 1:]) <= 0)
            tx = idx_t + (xgrid[idx_grid] - xpath[idx_t]) / (xpath[idx_t+1] - xpath[idx_t])
            
            idx_grid, idx_t = np.where((ygrid[:, None] - ypath[None, :-1]) * 
                                       (ygrid[:, None] - ypath[None, 1:]) <= 0)
            ty = idx_t + (ygrid[idx_grid] - ypath[idx_t]) / (ypath[idx_t+1] - ypath[idx_t])
            
            idx_grid, idx_t = np.where((zgrid[:, None] - zpath[None, :-1]) * 
                                       (zgrid[:, None] - zpath[None, 1:]) <= 0)
            tz = idx_t + (zgrid[idx_grid] - zpath[idx_t]) / (zpath[idx_t+1] - zpath[idx_t])
            
            # stack all gridline crossings in a sorted array
            t2 = np.hstack((0,np.sort(np.r_[tx, ty, tz]),len(xpath)))
            valid = np.append(np.diff(t2)>0,True)
            t2 = t2[valid]
            # interpolate from the relative units (distance along path in t)
            # to units of x,y,z
            x2 = np.interp(t2, t, xpath)
            y2 = np.interp(t2, t, ypath)
            z2 = np.interp(t2, t, zpath)
                        
            # find the cells corresponding to each path segment
            midpoints = np.column_stack((x2[:-1]+np.diff(x2)/2.,
                                         y2[:-1]+np.diff(y2)/2.,
                                         z2[:-1]+np.diff(z2)/2.))
            nndist,nnidx = tree.query(midpoints)
               
            dz2 = np.diff(z2)
            # compensate for sphericity (distances get smaller with increasing depth)
            dx2 = np.diff(x2) * (6371-z2[:-1]+dz2/2.)/6371.
            dy2 = np.diff(y2) * (6371-z2[:-1]+dz2/2.)/6371.
            dists = np.sqrt( dx2**2 + dy2**2 + dz2**2)      
            if np.sum(dists) == 0:
                print(i,[xpath,y,z])
                raise Exception("a zero distance path is not valid!")
                

            if False:
                fig = plt.figure()
                ax = fig.add_subplot(1,3,1)
                for i in range(len(x2)-1):
                    xp,yp,zp = ([x2[i],x2[i+1]],[y2[i],y2[i+1]],[z2[i],z2[i+1]])
                    plt.plot(xp, yp)
                    plt.text(np.mean(xp), np.mean(yp), np.around(dists[i],2))
                lines = []
                for xi in xgrid:
                    lines.append(((xi, np.min(y)), (xi, np.max(y))))
                for yi in ygrid:
                    lines.append(((np.min(x), yi), (np.max(x), yi)))
                lc = LineCollection(lines, color="gray", lw=1, alpha=0.5)
                ax.add_collection(lc)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_aspect('equal')
                ax = fig.add_subplot(1,3,2)
                for i in range(len(x2)-1):
                    xp,yp,zp = ([x2[i],x2[i+1]],[y2[i],y2[i+1]],[z2[i],z2[i+1]])
                    plt.plot(xp, zp)
                    plt.text(np.mean(xp), np.mean(zp), np.around(dists[i],2))
                lines = []
                for xi in xgrid:
                    lines.append(((xi, np.min(z)), (xi, np.max(z))))
                for zi in zgrid:
                    lines.append(((np.min(x), zi), (np.max(x), zi)))
                lc = LineCollection(lines, color="gray", lw=1, alpha=0.5)
                ax.add_collection(lc)
                ax.set_xlabel("x")
                ax.set_ylabel("z")
                ax.set_aspect('equal')
                ax = fig.add_subplot(1,3,3)
                for i in range(len(x2)-1):
                    xp,yp,zp = ([x2[i],x2[i+1]],[y2[i],y2[i+1]],[z2[i],z2[i+1]])
                    plt.plot(yp, zp)
                    plt.text(np.mean(yp), np.mean(zp), np.around(dists[i],2))
                lines = []
                for yi in ygrid:
                    lines.append(((yi, np.min(z)), (yi, np.max(z))))
                for zi in zgrid:
                    lines.append(((np.min(y), zi), (np.max(y), zi)))
                lc = LineCollection(lines, color="gray", lw=1, alpha=0.5)
                ax.add_collection(lc)
                ax.set_xlabel("y")
                ax.set_ylabel("z")
                ax.set_aspect('equal')

            # sometimes a path can traverse a grid cell twice
            idxunique, sortidx, cntunique= np.unique(nnidx,return_counts=True,
                                                     return_index=True)
            if (cntunique>1).any():
                for doubleidx in np.where(cntunique>1)[0]:
                    dists[nnidx==idxunique[doubleidx]] = np.sum(
                        dists[nnidx==idxunique[doubleidx]])

            if (dists==0.).any():
                print("Warning! One of the weights in the A matrix is zero!")
            if not np.array_equal(idxunique,nnidx[sortidx]):
                raise Exception("should be the same")
            A[dataset][i,nnidx[sortidx]] = dists[sortidx]
                            
        if verbose:
            print("Matrix filling done")
            
        A[dataset] = A[dataset].tocsc() # much faster calculations. But slower while filling
        # csr would be even a little bit faster with matrix vector product
        # calculations, but csc is much faster when extracting columns which is
        # necessary when updating the model predictions (see calculate_ttimes)

    return A
            

def forward_module(p_slo,s_slo,A,ttimes_base=None,idx_mod_gpts=None,
                   ttimes_ref=None,p_slo_ref=None,s_slo_ref=None):
    
    ttimes = {}
    
    for dataset in A:
        
        if dataset[:2]=="s ":
            slowness_model = s_slo
            model_ref = s_slo_ref
        elif dataset[:2]=="p ":
            slowness_model = p_slo
            model_ref = p_slo_ref
        else:
            raise Exception(f"did not recognize whether {dataset} is a P or S travel time dataset.")
        
        if idx_mod_gpts is None:
            ttimes[dataset] = ttimes_base[dataset] + A[dataset] * slowness_model
            
        else: # slowness_model and model_ref are already expected to be reduced to the indices in idx_mod_gpts
            ttimes[dataset] = (
                ttimes_ref[dataset] + A[dataset][:,idx_mod_gpts] * 
                (slowness_model-model_ref) )
            
    return ttimes


# using a matrix is much faster than finding the indices of each event to
# calcualte the event-wise means.
def demean_ttimes(ttimes,M):
    means = M*ttimes # has the same size as the number of events
    return ttimes - np.repeat(means,np.diff(M.indptr))

# @jit(nopython=True)
# def demean_loop(ttimes,event_idx):
#     for i in range(len(ttimes)):

# function just for testing, otherwise unused
def get_phi(residuals_obs,ttimes_syn,datastd,M):
    
    phi = 0
    for dataset in residuals_obs:
        residuals_syn = demean_ttimes(ttimes_syn[dataset],M[dataset])
        residuals = (residuals_obs[dataset] - residuals_syn)
        phi += np.sum((residuals/datastd[dataset])**2)   
    return phi


def loglikelihood(residuals_obs,ttimes_syn,datastd,M,
                  norm='L2',distribution=None,foutlier=None,widthoutlier=None,
                  return_datastd=False):
    
    loglikelihood = 0.
    total_residuals = 0.
    stadev = []
    
    for dataset in residuals_obs:
        
        # splitidx = np.where(np.diff(event_idx[dataset])!=0)[0]+1
        # event_ttimes = np.split(ttimes_syn[dataset],splitidx)
        # repeats = np.diff(np.concatenate((np.array([0]),splitidx,np.array([len(event_idx[dataset])]))))
        # means = np.array(list(map(np.mean,event_ttimes)))
        # residuals_syn = ttimes_syn[dataset] - np.repeat(means,repeats)
        
        residuals_syn = demean_ttimes(ttimes_syn[dataset],M[dataset])
        
        residuals = (residuals_obs[dataset] - residuals_syn)
        stadev.append(residuals)
        total_residuals += np.sum(np.abs(residuals))
        
        if distribution != 'outliers':
            if norm == 'L2':
                # the np.sqrt(2*np.pi) term is not necessary (always cancels out in the
                # likelihood ratios), but is kept here to make it more comparable to
                # the outlier model likelihoods which include the 2pi term.
                loglikelihood += np.sum(-np.log(np.sqrt(2*np.pi)*datastd[dataset]) - 
                                        0.5*(residuals/datastd[dataset])**2)
    
            elif norm == 'L1':
                loglikelihood += np.sum(-np.log(2*datastd[dataset]) - 
                                        0.5*np.abs(residuals/datastd[dataset]))
                
            else:
                raise Exception("not a valid norm, choose from 'L1', 'L2'.")
                
        else:
            # data_std can be an array or a single value
            # what happens if a residual is outside the widthoutlier range?
            # the loglikelihood_uniform should be 0 for these residuals, but that would only happen rarely...
            likelihood_uniform = foutlier[dataset]/widthoutlier[dataset]
            if norm == 'L2':
                likelihood_gauss = ((1-foutlier[dataset]) *
                                    1./(np.sqrt(2*np.pi)*datastd[dataset]) *
                                    np.exp(-0.5*(residuals/datastd[dataset])**2))
            elif norm == 'L1':
                likelihood_gauss = ((1-foutlier[dataset]) *
                                    1./(2*datastd[dataset]) *
                                    np.exp(-0.5*np.abs(residuals/datastd[dataset])))
            # waterlevel 1e-300 to make sure that there is no inf in np.log
            loglikelihood += np.sum( np.log( likelihood_gauss + likelihood_uniform + 1e-300 ) )
        
    if return_datastd:
        return total_residuals,loglikelihood,np.std(np.hstack(stadev))
    else:        
        return total_residuals,loglikelihood
        
   
# using jit, but that doesn't make a significant difference
def loglikelihood_notused(residuals_obs,ttimes_syn,datastd,M,
                  norm='L2',distribution=None):
    
    loglikelihood = 0.
    total_residuals = 0.
    
    for dataset in residuals_obs:
        
        residuals_syn = demean_ttimes(ttimes_syn[dataset],M[dataset])
        ll,res = loglike_jit(residuals_obs[dataset],residuals_syn,
                             datastd[dataset],norm,distribution)
        loglikelihood += ll
        total_residuals += res
        
    return total_residuals,loglikelihood
        
   
#@jit(nopython=True) # using just-in-time compilation to speed up the loglikelihood calculations
def loglike_jit(residuals_obs,residuals_syn,datastd,
                norm,distribution):
    
    residuals = (residuals_obs - residuals_syn)
    residualsum = np.sum(np.abs(residuals))
    if distribution == 'normal':
        if norm == 'L2':
            # the np.sqrt(2*np.pi) term is not necessary (always cancels out in the
            # likelihood ratios), but is kept here to make it more comparable to
            # the outlier model likelihoods which include the 2pi term.
            loglikelihood = np.sum(-np.log(np.sqrt(2*np.pi)*datastd) - 
                                    ((residuals/datastd)**2)/2.)

        elif norm == 'L1':
            loglikelihood = np.sum(-np.log(2*datastd) - 
                                    np.abs(residuals/datastd)/2.)
            
        else:
            raise Exception("not a valid norm, choose from 'L1', 'L2'.")
    else:
        raise Exception("not a valid distribution type, choose from 'normal',")
        
    return loglikelihood,residualsum

def maximum_likelihood_datastd(residuals):
    # eq. (B5) from Sambridge 2014, A Parallel Tempering algorithm for ...
    return np.sqrt(1./len(residuals) * np.sum(residuals**2))
    

def read_bodywave_module(output_folder):
    
    module_fname = os.path.join(output_folder,"data_modules","bodywavedata.pgz")
    if os.path.isfile(module_fname):
        print("found bodywave module file, reading...")
        with gzip.open(module_fname, "rb") as f:
            return pickle.load(f)
    else:
        return None
    
def dump_bodywave_module(output_folder,module):
    
    module_fname = os.path.join(output_folder,"data_modules","bodywavedata.pgz")
    if not os.path.exists(os.path.dirname(module_fname)):
        os.makedirs(os.path.dirname(module_fname))
    with gzip.open(module_fname,"wb") as f:
        pickle.dump(module,f)
    
    
def read_data(directory_list,mpi_rank,mpi_comm,model_bottom_depth=700,
              stationlist=None, projection=None,plotting=False,output_folder=None):
    
    if directory_list is None or directory_list == []:
        bwd = BodywaveData()
        if mpi_rank==0:
            dump_bodywave_module(output_folder, bwd)
        return bwd

    bwd = None
    p = None
    readdata = False
    if mpi_rank==0:
        bwd = read_bodywave_module(output_folder)
        if bwd is not None:
            if not bwd.isactive:
                bwd = None
        if bwd is None:
            readdata = True
    readdata = mpi_comm.bcast(readdata,root=0)
    
    # if bodywave module file was found, no need to re-read the data
    if not readdata:
        return bwd
    
    # read data (only mpi rank 0) and calculate paths (all mpi ranks in parallel)
    if type(directory_list) == str:
        directory_list = [directory_list]
    
    stationdict = {}
    sources = {}
    receivers = {}
    paths = {}
    event_idx = {}
    data_bw = {}
    phase_info = {}
    stats = []
    datasets = []
    if mpi_rank == 0:
        print("\n# Reading Body Wave Input Files #")
        for xml_folder in directory_list:

            print(f"Reading event xml files in {xml_folder}")
            
            eventfiles = glob.glob(os.path.join(xml_folder,"*.xml"))
        
            if len(eventfiles)==0:
                print("no .xml files found in",xml_folder)
                continue
        
            print("checking station dictionary")
            if stationlist is not None:
                with open(stationlist,"r") as f:
                    for line in f.readlines():
                        staid,lat,lon, elevation = line.split()
                        stationdict[staid] = {}
                        stationdict[staid]['longitude'] = float(lon)
                        stationdict[staid]['latitude'] = float(lat)
                        stationdict[staid]['elevation'] = float(elevation)
            elif os.path.isfile(os.path.join(xml_folder,"stationlist.dat")):
                with open(os.path.join(xml_folder,"stationlist.dat"),"r") as f:
                    for line in f.readlines():
                        staid,lat,lon,elevation = line.split()
                        stationdict[staid] = {}
                        stationdict[staid]['longitude'] = float(lon)
                        stationdict[staid]['latitude'] = float(lat)  
                        stationdict[staid]['elevation'] = float(elevation)
         
            print_info = True
            for eventfile in eventfiles:
                cat = read_events(eventfile)
                for event in cat:
                    for pick in event.picks:
                        staid = pick.waveform_id.network_code+"."+pick.waveform_id.station_code
                        try:
                            stationdict[staid]
                        except:
                            if print_info:
                                print("Stationfile missing/incomplete. Trying to download station information from FDSN webservices.")
                                print_info = False
                            for cl in ["IRIS","GFZ","ORFEUS","RESIF","LMU","ETH","BGR","INGV"]:
                                client = Client(cl)
                                try:
                                    inventory = client.get_stations(
                                        network=pick.waveform_id.network_code, 
                                        station=pick.waveform_id.station_code,
                                        starttime=pick.time-5,endtime=pick.time+5)
                                    break
                                except:
                                    continue
                            else:
                                print("station not found!",pick.waveform_id)
                                continue
                            stationdict[staid] = {}
                            stationdict[staid]['longitude'] = inventory[0][0].longitude
                            stationdict[staid]['latitude'] = inventory[0][0].latitude
                            stationdict[staid]['elevation'] = inventory[0][0].elevation
                        
            # write station information to file
            if not print_info:
                with open(os.path.join(xml_folder,"stationlist.dat"),"w") as f:
                    for staid in stationdict:
                        f.write("{:<9}".format(staid)+"\t%8.5f\t%9.5f\t%6.1f\n" %(
                            stationdict[staid]['latitude'],
                            stationdict[staid]['longitude'],
                            stationdict[staid]['elevation']))
    
            elevation_correction = True
            if elevation_correction:
                print("correcting for station elevations using constant velocity (Vp=5.0 km/s Vs=2.8 km/s)")
            if xml_folder.endswith("/"):
                dataset = xml_folder.split("/")[-2]
            else:
                dataset = xml_folder.split("/")[-1]
                
            dataset_s = "s "+dataset
            dataset_p = "p "+dataset
            datasets = [dataset_s,dataset_p]
            
            for dataset in datasets:
                paths[dataset] = []
                sources[dataset] = []
                receivers[dataset] = []
                event_idx[dataset] = []
                data_bw[dataset] = []
                phase_info[dataset] = []
            for ievent,eventfile in enumerate(eventfiles):
                cat = read_events(eventfile)
                event = cat[0]
                source = (event.origins[0].longitude,event.origins[0].latitude,event.origins[0].depth)
                sourcetime = event.origins[0].time
                for pick in event.picks:
                    if pick.phase_hint not in ['S','SKS','P']:
                        print(pick.phase_hint)
                        raise Exception("currently only P,Pdiff, S and SKS phases are accepted (quakeML.event.pick.phase_hint has to be 'P', 'S' or 'SKS')")
                    staid = pick.waveform_id.network_code+"."+pick.waveform_id.station_code
                    try:
                        if stationdict[staid]['elevation']<0. and elevation_correction:
                            receiver = (stationdict[staid]['longitude'],stationdict[staid]['latitude'],-stationdict[staid]['elevation']/1000.)
                        else:
                            receiver = (stationdict[staid]['longitude'],stationdict[staid]['latitude'],0)
                    except:
                        print("no receiver information, skipping",staid)
                        continue
                    if pick.phase_hint[-1] == 'S':
                        dataset = dataset_s
                        vcorr = 2800
                        pickphase = pick.phase_hint
                    elif pick.phase_hint[-1] == 'P':
                        dataset = dataset_p
                        vcorr = 5000
                        pickphase = ("P","Pdiff") # can be both
                    else:
                        raise Exception("unknown phase")
                    sources[dataset].append(source)
                    receivers[dataset].append(receiver)
                    event_idx[dataset].append(ievent)
                    phase_info[dataset].append(pickphase)
                    if elevation_correction:
                        if stationdict[staid]['elevation'] > 0:
                            elev_corr = stationdict[staid]['elevation']/vcorr
                    else:
                        elev_corr = 0.
                    data_bw[dataset].append(pick.time-sourcetime-elev_corr)
                    
        # done reading datasets
        
        # remove empty dictionaries
        keys_to_remove = []
        for dataset in data_bw:
            if len(data_bw[dataset])==0:
                keys_to_remove.append(dataset)
        for key in keys_to_remove:
            del data_bw[key]
            del event_idx[key]
            del sources[key]
            del receivers[key]
            del phase_info[key]
            del paths[key]        
        
        # demean observed data
        for dataset in data_bw:
            data_bw[dataset] = np.array(data_bw[dataset])
            event_idx[dataset] = np.array(event_idx[dataset])
            sources[dataset] = np.vstack(sources[dataset])
            receivers[dataset] = np.vstack(receivers[dataset])
            phase_info[dataset] = np.vstack(phase_info[dataset])
            for ievent in np.unique(event_idx[dataset]):
                idx = event_idx[dataset]==ievent
                data_bw[dataset][idx] -= np.mean(data_bw[dataset][idx])
                stats.append(receivers[dataset])
        
        # get projection
        stats = np.vstack(stats)
        stats = np.unique(stats,axis=0)
        meanlon = np.around(np.median(stats[:,0]))
        meanlat = np.around(np.median(stats[:,1]))
        if projection is not None:
            projection_str = projection
        else:
            projection_str = "+proj=tmerc +datum=WGS84 +lat_0=%f +lon_0=%f" %(meanlat,meanlon)
        p = pyproj.Proj(projection_str)
        
        ttimes_base = {}
        for dataset in data_bw:
            ttimes_base[dataset] = np.zeros(len(sources[dataset]))
        
        mpi_size = mpi_comm.Get_size()
        print(f"calculating bodywave paths using Tau-P on {mpi_size} parallel processes.")
        
        # mpi rank has finished its solo work, going back to all mpi processes
        
    # distribute projection information
    p = mpi_comm.bcast(p,root=0)
    datasets = mpi_comm.bcast(list(data_bw.keys()),root=0)
    
    # this is the part where the other mpi processes also play a role
    for dataset in datasets:
        
        # mpi rank 0 sends information
        if mpi_rank==0:
    
            for rank in range(mpi_size):
                startidx = rank*int(len(sources[dataset])/mpi_size)
                if rank==mpi_size-1:
                    endidx=None
                else:
                    endidx = (rank+1)*int(len(sources[dataset])/mpi_size)
                if rank>0:
                    #print("mpi rank 0 sending data to mpi rank",rank)
                    #print("first data point is",sources[dataset][startidx:endidx][0],receivers[dataset][startidx:endidx][0])
                    mpi_comm.send(sources[dataset][startidx:endidx],rank,tag=rank*3)
                    mpi_comm.send(receivers[dataset][startidx:endidx],rank,tag=rank*3+1)
                    mpi_comm.send(phase_info[dataset][startidx:endidx],rank,tag=rank*3+2)
                else:
                    sources_subset = sources[dataset][startidx:endidx]
                    receivers_subset = receivers[dataset][startidx:endidx]
                    phase_info_subset = phase_info[dataset][startidx:endidx]
      
        # other ranks receive information
        else:
            sources_subset = mpi_comm.recv(tag=mpi_rank*3)
            receivers_subset = mpi_comm.recv(tag=mpi_rank*3+1)
            phase_info_subset = mpi_comm.recv(tag=mpi_rank*3+2)
        
        valid = np.ones(len(sources_subset),dtype=bool)
        path_coords = []
        path_ttimes = np.zeros(len(sources_subset))
        #print("I am mpi rank",mpi_rank,"and I am working on a set of",len(sources_subset),"data points. My first data point is",sources_subset[0],receivers_subset[0])
        for i in range(len(sources_subset)):
            if mpi_rank==0:
                if i*mpi_size % 1000 == 0:
                    print("    ",i*mpi_size,"/",len(sources[dataset]),"(%s)" %dataset)
            path = trace_direct_ray(sources_subset[i],receivers_subset[i],
                                    phase_info_subset[i],model_bottom_depth=model_bottom_depth)
            if path is None:
                print("could not get path in reference model!",sources_subset[i],receivers_subset[i])
                valid[i] = False
                continue
            pathx,pathy = p(path[:,0],path[:,1])
            path_coords.append(np.column_stack((pathx/1000.,pathy/1000,path[:,2])))
            path_ttimes[i] = path[0][-1]
            
        # the gathered arrays are sorted according to the rank which is important
        path_coords = mpi_comm.gather(path_coords,root=0)
        path_ttimes = mpi_comm.gather(path_ttimes,root=0)
        valid = mpi_comm.gather(valid,root=0)
        # remaining part is again only done on rank 0
    
        if mpi_rank==0:
            print("Done getting Tau-P paths.\n")
            for pathlist in path_coords:
                paths[dataset] = paths[dataset] + list(pathlist)
            ttimes_base[dataset] = np.hstack(path_ttimes)
            valid = np.hstack(valid)
            data_bw[dataset] = data_bw[dataset][valid]
            event_idx[dataset] = event_idx[dataset][valid]
            sources[dataset] = sources[dataset][valid]
            receivers[dataset] = receivers[dataset][valid]    
        
    # for i in range(len(sources[dataset])):
    #     if i%1000==0:
    #         print(i,"/",len(sources[dataset]))
    #     # taking P and Pdiff into account, since the ttimes are continuous
    #     path = trace_direct_ray(sources[dataset][i],
    #                             receivers[dataset][i],("P","Pdiff"),
    #                             model_bottom_depth=model_bottom_depth)
    #     if path is None:
    #         print("could not get path in reference model!",sources[dataset][i],receivers[dataset][i])
    #         idx_invalid.append(i)
    #         continue
    #     pathx,pathy = p(path[:,0],path[:,1])
    #     paths[dataset].append(np.column_stack((pathx/1000.,pathy/1000,path[:,2])))
    #     ttimes_base[dataset][i] = path[0][-1]
    # ttimes_base[dataset] = np.delete(ttimes_base[dataset],idx_invalid)
    # data_bw[dataset] = np.delete(data_bw[dataset],idx_invalid)
    # event_idx[dataset] = np.delete(event_idx[dataset],idx_invalid)
    # sources[dataset] = np.delete(sources[dataset],idx_invalid,axis=0)
    # receivers[dataset] = np.delete(receivers[dataset],idx_invalid,axis=0)
    
    if mpi_rank==0:
        pierce_points = {}
        for dataset in paths:
            pierce_points[dataset] = np.zeros((len(paths[dataset]),3))
            for i,path in enumerate(paths[dataset]):
                pierce_points[dataset][i] = path[0]
                
                
        if plotting:
            figpath = os.path.join(output_folder,"indata_figures")
            if not os.path.exists(figpath):
                os.makedirs(figpath)
                
            # uses a WGS84 ellipsoid as default to create the transverse mercator projection
            proj = ccrs.TransverseMercator(central_longitude=meanlon,central_latitude=meanlat)
              
            plt.ioff()
            for dataset in data_bw:
            
                fig = plt.figure(figsize=(10,8))
                axm = plt.axes(projection=proj)

                recx,recy = p(receivers[dataset][:,0],receivers[dataset][:,1])
                recxy = np.column_stack((recx,recy))
                boxpaths = np.hstack((np.split(pierce_points[dataset][:,:2]*1000,len(recxy)),
                                      np.split(recxy,len(recxy))))
                lc = LineCollection(boxpaths, linewidths=0.2, color='black',alpha=0.5)
                axm.add_collection(lc)

                #axm.plot(np.unique(sources[dataset],axis=0)[:,0],
                #         np.unique(sources[dataset],axis=0)[:,1],'yo',
                #         transform = ccrs.PlateCarree(),label='EQ sources')
                piercelons,piercelats = p(pierce_points[dataset][:,0]*1000,
                                          pierce_points[dataset][:,1]*1000,
                                          inverse=True)
                axm.plot(piercelons,piercelats,'r.',ms=1,alpha=0.5,
                         transform = ccrs.PlateCarree(),label='pierce points')
            
                axm.plot(stats[:,0],stats[:,1],'gv',ms = 2,label='stations',
                         transform = ccrs.PlateCarree())
                gl = axm.gridlines(crs=ccrs.PlateCarree(), draw_labels=["bottom","left"],
                                   #xlocs=np.arange(-5,25,1),ylocs=np.arange(39,55,1),
                                   linewidth=0.5, color='black', alpha=0.5, linestyle='--')
                axm.coastlines(resolution='50m')
                axm.add_feature(cf.BORDERS.with_scale('50m'))
                axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
                axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
                axm.legend(loc='upper right')
                plt.savefig(os.path.join(figpath,"figure bodywave indata %s.jpg" %dataset),
                            bbox_inches='tight',dpi=200)
                plt.close(fig)
            
                for testdepth in [100,200,300]:
                    for dataset in paths:
                        fig = plt.figure(figsize=(10,8))
                        axm = plt.axes(projection=proj)
                        lines = []
                        for path in paths[dataset]:
                            subpath = path[path[:,2]<=testdepth]
                            lines.append(subpath[:,:2]*1000.)
                        lc = LineCollection(lines, linewidths=0.2, color='black',alpha=0.5)
                        axm.add_collection(lc)
                        piercelons,piercelats = p(pierce_points[dataset][:,0]*1000,
                                                  pierce_points[dataset][:,1]*1000,
                                                  inverse=True)
                        axm.plot(piercelons,piercelats,'r.',ms=1, alpha=0.5,
                                 transform = ccrs.PlateCarree(),label='pierce points')
                        gl = axm.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                           linewidth=0.5, color='black', alpha=0.5, linestyle='--')
                        axm.plot(stats[:,0],stats[:,1],'gv',ms = 2,label='stations',
                                 transform = ccrs.PlateCarree())
                        axm.coastlines(resolution='50m')
                        axm.add_feature(cf.BORDERS.with_scale('50m'))
                        axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
                        axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
                        axm.legend(loc='upper right')
                        plt.savefig(os.path.join(figpath,"figure bodywave indata above %d km %s.jpg" %(testdepth,dataset)),
                                    bbox_inches='tight',dpi=200)
                        plt.close(fig)
                
             
        bwd =  BodywaveData(sources=sources,receivers=receivers,
                            piercepoints=pierce_points,paths=paths,
                            ttimes_base=ttimes_base,residuals_obs=data_bw,
                            model_bottom_depth=model_bottom_depth,
                            event_idx=event_idx,projection=projection_str)
        
        dump_bodywave_module(output_folder, bwd)
            
    return bwd
    

def test():
    
    eventfiles = glob.glob("../dataset_teleseismic_p/*.xml")
    
    stationdict = {}
    with open("stationlist_bodywavedataset.dat","r") as f:
        for line in f.readlines():
            staid,lat,lon = line.split()
            stationdict[staid] = {}
            stationdict[staid]['longitude'] = float(lon)
            stationdict[staid]['latitude'] = float(lat)
    for eventfile in eventfiles:
        cat = read_events(eventfile)
        for event in cat:
            for pick in event.picks:
                staid = pick.waveform_id.network_code+"."+pick.waveform_id.station_code
                try:
                    stationdict[staid]
                except:
                    for cl in ["IRIS","GFZ","ORFEUS","RESIF","LMU","ETH","BGR","INGV"]:
                        client = Client(cl)
                        try:
                            inventory = client.get_stations(
                                network=pick.waveform_id.network_code, 
                                station=pick.waveform_id.station_code,
                                starttime=pick.time-5,endtime=pick.time+5)
                            break
                        except:
                            continue
                    else:
                        print("station not found!",pick.waveform_id)
                        continue
                    stationdict[staid] = {}
                    stationdict[staid]['longitude'] = inventory[0][0].longitude
                    stationdict[staid]['latitude'] = inventory[0][0].latitude
    with open("stationlist_bodywavedataset.dat","w") as f:
        for staid in stationdict:
            f.write(staid+"\t%8.5f\t%8.5f\n" %(stationdict[staid]['latitude'],
                                               stationdict[staid]['longitude']))
            
    projection_str = "+proj=tmerc +datum=WGS84 +lat_0=%f +lon_0=%f" %(47,10)
    p = pyproj.Proj(projection_str)   

    sources = {}
    receivers = {}
    paths = {}
    event_idx = {}
    data_bw = {}
    dataset = "P wave AlpArray"
    paths[dataset] = []
    sources[dataset] = []
    receivers[dataset] = []
    event_idx[dataset] = []
    data_bw[dataset] = []
    for ievent,eventfile in enumerate(eventfiles):
        cat = read_events(eventfile)
        event = cat[0]
        source = (event.origins[0].longitude,event.origins[0].latitude,event.origins[0].depth)
        sourcetime = event.origins[0].time
        for pick in event.picks:
            if pick.phase_hint!='P':
                print(pick.phase_hint)
                raise Exception("only accepts P phases")
            staid = pick.waveform_id.network_code+"."+pick.waveform_id.station_code
            try:
                receiver = (stationdict[staid]['longitude'],stationdict[staid]['latitude'],0)
            except:
                print("no receiver information, skipping",staid)
            sources[dataset].append(source)
            receivers[dataset].append(receiver)
            event_idx[dataset].append(ievent)
            data_bw[dataset].append(pick.time-sourcetime)
    data_bw[dataset] = np.array(data_bw[dataset])
    event_idx[dataset] = np.array(event_idx[dataset])
    sources[dataset] = np.vstack(sources[dataset])
    receivers[dataset] = np.vstack(receivers[dataset])
    # demean observed data
    for dataset in data_bw:
        for ievent in np.unique(event_idx[dataset]):
            idx = event_idx[dataset]==ievent
            data_bw[dataset][idx] -= np.mean(data_bw[dataset][idx])
            
    
    paths = {}
    ttimes_base = {}
    paths[dataset] = []
    ttimes_base[dataset] = np.zeros(len(sources[dataset]))
    idx_invalid = []
    for i in range(len(sources[dataset])):
        path = trace_direct_ray(sources[dataset][i],
                                receivers[dataset][i],"P",
                                model_bottom_depth=700)
        if path is None:
            print("could not get path in reference model!",sources[dataset][i],receivers[dataset][i])
            idx_invalid.append(i)
            continue
        pathx,pathy = p(path[:,0],path[:,1])
        paths[dataset].append(np.column_stack((pathx/1000.,pathy/1000,path[:,2])))
        ttimes_base[dataset][i] = path[0][-1]
    ttimes_base[dataset] = np.delete(ttimes_base[dataset],idx_invalid)
    data_bw[dataset] = np.delete(data_bw[dataset],idx_invalid)
    event_idx[dataset] = np.delete(event_idx[dataset],idx_invalid)
    sources[dataset] = np.delete(sources[dataset],idx_invalid)
    receivers[dataset] = np.delete(receivers[dataset],idx_invalid)
    
    minx = 1e99
    maxx = -1e99
    miny = 1e99
    maxy = -1e99
    for dataset in paths:
        for path in paths[dataset]:
            minx = np.min([np.min(path[:,0]),minx])
            maxx = np.max([np.max(path[:,0]),maxx])
            miny = np.min([np.min(path[:,1]),miny])
            maxy = np.max([np.max(path[:,1]),maxy])            
    
    
    x = np.arange(minx,maxx+25,25)
    y = np.arange(miny,maxy+25,25)
    # irregular spaced z-axis
    z = [0,1,2,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,33,36,39,42,46,50,55,
         60,65,70,80,90,100,120,140,160,180,200,240,280,350,700]
    X,Y,Z = np.meshgrid(x,y,z)
    gridpoints = np.column_stack((X.flatten(),Y.flatten(),Z.flatten()))
    
    A = create_A_matrix(gridpoints, paths)

    
    
    # sources = [(0,0,50)]
    # receivers = [(12,46,0)] 
    # datasets = ["test dataset"]
    # paths = {}
    # ttimes_base = {}
    # for dataset in datasets:
    #     paths[dataset] = []
    #     ttimes_base[dataset] = np.zeros(len(sources))
    #     for i in range(len(sources)):
    #         path = trace_direct_ray(model,source,receiver,"P",
    #                                 model_bottom_depth=700)
    #         pathx,pathy = p(path[:,0],path[:,1])
    #         paths[dataset].append(np.column_stack((pathx/1000.,pathy/1000,path[:,2])))
    #         ttimes_base[dataset][i] = path[0][-1]
            
    # lons = path[:,0]
    # lats = path[:,1]
    
    # r = 6371.-path[:,2]
    # x = r*np.sin(np.radians(90.-lats))*np.cos(np.radians(lons))
    # y = r*np.sin(np.radians(90.-lats))*np.sin(np.radians(lons))
    # z = r*np.cos(np.radians(90.-lats))
    # pathdistance = np.sum(np.sqrt(np.diff(x)**2+np.diff(y)**2+np.diff(z)**2))
    
    
    # A = create_A_matrix(gridpoints, paths)
    
    slowness = 1./np.random.uniform(5.5,10.0,size=len(gridpoints))
    
    ttimes = forward_module(slowness, A, ttimes_base=ttimes_base)
    
    res_syn = {}
    for dataset in ttimes:
        print("this needs to be changed to adapt to the new demean_ttimes function using the M matrix")
        res_syn[dataset] = demean_ttimes(ttimes[dataset],event_idx[dataset])
    
    
    
#%%
def test2():
    x0, y0, x1, y1, z0, z1 = -10, -10, 10, 10, 0, 8
    xgrid = np.linspace(x0, x1, 11)
    ygrid = np.linspace(y0, y1, 11)
    zgrid = np.array([0,1,2,3,4,6,8,10,15,20])
    npath = 10
    x = np.linspace(-9, 9, npath)
    y = np.linspace(-9, 9, npath)
    z = np.linspace(8.1,0.1,npath)
    t = np.arange(len(x))
    
    xthresh = np.diff(xgrid)[0]/1000.
    ythresh = np.diff(ygrid)[0]/1000.
    zthresh = np.diff(zgrid)[0]/1000.
    # x-points directly on one of the xgridlines:
    idx_x = np.where(np.abs(xgrid[:, None] - x[None, :]) < xthresh)[1]
    x[idx_x] += xthresh
    # y-points directly on one of the ygridlines:
    idx_y = np.where(np.abs(ygrid[:, None] - y[None, :]) < ythresh)[1]
    y[idx_y] += ythresh 
    # z-points directly on one of the ygridlines:
    idx_z = np.where(np.abs(zgrid[:, None] - y[None, :]) < zthresh)[1]
    z[idx_z] += zthresh 
    
    idx_grid, idx_t = np.where((xgrid[:, None] - x[None, :-1]) * (xgrid[:, None] - x[None, 1:]) <= 0)
    tx = idx_t + (xgrid[idx_grid] - x[idx_t]) / (x[idx_t+1] - x[idx_t])
    
    idx_grid, idx_t = np.where((ygrid[:, None] - y[None, :-1]) * (ygrid[:, None] - y[None, 1:]) <= 0)
    ty = idx_t + (ygrid[idx_grid] - y[idx_t]) / (y[idx_t+1] - y[idx_t])
    
    idx_grid, idx_t = np.where((zgrid[:, None] - z[None, :-1]) * (zgrid[:, None] - z[None, 1:]) <= 0)
    tz = idx_t + (zgrid[idx_grid] - z[idx_t]) / (z[idx_t+1] - z[idx_t])
    
    t2 = np.hstack((0,np.sort(np.r_[tx, ty, tz]),len(x)))
    valid = np.append(np.diff(t2)>0,True)
    t2 = t2[valid]
    # # if the path crosses exactly a grid crossing, it creates a double
    # # point which can cause problems
    # t2unique,idxunique,cntunique = np.unique(np.around(t2,5),return_index=True,return_counts=True)
    # idx_double = idxunique[np.where(cntunique==4)[0]]
    # t2 = np.delete(t2,np.hstack((idx_double,idx_double+1)))
    
    x2 = np.interp(t2, t, x)
    y2 = np.interp(t2, t, y)
    z2 = np.interp(t2, t, z)
    
    #loc = np.where(np.diff(t2) == 0)[0] + 1
    
    dists = np.sqrt( np.diff(x2)**2 + np.diff(y2)**2 + np.diff(z2)**2)
    #distlist = np.split(dists, loc)
    #d = np.array(list(map(np.sum,distlist)))
    d = dists
    
    # xlist = np.split(x2, loc)
    # ylist = np.split(y2, loc)
    # zlist = np.split(z2, loc)
    
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    for i in range(len(x2)-1):
        xp,yp,zp = ([x2[i],x2[i+1]],[y2[i],y2[i+1]],[z2[i],z2[i+1]])
        ax.plot(xp,yp,zp)
        ax.text(np.mean(xp), np.mean(yp), np.mean(zp), str(i))
    xx,yy = np.meshgrid([x0,x1],[y0,y1])
    for zi in zgrid:
        ax.plot_surface(xx, yy, np.ones_like(xx)*zi, alpha=0.2)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(1,3,1)
    for i in range(len(x2)-1):
        xp,yp,zp = ([x2[i],x2[i+1]],[y2[i],y2[i+1]],[z2[i],z2[i+1]])
        plt.plot(xp, yp)
        plt.text(np.mean(xp), np.mean(yp), np.around(d[i],2))
    lines = []
    for xi in xgrid:
        lines.append(((xi, y0), (xi, y1)))
    for yi in ygrid:
        lines.append(((x0, yi), (x1, yi)))
    lc = LineCollection(lines, color="gray", lw=1, alpha=0.5)
    ax.add_collection(lc)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    ax = fig.add_subplot(1,3,2)
    for i in range(len(x2)-1):
        xp,yp,zp = ([x2[i],x2[i+1]],[y2[i],y2[i+1]],[z2[i],z2[i+1]])
        plt.plot(xp, zp)
        plt.text(np.mean(xp), np.mean(zp), np.around(d[i],2))
    lines = []
    for xi in xgrid:
        lines.append(((xi, z0), (xi, z1)))
    for zi in zgrid:
        lines.append(((x0, zi), (x1, zi)))
    lc = LineCollection(lines, color="gray", lw=1, alpha=0.5)
    ax.add_collection(lc)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect('equal')
    ax = fig.add_subplot(1,3,3)
    for i in range(len(x2)-1):
        xp,yp,zp = ([x2[i],x2[i+1]],[y2[i],y2[i+1]],[z2[i],z2[i+1]])
        plt.plot(yp, zp)
        plt.text(np.mean(yp), np.mean(zp), np.around(d[i],2))
    lines = []
    for yi in ygrid:
        lines.append(((yi, z0), (yi, z1)))
    for zi in zgrid:
        lines.append(((y0, zi), (y1, zi)))
    lc = LineCollection(lines, color="gray", lw=1, alpha=0.5)
    ax.add_collection(lc)
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_aspect('equal')
    #%%
    x0, y0, x1, y1 = -10, -10, 10, 10
    n = 11
    xgrid = np.linspace(x0, x1, n)
    ygrid = np.linspace(y0, y1, n)
    x = np.linspace(-9, 9, 4)
    y = x/2.2
    t = np.arange(len(x))
    
    idx_grid, idx_t = np.where((xgrid[:, None] - x[None, :-1]) * (xgrid[:, None] - x[None, 1:]) <= 0)
    tx = idx_t + (xgrid[idx_grid] - x[idx_t]) / (x[idx_t+1] - x[idx_t])
    
    idx_grid, idx_t = np.where((ygrid[:, None] - y[None, :-1]) * (ygrid[:, None] - y[None, 1:]) <= 0)
    ty = idx_t + (ygrid[idx_grid] - y[idx_t]) / (y[idx_t+1] - y[idx_t])
    
    t2 = np.sort(np.r_[t, tx, tx, ty, ty])
    
    x2 = np.interp(t2, t, x)
    y2 = np.interp(t2, t, y)
    dists = np.sqrt( np.diff(x2)**2 + np.diff(y2)**2)
    distlist = np.split(dists, loc)
    d = np.array(list(map(np.sum,distlist)))
                        
    loc = np.where(np.diff(t2) == 0)[0] + 1
    
    xlist = np.split(x2, loc)
    ylist = np.split(y2, loc)
    
    
    fig, ax = plt.subplots()
    for i, (xp, yp) in enumerate(zip(xlist, ylist)):
        plt.plot(xp, yp)
        plt.text(np.mean(xp), np.mean(yp), np.around(d[i],2))
    
    
    lines = []
    for x in np.linspace(x0, x1, n):
        lines.append(((x, y0), (x, y1)))
    
    for y in np.linspace(y0, y1, n):
        lines.append(((x0, y), (x1, y)))
    
    lc = LineCollection(lines, color="gray", lw=1, alpha=0.5)
    ax.add_collection(lc)
    ax.set_aspect('equal')
    
    
    
