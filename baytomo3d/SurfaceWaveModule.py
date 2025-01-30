#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:08:47 2022

@author: emanuel
"""

import numpy as np
import pyproj, os, gzip, pickle, sys, time
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull, Delaunay
from scipy.sparse import csc_matrix
from scipy.stats import truncnorm
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.special import gammaln
from . import FMM
import cartopy.crs as ccrs
import cartopy.feature as cf
# either use pysurf
if True:
    from pysurf96aa import surf96
# or dccurve
else:
    from . import dccurve
    dccurve.init_dccurve(0) # verbose = 1

# the statement below sometimes causes problems on the cluster, better
# switch manually between modules?
# if not 'baytomo3d.dccurve' in sys.modules:
#     try:
#         from . import dccurve
#         # initialize. This mustn't be made more than once!
#         dccurve.init_dccurve(0) # verbose = 1
#     except:
#         print("Did not find dccurve module. Using pysurf96.")
#         from pysurf96aa import surf96
        
"""
CLASS
"""

# This class is used to handle the surface-wave data and the matrix for the
# forward calculations. The values stored in this class are identical for all
# chains and do not change during iterations. Only when paths are updates, it
# will be updated accordingly
class SurfacewaveData(object):
    
    def __init__(self,sources=None,receivers=None,traveltimes=None,
                 datastd=None,datacount=None,periods=None,
                 stations=None,projection=None):
        
        self.type = 'surfacewaves'
        
        if sources is None:
            self.isactive = False
        else:
            self.isactive = True
            args = locals()
            for arg in args:
                if args[arg] is None:
                    raise Exception("Not all values in SurfacewaveData are defined (%s)." %arg)
        self.sources = sources
        self.receivers = receivers
        self.data = traveltimes
        # is the initial/fixed data std. If datastd is a hyperparameter, it is
        # handled by the individual chains
        self.datastd = datastd
        self.projection = projection
        self.stations = stations
        self.periods = periods
        # initialize with an empty matrix
        self.A = None
        
        if self.isactive:
            self.datasets = []
            for dataset in self.data:
                self.datasets.append(dataset)
            # make sure periods are strictly decreasing, important for dccurve forward calculations
            self.periods = self.periods[self.periods.argsort()[::-1]]
            self.minx = 1e99
            self.maxx = -1e99
            self.miny = 1e99
            self.maxy = -1e99
            for stat in stations:
                self.minx = np.min([np.min(stat[0]),self.minx])
                self.maxx = np.max([np.max(stat[0]),self.maxx])
                self.miny = np.min([np.min(stat[1]),self.miny])
                self.maxy = np.max([np.max(stat[1]),self.maxy])
                
                
    #         self.set_collection_datastd()
            
            
    # def set_collection_datastd(self):
    #     if self.isactive:
    #         self.collection_datastd = {}
    #         for dataset in self.data:
    #             self.collection_datastd[dataset] = {}
    #             for period in self.periods:
    #                 if period in self.data[dataset].keys():
    #                     self.collection_datastd[dataset][period] = []
                        
    # def append_collection_datastd(self):
    #     if self.isactive:
    #         for dataset in self.datastd:
    #             for period in self.datastd[dataset]:
    #                 self.collection_datastd[dataset][period].append(
    #                     np.mean(self.datastd[dataset][period]))

            


"""
FUNCTIONS
"""

def get_phasevel_dataranges(surfacewaves):
    """
    Automatically checks the range of phase velocities in the datasets.
    This information can be used to set a prior on the allowed phase velocites.

    Returns
    -------
    phasevel ranges per dataset (love/rayleigh) and period.

    """
    
    print("Warning! setting a prior on the phase velocities!")
    
    phasevel_prior = {}
    phasevel_prior["rayleigh"] = {}
    phasevel_prior["love"] = {}
    
    for dataset in surfacewaves.data:
        
        for period in surfacewaves.data[dataset]:
            
            try:
                path_dists = surfacewaves.path_dists[dataset][period]
            except:
                path_dists = np.sqrt(np.sum(
                    (surfacewaves.sources[dataset][period]-surfacewaves.receivers[dataset][period])**2,axis=1))
            vels = path_dists/surfacewaves.data[dataset][period]
        
            if "rayleigh" in dataset:
                
                try:
                    phasevel_prior["rayleigh"][period].append(vels)
                except:
                    phasevel_prior["rayleigh"][period] = list(vels)
                    
            elif "love" in dataset:
                
                try:
                    phasevel_prior["love"][period].append(vels)
                except:
                    phasevel_prior["love"][period] = list(vels)
                    
    for wavetype in phasevel_prior:
        for period in phasevel_prior[wavetype]:
            phasevel_prior[wavetype][period] = np.hstack(phasevel_prior[wavetype][period])
                    
    phvel_ray_prior = []
    phvel_lov_prior = []
    
    for period in surfacewaves.periods:
        
        for wavetype in ["rayleigh","love"]:
        
            try:
                vels = phasevel_prior[wavetype][period]
                vmin = np.percentile(vels,0.5) 
                vmax = np.percentile(vels,99.5)
                dv = np.max([(vmax-vmin),1.0])
                if period<5:
                    vmin = vmin-(0.7*dv)
                else:
                    vmin = vmin-(0.5*dv)
                vmax = vmax+(0.5*dv)
            except:
                vmin = -99
                vmax = 99
            
            if wavetype == "rayleigh":
                phvel_ray_prior.append([vmin,vmax])
            else:
                phvel_lov_prior.append([vmin,vmax])
                
    phvel_ray_prior = np.vstack(phvel_ray_prior)
    phvel_lov_prior = np.vstack(phvel_lov_prior)
    phvel_ray_prior[surfacewaves.periods.argmax(),1] *= 1.1
    phvel_lov_prior[surfacewaves.periods.argmax(),1] *= 1.1
    
    return phvel_ray_prior,phvel_lov_prior


# Calculate a single straight path (on a regular, equidistant grid)
def calc_straight_ray_path(source,receiver,stepsize):
    
    if source[0]==receiver[0] and source[1]==receiver[1]:
        raise Exception("Source and receiver position are identical.")
        
    distance = np.sqrt((source[0]-receiver[0])**2 + (source[1]-receiver[1])**2)
    if distance <= stepsize*2.:
        stepsize = distance/2.

    x_regular = np.interp(np.linspace(0,1,int(distance/stepsize)),[0,1],[source[0],receiver[0]])
    y_regular = np.interp(np.linspace(0,1,int(distance/stepsize)),[0,1],[source[1],receiver[1]])        
    
    return np.column_stack((x_regular,y_regular))

    
    
# Calculate a single Eikonal path between a source and a receiver
def calc_eikonal_path(x,y,model,source,receiver,fmm_out=None):
    """
    x: 1-d array x-axis
    y: 1-d array y-axis
    model: 2-d array velocity model
    source: tuple of source coordinates (x,y)
    receiver: tuple of receiver coordinates (x,y)
    fmm_out: output from a previous traveltime field calculation (travel-
             time field is identical if only the receiver position changes)
    """
    
    if fmm_out is not None:
        xnew,ynew,ttimefield = fmm_out
    else:
        xnew,ynew,ttimefield = FMM.calculate_ttime_field(x,y,model,source)
        if type(ttimefield) == type(None):
            raise Exception("Could not calculate travel time field!")
        
    path = FMM.shoot_ray(xnew,ynew,ttimefield,source,receiver)[0] # FMM.shoot_ray returns a list      
    
    return path,(xnew,ynew,ttimefield)    


# Calculate paths for all source-receiver combinations
def get_paths(sources,receivers,rays='straight',x=None,y=None,model=None,
              return_ttimes=False,stepsize=None,verbose=True):
    """
    it is assumed that sources has the same length as receivers. Each
    source-receiver couple represents one travel-time measurement
    """
    
    paths = []
    ttimes = []
    
    if rays=='straight':
        if return_ttimes:
            raise Exception("return_traveltimes works only in combination with the FMM option.")
        if verbose:
            print("Calculating straight rays")
        for i in range(len(sources)):
            paths.append(calc_straight_ray_path(sources[i],receivers[i],stepsize))
    
    elif rays=='fmm':
        t0 = time.time()
        if x is None or y is None or model is None:
            raise Exception("Please define x,y and the model for the FMM calculations.")
        if verbose:
            print("Calculating Eikonal rays with the Fast Marching Method.")
        for i in range(len(sources)):           
            if i==0 or (sources[i]!=sources[i-1]).any():
                fmm_out = None # if the source location changes, we need to recalculate the ttime field
                ttime_interpolation_func = None
            path,fmm_out = calc_eikonal_path(x,y,model,sources[i],receivers[i],fmm_out=fmm_out)
            if return_ttimes:
                if ttime_interpolation_func is None:
                    ttimefunc = RectBivariateSpline(fmm_out[0],fmm_out[1],fmm_out[2].T)
                ttimes.append(ttimefunc(receivers[i][0],receivers[i][1]))
                
            paths.append(path)
            if verbose:
                if (i+1)%1000 == 0:
                    calc_time = time.time()-t0
                    print("  Eikonal path: %d/%d (elapsed time: %ds)" %(i+1,len(sources),calc_time))
                    
        if verbose:
            calc_time = time.time()-t0
            print("  Eikonal path: %d/%d (elapsed time: %ds)" %(i+1,len(sources),calc_time))
    else:
        raise Exception("Undefined ray type for path calculation! Choose between 'straight' and 'fmm'. Current value: %s" %(rays))
    
    if return_ttimes:
        return paths,ttimes
    else:
        return paths

#%%
def create_A_matrix(x,y,paths,wavelength_smoothing=False,verbose=False,
                    plotting=False,plotpath=None):
    
    if verbose:
        print("Filling A matrix...")
        
    #wavelength_smoothing=True
    if wavelength_smoothing:
        print("    Warning: Applying wavelength smoothing to the A matrices.")
        # get typical wavelengths at different periods
        periods =   np.array([1,   3,   5,   10,   15,  20,  30,  40., 75., 125, 200, 300.])
        phvel_ray = np.array([2.,  2.7, 2.9, 3.05, 3.2, 3.5, 3.7, 3.8, 4.0, 4.2, 4.6, 4.9 ])
        phvel_lov = np.array([2.3, 3.0, 3.3, 3.4,  3.6, 3.8, 4.0, 4.2, 4.5, 4.7, 4.8, 5.0 ])
        wlen_ray = interp1d(periods,periods*phvel_ray)
        wlen_lov = interp1d(periods,periods*phvel_lov)
        X,Y = np.meshgrid(x,y)
        gridpoints = np.column_stack((X.flatten(),Y.flatten()))
            
    
    A = {}
    PHI = {}
    
    # cells = []
    # corners = []
    # for gp in self.gridpoints:
    #     cell = [(gp[0]-self.xgridspacing/2.,gp[1]+self.ygridspacing/2.),
    #             (gp[0]+self.xgridspacing/2.,gp[1]+self.ygridspacing/2.),
    #             (gp[0]+self.xgridspacing/2.,gp[1]-self.ygridspacing/2.),
    #             (gp[0]-self.xgridspacing/2.,gp[1]-self.ygridspacing/2.)]
    #     corners.append(cell)
    #     cells.append(Polygon(cell))
        
    # gridtree = cKDTree(self.gridpoints)
       
    # for dataset in paths:
        
    #     # initializing sparse matrix
    #     self.A[dataset] = lil_matrix((len(self.data[dataset]),len(self.gridpoints)))
    #     PHI = lil_matrix(self.A[dataset].shape)
 
    #     if len(paths[dataset][period]) != len(self.data[dataset][period]):
    #         raise Exception("Number of paths and number of data points is not identical! Cannot create A matrix.")

    #     for i in range(len(self.data[dataset])):
            
    #         if i%1000==0:
    #             print(i,"/",len(self.data[dataset]))
                
    #         x = paths[dataset][period][i][:,0]
    #         y = paths[dataset][i][:,1]
                                
    #         pathdist = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    #         steps = np.linspace(0,1,int(pathdist/np.min([self.xgridspacing/4.,self.ygridspacing/4.])))
    #         x_reg = np.interp(steps,np.linspace(0,1,len(x)),x)
    #         y_reg = np.interp(steps,np.linspace(0,1,len(y)),y)
            
    #         k=16
    #         distance_upper_bound = np.max([wavelength/7.,np.sqrt(self.xgridspacing**2+self.ygridspacing**2)])
    #         nndist,nnidx = gridtree.query(np.column_stack((x_reg,y_reg)),k=k,distance_upper_bound=distance_upper_bound)
    #         while not np.isinf(nndist[:,-1]).all():
    #             k*=2
    #             nndist,nnidx = gridtree.query(np.column_stack((x_reg,y_reg)),k=k,distance_upper_bound=distance_upper_bound)
    #             if k>=len(self.gridpoints):
    #                 break
    #         idx = np.unique(nnidx)[:-1] # last index is for the inf points
                   
    #         # normal vectors
    #         norm = np.arctan2(np.gradient(y_reg),np.gradient(x_reg)) + np.pi/2.
            
    #         parallel1 = np.column_stack((x_reg+wavelength/8*np.cos(norm),
    #                                      y_reg+wavelength/8*np.sin(norm)))
    #         parallel2 = np.column_stack((x_reg-wavelength/8*np.cos(norm),
    #                                      y_reg-wavelength/8*np.sin(norm)))
            
    #         # shapely polygon
    #         polyshape = np.vstack((parallel1,parallel2[::-1]))
    #         poly = Polygon(polyshape)
            
    #         weights = np.zeros(len(self.gridpoints))
    #         idx_valid = []
    #         for grididx in idx:
    #             area = poly.intersection(cells[grididx]).area
    #             if area>0.:
    #                 weights[grididx] = area
    #                 idx_valid.append(grididx)
                
    #         idx = idx_valid
    #         weights /= np.sum(weights)
    #         weights *= pathdist
                    
                    
    #         self.A[dataset][i,idx] = weights[idx]

    #         # plt.figure()
    #         # patch = patches.Polygon(polyshape,alpha=0.6)
    #         # for xg in self.xgrid:
    #         #     plt.plot([xg,xg],[np.min(self.ygrid),np.max(self.ygrid)],'k')
    #         # for yg in self.ygrid:
    #         #     plt.plot([np.min(self.xgrid),np.max(self.xgrid)],[yg,yg],'k')
    #         # plt.plot(self.gridpoints[:,0],self.gridpoints[:,1],'k.',ms=2)
    #         # plt.plot(paths[dataset][i][:,0],paths[dataset][i][:,1])
    #         # plt.plot(parallel1[:,0],parallel1[:,1],'--')
    #         # plt.plot(parallel2[:,0],parallel2[:,1],'--')
    #         # plt.plot(self.gridpoints[idx][:,0],self.gridpoints[idx][:,1],'g.')
    #         # for i in idx:
    #         #     plt.text(self.gridpoints[i,0],self.gridpoints[i,1],"%.3f" %weights[i])
    #         # plt.gca().add_patch(patch)
    #         # plt.gca().set_aspect('equal')
    #         # plt.show()    
    #         # pause
            
            
    if not (np.allclose(np.std(np.diff(x)),0.) and np.allclose(np.std(np.diff(x)),0.)):
        raise Exception("x and y need to be on a regular axis.")
        
    xgridspacing = np.diff(x)[0]
    ygridspacing = np.diff(y)[0]
    
    # gridlines (lines between gridpoints, delimiting cells) 
    xgrid = np.append(x-xgridspacing/2.,x[-1]+xgridspacing/2.)
    ygrid = np.append(y-ygridspacing/2.,y[-1]+ygridspacing/2.)
        
    for dataset in paths:
        
        A[dataset] = {}
        PHI[dataset] = {}
        
        for period in paths[dataset]:
            
            print(f"    Filling matrix. dataset: {dataset} period: {period}")
    
            # these arrays will be filled with the non-zero indices and values
            # of the A and the PHI matrices
            col = []
            row = []
            data_a = []
            data_phi = []
            shape = (len(paths[dataset][period]),len(x)*len(y))
            
            # initializing sparse matrix
            #A[dataset][period] = lil_matrix((len(paths[dataset][period]),len(x)*len(y)),dtype='float32')
            #PHI[dataset][period] = lil_matrix(A[dataset][period].shape,dtype='float32')
     
            # it can cause problems if the start or end points of the path are 
            # exactly on the grid lines. Points are then moved by this threshold
            xthresh = xgridspacing/1000.
            ythresh = ygridspacing/1000.
    
            for i in range(len(paths[dataset][period])):
                            
                xpath = paths[dataset][period][i][:,0]
                ypath = paths[dataset][period][i][:,1]
                
                # x-points directly on one of the xgridlines:
                idx_x = np.where(np.abs(xgrid[:, None] - xpath[None, :]) < xthresh)[1]
                xpath[idx_x] += xthresh
                # y-points directly on one of the ygridlines:
                idx_y = np.where(np.abs(ygrid[:, None] - ypath[None, :]) < ythresh)[1]
                ypath[idx_y] += ythresh            
                
                t = np.arange(len(xpath))
                
                idx_grid_x, idx_t = np.where((xgrid[:, None] - xpath[None, :-1]) * (xgrid[:, None] - xpath[None, 1:]) <= 0)
                tx = idx_t + (xgrid[idx_grid_x] - xpath[idx_t]) / (xpath[idx_t+1] - xpath[idx_t])
                    
                idx_grid_y, idx_t = np.where((ygrid[:, None] - ypath[None, :-1]) * (ygrid[:, None] - ypath[None, 1:]) <= 0)
                ty = idx_t + (ygrid[idx_grid_y] - ypath[idx_t]) / (ypath[idx_t+1] - ypath[idx_t])
                                
                t2 = np.sort(np.r_[t, tx, tx, ty, ty])
                
                # if the path crosses exactly a grid crossing, it creates a double
                # point which can cause problems
                t2unique,idxunique,cntunique = np.unique(np.around(t2,5),
                                                         return_index=True,
                                                         return_counts=True)
                idx_double = idxunique[np.where(cntunique==4)[0]]
                t2 = np.delete(t2,np.hstack((idx_double,idx_double+1)))
                
                x2 = np.interp(t2, t, xpath)
                y2 = np.interp(t2, t, ypath)
                
                dists = np.sqrt( np.diff(x2)**2 + np.diff(y2)**2 )
      
                # loc gives the x2-/y2-indices where the path is crossing a gridline
                loc = np.where(np.diff(t2) == 0)[0] + 1     
                
                pnts = np.column_stack((x2[np.hstack((0,loc,-1))],
                                        y2[np.hstack((0,loc,-1))]))
                
                # the angles from np.arctan lie always in the range [-pi/2,+pi/2]
                phi = np.arctan(np.diff(pnts[:,1])/(np.diff(pnts[:,0])+1e-16))
                # the azimuthal coverage check will not work with arctan2!
                # arctan2 determines the correct quadrant, angles are in [-pi,pi]
                #phi = np.arctan2(np.diff(pnts[:,1]),np.diff(pnts[:,0]))
                
                midpoints = pnts[:-1]+np.diff(pnts,axis=0)/2.
           
                xind = np.abs(midpoints[:,0][None,:]-x[:,None]).argmin(axis=0)
                yind = np.abs(midpoints[:,1][None,:]-y[:,None]).argmin(axis=0)
                
                # idx gives the nearest neighbor index of the self.gridpoints array
                idx = yind*len(x)+xind
                    
                distlist = np.split(dists, loc)
                d = np.array(list(map(np.sum,distlist)))
                if np.sum(d) == 0:
                    print(i,[xpath,ypath])
                    raise Exception("a zero distance path is not valid!")
                weights = d.copy()
           
                # plt.figure()
                # for xg in self.xgrid:
                #     plt.plot([xg,xg],[np.min(self.ygrid),np.max(self.ygrid)],'k')
                # for yg in self.ygrid:
                #     plt.plot([np.min(self.xgrid),np.max(self.xgrid)],[yg,yg],'k')
                # plt.plot(self.X.flatten(),self.Y.flatten(),'k.',ms=2)
                # plt.plot(x2[loc],y2[loc],'o')
                # plt.plot(midpoints[:,0],midpoints[:,1],'.')
                # plt.plot(self.gridpoints[idx][:,0],self.gridpoints[idx][:,1],'g.')
                # plt.plot([x2[0],x2[-1]],[y2[0],y2[-1]],'r.')
                # for mp in range(len(midpoints)):
                #     plt.text(midpoints[mp,0],midpoints[mp,1],"%.3f" %d[mp])
                # plt.gca().set_aspect('equal')
                # plt.show()    
                # pause
           
                # sometimes a path can traverse a grid cell twice
                idxunique, sortidx, cntunique= np.unique(idx,return_counts=True,return_index=True)
                if (cntunique>1).any():
                    for doubleidx in np.where(cntunique>1)[0]:
                        weights[idx==idxunique[doubleidx]] = np.sum(weights[idx==idxunique[doubleidx]])
                        # taking the mean angle
                        phi[idx==idxunique[doubleidx]] = np.arctan2(np.sum(np.sin(phi[idx==idxunique[doubleidx]])),
                                                                    np.sum(np.cos(phi[idx==idxunique[doubleidx]])))
       
                if not np.allclose(np.sum(weights[sortidx]),np.sum(d)):
                    print(np.sum(weights[sortidx]),np.sum(d))
                    raise Exception("matrix filling error")
                if (weights==0.).all():
                    raise Exception("weights have to be greater than zero!")
        
                # we have to make sure that none of the phi values is exactly
                # zero by coincidence. This would result in different matrix
                # shapes between the A matrix and the PHI matrix
                phi[phi==0.] += 0.001
                if (weights==0.).any():
                    print("Warning! One of the weights in the A matrix is zero!")
                col.append(idxunique)
                row.append(np.repeat(i,len(sortidx)))
                data_a.append(weights[sortidx])
                data_phi.append(phi[sortidx])
               
            # done calculating the matrix elements. Putting matrix together
            col = np.hstack(col)
            row = np.hstack(row)
            data_a = np.hstack(data_a)
            data_phi = np.hstack(data_phi)
            # csr would be even a little bit faster with matrix vector product
            # calculations, but csc is much faster when extracting columns
            # which is necessary when updating the model predictions
            A[dataset][period] = csc_matrix((data_a,(row,col)),shape=shape,dtype='float32')
            PHI[dataset][period] = csc_matrix((data_phi,(row,col)),shape=shape,dtype='float32')
            # float32 used to save some memory and disk space

            # create a smoothing matrix with truncated Gaussian kernels
            if wavelength_smoothing:
                if "ray" in dataset:
                    truncated_normal = truncnorm(-2,2,0,wlen_ray(period)/4.)
                elif "lov" in dataset:
                    truncated_normal = truncnorm(-2,2,0,wlen_lov(period)/4.)
                else:
                    raise Exception(dataset,"not recognized whether this is love or rayleigh.")
                maxweight = truncated_normal.pdf(0)
                row = []
                col = []
                data_s = []
                # smoothing matrix is symmetric. We put it together column wise
                # to avoid having to store a huge len(gridpoints)*len(gridpoints)
                # matrix in memory. 
                for colidx,(xj,yj) in enumerate(gridpoints):
                    #distm = np.sqrt(np.sum((gridpoints-np.array([xj,yj]))**2,axis=1))
                    weights = truncated_normal.pdf(np.sqrt(np.sum((gridpoints-np.array([xj,yj]))**2,axis=1)))
                    # ignore weights that are very small and have almost no contribution
                    rowind = np.where(weights>0.05*maxweight)[0]
                    weights = weights[rowind]
                    colind = np.ones_like(rowind,dtype=int)*colidx
                    col.append(colind)
                    row.append(rowind)
                    data_s.append(weights)
                    # because of symmetry, add the same from the right
                    colidx_right = len(gridpoints)-colidx-1
                    if colidx_right <= colidx:
                        # no need to go through all columns because of symmetry
                        break
                    else:
                        col.append(len(gridpoints)-colind-1)
                        row.append(len(gridpoints)-rowind-1)
                        data_s.append(weights)
                col = np.hstack(col)
                row = np.hstack(row)
                data_s = np.hstack(data_s)
                smoothing_matrix = csc_matrix((data_s,(row,col)),shape=(len(gridpoints),len(gridpoints)),dtype='float32')
                
                # transforming the normal A matrix to the finite-ray-width matrix
                dists = A[dataset][period].sum(axis=1)
                A[dataset][period] = A[dataset][period]*smoothing_matrix
                # smoothing changes the distance, because the weights are not
                # correctly normalized to 1 (which is difficult if the smoothing
                # kernel goes beyond the model borders)
                correction_factor = dists/A[dataset][period].sum(axis=1)
                A[dataset][period] = A[dataset][period].multiply(correction_factor).tocsc()
                
            if plotting:
                iplot = np.random.randint(0,len(paths[dataset][period]))
                weights = np.array(A[dataset][period][iplot].todense())[0]
                path = paths[dataset][period][iplot]
                plt.ioff()
                fig = plt.figure(figsize=(12,10))
                plt.pcolormesh(x,y,np.reshape(weights,(len(y),len(x))),
                               cmap=plt.cm.YlOrBr,shading='nearest')
                for xg in xgrid:
                    plt.plot([xg,xg],[np.min(ygrid),np.max(ygrid)],'k',linewidth=0.1)
                for yg in ygrid:
                    plt.plot([np.min(xgrid),np.max(xgrid)],[yg,yg],'k',linewidth=0.1)
                plt.plot(path[:,0],path[:,1],'k',linewidth=1)
                plt.gca().set_aspect('equal')
                #plt.show()
                #pause
                plt.savefig(os.path.join(plotpath,"A_matrix_surfwaves_%s_%s_%d.jpg" %(dataset,period,iplot)),
                            bbox_inches='tight',dpi=200)
                plt.close(fig)
                    
            if verbose:
                print("Matrix filling done")
    
    return A,PHI


def create_datahull(gridpoints2d,A):
    """

    Parameters
    ----------
    gridpoints2d : numpy 2D array
        Array containing the 2D gridpoints obtained from:
        X,Y = np.meshgrid(x,y)
        gridpoints2d = np.column_stack((X.flatten(),Y.flatten()))
    A : dictionary
        Dictionary containing the A matrix for the surface-wave data sets.

    Returns
    -------
    Delaunay triangulation object.

    """
    # define a hull around the region covered by surface wave data. 
    # areas outside this hull are not relevant for the phase-velocity
    # calculations and can be skipped
    x = np.unique(gridpoints2d[:,0])
    y = np.unique(gridpoints2d[:,1])
    xgridspacing = np.diff(x)[0]
    ygridspacing = np.diff(y)[0]    
    data_coverage_sw = np.zeros(len(gridpoints2d))
    for dataset in A:
        for period in A[dataset]:
            data_coverage_sw += np.array(A[dataset][period].sum(axis=0))[0]
    hull = ConvexHull(gridpoints2d[data_coverage_sw>0])
    # add a margin around the hull (2 gridpoints in size)
    edgepoints = []
    for i in range(len(hull.vertices)):
        vertex_xy = hull.points[hull.vertices[i]]
        neighbor1 = hull.points[np.roll(hull.vertices,-1)[i]]
        neighbor2 = hull.points[np.roll(hull.vertices, 1)[i]]
        vector_angle = np.arctan2(neighbor1[1]-neighbor2[1],neighbor1[0]-neighbor2[0]) - np.pi/2.
        #intercept = vertex_xy[1]-slope*vertex_xy[0]
        vector_length = np.sqrt(np.sum((2*xgridspacing)**2 + (2*ygridspacing)**2)) 
        edgepoints.append(vertex_xy+[np.cos(vector_angle)*vector_length,
                                     np.sin(vector_angle)*vector_length])
    return Delaunay(edgepoints)
    
            
           
def forward_module(z,periods,modelshape,vs_field,vpvs_field,A,
                   phslo_ray=None,phslo_lov=None,phttime_predictions=None,
                   phvel_prior_ray=None,phvel_prior_lov=None,
                   idx_modified_gridpoints=None,anisotropic=False,
                   selfcheck=False,inhull_idx=None):
    
    if idx_modified_gridpoints is not None:
        if len(idx_modified_gridpoints)==0: 
            if (phttime_predictions is not None and
                phslo_ray is not None and phslo_lov is not None):
                return phttime_predictions.copy(),phslo_ray.copy(),phslo_lov.copy(),0
            
    # this calculates the phase-vel predictions for Rayleigh and Love
    valid,idx_2d,prop_phslo_ray,prop_phslo_lov,nmods = get_phasevel_maps(
        z,periods,modelshape,vs_field=vs_field,vpvs_field=vpvs_field,
        phslo_ray=phslo_ray,phslo_lov=phslo_lov,
        phvel_prior_ray=phvel_prior_ray,phvel_prior_lov=phvel_prior_lov,
        idx_modified_gridpoints=idx_modified_gridpoints,
        selfcheck=selfcheck,inhull_idx=inhull_idx)
    if valid:
        prop_phttime_predictions = calculate_ttimes(
            prop_phslo_ray,prop_phslo_lov,A,
            phslo_ray=phslo_ray,phslo_lov=phslo_lov,
            phttime_predictions=phttime_predictions,
            idx_modified=idx_2d,selfcheck=selfcheck)
    else:
        prop_phttime_predictions = None
    
    return prop_phttime_predictions,prop_phslo_ray,prop_phslo_lov,nmods


def get_phasevel_maps(z,periods,modelshape,vs_field=None,vp_field=None,vpvs_field=None,
                      phslo_ray=None,phslo_lov=None,
                      phvel_prior_ray=None,phvel_prior_lov=None,
                      idx_modified_gridpoints=None,selfcheck=False,
                      inhull_idx=None):
    """
    The inhull_idx is only needed to make sure that the indices outside the
    hull are not causing the phase-velocity calculation to fail.
    Otherwise, the idx_modified_gridpoints is already limited to those gridpoints
    that are within the hull.

    """
    
    if ((vp_field is None and vpvs_field is None) or
        (vs_field is None and vpvs_field is None)):
        raise Exception("you have to define two of the three fields (vp_field,vs_field,vpvs_field)")
        
    # if the velocity profiles are bad, it may not be possible to obtain
    # phase velocity curves
    valid_result = True
        
    xn,yn,zn = modelshape    
        
    # Update the dispersion curves at all available grid locations
    if phslo_ray is None or phslo_lov is None:
        
        flat_idx = None
        
        prop_phslo_ray = {}
        prop_phslo_lov = {}
        for period in periods:
            prop_phslo_ray[period] = np.zeros(xn*yn)
            prop_phslo_lov[period] = np.zeros(xn*yn)
            
        if vpvs_field is None:
            vsprofiles = np.reshape(vs_field,(xn*yn,zn))
            vpprofiles = np.reshape(vp_field,(xn*yn,zn))
        elif vs_field is None:
            vpprofiles = np.reshape(vp_field,(xn*yn,zn))
            vpvsprofiles = np.reshape(vpvs_field, (xn*yn, zn))
            vsprofiles = vpprofiles/vpvsprofiles
        elif vp_field is None:
            vsprofiles = np.reshape(vs_field,(xn*yn,zn))
            vpvsprofiles = np.reshape(vpvs_field, (xn*yn, zn))
            vpprofiles = vsprofiles*vpvsprofiles            

        #t0 = time.time()
        raycurves = np.ones((len(vsprofiles),len(periods)))*np.inf
        lovcurves = np.ones((len(vsprofiles),len(periods)))*np.inf
        for i in range(len(vsprofiles)):
            slowness_ray, slowness_lov = profile_to_phasevelocity(
                    z,vsprofiles[i],vpprofiles[i],periods,
                    phvel_prior_ray,phvel_prior_lov)
            if (slowness_ray==0.).any():
                valid_result = False
            else:
                raycurves[i] = slowness_ray
            if (slowness_lov==0.).any():
                valid_result = False
            else:
                lovcurves[i] = slowness_lov
            # ignore profiles that are not inside the data-coverage hull
            if not valid_result and inhull_idx is not None:
                if i not in inhull_idx:
                    valid_result = True
                    lovcurves[i][:] = 3. # this only affects profiles outside
                    raycurves[i][:] = 3. # the hull, where no data points are
        
        #print(time.time()-t0)   
        for i,period in enumerate(periods):
            prop_phslo_ray[period] = raycurves[:,i]
            prop_phslo_lov[period] = lovcurves[:,i]
    
    # update the dispersion curves only at idx_modified_gridpoints
    else:
        if idx_modified_gridpoints is None:
            print("ERROR")
        # get the x,y indices of the modified gridpoints
        idx_2d = tuple(np.unique(np.unravel_index(
            idx_modified_gridpoints,modelshape)[:2],axis=1))
        # same indices in a flattened 2D array representation
        flat_idx = np.ravel_multi_index(idx_2d,modelshape[:2])
    
        # deepcopy seems to be necessary here, otherwise the phslo arrays
        # may change unintendendly (simple copy does not work with dictionaries)
        prop_phslo_ray = deepcopy(phslo_ray)
        prop_phslo_lov = deepcopy(phslo_lov)
        if selfcheck:
            for period in periods:
                if np.inf in phslo_ray[period]:
                    print("inf value at the beginning of getphasevel maps")
                    break
        
        if vpvs_field is None:
            vpmod = np.reshape(vp_field,modelshape)[idx_2d]
            vsmod = np.reshape(vs_field,modelshape)[idx_2d]
        elif vp_field is None:
            vsmod = np.reshape(vs_field,modelshape)[idx_2d]
            vpvsmod = np.reshape(vpvs_field,modelshape)[idx_2d]
            vpmod = vsmod*vpvsmod
        elif vs_field is None:
            vpmod = np.reshape(vp_field,modelshape)[idx_2d]
            vpvsmod = np.reshape(vpvs_field,modelshape)[idx_2d]
            vsmod = vpmod/vpvsmod            
            
        # if self.gridsizefactor > 1.:
        #     self.gridsizefactor += 2*self.gridsizefactor_inc*self.total_steps
        #     minx,miny,minz = np.min(self.gridpoints[idx_modified_gridpoints],axis=0)
        #     maxx,maxy,maxz = np.max(self.gridpoints[idx_modified_gridpoints],axis=0)
        #     Xlow,Ylow = np.meshgrid(np.arange(minx,maxx+self.xgridspacing,self.xgridspacing*self.gridsizefactor),
        #                             np.arange(miny,maxy+self.ygridspacing,self.ygridspacing*self.gridsizefactor))
        #     kdtree = cKDTree(self.gridpoints2d[flat_idx])
        #     nei_dist, nei_idx1 = kdtree.query(np.column_stack((Xlow.flatten(),Ylow.flatten())))            
        #     kdtree = cKDTree(np.column_stack((Xlow.flatten(),Ylow.flatten())))
        #     nei_dist, nei_idx2 = kdtree.query(self.gridpoints2d[flat_idx])
        #     vel_profiles = vel_profiles[nei_idx1[nei_idx2]]
        #     self.prop_model_vs.reshape(np.shape(self.X))[idx_2d] = vel_profiles
        
        # there is a chance to reduce the number of necessary calculations if
        # we only take the unique depth profiles into account
        _,idxunique,idxinverse = np.unique(
            np.hstack((vsmod,vpmod)),return_index=True,return_inverse=True,axis=0)
        vsprofiles = vsmod[idxunique]
        vpprofiles = vpmod[idxunique]
                  
        #t0 = time.time()
        raycurves = np.ones((len(vsprofiles),len(periods)))*np.inf
        lovcurves = np.ones((len(vsprofiles),len(periods)))*np.inf
        for i in range(len(vsprofiles)):  
            slowness_ray, slowness_lov = profile_to_phasevelocity(
                    z,vsprofiles[i],vpprofiles[i],periods,
                    phvel_prior_ray,phvel_prior_lov)
            if (slowness_ray==0.).any():
                raycurves[:] = np.inf
                lovcurves[:] = np.inf
                valid_result = False
                break
            else:
                raycurves[i] = slowness_ray
            if (slowness_lov==0.).any():
                raycurves[:] = np.inf
                lovcurves[:] = np.inf
                valid_result = False
                break
            else:
                lovcurves[i] = slowness_lov
        #print(time.time()-t0) 

        # plt.figure()
        # for i,curve in enumerate(raycurves):
        #     plt.plot(1./curve)
        # plt.show()

        for i,period in enumerate(periods):
            prop_phslo_ray[period][flat_idx] = raycurves[:,i][idxinverse]  
            prop_phslo_lov[period][flat_idx] = lovcurves[:,i][idxinverse]
                
    # SELFCHECK
    if selfcheck and valid_result:
        if phslo_ray is not None and phslo_lov is not None:
            for period in periods:
                if (np.inf in phslo_ray[period] or
                    np.inf in phslo_lov[period]):
                    raise Exception("there should be no inf in the phslo arrays")
                          
        if vpvs_field is None:
            vsprofiles = np.reshape(vs_field,(xn*yn,zn))
            vpprofiles = np.reshape(vp_field,(xn*yn,zn))
        elif vs_field is None:
            vpprofiles = np.reshape(vp_field,(xn*yn,zn))
            vpvsprofiles = np.reshape(vpvs_field, (xn*yn, zn))
            vsprofiles = vpprofiles/vpvsprofiles
        elif vp_field is None:
            vsprofiles = np.reshape(vs_field,(xn*yn,zn))
            vpvsprofiles = np.reshape(vpvs_field, (xn*yn, zn))
            vpprofiles = vsprofiles*vpvsprofiles 
        #t0 = time.time()
        raycurves = np.ones((len(vsprofiles),len(periods)))*np.inf
        lovcurves = np.ones((len(vsprofiles),len(periods)))*np.inf
        for i in range(len(vsprofiles)):
            slowness_ray, slowness_lov = profile_to_phasevelocity(
                z,vsprofiles[i],vpprofiles[i],periods,
                phvel_prior_ray,phvel_prior_lov)
            if (slowness_ray==0.).any():
                valid_result = False
            else:
                raycurves[i] = slowness_ray
            if (slowness_lov==0.).any():
                valid_result = False
            else:
                lovcurves[i] = slowness_lov
                
        for i,period in enumerate(periods):
            if not np.array_equal(prop_phslo_ray[period][inhull_idx],raycurves[:,i][inhull_idx]):
                raise Exception("selfcheck error: prop_phslo_ray is wrong")
            if not np.array_equal(prop_phslo_lov[period][inhull_idx],lovcurves[:,i][inhull_idx]):
                raise Exception("selfcheck error: prop_phslo_lov is wrong")
           
    nmods = len(vsprofiles)
    
    return valid_result,flat_idx,prop_phslo_ray,prop_phslo_lov,nmods
    

def vp2rho(vpmod):
    # Bertheussen 1977 "Moho depth determinations based on spectral-ratio analysis of NORSAR long-period P waves"
    return vpmod * 0.32 + 0.77


def profile_to_phasevelocity(z,vsprofile,vpprofile,periods,
                             phvel_prior_ray=None,phvel_prior_lov=None):
 
    if False: # simplify profiles has no significant speedup effect
        raise Exception("TODO: vpmod needs to be added")
        thickmod, vsmod = simplify_profile(z[:-1], vsprofile[:-1], return_layers=True)
        thickmod = np.append(thickmod,999.)
        vsmod = np.append(vsmod,vsprofile[-1])
    else:
        idx=np.where(np.around(np.diff(vsprofile*vpprofile),3)!=0)[0]
        layermodel = np.append(0,(z[:-1]+np.diff(z)/2.)[idx])
        vsmod = np.append(vsprofile[idx],vsprofile[-1])
        vpmod = np.append(vpprofile[idx],vpprofile[-1])
        thickmod = np.append(np.diff(layermodel),999.)
    
    densmod = vp2rho(vpmod)
    #np.savetxt("layermodel.txt",np.column_stack((thickmod,vpmod,vsmod,densmod)))
    if len(vsmod)>1 and (vpmod>0.).all() and (vsmod>0.).all():
        
        if 'baytomo3d.dccurve' in sys.modules:
            if (np.diff(periods)>0.).any():
                raise Exception("periods array has to be strictly decreasing! Otherwise dccurve will not work properly.")
            # size of these arrays have to be multiplied with number of modes
            # here, we just take the first mode into account
            slowness_ray = np.zeros(len(periods))
            slowness_lov = np.zeros(len(periods))
            # dccurve.get_disp_curve(nmodes,group,thickness,vp,vs,rho,periods,slowness, ray)
            # ray=1 for rayleigh, 0 for love
            dccurve.get_disp_curve(1,0,thickmod,vpmod,vsmod,densmod,
                                   periods,slowness_ray,1)
            dccurve.get_disp_curve(1,0,thickmod,vpmod,vsmod,densmod,
                                   periods,slowness_lov,0)
            
        else:
            if len(thickmod)>200 or len(periods)>60:
                raise Exception("pysurf currently cannot handle models with more than 200 layers or calculations at more than 60 periods.")
            phvel_lov = surf96(thickmod, vpmod, vsmod, densmod, periods,
                               wave='love',mode=1,velocity='phase',flat_earth=False)
            phvel_ray = surf96(thickmod, vpmod, vsmod, densmod, periods,
                               wave='rayleigh',mode=1,velocity='phase',flat_earth=False)
            slowness_ray = 1./phvel_ray
            slowness_lov = 1./phvel_lov
            
    else:
        slowness_ray[:] = slowness_lov[:] = 0.
        return slowness_ray, slowness_lov
    
    if (slowness_ray==0.).any() or (slowness_lov==0).any():
        return slowness_ray, slowness_lov
    fail = False
    if phvel_prior_ray is not None:
        if (1./slowness_ray < phvel_prior_ray[:,0]).any():
            fail=True
        elif (1./slowness_ray > phvel_prior_ray[:,1]).any():
            fail=True
    if phvel_prior_lov is not None:
        if (1./slowness_lov < phvel_prior_lov[:,0]).any():
            fail=True
        elif (1./slowness_lov > phvel_prior_lov[:,1]).any():
            fail=True
    if fail:
        # plt.ioff()
        # fig=plt.figure()
        # plt.plot(periods,1./slowness_ray)
        # plt.plot(periods,self.phvel_ray_prior[:,0],'k--')
        # plt.plot(periods,self.phvel_ray_prior[:,1],'k--')
        # plt.ylim(1.,5.)
        # plt.savefig("testplot_%d_%d.jpg" %(self.chain_no,self.total_steps),bbox_inches='tight')
        # plt.close(fig)                
        # print(self.total_steps,"model not in prior range",self.para.action)
        slowness_ray[:] = slowness_lov[:] = 0.

    return slowness_ray, slowness_lov
        

# simplify profile such that layers need to have a minimum thickness of
# at least minthick*depth (recommended minthick: 0.2)
# a lower number of layers speeds up the phase-velocity calculations
# this function will iteratively add new layerboundaries, starting with
# the maximum gradient until no more layers can be added 
# the phase-vel calculations are faster with fewer layers, but the speedup
# is minimal given the calculation time of this function
def simplify_profile( z, vel, minthick=0.2, mingrad=0.01, 
                     nlayers=None, return_layers=False):
            
    if nlayers is None:
        nlayers=len(z)
    
    dv = np.abs(np.diff(vel))
    grad = dv/np.diff(z)
    gradsort = grad.argsort()[::-1]

    layerbounds = [(z[gradsort[0]]+z[gradsort[0]+1])/2]

    idx = [gradsort[0]+1]
    
    for i in gradsort[1:]:
            
        # ignore velocity differences smaller than 0.01 km/s
        if dv[i]<mingrad:
            break
        
        newbound = (z[i]+z[i+1])/2
        
        if np.min(np.abs(newbound - layerbounds)) < minthick*newbound:
            continue
        
        layerbounds.append(newbound)
        
        idx.append(i+1)
        
        if len(idx) == nlayers-1:
            break
       
    if return_layers: #return thickness and layer velocities
        bounds = np.r_[np.min(z),np.sort(layerbounds),np.max(z)]   
        layervels = np.zeros(len(bounds)-1)
        for i,subidx in enumerate(np.split(np.arange(len(z)),np.sort(idx))):
            layervels[i] = np.mean(vel[subidx])
        return np.diff(bounds),layervels
    else: #return full profile (same z levels as input vs model)
        vel_simple = np.zeros(len(z))
        for i,subidx in enumerate(np.split(np.arange(len(z)),np.sort(idx))):
            vel_simple[subidx] = np.mean(vel[subidx]) 
        return vel_simple
    
        

def calculate_ttimes(prop_phslo_ray,prop_phslo_lov,A,
                     phslo_ray=None,phslo_lov=None,phttime_predictions=None,
                     idx_modified=None,anisotropic=False,anisochange=None,
                     selfcheck=False):
    # idx modified gives the indices of the modified gridpoints
    # idx_modified are the indices of the flat 2D arrays (at each period)
    # if nothing is given, it is assumed that all gridpoints have changed

    prop_phttime_predictions = {}
    
    if anisotropic:
        
        raise Exception("currently not implemented")
       
    # isotropic case         
    else:
        
        # matrix vector calculation assumes slownesses
        if phttime_predictions is None:
            # calculating model predictions from scratch. faster if more than
            # about 1/3 of the points have changed
            for dataset in A:
                prop_phttime_predictions[dataset] = {}
                for period in A[dataset]:
                    if "ray" in dataset:
                        prop_phttime_predictions[dataset][period] = A[dataset][period]*prop_phslo_ray[period]
                    elif "lov" in dataset:
                        prop_phttime_predictions[dataset][period] = A[dataset][period]*prop_phslo_lov[period]
                    else:
                        raise Exception(f"Cannot recognize whether dataset {dataset} is Rayleigh or Love")
        else:
            # updating previous model predictions. faster if only a few (less
            # than 1/3rd) of the points in the model vector have changed
            for dataset in A:
                prop_phttime_predictions[dataset] = {}
                for period in A[dataset]:
                    if "ray" in dataset:
                        prop_phttime_predictions[dataset][period] = (
                                phttime_predictions[dataset][period] + 
                                A[dataset][period][:,idx_modified] * 
                                (prop_phslo_ray[period][idx_modified] -
                                 phslo_ray[period][idx_modified]))
                        if selfcheck:
                            if not np.allclose(A[dataset][period]*prop_phslo_ray[period], 
                                               prop_phttime_predictions[dataset][period]):
                                raise Exception("selfcheck failed for matrix vector multiplication")
                    elif "lov" in dataset:
                        prop_phttime_predictions[dataset][period] = (
                                phttime_predictions[dataset][period] + 
                                A[dataset][period][:,idx_modified] * 
                                (prop_phslo_lov[period][idx_modified] -
                                 phslo_lov[period][idx_modified]))
                        if selfcheck:
                            if not np.allclose(A[dataset][period]*prop_phslo_lov[period], 
                                               prop_phttime_predictions[dataset][period]):
                                raise Exception("selfcheck failed for matrix vector multiplication")
                    else:
                        raise Exception(f"Cannot recognize whether dataset {dataset} is Rayleigh or Love")


    return prop_phttime_predictions


def get_loglikelihood_alt(data,predictions,data_std,path_dists,
                          norm='L2',distribution='normal',foutlier=None,
                          widthoutlier=None,degfree=None,minerror=None,
                          norm_coeffs=None):
    
        loglikelihood = 0.
        total_residual = 0.
        ncs = {} # a separate dictionary guarantees that the original one is not unintentionally modified
        for dataset in data:
            ncs[dataset] = {}
            for period in data[dataset]:
                loglikes,residuals,normcoeffs = calc_loglike(
                    dataset, period, data, predictions, path_dists, data_std, 
                    norm=norm, distribution=distribution, 
                    norm_coeffs=norm_coeffs, foutlier=foutlier, 
                    widthoutlier=widthoutlier, df=degfree)
                total_residual += np.sum(residuals)
                loglikelihood += np.sum(loglikes)
                ncs[dataset][period] = normcoeffs
                
        return total_residual,loglikelihood,ncs

        # empty dictionaries just to have the same return parameters as the new function
        return total_residual,loglikelihood,{},{},ncs
    

def get_loglikelihood(loglikelihood_dict0,residual_dict0,norm_coeffs0,
                      data,predictions,path_dists,data_std,
                      update=None,predictions0=None,foutlier=None,
                      widthoutlier=None,degfree=None,distribution='normal',
                      norm='L2',selfcheck=False):
    """
    loglikelihood_dict0 = self.loglikelihood_dict_sw
    residual_dict0 = self.residual_dict_sw
    data_std = self.datastd
    update=None
    data=self.surfacewaves.data
    predictions=self.prop_phttime_predictions
    path_dists = self.surfacewaves.path_dists
    predictions0=None
    foutlier=self.foutlier
    widthoutlier=self.widthoutlier
    degfree=self.degfree
    norm_coeffs0=self.likedist_normcoeffs
    distribution=self.likedist
    norm=self.misfit_norm
    """
    
    ll_sum = 0. # total loglikelihood
    res_sum = 0. # total residual

    loglikelihood_dict = {}
    residual_dict = {}
    norm_coeffs = {}
    for dataset in loglikelihood_dict0:
        loglikelihood_dict[dataset] = {}
        residual_dict[dataset] = {}
        norm_coeffs[dataset] = {}
         
    # if type(update)==tuple, update only the loglikelihoods for a certain 
    # combination of dataset and period. This applies for hyperparameter updates
    # otherwise, update everything
    if type(update)==tuple:
        for dataset in loglikelihood_dict0:
            for period in loglikelihood_dict0[dataset]:
                if dataset==update[0] and period in update[1]:
                    loglike,residuals,normcoeffs = calc_loglike(
                        dataset,period,data,predictions,path_dists,data_std,
                        distribution=distribution,norm=norm,norm_coeffs=None,
                        foutlier=foutlier,widthoutlier=widthoutlier,df=degfree)
                    residual_dict[dataset][period] = residuals
                    loglikelihood_dict[dataset][period] = loglike
                    norm_coeffs[dataset][period] = normcoeffs
                else:
                    norm_coeffs[dataset][period] = norm_coeffs0[dataset][period]
                    loglikelihood_dict[dataset][period] = loglikelihood_dict0[dataset][period]
                    residual_dict[dataset][period] = residual_dict0[dataset][period]
                res_sum += np.sum(residual_dict[dataset][period])
                ll_sum += np.sum(loglikelihood_dict[dataset][period])
                
    elif update is None or distribution == 'normal':
        if update == 'predictions':
            nc = norm_coeffs0 # use norm coeffs
        else:
            nc = None # re-calculate norm coeffs
        for dataset in loglikelihood_dict0:
            for period in loglikelihood_dict0[dataset]:
                loglike,residuals,normcoeffs = calc_loglike(
                    dataset,period,data,predictions,path_dists,data_std,
                    distribution=distribution,norm=norm,norm_coeffs=nc,
                    foutlier=foutlier,widthoutlier=widthoutlier,df=degfree)
                residual_dict[dataset][period] = residuals
                loglikelihood_dict[dataset][period] = loglike
                norm_coeffs[dataset][period] = normcoeffs
                res_sum += np.sum(np.abs(residual_dict[dataset][period]))
                ll_sum += np.sum(loglikelihood_dict[dataset][period])
        
    # if updating the predictions, the hyperparameters do not change
    # taking only the indices into account where the data actually changed
    # this speeds up calculation for the more complicated distributions 
    # (student's t or outlier distribution). For the normal distribution, this
    # has no benefit and is even slower because of the overhead.
    elif update == 'predictions':
        for dataset in loglikelihood_dict0:
            for period in loglikelihood_dict0[dataset]:
                residual_dict[dataset][period] = residual_dict0[dataset][period].copy()
                loglikelihood_dict[dataset][period] = loglikelihood_dict0[dataset][period].copy()
                norm_coeffs[dataset][period] = norm_coeffs0[dataset][period] # norm_coeffs do not change
                idx_mod = np.where(predictions0[dataset][period]!=predictions[dataset][period])[0]
                if len(idx_mod)>0:
                    loglike,residuals,normcoeffs = calc_loglike(dataset,period,
                        data,predictions,path_dists,data_std,
                        distribution=distribution,norm=norm,norm_coeffs=norm_coeffs0,
                        foutlier=foutlier,widthoutlier=widthoutlier,df=degfree,
                        idx_mod=idx_mod)
                    residual_dict[dataset][period][idx_mod] = residuals
                    loglikelihood_dict[dataset][period][idx_mod] = loglike
                res_sum += np.sum(np.abs(residual_dict[dataset][period]))
                ll_sum += np.sum(loglikelihood_dict[dataset][period])

    else:
        raise Exception()
        
    if selfcheck:
        total_residual,loglike,ncs = get_loglikelihood_alt(
            data,predictions,data_std,path_dists,norm=norm,
            distribution=distribution,foutlier=foutlier,widthoutlier=widthoutlier,
            degfree=degfree,norm_coeffs=None)
        if total_residual != res_sum or not np.allclose(loglike,ll_sum):
            raise Exception("error in calculation of loglikelihoods!")
        
    return res_sum,ll_sum,residual_dict,loglikelihood_dict,norm_coeffs
 

# function just for testing, otherwise unused
def get_phi(data,predictions,path_dists,data_std):
    
    phi = 0
    for dataset in data:
        for period in data[dataset]:
            residuals = np.abs(path_dists[dataset][period] / predictions[dataset][period] -
                               path_dists[dataset][period] / data[dataset][period])
            phi += np.sum((residuals/data_std[dataset][period])**2)   
    return phi

            
def calc_loglike(dataset,period,data,predictions,path_dists,sigma,
                 norm='L2',distribution='normal',norm_coeffs=None,
                 foutlier=None,widthoutlier=None,df=None,idx_mod=(),
                 minerror=None):
    
    # this is the likelihood P(d|m) for a hierarchical Bayesian formulation
    # meaning that the std is also considered unknown.
    # however, if it is fixed during the model search the first term will just cancel out   

    # For testing: reconstruct the prior by setting the likelihood to a 
    # fixed value. This way it cancels out in all calculations above
    # equivalent to having no data. The algorithm will just sample
    # the prior. For the number of cells this means a Gaussian around the
    # initial number of cells (is this correct?). For the data std a uniform
    # distribution in case data_std = 'absolute'
    # return 0,0,[None] # for testing, should reconstruct the prior

    # the standard deviation can be a vector of the same size as data or an
    # array with a single value (same std for all measurements).
    #              P(d|m) ~ prod(1/sigma**N * exp(-0.5*(G(m)-d)**2/sigma**2)) 
    # -> logspace: P(d|m) ~ sum(-log(sigma) - 0.5*(G(m)-d)**2)/sigma**2
    #  norm_coeffs contains the (log) normalization coefficients for the
    # distribution. By storing them, we save some computational time.
 
    residuals = np.abs(path_dists[dataset][period][idx_mod] / predictions[dataset][period][idx_mod] -
                       path_dists[dataset][period][idx_mod] / data[dataset][period][idx_mod])
    if minerror is not None:
        residuals[residuals<minerror] = minerror
         
    sig = sigma[dataset][period]
    if len(sig) > 1:
        sig = sig[idx_mod]
    
    if distribution == 'normal':
        if norm == 'L2':
            # the np.sqrt(2*np.pi) term is not necessary (always cancels out in the
            # likelihood ratios), but is kept here to make it more comparable to
            # the outlier model likelihoods which include the 2pi term.
            if norm_coeffs is None:
                nc = -np.log(np.sqrt(2*np.pi)*sig)
            else:
                nc = norm_coeffs[dataset][period]
            loglikelihood = nc - 0.5*(residuals/sig)**2
        elif norm == 'L1':
            if norm_coeffs is None:
                nc = -np.log(2*sig)
            else:
                nc = norm_coeffs[dataset][period]
            loglikelihood = nc - np.abs(residuals/sig)

    elif distribution == 'outliers':
        nc = [None] # doesn't work with this kind of distribution
        # data_std can be an array or a single value
        if norm == 'L2':
            likelihood = ((1-foutlier[dataset][period]) * 1./(np.sqrt(2*np.pi)*sig) *
                          np.exp((-(residuals/sig)**2)/2.) +
                          foutlier[dataset][period]/widthoutlier[dataset][period])
        elif norm == 'L1':
            likelihood = ((1-foutlier[dataset][period]) * 1./sig *
                          np.exp(-residuals/sig) + 
                          foutlier[dataset][period]/widthoutlier[dataset][period])
        # waterlevel (1e-300) to make sure that the log doesn't give -inf
        loglikelihood = np.log(likelihood+1e-300)
                
    # calculation of gammaln is rather slow, therefore it makes
    # sense to store the norm coefficients so that they do not
    # have to be recalculated when only updating the residuals            
    elif distribution == 'students_t':
        # data_std can be an array or a single value
        if norm_coeffs is None:
            nc = (gammaln((df[dataset][period]+1)/2) - 
                  gammaln(df[dataset][period]/2) -
                  0.5*np.log(df[dataset][period]*np.pi*sig**2) )
        else:
            nc = norm_coeffs[dataset][period]
            
        loglikelihood = (
            nc - (df[dataset][period]+1)/
                  2*np.log(1+residuals**2 / (df[dataset][period]*sig**2)) )
            
        # nc2 = (gammaln((df+1)/2) - gammaln(df/2)
        #       - 0.5*np.log(df*np.pi*data_std[dataset][period]**2) )    
        # ll = np.sum(
        #     nc - (df+1)/2*np.log(1+residuals**2 /
        #                           (df*data_std[dataset][period]**2)))
        # ll_test = np.sum(
        #       studt.logpdf(residuals,df,loc=0,scale=data_std[dataset][period]))
        # if not np.array_equal(nc,nc2):
        #     print("norm coeff error",nc,nc2)
        #     raise Exception()
        # if np.around(ll,8)!=np.around(ll_test,8):
        #     print(ll,ll_test)
        #     raise Exception()
        # loglikelihood += ll
        if norm=='L1':
            raise Exception("L1 norm not implemented for Student's t-distribution.")
            
    return loglikelihood,residuals,nc
        

def check_prior_range(z,periods,min_vsmodel,max_vsmodel,
                      data_dict,path_dists_dict,printfile=None):

    if (np.diff(periods)>0.).any():
        raise Exception("self.surfacewaves.periods array has to be strictly decreasing! Otherwise dccurve will not work properly.")

    slowness_ray_max = np.zeros(len(periods))
    slowness_lov_max = np.zeros(len(periods))
    slowness_ray_min = np.zeros(len(periods))
    slowness_lov_min = np.zeros(len(periods))
    
    min_vpprofile = min_vsmodel*1.7
    slowness_ray_min,slowness_lov_min = profile_to_phasevelocity(
        z,min_vsmodel,min_vpprofile,periods)
    max_vpprofile = max_vsmodel*1.9
    slowness_ray_max,slowness_lov_max = profile_to_phasevelocity(
        z,max_vsmodel,max_vpprofile,periods)
        
    with open(printfile,"w") as f:
        for dataset in data_dict:
            f.write(f"Checking dataset {dataset}\n")
            for pi,period in enumerate(periods):
                try:
                    phasevels = path_dists_dict[dataset][period]/data_dict[dataset][period]
                except:
                    continue
                if "ray" in dataset:
                    if (phasevels > 1./slowness_ray_max[pi]).any():
                        f.write("Warning: the maximum observed Rayleigh phase velocity at period=%.1fs is %.3fkm/s but the prior allows only a phase velocity up to %.3fkm/s. Mean phasevel = %.3fkm/s\n" %(period,np.max(phasevels),1./slowness_ray_max[pi],np.mean(phasevels)))
                    if (phasevels < 1./slowness_ray_min[pi]).any():
                        f.write("Warning: the minimum observed Rayleigh phase velocity at period=%.1fs is %.3fkm/s but the prior allows only a phase velocity down to %.3fkm/s. Mean phasevel = %.3fkm/s\n" %(period,np.min(phasevels),1./slowness_ray_min[pi],np.mean(phasevels)))
                if "lov" in dataset:
                    if (phasevels > 1./slowness_lov_max[pi]).any():
                        f.write("Warning: the maximum observed Love phase velocity at period=%.1fs is %.3fkm/s but the prior allows only a phase velocity up to %.3fkm/s. Mean phasevel = %.3fkm/s\n" %(period,np.max(phasevels),1./slowness_lov_max[pi],np.mean(phasevels)))
                    if (phasevels < 1./slowness_lov_min[pi]).any():
                        f.write("Warning: the minimum observed Love phase velocity at period=%.1fs is %.3fkm/s but the prior allows only a phase velocity down to %.3fkm/s. Mean phasevel = %.3fkm/s\n" %(period,np.min(phasevels),1./slowness_lov_min[pi],np.mean(phasevels)))
    
    

def read_surfacewave_module(output_folder,verbose=True):
    
    module_fname = os.path.join(output_folder,"data_modules","surfacewavedata.pgz")
    if os.path.isfile(module_fname):
        if verbose:
            print("found surfacewave module file, reading...")
        with gzip.open(module_fname, "rb") as f:
            return pickle.load(f)
    else:
        return None


def dump_surfacewave_module(output_folder,module):
    
    module_fname = os.path.join(output_folder,"data_modules","surfacewavedata.pgz")
    if not os.path.exists(os.path.dirname(module_fname)):
        os.makedirs(os.path.dirname(module_fname))
    with gzip.open(module_fname,"wb") as f:
        pickle.dump(module,f)
        

def read_data(input_files,coordinates='latlon',projection=None,
              plotting=False,output_folder=None):
    
    if input_files == [] or input_files is None:
        swd = SurfacewaveData()
        dump_surfacewave_module(output_folder, swd)
        return swd
    
    swd = read_surfacewave_module(output_folder)
    if swd is not None:
        return swd
    
    periods = []
    
    traveltimes = {}
    sources = {}
    receivers = {}
    datastd = {}
    datacount = {}
    
    # READ INPUT DATA
    i = 0
    for fpath in input_files: 
        i += 1
        # read input file header
        with open(fpath,"r") as f:
            line1 = f.readline()            
            line2 = f.readline()
            line3 = f.readline()
            line4 = f.readline()
            if not(line1[0] == '#' and line2[0] == '#' and line3[0] == '#' and line4[0] == '#'):
                raise Exception("Input file should contain 4 header lines starting with #")

            dataset_name = line1[1:].strip().lower()
            if dataset_name in traveltimes.keys():
                dataset_name += " %d" %i
            if not ("rayleigh" in dataset_name or "love" in dataset_name):
                print(f"first header line: {dataset_name}")
                raise Exception("The first header line should contain either a 'rayleigh' or 'love' key word to identify the data type.") 
            traveltimes[dataset_name] = {}
            sources[dataset_name] = {}
            receivers[dataset_name] = {}
            datastd[dataset_name] = {}
            datacount[dataset_name] = 0
            
            dataset_periods = np.array(line3.split()[2:]).astype(float)
            periods.append(dataset_periods)
            dataset_data = np.loadtxt(fpath)
            if np.shape(dataset_data)[1] < len(dataset_periods)+4:
                raise Exception("number of columns and periods in header do not match (LAT1 LON1 LAT2 LON2 TRAVELTIME TRAVELTIME ...).")

    
            for pidx,period in enumerate(dataset_periods):
                
                data = dataset_data.copy()
                if np.shape(dataset_data)[1] == len(periods)+5:
                    data = data[:,(0,1,2,3,pidx+4,-1)]
                else:
                    data = np.column_stack((data[:,(0,1,2,3,pidx+4)],np.ones(len(data))))
                data = data[~np.isnan(data[:,-2])]
                data = data[data[:,-2]>0.0]
                
                # if len(data)>10000:
                #     print(dataset_name,period,len(data))
                #     print("warning: limiting the dataset size to 10000 per period")
                #     data = data[:10000]
                
                if len(data)>0:
                    traveltimes[dataset_name][period] = data[:,-2]
                    sources[dataset_name][period] = data[:,:2]
                    receivers[dataset_name][period] = data[:,2:4]
                    datastd[dataset_name][period] = data[:,-1]
                else:
                    continue
                
                datacount[dataset_name] += len(data)
              
    periods = np.unique(np.hstack(periods)) 
    
    print(f"Successfully read {len(traveltimes)} surface-wave datasets:")
    for dataset in traveltimes:
        print(f"{dataset}: {datacount[dataset]} measurements")
        
    stations = []
    for dataset in traveltimes:
        for period in traveltimes[dataset]:
            stats = np.unique(np.vstack((sources[dataset][period],
                                         receivers[dataset][period])),axis=0)
            if len(stations) == 0:
                stations = stats
            else:
                stations = np.unique(np.vstack((stations,stats)),axis=0)
                
    print(f"The datasets contain a total of {len(stations)} sources and receivers")
    
    if coordinates == 'latlon':
        
        g = pyproj.Geod(ellps='WGS84')
        if projection is None:
            central_lon = np.around(np.mean(stations[:,1]),1)
            central_lat = np.around(np.mean(stations[:,0]),1)
            projection_str = "+proj=tmerc +datum=WGS84 +lat_0=%f +lon_0=%f" %(central_lat,central_lon)
        else:
            projection_str = projection
            
        p = pyproj.Proj(projection_str)
        
        statsxy = p(stations[:,1],stations[:,0])

        for dataset in traveltimes:
            for period in traveltimes[dataset]:
 
                az,baz,dist_real = g.inv(sources[dataset][period][:,1],
                                         sources[dataset][period][:,0],
                                         receivers[dataset][period][:,1],
                                         receivers[dataset][period][:,0]) 
                     
                sources[dataset][period] = np.column_stack(p(sources[dataset][period][:,1],
                                                             sources[dataset][period][:,0]))/1000.
                receivers[dataset][period] = np.column_stack(p(receivers[dataset][period][:,1],
                                                               receivers[dataset][period][:,0]))/1000.
        
                dist_proj = np.sqrt((sources[dataset][period][:,0] - 
                                     receivers[dataset][period][:,0])**2 + 
                                     (sources[dataset][period][:,1] - 
                                      receivers[dataset][period][:,1])**2)
                
                distortion_factor = dist_proj/(dist_real/1000.)
                if np.max(distortion_factor)>1.02 or np.min(distortion_factor)<0.98:
                    print("WARNING: significant distortion by the coordinate projection!")

                traveltimes[dataset][period] *= distortion_factor
                
    else:
        central_lon = 0.
        central_lat = 0.            
        statsxy = (stations[:,0]*1000.,stations[:,1]*1000.)
        g = pyproj.Geod(ellps='WGS84')
        projection_str = "+proj=tmerc +datum=WGS84 +lat_0=%f +lon_0=%f" %(central_lat,central_lon)
        p = pyproj.Proj(projection_str)
        
        

    if plotting:
        if not os.path.exists(os.path.join(output_folder,"indata_figures")):
            os.makedirs(os.path.join(output_folder,"indata_figures"))
        
        for dataset in traveltimes:
            for period in traveltimes[dataset]:
                
                data = traveltimes[dataset][period]
                if len(data)>5000:
                    randidx = np.random.choice(np.arange(len(data)),5000,replace=False)
                else:
                    randidx = np.arange(len(data))
                plt.ioff()
                fig = plt.figure(figsize=(12,10))
                
                dist = np.sqrt((sources[dataset][period][randidx,0] - 
                                receivers[dataset][period][randidx,0])**2 + 
                                (sources[dataset][period][randidx,1] - 
                                 receivers[dataset][period][randidx,1])**2)
                velocities = dist/traveltimes[dataset][period][randidx]
                
                if coordinates=='latlon':
                    # uses a WGS84 ellipsoid as default to create the transverse mercator projection
                    central_lon = None
                    central_lat = None
                    for element in p.definition_string().split():
                        if element.split("=")[0]=="lon_0":
                            central_lon = float(element.split("=")[1])
                        elif element.split("=")[0]=="lat_0":
                            central_lat = float(element.split("=")[1])
                    if central_lon is None or central_lat is None:
                        raise Exception("could not read lat_0 and lon_0 from provided projection")
                    proj = ccrs.TransverseMercator(central_longitude=central_lon,central_latitude=central_lat)
                    axm = plt.axes(projection=proj)
                    segments = np.hstack((np.split(1000*sources[dataset][period][randidx],len(randidx)),
                                          np.split(1000*receivers[dataset][period][randidx],len(randidx))))
                    lc = LineCollection(segments, linewidths=0.3)
                    lc.set(array=velocities,cmap='jet_r')
                    # PlateCarree is the projection of the input coordinates, i.e. "no" projection (lon,lat)
                    # it is also possible to use ccrs.Geodetic() here 
                    axm.add_collection(lc)
                    axm.plot(stations[:,1],stations[:,0],'rv',ms = 2,transform = ccrs.PlateCarree())
                    axm.coastlines(resolution='50m')
                    axm.add_feature(cf.BORDERS.with_scale('50m'))
                    axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
                    axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
                    gl = axm.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                       linewidth=0.5, color='black', alpha=0.5, linestyle='--')
                else:
                    axm = fig.add_subplot(111)
                    segments = np.hstack((np.split(sources[dataset][period][randidx],len(randidx)),
                                          np.split(receivers[dataset][period][randidx],len(randidx))))
                    lc = LineCollection(segments, linewidths=0.3)
                    lc.set(array=velocities,cmap='jet_r')
                    axm.add_collection(lc)
                    axm.plot(stations[:,0],stations[:,1],'rv',ms=2)
                    axm.set_aspect('equal')
                plt.colorbar(lc,fraction=0.05,shrink=0.5,label='velocity')
                plt.savefig(os.path.join(output_folder,"indata_figures",f"input_data_{dataset}_{period}.jpg"),dpi=200,bbox_inches='tight')
                plt.close(fig)
                
    

    stationsxy = np.column_stack((statsxy[0]/1000.,statsxy[1]/1000.))
    swd =  SurfacewaveData(sources=sources,receivers=receivers,
                           traveltimes=traveltimes,datastd=datastd,
                           datacount=datacount,periods=periods,
                           stations=stationsxy,projection=projection_str)
    
    dump_surfacewave_module(output_folder, swd)
 
    return swd
    
    
    
