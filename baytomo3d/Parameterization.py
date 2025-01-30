#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:35:55 2021

@author: emanuel
"""

import numpy as np
from operator import itemgetter
from copy import deepcopy
from scipy.spatial import KDTree, Delaunay, ConvexHull, distance
from scipy.spatial.distance import cdist
from scipy.interpolate import NearestNDInterpolator,interp1d
from scipy.ndimage import gaussian_filter
from scipy.stats import truncnorm
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import time
try:
    import pywt
except:
    pass
    #print("pywavelets module not installed.")


class voronoi_cells(object):
    
    def __init__(self,gridpoints,shape,convexhull,velmin,velmax,vpvs,
                 init_no_points,moho_depths=None,crustal_model=None,
                 inverse_z_scaling=None,
                 vpvsmin=None,vpvsmax=None,mingrad=None,maxgrad=None): 
        
        self.gridpoints = gridpoints
        dx = np.diff(np.unique(gridpoints[:,0]))[0]
        dy = np.diff(np.unique(gridpoints[:,1]))[0]
        dz = np.diff(np.unique(gridpoints[:,2]))[0]
        self.shape = shape
        
        self.minx = np.min(self.gridpoints[:,0])
        self.maxx = np.max(self.gridpoints[:,0])
        self.miny = np.min(self.gridpoints[:,1])
        self.maxy = np.max(self.gridpoints[:,1])
        self.z = np.unique(self.gridpoints[:,2])
        self.z_unscaled = inverse_z_scaling(self.z)
        self.minz = self.z[0]
        self.maxz = self.z[-2]
        self.inverse_z_scaling = inverse_z_scaling
        
        self.vsprecision = 0.01 # round vs and vpvs in 0.05 steps (2.4, 2.45, 2.5, ...)
        
        # experimental, treat points in the halfspace separately
        self.separate_halfspace = True
        
        # experimental
        self.smoothmodel = False
        if self.smoothmodel:
            print("warning:smoothing matrix active")
            self.create_smoothing_matrix()
        
        # these are functions of the type vmin = self.vsmin(depth)
        self.vsmin = velmin
        self.vsmax = velmax
        self.vs_range = interp1d(self.z,self.vsmax(self.z)-self.vsmin(self.z))
        # this function returns the mean vs value for a given depth
        self.mean_vs = interp1d(self.z,self.vsmin(self.z)+0.5*self.vs_range(self.z))
        self.p_range = 1 # between 0 and 1
        
        self.mingrad = mingrad
        self.maxgrad = maxgrad
        
        self.vpvs = vpvs # can be a function of vs or a fixed vpvs ratio or an array of vpvs values
        self.vpvsmin = vpvsmin # min prior on vpvs ratio
        self.vpvsmax = vpvsmax # max prior on vpvs ratio
        if self.vpvsmin is None and self.vpvsmax is None:
            self.vpvs_range = None
        else:
            self.vpvs_range = self.vpvsmax-self.vpvsmin
        
        # convexhull is a 3D hull in scaled coordinates, same as self.gridpoints
        self.convexhull = convexhull
        # get all gridpoints that are within the hull
        # outside gridpoints are ignored in the calculations for speedup
        self.valid_gpts = self.convexhull.find_simplex(self.gridpoints)>-1
        
        # initialize points
        points_x = np.random.uniform(low=self.minx,high=self.maxx,
                                     size=init_no_points)
        points_y = np.random.uniform(low=self.miny,high=self.maxy,
                                     size=init_no_points)
        # higher probability for the creation of new cells at shallow depths
        points_z = np.random.triangular(self.minz,self.minz,self.maxz,
                                        size=init_no_points)
        points = np.column_stack((points_x,points_y,points_z))
        while True:
            points = np.column_stack((points_x,points_y,points_z))
            simplex = convexhull.find_simplex(points)
            if (simplex >= 0).all():
                break # all points are inside the hull
            outside_idx = np.where(simplex < 0.)[0]
            points_x[outside_idx] = np.random.uniform(
                low=self.minx,high=self.maxx,size=len(outside_idx))
            points_y[outside_idx] = np.random.uniform(
                low=self.miny,high=self.maxy,size=len(outside_idx))
            points_z[outside_idx] = np.random.triangular(
                self.minz,self.minz,self.maxz,size=len(outside_idx))
        self.points = points

        while True:
            # remove empty Voronoi cells and add more points where necessary
            kdt = KDTree(self.points)
            nndist,nnidx = kdt.query(self.gridpoints)
            #gpts_dict = self.get_gpts_dictionary()
            remove_idx = []
            newpoints = []
            for pnt_idx in range(len(self.points)):
                if np.sum(nnidx==pnt_idx) == 0:
                    remove_idx.append(pnt_idx)
                    continue
                # the following is to make sure that within a single voronoi cell
                # all velocities are within the vsmin-vsmax range
                # if the cell extends across a large depth range, it may be that there
                # is no solution, in that case, try adding more cells
                # this is especially a problem if the z_scaling is low (cells are vertically stretched)
                # THIS IS NOT NEEDED WITH THE P_RANGE APPROACH
                # vlim1 = np.max(self.vsmin(self.gridpoints[nnidx==pnt_idx,2]))
                # vlim2 = np.min(self.vsmax(self.gridpoints[nnidx==pnt_idx,2]))
                # if vlim2<vlim1:
                #     x,y,z = self.gridpoints[np.random.choice(np.where(nnidx==pnt_idx)[0])]
                #     newpoint = np.array([
                #         np.random.uniform(np.max([x-dx,self.minx]),np.min([x+dx,self.maxx])),
                #         np.random.uniform(np.max([y-dy,self.miny]),np.min([y+dy,self.maxy])),
                #         np.random.uniform(np.max([z-dz,self.minz]),np.min([z+dz,self.maxz]))])
                #     if convexhull.find_simplex(newpoint[:2])>=0.:
                #         newpoints.append(newpoint)
            if len(remove_idx) > 0:
                self.points = np.delete(self.points,remove_idx,axis=0)
            if len(newpoints)==0:
                break
            else:
                self.points = np.vstack((self.points,newpoints))
            if len(self.points)>10000:
                raise Exception()
        #self.vertex_neighbors,_ = self.get_neighbors(0,points=self.points)
        
        # if self.Nnearest=1, it's a classical nearest-neighbor interpolation (Voronoi cells)
        # if self.Nnearest>1, the contributions of the closest N cells are taken into account
        # EXPERIMENTAL
        self.Nnearest = 1
        if self.Nnearest>1:
            print(f"Warning: taking the contributions of the {self.Nnearest} nearest cells into account for the model interpolation.")
        self.setup_nearest_neighbor_arrays()    
        
        if moho_depths is not None:
            if len(moho_depths.shape)!=1 or len(moho_depths)!=len(gridpoints):
                raise Exception("Moho depths have to be provided as 1D array of the same lengths as gridpoints.")
            if np.max(moho_depths) > np.max(self.z):
                print("Warning: Moho depths are larger than the maximum model depth.")
            self.fixed_moho = True
            self.crustidx = np.where(gridpoints[:,-1]<=moho_depths)[0]
            self.mantleidx = np.where(gridpoints[:,-1]>moho_depths)[0]
            self.moho_func = NearestNDInterpolator(gridpoints[:,:2],moho_depths)
        else:
            self.fixed_moho = False
        if crustal_model[0] is not None:
            self.crust_idx = ~np.isnan(crustal_model[0])
            self.crust_vs = crustal_model[1][self.crust_idx]
            self.crust_vpvs = crustal_model[0][self.crust_idx]/crustal_model[1][self.crust_idx]
        else:
            self.crust_idx = None
                    
        self.points_backup = None
        self.parameters_backup = None
        self.vpvs_backup = None
        self.idx_mod_gpts = ()
        self.vsfield = np.zeros(len(self.gridpoints))
        self.vpvsfield = np.zeros(len(self.gridpoints))
        #self.subidx_gridpts_new_cell = None
        #self.subgrid_lengths = None        
        self.action = None
        #self.idx_mod_gpts = None
        #self.idx_mod = None
        #self.idx_subset_points = None
        self.psi2amp = None
        self.psi2 = None
        self.psi2amp_backup = None
        self.psi2_backup = None                        
        self.propstd_dimchange = 'uniform' # will be adapted when running
        #self.vertex_neighbors_backup = None
        self.grid_nnidx_backup = None
        self.grid_nndists_backup = None
        self.propdist_std = None
        self.dx = None
        
    # def psi2amp_update(self,idx,psi2ampnew,backup=True):
        
    #     if backup:
    #         self.psi2amp_backup = self.psi2amp.copy()
            
    #     self.psi2amp[idx] = psi2ampnew
    #     self.action='psi2amp_update'
    #     self.idx_mod = idx
        
    #     self.idx_mod_gpts = self.gpts_dict[idx]
        
        
    # def psi2_update(self,idx,psi2new,backup=True):
        
    #     if backup:
    #         self.psi2_backup = self.psi2.copy()
            
    #     self.psi2[idx] = psi2new
    #     if self.action=='psi2amp_update':
    #         self.action='anisotropy_birth_death'
    #     else:
    #         self.action='psi2_update'
    #     self.idx_mod = idx
        
    #     self.idx_mod_gpts = self.gpts_dict[idx]

    def setup_nearest_neighbor_arrays(self):
                
        kdt = KDTree(self.points)
        self.grid_nndists,self.grid_nnidx = kdt.query(self.gridpoints,k=self.Nnearest)
        if self.Nnearest==1:
            # make sure the shape are always of type (Ngridpoints,Nnearest)
            self.grid_nndists = self.grid_nndists[None].T # add one axis
            self.grid_nnidx = self.grid_nnidx[None].T # add one axis
            
        
    def get_vertical_gradient(self,model=None):
        
        if model is None:
            vs = myround(self.vsmin(self.points[:,-1]) + self.vs_range(self.points[:,-1])*self.parameters,self.vsprecision)
            if self.Nnearest>1:       
                # this is probably uite slow and should be avoided
                model = np.average(vs[self.grid_nnidx],weights=1./(1+self.grid_nndists),axis=1)
            else:
                model = vs[self.grid_nnidx][:,0]
        zdiff = np.diff(np.reshape(model,self.shape),axis=2)
        zgrad = zdiff/np.diff(self.z_unscaled)
        
        return np.min(zgrad),np.max(zgrad)
        
    
    def check_gradient(self,point,mingrad,maxgrad,idx=None,neighbors=None):
        
        if mingrad is None or maxgrad is None:
            raise Exception("define both mingrad and maxgrad")
            
        velmin = self.vsmin(point[2])
        velmax = self.vsmax(point[2])
        
        #if self.idx_subset_points is None:
        if neighbors is None:
            _,neighbors = self.get_neighbors(idx)
        #else:
        #    neighbors = self.idx_subset_points
            
        above = self.points[neighbors,2]<point[2]
        below = self.points[neighbors,2]>point[2]
        idx_below = neighbors[below]
        idx_above = neighbors[above]
        
        dz = (self.inverse_z_scaling(point[2]) - 
              self.inverse_z_scaling(self.points[neighbors,2]))
        velmin = np.max(np.hstack((
            self.parameters[idx_above] + mingrad * dz[above],
            self.parameters[idx_below] + maxgrad * dz[below],
            velmin)))
        velmax = np.min(np.hstack((
            self.parameters[idx_above] + maxgrad * dz[above],
            self.parameters[idx_below] + mingrad * dz[below],
            velmax)))
        
        return velmin,velmax
        

    def get_mod_point_coords(self,idx=None):
        
        if idx is not None:
            return self.points[idx,:2],self.points[idx,2]
        
        if self.action == 'death':
            return self.points_backup[self.idx_mod,:2],self.points_backup[self.idx_mod,2]
        else:
            return self.points[self.idx_mod,:2],self.points[self.idx_mod,2]
    
    
    def parameter_update(self,idx,propdist_std,backup=True):
        
        self.action='velocity_update'
        if backup:
            self.backup_mod()            
        self.idx_mod = idx
        self.propdist_std = propdist_std # store for delayed rejection scheme
        # store dx for delayed rejection scheme
        self.dx = np.random.normal(loc=0.0,scale=propdist_std)
        self.parameters[idx] += self.dx
        #self.parameters[idx] = myround(self.parameters[idx],self.vsprecision)
        
        if self.parameters[idx]<0 or self.parameters[idx]>1:
            return False
        
        if self.fixed_moho:
            if not self.moho_check():
                return False
            
        self.idx_mod_gpts = np.any(self.grid_nnidx==idx,axis=1)
                
        return True
        
    
    def vpvs_ratio_update(self,idx,propdist_std,backup=True):
        
        self.action='vpvs_update'
        if backup:
            self.backup_mod()
        self.idx_mod = idx
        
        self.vpvs[idx] += np.random.normal(loc=0.0,scale=propdist_std)
        self.vpvs[idx] = myround(self.vpvs[idx],self.vsprecision)
        
        if self.vpvs[idx]<self.vpvsmin or self.vpvs[idx]>self.vpvsmax:
            return False
        
        self.idx_mod_gpts = np.any(self.grid_nnidx==idx,axis=1)
                
        return True
    
        
    def add_point(self,anisotropic=False,birth_prop='uniform', backup=True):

        self.action='birth'        
        if backup:
            self.backup_mod()
           
        #prop_point_z = np.random.uniform(self.minz,self.maxz)       
        # higher probability for the creation of new cells at shallow depths
        #prop_point_z = np.random.triangular(self.minz_scaled,self.minz_scaled,self.maxz_scaled)
        # alternative 1:
        # this gives a combination of uniform and triangular distributions
        # self.total_steps%5 means that a value of z=minz is five times more
        # likely than a value of z=maxz
        #if self.total_steps%5 == 0:
        #prop_point_z = np.random.uniform(self.minz_scaled,self.maxz_scaled)
        #else:
        #    prop_point_z = np.random.triangular(self.minz_scaled,self.minz_scaled,self.maxz_scaled)
        # alternative 2: controlled by the distribution of z points
        #if self.mindepth is not None:
        #    zi = np.random.randint(np.abs(self.mindepth-self.z).argmin(),len(self.z)-1)
        #else:
        # if self.total_steps%2 == 0:
        #     prop_point_z = np.random.uniform(self.minz_scaled,self.maxz_scaled)
        #     #prop_point_z = self.z[np.abs(prop_point_z-self.z).argmin()]
        # else:
        #     zi = np.random.randint(0,len(self.z_scaled))
        #     prop_point_z = self.z_scaled[zi]
        #prop_point_z = self.z[zi] + np.random.uniform(0,self.z[zi+1]-self.z[zi]) 
           
        while True: # make sure point is within prior (convex hull)
            prop_point_x = np.random.uniform(self.minx,self.maxx)
            prop_point_y = np.random.uniform(self.miny,self.maxy)
            zi = np.random.randint(0,len(self.z)-1)
            if self.separate_halfspace and zi==len(self.z)-2: # points at the halfspace layer are treated separately
                prop_point_z = self.z[-1]
            else:
                prop_point_z = self.z[zi] + np.random.uniform(0,self.z[zi+1]-self.z[zi])
            point = np.hstack((prop_point_x,prop_point_y,prop_point_z))
            if self.convexhull.find_simplex(point) >= 0:
                break
            
        self.idx_mod = len(self.parameters)
        
        self.points = np.vstack((self.points,point))
        if anisotropic:
            # newborn cell has no anisotropy
            self.psi2amp = np.append(self.psi2amp,0.)
            self.psi2 = np.append(self.psi2,0.)
        
        # getting all neighbours of new point
        #self.vertex_neighbors,self.idx_subset_points = self.get_neighbors(
        #    self.idx_mod,points=self.points)
        self.update_neighbor_lists(self.points, 'birth', self.idx_mod)
               
        if type(birth_prop[1])==type('uniform'):
            self.prop_p = np.random.uniform(0,1)
            propstd = birth_prop[1]
            if not callable(self.vpvs):
                self.prop_vpvs = np.random.uniform(self.vpvsmin,self.vpvsmax)
        else:
            if type(birth_prop[1])==type(np.array([])):
                zidx = np.abs(prop_point_z - birth_prop[0]).argmin()
                propstd = birth_prop[1][zidx]
            else:
                propstd = birth_prop[1]
            # the velocity for the newborn cell is drawn from a normal 
            # distribution around the average of the previous velocity values
            # in that area
            if np.sum(self.idx_mod_gpts) > 0:
                idx_neighbor = self.grid_nnidx_backup[self.idx_mod_gpts]
            else:
                idx_neighbor = np.sum((point-self.points_backup)**2,axis=1).argmin()                
            p_base = np.mean(self.parameters_backup[idx_neighbor])
            self.tn_p = rv_truncnorm(p_base,propstd,0,1)
            try:
                self.prop_p = self.tn_p.rvs()
            except:
                print(self.idx_mod_gpts)
                print(idx_neighbor)
                print(self.parameters_backup[idx_neighbor])
                print(p_base,propstd,birth_prop)
                raise Exception("truncnorm error")
            #self.prop_dp = prop_p - p_base
            if self.prop_p > 1 or self.prop_p < 0:
                raise Exception("cannot happen")  
            if not callable(self.vpvs):
                vpvs_base = np.mean(self.vpvs_backup[idx_neighbor])
                self.tn_vpvs = rv_truncnorm(vpvs_base,propstd,self.vpvsmin,self.vpvsmax)
                self.prop_vpvs = self.tn_vpvs.rvs()
                #self.prop_dvpvs = prop_vpvs - vpvs_base
                if self.prop_vpvs > self.vpvsmax or self.prop_vpvs < self.vpvsmin:
                    raise Exception("cannot happen") 
            else:
                self.prop_vpvs = None
                
                
            #self.prop_dp, self.prop_dvpvs = np.random.normal(loc=0.0,scale=propstd,size=2)
            # the velocity for the newborn cell is drawn from a normal 
            # distribution around the average of the previous velocity values
            # in that area
            #if len(idx_mod_gpts) > 0:
            #    idx_neighbor = self.grid_nnidx_backup[idx_mod_gpts]
            #else:
            #    idx_neighbor = np.sum((point-self.points_backup)**2,axis=1).argmin()
            #p_base = np.mean(self.parameters_backup[idx_neighbor])
            #prop_p = p_base + self.prop_dp
            #if prop_p > 1 or prop_p < 0:
            #    return False
           # if not callable(self.vpvs):
           #     vpvs_base = np.mean(self.vpvs_backup[idx_neighbor])
           #     prop_vpvs = vpvs_base + self.prop_dvpvs
           #     if prop_vpvs > self.vpvsmax or prop_vpvs < self.vpvsmin:
           #         return False
            # else:
            #     self.prop_dvpvs = None

        self.parameters = np.append(self.parameters,self.prop_p)
        if not callable(self.vpvs):
            self.vpvs = np.append(self.vpvs,self.prop_vpvs)
                        
        # if self.maxgrad is not None or self.mingrad is not None:
        #     mingrad,maxgrad = self.get_vertical_gradient()
        #     if mingrad<self.mingrad or maxgrad>self.maxgrad:
        #         return False
            # velmin,velmax = self.check_gradient(point,self.mingrad,self.maxgrad,
            #                                     neighbors=self.idx_subset_points)
            # if velmin>velmax or vs_birth>velmax or vs_birth<velmin:
            #     return False    
            
        if self.fixed_moho:
            if not self.moho_check():
                return False
        
        # # interpolate new velocities only in the area affected by the change
        # idx_points_subset = np.append(self.idx_mod,self.idx_subset_points)

        # # getting all indices of all gridpoints of the neighboring points
        # # (this automatically includes the gridpoints in the new area)
        # idx_modified_gridpoints_all = itemgetter(*self.idx_subset_points)(self.gpts_dict)
        # self.subgrid_lengths = list(map(len,idx_modified_gridpoints_all))
        # idx_modified_gridpoints_all = np.hstack(idx_modified_gridpoints_all)
        
        # subset_voronoi_kdt = KDTree(self.points[idx_points_subset])           
        # update_point_dist, update_point_regions = subset_voronoi_kdt.query(
        #     self.gridpoints[idx_modified_gridpoints_all],eps=0)
        # update_point_regions = idx_points_subset[update_point_regions]
        # # getting all gridpoints that are nearest neighbors to the new point
        # self.subidx_gridpts_new_cell = update_point_regions==self.idx_mod
        # self.idx_mod_gpts = idx_modified_gridpoints_all[self.subidx_gridpts_new_cell]
        
        # for the prior_proposal_ratio calculation:
        self.propstd_dimchange = propstd
        
        return True
        
        
    def remove_point(self,anisotropic=False,
                     backup=True,death_prop='uniform'):
        
        self.action='death'
        if backup:
            self.backup_mod()
    
        if anisotropic:
            # choose only points without anisotropy
            ind_pnts = np.where(self.psi2amp == 0.)[0]
            if len(ind_pnts) > 0:
                self.idx_mod = np.random.choice(ind_pnts)
            else:
                return False
        else:
            # choose point to remove randomly
            self.idx_mod = np.random.randint(0,len(self.points))
        
        pnt_remove = self.points[self.idx_mod]
        p_remove = self.parameters[self.idx_mod]
        if not callable(self.vpvs):
            vpvs_remove = self.vpvs[self.idx_mod]
        
        # getting all neighbor points of removed vertice
        #_,self.idx_subset_points = self.get_neighbors(
        #    self.idx_mod)

        # getting indices of all gridpoints that will change their velocity
        #self.idx_mod_gpts = self.gpts_dict[self.idx_mod]
        
        self.points = np.delete(self.points,self.idx_mod,axis=0)
        self.parameters = np.delete(self.parameters,self.idx_mod)
        if not callable(self.vpvs):
            self.vpvs = np.delete(self.vpvs,self.idx_mod)
        if anisotropic:
            self.psi2amp = np.delete(self.psi2amp,self.idx_mod)
            self.psi2 = np.delete(self.psi2,self.idx_mod)
            
        self.update_neighbor_lists(self.points, 'death', self.idx_mod)
        
        if self.fixed_moho:
            if not self.moho_check():
                return False


        #self.idx_subset_points[self.idx_subset_points>self.idx_mod] -= 1
        
        #self.vertex_neighbors,_ = self.get_neighbors(0,points=self.points)
        
        # # now compare the velocity of the removed point with the velocity
        # # at the empty spot (inverse birth operation)
        # # the given idx_subset_points are for the NEW point indices of
        # # the neighbors (after point removal)
        # if len(self.idx_subset_points)>0:
        #     dists = np.sqrt(np.sum((self.points[self.idx_subset_points] -
        #                             pnt_remove)**2,axis=1))
        #     self.prop_dv = (vs_remove - 
        #                     np.average(self.parameters[self.idx_subset_points],
        #                                 weights=1./dists) )
        # else:
        #     self.prop_dv = 0.
            
        # if self.maxgrad is not None or self.mingrad is not None:
        #     mingrad,maxgrad = self.get_vertical_gradient()
        #     if mingrad<self.mingrad or maxgrad>self.maxgrad:
        #         return False
            # velmin,velmax = self.check_gradient(pnt_remove,self.mingrad,self.maxgrad,
            #                                     neighbors=self.idx_subset_points)
            # if vs_remove>velmax or vs_remove<velmin:
            #     raise Exception("should not happen")
            # # in this case we have to do an extra loop to check that the
            # # gradient between all other neighbors is not affected either
            # for idx in self.idx_subset_points:
            #     vmin,vmax = self.check_gradient(self.points[idx],self.mingrad,
            #                                     self.maxgrad,idx=idx)
            #     if self.parameters[idx]<vmin or self.parameters[idx]>vmax:
            #         return False
        
        # These values are only needed if not drawing from a uniform distribution
        if type(death_prop[1])!=type('uniform'):
            if type(death_prop[1])==type(np.array([])):
                zidx = np.abs(pnt_remove[2]-death_prop[0]).argmin()
                self.propstd_dimchange = death_prop[1][zidx]
            else:
                self.propstd_dimchange = death_prop[1]
            if np.sum(self.idx_mod_gpts)>0:
                idx_neighbor = self.grid_nnidx[self.idx_mod_gpts]
            else:
                idx_neighbor = np.sum((pnt_remove-self.points)**2,axis=1).argmin()      
            p_base = np.mean(self.parameters[idx_neighbor])
            #self.prop_dp = p_base-p_remove
            self.tn_p = rv_truncnorm(p_base,self.propstd_dimchange,0,1)
            self.prop_p = p_remove
            if not callable(self.vpvs):
                vpvs_base = np.mean(self.vpvs[idx_neighbor])
                #self.prop_dvpvs = vpvs_base-vpvs_remove
                self.tn_vpvs = rv_truncnorm(vpvs_base,self.propstd_dimchange,
                                            self.vpvsmin,self.vpvsmax)
                self.prop_vpvs = vpvs_remove
            else:
                #self.prop_dvpvs = None
                self.prop_vpvs = None
                    
        return True
                    
            
    def move_point(self,propmovestd,index=None,backup=True):

        self.action = 'move'
        if backup:
            self.backup_mod()        
        self.propdist_std = propmovestd
        
        if index is None:
            index = np.random.randint(0,len(self.points))
        self.idx_mod = index

        oldxy = self.points[index]
        
        self.dx = np.random.normal(loc=0.0,scale=propmovestd,size=3)
        # if point is at halfspace depth, no z movement
        if self.separate_halfspace and oldxy[2] == self.z[-1]:
            self.dx[2] = 0.
        newxy = oldxy + self.dx
        
        if self.separate_halfspace and oldxy[2] == self.z[-1]:
            if (newxy[0]>self.maxx or newxy[0]<self.minx or
                newxy[1]>self.maxy or newxy[1]<self.miny
                or self.convexhull.find_simplex(newxy) < 0 
                ):
                return False
        else:
            if (newxy[0]>self.maxx or newxy[0]<self.minx or
                newxy[1]>self.maxy or newxy[1]<self.miny or
                newxy[2]>self.maxz or newxy[2]<self.minz
                or self.convexhull.find_simplex(newxy) < 0 
                ):
                return False
            
        #if (self.parameters[index] > self.velmax(newxy[2]) or 
        #    self.parameters[index] < self.velmin(newxy[2]) ):
        #    return (np.nan,np.nan)

        # _,old_neighbors = self.get_neighbors(index)
        # self.points[index] = newxy
        # self.vertex_neighbors,new_neighbors = self.get_neighbors(index,points=self.points)
        
        
        if self.fixed_moho:
            if not self.moho_check():
                return False
        
        # idx_neighbor_indices = np.unique(np.hstack((old_neighbors,new_neighbors)))
        
        # # indices of all points that might be affected by the change
        # # (old neighbors, new neighbors and move point itself) 
        # # make sure the idx_move index is the first in the array
        # self.idx_subset_points = np.append(index,idx_neighbor_indices)
        
        # # indices of all gridpoints that may be affected by this
        # # operation, i.e. gridpoints in idx_move and in new neighbors
        # idx_modified_gridpoints_all = np.unique(np.hstack(itemgetter(*np.hstack((new_neighbors,index)))(self.gpts_dict)))
        # # we can reduce this to the gridpoints that are actually related to the new point position                           
        
        # subset_voronoi_kdt = KDTree(self.points[self.idx_subset_points])
        # update_point_dist, update_point_regions = subset_voronoi_kdt.query(self.gridpoints[idx_modified_gridpoints_all],eps=0)
        # # with the following step, the point_region indices are again
        # # related to the global point indices and not the subset indices
        # update_point_regions = self.idx_subset_points[update_point_regions]
        # idx_mod_gpts_oldpos = self.gpts_dict[index]
        # idx_mod_gpts_newpos = idx_modified_gridpoints_all[update_point_regions==index]
        # self.idx_mod_gpts = np.unique(np.append(idx_mod_gpts_newpos,
        #                                         idx_mod_gpts_oldpos))
        
        # if self.maxgrad is not None or self.mingrad is not None:
        #     for idx in self.idx_subset_points:
        #         vmin,vmax = self.check_gradient(self.points[idx],self.mingrad,
        #                                         self.maxgrad,idx=idx)
        #         if self.parameters[idx]<vmin or self.parameters[idx]>vmax:
        #             return (np.nan,np.nan)
        
        self.points[index] = newxy
        self.update_neighbor_lists(self.points, 'move', self.idx_mod)
        
        return True
        

    def moho_check(self):
        
        moho_depth = self.moho_func(self.points[self.idx_mod,:2])
        vs = myround(self.vsmin + self.vs_range(self.points[self.idx_mod,-1])*self.parameters[self.idx_mod],self.vsprecision)
        if (self.points[self.idx_mod,2] > moho_depth and vs < 4.1 ):
            return False
        if (self.points[self.idx_mod,2] <= moho_depth and vs > 4.1 ):
            return False
        return True
    
    
    def update_neighbor_lists(self,points,action,idx_point):

        if action=='birth':
            idx_mod_gpts = np.zeros(len(self.gridpoints),dtype=bool)
            dists = np.ones(len(self.gridpoints))*np.inf
            # doing the distance calculation only at valid gridpoints (where a ray passes)
            # speeds up calculations.
            dists[self.valid_gpts] = np.sqrt(np.sum((self.gridpoints[self.valid_gpts]-points[-1])**2,axis=1))
            for i in range(self.Nnearest):
                nearest = dists<self.grid_nndists[:,i]
                # if the new point is the new closest neighbor, shift the distance and index arrays by one
                # so that the previous closest point is now the second closest and so on
                self.grid_nndists[nearest,i+1:] = self.grid_nndists[nearest,i:-1]
                self.grid_nnidx[nearest,i+1:] = self.grid_nnidx[nearest,i:-1]
                # now set the new closest neighbor distance and index
                self.grid_nndists[nearest,i] = dists[nearest]
                self.grid_nnidx[nearest,i] = idx_point
                dists[nearest] = np.inf
                idx_mod_gpts[nearest] = True
        elif action=='death':
            # re-calculate the nearest neighbor interpolation, but only for the
            # gridpoints that had the removed nucleus as one of his nearest neighbors
            idx_mod_gpts = np.any(self.grid_nnidx==idx_point,axis=1)
            kdt = KDTree(points)
            nndist,nnidx = kdt.query(self.gridpoints[idx_mod_gpts],k=self.Nnearest)
            if self.Nnearest==1:
                nndist = nndist[None].T
                nnidx = nnidx[None].T
            self.grid_nndists[idx_mod_gpts] = nndist
            self.grid_nnidx[self.grid_nnidx>idx_point] -= 1
            self.grid_nnidx[idx_mod_gpts] = nnidx
        elif action=='move':
            # get all potentially modified gridpoints from the old cell 
            # location and the new cell location
            idx_mod_gpts = np.any(self.grid_nnidx==idx_point,axis=1)
            dists = np.ones(len(self.gridpoints))*np.inf
            dists[self.valid_gpts] = np.sqrt(np.sum((self.gridpoints[self.valid_gpts]-points[idx_point])**2,axis=1))
            for i in range(self.Nnearest):
                idx_new = dists<self.grid_nndists[:,i]
                idx_mod_gpts[idx_new] = True
            # now re-calculate the nearest neighbor interpolation for the 
            # affected gridpoints
            kdt = KDTree(points)
            nndist,nnidx = kdt.query(self.gridpoints[idx_mod_gpts],k=self.Nnearest)
            if self.Nnearest==1:
                nndist = nndist[None].T
                nnidx = nnidx[None].T
            self.grid_nndists[idx_mod_gpts] = nndist
            self.grid_nnidx[idx_mod_gpts] = nnidx
        else:
            raise Exception("action undefined!",action)
            
        self.idx_mod_gpts = idx_mod_gpts

        # kdt = KDTree(points)
        # nndist,nnidx = kdt.query(self.gridpoints,k=self.Nnearest)
        # if self.Nnearest==1:
        #     nndist = nndist[None].T
        #     nnidx = nnidx[None].T
        # if not np.array_equal(nndist,self.grid_nndists) or not np.array_equal(nnidx,self.grid_nnidx):
        #     raise Exception("!")
            
        # idx_mod_test = np.where(np.any(self.grid_nndists!=self.grid_nndists_backup,axis=1))[0]
        # if not np.array_equal(idx_mod_test,np.sort(self.idx_mod_gpts)):
        #     self.a = idx_mod_gpts
        #     self.b = idx_mod_test
        #     raise Exception("!2")
                    
        # self.grid_nndists = self.grid_nndists[:,0]
        # self.grid_nnidx = self.grid_nnidx[:,0]
        
        # if action=='birth':
        #     dists = cdist(self.gridpoints,points[-1:]).flatten()
        #     #dists = np.sqrt(np.sum((self.gridpoints-points[-1])**2,axis=1))
        #     idx_mod_gpts = np.where(dists<self.grid_nndists)[0]
        #     self.grid_nndists[idx_mod_gpts] = dists[idx_mod_gpts]
        #     self.grid_nnidx[idx_mod_gpts] = idx_point
        # elif action=='death':
        #     idx_mod_gpts = np.where(self.grid_nnidx==idx_point)[0]
        #     kdt = KDTree(points)
        #     nndist,nnidx = kdt.query(self.gridpoints[idx_mod_gpts])
        #     self.grid_nndists[idx_mod_gpts] = nndist
        #     self.grid_nnidx[self.grid_nnidx>idx_point] -= 1
        #     self.grid_nnidx[idx_mod_gpts] = nnidx
        # elif action=='move':
        #     idx_old = np.where(self.grid_nnidx==idx_point)[0]
        #     kdt = KDTree(points)
        #     nndist,nnidx = kdt.query(self.gridpoints[idx_old])
        #     self.grid_nndists[idx_old] = nndist
        #     self.grid_nnidx[idx_old] = nnidx
        #     dists = cdist(self.gridpoints,points[idx_point:idx_point+1]).flatten()
        #     #dists = np.sqrt(np.sum((self.gridpoints-points[idx_point])**2,axis=1))
        #     idx_new = np.where(dists<self.grid_nndists)[0]
        #     self.grid_nndists[idx_new] = dists[idx_new]
        #     self.grid_nnidx[idx_new] = idx_point
        #     idx_mod_gpts = np.append(
        #         idx_old[self.grid_nnidx[idx_old]!=idx_point],idx_new)
        # else:
        #     raise Exception("action undefined!",action)
            
        # self.grid_nndists = self.grid_nndists[None].T
        # self.grid_nnidx = self.grid_nnidx[None].T
            
        # return idx_mod_gpts
            
            
        
    # def get_modified_gridpoints(self,points,action,idx_point):
        
    #     if False and self.points_backup is not None:
    #         kdt = KDTree(self.points_backup)
    #         nndist,nnidx = kdt.query(self.gridpoints)
    #         if not np.array_equal(self.grid_nnidx,nnidx):
    #             raise Exception("01",action)
    #         if not np.array_equal(self.grid_nndists,nndist):
    #             raise Exception("02",action)
    #         print("0 check okay",action)
        
    #     if action=='update':
    #         idx_mod_gpts = np.where(self.grid_nnidx==idx_point)[0]
        
    #     elif action=='birth':
    #         dists = np.sqrt(np.sum((self.gridpoints-points[-1])**2,axis=1))
    #         idx_mod_gpts = np.where(dists<self.grid_nndists)[0]
            
    #         self.grid_nndists[idx_mod_gpts] = dists[idx_mod_gpts]
    #         self.grid_nnidx[idx_mod_gpts] = idx_point
            
    #     elif action=='death':
    #         idx_mod_gpts = np.where(self.grid_nnidx==idx_point)[0]
    #         kdt = KDTree(points)
    #         nndist,nnidx = kdt.query(self.gridpoints[idx_mod_gpts])
            
    #         self.grid_nndists[idx_mod_gpts] = nndist
    #         self.grid_nnidx[self.grid_nnidx>idx_point] -= 1
    #         self.grid_nnidx[idx_mod_gpts] = nnidx
            
    #     elif action=='move':
    #         idx_old = np.where(self.grid_nnidx==idx_point)[0]
    #         kdt = KDTree(points)
    #         nndist,nnidx = kdt.query(self.gridpoints[idx_old])
    #         self.grid_nndists[idx_old] = nndist
    #         self.grid_nnidx[idx_old] = nnidx
    #         dists = np.sqrt(np.sum((self.gridpoints-points[idx_point])**2,axis=1))
    #         idx_new = np.where(dists<self.grid_nndists)[0]
    #         self.grid_nndists[idx_new] = dists[idx_new]
    #         self.grid_nnidx[idx_new] = idx_point
    #         idx_mod_gpts = np.append(
    #             idx_old[self.grid_nnidx[idx_old]!=idx_point],idx_new)
    #     else:
    #         raise Exception("action undefined!")
            
    #     if False:
    #         kdt = KDTree(points)
    #         nndist,nnidx = kdt.query(self.gridpoints)
    #         if not np.array_equal(self.grid_nnidx,nnidx):
    #             raise Exception("1",action)
    #         if not np.array_equal(self.grid_nndists,nndist):
    #             raise Exception("2",action)
    #         #print("check okay",action)
    #     self.idx_mod_gpts_new = idx_mod_gpts
        
    #     return idx_mod_gpts
    

    def get_neighbors(self,ind,points=None):
       
        if points is None:
            vertex_neighbor_vertices = self.vertex_neighbors
        else:
            tri = Delaunay(points)
            if len(tri.coplanar)>0:
                print("Warning: coplanar points!")
            vertex_neighbor_vertices = list(tri.vertex_neighbor_vertices)
            
        intptr,neighbor_indices = vertex_neighbor_vertices
        # it's important to return copies here, otherwise changes in one of the
        # returned objects could cause a change in the other returned object.
        return (vertex_neighbor_vertices.copy(),
                neighbor_indices[intptr[ind]:intptr[ind+1]].copy())
  

    # def get_neighbors(self,points,ind):
        

    #     # # STANDARD PROCEDURE TO GET NEIGHBORS (but slower since the
    #     # delaunay triangulation has to be recalculated)
    #     def test1(points,ind):
    #         tri = Delaunay(points)
    #         if len(tri.coplanar)>0:
    #             print("Warning: coplanar points!")
    #         intptr,neighbor_indices = tri.vertex_neighbor_vertices
    #         return neighbor_indices[intptr[ind]:intptr[ind+1]]
        
    #     # Get the distance to all other points
    #     # brute force seems to be faster than a KDTree
    #     nnidx = np.sum((points-points[ind])**2,axis=1).argsort()
        
    #     # get the nearest neighbors in all 6 possible directions. This should
    #     # include the neighbors we are looking for. 20 in each direction should
    #     # be enough
    #     nnidx = np.unique(np.hstack((nnidx[points[nnidx,2]>points[ind,2]][:100],
    #                                  nnidx[points[nnidx,2]<points[ind,2]][:100],
    #                                  nnidx[points[nnidx,0]>points[ind,0]][:100],
    #                                  nnidx[points[nnidx,0]<points[ind,0]][:100],
    #                                  nnidx[points[nnidx,1]>points[ind,1]][:100],
    #                                  nnidx[points[nnidx,1]<points[ind,1]][:100])))
                                    
    #     # include also the points at the edges (we use a hull to find
    #     # these points)
    #     hull = ConvexHull(points)
    #     nnidx = np.unique(np.append(hull.vertices[hull.vertices!=ind],nnidx))
    #     nnidx = np.append(ind,nnidx)

    #     # using only the subset of points, the Delaunay triangulation is much
    #     # faster and the result should be the same.        
    #     tri = Delaunay(points[nnidx])
    #     if len(tri.coplanar)>0:
    #         print("Warning: coplanar points!")
    #     intptr,neighbor_indices = tri.vertex_neighbor_vertices
    #     neighbors = neighbor_indices[intptr[0]:intptr[1]]
    #     # translate back to global indices
    #     neighbors = nnidx[neighbors]
        
    #     if not np.array_equal(np.sort(test1(points,ind)),np.sort(neighbors)):
    #         raise Exception(ind)
          
    #     return neighbors
           
    #     """
    #     plt.figure()
    #     plt.scatter(X.flatten(),Y.flatten(),c=self.ind)
    #     plt.plot(self.gridpoints[grid_neighbor,0],self.gridpoints[grid_neighbor,1],'ro')
    #     plt.plot(points[ind_neighbors,0],points[ind_neighbors,1],'ko')
    #     plt.show()
    #     """
        
    def get_prior_proposal_ratio(self,delayed=False):
        
        # this function returns the log prior ratio plus the log proposal ratio
        # which is needed to calculate the acceptance probability
        if 'update' in self.action or self.action == 'move':
            if not delayed:
                return 0
            else:
                # this is log(q1(m'|m'')/q1(m'|m)) from eq. 3.39 of the thesis of T.Bodin
                return 1./(2*self.propdist_std1**2)*(np.sum(self.dx1**2)-np.sum(self.dx**2))
        if delayed:
            raise Exception("delayed rejection for transdimensional changes (birth/death) not implemented.")
          
        if self.propstd_dimchange == 'uniform':
            return 0
        elif self.prop_vpvs is None:
            prior_prop_ratio = np.log(self.p_range) + self.tn_p.logpdf(self.prop_p)
        else:
            prior_prop_ratio = (np.log(self.p_range*self.vpvs_range) + 
                                self.tn_p.logpdf(self.prop_p) +
                                self.tn_vpvs.logpdf(self.prop_vpvs) )
        
        if self.action == 'birth':
            return -prior_prop_ratio
        elif self.action == 'death':
            return prior_prop_ratio
        else:
            raise Exception("unknown update type")
            
          
        # elif self.action == 'birth':
        #     if self.propstd_dimchange == 'uniform':
        #         # if we draw from a uniform prior, everything cancels out
        #         return 0
        #     elif self.prop_dvpvs is None:  # if vpvs is fixed
        #         # see for example equation A.34 of the PhD thesis of Thomas Bodin
        #         return (np.log(self.propstd_dimchange*np.sqrt(2.*np.pi) / self.p_range) +
        #                 self.prop_dp**2 / (2*self.propstd_dimchange**2))
        #     else:
        #         return (np.log(self.propstd_dimchange**2*2.*np.pi / 
        #                         (self.p_range * self.vpvs_range)) +
        #                 (self.prop_dp**2 + self.prop_dvpvs**2) / (2*self.propstd_dimchange**2))
        # elif self.action == 'death':
        #     if self.propstd_dimchange == 'uniform':
        #         return 0
        #     elif self.prop_dvpvs is None: # if vpvs is fixed
        #         return (np.log((self.p_range)/(self.propstd_dimchange*np.sqrt(2.*np.pi))) -
        #                 self.prop_dp**2 / (2*self.propstd_dimchange**2))
        #     else:
        #         return (np.log((self.p_range*self.vpvs_range) / 
        #                         (self.propstd_dimchange**2*2.*np.pi)) -
        #                 (self.prop_dp**2 + self.prop_dvpvs)**2 / (2*self.propstd_dimchange**2))
    
    
    # def get_model_old(self,points=None,vs=None,vpvs=None,
    #               psi2amp=None,psi2=None,anisotropic=False):
        
    #     if points is None:
    #         points = self.points
    #     if vs is None:
    #         vs = self.parameters
    #     if vpvs is None:
    #         vpvs = self.vpvs       
            
    #     func = NearestNDInterpolator(points,np.arange(len(vs)))
    #     field = func(self.gridpoints)
        
    #     velfield = vs[field]
    #     if callable(vpvs):
    #         vpvsfield = vpvs(vs)[field]
    #     else:
    #         vpvsfield = vpvs[field]
        
    #     if self.fixed_moho:
    #         idx_mod_crust = np.where(velfield[self.crustidx] > 4.0)[0]
    #         velfield[self.crustidx][idx_mod_crust] = 3.9
    #         idx_mod_mantle = np.where(velfield[self.mantleidx] < 4.1)[0]
    #         velfield[self.mantleidx][idx_mod_mantle] = 4.1
        
    #     if anisotropic:
    #         if psi2amp is None:
    #             psi2amp = self.psi2amp 
    #         if psi2 is None:
    #             psi2 = self.psi2
            
    #         psi2amp = psi2amp[field]
    #         psi2 = psi2[field]
    #         return (velfield,vpvsfield,psi2amp,psi2)
        
    #     else:
    #         return velfield,vpvsfield
    
    
    def get_model(self,points=None,params=None,vpvs=None,
                  psi2amp=None,psi2=None,anisotropic=False,recalculate=False):
        
        if anisotropic:
            print("currently no anisotropy implemented.")
        if points is not None or recalculate:
            if points is None:
                points = self.points
                z = points[:,-1]
            kdt = KDTree(points)
            nndist,nnidx = kdt.query(self.gridpoints,k=self.Nnearest)
            idx_mod_gpts = ()
        else:
            z = self.points[:,-1]
            idx_mod_gpts = self.idx_mod_gpts
            if len(idx_mod_gpts) > 0:
                if len(idx_mod_gpts)!=len(self.gridpoints):
                    raise Exception("should be boolean of same size as gridpoints") 
                idx_mod_gpts *= self.valid_gpts
            if self.Nnearest==1:
                # for performance reasons, it is very important that the output
                # array has shape (Ngridpoints,) and not (Ngridpoints,1)
                # the latter is MUCH slower. Therefore take index 0 (i.e. first
                # column of single column vector).
                nnidx = self.grid_nnidx[idx_mod_gpts][:,0]
            else:
                nndist = self.grid_nndists[idx_mod_gpts]
                nnidx = self.grid_nnidx[idx_mod_gpts]               

        if params is None:
            params = self.parameters
        if vpvs is None:
            vpvs = self.vpvs
            
        # at this point, we translate the parameters (defined between 0 and 1)
        # to a vs value that is between vsmin(corresponds to 0) and vsmax(corresponds to 1)
        vs = myround(self.vsmin(z) + self.vs_range(z)*params,self.vsprecision)
        if callable(vpvs):
            vpvs_cells = myround(vpvs(vs),self.vsprecision)
        else:
            vpvs_cells = vpvs
        # the velfield contains the weighted contributions of the self.Nnearest
        # neigbor points. The weighting can also be defined differently    
        if self.Nnearest>1:
            weights = 1./(1+nndist)
            weights /= np.sum(weights,axis=1)[None].T
            self.vsfield[idx_mod_gpts] = myround(np.sum(vs[nnidx]*weights,axis=1),self.vsprecision)
            self.vpvsfield[idx_mod_gpts] = myround(np.sum(vpvs_cells[nnidx]*weights,axis=1),self.vsprecision)
        else:
            self.vsfield[idx_mod_gpts] = vs[nnidx]
            self.vpvsfield[idx_mod_gpts] = vpvs_cells[nnidx]
            
        # gridpoints that are not in self.valid_gpts (those without ray coverage)
        # are ignored and the model is not updated in these regions. For
        # consistency, make sure that these model regions are always kept at
        # the mean value
        if len(idx_mod_gpts) == 0:
            zinvalid = self.gridpoints[~self.valid_gpts,2]
            self.vsfield[~self.valid_gpts] = myround(self.mean_vs(zinvalid),self.vsprecision)
            self.vpvsfield[~self.valid_gpts] = 1.75
            
        if self.smoothmodel:
            self.vsfield = myround(self.smoothing_matrix*self.vsfield,self.vsprecision)
            self.vpvsfield = myround(self.smoothing_matrix*self.vpvsfield,self.vsprecision)
          
        if self.fixed_moho:
            idx_mod_crust = np.where(self.vsfield[self.crustidx] > 4.1)[0]
            self.vsfield[self.crustidx][idx_mod_crust] = 4.0
            idx_mod_mantle = np.where(self.vsfield[self.mantleidx] < 4.1)[0]
            self.vsfield[self.mantleidx][idx_mod_mantle] = 4.1
            
        if self.crust_idx is not None:
            if len(idx_mod_gpts) > 0:
                idx_mod_crust = idx_mod_gpts*self.crust_idx
                subidx = idx_mod_gpts[self.crust_idx]
            else:
                subidx = ()
                idx_mod_crust = self.crust_idx
            if False:
                # allow up to 0.4km/s variations from the input crustal model
                dvmax = 0.4
                dv_crust = myround(-dvmax + 2*dvmax*params, self.vsprecision)
                dv_crust = dv_crust[self.grid_nnidx[idx_mod_crust,0]]
                if len(subidx)==0: # crust indices outside valid regions are not modified
                    dv_crust[~self.valid_gpts[self.crust_idx]] = 0.
            else:
                dv_crust = 0.
            self.vsfield[idx_mod_crust] = self.crust_vs[subidx]+dv_crust
            self.vpvsfield[idx_mod_crust] = self.crust_vpvs[subidx]

        # idx_mod_gpts often contains indices that have not changed
        # its velocity. we can therefore reduce the number of indices by the
        # following test. Commented, because it doesn't seem to yield a speedup
        # if idx_mod_gpts != ():
        #     self.idx_mod_gpts = (self.vsfield!=self.vsfield_backup)+(self.vpvsfield!=self.vpvsfield_backup)
        
        return True
            
    # def update_model(self,fields=None,anisotropic=False):
        
    #     if anisotropic:
    #         velfield, vpvs, psi2amp, psi2 = fields
    #     else:
    #         velfield, vpvs = fields
        
    #     velfield_cp = velfield.copy()
    #     vpvs_cp = vpvs.copy()
    #     if anisotropic:
    #         psi2amp_cp = psi2amp.copy()
    #         psi2_cp = psi2.copy()
            
            
    #     if len(self.idx_mod_gpts) == 0:
    #         if anisotropic:   
    #             return velfield_cp, vpvs_cp, psi2amp_cp, psi2_cp
    #         else:
    #             return velfield_cp, vpvs_cp
        
    #     if self.action=='velocity_update':
    #         velfield_cp[self.idx_mod_gpts] = self.parameters[self.idx_mod]
            
    #     elif self.action=='vpvs_update':
    #         vpvs_cp[self.idx_mod_gpts] = self.vpvs[self.idx_mod]
            
    #     elif self.action=='birth':
    #         velfield_cp[self.idx_mod_gpts] = self.parameters[-1]
    #         if not callable(self.vpvs):
    #             vpvs_cp[self.idx_mod_gpts] = self.vpvs[-1]
    #         if anisotropic:
    #             psi2amp_cp[self.idx_mod_gpts] = self.psi2amp[-1]
    #             psi2_cp[self.idx_mod_gpts] = self.psi2[-1]
            
    #     elif self.action=='death' or self.action=='move':
    #         field = self.grid_nnidx[self.idx_mod_gpts]
    #         ## interpolate new velocities only in the area affected by the change
    #         #func = NearestNDInterpolator(self.points[self.idx_subset_points],
    #         #                             self.idx_subset_points)
    #         #field = func(self.gridpoints[self.idx_mod_gpts])
    #         velfield_cp[self.idx_mod_gpts] = self.parameters[field]
    #         if not callable(self.vpvs):
    #             vpvs_cp[self.idx_mod_gpts] = self.vpvs[field]
    #         if anisotropic:
    #             psi2amp_cp[self.idx_mod_gpts] = self.psi2amp[field]
    #             psi2_cp[self.idx_mod_gpts] = self.psi2[field]
            
    #     elif self.action == 'psi2amp_update':
    #         psi2amp_cp[self.idx_mod_gpts] = self.psi2amp[self.idx_mod]

    #     elif self.action == 'psi2_update':
    #         psi2_cp[self.idx_mod_gpts] = self.psi2[self.idx_mod]
        
    #     elif self.action == 'anisotropy_birth_death':
    #         psi2amp_cp[self.idx_mod_gpts] = self.psi2amp[self.idx_mod]
    #         psi2_cp[self.idx_mod_gpts] = self.psi2[self.idx_mod]
          
    #     if self.fixed_moho:
    #         idx_mod_crust = np.where(velfield_cp[self.crustidx] > 4.0)[0]
    #         velfield_cp[self.crustidx][idx_mod_crust] = 3.9
    #         idx_mod_mantle = np.where(velfield_cp[self.mantleidx] < 4.1)[0]
    #         velfield_cp[self.mantleidx][idx_mod_mantle] = 4.1
            
    #     idx_modified = np.where(velfield!=velfield_cp)[0]
    #     if len(idx_modified)>0 and not np.array_equal(np.sort(idx_modified),np.sort(self.idx_mod_gpts_new)):
    #         #raise Exception("idx mod not good",self.action,velfield[idx_modified],
    #         #                velfield_cp[idx_modified])
    #         print("idx mod not good",len(idx_modified),len(self.idx_mod_gpts))
    #         if not np.array_equal(velfield_cp,self.parameters[self.grid_nnidx]):
    #             print("bad velocity field!",self.action)
          
    #     if anisotropic:
    #         return velfield_cp, vpvs_cp, psi2amp_cp, psi2_cp
    #     else:
    #         return velfield_cp, vpvs_cp
                      
        
    # def get_gpts_dictionary(self,points=None,gridpoints=None):
        
    #     gpts_dict = {}
        
    #     if points is None:
    #         points = self.points
    #     if gridpoints is None:
    #         gridpoints = self.gridpoints
        
    #     kdt = KDTree(points)
    #     point_dist, point_regions = kdt.query(gridpoints)
    #     for i in range(len(points)):
    #         gpts_dict[i] = np.isin(point_regions,i).nonzero()[0]

    #     return gpts_dict
    
    
    # def check_gpts_dictionary(self,action):
        
    #     test_gpts_dictionary = self.get_gpts_dictionary(self.points,self.gridpoints)
        
    #     for i in test_gpts_dictionary:
    #         if not np.array_equal(np.sort(self.gpts_dict[i]), test_gpts_dictionary[i]):
    #             raise Exception(f"gpts dict not right after {action} operation")
        
        
    # def update_gpts_dict(self,selfcheck=False):
        
    #     if self.action is None or self.idx_mod_gpts is None:
    #         raise Exception("cannot update gpts dictionary!")
        
    #     if self.action=='birth':
    #         if (self.subidx_gridpts_new_cell is None or
    #             self.subgrid_lengths is None or
    #             self.idx_subset_points is None):
    #             raise Exception("cannot update gpts dictionary!")

    #         subidx_gridpts_new_cell = np.split(self.subidx_gridpts_new_cell,
    #                                            np.cumsum(self.subgrid_lengths))
    #         for j,idx in enumerate(self.idx_subset_points):
    #             self.gpts_dict[idx] = self.gpts_dict[idx][~subidx_gridpts_new_cell[j]]

    #         self.gpts_dict[len(self.points)-1] = self.idx_mod_gpts
            
    #     elif self.action=='death':
    #         if (self.idx_mod is None or self.idx_subset_points is None):
    #             raise Exception("cannot update gpts dictionary!")
    #         # correct the indices of the gpts_idx dictionary
    #         for idx in range(self.idx_mod,len(self.points)):
    #             self.gpts_dict[idx] = self.gpts_dict[idx+1]
    #         del self.gpts_dict[len(self.points)]
                
    #         # add the gridpoints that were in the death cell to the
    #         # neighboring cells
    #         subset_voronoi_kdt = KDTree(self.points[self.idx_subset_points])            
    #         update_point_dist, update_point_regions = subset_voronoi_kdt.query(self.gridpoints[self.idx_mod_gpts],eps=0)
    #         update_point_regions = self.idx_subset_points[update_point_regions]
                  
    #         for idx in self.idx_subset_points:
    #             self.gpts_dict[idx] = np.hstack((self.gpts_dict[idx],self.idx_mod_gpts[update_point_regions==idx]))
      
    #     elif self.action=='move':
    #         if (self.idx_subset_points is None):
    #             raise Exception("cannot update gpts dictionary!")
    #         idx_modified_gridpoints_all = np.unique(np.hstack((itemgetter(*self.idx_subset_points)(self.gpts_dict))))
    #         subset_voronoi_kdt = KDTree(self.points[self.idx_subset_points])
    #         update_point_dist, update_point_regions = subset_voronoi_kdt.query(self.gridpoints[idx_modified_gridpoints_all],eps=0)
    #         # relate the subset indices to the global point indices
    #         update_point_regions = self.idx_subset_points[update_point_regions]
    #         for idx in self.idx_subset_points:
    #             self.gpts_dict[idx] = idx_modified_gridpoints_all[update_point_regions==idx]

    #     if selfcheck:
    #         kdt = KDTree(self.points)
    #         point_dist, point_regions = kdt.query(self.gridpoints)
    #         for i in range(len(self.points)):
    #             if not np.array_equal(np.sort(self.gpts_dict[i]),np.isin(point_regions,i).nonzero()[0]):
    #                 raise Exception("gpts dict not right after",self.action,"operation")
        
        
    def create_smoothing_matrix(self):
        
        std = 25.
        ntrunc = 2
        truncated_normal = truncnorm(-ntrunc,ntrunc,0,std)
        self.smoothing_matrix = lil_matrix((len(self.gridpoints),len(self.gridpoints)))
        for i in range(len(self.gridpoints)):
            x,y,z = self.gridpoints[i]
            subidx = np.where((self.gridpoints[:,0]<=x+ntrunc*std)*(self.gridpoints[:,0]>=x-ntrunc*std)*
                              (self.gridpoints[:,1]<=y+ntrunc*std)*(self.gridpoints[:,1]>=y-ntrunc*std)*
                              (self.gridpoints[:,2]<=z+ntrunc*std)*(self.gridpoints[:,2]>=z-ntrunc*std))[0]
            cdist = distance.cdist(self.gridpoints[subidx],self.gridpoints[i:i+1]).flatten()
            weights = truncated_normal.pdf(cdist)
            weights /= np.sum(weights)
            self.smoothing_matrix[i,subidx] = weights
            if i%10000==0:
                print(i,"/",len(self.gridpoints))
        self.smoothing_matrix = self.smoothing_matrix.tocsr()
        

    def backup_mod(self):
        
        self.points_backup = self.points.copy()
        self.parameters_backup = self.parameters.copy()
        #self.vertex_neighbors_backup = self.vertex_neighbors.copy()
        if not 'update' in self.action:
            self.grid_nnidx_backup = self.grid_nnidx.copy()
            self.grid_nndists_backup = self.grid_nndists.copy()
        self.vsfield_backup = self.vsfield.copy()
        self.vpvsfield_backup = self.vpvsfield.copy()
        if not callable(self.vpvs):
            self.vpvs_backup = self.vpvs.copy()
        if self.psi2amp is not None:
            self.psi2amp_backup = self.psi2amp.copy()
        if self.psi2 is not None:
            self.psi2_backup = self.psi2.copy()
            
        
    def reject_mod(self):

        if self.points_backup is not None:
            self.points = self.points_backup
        if self.parameters_backup is not None:
            self.parameters = self.parameters_backup
        if self.vpvs_backup is not None:
            self.vpvs = self.vpvs_backup
        if self.psi2amp_backup is not None:
            self.psi2amp = self.psi2amp_backup
        if self.psi2_backup is not None:
            self.psi2 = self.psi2_backup
        #if self.vertex_neighbors_backup is not None:
        #    self.vertex_neighbors = self.vertex_neighbors_backup
        if self.grid_nnidx_backup is not None:
            self.grid_nnidx = self.grid_nnidx_backup
        if self.grid_nndists_backup is not None:
            self.grid_nndists = self.grid_nndists_backup
        if self.vsfield_backup is not None:
            self.vsfield = self.vsfield_backup
        if self.vpvsfield_backup is not None:
            self.vpvsfield = self.vpvsfield_backup
        # keep the value for potential delayed rejection scheme
        self.propdist_std1 = self.propdist_std
        self.dx1 = self.dx
           
        self.reset_variables()

    def accept_mod(self):
                
        #self.update_gpts_dict(selfcheck=selfcheck)
        #if selfcheck:
        #    self.check_gpts_dictionary(self.action)
        #    tri = Delaunay(self.points)
        #    if not np.array_equal(self.vertex_neighbors[1],tri.vertex_neighbor_vertices[1]):
        #        raise Exception("triangulation error")
            
        # tri = Delaunay(self.points)
        # intptr,neighbor_indices = tri.vertex_neighbor_vertices
        # for pnt_idx in range(len(self.points)):
        #     idx_neighbor_points = neighbor_indices[intptr[pnt_idx]:intptr[pnt_idx+1]]
        #     dvs = self.parameters[idx_neighbor_points] - self.parameters[pnt_idx]
        #     dz = (self.inverse_z_scaling(self.points[idx_neighbor_points,2]) - 
        #           self.inverse_z_scaling(self.points[pnt_idx,2]))
            
        #     gradients = dvs[dz!=0.]/dz[dz!=0.]
        #     if (gradients < -0.5).any() or (gradients > 1.0).any():
        #         raise Exception(f"{self.action} needs to be fixed!")
        
        # when accepted, no delayed rejection scheme can be applied anymore
        self.propdist_std1 = self.dx1 = None
        self.reset_variables()
        
    def reset_variables(self):
            
        self.points_backup = None
        self.parameters_backup = None
        self.vpvs_backup = None
        self.vsfield_backup = None
        self.vpvsfield_backup = None
        self.idx_mod_gpts = '' # make sure that it throws an error if nothing is assigned
        #self.subidx_gridpts_new_cell = None
        #self.subgrid_lengths = None
        self.action = None
        #self.idx_mod_gpts = None
        #self.idx_mod = None
        #self.idx_subset_points = None
        self.psi2amp_backup = None
        self.psi2_backup = None
        self.velrange = None
        #self.vertex_neighbors_backup = None
        self.grid_nndists_backup = None
        self.grid_nnidx_backup = None
        #self.prop_dp = None
        self.prop_p = np.nan
        self.tn_p = None
        self.prop_vpvs = np.nan
        self.tn_vpvs = None
        self.propdist_std = None
        self.dx = None

    

    def plot(self,idx_mod_gpts=None,idx_neighbor_points=None):
        
        from scipy.spatial import delaunay_plot_2d
        tri = Delaunay(self.points)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        _ = delaunay_plot_2d(tri,ax=ax)
        ax.plot(self.gridpoints[:,0],self.gridpoints[:,1],'k.')
        if idx_mod_gpts is not None:
            ax.plot(self.gridpoints[idx_mod_gpts,0],self.gridpoints[idx_mod_gpts,1],'rx',zorder=3)
        cbar = ax.scatter(self.points[:,0],self.points[:,1],c=self.parameters,s=40,zorder=3)
        if idx_neighbor_points is not None:
            ax.scatter(self.points[:,0],self.points[:,1],c=self.parameters,s=60,zorder=3)
        plt.colorbar(cbar)
        plt.show()

        
##############################################################################
##############################################################################
##############################################################################

class profiles(object):
    
    def __init__(self,gridpoints,shape,convexhull,velmin,velmax,vpvs,
                 init_no_points):
        
        raise Exception("currently not working, convexhull is now a 3D hull (previously was 2D)")
        
        x = np.unique(gridpoints[:,0])
        y = np.unique(gridpoints[:,1])
        X,Y = np.meshgrid(x,y) # 2d gridpoint field
        self.gridpoints = np.column_stack((X.flatten(),Y.flatten()))
        self.shape = shape
        self.shape2d = shape[:2]
        
        self.minx = np.min(x)
        self.maxx = np.max(x)
        self.miny = np.min(y)
        self.maxy = np.max(y)
        self.z = np.unique(gridpoints[:,2])
        self.minz = self.z[0]
        self.maxz = self.z[-2]
        
        self.velmin = velmin
        self.velmax = velmax
        
        self.vpvs = vpvs
        
        self.convexhull = convexhull
        
        points_x = np.random.uniform(low=self.minx,high=self.maxx,
                                     size=init_no_points)
        points_y = np.random.uniform(low=self.miny,high=self.maxy,
                                     size=init_no_points)
        while (self.convexhull.find_simplex(np.column_stack((points_x,points_y))) < 0.).any():
            outside_idx = np.where(self.convexhull.find_simplex(np.column_stack((points_x,points_y))) < 0.)[0]
            points_x[outside_idx] = np.random.uniform(low=self.minx,
                                                      high=self.maxx,
                                                      size=len(outside_idx))
            points_y[outside_idx] = np.random.uniform(low=self.miny,
                                                      high=self.maxy,
                                                      size=len(outside_idx))
        
        self.points = np.column_stack((points_x,points_y))
        kdt = KDTree(self.points)
        self.grid_nndists,self.grid_nnidx = kdt.query(self.gridpoints)
        
        self.parameters = []
        for i in range(init_no_points):
            self.parameters.append(self.get_new_profile())
        
        self.points_backup = None
        self.parameters_backup = None
        self.vpvs_backup = None
        self.subidx_gridpts_new_cell = None
        self.subgrid_lengths = None        
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod_gpts2d = None
        self.idx_mod = None
        self.idx_subset_points = None
        self.psi2amp = None
        self.psi2 = None
        self.psi2amp_backup = None
        self.psi2_backup = None                        
        self.propstd_dimchange = 'uniform' # will be adapted when running
        
        
    def get_new_profile(self):
        
        z = np.sort(np.random.choice(self.z,8,replace=False))
        vs = np.sort(np.random.uniform(self.velmin(z),self.velmax(z)))
        if callable(self.vpvs):
            vpvs = self.vpvs(vs)
        
        return np.column_stack((z,vs,vpvs))
        
    def get_mod_point_coords(self,idx=None):
        
        if idx is not None:
            return self.points[idx],0
        
        if self.action == 'death':
            return self.points_backup[self.idx_mod],0
        else:
            return self.points[self.idx_mod],0

    def parameter_update(self,idx,dp,backup=True):
        
        if backup:
            self.parameters_backup = deepcopy(self.parameters)
            
        self.action='parameter_update'
        self.idx_mod = idx
            
        param = np.random.choice(["depth","vs"])
        depth_idx = np.random.randint(0,len(self.parameters[self.idx_mod]))
        
        if param=="vs":
            self.parameters[self.idx_mod][depth_idx,1] += dp
            vmin = self.velmin(self.parameters[self.idx_mod][depth_idx,0])
            vmax = self.velmax(self.parameters[self.idx_mod][depth_idx,0])
            if (self.parameters[self.idx_mod][depth_idx,1]<vmin or 
                self.parameters[self.idx_mod][depth_idx,1]>vmax ):
                return False
        else:
            if depth_idx==0:
                zmin=0
            else:
                zmin = self.parameters[self.idx_mod][depth_idx-1,0]
            if depth_idx==len(self.parameters[self.idx_mod])-1:
                zmax=self.z[-1]
            else:
                zmax = self.parameters[self.idx_mod][depth_idx+1,0]
            self.parameters[self.idx_mod][depth_idx,0] = np.random.uniform(zmin,zmax)
         
        return True
    
    # def vpvs_ratio_update(self,idx,dvpvs,backup=True):
        
    #     if backup:
    #         self.vpvs_backup = self.vpvs.copy()
            
    #     self.action='vpvs_update'
    #     self.idx_mod = idx
        
    #     self.vpvs[idx] += dvpvs
        
    #     self.idx_mod_gpts = self.gpts_dict[idx]


    def add_point(self,anisotropic=False,birth_prop='uniform',backup=True):
        
        if backup:
            self.backup_mod()
            
        prop_point_x = np.random.uniform(self.minx,self.maxx)
        prop_point_y = np.random.uniform(self.miny,self.maxy)
        while self.convexhull.find_simplex(np.array([prop_point_x,prop_point_y])) < 0:
            prop_point_x = np.random.uniform(self.minx,self.maxx)
            prop_point_y = np.random.uniform(self.miny,self.maxy)
        point = np.hstack((prop_point_x,prop_point_y))
            
        self.action='birth'
        self.idx_mod = len(self.parameters)
        
        self.points = np.vstack((self.points,point))
        
        # getting all neighbours of new point
        idx_mod_gpts = self.get_modified_gridpoints(
            self.points,'birth',self.idx_mod)
                
        if birth_prop=='uniform':
            parameters_birth = self.get_new_profile()
        else:
            birth_params = np.zeros_like(self.parameters[0])
            for idx_neighbor in self.idx_subset_points:
                birth_params += self.parameters[idx_neighbor]
            birth_params /= len(self.idx_subset_points)
            raise Exception("not implemented yet")
            #if vs_birth > self.velmax or vs_birth < self.velmin:
            #    return False
            
        self.parameters.append(parameters_birth)

        # for the prior_proposal_ratio calculation:
        self.propstd_dimchange = birth_prop
        
        return True
        
        
        
    def remove_point(self,anisotropic=False,backup=True):
        
        if backup:
            self.backup_mod()
            
        self.action='death'
        
        if anisotropic:
            # choose only points without anisotropy
            ind_pnts = np.where(self.psi2amp == 0.)[0]
            if len(ind_pnts) > 0:
                self.idx_mod = np.random.choice(ind_pnts)
            else:
                return False
        else:
            # choose point to remove randomly
            self.idx_mod = np.random.randint(0,len(self.points))
        
        self.points = np.delete(self.points,self.idx_mod,axis=0)
        parameters_removed = self.parameters.pop(self.idx_mod)

        idx_mod_gpts = self.get_modified_gridpoints(
            self.points,'death',self.idx_mod)
                
        return True
                    
            
    def move_point(self,propmovestd,index=None,backup=True):

        if backup:
            self.backup_mod()
        
        self.action = 'move'
        
        if index is None:
            index = np.random.randint(0,len(self.points))
        self.idx_mod = index

        oldxy = self.points[index]
        
        dx = np.random.normal(loc=0.0,scale=propmovestd,size=2)
        newxy = oldxy + dx
        
        if (newxy[0]>self.maxx or newxy[0]<self.minx or
            newxy[1]>self.maxy or newxy[1]<self.miny or
            self.convexhull.find_simplex(newxy) < 0 ):
            return False

        self.points[index] = newxy
        self.idx_mod_gpts = self.get_modified_gridpoints(
            self.points, 'move', self.idx_mod)
        
        return True
        

    # def get_neighbors(self,points,ind):
            
    #     # FASTER FOR MANY POINTS (>500 points)
    #     # brute force seems to be faster than a KDTree
    #     nndist = np.sum((points-points[ind])**2,axis=1)
    #     # 100 nearest neighbors of new point. should include enough neighbors
    #     # so that the Delaunay triangulation in the area is the same as if all 
    #     # points were included
    #     nnidx = nndist.argsort()[1:100]
        
    #     # include also the points at the edges (we use a hull to find
    #     # these points)
    #     hull = ConvexHull(points)
    #     nnidx = np.unique(np.append(hull.vertices[hull.vertices!=ind],nnidx))
    #     nnidx = np.append(ind,nnidx)
        
    #     tri = Delaunay(points[nnidx])
    #     if len(tri.coplanar)>0:
    #         print("Warning: coplanar points!")
    #     intptr,neighbor_indices = tri.vertex_neighbor_vertices
    #     neighbors = neighbor_indices[intptr[0]:intptr[1]]
    #     # translate back to global indices
    #     neighbors = nnidx[neighbors]
          
    #     return neighbors
           
        """
        plt.figure()
        plt.scatter(X.flatten(),Y.flatten(),c=self.ind)
        plt.plot(self.gridpoints[grid_neighbor,0],self.gridpoints[grid_neighbor,1],'ro')
        plt.plot(points[ind_neighbors,0],points[ind_neighbors,1],'ko')
        plt.show()
        """
    def get_modified_gridpoints(self,points,action,idx_point):
              
        if action=='update':
            idx_mod_gpts = np.where(self.grid_nnidx2d==idx_point)[0]
        elif action=='birth':
            dists = distance.cdist(self.gridpoints,points[-1:]).flatten()
            idx_mod_gpts = np.where(dists<self.grid_nndists2d)[0]
            self.grid_nndists2d[idx_mod_gpts] = dists[idx_mod_gpts]
            self.grid_nnidx2d[idx_mod_gpts] = idx_point
            indices = np.unravel_index(self.grid_nnidx2d[idx_mod_gpts],self.shape2d)
            fu = interp1d(parameters[i][:,0],parameters[i][:,1],kind='nearest',
                          bounds_error=False,fill_value='extrapolate')
        elif action=='death':
            idx_mod_gpts = np.where(self.grid_nnidx==idx_point)[0]
            kdt = KDTree(points)
            nndist,nnidx = kdt.query(self.gridpoints[idx_mod_gpts])
            self.grid_nndists[idx_mod_gpts] = nndist
            self.grid_nnidx[self.grid_nnidx>idx_point] -= 1
            self.grid_nnidx[idx_mod_gpts] = nnidx
        elif action=='move':
            idx_old = np.where(self.grid_nnidx==idx_point)[0]
            kdt = KDTree(points)
            nndist,nnidx = kdt.query(self.gridpoints[idx_old])
            self.grid_nndists[idx_old] = nndist
            self.grid_nnidx[idx_old] = nnidx
            dists = distance.cdist(self.gridpoints,points[idx_point:idx_point+1]).flatten()
            #dists = np.sqrt(np.sum((self.gridpoints-points[idx_point])**2,axis=1))
            idx_new = np.where(dists<self.grid_nndists)[0]
            self.grid_nndists[idx_new] = dists[idx_new]
            self.grid_nnidx[idx_new] = idx_point
            idx_mod_gpts = np.append(
                idx_old[self.grid_nnidx[idx_old]!=idx_point],idx_new)
        else:
            raise Exception("action undefined!",action)
            
        return idx_mod_gpts
    


    def get_prior_proposal_ratio(self,delayed=False):
        
        # this function returns the prior ratio times the proposal ratio which
        # is needed to calculate the acceptance probability       
        if 'update' in self.action or self.action == 'move':
            if not delayed:
                return 0
            else:
                # this is log(q1(m'|m'')/q1(m'|m)) from eq. 3.39 of the thesis of T.Bodin
                # notice that it's both q1, meaning that most terms cancel out, only exp terms remain
                return (np.sum(self.dx1**2)-np.sum(self.dx**2))/(2*self.propdist_std1**2)
        if delayed:
            raise Exception("delayed rejection for transdimensional changes (birth/death) not implemented.")
          
        if self.propstd_dimchange == 'uniform':
            return 0
        elif self.prop_vpvs is None:
            prior_prop_ratio = np.log(self.p_range) + self.tn_p.logpdf(self.prop_p)
        else:
            prior_prop_ratio = (np.log(self.p_range*self.vpvs_range) + 
                                self.tn_p.logpdf(self.prop_p) +
                                self.tn_vpvs.logpdf(self.prop_vpvs) )
        
        if self.action == 'birth':
            return -prior_prop_ratio
        elif self.action == 'death':
            return prior_prop_ratio
        else:
            raise Exception("unknown update type")
            
            
    # def get_prior_proposal_ratio(self):
        
    #     # this function returns the prior ratio times the proposal ratio which
    #     # is needed to calculate the acceptance probability
        
    #     if self.action == 'update' or self.action == 'move':
    #         # currently this is included in the main script (rjtransdim3d.py)
    #         # because of the delayed rejection scheme.
    #         raise Exception("not implemented")
    #     elif self.action == 'birth':
    #         if self.propstd_dimchange == 'uniform':
    #             # if we draw from a uniform prior, everything cancels out
    #             return 1
    #         else:
    #             # see for example equation A.34 of the PhD thesis of Thomas Bodin
    #             return ((self.propstd_dimchange*np.sqrt(2.*np.pi) / 
    #                      self.velrange) *  
    #                     np.exp(self.prop_dv**2 / (2*self.propstd_dimchange**2)))
    #     elif self.action == 'death':
    #         if self.propstd_dimchange == 'uniform':
    #             return 1
    #         else:
    #             return ((self.velrange/(self.propstd_dimchange*np.sqrt(2.*np.pi))) /
    #                     np.exp(self.prop_dv**2 / (2*self.propstd_dimchange**2)))
    #     else:
    #         raise Exception("unknown update type")

    def get_model(self,points=None,params=None,vpvs=None,
                  psi2amp=None,psi2=None,anisotropic=False,recalculate=False):
        
        if anisotropic:
            print("currently no anisotropy implemented.")
        if points is not None or recalculate:
            if points is None:
                points = self.points
                z = points[:,-1]
            kdt = KDTree(points)
            nndist,nnidx = kdt.query(self.gridpoints)
        else:
            z = self.points[:,-1]
            nnidx = self.grid_nnidx
        if params is None:
            params = self.parameters
        if vpvs is None:
            vpvs = self.vpvs
            
        # at this point, we translate the parameters (defined between 0 and 1)
        # to a vs value that is between vsmin(corresponds to 0) and vsmax(corresponds to 1)
        vs = myround(self.vsmin(z) + self.vs_range(z)*params,self.vsprecision)
        velfield = vs[nnidx]
        if callable(vpvs):
            vpvsfield = myround(vpvs(vs)[nnidx],self.vsprecision)
        else:
            vpvsfield = vpvs[nnidx]
            if np.isnan(self.vpvs).any():
                raise Exception("nan in para.vpvs after",self.action)
        if np.isnan(vpvsfield).any():
            raise Exception("nan in vpvsfield after",self.action)
          
        if self.fixed_moho:
            idx_mod_crust = np.where(velfield[self.crustidx] > 4.1)[0]
            velfield[self.crustidx][idx_mod_crust] = 4.0
            idx_mod_mantle = np.where(velfield[self.mantleidx] < 4.1)[0]
            velfield[self.mantleidx][idx_mod_mantle] = 4.1
            
        if self.crust_idx is not None:
            velfield[self.crust_idx] = self.crust_vs
            vpvsfield[self.crust_idx] = self.crust_vpvs
            
        return velfield,vpvsfield
    
    
    def get_model(self,points=None,parameters=None):
        
        if points is None:
            points = self.points
        if parameters is None:
            parameters = self.parameters
             
        velfield = np.zeros(self.shape)
        vpvsfield = np.zeros(self.shape)
     
        for i in range(len(points)):
            indices = gpts_dict[i]
            indices = np.unravel_index(indices,self.shape2d)
            #indices = (np.repeat(indices*len(self.z),len(self.z)) + 
            #           np.tile(np.arange(len(self.z)),len(indices)))
            fu = interp1d(parameters[i][:,0],parameters[i][:,1],kind='nearest',
                          bounds_error=False,fill_value='extrapolate')
            vs = fu(self.z)
            velfield[indices] = vs
            
            if callable(self.vpvs):
                parameters[i][:,2] = self.vpvs(parameters[i][:,1])
            
            fu = interp1d(parameters[i][:,0],parameters[i][:,2],kind='nearest',
                          bounds_error=False,fill_value='extrapolate')
            vpvs = fu(self.z)
            vpvsfield[indices] = vpvs

        return velfield.flatten(),vpvsfield.flatten()
    
    
    def update_model(self,fields=None,anisotropic=False):
        
        if anisotropic:
            velfield, vpvs, psi2amp, psi2 = fields
        else:
            velfield, vpvs = fields
        
        velfield_cp = velfield.copy().reshape(self.shape)
        vpvs_cp = vpvs.copy().reshape(self.shape)
        if anisotropic:
            psi2amp_cp = psi2amp.copy().reshape(self.shape)
            psi2_cp = psi2.copy().reshape(self.shape)
 
        if len(self.idx_mod_gpts) == 0:
            if anisotropic:   
                return velfield_cp.flatten(), vpvs_cp.flatten(), psi2amp_cp.flatten(), psi2_cp.flatten()
            else:
                return velfield_cp.flatten(), vpvs_cp.flatten()
        
        if self.action=='parameter_update' or self.action=='birth':  
            idx_mod_gpts = np.unravel_index(self.idx_mod_gpts2d,self.shape2d)
            fu = interp1d(self.parameters[self.idx_mod][:,0],
                          self.parameters[self.idx_mod][:,1],kind='nearest',
                          bounds_error=False,fill_value='extrapolate')
            vs = fu(self.z)
            velfield_cp[idx_mod_gpts] = vs
            fu = interp1d(self.parameters[self.idx_mod][:,0],
                          self.parameters[self.idx_mod][:,2],kind='nearest',
                          bounds_error=False,fill_value='extrapolate')
            vpvs = fu(self.z)
            vpvs_cp[idx_mod_gpts] = vpvs
            
        elif self.action=='death':
            
            subset_voronoi_kdt = KDTree(self.points[self.idx_subset_points])            
            update_point_dist, update_point_regions = (
                subset_voronoi_kdt.query(self.gridpoints[self.idx_mod_gpts2d],eps=0))
            update_point_regions = self.idx_subset_points[update_point_regions]
            
            for idx in self.idx_subset_points:
                idx_mod_gpts = self.idx_mod_gpts2d[update_point_regions==idx]
                idx_mod_gpts = np.unravel_index(idx_mod_gpts,self.shape2d)
                fu = interp1d(self.parameters[idx][:,0],
                              self.parameters[idx][:,1],kind='nearest',
                              bounds_error=False,fill_value='extrapolate')
                vs = fu(self.z)
                velfield_cp[idx_mod_gpts] = vs
                fu = interp1d(self.parameters[idx][:,0],
                              self.parameters[idx][:,2],kind='nearest',
                              bounds_error=False,fill_value='extrapolate')
                vpvs = fu(self.z)
                vpvs_cp[idx_mod_gpts] = vpvs
            
        elif self.action=='move':
            
            idx_modified_gridpoints_all = np.unique(np.hstack((
                itemgetter(*self.idx_subset_points)(self.gpts_dict))))
            subset_voronoi_kdt = KDTree(self.points[self.idx_subset_points])
            update_point_dist, update_point_regions = subset_voronoi_kdt.query(
                self.gridpoints[idx_modified_gridpoints_all],eps=0)
            update_point_regions = self.idx_subset_points[update_point_regions]
            
            for idx in self.idx_subset_points:
                idx_mod_gpts = idx_modified_gridpoints_all[update_point_regions==idx]
                idx_mod_gpts = np.unravel_index(idx_mod_gpts,self.shape2d)
                fu = interp1d(self.parameters[idx][:,0],
                              self.parameters[idx][:,1],kind='nearest',
                              bounds_error=False,fill_value='extrapolate')
                vs = fu(self.z)
                velfield_cp[idx_mod_gpts] = vs
                fu = interp1d(self.parameters[idx][:,0],
                              self.parameters[idx][:,2],kind='nearest',
                              bounds_error=False,fill_value='extrapolate')
                vpvs = fu(self.z)
                vpvs_cp[idx_mod_gpts] = vpvs
                
        else:
            raise Exception("action undefined",self.action)

        if callable(self.vpvs):
            vpvs_cp = self.vpvs(velfield_cp)

        if anisotropic:   
            return velfield_cp.flatten(), vpvs_cp.flatten(), psi2amp_cp.flatten(), psi2_cp.flatten()
        else:
            return velfield_cp.flatten(), vpvs_cp.flatten()
                      
        
    # def get_gpts_dictionary(self,points=None,gridpoints=None):
        
    #     gpts_dict = {}
        
    #     if points is None:
    #         points = self.points
    #     if gridpoints is None:
    #         gridpoints = self.gridpoints
                    
    #     kdt = KDTree(points)
    #     point_dist, point_regions = kdt.query(gridpoints)
    #     for i in range(len(points)):
    #         gpts_dict[i] = np.isin(point_regions,i).nonzero()[0]

    #     return gpts_dict
    
    
    # def check_gpts_dictionary(self,action):
        
    #     test_gpts_dictionary = self.get_gpts_dictionary(self.points,self.gridpoints)
        
    #     for i in test_gpts_dictionary:
    #         if not np.array_equal(np.sort(self.gpts_dict[i]), test_gpts_dictionary[i]):
    #             raise Exception(f"gpts dict not right after {action} operation")
        
        
    # def update_gpts_dict(self,selfcheck=False):
        
    #     if self.action is None or self.idx_mod_gpts is None:
    #         raise Exception("cannot update gpts dictionary!")
        
    #     if self.action=='birth':
    #         if (self.subidx_gridpts_new_cell is None or
    #             self.subgrid_lengths is None or
    #             self.idx_subset_points is None):
    #             raise Exception("cannot update gpts dictionary!")

    #         subidx_gridpts_new_cell = np.split(self.subidx_gridpts_new_cell,
    #                                            np.cumsum(self.subgrid_lengths))
    #         for j,idx in enumerate(self.idx_subset_points):
    #             self.gpts_dict[idx] = self.gpts_dict[idx][~subidx_gridpts_new_cell[j]]

    #         self.gpts_dict[len(self.points)-1] = self.idx_mod_gpts2d
            
    #     elif self.action=='death':
    #         if (self.idx_mod is None or self.idx_subset_points is None):
    #             raise Exception("cannot update gpts dictionary!")
    #         # correct the indices of the gpts_idx dictionary
    #         for idx in range(self.idx_mod,len(self.points)):
    #             self.gpts_dict[idx] = self.gpts_dict[idx+1]
    #         del self.gpts_dict[len(self.points)]
                
    #         # add the gridpoints that were in the death cell to the
    #         # neighboring cells
    #         subset_voronoi_kdt = KDTree(self.points[self.idx_subset_points])            
    #         update_point_dist, update_point_regions = subset_voronoi_kdt.query(self.gridpoints[self.idx_mod_gpts2d],eps=0)
    #         update_point_regions = self.idx_subset_points[update_point_regions]
                  
    #         for idx in self.idx_subset_points:
    #             self.gpts_dict[idx] = np.hstack((self.gpts_dict[idx],self.idx_mod_gpts2d[update_point_regions==idx]))
      
    #     elif self.action=='move':
    #         if (self.idx_subset_points is None):
    #             raise Exception("cannot update gpts dictionary!")
    #         idx_modified_gridpoints_all = np.unique(np.hstack((itemgetter(*self.idx_subset_points)(self.gpts_dict))))
    #         subset_voronoi_kdt = KDTree(self.points[self.idx_subset_points])
    #         update_point_dist, update_point_regions = subset_voronoi_kdt.query(self.gridpoints[idx_modified_gridpoints_all],eps=0)
    #         # relate the subset indices to the global point indices
    #         update_point_regions = self.idx_subset_points[update_point_regions]
    #         for idx in self.idx_subset_points:
    #             self.gpts_dict[idx] = idx_modified_gridpoints_all[update_point_regions==idx]

    #     if selfcheck:
    #         kdt = KDTree(self.points)
    #         point_dist, point_regions = kdt.query(self.gridpoints)
    #         for i in range(len(self.points)):
    #             if not np.array_equal(np.sort(self.gpts_dict[i]),np.isin(point_regions,i).nonzero()[0]):
    #                 raise Exception("gpts dict not right after",self.action,"operation")
        
        

    def backup_mod(self):
        
        self.points_backup = self.points.copy()
        self.parameters_backup = deepcopy(self.parameters)
        self.grid_nndists_backup = self.grid_nndists.copy()
        self.grid_nnidx_backup = self.grid_nnidx.copy()
        if self.psi2amp is not None:
            self.psi2amp_backup = self.psi2amp.copy()
        if self.psi2 is not None:
            self.psi2_backup = self.psi2.copy()
        
        
    def reject_mod(self):
        
        if self.points_backup is not None:
            self.points = self.points_backup
        if self.parameters_backup is not None:
            self.parameters = self.parameters_backup
        if self.psi2amp_backup is not None:
            self.psi2amp = self.psi2amp_backup
        if self.psi2_backup is not None:
            self.psi2 = self.psi2_backup
        if self.grid_nndists_backup is not None:
            self.grid_nndists = self.grid_nndists_backup
        if self.grid_nnidx_backup is not None:
            self.grid_nnidx = self.grid_nnidx_backup
            
        self.reset_backup()
        
    def accept_mod(self,selfcheck=False):
        
        # self.update_gpts_dict(selfcheck=selfcheck)
        # if selfcheck:
        #     self.check_gpts_dictionary(self.action)
        self.reset_backup()
 
    def reset_backup(self):
           
        self.points_backup = None
        self.parameters_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod_gpts2d = None
        self.idx_mod = None
        self.psi2amp_backup = None
        self.psi2_backup = None
        self.velrange = None
        self.grid_nndists_backup = None
        self.grid_nnidx_backup = None
        

        
    def plot(self,idx_mod_gpts=None,idx_neighbor_points=None):
        
        from scipy.spatial import delaunay_plot_2d
        tri = Delaunay(self.points)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        _ = delaunay_plot_2d(tri,ax=ax)
        ax.plot(self.gridpoints[:,0],self.gridpoints[:,1],'k.')
        if idx_mod_gpts is not None:
            ax.plot(self.gridpoints[idx_mod_gpts,0],self.gridpoints[idx_mod_gpts,1],'rx',zorder=3)
        cbar = ax.scatter(self.points[:,0],self.points[:,1],c=self.parameters,s=40,zorder=3)
        if idx_neighbor_points is not None:
            ax.scatter(self.points[:,0],self.points[:,1],c=self.parameters,s=60,zorder=3)
        plt.colorbar(cbar)
        plt.show()
        

##############################################################################
##############################################################################
##############################################################################

class wavelets(object):
    
    def __init__(self,gridpoints,shape,convexhull,velmin,velmax,
                 vs2vpvs,init_no_coeffs=None,
                 startmodel=None): 
        
        
        self.gridpoints = gridpoints
        self.z = np.unique(gridpoints[:,2])
        self.shape = shape
        self.convexhull = convexhull # unused
        
        self.wavelet = 'bior4.4'#'bior4.4'
        # decomposition level controls into how many levels the 3D grid is split
        # It can be decomposed so that at the base level there are only a few
        # parameters (few long period wavelets). But that means that at every
        # iteration, a very large part of the model has to be re-calculated,
        # which is very slow...here, I subtract 2 levels to avoid this
        decomposition_level = int(np.log2(np.min(self.shape)))-2
        # minlevel gives the minimum level of the decomposition tree in which
        # the birth/death operations may take place. nodes smaller than
        # minlevel are always filled. Normally, minlevel=0 (no restriction)
        self.minlevel=0
        
        self.vsmin = velmin(self.z)
        self.vsmax = velmax(self.z)
        self.vs_range = self.vsmax-self.vsmin
        
        self.vsprecision = 0.05 # round vs in 0.05 steps (2.35, 2.4, 2.45, ...)
        
        self.vpvs = vs2vpvs
        
        self.coeff_range = {}
        for i in range(2):
            if i==0:
                start_mod = np.zeros(self.shape)
            else:
                start_mod = np.ones(self.shape)
            coeffs = pywt.wavedecn(start_mod, self.wavelet,
                                    level=decomposition_level,
                                    mode='periodization')
            for level in range(len(coeffs)):
                try:
                    self.coeff_range[level]
                except:
                    self.coeff_range[level] = -1e99
                if level==0:
                    self.coeff_range[0] = np.max([
                            self.coeff_range[0],np.max(np.abs(coeffs[level]))])
                else:
                    for direction in list(coeffs[-1]):
                        self.coeff_range[level] = np.max([
                                self.coeff_range[level],np.max(np.abs(coeffs[level][direction]))])
                     
        if startmodel is not None:
            startmodel = (startmodel-self.vsmin)/self.vs_range
        else:
            startmodel = np.random.uniform(0,1,len(self.z))
        startmodel = np.reshape(
            np.tile(startmodel,self.shape[0]*self.shape[1]),self.shape)
        coeffs = pywt.wavedecn(startmodel, self.wavelet,
                               level=decomposition_level,
                               mode='periodization')
        self.coeffs = coeffs
        self.directions = list(self.coeffs[-1])
        
        #self.vs = []
        self.points = []
        for level in range(len(self.coeffs)):
            
            if level==0:
                shape = self.coeffs[level].shape
            elif level==1:
                shape = (len(self.directions),) + shape
            else:
                shape = (len(self.directions), shape[1]*2,shape[2]*2,shape[3]*2)
                
            if level==0:# or level<=self.minlevel:
                for idir,direction in enumerate(self.directions):
                    if level==0:
                        direction=None
                    indices = np.column_stack(np.unravel_index(
                        np.arange(len(self.coeffs[level][direction].flatten())),
                        shape[-3:]))
                    #self.vs.append(self.coeffs[level][direction].flatten())
                    self.points.append(np.column_stack((np.ones(len(indices))*level,
                                                        np.ones(len(indices))*idir,
                                                        indices,
                                                        np.zeros(len(indices)))))
                    if level==0:
                        break
            else:
                for direction in self.directions:
                    self.coeffs[level][direction][:] = 0.
        
        # points has 6 columns: level, direction, yind, xind, zind, isleaf
        # at level=0, the direction does not apply
        #self.vs = np.hstack(self.vs)
        self.points = np.vstack(self.points).astype(int)
        self.points[self.points[:,0]==self.minlevel,-1] = 1
   
        # number of decomposition levels
        self.levels = level+1
        
        self.maxlevel = self.minlevel
        print("Setting maximum level to",self.maxlevel)
        print("Maxlevel will increase during burn-in phase, so that at the "+
              "beginning, the large-scale structures are fit, and fine-scale structures later.")
        
        self.vsfield_backup = np.ones(len(self.gridpoints))*-1
        valid = self.get_model()
        self.update_coeff_range()
        
        # potential children includes all new nodes that can be chosen from
        # in a birth operation (children of currently active nodes)
        self.potential_children = []
        for point in self.points[self.points[:,0]>=self.minlevel]:
            children = self.get_children(point)
            if len(children)>0:
                self.potential_children.append(children)
        if len(self.potential_children)>0:
            self.potential_children = np.vstack(self.potential_children)
        else:
            self.potential_children = np.empty((0,6),dtype=int)
            
        # dictionary that stores the number of possible arrangements of nodes
        # for a certain tree structure
        self.D = {}
        # getting the tree structure. the first number in the array gives the
        # total number of root nodes. The following number give the number of
        # children for each root node.
        # In a binary tree that has a maximum depth of 3, this would look like
        # self.tree = [1,2,2,2]
        # However, in this case we have already an array at the root, e.g.
        # 6*7 elements, resulting in 42 root nodes. Each root node has 3 child-
        # ren, one for each direction. For each direction, each node has 4
        # children, on the succesively finer grids
        # self.tree = [42, 3, 4, 4, 4]
        self.full_tree = np.array([len(self.coeffs[0].flatten()),7] + 
                                  (self.levels-2)*[8])
        # adapt tree to the minlevel and the maxlevel
        self.tree = np.append(np.product(self.full_tree[:self.minlevel+1]),
                              self.full_tree[self.minlevel+1:self.maxlevel+1])
        self.kmin = 0
        for i in range(self.minlevel):
            self.kmin += np.product(self.full_tree[:i+1])
        
        self.points_backup = None
        #self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.psi2amp = None
        self.psi2 = None
        self.psi2amp_backup = None
        self.psi2_backup = None  
        self.coeffs_backup = None
        self.potential_children_backup = None
        self.vsfield_backup = None
        self.vpvsfield_backup = None
        

        # add points until the number of initial coefficients is reached
        if init_no_coeffs is not None:
            init_coeffs = np.max([0,init_no_coeffs-len(self.points)])
        else:
            init_coeffs = 0
        for i in range(init_coeffs):
            self.add_point()
            valid = self.get_model()
            if valid:
                self.accept_mod()
            else:
                self.reject_mod()
        
        
        x = np.unique(self.gridpoints[:,0])
        y = np.unique(self.gridpoints[:,1])
        self.wavelet_coords = {}
        for level in range(self.levels):
            if level==0:
                shape = self.coeffs[level].shape[:2]
            else:
                shape = self.coeffs[level]['ddd'].shape[:2]
            dy = (y[-1]-y[0])/shape[0]
            dx = (x[-1]-x[0])/shape[1]
            xcoords = np.arange(x[0]+dx/2.,x[-1],dx)
            ycoords = np.arange(y[0]+dy/2.,y[-1],dy)
            self.wavelet_coords[level] = np.meshgrid(xcoords,ycoords)
        
        
        #slowfield = self.get_model()
        #if (1./slowfield > velmax).any() or (1./slowfield < velmin).any():
        #    raise Exception("bad starting model")        
        
        self.acceptance_rate = {}
        self.accepted_steps = {}
        self.rejected_steps = {}
        for level in range(self.levels):
            self.acceptance_rate[level] = np.zeros(100,dtype=int)
            self.accepted_steps[level] = 0
            self.rejected_steps[level] = 0
      
        
    def update_coeff_range(self):
        
        for level in range(self.levels):
            if level==0:
                continue
            maxcoeff = 0.
            for direction in self.coeffs[level]:
                mc = np.max(np.abs(self.coeffs[level][direction]))
                maxcoeff = np.max([maxcoeff,mc])
            if maxcoeff == 0.:
                maxcoeff = self.coeff_range[level-1]/3.
            self.coeff_range[level] = maxcoeff*1.1
                    
     
    def update_maxlevel(self):
        
        if self.maxlevel+1<self.levels:
            print("setting maximum decomposition level from:")
            print(self.maxlevel,self.tree)
            self.maxlevel += 1
            self.tree = np.append(np.product(self.full_tree[:self.minlevel+1]),
                              self.full_tree[self.minlevel+1:self.maxlevel+1])
            print("to")
            print(self.maxlevel,self.tree)
            
            for point in self.points[self.points[:,0]==self.maxlevel-1]:
                children = self.get_children(point)
                self.potential_children = np.vstack((self.potential_children,
                                                     children))
     
        
    def optimize_gridsize(minx,maxx,miny,maxy,zaxis,
                          xgridspacing,ygridspacing):
        
        # function to optimize the gridsize so that we can construct a tree
        # that has 4 children for every node.
        # During the wavelet decomposition the grid is split in successively
        # rougher grids by taking only every second sample of the original grid
        # 16 -> 8 -> 4 -> 2 -> 1
        # If the gridsize is not a power of 2, this will fail at some point.
        # 18 -> 9 -> !
        # In general, this is not a problem, because the wavelet decomposition
        # works also with non-integer decimations:
        # 18 -> 9 -> 5 -> 3 -> 2 -> 1
        # however, this means that not every node has exactly four children
        # which makes the calculations more difficult
        # this function will therefore try to avoid this by decreasing the
        # grid spacing, i.e. increasing the number of samples
        # 18 samples increased to 20
        # 20 -> 10 -> 5
        # the optimization will stop at some point, meaning that it will not
        # be possible to have only one single root node
        # instead, in the example above, there would be 5 root nodes

        #minx = -110.75607181407213
        #maxx = 104.96901884072297
        #xgridspacing = 1.
        
        xpoints = len(np.arange(minx,maxx+xgridspacing,xgridspacing))
        partition_levels = 0
        while True:
            if xpoints==1:
                xpartitions = int(xpoints * 2**partition_levels)
                break
            if xpoints%2==0:
                xpoints/=2
                partition_levels += 1
            else:
                xpoints = int(xpoints+1)
                xnew = np.linspace(minx,maxx,
                                   xpoints * 2**partition_levels)
                dxnew = xnew[1]-xnew[0]
                if dxnew < 0.5*xgridspacing:
                    xpoints-=1
                    xnew = np.linspace(minx,maxx,
                                       xpoints * 2**partition_levels)
                    dxnew = xnew[1]-xnew[0]
                    xpartitions = (xpoints * 2**partition_levels)
                    break
                
        ypoints = len(np.arange(miny,maxy+ygridspacing,ygridspacing))
        partition_levels = 0
        while True:
            if ypoints==1:
                ypartitions = int(ypoints * 2**partition_levels)
                break
            if ypoints%2==0:
                ypoints/=2
                partition_levels += 1
            else:
                ypoints = int(ypoints+1)
                ynew = np.linspace(miny,maxy,
                                   ypoints * 2**partition_levels)
                dynew = ynew[1]-ynew[0]
                if dynew < 0.5*ygridspacing:
                    ypoints-=1
                    ynew = np.linspace(miny,maxy,
                                       ypoints * 2**partition_levels)
                    dynew = ynew[1]-ynew[0]
                    ypartitions = (ypoints * 2**partition_levels)
                    break

        i=0
        while 2**i<len(zaxis):
            i+=1
        nzpoints = 2**i
        if nzpoints!=len(zaxis):
            zrange = np.max(zaxis)-np.min(zaxis)
            rel_spacing = np.diff(zaxis)/zrange
            rel_spacing_new = np.interp(np.linspace(0,1,nzpoints-1),
                                        np.linspace(0,1,len(rel_spacing)),
                                        rel_spacing)
            rel_spacing_new /= np.sum(rel_spacing_new)
            
            zaxis_new = np.append(zaxis[0],zaxis[0]+
                                  np.cumsum(zrange*rel_spacing_new))
            # round values
            precision = np.min(np.diff(zaxis_new))/100.
            if precision<1.:
                decimals = int(np.abs(np.log10(precision)))+1
            else:
                decimals = 0
            zaxis_new = np.around(zaxis_new,decimals)
        else:
            zaxis_new = zaxis
        
        
        return xpartitions,ypartitions,zaxis_new
                
                
    
    def memoize_arrangements(self,tree,k):
        
        # this function is computationally demanding and requires many loops.
        # could potentially profit from just-in-time compilation (numba jit)
        # but that makes the usage of dictionaries and self difficult.
        
        # based on "Geophysical imaging using trans-dimensional trees" by
        # Hawkins and Sambridge, 2015, appendix A

        # this function will successively fill the D dictionary where the
        # number of possible arrangements in a tree is stored
        # for example in a ternary tree with 3 active nodes, there are 12
        # possible arrangements
        # tree = (1, 3, 3, 3, 3); k=3
        # D[((1,3,3,3,3),3)] = 12    

        kmax = 0
        for i in range(len(tree)):
            kmax += np.product(tree[:i+1])
        if k==0 or k==kmax:
            return 1
        if k<0 or k>kmax:
            return 0
        
        try:
            return self.D[(tuple(tree),k)]
        except:
            j = tree[0]
            if j==1:
                A = tree[1:]
                self.D[(tuple(tree),k)] = self.memoize_arrangements(A,k-1)
            elif j%2==1:
                A = tree.copy()
                A[0] = 1
                B = tree.copy()
                B[0] -= 1
                self.D[(tuple(tree),k)] = self.compute_subtrees(A,B,k)
            else:
                A = tree.copy()
                A[0] = tree[0]/2
                B = tree.copy()
                B[0] = tree[0]/2
                self.D[(tuple(tree),k)] = self.compute_subtrees(A,B,k)
            
        return self.D[(tuple(tree),k)]
            
            
    def compute_subtrees(self,A,B,k):
        arrangements = 0
        for i in range(k+1):
            a = self.memoize_arrangements(A,i)
            b = self.memoize_arrangements(B,k-i)
            arrangements += a*b
        return arrangements
    
        # # for comparison, possible arrangements in a ternary tree:
        # def compute_ternary(k):
        #     if k<=0:
        #         return 1
        #     arrangements = 0
        #     for i in range(k):
        #         a = compute_ternary(i)
        #         subsum = 0
        #         for j in range(k-i):
        #             b = compute_ternary(j)
        #             c = compute_ternary(k-i-j-1)
        #             subsum += b*c
        #         arrangements += a*subsum
        #     return arrangements
                
         
    def get_children(self,point):
        
        level,direction,yind,xind,zind,isleaf = point
        
        if level+1 > self.maxlevel or level+1 < self.minlevel:
            return []
        
        # a level 0 node has 7 children (directions)
        if level==0:
            return np.array([[level+1,0,yind,xind,zind,1],
                             [level+1,1,yind,xind,zind,1],
                             [level+1,2,yind,xind,zind,1],
                             [level+1,3,yind,xind,zind,1],
                             [level+1,4,yind,xind,zind,1],
                             [level+1,5,yind,xind,zind,1],
                             [level+1,6,yind,xind,zind,1]])
        
        # a higher level node has 8 children, the direction is the same as the
        # parent direction but the "pixel" gets split into 8 smaller ones
        children = np.array([[level+1,direction,yind*2,xind*2,zind*2,1],
                             [level+1,direction,yind*2+1,xind*2,zind*2,1],
                             [level+1,direction,yind*2,xind*2+1,zind*2,1],
                             [level+1,direction,yind*2,xind*2,zind*2+1,1],
                             [level+1,direction,yind*2+1,xind*2+1,zind*2,1],
                             [level+1,direction,yind*2,xind*2+1,zind*2+1,1],
                             [level+1,direction,yind*2+1,xind*2,zind*2+1,1],
                             [level+1,direction,yind*2+1,xind*2+1,zind*2+1,1]])
                             
        
        return children
    

    def get_parent_idx(self,point):
        
        level,direction,yind,xind,zind,isleaf = point
        
        if level==0:
            raise Exception("level 0 has no parent")
            
        if level==1:
            idx_parent = np.where((self.points[:,(0,2,3,4)]==
                            np.array([level-1,yind,xind,zind])).all(axis=1))[0]
        
        else:
            idx_parent = np.where((self.points[:,:5]==np.array([
                level-1,direction,int(yind/2.),int(xind/2.),int(zind/2.)])).all(axis=1))[0]
        
        if len(idx_parent) > 1:
            raise Exception("there should only be one parent!")
        
        return idx_parent[0]
    
        
    def get_mod_point_coords(self,idx=None):
        
        if idx is not None:
            level,direction,yind,xind,zind,isleaf = self.points[idx]
        elif self.action == 'death':
            level,direction,yind,xind,zind,isleaf = self.point_removed
        else:
            level,direction,yind,xind,zind,isleaf = self.points[self.idx_mod]

        return ((self.wavelet_coords[level][0][yind,xind],
                 self.wavelet_coords[level][1][yind,xind]),
                level)
    
    def parameter_update(self,idx,propdist_std,backup=True):
        
        self.action = 'update'
        if backup:
            self.backup_mod()
            
        self.propdist_std = propdist_std
        self.dx = np.random.normal(loc=0, scale=propdist_std)
          
        # with increasing level, the coefficient ranges become smaller
        # this happens with an approximate factor of 0.5
        # this way, the acceptance at each level should be approximately equal
        level,direction,yind,xind,zind,isleaf = self.points[idx]
        #self.vs[idx] += dcoeff#*1.2**(8-level) # 5 could be replaced by any value, but should be adjusted to the proposal ratio
        if level==0:
            self.coeffs[level][yind,xind,zind] += self.dx
            coeff_new = self.coeffs[level][yind,xind,zind]
        else:
            self.coeffs[level][self.directions[direction]][yind,xind,zind] += self.dx
            coeff_new = self.coeffs[level][self.directions[direction]][yind,xind,zind]
        
        self.idx_mod = idx      
        
        if np.abs(coeff_new)>self.coeff_range[level]:
            return False
        else:
            return True
        
        
    def add_point(self,birth_prop='uniform',anisotropic=False,backup=True):
        
        if type(birth_prop[1])!=type('uniform'):
            raise Exception("only uniform birth proposals are currently implemented")
        
        # do a better implementation of the case where the maximum tree 
        # height is reached
        if len(self.potential_children) == 0:
            return False
        
        self.action='birth'
        if backup:
            self.backup_mod()            
        self.idx_mod = len(self.points)
           
        # randomly choose one of the potential children
        idx_birth = np.random.randint(0,len(self.potential_children))
        
        self.points = np.vstack((self.points,self.potential_children[idx_birth]))
        level,direction,yind,xind,zind,isleaf = self.points[-1]
        coeff_new = np.random.uniform(-self.coeff_range[level],self.coeff_range[level])
        if level==0:
            self.coeffs[level][yind,xind,zind] = coeff_new
        else:
            self.coeffs[level][self.directions[direction]][yind,xind,zind] = coeff_new
        #self.vs = np.append(self.vs,np.random.uniform(
        #    -self.coeff_range[level][zind],
        #     self.coeff_range[level][zind]))
        
        # needed for the proposal ratio calculation
        self.no_birthnodes = len(self.potential_children)
        
        self.potential_children = np.delete(self.potential_children,
                                            idx_birth,axis=0)
        #self.potential_children.remove(tuple(self.points[-1]))
        children = self.get_children(self.points[-1])
        if len(children) > 0:
            self.potential_children = np.vstack((self.potential_children,
                                                 children))
        if level>0:
            idx_parent = self.get_parent_idx(self.points[-1])
            self.points[idx_parent,-1] = 0
        
        return True


    def remove_point(self,anisotropic=False,death_prop=None,backup=True):
        
        self.action='death'
        if backup:
            self.backup_mod()
                       
        # randomly choose one of the leaf nodes to remove
        # leaf nodes are those that have no children
        self.no_deathnodes = np.sum(self.points[:,-1])
        if self.no_deathnodes > 0:
            idx_death = np.random.choice(np.where(self.points[:,-1])[0])
        else:
            return False
        
        self.point_removed = self.points[idx_death]
        level,direction,yind,xind,zind,isleaf = self.point_removed
        if level<self.minlevel:
            raise Exception("should not happen!")
        
        self.points = np.delete(self.points,idx_death,axis=0)
        #self.vs = np.delete(self.vs,idx_death)
        if level==0:
            self.coeffs[level][yind,xind,zind] = 0.
        else:
            self.coeffs[level][self.directions[direction]][yind,xind,zind] = 0.
        
        # the children of the removed point have to be removed from the list
        # of potential children (in a birth step a child from potential
        # children is chosen)
        children = self.get_children(self.point_removed)
        if len(children)>0:
            idx_remove = np.where(
                (self.potential_children[:,0]==children[0,0]) * 
                np.in1d(self.potential_children[:,1],children[:,1]) *
                np.in1d(self.potential_children[:,2],children[:,2]) *
                np.in1d(self.potential_children[:,3],children[:,3]) *
                np.in1d(self.potential_children[:,4],children[:,4]))[0]
            self.potential_children = np.delete(self.potential_children,
                                                idx_remove,0)
        # the removed point has to be added to the list of potential children
        self.potential_children = np.vstack((self.potential_children,
                                             self.point_removed))
        # the parent might become a leaf node now (removable), do a check
        if self.point_removed[0]>self.minlevel+1:
            # check if the removed node had any siblings 
            level,direction,yind,xind,zind,isleaf = self.point_removed
            direction = self.directions[direction]
            coeff_sum_siblings = (
                # removed point and its seven siblings
                self.coeffs[level][direction][int(yind/2)*2,int(xind/2)*2,int(zind/2)*2] +
                self.coeffs[level][direction][int(yind/2)*2+1,int(xind/2)*2,int(zind/2)*2] +
                self.coeffs[level][direction][int(yind/2)*2,int(xind/2)*2+1,int(zind/2)*2] +
                self.coeffs[level][direction][int(yind/2)*2+1,int(xind/2)*2+1,int(zind/2)*2] +
                self.coeffs[level][direction][int(yind/2)*2,int(xind/2)*2,int(zind/2)*2+1] +
                self.coeffs[level][direction][int(yind/2)*2+1,int(xind/2)*2,int(zind/2)*2+1] +
                self.coeffs[level][direction][int(yind/2)*2,int(xind/2)*2+1,int(zind/2)*2+1] +
                self.coeffs[level][direction][int(yind/2)*2+1,int(xind/2)*2+1,int(zind/2)*2+1] -
                # subtract the coefficient of the removed point again
                self.coeffs[level][direction][yind,xind,zind])
            # if this sums up to zero, all sibiling coefficients must be zero
            if coeff_sum_siblings == 0.: # i.e. no siblings
                # make parent node a leaf node
                idx_parent = self.get_parent_idx(self.point_removed)
                self.points[idx_parent,-1] = 1
 
        return True
         
        
    def get_model(self,coeffs=None,vpvs=None,
                  psi2amp=None,psi2=None,anisotropic=False):
        
        if coeffs is None:
            coeffs = self.coeffs
        if vpvs is None:
            vpvs = self.vpvs

        # reconstruct the field from the coefficients
        rel_velfield = pywt.waverecn(coeffs, self.wavelet, mode='periodization')
        if np.any(rel_velfield<0) or np.any(rel_velfield>1):
            return False
        self.vsfield = myround(rel_velfield*self.vs_range + self.vsmin,self.vsprecision).flatten()
        
        if callable(vpvs):
            self.vpvsfield = vpvs(self.vsfield)
            self.idx_mod_gpts = self.vsfield!=self.vsfield_backup
        else:
            raise Exception("variable vpvs currently not implemented")
            
        return True
       
    
    def smooth_coefficients(self,idx_2d,idepth):
        
        yidx,xidx = np.unravel_index(idx_2d,self.shape[:2])
        for i in range(len(idx_2d)):
            for j,level in enumerate(np.arange(self.levels)[::-1]):
                idx_coeffs = np.where((self.points[:,0]==level)*
                                      (self.points[:,2]>=int(np.floor(yidx[i]/(2**(j+1)))))*
                                      (self.points[:,2]<=int(np.ceil(yidx[i]/(2**(j+1)))))*                                      
                                      (self.points[:,3]>=int(np.floor(xidx[i]/(2**(j+1)))))*
                                      (self.points[:,3]<=int(np.ceil(xidx[i]/(2**(j+1))))))[0]
                if len(idx_coeffs)>0:
                    for idx in idx_coeffs:
                        level,direction,yind,xind,zind,isleaf = self.points[idx]
                        if level==0:
                            self.coeffs[level][yind,xind,zind] *= 0.9
                        else:
                            self.coeffs[level][self.directions[direction]][yind,xind,zind] *= 0.9
                        print("smoothing",self.points[idx])
                    if j >= idepth or level==0:
                        break
            else:
                raise Exception("should break somewhere")
            
            
            
    def get_prior_proposal_ratio(self):
        # returns the log of the prior ratio and the proposal ratio
        # delayed rejection is currently not implemented
        if self.action == 'update':
            # if the number of active nodes is unchanged and only the 
            # coefficients are changed, the probability from going from one
            # coefficient to another one is equal to the reverse step
            return 0
        elif self.action == 'birth':
            # assuming we draw from a uniform distribution
            # nominator: number of possible nodes to choose from during birth
            # denominator: number of possible death nodes to choose from after
            #              birth
            proposal_ratio = self.no_birthnodes / np.sum(self.points[:,-1])
            # eqs 13 & 16 from Hawkins & Sambridge 2015
            prior_ratio = (
                self.memoize_arrangements(self.tree, len(self.points)-1-self.kmin) /
                self.memoize_arrangements(self.tree, len(self.points)-self.kmin) )
        elif self.action == 'death':
            # same as birth but inverse
            proposal_ratio = self.no_deathnodes / len(self.potential_children)
            prior_ratio = (
                self.memoize_arrangements(self.tree, len(self.points)+1-self.kmin) /
                self.memoize_arrangements(self.tree, len(self.points)-self.kmin) )
        else:
            raise Exception("unknown update type")

        return np.log(proposal_ratio * prior_ratio)
    
    
    def backup_mod(self):
        
        self.points_backup = self.points.copy()
        #self.vs_backup = self.vs.copy()
        self.coeffs_backup = deepcopy(self.coeffs)
        self.potential_children_backup = self.potential_children.copy()
        self.vsfield_backup = self.vsfield.copy()
        self.vpvsfield_backup = self.vpvsfield.copy()
        if self.psi2amp is not None:
            self.psi2amp_backup = self.psi2amp.copy()
        if self.psi2 is not None:
            self.psi2_backup = self.psi2.copy()
        
        
    def reject_mod(self):
        
        if self.action=='update':
            self.rejected_steps[self.points[self.idx_mod,0]] += 1
            self.acceptance_rate[self.points[self.idx_mod,0]][0] = 0
            self.acceptance_rate[self.points[self.idx_mod,0]] = np.roll(
                self.acceptance_rate[self.points[self.idx_mod,0]],1)
                                
        if self.points_backup is not None:
            self.points = self.points_backup
        if self.vsfield_backup is not None:
            self.vsfield = self.vsfield_backup
        if self.vpvsfield_backup is not None:
            self.vpvsfield = self.vpvsfield_backup
        #if self.vs_backup is not None:
        #    self.vs = self.vs_backup
        if self.psi2amp_backup is not None:
            self.psi2amp = self.psi2amp_backup
        if self.psi2_backup is not None:
            self.psi2 = self.psi2_backup
        if self.coeffs_backup is not None:
            self.coeffs = self.coeffs_backup
        if self.potential_children_backup is not None:
            self.potential_children = self.potential_children_backup
         
        # for point in self.points:
        #     if point[-1]==0:
        #         continue
        #     children = self.get_children(point)
        #     for child in children:
        #         if not child in self.potential_children:
        #             raise Exception(f"error after reject {self.action}") 
                    
        self.points_backup = None
        #self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.psi2amp_backup = None
        self.psi2_backup = None
        self.point_removed = None
        self.coeffs_backup = None
        self.no_birthnodes = None
        self.no_deathnodes = None
        self.potential_children_backup = None
        self.vsfield_backup = None
        self.vpvsfield_backup = None
    
        
    def accept_mod(self,selfcheck=False):
        
        # testlist = []
        # for item in self.points:
        #     testlist.append(tuple(item[:-1]))
        # for point in self.points:
        #     if point[0]==0:
        #         continue
        #     if point[-1]==0:
        #         children = self.get_children(point)
        #         for child in children:
        #             if tuple(child[:-1]) in testlist:
        #                 break
        #         else:
        #             print(point)
        #             raise Exception("this node should be marked as leaf! error after",self.action) 
                    
        if self.action=='update':
            self.accepted_steps[self.points[self.idx_mod,0]] += 1
            self.acceptance_rate[self.points[self.idx_mod,0]][0] = 1
            self.acceptance_rate[self.points[self.idx_mod,0]] = np.roll(
                self.acceptance_rate[self.points[self.idx_mod,0]],1)
            
        if np.random.rand()<0.05:
            self.update_coeff_range()
         
        if selfcheck:
            for point in self.points:
                if (self.potential_children[:,:-1]==point[:-1]).all(axis=1).any():
                    idx_child = np.where((self.potential_children[:,:-1]==point[:-1]).all(axis=1))[0]
                    print(idx_child,self.potential_children[idx_child],point)
                    raise Exception(f"point is also in potential children after {self.action}")
                if np.sum((self.points[:,:-1]==point[:-1]).all(axis=1)) > 1:
                    idx_double = np.where((self.points==point).all(axis=1))[0]
                    print(idx_double,self.points[idx_double])
                    raise Exception(f"double points after {self.action}")
            
        # for point in self.points:
        #     if point[-1]==0:
        #         continue
        #     children = self.get_children(point)
        #     for child in children:
        #         if not (self.potential_children==child).all(axis=1).any():
        #             raise Exception(f"error after accept {self.action}")
        #         if (self.points==child).all(axis=1).any():
        #             raise Exception(f"error after accept {self.action}, should not be leaf node!")
                   
        # for point in self.points:
        #     if point[0]==0:
        #         continue
        #     parent_idx = self.get_parent_idx(point)
        #     if len(parent_idx)!=1:
        #         raise Exception("error after",self.action)
        
        self.points_backup = None
        #self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.psi2amp_backup = None
        self.psi2_backup = None
        self.point_removed = None
        self.coeffs_backup = None
        self.no_birthnodes = None
        self.no_deathnodes = None
        self.potential_children_backup = None
        self.vsfield_backup = None
        self.vpvsfield_backup = None        
        
            
    def plot(self,idx_mod_gpts=None,idx_neighbor_points=None):

        #slowfield = self.get_model()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cbar = ax.pcolormesh(1./slowfield.reshape(self.shape),cmap=plt.cm.seismic_r)
        plt.colorbar(cbar)
        plt.show()
        
def myround(x, precision):
    return precision * np.around(x/precision)

def rv_truncnorm(loc,scale,clip_a,clip_b):
    a, b = (clip_a - loc) / scale, (clip_b - loc) / scale
    return truncnorm(a,b,loc=loc,scale=scale)
