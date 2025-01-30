#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 17:18:22 2018

@author: emanuel
"""

import numpy as np
#np.seterr(under='ignore')
#np.seterr(over='ignore')
from scipy.interpolate import interp1d, griddata
from scipy.spatial import Delaunay, ConvexHull, KDTree
import matplotlib, os
if not bool(os.environ.get('DISPLAY', None)):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from copy import deepcopy
from . import Parameterization
from . import BodyWaveModule as bwm
from . import SurfaceWaveModule as swm
    
class RJMC(object):
    
    def __init__(self,chain_no,targets,prior,params):
        
        self.chain_no = chain_no        
        self.logfile = os.path.join(params.logfile_path,"logfile_%d.txt" %(self.chain_no))
        self.dirpath = params.logfile_path
        self.print_stats_step = params.print_stats_step       
        
        # interpolation method
        self.parameterization = params.parameterization
        
        self.misfit_norm = params.misfit_norm
        self.likedist = params.likedist
        
        self.target_iterations = params.target_iterations
        self.nburnin = params.nburnin
        self.update_paths_interval = params.update_paths_interval
        self.init_no_points = params.init_no_points
        self.init_vel_points = params.init_vel_points
        self.startmodel_depth = params.startmodel_depth
        self.startmodel_vs = params.startmodel_vs
        
        if params.delayed_rejection and self.parameterization!='voronoi':
            print("delayed rejection scheme is currently only implemented for a Voronoi parameterization.")
            self.delayed_rejection = False
        else:
            self.delayed_rejection = params.delayed_rejection
        self.store_models = params.store_models

        # get all prior values
        self.min_datastd = prior.min_datastd
        self.max_datastd = prior.max_datastd
        self.nmin_points = prior.nmin_points
        self.nmax_points = prior.nmax_points
        
        self.projection = params.projection
        
        # set up model area as regular grid in x,y,z coordinates (km scale)
        self.minx,self.maxx,self.miny,self.maxy = params.minx,params.maxx,params.miny,params.maxy
        self.xgridspacing = params.xgridspacing
        self.ygridspacing = params.ygridspacing
        self.x = np.arange(params.minx, params.maxx+params.xgridspacing, params.xgridspacing)
        self.y = np.arange(params.miny, params.maxy+params.ygridspacing, params.ygridspacing)
        self.z = params.z
        if (np.diff(self.z)<=0.).any():
            raise Exception("The z axis must be strictly increasing!")
        self.minz = np.min(self.z)
        self.maxz = np.max(self.z)
        zhalfspace = self.maxz*2
        for target in targets:
            if target.type == 'bodywaves' and target.isactive:
                zhalfspace = np.max([zhalfspace,target.model_bottom_depth*1.05])
        self.z = np.append(self.z,zhalfspace)
        if self.parameterization=='wavelets':
            xsize,ysize,znew = Parameterization.wavelets.optimize_gridsize(
                self.minx,self.maxx,self.miny,self.maxy,self.z,
                self.xgridspacing,self.ygridspacing)
            self.x = np.linspace(self.minx,self.maxx,xsize)
            self.xgridspacing = self.x[1]-self.x[0]
            self.y = np.linspace(self.miny,self.maxy,ysize)
            self.ygridspacing = self.y[1]-self.y[0]
            if self.chain_no==1:
                print(f"Warning: setting xgridspacing from {params.xgridspacing} "+
                      f"to {self.xgridspacing} to make the x-axis length a "+
                      "power of 2 (necessary for the wavelet transform).")
                print(f"Warning: setting ygridspacing from {params.ygridspacing} "+
                      f"to {self.ygridspacing} to make the y-axis length a "+
                      "power of 2 (necessary for the wavelet transform).")
                print(f"Warning: setting the z-axis from \n{self.z}\nto\n{znew}\n"+
                      "to make the z-axis length a power of 2.")
            self.z_scaled = self.z = znew
            self.z_scaling = self.inverse_z_scaling = interp1d(self.z,self.z_scaled)
        else:
            # z scaling is useful because the earth stucture is typically layered
            # meaning that we have most velocity variation in z direction
            deep_z_scale = np.min([1,params.z_scale]) # below 100km, fix z scale
            if params.z_scale>deep_z_scale:
                z_func = interp1d([self.z[0],100,self.z[-1]],
                                  [params.z_scale,deep_z_scale,deep_z_scale],
                                  kind='linear')
            else:
                z_func = interp1d([self.z[0],self.z[-1]],
                                  [params.z_scale,params.z_scale],
                                  kind='linear')
            self.z_scaled = np.append(self.z[0],np.cumsum(np.diff(self.z)*z_func(self.z[:-1])))
            self.z_scaling = interp1d(self.z,self.z_scaled)
            self.inverse_z_scaling = interp1d(self.z_scaled,self.z)
            self.minz_scaled = self.z_scaled[0]
            self.maxz_scaled = self.z_scaled[-2]
        
        self.X,self.Y,self.Z_scaled = np.meshgrid(self.x,self.y,self.z_scaled)
        self.shape = self.X.shape
        self.gridpoints = np.column_stack((self.X.flatten(),self.Y.flatten(),
                                           self.Z_scaled.flatten()))
        self.gridpoints2d = np.column_stack((self.X[:,:,0].flatten(),self.Y[:,:,0].flatten()))

        # Vs priors has 3 columns: depth [km], vsmin [km/s], vsmax [km/s]
        vs_priors = prior.vs_priors
        if np.max(vs_priors[:,0])<self.maxz:
            raise Exception("The maximum depth of the vs priors should be equal to or greater than the maximum depth of the model search.")
        if np.min(vs_priors[:,0])>self.minz:
            raise Exception("The minimum depth of the vs priors should be equal to or smaller than the minimum depth of the model search.")
        vs_priors = vs_priors[vs_priors[:,0]<=self.maxz]
        vs_priors[-1,2] *= 1.1 # 10 percent extra for bottom layer
        vs_priors = np.vstack((vs_priors,(zhalfspace,vs_priors[-1,1],vs_priors[-1,2]*1.1)))
        self.minvs = interp1d(self.z_scaling(vs_priors[:,0]),vs_priors[:,1])
        self.maxvs = interp1d(self.z_scaling(vs_priors[:,0]),vs_priors[:,2])
        vs_conversion = prior.vs_conversion
        self.flexible_vpvs = prior.flexible_vpvs
        if self.flexible_vpvs:
            # start with a sqrt(3) relationship, then controlled by the model search
            meanvpvs = np.mean([prior.vpvsmin,prior.vpvsmax])
            self.vs2vpvs = interp1d([1,8],[meanvpvs,meanvpvs],kind='nearest',fill_value='extrapolate')
            self.vpvsmin = prior.vpvsmin
            self.vpvsmax = prior.vpvsmax
        else:
            self.vs2vpvs = interp1d(vs_conversion[:,0],vs_conversion[:,1],kind='nearest',fill_value='extrapolate')
            self.vpvsmin = self.vpvsmax = None
        self.vs2rho = interp1d(vs_conversion[:,0],vs_conversion[:,2],kind='nearest',fill_value='extrapolate')
        
        if prior.crust_model is not None and prior.moho_model is not None:
            raise Exception("Either a Moho model OR a crustal model should be provided, not both!")
        if prior.moho_model is not None:
            self.moho_depths = griddata(prior.moho_model[:,:2],
                                        self.z_scaling(prior.moho_model[:,2]),
                                        self.gridpoints[:,:2],
                                        fill_value=self.z_scaling(30.))
        else:
            self.moho_depths = None
        if prior.crust_model is not None:
            zz = self.inverse_z_scaling(self.gridpoints[:,2])
            tree = KDTree(prior.crust_model[:,:3])
            nndist,nnidx = tree.query(np.column_stack((self.gridpoints[:,:2],zz)))
            zdist = zz-prior.crust_model[nnidx,2]
            dzmax = np.max(np.diff(np.unique(prior.crust_model[:,2]))/2.)
            invalid = np.logical_or(zdist>dzmax,zz>np.max(prior.crust_model[:,2]))
            self.vp_crust = prior.crust_model[nnidx,3]
            self.vp_crust[invalid] = np.nan
            self.vs_crust = prior.crust_model[nnidx,4]
            self.vs_crust[invalid] = np.nan
            test = []
            vp3d = np.reshape(self.vp_crust,self.shape)
            for iy,y in enumerate(self.y):
                for ix,x in enumerate(self.x):
                    if np.sum(np.isnan(vp3d[iy,ix,:]))==0:
                        zcrust=np.max(self.z)
                    else:
                        zcrust = self.z[np.where(np.isnan(vp3d[iy,ix,:]))[0][0]]
                    test.append([x,y,zcrust])
            test = np.vstack(test)
            plt.ioff()
            fig = plt.figure(figsize=(14,8))
            plt.subplot(121)
            plt.scatter(test[:,0],test[:,1],c=test[:,2])
            plt.colorbar(fraction=0.05,shrink=0.4,label='crustal model depth [km]')
            plt.gca().set_aspect('equal')
            plt.subplot(122)
            idx5 = np.argmin(np.abs(self.z-5.))
            plt.pcolormesh(self.x,self.y,vp3d[:,:,idx5],cmap=plt.cm.seismic_r)
            plt.colorbar(fraction=0.05,shrink=0.4,label='Vp [km/s] at 5 km depth')
            plt.gca().set_aspect('equal')
            plt.savefig(os.path.join(self.dirpath,"testplot_crustal_model.jpg"),bbox_inches='tight')
            plt.close(fig)
        else:
            self.vp_crust = self.vs_crust = None
        
        self.error_type = params.data_std_type
        self.sw_hyperparameter_level = params.sw_hyperparameter_level
        if self.sw_hyperparameter_level not in ['dataset','period'] and self.error_type!='fixed':
            raise Exception(f"The sw_hyperparameter_level has to be either 'dataset' or 'period', unknown option {self.sw_hyperparameter_level}.")
        self.bw_hyperparameter_level = params.bw_hyperparameter_level
        if self.bw_hyperparameter_level not in ['dataset','event'] and self.error_type!='fixed':
            raise Exception(f"The bw_hyperparameter_level has to be either 'dataset' or 'event', unknown option {self.bw_hyperparameter_level}.")
        self.propsigmastd = params.propsigmastd

        self.temperature = params.temperature
        
        if params.propvelstd == 'adaptive':
            self.adaptive_proposalstd = True
        else:
            self.adaptive_proposalstd = False
        self.propvelstd_pointupdate = params.propvelstd_pointupdate
        self.propvelstd_dimchange = params.propvelstd_dimchange
        self.propmovestd = params.propmovestd
        
        # if user wants to visualize progress:
        self.visualize = params.visualize
        self.visualize_step = params.visualize_step
        
        self.collect_step = params.collect_step
        self.collection_vs_models = []
        self.collection_vpvs_models = []
        self.collection_no_points = []
        self.collection_loglikelihood = []
        self.collection_residuals = []

        
    
    def initialize(self,targets):
        
        print("Chain %d: initialize..." %self.chain_no)
        self.total_steps = 0
        with open(self.logfile,"w") as f:
            f.write("%s\nStarting chain %d\n\n" %(time.ctime(),self.chain_no))
            
        self.selfcheck=False
        if self.selfcheck:
            print("Warning: selfcheck turned on. Calculations will take much longer.")

        # this is currently not working properly
        # the idea is to reduce the grid spacing during the burnin
        # phase so that the calculations are faster
        self.gridsizefactor = 1
        #self.gridsizefactor_inc = (1-self.gridsizefactor)/(self.nburnin*0.9) # linear
        self.gridsizefactor_inc = (1-self.gridsizefactor)/(self.nburnin*0.9)**2 # quadratic
                
        self.likedist_updates = []
        self.datastd = {}
        self.foutlier = {}
        self.widthoutlier = {}
        self.degfree = self.std_scaling = self.std_intercept = None
        self.collection_datastd = {}

        # Initialize targets for model search
        for target in targets:
            if target.type=='bodywaves':
                bodywaves = target
            elif target.type=='surfacewaves':
                surfacewaves = target
        
        # Initialize Bodywave dataset
        if bodywaves.isactive:
            if self.bw_hyperparameter_level=='event':
                print("chain",self.chain_no,"each earthquake event has a unique data standard deviation")
            self.datastd.update(deepcopy(bodywaves.datastd))
            if self.error_type != 'fixed':
                for dataset in bodywaves.data:
                    self.datastd[dataset] = np.ones(len(bodywaves.data[dataset]))*np.mean(
                        [self.min_datastd,self.max_datastd])
                    ievents = np.unique(bodywaves.event_idx[dataset])
                    if self.bw_hyperparameter_level=='event':
                        for ievent in ievents:
                            self.likedist_updates.append(('bodywaves',dataset,ievent,'stadev'))
                    else: # all events have the same standard deviation
                        self.likedist_updates.append(('bodywaves',dataset,(),'stadev'))
                    # outlier model
                    if self.likedist == 'outliers':
                        self.foutlier[dataset] = np.ones(len(bodywaves.data[dataset]))*0.1
                        vpmod,vsmod = bwm.get_iasp91(self.inverse_z_scaling(self.gridpoints[:,2]))
                        ttimes_syn = bwm.forward_module(
                            1./vpmod,1./vsmod,bodywaves.A,
                            ttimes_base=bodywaves.ttimes_base)
                        residuals = bodywaves.data[dataset] - bwm.demean_ttimes(ttimes_syn[dataset],bodywaves.M[dataset])
                        self.widthoutlier[dataset] = np.max(residuals)-np.min(residuals)
                        if self.bw_hyperparameter_level=='event':
                            for ievent in ievents:
                                self.likedist_updates.append(('bodywaves',dataset,ievent,'foutlier'))
                        else: # all events have the same standard deviation
                            self.likedist_updates.append(('bodywaves',dataset,(),'foutlier'))
            for dataset in bodywaves.data:
                self.collection_datastd[dataset] = []

        # Initialize Surfacewave dataset
        if surfacewaves.isactive:
            self.maxlendata = 0
            self.loglikelihood_dict_sw = {}
            self.residual_dict_sw = {}
            for dataset in surfacewaves.data:
                self.loglikelihood_dict_sw[dataset] = {}
                self.residual_dict_sw[dataset] = {}
                for period in surfacewaves.data[dataset]:
                    ndata = len(surfacewaves.data[dataset][period])
                    if ndata > self.maxlendata:
                        self.maxlendata = ndata
                    self.loglikelihood_dict_sw[dataset][period] = np.ones(ndata)*np.nan
                    self.residual_dict_sw[dataset][period] = np.ones(ndata)*np.nan
            self.likedist_normcoeffs = deepcopy(surfacewaves.datastd)
            self.datastd.update(deepcopy(surfacewaves.datastd))
            # data errors scale with the interstation distance
            if self.error_type == 'relative':
                self.std_intercept = {}
                self.std_scaling = {}
                for dataset in surfacewaves.data:
                    self.std_intercept[dataset] = {}
                    self.std_scaling[dataset] = {}
                    if self.sw_hyperparameter_level=='dataset':
                        self.likedist_updates.append(('surfwaves',dataset,None,'stadev_intercept'))
                        self.likedist_updates.append(('surfwaves',dataset,None,'stadev_scaling'))
                    for period in surfacewaves.data[dataset]:
                        self.std_intercept[dataset][period] = np.mean([
                            self.max_datastd,self.min_datastd])
                        self.std_scaling[dataset][period] = 0.
                        if self.sw_hyperparameter_level=='period':
                            self.likedist_updates.append(('surfwaves',dataset,period,'stadev_intercept'))
                            self.likedist_updates.append(('surfwaves',dataset,period,'stadev_scaling'))
                        self.datastd[dataset][period] = np.ones(len(surfacewaves.data[dataset][period])) * self.std_intercept[dataset][period]
            # station errors have no linear scaling
            elif self.error_type == 'absolute':
                for dataset in surfacewaves.data:
                    if self.sw_hyperparameter_level == 'dataset':
                        self.likedist_updates.append(('surfwaves',dataset,None,'stadev'))
                    for period in surfacewaves.data[dataset]:
                        self.datastd[dataset][period] = np.array([
                            np.mean([self.max_datastd,self.min_datastd])])
                        if self.sw_hyperparameter_level == 'period':
                            self.likedist_updates.append(('surfwaves',dataset,period,'stadev'))
            self.phttime_predictions = {}
            for dataset in surfacewaves.datasets:
                self.phttime_predictions[dataset] = {}
                self.collection_datastd[dataset] = {}
                for period in surfacewaves.data[dataset]:
                    self.phttime_predictions[dataset][period] = np.zeros(np.shape(surfacewaves.data[dataset][period]))
                    self.collection_datastd[dataset][period] = []
            if self.likedist == 'outliers':
                for dataset in surfacewaves.data:
                    self.foutlier[dataset] = {}
                    self.widthoutlier[dataset] = {}
                    residuals = np.array([])
                    for period in surfacewaves.data[dataset]:
                        self.foutlier[dataset][period] = np.array([0.])
                        path_dists = np.sqrt(np.sum((surfacewaves.sources[dataset][period]-
                                                      surfacewaves.receivers[dataset][period])**2,axis=1))
                        res = (path_dists/surfacewaves.data[dataset][period] - 
                               np.mean(path_dists/surfacewaves.data[dataset][period]))
                        if self.sw_hyperparameter_level=='dataset':
                            residuals = np.append(residuals,res)
                        else:
                            self.widthoutlier[dataset][period] = np.max(res)-np.min(res)
                            self.likedist_updates.append(('surfwaves',dataset,period,'foutlier'))
                    if self.sw_hyperparameter_level=='dataset':
                        self.likedist_updates.append(('surfwaves',dataset,None,'foutlier'))
                        for period in surfacewaves.data[dataset]:
                           self.widthoutlier[dataset][period] = np.max(residuals)-np.min(residuals) 
            elif self.likedist == 'students_t':
                # if degfree -> infinity, the t distribution becomes a normal distribution
                # for small degfree (e.g. 1), it has a longer tail, i.e. more outlier resistant
                self.degfree = {}
                for dataset in surfacewaves.data:
                    self.degfree[dataset] = {}
                    if self.sw_hyperparameter_level == 'dataset':
                        self.likedist_updates.append(('surfwaves',dataset,None,'degfree'))
                    for period in surfacewaves.data[dataset]:
                        self.degfree[dataset][period] = 1000. # close to a normal distribution
                        if self.sw_hyperparameter_level=='period':
                            self.likedist_updates.append(('surfwaves',dataset,period,'degfree'))
            
            # for info purposes. Check whether the measured phase velocities
            # can be produced with the chosen Vs prior range.
            if self.chain_no==1:
                self.check_prior_range(surfacewaves)
              
            # Optional: Set a prior on the allowed phase velocities at each period
            phasevel_prior=False
            if phasevel_prior:
                self.phvel_ray_prior,self.phvel_lov_prior = swm.get_phasevel_dataranges(surfacewaves)
            else:
                self.phvel_ray_prior = None
                self.phvel_lov_prior = None
    
        
        if True and bodywaves.isactive and surfacewaves.isactive:
            print("Warning, setting weights!")
            Nsurf = 0
            for dataset in surfacewaves.data:
                for period in surfacewaves.data[dataset]:
                    if period <= 30:
                        continue
                    Nsurf += len(surfacewaves.data[dataset][period])
            Nbody = 0
            for dataset in bodywaves.data:
                Nbody += len(bodywaves.data[dataset])
            print("Nsurf (>30s) =",Nsurf,"Nbody =",Nbody)
            self.relweight_bodywaves = Nsurf/Nbody
            self.relweight_surfwaves = 1#Nbody/Nsurf
            print("relative weight of bodywaves:",self.relweight_bodywaves)
            print("relative weight of surfacewaves:",self.relweight_surfwaves)
        else:
            self.relweight_bodywaves = 1.#415.
            #print("Warning: bodywaveweight",self.relweight_bodywaves)
            self.relweight_surfwaves = 1
                                    
        self.limit_gradients = False
        if self.limit_gradients:
            # gradients are given as dVs/dz (unscaled) [km/s / km]
            mingrad = -0.5
            maxgrad = 1.0
            print("Warning: Experimentally limiting the velocity gradients between layers!")
        else:
            mingrad = None
            maxgrad = None

        self.separate_mantle = False
        if self.separate_mantle:
            self.mantle_vel_limit = 4.1
            print("Warning: Experimentally restricting crust-mantle transitions. " +
                  "No cells with crustal velocities are allowed below cells with " +
                  f"mantle velocities. Velocity threshold: {self.mantle_vel_limit}km/s.")
        
            
        # Algorithm tries not to fit data beyond the minerror
        # can also be set to None
        self.minerror = None
        if self.minerror is not None:
            print("Setting minimum error for the measurements to",self.minerror,"km/s.")
        
        
        # create a convex hull around the region covered by data
        convexhull = self.get_data_coverage_convexhull(targets)
            
        if self.parameterization == 'voronoi':
            
            cnt = 0
            while True:
                                
                if cnt>0: # try adding more points
                    self.init_no_points = np.min([self.nmax_points,int(self.init_no_points*1.01)+10])                
                if cnt>100:
                    raise Exception(f"chain {self.chain_no}: not able to find a good starting model. Try increasing the z_scaling or removing restrictions such as the phasevel prior.")
                
                self.para = Parameterization.voronoi_cells(
                    self.gridpoints,self.shape,convexhull,self.minvs,self.maxvs,
                    self.vs2vpvs,self.init_no_points,moho_depths=self.moho_depths,
                    crustal_model=(self.vp_crust,self.vs_crust),
                    inverse_z_scaling=self.inverse_z_scaling,
                    vpvsmin=self.vpvsmin,vpvsmax=self.vpvsmax,mingrad=mingrad,maxgrad=maxgrad)
                if self.init_vel_points not in ['random','random_restricted']:
                    startmodel_function = interp1d(self.z_scaling(self.startmodel_depth),self.startmodel_vs)
                    if self.limit_gradients:
                        gradients = np.diff(startmodel_function(self.z_scaled)) / np.diff(self.z)
                        if (gradients > maxgrad).any() or (gradients < mingrad).any():
                            raise Exception("The given starting model is incompatible with the defined gradient limits.")
                else:
                    startmodel_function = self.get_startmodel(
                        surfacewaves,mingrad=mingrad,maxgrad=maxgrad)

                # # #
                if False: # for testing
                    import pyproj
                    print("Warning, setting a 3D startmodel")
                    fullmod = np.loadtxt('./fullmod_syn.txt')
                    p = pyproj.Proj(self.projection)
                    fullmod_x,fullmod_y = p(fullmod[:,1],fullmod[:,0])
                    fullmod[:,0] = fullmod_x/1000.
                    fullmod[:,1] = fullmod_y/1000.
                    zz = self.inverse_z_scaling(self.para.points[:,2])
                    tree = KDTree(fullmod[:,:3])
                    nndist,nnidx = tree.query(np.column_stack((self.para.points[:,:2],zz)))
                    init_vels = fullmod[nnidx,4]                    
                else:
                # # #


                    init_vels = startmodel_function(self.para.points[:,2])
                # search parameters are relative parameters between 0 and 1
                # meaning between vsmin and vsmax
                init_params = (init_vels - self.para.vsmin(self.para.points[:,2])) / self.para.vs_range(self.para.points[:,2])
                if np.any(init_params<0) or np.any(init_params>1):
                    cnt += 1
                    continue
                self.para.get_model(params=init_params,vpvs=self.vs2vpvs)
                if self.limit_gradients:
                    mingrad_model,maxgrad_model = self.para.get_vertical_gradient(self.para.vsfield)
                    if mingrad_model<self.para.mingrad or maxgrad_model>self.para.maxgrad:
                        cnt += 1
                        continue
                if surfacewaves.isactive:
                    valid,idx_2d,prop_phslo_ray,prop_phslo_lov,nmods = swm.get_phasevel_maps(
                        self.z,surfacewaves.periods,self.shape,
                        vs_field=self.para.vsfield,vpvs_field=self.para.vpvsfield,
                        phvel_prior_ray=self.phvel_ray_prior,phvel_prior_lov=self.phvel_lov_prior)
                    if not valid:
                        cnt += 1
                        continue
                self.para.parameters = init_params
                if self.flexible_vpvs:
                    self.para.vpvs = self.vs2vpvs(init_vels)

                break
            
            
            # double check whether starting model is valid
            if self.limit_gradients or self.separate_mantle:
                mingrad,maxgrad = self.para.get_vertical_gradient(self.para.vsfield)
                if mingrad<self.para.mingrad or maxgrad>self.para.maxgrad:
                    raise Exception("gradient not okay!")
                tri = Delaunay(self.para.points)
                intptr,neighbor_indices = tri.vertex_neighbor_vertices
                for pnt_idx in range(len(self.para.points)):
                    idx_neighbor_points = neighbor_indices[intptr[pnt_idx]:intptr[pnt_idx+1]]
                    if self.limit_gradients:
                        dvs = (init_vels[idx_neighbor_points] - init_vels[pnt_idx])
                        dz = (self.inverse_z_scaling(self.para.points[idx_neighbor_points,2]) - 
                              self.inverse_z_scaling(self.para.points[pnt_idx,2]))
                        gradients = dvs[dz!=0.] / dz[dz!=0.]
                        if (gradients < mingrad).any() or (gradients > maxgrad).any():
                            raise Exception("needs to be fixed!")
                    if self.separate_mantle:
                        idx_above = idx_neighbor_points[self.para.points[idx_neighbor_points,2]<self.para.points[pnt_idx,2]]
                        if (init_vels[idx_above] >= self.mantle_vel_limit).any() and init_vels[pnt_idx] < self.mantle_vel_limit:
                            raise Exception("needs to be fixed (separate mantle)!")
                    
            # could also add more "velocity_update" steps?
            self.actions = ["birth","death","move","velocity_update","velocity_update"]
                                 
            
        elif self.parameterization == 'profiles':
            
            self.para = Parameterization.profiles(
                self.gridpoints,self.shape,convexhull,self.minvs,self.maxvs,
                self.vs2vpvs,self.init_no_points)
            
            while True:
                for i in range(len(self.para.parameters)):
                    profile = self.para.parameters[i]
                    fu = interp1d(profile[:,0],profile[:,1],kind='nearest',
                                  bounds_error=False,fill_value='extrapolate')
                    vsprofile = fu(self.z)
                    fu = interp1d(profile[:,0],profile[:,2],kind='nearest',
                                  bounds_error=False,fill_value='extrapolate')
                    vpprofile = vsprofile*fu(self.z)
                    slowness_ray, slowness_lov = swm.profile_to_phasevelocity(
                        self.z,vsprofile,vpprofile,surfacewaves.periods,
                        self.phvel_ray_prior,self.phvel_lov_prior)
                    if (slowness_ray==0.).any() or (slowness_lov==0.).any():
                        self.para.parameters[i] = self.para.get_new_profile()
                        break
                else:
                    break
                
            # could also add more "velocity_update" steps?
            self.actions = ["birth","death","move","velocity_update"]
            
        elif self.parameterization == 'wavelets':
            
            valid = False
            while not valid:
                reference_model = None
                if surfacewaves.isactive:
                    if phasevel_prior:
                        startmodel_function = self.get_startmodel(surfacewaves)
                        reference_model = startmodel_function(self.z)              
                self.para = Parameterization.wavelets(
                    self.gridpoints,self.shape,convexhull,self.minvs,self.maxvs,
                    self.vs2vpvs,init_no_coeffs=0,#params.init_no_points,
                    startmodel=reference_model)
                check_mod = self.para.get_model()
                if not check_mod:
                    continue
                if surfacewaves.isactive:
                    valid,idx_2d,prop_phslo_ray,prop_phslo_lov,nmods = swm.get_phasevel_maps(
                        self.z,surfacewaves.periods,self.shape,
                        vs_field=self.para.vsfield,vpvs_field=self.para.vpvsfield)
            self.nmax_points = 0
            for level in range(len(self.para.tree)):
                self.nmax_points += np.product(self.para.tree[:level+1])
                
            self.propvelstd_pointupdate = self.para.coeff_range[0]/10.
            
            # could also add more "velocity_update" steps?
            self.actions = ["birth","death","velocity_update"]
            
        else:
            
            raise Exception("parameterization type unknown, choose between "+
                            "voronoi, profiles and wavelets.")
          
        if self.parameterization=='voronoi':
            self.propstd_depthlevels = np.cumsum(np.linspace(
                self.z_scaled[0],2*self.z_scaled[-2]/10.,10))
        elif self.parameterization=='wavelets':
            self.propstd_depthlevels = np.arange(self.para.levels)
        else:
            self.propstd_depthlevels = np.array([0])
            
        # if the proposal standard deviation is not fixed, make it dependent
        # on the depth level
        if self.adaptive_proposalstd:
            self.propvelstd_pointupdate = np.ones(len(self.propstd_depthlevels))*self.propvelstd_pointupdate
            if self.propvelstd_dimchange != 'uniform':
                self.propvelstd_dimchange = np.ones(len(self.propstd_depthlevels))*self.propvelstd_dimchange
            #self.propstd_birth = np.ones((len(self.propstd_depthlevels),30))*params.propvelstd_dimchange
            #self.propstd_death = np.ones((len(self.propstd_depthlevels),30))*params.propvelstd_dimchange            
            self.propmovestd = np.ones(len(self.propstd_depthlevels))*self.propmovestd
        
        
        if False:
            #from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.X.flatten(),self.Y.flatten(),self.Z.flatten(), c=self.para.vsfield, marker='o')
            plt.show()
            
        if np.isnan(self.para.vsfield).any():
            raise Exception("There should be no nan value in the velocity field")
        if np.isnan(self.para.vpvsfield).any():
            raise Exception("There should be no nan value in the vpvs field")
            
        # initialize average model        
        self.average_vs = self.para.vsfield.copy()
        self.vssumsquared = self.average_vs.copy()**2
        self.average_vpvs = self.vs2vpvs(self.para.vsfield)
        self.vpvssumsquared = self.average_vpvs.copy()
        self.average_vp = self.average_vs*self.average_vpvs
        self.vpsumsquared = (self.average_vs*self.average_vpvs)**2
        self.average_model_counter = 1
        
        if self.flexible_vpvs:
            self.actions += ["vpvsratio_update"]
        
        self.accepted_steps = {}
        self.rejected_steps = {}
        self.avg_acceptance_rate = {}
        self.acceptance_rate = {}
        self.meantimes = {}
                
        if self.error_type != 'fixed':
            self.actions += ["hyperparameter_update"]
            
        for action in np.unique(self.actions):    
            self.accepted_steps[action] = 0
            self.rejected_steps[action] = 1
            self.avg_acceptance_rate[action] = np.zeros(100)
            self.avg_acceptance_rate[action][::2] = 1
            self.acceptance_rate[action] = np.zeros((len(self.propstd_depthlevels),100))
            self.acceptance_rate[action][:,::2] = 1
            self.meantimes[action] = 0.
        
        self.modellengths = []
        
        # if user wants to visualize progress:
        if self.visualize:
            self.visualize_progress(surfacewaves,initialize=True)
        
        self.anisotropic = False
        
        # calculate initial residual and likelihood
        self.update_current_likelihood(targets)

        self.cleanup(targets) # set all the proposal variables to None
        

    def resume(self,targets,prior,params):
        
        #self.selfcheck=True
        #print("warning: selfcheck turned on!")
        #if self.parameterization=='wavelets':
        #    for i in range(10):
        #        self.para.update_maxlevel()
                       
        self.target_iterations = self.update_parameter(self.target_iterations,params.target_iterations,
                                                       "parameter(target iterations)")
        self.nburnin = self.update_parameter(self.nburnin, params.nburnin,
                                             "parameter(nburnin)")
        
        with open(self.logfile,"a") as f:
            f.write(f"\nResuming at {self.total_steps} iterations. Calculating additonal {self.target_iterations-self.total_steps} iterations.\n")
        
        # can cause problems if the new boundaries are narrower than the old
        # ones so that some values ar not within the prior bounds anymore.
        if prior.min_datastd>self.min_datastd:
            print(f"warning: changing lower prior boundary on the data std from {self.min_datastd} to {prior.min_datastd}. This can cause problems if current datastd is below the new boundary.")
        self.min_datastd = self.update_parameter(self.min_datastd, prior.min_datastd,
                                                 "prior(min data std)")
        if prior.max_datastd<self.max_datastd:
            print(f"warning: changing upper prior boundary on the data std from {self.max_datastd} to {prior.max_datastd}. This can cause problems if current datastd is above the new boundary.")
        self.max_datastd = self.update_parameter(self.max_datastd, prior.max_datastd,
                                                 "prior(max data std)")       
        self.nmin_points = self.update_parameter(self.nmin_points, prior.nmin_points,
                                                 "prior(min no points)")
        self.nmax_points = self.update_parameter(self.nmax_points, prior.nmax_points,
                                                 "prior(max no points)")
        self.flexible_vpvs = self.update_parameter(self.flexible_vpvs, prior.flexible_vpvs,
                                                   "prior(flexible vpvs ratio)")
        if self.flexible_vpvs and "vpvsratio_update" not in self.actions:
            self.actions += ["vpvsratio_update"]
    
        self.propsigmastd = self.update_parameter(self.propsigmastd, params.propsigmastd,
                                                  "parameter(proposal std data std)")
        if not self.adaptive_proposalstd:
            if params.propvelstd == "adaptive":
                self.adaptive_proposalstd = self.update_parameter(self.adaptive_proposalstd, True,
                                                                   "parameter(adaptive proposal std)")
                self.propvelstd_dimchange = np.ones(len(self.propstd_depthlevels))*self.propvelstd_dimchange
                self.propvelstd_pointupdate = np.ones(len(self.propstd_depthlevels))*self.propvelstd_pointupdate
                self.propmovestd = np.ones(len(self.propstd_depthlevels))*self.propmovestd
            else:
                self.propvelstd_dimchange = self.update_parameter(self.propvelstd_dimchange, params.propvelstd_dimchange,
                                                              "parameter(proposal std birth/death)")
                self.propvelstd_pointupdate = self.update_parameter(self.propvelstd_pointupdate, params.propvelstd_pointupdate,
                                                                "parameter(proposal std velocity update)")
                self.propmovestd = self.update_parameter(self.propmovestd, params.propmovestd,
                                                     "parameter(proposal std move)")
           
        # rather not change the temperatures when resuming
        #self.temperature = self.update_parameter(self.temperature, params.temperature,
        #                                         "parameter(temperature)")

        self.update_paths_interval = self.update_parameter(self.update_paths_interval, params.update_paths_interval,
                                                           "parameter(update path interval)")

        self.collect_step = self.update_parameter(self.collect_step, params.collect_step,
                                                  "parameter(model collect interval)")
        self.print_stats_step = self.update_parameter(self.print_stats_step, params.print_stats_step,
                                                      "parameter(print stats interval)")

       # include 2 psi anisotropy
        self.anisotropic = self.update_parameter(self.anisotropic, params.anisotropic,
                                                 "parameter(anisotropic search)")
        if self.anisotropic:
            self.propstd_anisoamp = self.update_parameter(self.propstd_anisoamp, params.propstd_anisoamp,
                                                          "parameter(proposal std anisotropic amplitude)")
            self.propstd_anisodir = self.update_parameter(self.propstd_anisodir, params.propstd_anisodir,
                                                          "parameter(proposal std anisotropic direction)")
            self.aniso_ampmin = self.update_parameter(self.aniso_ampmin, prior.aniso_ampmin,
                                                      "prior(min aniso amplitude)")
            if self.aniso_ampmin != 0.:
                print("Warning: minimum anisotropic amplitude is set to zero")
                self.aniso_ampmin = 0.
                # could otherwise cause problems if I start with a purely isotropic search
            self.aniso_ampmax = self.update_parameter(self.aniso_ampmax, prior.aniso_ampmax,
                                                      "parameter(prior(max aniso amplitude)")
        loglike_before = self.loglikelihood_current
        self.update_current_likelihood(targets)
        print("loglike of chain:",loglike_before,"loglike re-calculated (should be identical):",self.loglikelihood_current)
        
    def update_parameter(self,oldparam,newparam,varname):
        if oldparam != newparam:
            with open(self.logfile,"a") as f:
                f.write(f"    Updating {varname}: {oldparam} -> {newparam}\n")
        return newparam  
            

         
    
    def propose_jump(self,targets,action='random'):
        
        self.t0 = time.time()
        self.total_steps += 1
        
        for target in targets:
            if target.type == 'surfacewaves':
                surfacewaves = target
            elif target.type == 'bodywaves':
                bodywaves = target
                
        self.mindepth = None
        if self.mindepth is not None and self.total_steps % 10000 == 0:
            print("Warning: mindepth function turned on!")

        if self.parameterization == 'wavelets':
            if self.total_steps%100000==0:
                self.para.update_maxlevel()
                self.nmax_points = 0
                for level in range(len(self.para.tree)):
                    self.nmax_points += np.product(self.para.tree[:level+1])

        #if self.total_steps == int(self.nburnin*(3/4.)):
        #    self.limit_gradients=False
        #    print("limit gradients is now OFF")                
                  
        if action=='random':
            action = np.random.choice(self.actions)
            #while action == "vpvsratio_update" and self.total_steps < int(self.nburnin/100):
             #   action = np.random.choice(self.actions)
            while not 'update' in action and self.total_steps < int(self.nburnin/40):
                action = np.random.choice(self.actions)
                   
        #print(action))
        
        
        #if self.outlier_model:
        #    if self.total_steps < 2000 or self.total_steps%1000==0:
        #        self.update_widthoutlier()
        

        # # # # HYPER-PARAMETER UPDATE # # # # #
        if action == 'hyperparameter_update':
            
            randidx = np.random.randint(0,len(self.likedist_updates))
            update = self.likedist_updates[randidx]
            wavetype,dataset,subset,update_type = update
            if update[0]=='bodywaves':
                if type(subset)==int: # 'period' is the event index
                    # events have invididual standard deviations
                    subset = bodywaves.event_idx[dataset]==subset
                    
            # proposed values are identical to previous values at the beginning
            prop_datastd = self.datastd
            prop_foutlier = self.foutlier
            prop_degfree = self.degfree
            prop_std_scaling = self.std_scaling
            prop_std_intercept = self.std_intercept
            
            if update[0]=='surfwaves' and self.sw_hyperparameter_level=='dataset':
                keys = list(surfacewaves.data[dataset].keys())
            else:
                keys = [subset]
                
            # update hyperparameters
            if update_type == 'stadev': # can be body or surface waves
                prop_datastd = deepcopy(self.datastd)
                dstadev = np.random.normal(loc=0.0,scale=self.propsigmastd)
                for key in keys:
                    prop_datastd[dataset][key] += dstadev
                    
            elif update_type == 'stadev_intercept': # only for surface waves
                prop_intercept = (self.std_intercept[dataset][keys[0]] + 
                                  np.random.normal(loc=0.0,scale=self.propsigmastd))
                if prop_intercept <= 0.:
                    self.rejected_steps[action] += 1
                    self.collect_model(targets,action)
                    return 0,self.loglikelihood_current
                prop_datastd = deepcopy(self.datastd)
                prop_std_intercept = deepcopy(self.std_intercept)
                for key in keys:
                    prop_std_intercept[dataset][key] = prop_intercept
                    prop_datastd[dataset][key] = (
                        self.std_scaling[dataset][key] * 
                        surfacewaves.path_dists[dataset][key] + 
                        prop_std_intercept[dataset][key])
                
            elif update_type == 'stadev_scaling': # only for surface waves
                prop_scaling_factor = (
                    self.std_scaling[dataset][keys[0]] + 
                    np.random.normal(loc=0.0,scale=self.propsigmastd/np.mean(
                        surfacewaves.path_dists[dataset][keys[0]])))
                if prop_scaling_factor <= 0.:
                    self.rejected_steps[action] += 1
                    self.collect_model(targets,action)
                    return 0,self.loglikelihood_current
                prop_datastd = deepcopy(self.datastd)
                prop_std_scaling = deepcopy(self.std_scaling)
                for key in keys:
                    prop_std_scaling[dataset][key] = prop_scaling_factor
                    prop_datastd[dataset][key] = (
                        prop_std_scaling[dataset][key] * 
                        surfacewaves.path_dists[dataset][key] + 
                        self.std_intercept[dataset][key])
                
            elif update_type == 'foutlier':
                dfoutlier = np.random.normal(loc=0.0,scale=0.01)
                if ((self.foutlier[dataset][keys[0]]+dfoutlier < 0.).any() or 
                    (self.foutlier[dataset][keys[0]]+dfoutlier > 0.8).any() ):
                    self.rejected_steps[action] += 1
                    self.collect_model(targets,action)
                    return 0,self.loglikelihood_current
                prop_foutlier = deepcopy(self.foutlier)
                for key in keys:
                    prop_foutlier[dataset][key] += dfoutlier
                
            elif update_type == 'degfree':
                ddegfree = np.random.uniform(0.5,1.5)
                if self.degfree[dataset][keys[0]]*ddegfree <= 0.:
                    self.rejected_steps[action] += 1
                    self.collect_model(targets,action)
                    return 0,self.loglikelihood_current     
                prop_degfree = deepcopy(self.degfree)
                for key in keys:
                    prop_degfree[dataset][key] *= ddegfree
                
            # check that the standard deviation is within the prior bounds
            if 'stadev' in update_type:
                if ((np.min(prop_datastd[dataset][keys[0]]) < self.min_datastd).any() or
                    (np.max(prop_datastd[dataset][keys[0]]) > self.max_datastd).any() or
                    (prop_datastd[dataset][keys[0]] <= 0.0).any()):
                    self.rejected_steps[action] += 1
                    self.collect_model(targets,action)
                    return 0,self.loglikelihood_current

            # udate likelihoods
            # the normalization coefficients change only during hyperparameter updates
            if surfacewaves.isactive:
                (res_sw,ll_sw,self.prop_residual_dict_sw,
                 self.prop_loglike_dict_sw,prop_norm_coeffs) = swm.get_loglikelihood(
                    self.loglikelihood_dict_sw,self.residual_dict_sw,
                    self.likedist_normcoeffs,surfacewaves.data,
                    self.phttime_predictions,surfacewaves.path_dists,
                    prop_datastd,update=(dataset,keys),
                    foutlier=prop_foutlier,widthoutlier=self.widthoutlier,
                    degfree=prop_degfree,distribution=self.likedist,
                    norm=self.misfit_norm,selfcheck=self.selfcheck)
                ll_sw *= self.relweight_surfwaves
            else:
                res_sw = ll_sw = 0.
            if bodywaves.isactive:
                res_bw,ll_bw = bwm.loglikelihood(
                    bodywaves.data,self.p_ttimes,prop_datastd,
                    bodywaves.M,norm=self.misfit_norm,
                    distribution=self.likedist,foutlier=prop_foutlier,
                    widthoutlier=self.widthoutlier)
                ll_bw *= self.relweight_bodywaves
            else:
                res_bw = ll_bw = 0.
            loglikelihood_prop = ll_sw+ll_bw
            
            # accept or reject model
            log_acceptance_prob = loglikelihood_prop - self.loglikelihood_current
            if np.log(np.random.rand(1)) <= 1./self.temperature * log_acceptance_prob:     
                #accepted
                self.loglikelihood_current = loglikelihood_prop
                self.loglike_curr_bw = ll_bw
                self.loglike_curr_sw = ll_sw
                self.residual_current = res_sw+res_bw
                if surfacewaves.isactive:
                    self.loglikelihood_dict_sw = self.prop_loglike_dict_sw
                    self.residual_dict_sw = self.prop_residual_dict_sw
                    self.likedist_normcoeffs = prop_norm_coeffs
                self.foutlier = prop_foutlier
                self.degfree = prop_degfree
                self.datastd = prop_datastd
                self.std_scaling = prop_std_scaling
                self.std_intercept = prop_std_intercept
                self.collect_model(targets,action)
                self.meantimes[action] += time.time()-self.t0 
                self.accepted_steps[action] += 1
                return 1,self.loglikelihood_current
            else:
                self.rejected_steps[action] += 1
                self.collect_model(targets,action)
                return 0,self.loglikelihood_current
            
       
        # # # # VP/VS RATIO UPDATE # # # # 
        if action == 'vpvsratio_update':
            
            idx = np.random.randint(0,len(self.para.points))
            zidx = np.abs(self.para.points[idx,2]-self.propstd_depthlevels).argmin()
            pnt_xy = self.para.points[idx,:2]
            
            prop_std = 0.04 # hard coded currently
            
            check_update = self.para.vpvs_ratio_update(idx,prop_std)
            if not check_update:
                self.reject_model(targets,action,zidx,pnt_xy)
                return 0,self.loglikelihood_current
            
            check_mod = self.para.get_model()
            if not check_mod:
                self.reject_model(targets,action,zidx,pnt_xy)
                return 0,self.loglikelihood_current
            
            valid = self.update_model_predictions(targets,self.para.idx_mod_gpts)
            if not valid:
                self.reject_model(targets,action,zidx,pnt_xy)
                return 0,self.loglikelihood_current
            
            if self.check_acceptance():
                self.accept_model(targets,action,zidx,pnt_xy)
                return 1,self.loglikelihood_current
            else:
                self.reject_model(targets,action,zidx,pnt_xy)                
                return 0,self.loglikelihood_current             
        
        
        
        if action == 'velocity_update':
                      
            if self.mindepth is not None:
                idx = np.random.choice(np.where(self.para.points[:,2] > 
                                                self.mindepth)[0])
            else:
                idx = np.random.randint(0,len(self.para.points))
                
            pnt_xy,pnt_z = self.para.get_mod_point_coords(idx=idx)
            zidx = np.abs(pnt_z-self.propstd_depthlevels).argmin()
                
            if self.adaptive_proposalstd:
                propvelstd_pointupdate = self.propvelstd_pointupdate[zidx]
            else:
                propvelstd_pointupdate = self.propvelstd_pointupdate
            
            for second_try in [False,True]: #delayed rejection

                # if first move was rejected, try a second move with smaller std
                if second_try:
                    propvelstd = propvelstd_pointupdate * np.random.rand()
                else:
                    propvelstd = propvelstd_pointupdate

                check_update = self.para.parameter_update(idx,propvelstd)
                if not check_update:
                    self.reject_model(targets,action,zidx,pnt_xy)
                    return 0,self.loglikelihood_current
                                    
                if self.anisotropic:
                    print("not implemented")
                else:
                    check_mod = self.para.get_model()

                if not check_mod:
                    self.reject_model(targets,action,zidx,pnt_xy)
                    return 0,self.loglikelihood_current
                    
                if self.limit_gradients:
                    mingrad,maxgrad = self.para.get_vertical_gradient(self.para.vsfield)
                    if mingrad<self.para.mingrad or maxgrad>self.para.maxgrad:
                        self.reject_model(targets,action,zidx,pnt_xy)
                        return 0,self.loglikelihood_current
                    
                if self.selfcheck:
                    # double check
                    self.test_vs_field = self.para.vsfield.copy()
                    self.test_vpvs = self.para.vpvsfield.copy()
                    self.para.get_model(recalculate=True)
                    if (np.abs(self.para.vsfield-self.test_vs_field) > 1e-10).any():
                        print(idx)
                        raise Exception("error velocity update")
                    
                valid = self.update_model_predictions(targets,self.para.idx_mod_gpts)
                if not valid:
                    self.reject_model(targets,action,zidx,pnt_xy)
                    return 0,self.loglikelihood_current
                                
                if self.check_acceptance(delayed=second_try):
                    self.accept_model(targets,action,zidx,pnt_xy)
                    return 1,self.loglikelihood_current
                else:
                    if self.delayed_rejection and not(second_try):
                        self.para.reject_mod()
                        continue
                    else:   
                        self.reject_model(targets,action,zidx,pnt_xy)                
                        return 0,self.loglikelihood_current     
        

   
        # # # # BIRTH # # # # #
        if action=='birth':
                        
            if len(self.para.points)+1 > self.nmax_points:
                self.reject_model(targets,action,0,(0,0))
                return 0,self.loglikelihood_current
                            
            # add a new point at a random location
            birth_check = self.para.add_point(
                anisotropic=self.anisotropic,
                birth_prop=(self.propstd_depthlevels,self.propvelstd_dimchange))
            
            pnt_xy,pnt_z = self.para.get_mod_point_coords()
            zidx = np.abs(pnt_z-self.propstd_depthlevels).argmin()
            
            # reject model if the new parameter is outside the prior range or
            # if there is no free node in the tree (wavelet option)
            if not birth_check:
                self.reject_model(targets,action,zidx,pnt_xy)
                return 0,self.loglikelihood_current

            check_mod = self.para.get_model()
            if not check_mod:
                self.reject_model(targets,action,zidx,pnt_xy)
                return 0,self.loglikelihood_current
            
            if self.limit_gradients:
                mingrad,maxgrad = self.para.get_vertical_gradient(self.para.vsfield)
                if mingrad<self.para.mingrad or maxgrad>self.para.maxgrad:
                    self.reject_model(targets,action,zidx,pnt_xy)
                    return 0,self.loglikelihood_current

            if self.selfcheck:
                # double check
                test_vs_field = self.para.vsfield.copy()
                test_vpvs = self.para.vpvsfield.copy()
                self.para.get_model(recalculate=True)
                if (np.abs(self.para.vsfield-test_vs_field) > 1e-10).any():
                    print("birth cell:",self.para.points[-1])
                    raise Exception("birth")
                
            valid = self.update_model_predictions(targets,self.para.idx_mod_gpts)
            if not valid:
                self.reject_model(targets,action,zidx,pnt_xy)
                return 0,self.loglikelihood_current
    
            if self.check_acceptance():
                # model accepted
                self.accept_model(targets,action,zidx,pnt_xy)
                return 1,self.loglikelihood_current
            else:
                self.reject_model(targets,action,zidx,pnt_xy)
                return 0,self.loglikelihood_current



# DEATH ####################
        if action == 'death':
            
            # minimum 4 points are needed for the 3D triangulation
            if (len(self.para.points)-1 < self.nmin_points or 
                len(self.para.points)-1 < 4):    
                self.reject_model(targets,action,0,(0,0))
                return 0,self.loglikelihood_current
           
            # remove a randomly chosen point
            remove_check = self.para.remove_point(
                anisotropic=self.anisotropic,
                death_prop=(self.propstd_depthlevels,self.propvelstd_dimchange))
            
            pnt_xy,pnt_z = self.para.get_mod_point_coords()
            zidx = np.abs(pnt_z-self.propstd_depthlevels).argmin()
            # check whether it was possible to remove a point
            if not remove_check:
                self.reject_model(targets,action,zidx,pnt_xy)
                return 0,self.loglikelihood_current

            if self.anisotropic:
                print("not implemented")
            else:
                check_mod = self.para.get_model()
                
            if not check_mod:
                self.reject_model(targets,action,zidx,pnt_xy)
                return 0,self.loglikelihood_current     
            
            if self.limit_gradients:
                mingrad,maxgrad = self.para.get_vertical_gradient(self.para.vsfield)
                if mingrad<self.para.mingrad or maxgrad>self.para.maxgrad:
                    self.reject_model(targets,action,zidx,pnt_xy)
                    return 0,self.loglikelihood_current

            if self.selfcheck:
                test_vs_field = self.para.vsfield.copy()
                test_vpvs = self.para.vpvsfield.copy()
                self.para.get_model(recalculate=True)
                if (self.para.vsfield-test_vs_field > 1e-10).any():
                    self.para.plot(idx_mod_gpts=self.para.idx_mod_gpts)
                    raise Exception("death")
                if np.isnan(self.para.vsfield).any():
                    raise Exception(f"nan in vel mod during {action}")
            
            valid = self.update_model_predictions(targets,self.para.idx_mod_gpts)
            if not valid:
                self.reject_model(targets,action,zidx,pnt_xy)
                return 0,self.loglikelihood_current
    
            if self.check_acceptance():
                # model accepted
                self.accept_model(targets,action,zidx,pnt_xy)
                return 1,self.loglikelihood_current
            else:
                self.reject_model(targets,action,zidx,pnt_xy)
                return 0,self.loglikelihood_current
              
                
# MOVE ####################
        if action == 'move':
               
            idx_move = np.random.randint(0,len(self.para.points))
                
            pnt_xy,pnt_z = self.para.get_mod_point_coords(idx=idx_move)
            zidx = np.abs(pnt_z-self.propstd_depthlevels).argmin()
            
            if self.adaptive_proposalstd:
                propstd_move = self.propmovestd[zidx]
            else:
                propstd_move = self.propmovestd
     
            for second_try in [False,True]:
            
                # if first move was rejected, try a second move with smaller std
                if second_try:
                    propmovestd = propstd_move * np.random.rand()             
                else:
                    propmovestd = propstd_move
                    
                move_check = self.para.move_point(propmovestd,index=idx_move)
                # if the new position is outsite the coordinate limits
                if not move_check:
                    self.reject_model(targets,action,zidx,pnt_xy)
                    return 0,self.loglikelihood_current

                if self.anisotropic:
                    print("not implemented")
                else:
                    check_mod = self.para.get_model()
                    
                if not check_mod:
                    self.reject_model(targets,action,zidx,pnt_xy)
                    return 0,self.loglikelihood_current
                
                if self.limit_gradients:
                    mingrad,maxgrad = self.para.get_vertical_gradient(self.para.vsfield)
                    if mingrad<self.para.mingrad or maxgrad>self.para.maxgrad:
                        self.reject_model(targets,action,zidx,pnt_xy)
                        return 0,self.loglikelihood_current
                    
                if self.selfcheck:
                    test_vs_field = self.para.vsfield.copy()
                    test_vpvs = self.para.vpvsfield.copy()
                    self.para.get_model(recalculate=True)
                    if (np.abs(self.para.vsfield-test_vs_field) > 1e-10).any():
                        raise Exception(action)
                    if np.isnan(self.para.vsfield).any():
                        raise Exception(f"nan in vel mod during {action}")
                    
                valid = self.update_model_predictions(targets,self.para.idx_mod_gpts)
                if not valid:
                    self.reject_model(targets,action,zidx,pnt_xy)
                    return 0,self.loglikelihood_current

                if self.check_acceptance(delayed=second_try):

                # # first try with normal proposal standard deviation
                # if not second_try:
                #     log_acceptance_prob = self.get_acceptance_prob()
                #     log_a_m1_m0 = log_acceptance_prob
                # else:
                #     log_a_m1_m2 = 1./self.temperature * (loglikelihood_prop1-
                #                                          self.loglikelihood_prop)
                #     log_acceptance_prob = self.get_acceptance_prob(
                #         delayed=True,log_a_m1_m0=log_a_m1_m0,log_a_m1_m2=log_a_m1_m2)
                
                
                # if not second_try:
                #     dx1 = dx
                #     log_acceptance_prob = (1./self.temperature * 
                #                            (self.loglikelihood_prop - 
                #                             self.loglikelihood_current))
                #     loglikelihood_prop1 = self.loglikelihood_prop

                # # second try with smaller proposal standard deviation
                # else:
                #     # see thesis T. Bodin p. 60 eq. 3.39
                #     # see also thesis T. Bodin p. 48 eq. (3.15)
                #     prop_ratio = (np.log(propstd_move) - np.log(propmovestd) + 
                #                   np.sum(dx1**2)/(2*propstd_move**2) - 
                #                   np.sum(dx**2)/(2*propmovestd**2))

                #     a_m1_m2 = np.min([1,np.exp(1./self.temperature * 
                #                                (loglikelihood_prop1-
                #                                 self.loglikelihood_prop))])
                #     a_m1_m = np.min([1,np.exp(log_acceptance_prob)])
                    
                #     if a_m1_m < 1. and a_m1_m2 < 1.:
                #         log_acceptance_prob = (1./self.temperature * 
                #             (self.loglikelihood_prop-self.loglikelihood_current) +
                #             prop_ratio + (np.log((1.-a_m1_m2)/(1.-a_m1_m))))
                #     else:
                #         #print("update cell divide by 0:",log_acceptance_prob)
                #         log_acceptance_prob = -np.inf

                # if np.log(np.random.rand(1)) <= log_acceptance_prob:
                    # model accepted                    
                    self.accept_model(targets,action,zidx,pnt_xy)
                    return 1,self.loglikelihood_current
                    
                else:                                        
                    if self.delayed_rejection and not(second_try):
                        self.para.reject_mod()
                        continue           
                    else:
                        self.reject_model(targets,action,zidx,pnt_xy)          
                        return 0,self.loglikelihood_current


            

    # # function to get all the neighboring voronoi cells of a query point
    # def get_neighbors(self,points,idx_mod):
                
    #     # nearest neighbor lookup tree (fast also for many points)      
    #     tree = cKDTree(points[:,:3])
    #     # 100 nearest neighbors of new point. should include enough neighbors so
    #     # that the Delaunay triangulation in the area is the same as if all points
    #     # were included
    #     nndist,nnidx = tree.query(points[idx_mod,:3],k=np.min([100,len(points)]))
    #     tri = Delaunay(points[nnidx,:3])
    #     if len(tri.coplanar)>0:
    #         print("Warning: coplanar points!")
    #     intptr,neighbor_indices = tri.vertex_neighbor_vertices
    #     neighbors = neighbor_indices[intptr[0]:intptr[1]]
    #     # translate back to global indices (old means prior to move/of the removed point)
    #     neighbors = nnidx[neighbors]
          
    #     return neighbors


    # def get_gpts_dictionary(self,points,gridpoints):
        
    #     gpts_dict = {}
        
    #     if self.pointlocations == 'random':
    #         kdtree = cKDTree(points[:,:3])
    #         point_dist, point_regions = kdtree.query(gridpoints)
    #         for i in range(len(points)):
    #             gpts_dict[i] = np.isin(point_regions,i).nonzero()[0]
                        
    #     elif self.pointlocations == 'profiles':
    #         profile_locs, idxunique = np.unique(points[:,:2],return_inverse=True,axis=0)
    #         kdtree = cKDTree(profile_locs)
    #         for profileidx,profilexy in enumerate(profile_locs):
    #             pointidx = np.where(profileidx==idxunique)[0]
    #             point_dist, point_regions = kdtree.query(gridpoints[:,:2])
    #             gridpoints_idx = np.where(point_regions==profileidx)[0]
    #             gridpoints_subset = gridpoints[gridpoints_idx]
    #             func = interp1d(points[pointidx,2],pointidx,
    #                             kind='nearest',fill_value='extrapolate')
    #             gridpoints_subidx = func(gridpoints_subset[:,2]).astype(int)
    #             for i in pointidx:  
    #                 gpts_dict[i] = np.sort(gridpoints_idx[gridpoints_subidx==i])
    #     else:
    #         raise Exception("unknown parameter for self.pointlocations:",self.pointlocations)

    #     return gpts_dict

    # def check_gpts_dictionary(self,action):
        
    #     test_gpts_dictionary = self.get_gpts_dictionary(self.points,self.gridpoints)
        
    #     for i in test_gpts_dictionary:
    #         if not np.array_equal(np.sort(self.gpts_idx[i]), test_gpts_dictionary[i]):
    #             raise Exception(f"gpts dict not right after {action} operation")


    # def points_to_slownessfield(self,points,intp_points=None):
        
    #     if intp_points is None:
    #         intp_points = self.gridpoints
    #     else:
    #         intp_points = intp_points
        
    #     if self.pointlocations == 'profiles':
            
    #         gpts_dictionary = self.get_gpts_dictionary(points,intp_points)
    #         slowness_field = np.zeros(len(intp_points))
    #         for i in gpts_dictionary:
    #             slowness_field[gpts_dictionary[i]] = 1./points[i,3]
                
    #         if (slowness_field==0).any():
    #             raise Exception("")
                
    #     else:
        
    #         func = self.interpolator(points[:,:3],1./points[:,3])
    #         slowness_field =  func(intp_points)
            
    #     return slowness_field

 
    def update_model_predictions(self,targets,idx_mod_gpts):
        
        for target in targets:
            if target.type=='bodywaves':
                bodywaves = target
            elif target.type=='surfacewaves':
                surfacewaves = target

        # idx_mod_gpts is a boolean array of the same length as self.gridpoints
        if surfacewaves.isactive:
            # ignore modified gridpoints if they are outside the region
            # covered by surfacewaves to speed up the calculations
            if surfacewaves.convexhull is not None:
                inhull = surfacewaves.convexhull.find_simplex(self.gridpoints[idx_mod_gpts,:2])
                idx_mod_gpts_surfwaves = np.where(idx_mod_gpts)[0][inhull>-1]
            else:
                idx_mod_gpts_surfwaves = idx_mod_gpts
            # this leads to an error during the selfcheck since the phasevelocity 
            # maps are incomplete (outside coverage region). To avoid this:
            if self.selfcheck:
                inhull_idx = np.where(surfacewaves.convexhull.find_simplex(self.gridpoints2d)>-1)[0]
            else:
                inhull_idx = None
            self.prop_phttime_predictions,self.prop_phslo_ray,self.prop_phslo_lov,nmods = swm.forward_module(
                self.z,surfacewaves.periods,self.shape,self.para.vsfield,
                self.para.vpvsfield,surfacewaves.A,
                phslo_ray=self.phslo_ray,phslo_lov=self.phslo_lov,
                phttime_predictions=self.phttime_predictions,
                phvel_prior_ray=self.phvel_ray_prior,phvel_prior_lov=self.phvel_lov_prior,
                idx_modified_gridpoints=idx_mod_gpts_surfwaves,
                selfcheck=self.selfcheck,inhull_idx=inhull_idx)
            self.modellengths.append(nmods)
            if self.prop_phttime_predictions is None:
                return False
            (res_sw,self.loglike_prop_sw,self.prop_residual_dict_sw,
             self.prop_loglikelihood_dict_sw,_) = swm.get_loglikelihood(
                self.loglikelihood_dict_sw,self.residual_dict_sw,
                self.likedist_normcoeffs,surfacewaves.data,
                self.prop_phttime_predictions,surfacewaves.path_dists,
                self.datastd,update='predictions',predictions0=self.phttime_predictions,
                foutlier=self.foutlier,widthoutlier=self.widthoutlier,
                degfree=self.degfree,distribution=self.likedist,
                norm=self.misfit_norm,selfcheck=self.selfcheck)
            self.loglike_prop_sw *= self.relweight_surfwaves
        else:
            self.prop_loglikelihood_dict_sw = None
            res_sw = self.loglike_prop_sw = 0.
                
        if bodywaves.isactive:
            if len(idx_mod_gpts)>0:
                currslow_s = 1./self.para.vsfield_backup[idx_mod_gpts]
                propslow_s = 1./self.para.vsfield[idx_mod_gpts]
                currslow_p = currslow_s / self.para.vpvsfield_backup[idx_mod_gpts]
                propslow_p = propslow_s / self.para.vpvsfield[idx_mod_gpts]
                self.prop_p_ttimes = bwm.forward_module(
                    propslow_p,propslow_s,bodywaves.A,
                    ttimes_ref=self.p_ttimes,idx_mod_gpts=idx_mod_gpts,
                    p_slo_ref=currslow_p,s_slo_ref=currslow_s)
            else:
                self.prop_p_ttimes = self.p_ttimes.copy()
            res_bw, self.loglike_prop_bw = bwm.loglikelihood(
                bodywaves.data,self.prop_p_ttimes,
                self.datastd,bodywaves.M,
                norm=self.misfit_norm,distribution=self.likedist,
                foutlier=self.foutlier,widthoutlier=self.widthoutlier)
            self.loglike_prop_bw *= self.relweight_bodywaves
        else:
            res_bw = self.loglike_prop_bw = 0.
            
        self.residual_prop = res_sw + res_bw
        self.loglikelihood_prop = self.loglike_prop_sw + self.loglike_prop_bw
        return True


    def check_acceptance(self,delayed=False):
        
        # EXPERIMENTAL
        # to give equal weights to the two datasets, make the acceptance
        # dependent on one or the other dataset, interchanging at every iteration
        if False: #surfacewaves.isactive and bodywaves.isactive:
            if self.total_steps%2 == 0:
                llike_proposed = self.loglike_prop_sw
                llike_current = self.loglike_curr_sw
            else:
                llike_proposed = self.loglike_prop_bw
                llike_current = self.loglike_curr_bw          
        else: # normal approach:
            llike_current = self.loglikelihood_current
            llike_proposed = self.loglikelihood_prop
        
        if delayed:
            log_a_m1_m2 = 1./self.temperature * (self.llike_prop_m1-llike_proposed)
            if self.log_a_m1_m0 >= 0. or log_a_m1_m2 >=0.:
                return False
            log_prior_prop_ratios = (
                self.para.get_prior_proposal_ratio(delayed=delayed) +
                # this is {1-a1(m'|m'')}/{1-a1(m'|m)} from eq. 3.39 of the thesis of T.Bodin
                np.log((1-np.exp(log_a_m1_m2))/(1-np.exp(self.log_a_m1_m0))) )
        else:
            log_prior_prop_ratios = self.para.get_prior_proposal_ratio()
        
        log_acceptance_prob = (
            log_prior_prop_ratios + 1./self.temperature * 
            (llike_proposed - llike_current) )
            
        # store the value for the delayed rejection scheme
        self.log_a_m1_m0 = log_acceptance_prob
        self.llike_prop_m1 = llike_proposed
        
        return np.log(np.random.rand(1)) <= log_acceptance_prob


    # Unused: modify the width of the uniform distribution in regular intervals? 
    # def update_widthoutlier(self):

    #     for dataset in self.surfacewaves.datasets:
    #         try:
    #             self.collection_widthoutlier[dataset]
    #         except:
    #             self.collection_widthoutlier = {}
    #             self.collection_widthoutlier[dataset] = {}
    #         for period in self.outlierwidth[dataset]:

    #             residuals = (self.surfacewaves.path_dists[dataset][period] / self.surfacewaves.data[dataset][period] -
    #                          self.surfacewaves.path_dists[dataset][period] / self.phttime_predictions[dataset][period])
    #             if (residuals > self.outlierwidth[dataset][period][1]).any():
    #                 self.outlierwidth[dataset][period][1] = np.max(residuals)
    #                 self.widthoutlier[dataset][period] = np.diff(self.outlierwidth[dataset][period])
    #             elif (residuals < self.outlierwidth[dataset][period][0]).any():
    #                 self.outlierwidth[dataset][period][0] = np.min(residuals)
    #                 self.widthoutlier[dataset][period] = np.diff(self.outlierwidth[dataset][period])
    #             try:
    #                 self.collection_widthoutlier[dataset][period].append(self.widthoutlier[dataset][period])
    #             except:
    #                 self.collection_widthoutlier[dataset][period] = [self.widthoutlier[dataset][period]]
    #     raise Exception("needs to be fixed (bwaves missing)")
    #     self.residual_current,self.loglikelihood_current,_ = swm.loglikelihood(
    #         self.surfacewaves.data,self.phttime_predictions,self.datastd,
    #         self.surfacewaves.path_dists,norm=self.misfit_norm,distribution=self.likedist,
    #         foutlier=self.foutlier,widthoutlier=self.widthoutlier,
    #         degfree=self.degfree,norm_coeffs=self.likedist_normcoeffs)
    #     self.loglikelihood_current *= self.relweight_



    def update_current_likelihood(self,targets):
        # this is a complete update of model predictions and likelihoods,
        # necessary after the paths have changed
        with open(self.logfile,"a") as f:
            f.write("\nUpdating model predictions and current loglikelihood.\n")
            
        for target in targets:
            if target.type=='bodywaves':
                bodywaves = target
            elif target.type=='surfacewaves':
                surfacewaves = target
        # if self.parameterization == 'wavelets' and self.total_steps>0:
        #     self.correct_wavelet_model()
            
        if surfacewaves.isactive:
            if surfacewaves.convexhull is not None:
                inhull_idx = np.where(surfacewaves.convexhull.find_simplex(self.gridpoints2d)>-1)[0]
            else:
                inhull_idx = None
            self.phttime_predictions,self.phslo_ray,self.phslo_lov,nmods = swm.forward_module(
                self.z,surfacewaves.periods,self.shape,self.para.vsfield,self.para.vpvsfield,
                surfacewaves.A,inhull_idx=inhull_idx,
                phvel_prior_ray=self.phvel_ray_prior,phvel_prior_lov=self.phvel_lov_prior)
            if self.phttime_predictions is None:
                # if only paths are updated, it should be no problem (phasevel maps do not change)
                # but if the slowfield is changing, the result may be invalid
                raise Exception("invalid value in phase slowness arrays")
            # path dists may have changed, therefore update also data stds
            if self.error_type=='relative':
                for dataset in surfacewaves.datasets:
                    for period in surfacewaves.data[dataset]:
                        self.datastd[dataset][period] = self.std_scaling[dataset][period] * surfacewaves.path_dists[dataset][period] + self.std_intercept[dataset][period]
            (res_sw,self.loglike_curr_sw,self.residual_dict_sw,
             self.loglikelihood_dict_sw,self.likedist_normcoeffs) = swm.get_loglikelihood(
                self.loglikelihood_dict_sw,self.residual_dict_sw,None,
                surfacewaves.data,self.phttime_predictions,
                surfacewaves.path_dists,self.datastd,
                update=None,foutlier=self.foutlier,widthoutlier=self.widthoutlier,
                degfree=self.degfree,distribution=self.likedist,
                norm=self.misfit_norm,selfcheck=self.selfcheck)
            self.loglike_curr_sw *= self.relweight_surfwaves
        else:
            res_sw = self.loglike_curr_sw = 0.

        if bodywaves.isactive:
            s_slow = 1./self.para.vsfield
            p_slow = s_slow / self.para.vpvsfield
            self.p_ttimes = bwm.forward_module(
                        p_slow,s_slow,bodywaves.A,
                        ttimes_base=bodywaves.ttimes_base)
            res_bw, self.loglike_curr_bw = bwm.loglikelihood(
                bodywaves.data,self.p_ttimes,
                self.datastd,bodywaves.M,
                norm=self.misfit_norm,distribution=self.likedist,
                foutlier=self.foutlier,widthoutlier=self.widthoutlier)
            self.loglike_curr_bw *= self.relweight_bodywaves
        else:
            res_bw = self.loglike_curr_bw = 0.

        self.residual_current = res_sw + res_bw
        self.loglikelihood_current = self.loglike_curr_sw + self.loglike_curr_bw

        self.para.accept_mod()
        
        # the goodcoverage_hull is used to make sure that the proposal stadevs
        # are adapted so that the acceptance rate is good for updates within
        # the central model region (within the good data coverage hull)
        data_coverage_sw = self.get_data_coverage_sw(surfacewaves)
        data_coverage_bw = self.get_data_coverage_bw(bodywaves)
        data_coverage_2d = np.sum(np.reshape(data_coverage_sw+data_coverage_bw,self.X.shape),axis=2).flatten()
        self.goodcoverage_hull = Delaunay(self.gridpoints2d[data_coverage_2d>0.1*np.max(data_coverage_2d)])
            
        if False:
            cov = np.reshape(surfacewaves.data_coverage.copy(),self.X[:,:,0].shape)
            cov[cov<0.1*np.max(cov)] = 0.
            stations = []
            for dataset in surfacewaves.sources:
                for period in surfacewaves.sources[dataset]:
                    stations.append(surfacewaves.sources[dataset][period])
                    stations.append(surfacewaves.receivers[dataset][period])
            stations = np.unique(np.vstack(stations),axis=0)       
            plt.figure()
            plt.pcolormesh(self.X[:,:,0],self.Y[:,:,0],cov,shading='nearest')
            #plt.pcolormesh(self.X[:,:,0],self.Y[:,:,0],np.reshape(test,self.X[:,:,0].shape),shading='nearest')
            for stat in stations:
                plt.plot(stat[0],stat[1],'rv')
            plt.show()
 
    
    # def correct_wavelet_model(self):
    #     # This function helps to correct invalid models for which the phasevel
    #     # calculation fails.
    #     idepth = 0
    #     self.para.get_model
    #     while True:
    #         valid,idx_2d,self.prop_phslo_ray,self.prop_phslo_lov,nmods = swm.get_phasevel_maps(
    #             self.z,self.surfacewaves.periods,self.shape,self.para.vsfield,self.para.vpvsfield)
    #         if valid:
    #             break
    #         else:
    #             raise Exception()
    #         print("smoothing wavelet model to avoid incorrect phase velocities")
    #         invalid_idx = np.where(np.isinf(self.prop_phslo_ray[self.surfacewaves.periods[0]])+
    #                                np.isinf(self.prop_phslo_lov[self.surfacewaves.periods[0]]))[0]
    #         self.para.smooth_coefficients(invalid_idx,idepth)
    #         self.para.get_model
    #         # check that we are updating the correct coefficients
    #         idx_mod = np.where(self.para.vsfield!=self.para.vsfield)[0]
    #         idx_mod_2d = tuple(np.unique(np.unravel_index(
    #             idx_mod,self.shape)[:2],axis=1))
    #         flat_idx = np.ravel_multi_index(idx_mod_2d,self.shape[:2])
    #         if not np.in1d(invalid_idx,flat_idx,assume_unique=True).any():
    #             raise Exception("not working")
    #         idepth += 1
                
            
    #     self.para.vsfield = self.para.vsfield
    #     self.para.vpvsfield = self.para.vpvsfield
        
    #     self.phttime_predictions = swm.calculate_ttimes(
    #         self.prop_phslo_ray,self.prop_phslo_lov,self.surfacewaves.A)
    #     self.residual,self.loglikelihood_current = self.loglikelihood(
    #         self.surfacewaves.data,self.phttime_predictions,self.datastd_sw,self.foutlier)    
    #     self.phslo_ray = self.prop_phslo_ray
    #     self.phslo_lov = self.prop_phslo_lov

    #     return
        
        
    def accept_model(self,targets,action,depthidx,pnt_xy):
        
        self.para.accept_mod()
        if self.selfcheck:
            if np.isnan(self.para.points).any():
                raise Exception(f"nan in points after {action}")
            if np.isnan(self.para.vsfield).any():
                raise Exception("nan in velocity model after",action)
        self.loglikelihood_current = self.loglikelihood_prop
        self.loglike_curr_sw = self.loglike_prop_sw
        self.loglike_curr_bw = self.loglike_prop_bw
        if type(self.residual_prop) == type([]):
            raise Exception("residual prop error")
        self.residual_current = self.residual_prop
        if self.prop_loglikelihood_dict_sw is not None:
            self.loglikelihood_dict_sw = self.prop_loglikelihood_dict_sw
            self.residual_dict_sw = self.prop_residual_dict_sw
        if self.prop_phttime_predictions is None:
            raise Exception("must not be None!",action)
        self.phttime_predictions = self.prop_phttime_predictions
        self.p_ttimes = self.prop_p_ttimes
        self.phslo_ray = self.prop_phslo_ray
        self.phslo_lov = self.prop_phslo_lov
        if self.anisotropic:
            self.psi2amp = self.prop_psi2amp
            self.psi2 = self.prop_psi2
            
        # the acceptance rate gives the CURRENT acceptance rate, i.e. relative to the last 100 steps
        if self.goodcoverage_hull.find_simplex(pnt_xy)>-1:
            self.avg_acceptance_rate[action][0] = 1
            self.avg_acceptance_rate[action] = np.roll(self.avg_acceptance_rate[action],1)
            self.acceptance_rate[action][depthidx,0] = 1
            self.acceptance_rate[action][depthidx] = np.roll(self.acceptance_rate[action][depthidx],1)
        self.accepted_steps[action] += 1
        if self.adaptive_proposalstd  and self.total_steps%100==0:
            self.update_proposal_stds(depthidx)

        self.meantimes[action] += time.time()-self.t0
        
        self.collect_model(targets,action)
        
        return ()

        
        

    def reject_model(self,targets,action,depthidx,pnt_xy):
 
        self.para.reject_mod()       
 
        if depthidx is not None:
            # the acceptance rate gives the CURRENT acceptance rate, i.e. relative to the last 100 steps
            if self.goodcoverage_hull.find_simplex(pnt_xy)>-1:
                self.avg_acceptance_rate[action][0] = 0
                self.avg_acceptance_rate[action] = np.roll(self.avg_acceptance_rate[action],1)
                self.acceptance_rate[action][depthidx,0] = 0
                self.acceptance_rate[action][depthidx] = np.roll(self.acceptance_rate[action][depthidx],1)
            if self.adaptive_proposalstd and self.total_steps%100==0:
                self.update_proposal_stds(depthidx)

        self.rejected_steps[action] += 1
                
        self.collect_model(targets,action)
        
        return ()
    

    def collect_model(self,targets,action): #store_model is executed after EVERY iteration (also error updates)
    
       # if self.selfcheck:
       #     if self.loglikelihood_current != self.loglikelihood(
       #             self.surfacewaves.data,self.phttime_predictions,
       #                                                         self.datastd_sw,self.foutlier)[1]:
       #         raise Exception(f"loglikelihood_current not right after {action}")
    
        # if self.parameterization == 'wavelets' and self.total_steps%1000==0:
        #     self.correct_wavelet_model()
        
        # in regular intervals, do a full re-calculation of the likelihoods
        # in theory this should not be necessary, but it seems there are
        # sometimes some numerical errors that accumulate.
        # if self.total_steps%10001==0:
        #     self.update_current_likelihood()
    
        if self.total_steps == self.nburnin: # pre-burnin average models are discarded (just a single average is kept)
            with open(self.logfile,"a") as f:
                f.write("\n###\nChain %d: Burnin phase finished.\n###\n" %self.chain_no)
            self.average_vs /= self.average_model_counter
            self.vssumsquared = self.average_vs.copy()**2
            self.average_vp /= self.average_model_counter
            self.vpsumsquared = self.average_vp.copy()**2
            self.average_vpvs /= self.average_model_counter
            self.vpvssumsquared = self.average_vpvs.copy()**2
            if self.anisotropic:
                self.average_anisotropy = np.column_stack((np.cos(2*self.psi2)*self.psi2amp,
                                                           np.sin(2*self.psi2)*self.psi2amp))
            self.average_model_counter = 1

            # resetting statistics
            self.collection_no_points = []
            for dataset in self.collection_datastd:
                if type(self.collection_datastd[dataset]) == type({}):
                    for period in self.collection_datastd[dataset]:
                        self.collection_datastd[dataset][period] = []
                else:
                    self.collection_datastd[dataset] = []

        if self.total_steps%self.collect_step == 0 and self.temperature==1:
            self.average_model_counter += 1
            self.average_vs += self.para.vsfield
            self.average_vp += self.para.vsfield*self.para.vpvsfield
            self.average_vpvs += self.para.vpvsfield
            # modelsumsquared is needed for the calculation of the model uncertainty
            self.vssumsquared += self.para.vsfield**2
            self.vpsumsquared += (self.para.vsfield*self.para.vpvsfield)**2
            self.vpvssumsquared += self.para.vpvsfield**2
            #if self.selfcheck:
            #    if (self.modelsumsquared - self.average_model**2/self.average_model_counter < -0.1).any():
            #        raise Exception("error in modeluncertainty calculation")
            if self.total_steps >= self.nburnin:
                if self.store_models:
                    # saving disk space by saving as float16, this means rounded to 2-3 digits
                    self.collection_vs_models.append(self.para.vsfield.astype('float16'))
                    self.collection_vpvs_models.append(self.para.vpvsfield.astype('float16'))

            for dataset in self.collection_datastd:
                if type(self.collection_datastd[dataset]) == type({}):
                    for period in self.collection_datastd[dataset]:
                        self.collection_datastd[dataset][period].append(
                            np.mean(self.datastd[dataset][period]))
                else:
                    self.collection_datastd[dataset].append(
                        np.mean(self.datastd[dataset]))
        
        self.collection_no_points.append(len(self.para.points))
        self.collection_loglikelihood.append(self.loglikelihood_current)
        try:
            self.collection_residuals.append(self.residual_current)
        except:
            self.collection_residuals = [self.residual_current]
    
        if self.total_steps % self.print_stats_step == 0 or self.total_steps==1:
            self.print_statistics(targets)
            
        self.cleanup(targets)
            

    def update_proposal_stds(self,depthidx):
        
        if self.delayed_rejection:
            acc_rate = 75
        else:
            acc_rate = 50
        
        if np.sum(self.acceptance_rate['velocity_update'][depthidx]) < 42:
            self.propvelstd_pointupdate[depthidx] *= 0.99
        elif np.sum(self.acceptance_rate['velocity_update'][depthidx]) > acc_rate:
            self.propvelstd_pointupdate[depthidx] *= 1.01
        # don't allow values below 0.01 km/s for the velocity updates
        self.propvelstd_pointupdate[depthidx] = np.max([0.01,self.propvelstd_pointupdate[depthidx]])
        # moving points
        if 'move' in self.acceptance_rate.keys():
            if np.sum(self.acceptance_rate['move'][depthidx]) < 42:
                self.propmovestd[depthidx] *= 0.99
            elif np.sum(self.acceptance_rate['move'][depthidx]) > 50:
                self.propmovestd[depthidx] *= 1.01
        # don't allow values below 1/5 of the horizontal gridspacing
        hor_gridspace = np.min([self.xgridspacing,self.ygridspacing])
        self.propmovestd[depthidx] = np.max([hor_gridspace/5.,self.propmovestd[depthidx]])
            
        # for birth/death steps it is often not possible to get a good
        # acceptance rate and thus there is no good way to modify the proposal
        # standard deviation. Therefore, we keep it fixed
        
        # or take the same as for the velocity updates
        #if type(self.propvelstd_dimchange)==type(np.array([])):
        #    self.propvelstd_dimchange[depthidx] = self.propvelstd_pointupdate[depthidx]

        # # experimental, to optimize the birth/death acceptance rate
        # # dimension change (birth/death)
        # if np.sum(self.acceptance_rate['birth']) < 40:
        #     if np.sum(self.acceptance_rate['birth']) < self.acceptance_rate_previous:
        #         self.propvelstd_dimchange_update *= -1
        #     self.propvelstd_dimchange +=  self.propvelstd_dimchange_update 
        #     if self.propvelstd_dimchange <= 0.:
        #         self.propvelstd_dimchange = 0.01
        #     self.acceptance_rate_previous = np.sum(self.acceptance_rate['birth'])
        
       

    def cleanup(self,targets):
        
        for target in targets:
            if target.type=='bodywaves':
                bodywaves = target
            elif target.type=='surfacewaves':
                surfacewaves = target
                
        if self.selfcheck:
            if surfacewaves.isactive:
                (res_sw,ll_sw,residual_dict_sw,
                 loglike_dict_sw,norm_coeffs) = swm.get_loglikelihood(
                    self.loglikelihood_dict_sw,self.residual_dict_sw,
                    self.likedist_normcoeffs,surfacewaves.data,
                    self.phttime_predictions,surfacewaves.path_dists,
                    self.datastd,foutlier=self.foutlier,
                    widthoutlier=self.widthoutlier,degfree=self.degfree,
                    distribution=self.likedist,norm=self.misfit_norm,
                    selfcheck=self.selfcheck)
                ll_sw *= self.relweight_surfwaves
            else:
                ll_sw = 0.
            for period in surfacewaves.periods:
                if (np.inf in self.phslo_ray[period] or
                    np.inf in self.phslo_lov[period]):
                    raise Exception("invalid value in phase velocity maps")
            if bodywaves.isactive:
                res_bw,ll_bw = bwm.loglikelihood(
                    bodywaves.data,self.p_ttimes,self.datastd,
                    bodywaves.M,norm=self.misfit_norm,
                    distribution=self.likedist,foutlier=self.foutlier,
                    widthoutlier=self.widthoutlier)
                ll_bw *= self.relweight_bodywaves
            else:
                ll_bw = 0.
            loglikelihood = ll_sw+ll_bw
            if loglikelihood != self.loglikelihood_current:
                print(loglikelihood,self.loglikelihood_current)
                raise Exception("loglikelihoods not up to date")

        # optional progress visualization every visualize_step iteration
        if self.visualize and self.total_steps%self.visualize_step == 0:
            self.visualize_progress(surfacewaves)

        # this is just to be sure that the variables are not reused and everything works properly
        # could be removed in the future
        self.loglikelihood_prop = []
        self.loglike_prop_bw = []
        self.loglike_prop_sw = []
        self.residual_prop = []
        self.prop_phttime_predictions = []
        self.prop_phslo_ray = []
        self.prop_phslo_lov = []
        self.prop_p_ttimes = []
        if self.anisotropic:
            self.prop_psi2amp = []
            self.prop_psi2 = []
        self.log_a_m1_m0 = None
        self.llike_prop_m1 = None
        
        return ()
    
   
    def get_data_coverage_sw(self,surfacewaves):

        data_coverage_sw = np.zeros(len(self.gridpoints))
        if surfacewaves.isactive:
            for dataset in surfacewaves.datasets:
                for period in surfacewaves.data[dataset]:
                    data_coverage_sw += np.repeat(np.array(surfacewaves.A[dataset][period].sum(axis=0))[0],len(self.z))       
        return data_coverage_sw
    
    def get_data_coverage_bw(self,bodywaves):
        
        data_coverage_bw = np.zeros(len(self.gridpoints))
        if bodywaves.isactive:
            for dataset in bodywaves.data:
                data_coverage_bw += np.array(bodywaves.A[dataset].sum(axis=0))[0]
        return data_coverage_bw
    
    def get_data_coverage_convexhull(self,targets):
        
        for target in targets:
            if target.type=='surfacewaves':
                data_coverage_sw = self.get_data_coverage_sw(target)
            elif target.type=='bodywaves':
                data_coverage_bw = self.get_data_coverage_bw(target)
        
        # create a convex hull
        hull = ConvexHull(self.gridpoints[(data_coverage_sw+data_coverage_bw)>0])
        # make the hull larger by two gridpoints in each direction
        zgridspacing = np.max(np.diff(self.z[:-1]))
        dX = (np.array([self.xgridspacing,self.ygridspacing,zgridspacing])*
              np.array([[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1],
                        [1,1,1],[1,1,-1],[1,-1,-1],[1,-1,1],[0,0,0]])*2)
        edgepoints = []
        for i in range(len(hull.vertices)):
            edgepoints.append(hull.points[hull.vertices[i]]+dX)
        edgepoints = np.vstack(edgepoints)
        hull = ConvexHull(edgepoints)
        # use Delaunay instead of ConvexHull because it allows easily to check
        # whether a point is inside the hull
        # also reduce the number of points to the actual edge points
        return Delaunay(hull.points[hull.vertices])


    def print_statistics(self,targets):
        
        for target in targets:
            if target.type=='bodywaves':
                bodywaves = target
            elif target.type=='surfacewaves':
                surfacewaves = target

        with open(self.logfile,"a") as f:
            f.write("\n################################\n")
            f.write(time.ctime()+"\n")
            f.write("Chain %d: (Temperature: %.1f, Loglikelihood: %.1f, Residual: %.1f)\n" %(
                self.chain_no,self.temperature,self.loglikelihood_current,self.residual_current))
            f.write("    Iteration: %d\n" %self.total_steps)
            f.write("    Avg acceptance rate velocity update: %d\n" %np.sum(self.avg_acceptance_rate['velocity_update']))
            f.write("    Avg acceptance rate birth: %d\n" %np.sum(self.avg_acceptance_rate['birth']))
            f.write("    Avg acceptance rate death: %d\n" %np.sum(self.avg_acceptance_rate['death']))
            if self.parameterization != 'wavelets':
                f.write("    Avg acceptance rate move: %d\n" %np.sum(self.avg_acceptance_rate['move']))
                f.write("    Acceptance rate (proposal std)\n       depth   cell update    birth       death        move\n")
            else:
                f.write("    Acceptance rate (proposal std)\n       level   coeff update    birth       death\n")
            for zi in range(len(self.propstd_depthlevels)):
                if self.adaptive_proposalstd:
                    propvelstd_pointupdate = self.propvelstd_pointupdate[zi]
                    if type(self.propvelstd_dimchange) == type(np.array([])): 
                        
                        propstd_birth = propstd_death = str(
                            np.around(np.mean(self.propvelstd_dimchange[zi]),2))
                    else:
                        propstd_birth = propstd_death = 'uniform'
                    #propstd_death = np.mean(self.propstd_death[zi])
                    propmovestd = self.propmovestd[zi]
                else:
                    propvelstd_pointupdate = self.propvelstd_pointupdate
                    propstd_birth = propstd_death = self.propvelstd_dimchange
                    if type(propstd_birth)!=str:
                        propstd_birth = propstd_death = str(np.around(propstd_birth,2))
                    propmovestd = self.propmovestd
                if self.parameterization != 'wavelets':
                    f.write("    %6.1f km   %3d (%.2f)   %3d (%s)   %3d (%s)   %3d (%.2f)\n" %(
                        self.inverse_z_scaling(self.propstd_depthlevels[zi]),
                        np.sum(self.acceptance_rate['velocity_update'][zi]), propvelstd_pointupdate,
                        np.sum(self.acceptance_rate['birth'][zi]), propstd_birth,
                        np.sum(self.acceptance_rate['death'][zi]), propstd_death,
                        np.sum(self.acceptance_rate['move'][zi]), propmovestd))
                else:
                    f.write("    %d    %3d (%.2f)   %3d (%s)   %3d (%s)\n" %(
                        self.inverse_z_scaling(self.propstd_depthlevels[zi]),
                        np.sum(self.acceptance_rate['velocity_update'][zi]), propvelstd_pointupdate,
                        np.sum(self.acceptance_rate['birth'][zi]), propstd_birth,
                        np.sum(self.acceptance_rate['death'][zi]), propstd_death))                    
            if self.anisotropic and self.total_steps>int(self.nburnin/2):
                f.write("    Acceptance rate anisotropy update direction: %d\n" %(np.sum(self.avg_acceptance_rate['anisotropy_update_direction'])))
                f.write("    Acceptance rate anisotropy update amplitude: %d\n" %(np.sum(self.avg_acceptance_rate['anisotropy_update_amplitude'])))
            #f.write("    Acceptance rate birth/death should be around 40-50%. The Acceptance rate move/cell update will be higher due to the delayed rejection.")
            #f.write("    Proposal standard deviation cell update: %.2f  move crust: %.2f\n" %(self.propvelstd_pointupdate,self.propmovestd))
            if self.total_steps>100:
                f.write("    Mean no of Voronoi cells / parameters: %d\n" %(np.mean(self.collection_no_points[-100:])))
            else:
                f.write("    Start no of Voronoi cells / parameters: %d\n" %(len(self.para.points)))
            f.write("    Mean data std:\n")
            if surfacewaves.isactive:
                for dataset in surfacewaves.data:
                    datalengths = []
                    mean_data_std = []
                    for period in surfacewaves.data[dataset]:
                        mean_data_std.append(np.mean(self.datastd[dataset][period]))
                        datalengths.append(len(surfacewaves.data[dataset][period]))
                    f.write("        %s: %.2f\n" %(dataset,np.average(mean_data_std,weights=datalengths)))
            if bodywaves.isactive:
                for dataset in bodywaves.data:
                    f.write("        %s: %.2f\n" %(dataset,np.mean(self.datastd[dataset])))                    
            if self.likedist == 'outliers':
                f.write("    Mean outlier fraction:\n")
                if surfacewaves.isactive:
                    for dataset in surfacewaves.data:
                        outlier_fraction = []
                        datalengths = []
                        for period in surfacewaves.data[dataset]:
                            outlier_fraction.append(self.foutlier[dataset][period])
                            datalengths.append(len(surfacewaves.data[dataset][period]))
                        f.write("        %s: %d\n" %(dataset,100*np.average(outlier_fraction,weights=datalengths,axis=0)[0]))
                if bodywaves.isactive:
                    for dataset in bodywaves.data:
                        f.write("        %s: %d\n" %(dataset,100*np.mean(self.foutlier[dataset])))                        
            if self.likedist == 'students_t' and surfacewaves.isactive:
                f.write("    t-distribution degrees of freedom:\n")
                for dataset in surfacewaves.data:
                    degfree = []
                    datalengths = []
                    for period in surfacewaves.data[dataset]:
                        degfree.append(self.degfree[dataset][period])
                        datalengths.append(len(surfacewaves.data[dataset][period]))     
                    f.write("        %s: %.2f\n" %(dataset,np.average(degfree,weights=datalengths)))
            f.write("    Avg. time per accepted step:")
            for key in self.meantimes:
                f.write(" %s: %.3fs" %(key,self.meantimes[key]/(self.accepted_steps[key]+1e-10)))
            f.write("\n")
            # if 'move' in self.meantimes.keys():
            #     f.write("    Avg. time per accepted step: update: %.3fs birth: %.3fs death: %.3fs move: %.3fs error update: %.3fs\n" 
            #             %(self.meantimes['velocity_update']/(self.accepted_steps['velocity_update']+1e-10),
            #               self.meantimes['birth']/(self.accepted_steps['birth']+1e-10),
            #               self.meantimes['death']/(self.accepted_steps['death']+1e-10),
            #               self.meantimes['move']/(self.accepted_steps['move']+1e-10),
            #               self.meantimes['error_update']/(self.accepted_steps['error_update']+1e-10)))
            # else:
            #     f.write("    Avg. time per accepted step: update: %.3fs birth: %.3fs death: %.3fs\n" 
            #             %(self.meantimes['velocity_update']/(self.accepted_steps['velocity_update']+1e-10),
            #               self.meantimes['birth']/(self.accepted_steps['birth']+1e-10),
            #               self.meantimes['death']/(self.accepted_steps['death']+1e-10)))                
            if len(self.modellengths)>1:
                if self.gridsizefactor > 1.:
                    f.write("    Avg. number of dispcurves: %d  gridsize factor: %.1f\n" %(np.mean(self.modellengths),self.gridsizefactor))
                else:
                    f.write("    Avg. number of dispersion curves calculated per iteration: %d\n" %np.mean(self.modellengths))
            self.modellengths = []
            f.write("# # # # # # # # # # # # # #\n")

    
    def get_startmodel(self,surfacewaves,mingrad=None,maxgrad=None):
        
        vspriormin = self.minvs(self.z_scaled)
        vspriormax = self.maxvs(self.z_scaled)
    
        if mingrad==None:
            mingrad = -1.0
        if maxgrad==None:
            maxgrad = 1.0
    
        # get a random 1D starting model
        while True:
            vsmod = np.random.uniform(vspriormin,vspriormax)
            vsmod = np.sort(vsmod)
            vpmod = np.sqrt(3)*vsmod
            
            if surfacewaves.isactive:
                ray,lov = swm.profile_to_phasevelocity(
                    self.z,vsmod,vpmod,surfacewaves.periods,
                    self.phvel_ray_prior,self.phvel_lov_prior)
                if (ray==0.).any() or (lov==0.).any():
                    continue
                
            startmodel_depth = np.linspace(self.minz,self.z[-1],1000)
            startmodel_vs = np.interp(startmodel_depth,self.z,vsmod,)

            #startmodel_vs = np.zeros(len(startmodel_depth))
            for zi in range(1,len(startmodel_depth)):
                dz = startmodel_depth[zi] - startmodel_depth[zi-1]
                minvs = np.max([startmodel_vs[zi-1]+dz*mingrad,
                                self.minvs(self.z_scaling(startmodel_depth[zi]))]) + 1e-5
                maxvs = np.min([startmodel_vs[zi-1]+dz*maxgrad,
                                self.maxvs(self.z_scaling(startmodel_depth[zi]))]) - 1e-5
                if maxvs < minvs:
                    break # start again from the start
                if startmodel_vs[zi]>minvs and startmodel_vs[zi]<maxvs:
                    continue # okay
                else:
                    for k in range(100):
                        startmodel_vs[zi] = np.random.uniform(minvs,maxvs)
                        vsmod = np.interp(self.z,startmodel_depth,startmodel_vs)
                        vpmod = np.sqrt(3)*vsmod
                        if surfacewaves.isactive:
                            ray,lov = swm.profile_to_phasevelocity(
                                self.z,vsmod,vpmod,surfacewaves.periods,
                                self.phvel_ray_prior,self.phvel_lov_prior)
                            if (ray==0.).any() or (lov==0.).any():
                                continue
                        break # found value that is okay
                    else:
                        break # start again from the start
            else:
                vsmod = np.interp(self.z,startmodel_depth,startmodel_vs)
                vpmod = np.sqrt(3)*vsmod
                if surfacewaves.isactive:
                    ray,lov = swm.profile_to_phasevelocity(
                    self.z,vsmod,vpmod,surfacewaves.periods,
                    self.phvel_ray_prior,self.phvel_lov_prior)
                    if (ray==0.).any() or (lov==0.).any():
                        continue # start again from the beginning
                return interp1d(self.z_scaling(startmodel_depth),startmodel_vs) 
                
                
    def check_prior_range(self,surfacewaves):
        
        min_vsmodel = self.minvs(self.z_scaled)
        max_vsmodel = self.maxvs(self.z_scaled)
        
        swm.check_prior_range(self.z,surfacewaves.periods,
                              min_vsmodel,max_vsmodel,
                              surfacewaves.data,
                              surfacewaves.path_dists,
                              printfile=self.logfile.replace("logfile","velocity_prior_warning_chain"))


    def visualize_progress(self,surfacewaves,initialize=False):
        
        if initialize:
            fig = plt.figure(figsize=(14,10))
            gs = gridspec.GridSpec(2,5,width_ratios=[1,1,1,1,0.1],height_ratios=[2,1],wspace=0.5)
            self.ax11 = fig.add_subplot(gs[0,0:2])
            self.ax12 = fig.add_subplot(gs[0,2:4])
            self.cbarax = fig.add_subplot(gs[0,4])
            self.ax11.set_aspect('equal')
            self.ax12.set_aspect('equal')
            self.ax21 = fig.add_subplot(gs[1,0])
            self.ax22 = fig.add_subplot(gs[1,1])
            self.ax23 = fig.add_subplot(gs[1,2])
            self.ax24 = fig.add_subplot(gs[1,3])
            plt.ion()    
            
            cbar = self.ax12.pcolormesh(self.X,self.Y,
                                        self.average_vs.reshape(np.shape(self.X)),
                                        vmin=self.velmin,vmax=self.velmax,shading='nearest')
            self.cbar = plt.colorbar(cbar,cax=self.cbarax,label='velocity')
            self.ax11.set_title('current proposal')
            self.ax12.set_title('average model')
            self.ax11.set_aspect('equal')
            self.ax12.set_aspect('equal')
            plt.draw()            

        else:
            avg_model = self.average_vs.reshape(np.shape(self.X))/self.average_model_counter
            vmax = np.max(avg_model)
            vmin = np.min(avg_model)
            self.ax11.clear()
            self.ax12.clear()
            self.ax11.pcolormesh(self.X,self.Y,self.para.vsfield,
                                 vmin=vmin,vmax=vmax,cmap=plt.cm.jet_r,
                                 shading='nearest')
            self.ax11.plot(self.para.points[:,0],self.para.points[:,1],'ko')
            self.ax11.set_title('current proposal')
            cbar = self.ax12.pcolormesh(self.X,self.Y,
                                        self.average_vs.reshape(np.shape(self.X))/self.average_model_counter,
                                        vmin=vmin,vmax=vmax,cmap=plt.cm.jet_r,
                                        shading='nearest')
            self.ax12.set_title('average model')
            stations = np.unique(np.vstack((surfacewaves.sources,surfacewaves.receivers)),axis=0)
            self.ax12.plot(stations[:,0],stations[:,1],'rv',markeredgecolor='black')
            
            self.cbar.update_bruteforce(cbar)
            
            self.ax21.clear()
            no_cells = np.min([len(np.unique(self.collection_no_points)),25])
            self.ax21.hist(self.collection_no_points,no_cells)
            self.ax21.set_xlabel('no of voronoi cells',fontsize=8)
            self.ax21.set_ylabel('no of models',fontsize=8)
            self.ax21.tick_params(axis='both', which='major', labelsize=8)
            
            self.ax22.clear()
            self.ax22.hist(self.collection_datastd,25)
            self.ax22.set_xlabel('data std',fontsize=8)
            self.ax22.set_ylabel('no of models',fontsize=8)
            self.ax22.tick_params(axis='both', which='major', labelsize=8)
            
            self.ax23.clear()
            self.ax23.plot(self.collection_loglikelihood)
            self.ax23.set_ylabel('log likelihood',fontsize=8)
            self.ax23.set_xlabel('iteration no',fontsize=8)
            self.ax23.tick_params(axis='both', which='major', labelsize=8)
            
            self.ax24.clear()
            self.ax24.text(0,0.95,"Chain number: %d" %self.chain_no)
            self.ax24.text(0,0.85,"Iterations: {:,}".format(self.total_steps))
            self.ax24.text(0,0.75,"Burnin samples: {:,}".format(self.nburnin))
            self.ax24.text(0,0.65,"Std proposal data uncertainty: %.3f" %self.propsigmastd)
            self.ax24.text(0,0.55,"Std proposal velocity update: %.3f" %np.mean(self.propvelstd_pointupdate))
            self.ax24.text(0,0.45, "Std proposal birth/death: %.3f" %np.mean([self.propstd_birth,self.propstd_death]))
            self.ax24.text(0,0.35,"Std proposal move cell: %.3f" %np.mean(self.propmovestd))
            self.ax24.text(0,0.25,"Temperature: %.1f" %self.temperature)
            self.ax24.text(0,0.15,"Average of every %dth model" %self.collect_step)
            if self.total_steps>self.update_paths_interval:
                self.ax24.text(0,0.05,"Paths updated every {:,}th iteration.".format(self.update_paths_interval))
            self.ax24.axis('off')
            plt.draw()
            plt.pause(0.1)


    def plot(self,targets, saveplot = False, output_location = None):
        
        if output_location is None and saveplot:
            output_location = os.path.dirname(self.logfile)
            
        if saveplot:
            plt.ioff()
            
        for target in targets:
            if target.type=='bodywaves':
                bodywaves = target
            elif target.type=='surfacewaves':
                surfacewaves = target
        #%%
        if self.anisotropic:
            # mean angle given by np.arctan(y/x) -> arctan(sum(sin(phi))/sum(cos(phi)))
            anisotropy_dirmean = 0.5*np.arctan(self.average_anisotropy[:,1]/
                                               self.average_anisotropy[:,0])
            anisotropy_ampmean = np.sqrt((self.average_anisotropy[:,0]/self.average_model_counter)**2 +
                                         (self.average_anisotropy[:,1]/self.average_model_counter)**2)
    
            fig = plt.figure(figsize=(14,6))
            ax1 = fig.add_subplot(121)
            cbar = ax1.pcolormesh(self.X,self.Y,100*anisotropy_ampmean.reshape(np.shape(self.X)),
                                     shading='nearest')
            plt.colorbar(cbar)
            ax2 = fig.add_subplot(122)
            ax2.quiver(self.X,self.Y,100*anisotropy_ampmean.reshape(np.shape(self.X)),
                        100*anisotropy_ampmean.reshape(np.shape(self.X)),
                        angles=anisotropy_dirmean.reshape(np.shape(self.X))/np.pi*180,
                        headwidth=0,headlength=0,headaxislength=0,
                        pivot='middle',width=0.005,#scale=70.,width=0.005,
                        color='yellow',edgecolor='k',#scale=80.,width=0.0035,
                        linewidth=0.5)
            
            plt.savefig(os.path.join(output_location,"result_chain%d_anisotropy.png" %self.chain_no), bbox_inches='tight')
            plt.show()

        #%%
        
        if surfacewaves.isactive:
            valid,idx_2d,prop_phslo_ray,prop_phslo_lov,nmods = swm.get_phasevel_maps(
                self.z,surfacewaves.periods,self.shape,
                vs_field=self.average_vs/self.average_model_counter,
                vpvs_field=self.average_vpvs/self.average_model_counter)

            if not valid:
                print("Warning: bad slowness field, phasevel maps may be corrupted")
              
        valid_gridpoints = self.para.convexhull.find_simplex(self.gridpoints)
        valid_gridpoints = (valid_gridpoints>-1).reshape(np.shape(self.X))
        if surfacewaves.isactive and surfacewaves.convexhull is not None:
            valid_gridpoints_sw = surfacewaves.convexhull.find_simplex(self.gridpoints[:,:2])
            valid_gridpoints_sw = (valid_gridpoints_sw>-1).reshape(np.shape(self.X))
        else:
            valid_gridpoints_sw = np.ones_like(self.X,dtype=bool)
          
        for k in range(5):
            
            if not surfacewaves.isactive and k<3:
                continue
            
            fig = plt.figure(figsize=(15,10))
            gs = gridspec.GridSpec(3,5,width_ratios=[1,1,1,1,0.05],height_ratios=[1.2,1.2,1],wspace=0.3)
            axlist = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),
                      fig.add_subplot(gs[0,2]),fig.add_subplot(gs[0,3]),
                      fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),
                      fig.add_subplot(gs[1,2]),fig.add_subplot(gs[1,3])]
            #cbarax = fig.add_subplot(gs[1,4])

            ax21 = fig.add_subplot(gs[2,0])
            ax22 = fig.add_subplot(gs[2,1])
            ax23 = fig.add_subplot(gs[2,2])
            ax24 = fig.add_subplot(gs[2,3])
            
            if k == 0:
                testidx = np.around(np.linspace(0,len(surfacewaves.periods)-1,8)).astype(int)
                periods = surfacewaves.periods[testidx]
                for i,period in enumerate(periods[::-1]):
                    ax = axlist[i]
                    ax.set_aspect("equal")
                    velfield = prop_phslo_ray[period]
                    velfield = np.ma.masked_where(np.isinf(velfield),velfield)
                    velfield = np.ma.masked_where(~valid_gridpoints_sw[:,:,0].flatten(),velfield)
                    velfield = np.reshape(1./velfield,np.shape(self.X[:,:,0]))
                    cbar = ax.pcolormesh(self.X[:,:,0],self.Y[:,:,0],
                                         #velfield,
                                         (velfield-np.mean(velfield))/np.mean(velfield)*100,
                                         cmap=plt.cm.jet_r,shading='nearest')
                    ax.text(0.75,0.05,"%.1fs" %period,fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                            transform = ax.transAxes)
                    inax = ax.inset_axes([0.05, 0.05, 0.03, 0.3])
                    plt.colorbar(cbar,cax=inax)
                    if i==0:
                        ax.set_title("Rayleigh phase velocity maps (relative velocities [%])",loc='left')
                    
            elif k==1:
                testidx = np.around(np.linspace(0,len(surfacewaves.periods)-1,8)).astype(int)
                periods = surfacewaves.periods[testidx]
                for i,period in enumerate(periods[::-1]):
                    ax = axlist[i]
                    ax.set_aspect("equal")
                    velfield = prop_phslo_lov[period]
                    velfield = np.ma.masked_where(np.isinf(velfield),velfield)
                    velfield = np.ma.masked_where(~valid_gridpoints_sw[:,:,0].flatten(),velfield)
                    velfield = np.reshape(1./velfield,np.shape(self.X[:,:,0])) 
                    cbar = ax.pcolormesh(self.X[:,:,0],self.Y[:,:,0],
                                         (velfield-np.mean(velfield))/np.mean(velfield)*100,
                                         cmap=plt.cm.jet_r,shading='nearest')
                    ax.text(0.75,0.05,"%.1fs" %period,fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                            transform = ax.transAxes)
                    inax = ax.inset_axes([0.05, 0.05, 0.03, 0.3])
                    plt.colorbar(cbar,cax=inax)
                    if i==0:
                        ax.set_title("Love phase velocity maps (relative velocities [%])",loc='left')
                        
            elif k==2:
                testidx = np.around(np.linspace(0,len(self.z)-1,8)).astype(int)
                meanmodel = np.reshape(self.average_vs/self.average_model_counter,np.shape(self.X))
                meanmodel[~valid_gridpoints] = np.nan
                for i,idx in enumerate(testidx):
                    ax = axlist[i]
                    ax.set_aspect("equal")
                    mod = meanmodel[:,:,idx]
                    cbar = ax.pcolormesh(self.X[:,:,0],self.Y[:,:,0],
                                         (mod-np.nanmean(mod))/np.nanmean(mod)*100,
                                         cmap=plt.cm.jet_r,shading='nearest')
                    ax.text(0.75,0.05,"$%.1f~km$" %self.z[idx],fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                            transform = ax.transAxes)
                    inax = ax.inset_axes([0.05, 0.05, 0.03, 0.3])
                    plt.colorbar(cbar,cax=inax)
                    if i==0:
                        ax.set_title("Shear velocity maps (relative velocities [%])",loc='left')

            elif k==3:
                testidx = np.around(np.linspace(0,len(self.z)-1,8)).astype(int)
                meanmodel = np.reshape(self.average_vs/self.average_model_counter,np.shape(self.X))
                meanmodel[~valid_gridpoints] = np.nan
                for i,idx in enumerate(testidx):
                    ax = axlist[i]
                    ax.set_aspect("equal")
                    mod = meanmodel[:,:,idx]
                    cbar = ax.pcolormesh(self.X[:,:,0],self.Y[:,:,0],mod,
                                         cmap=plt.cm.jet_r,shading='nearest')
                    ax.text(0.75,0.05,"$%.1f~km$" %self.z[idx],fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                            transform = ax.transAxes)
                    inax = ax.inset_axes([0.05, 0.05, 0.03, 0.3])
                    plt.colorbar(cbar,cax=inax)
                    if i==0:
                        ax.set_title("Shear velocity maps",loc='left')
            
            else:
                testidx = np.around(np.linspace(0,len(self.z)-1,8)).astype(int)
                # average model variance sum(model-meanmodel)**2 = 
                # = sum( model**2 - 2model*meanmodel + meanmodel**2) =
                # = sum(model**2) - sum(2*model*meanmodel) + sum(meanmodel**2)
                # model std = sqrt( variance / N)
                meanmodel = self.average_vs/self.average_model_counter
                variance = self.vssumsquared - 2*meanmodel*self.average_vs + self.average_model_counter*meanmodel**2
                valid = np.where(variance>1e-5)[0]
                modeluncertainties = np.ones_like(variance)*np.nan
                modeluncertainties[valid] = np.sqrt(variance[valid] / self.average_model_counter)
                modeluncertainties = np.reshape(modeluncertainties,self.shape)
                for i,idx in enumerate(testidx):
                    ax = axlist[i]
                    ax.set_aspect("equal")
                    mod = modeluncertainties[:,:,idx]
                    if np.isnan(mod).all():
                        vmax = 1
                    else:
                        vmax = np.nanmax(mod)/5.
                    cbar = ax.pcolormesh(self.X[:,:,0],self.Y[:,:,0],mod,
                                         vmin=0,vmax=vmax,
                                         cmap=plt.cm.cividis,shading='nearest')
                    ax.text(0.70,0.05,"%.1fkm" %self.z[idx],fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                            transform = ax.transAxes)
                    inax = ax.inset_axes([0.05, 0.05, 0.03, 0.3])
                    plt.colorbar(cbar,cax=inax)
                    if i==0:
                        ax.set_title("Model uncertainty",loc='left')

            no_cells = np.min([len(np.unique(self.collection_no_points)),25])
            ax21.hist(self.collection_no_points,no_cells)
            ax21.set_xlabel('no of points',fontsize=8)
            ax21.set_ylabel('no of models',fontsize=8)
            ax21.tick_params(axis='both', which='major', labelsize=8)
            
            if surfacewaves.isactive:
                for dataset in surfacewaves.data:
                    periods = list(surfacewaves.data[dataset])
                    mean_data_std = np.zeros(len(self.collection_datastd[dataset][periods[0]]))
                    no_data = 0.
                    for period in periods:
                        mean_data_std += (np.array(self.collection_datastd[dataset][period]) * 
                                          len(surfacewaves.data[dataset][period]))
                        no_data += len(surfacewaves.data[dataset][period])
                    mean_data_std = mean_data_std/no_data
                    ax22.hist(mean_data_std,25,alpha=0.8,label=dataset)
            if bodywaves.isactive:
                for dataset in bodywaves.data:
                    ax22.hist(self.collection_datastd[dataset],25,alpha=0.8,label=dataset)
            ax22.legend(loc='upper right',fontsize=6)
            ax22.set_xlabel('data std',fontsize=8)
            ax22.set_ylabel('no of models',fontsize=8)
            ax22.tick_params(axis='both', which='major', labelsize=8)
            
            ax23.clear()
            if self.total_steps>self.nburnin:
                if len(self.collection_loglikelihood) != self.total_steps:
                    print(f"Warning: setting self.total_steps from {self.total_steps} to {len(self.collection_loglikelihood)}.")
                    self.total_steps = len(self.collection_loglikelihood)
                ax23.plot(np.arange(int(self.nburnin/4.),self.total_steps,1),
                               self.collection_loglikelihood[int(self.nburnin/4.):])
            else:
                ax23.plot(self.collection_loglikelihood)
            ax23.set_ylabel('log likelihood',fontsize=8)
            ax23.set_xlabel('iteration no',fontsize=8)
            ax23.tick_params(axis='both', which='major', labelsize=8)
            
            ax24.text(0,0.95,"Chain number: %d" %self.chain_no)
            ax24.text(0,0.85,"Iterations: {:,}".format(self.total_steps))
            ax24.text(0,0.75,"Burnin samples: {:,}".format(self.nburnin))
            ax24.text(0,0.65,"Std proposal data uncertainty: %.3f" %self.propsigmastd)
            ax24.text(0,0.55,"Std proposal velocity update: %.3f" %np.mean(self.propvelstd_pointupdate))
            if type(self.propvelstd_dimchange)==str:
                ax24.text(0,0.45, "Std proposal birth/death: uniform")
            else:
                ax24.text(0,0.45, "Std proposal birth/death: %.1f" %np.mean(self.propvelstd_dimchange))
            ax24.text(0,0.35,"Std proposal move cell: %.3f" %np.mean(self.propmovestd))
            ax24.text(0,0.25,"Temperature: %.1f" %self.temperature)
            ax24.text(0,0.15,"Average of every %dth model" %self.collect_step)
            if self.total_steps>self.update_paths_interval:
                ax24.text(0,0.05,"Paths updated every {:,}th iteration.".format(self.update_paths_interval))
            ax24.axis('off')
            if saveplot:
                if k==0:
                    plt.savefig(os.path.join(output_location,"result_chain%d_rayleigh_phasevelocity.png" %self.chain_no), bbox_inches='tight')
                elif k==1:
                    plt.savefig(os.path.join(output_location,"result_chain%d_love_phasevelocity.png" %self.chain_no), bbox_inches='tight')
                elif k==2:
                    plt.savefig(os.path.join(output_location,"result_chain%d_dVs.png" %self.chain_no), bbox_inches='tight')
                elif k==3:
                    plt.savefig(os.path.join(output_location,"result_chain%d_Vs.png" %self.chain_no), bbox_inches='tight')
                else:
                    plt.savefig(os.path.join(output_location,"result_chain%d_modelstd.png" %self.chain_no), bbox_inches='tight')                    
                plt.close(fig)
            else:
                plt.show()
        #%%
        #self.ax24.text(0,0.1,"Iterations: %d" %self.total_steps)
        
        
        return()
          
