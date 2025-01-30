#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 17:14:27 2018

@author: emanuel
"""

#import multiprocessing
import os
import matplotlib
if not bool(os.environ.get('DISPLAY', None)):
    matplotlib.use('Agg')
from .ModelSearch import RJMC
import numpy as np
import pickle, copy
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import gzip, time, pyproj
from . import BodyWaveModule as bwm
from . import SurfaceWaveModule as swm
from scipy.sparse import vstack as sparse_vstack
import matplotlib.pyplot as plt

# CLASSES # # #
class initialize_params(object):
    
    def __init__(self, minx=None, maxx=None, miny=None, maxy=None, z=None,
                 z_scale=None,
                 xgridspacing = None, ygridspacing = None,
                 projection = None,
                 parameterization = None,
                 misfit_norm = None,
                 likelihood_distribution = None,
                 init_no_points = None,
                 init_vel_points = None,
                 propvelstd_velocity_update = None,
                 propvelstd_birth_death = None,
                 fixed_propvel_std = None,
                 propmove_std = None,
                 delayed_rejection = None,
                 number_of_iterations = None,
                 number_of_burnin_samples = None,
                 update_path_interval = None,
                 propsigma_std = None,
                 data_std_type = None,
                 sw_hyperparameter_level = None,
                 bw_hyperparameter_level = None,
                 temperature = None,
                 collect_step = None,
                 anisotropic = None,
                 propstd_anisoamp = None,
                 propstd_anisodir = None,
                 store_models = None,
                 print_stats_step = None,
                 logfile_path = None,
                 visualize = False,
                 visualize_step = None,
                 resume = False):
        
        loc = locals()
        exception = False
        for argument in loc:
            if (resume and
                argument in ["minx","maxx","miny","maxy","z","z_scale",
                             "xgridspacing","ygridspacing","projection"]):
                continue
            if loc[argument] is None and not "aniso" in argument:
                print(f"{argument} undefined in initialize_params!")
                exception = True
        if exception:
            raise Exception("Undefined parameters")
        
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        
        self.z = np.array(z).astype(float)
        self.z_scale = z_scale
        
        self.xgridspacing = xgridspacing
        self.ygridspacing = ygridspacing
           
        self.projection = projection
        
        self.parameterization = parameterization
        
        self.misfit_norm = misfit_norm
        
        self.likedist = likelihood_distribution
        
        self.init_no_points = init_no_points
        self.init_vel_points = init_vel_points
        if self.init_vel_points != 'random':
            starting_model = np.loadtxt(self.init_vel_points)
            self.startmodel_depth = starting_model[:,0]
            self.startmodel_vs = starting_model[:,1]
        else:
            self.startmodel_depth = self.startmodel_vs = None
        
        self.target_iterations = number_of_iterations
        self.nburnin = number_of_burnin_samples
        
        self.delayed_rejection = delayed_rejection
        
        self.update_paths_interval = update_path_interval
        
        if fixed_propvel_std:
            self.propvelstd = 'fixed'
        else:
            self.propvelstd = 'adaptive'
        self.propvelstd_pointupdate = propvelstd_velocity_update
        if propvelstd_birth_death!='uniform' and parameterization=='wavelets':
            self.propvelstd_dimchange = 'uniform'
            print("Warning: For the wavelet parameterization, currently only uniform proposals at birth steps are implemented. Setting birth proposal to 'uniform'.")
        else:
            self.propvelstd_dimchange = propvelstd_birth_death
        self.propmovestd = propmove_std
        self.propsigmastd = propsigma_std
        self.data_std_type = data_std_type
        self.sw_hyperparameter_level = sw_hyperparameter_level
        self.bw_hyperparameter_level = bw_hyperparameter_level
        
        self.temperature = temperature
                
        self.anisotropic = anisotropic
        self.propstd_anisoamp = propstd_anisoamp
        self.propstd_anisodir = propstd_anisodir
        
        self.collect_step = collect_step
        
        self.store_models = store_models
        
        self.print_stats_step = print_stats_step
        
        self.logfile_path = logfile_path
        
        self.visualize = visualize
        self.visualize_step = visualize_step



class initialize_prior(object):
    
    def __init__(self,
                 vs_priors=None,
                 vs_conversion=None,
                 flexible_vpvs=None,
                 vpvsmin=None,
                 vpvsmax=None,
                 moho_model=None,
                 crust_model=None,
                 min_no_points=None, max_no_points=None,
                 min_datastd=None, max_datastd=None,
                 aniso_ampmin=None, aniso_ampmax=None):

        loc = locals()
        exception = False
        for argument in loc:
            if loc[argument] is None and not ("aniso" in argument or "moho" in argument or "crust" in argument):
                print(f"{argument} undefined in initialize_prior!")
                exception = True
        if exception:
            raise Exception("Undefined priors")
            
        self.moho_model = moho_model
        self.crust_model = crust_model
            
        #self.velmin = min_velocity
        #self.velmax = max_velocity
        self.vs_priors = vs_priors
        self.vs_conversion = vs_conversion
        self.flexible_vpvs = flexible_vpvs
        self.vpvsmin = vpvsmin
        self.vpvsmax = vpvsmax
        
        self.min_datastd = min_datastd
        self.max_datastd = max_datastd
        
        if min_no_points < 3:
            print("WARNING: minimum number of points has been set to 3 for the triangulation to work.")
            min_no_points = 3
        self.nmin_points = min_no_points
        self.nmax_points = max_no_points
        
        self.aniso_ampmin = aniso_ampmin
        self.aniso_ampmax = aniso_ampmax
        
        

    
""" Class for an MPI parallelized search """
# works also if only one core is active
class parallel_search(object):
    
    def __init__(self,output_location,parallel_tempering,nchains,
                 update_path_interval,refine_fmm_grid_factor,anisotropic,
                 store_paths,save_to_disk_step,mpi_comm):
        
        self.output_location = output_location
        self.parallel_tempering = parallel_tempering
        self.no_chains = nchains        
        self.chainlist = []
        self.no_cores = mpi_comm.Get_size()
        self.update_path_interval = update_path_interval
        if np.size(update_path_interval) > 1:
            raise Exception("The paths should be updated at the same time for all chains! (Only single value for 'update_path_interval')")
        self.refine_fmm_grid_factor = refine_fmm_grid_factor
        self.anisotropic = anisotropic
        self.store_paths = store_paths
        self.save_to_disk_step = save_to_disk_step
        
        self.idle = False
        
        if self.parallel_tempering:
            
            self.accepted_swaps = 1
            self.rejected_swaps = 1
            self.loglikelihoods = np.zeros(nchains)
            self.temperatures = np.zeros(nchains)
        
        return
              
    
    def run_model_search(self,input_list,mpi_rank, mpi_size, mpi_comm, resume=False):
          
        if self.anisotropic:
            raise Exception("anisotropy is currently not implemented")
        
        time.sleep(mpi_rank/10.) # so that not all chains read at the same time
        bodywaves = bwm.read_bodywave_module(self.output_location)            
        surfacewaves = swm.read_surfacewave_module(self.output_location)
        self.targets = [surfacewaves,bodywaves]
        for target in self.targets:
            if target is None:
                raise Exception("target undefined!")
        
        # Resuming
        if resume:   
            # maybe this should also be done sequentially...
            for i,(chain_no,prior,params) in enumerate(input_list[mpi_rank::mpi_size]):
                
                fname_chain = os.path.join(self.output_location,"rjmcmc_chain%d.pgz" %(chain_no))
                if not os.path.isfile(fname_chain):
                    raise Exception(f"Trying to resume model search but file {fname_chain} cannot be found. Resuming model search failed.")
                with gzip.open(fname_chain, "rb") as f:
                    chain = pickle.load(f)
                chain.resume(self.targets,prior,params)
                self.chainlist.append(chain)

        # Starting a new model search
        else:
            if mpi_rank == 0:
                if not os.path.exists(os.path.join(self.output_location,"data_modules")):
                    os.makedirs(os.path.join(self.output_location,"data_modules"))
            # not resuming
            for i,(chain_no,prior,params) in enumerate(input_list[mpi_rank::mpi_size]):
   
                chain = RJMC(chain_no,self.targets,prior,params)
                self.chainlist.append(chain)
                     
            # distributed calculation of straight ray paths and matrices
            self.calculate_paths_and_matrices(mpi_rank, mpi_comm, rays='straight',
                                              surf=surfacewaves.A is None,
                                              body=bodywaves.A is None)
            for chain in self.chainlist:
                chain.initialize(self.targets)
          
        # Start model search
        t0 = time.time()
        mpi_comm.Barrier()
        if mpi_rank==0:
            print("\n# # # Starting model search # # #",flush=True)
        self.submit_jobs(mpi_rank,mpi_comm)
        
        if mpi_rank==0:
            print("total time:",np.around((time.time()-t0)/60.,1),"min")
        

    def dump_chain(self,chain):
        # Save chain to disk
        
        dumpchain = pickle.dumps(chain)
        dumpchain = gzip.compress(dumpchain)
        with open(os.path.join(self.output_location,"dumpchain%d.pgz" %(chain.chain_no)), "wb") as f:
            f.write(dumpchain)
        dumpchain = []
        os.rename(os.path.join(self.output_location,"dumpchain%d.pgz" %(chain.chain_no)),
                  os.path.join(self.output_location,"rjmcmc_chain%d.pgz" %(chain.chain_no)))

        print("    Chain %d: iteration %d/%d" %(chain.chain_no,chain.total_steps,chain.target_iterations),flush=True)

        
    def submit_jobs(self,mpi_rank,mpi_comm):
            
        chain_iterations = []
        chain_target_iterations = []
        nburnins = []

        for chain in self.chainlist:
            chain_iterations.append(chain.total_steps)
            chain_target_iterations.append(chain.target_iterations)
            nburnins.append(chain.nburnin)
            if self.parallel_tempering:
                self.temperatures[chain.chain_no-1] = chain.temperature
      
        nburnins = np.array(nburnins)
        
        # make sure that all chains resume at the same iteration number
        maxiterations = None
        maxiterations = mpi_comm.gather(np.max(chain_iterations),root=0)
        mpi_comm.Barrier()
        maxiterations = mpi_comm.bcast(maxiterations,root=0)
        maxiterations = np.max(maxiterations)
        if np.min(chain_iterations) != maxiterations:
            print("WARNING: MPI process",mpi_rank,"noticed that not all chains have the same number of starting iterations (",np.min(chain_iterations),maxiterations,"). This may cause unwanted behavior. Trying to catch up...",flush=True)
            for i in range(maxiterations-np.min(chain_iterations)):
                for j,chain in enumerate(self.chainlist):
                    if chain.total_steps < maxiterations:
                        acceptance,loglikelihood = chain.propose_jump(self.targets)
                        if chain.total_steps%self.save_to_disk_step == 0:
                            self.dump_chain(chain)
            print("    catching up done.")
        start_iterations = maxiterations

        for j,chain in enumerate(self.chainlist):
            print("MPI Process",mpi_rank,"working on chain:",chain.chain_no,flush=True)
        
        # make sure that the update_path_interval parameter is identical for all chains
        path_updates = None
        path_updates = mpi_comm.gather(self.update_path_interval,root=0)
        path_updates = mpi_comm.bcast(path_updates,root=0)
        if np.min(path_updates) != np.max(path_updates):
            raise Exception("Paths have to be updated at the same interval for all chains. Other options are currently not implemented. Aborting.")
            
        # communicate the minimum number of burnin steps
        min_nburnin = None
        min_nburnin = mpi_comm.gather(np.min(nburnins),root=0)
        min_nburnin = mpi_comm.bcast(np.min(min_nburnin),root=0)
        
        # communicate the maximum number of iterations to all processes
        max_iterations = None
        max_iterations = mpi_comm.gather(np.max(chain_target_iterations),root=0)
        max_iterations = mpi_comm.bcast(np.max(max_iterations),root=0)
        
        for chain in self.chainlist:
            if chain.target_iterations != (chain.total_steps + max_iterations-start_iterations):
                raise Exception("chain no:",chain.chain_no,"calculation of iterations went wrong!")
        
        
        for i in range(start_iterations,max_iterations):
                                        
            for j,chain in enumerate(self.chainlist):
                
                if chain.total_steps < chain.target_iterations:
                                                            
                    acceptance,loglikelihood = chain.propose_jump(self.targets)
                
                    if (chain.total_steps%self.save_to_disk_step == 0 or 
                        chain.total_steps == chain.target_iterations):
                        self.dump_chain(chain)

            # Recalculate Eikonal paths
            if i%self.update_path_interval == 0 and i>0:

                # the files in data_modules will be updated
                self.calculate_paths_and_matrices(mpi_rank, mpi_comm,rays='fmm')
                for chain in self.chainlist:
                    chain.update_current_likelihood(self.targets)
                        
            # parallel tempering, start after half the number of burnin steps
            # if the chains have different number of burnin steps, take the smallest number
            # currently every 100th step, maybe better every single step?
            if self.parallel_tempering and i >= 10:#min_nburnin/4.:

                maxtemp = self.swap_temperatures(mpi_rank,mpi_comm)
                
                if i%10000 == 0 and mpi_rank==0:
                    print("accepted swaps:",self.accepted_swaps)
                    print("the maximum temperature is",maxtemp)
                
            # average models from all chains to obtain faster convergence
            #self.mix_model_step = 2500005
            #if i%self.mix_model_step == 0 and i>0:
            #    self.mix_models(mpi_rank, mpi_comm)
            # probably not recommendable, as it will limit the parameter space
            # that is being explored...?
            ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##   

        # how can this happen?
        for chain in self.chainlist:
            if chain.total_steps != chain.target_iterations:
                print("warning! chain",chain.chain_no,"has not reached its target iterations",chain.total_steps,"/",chain.target_iterations)
                self.dump_chain(chain)
                

    def get_global_average_model(self,mpi_rank,mpi_comm):
        """
        This function takes the collected Vs and Vp models and calculates
        global averages.

        """
                    
        for i,chain in enumerate(self.chainlist):
            if i == 0:
                avg_vs = chain.average_vs.copy()
                avg_vp = chain.average_vp.copy()
                avg_model_counter = chain.average_model_counter
            else:
                avg_vs += chain.average_vs
                avg_vp += chain.average_vp
                avg_model_counter += chain.average_model_counter  
                
        recvbuf = None
        if mpi_rank == 0:
            recvbuf = np.empty((self.no_cores,len(chain.gridpoints)), dtype=float)
        mpi_comm.Gather(avg_vs, recvbuf, root=0)

        recvbuf2 = None
        if mpi_rank == 0:
            recvbuf2 = np.empty((self.no_cores,len(chain.gridpoints)), dtype=float)
        mpi_comm.Gather(avg_vp, recvbuf2, root=0)
            
        no_models = None
        no_models = mpi_comm.gather(avg_model_counter,root=0)
        no_models = np.sum(no_models)

        if mpi_rank == 0:
            global_average_vs = np.sum(recvbuf,axis=0)/no_models
        else:
            global_average_vs = np.empty(len(chain.gridpoints))
        mpi_comm.Bcast(global_average_vs, root=0)

        if mpi_rank == 0:
            global_average_vp = np.sum(recvbuf2,axis=0)/no_models
        else:
            global_average_vp = np.empty(len(chain.gridpoints))
        mpi_comm.Bcast(global_average_vp, root=0)
                        
        return global_average_vs,global_average_vp
    
    
    def get_average_phasevel_maps(self,surfacewaves,chain,mpi_rank,mpi_comm):
        """
        This function takes the global average Vs and VpVs models and calculates
        phase-velocity maps.

        """
        
        phasevel_ray = {}
        phasevel_lov = {}
        
        global_average_vs, global_average_vp = (
            self.get_global_average_model(mpi_rank, mpi_comm))
        
        sigma = 0.01
        # the global average model may have some vertical velocity jumps
        # so that the calculation of the phase velocities fails. In that 
        # case, smooth the velocity field to get valid phase-vel maps.
        while True:
            valid, _idx, phslo_ray,phslo_lov,_ = swm.get_phasevel_maps(
                chain.z,surfacewaves.periods,chain.shape,
                vs_field=global_average_vs,vp_field=global_average_vp)
            if valid:
                break
            global_average_vs = np.reshape(global_average_vs,chain.shape)
            global_average_vp = np.reshape(global_average_vp,chain.shape)
            global_average_vs = gaussian_filter(global_average_vs,sigma).flatten()
            global_average_vp = gaussian_filter(global_average_vp,sigma).flatten()
            sigma *= 2
            if sigma > 10:
                raise Exception("Failed to calculate the Eikonal paths because the phase velocity maps could not be calculated from the global average model.")
        if sigma > 0.01 and mpi_rank == 0:
            print("Had to smooth the Vs and the Vp fields to get valid phase velocity maps for the ray tracing (factor=%.2f)." %sigma)
            
        # the user can specify to refine the grid for the calculation of the
        # paths using the Fast Marching Method. Refining the grid gives
        # more accurate path calculations and avoids some kinks in the paths.   
        if self.refine_fmm_grid_factor:
            xgrid = np.linspace(chain.x[0],chain.x[-1],len(chain.x)*self.refine_fmm_grid_factor)
            ygrid = np.linspace(chain.y[0],chain.y[-1],len(chain.y)*self.refine_fmm_grid_factor)
            for period in surfacewaves.periods:
                func_ray = RectBivariateSpline(chain.x,chain.y,
                            phslo_ray[period].reshape(np.shape(chain.X[:,:,0])).T)
                func_lov = RectBivariateSpline(chain.x,chain.y,
                            phslo_lov[period].reshape(np.shape(chain.X[:,:,0])).T)     
                average_phslo_ray = func_ray(xgrid,ygrid).T
                average_phslo_lov = func_lov(xgrid,ygrid).T
                phasevel_ray[period] = 1./average_phslo_ray
                phasevel_lov[period] = 1./average_phslo_lov
        else:
            xgrid = chain.x
            ygrid = chain.y
            for period in surfacewaves.period:
                phasevel_ray[period] = 1./phslo_ray[period]
                phasevel_lov[period] = 1./phslo_lov[period]
    
        return xgrid,ygrid,phasevel_ray,phasevel_lov
    

    
    def calculate_paths_and_matrices(self,mpi_rank,mpi_comm,rays='straight',
                                     surf=True,body=True):
        
        # wait for all chains to do the path calculation in parallel
        mpi_comm.Barrier()
        
        for i,target in enumerate(self.targets):
            if target.type=='surfacewaves':
                isurf = i
            elif target.type=='bodywaves':
                ibody = i
        
        t0 = time.time()

        chain = self.chainlist[0]
        
        if self.no_cores > 1:
            verbose = False
        else:
            verbose = True
        
        if surf and self.targets[isurf].isactive:
            
            # remove old matrices from memory to avoid requiring double memory
            for chain in self.chainlist:
                self.targets[isurf].A = None
            
            if mpi_rank==0:
                print("\n# # # Calculating surfacewave paths. # # #\n")
                
            # for the calculation of Eikonal paths, get the average phase
            # velocity maps. For straight paths, the maps are not needed.
            if rays != 'straight':
                xgrid,ygrid,phasevel_maps_ray,phasevel_maps_lov = \
                    self.get_average_phasevel_maps(self.targets[isurf],chain,mpi_rank,mpi_comm)

            path_dictionary = {}
            path_dists = {}
            ranks = np.arange(self.no_cores)
            for period in self.targets[isurf].periods:
                                    
                for dataset in self.targets[isurf].data.keys():
                    
                    # in each iteration, a different mpi rank will collect the
                    # path data. This way, the paths are distributed approximately
                    # equal among mpi processes. Each process will then calculate
                    # the matrices based on the paths in their storage.
                    root_mpi = ranks[0]
                    
                    t1 = time.time()
                    if not (period in self.targets[isurf].data[dataset].keys()):
                        continue
                    # sorting can speed up the calculation of Eikonal paths.
                    # this is because for a given source station we only have
                    # to calculate the traveltime field once and can trace the
                    # path to all available receivers. For straight paths this
                    # does not have any benefit
                    sortind = np.lexsort((self.targets[isurf].receivers[dataset][period][:,0],
                                          self.targets[isurf].sources[dataset][period][:,1],
                                          self.targets[isurf].sources[dataset][period][:,0]))
        
                    startidx = mpi_rank*int(len(self.targets[isurf].data[dataset][period])/self.no_cores)
                    endidx = (mpi_rank+1)*int(len(self.targets[isurf].data[dataset][period])/self.no_cores)
                    if (mpi_rank+1) == self.no_cores:
                        endidx += len(self.targets[isurf].data[dataset][period])%self.no_cores

                    if endidx > 0:
                        
                        if rays == 'straight':
                            stepsize = np.min([chain.xgridspacing,chain.ygridspacing])/3.
                            paths = swm.get_paths(self.targets[isurf].sources[dataset][period][sortind][startidx:endidx],
                                                  self.targets[isurf].receivers[dataset][period][sortind][startidx:endidx],
                                                  rays='straight',stepsize=stepsize,verbose=False)
                        else:
                            if "love" in dataset:
                                model = phasevel_maps_lov[period]
                                if mpi_rank==0:
                                    print(f"    Dataset is '{dataset}' at period={period}s and using the Love phase velocity map.")
                            elif "rayleigh" in dataset:
                                model = phasevel_maps_ray[period]
                                if mpi_rank==0:
                                    print(f"    Dataset is '{dataset}' at period={period}s and using the Rayleigh phase velocity map.")
                            else:
                                raise Exception("The datataset should have one of the two wave-type keywords (love or rayleigh). Dataset name is:",dataset)
                            paths = swm.get_paths(self.targets[isurf].sources[dataset][period][sortind][startidx:endidx],
                                                  self.targets[isurf].receivers[dataset][period][sortind][startidx:endidx],
                                                  rays='fmm', x=xgrid,y=ygrid,
                                                  model=model,verbose=verbose)
                            #print("MPI Process No",mpi_rank+1,"calculating Eikonal paths from",startidx,"to",endidx,".")
                        
                    else: # if endidx <=0, meaning that there are fewer ray paths than mpi processes
                        
                        paths = np.array([])
                        
                        
                    all_paths = []
                    # gather gives a list ordered by rank (which is important!)
                    paths = mpi_comm.gather(paths, root=root_mpi)
                    
                    if mpi_rank == root_mpi:
                        for pathlist in paths:
                            all_paths = all_paths + list(pathlist)
                        if not dataset in path_dictionary.keys():
                            path_dictionary[dataset] = {}
                            path_dists[dataset] = {}
                        # undo sorting and restore original order
                        reverse_sortind = sortind.argsort()
                        if len(all_paths)>1:
                            all_paths = np.array(all_paths,dtype='object')[reverse_sortind]
                        path_dictionary[dataset][period] = all_paths
                        path_dists[dataset][period] = np.zeros(len(all_paths))
                        for pi,path in enumerate(path_dictionary[dataset][period]):
                            path_dists[dataset][period][pi] = np.sum(np.sqrt(np.diff(path[:,0])**2+np.diff(path[:,1])**2))
                            if (not np.array_equal(path[0],self.targets[isurf].sources[dataset][period][pi]) or
                                not np.array_equal(path[-1],self.targets[isurf].receivers[dataset][period][pi])):
                                print(path[0],path[-1],self.targets[isurf].sources[dataset][period][pi],self.targets[isurf].receivers[dataset][period][pi])
                                raise Exception("order is wrong!")
                        if self.store_paths:
                            if not os.path.exists(os.path.join(self.output_location,"paths")):
                                os.makedirs(os.path.join(self.output_location,"paths"))
                            np.save(os.path.join(self.output_location,"paths","%s_paths_%s_%s.npy" %(rays,dataset,period)),all_paths)
    
                    # advance the list by one, so that in the next iteration,
                    # a different mpi rank collects the path data.
                    ranks = np.roll(ranks,1)
                    
                    if not verbose and mpi_rank==0:
                        calc_time = time.time()-t1
                        print(f"    {dataset} {period} : Calculated {len(self.targets[isurf].data[dataset][period])} paths on {self.no_cores} parallel processes in {int(np.ceil(calc_time))}s.")
            
            # the path dists dictionary can already be collected from the processes
            path_dists = mpi_comm.gather(path_dists, root=0)
            if mpi_rank==0:
                self.targets[isurf].path_dists = {}
                for i in range(len(path_dists)):
                    for dataset in path_dists[i]:
                        if not dataset in self.targets[isurf].path_dists.keys():
                            self.targets[isurf].path_dists[dataset] = {}
                        self.targets[isurf].path_dists[dataset].update(path_dists[i][dataset])

            if mpi_rank == 0:
                print("Finished calculating surfacewave paths (time = %ds)" %(int(time.time()-t0)))
                t0 = time.time()
                print("\nCreating surfacewave matrices")
            mpi_comm.Barrier()
                             
            # each mpi process has a different path dictionary and calculates
            # the matrices for their respective dictionary. The matrices are 
            # joined in a later step
            A,PHI = swm.create_A_matrix(
                chain.x,chain.y,paths=path_dictionary,plotting=True,
                plotpath=os.path.join(self.output_location,"data_modules"))
   
            # matrices can be too large, which is a problem for the MPI protocol (max 2GB?)
            # A = mpi_comm.gather(A, root=0) # too large for gather
            # store matrix parts on disk
            with open(os.path.join(self.output_location,"data_modules","A_surfwaves_%d.pkl" %mpi_rank), 'wb') as f:
                pickle.dump(A, f)
            A = PHI = path_dictionary = None # empty memory
            
            # wait for all processes to save their matrix part
            mpi_comm.Barrier()
            
            # collect and join all matrix parts on rank 0
            if mpi_rank == 0:
                self.targets[isurf].A = {}
                for rank in range(self.no_cores):
                    matrix_part_path = os.path.join(self.output_location,"data_modules","A_surfwaves_%d.pkl" %rank)
                    with open(matrix_part_path, 'rb') as f:
                        A = pickle.load(f)
                        for dataset in A:
                            if not dataset in self.targets[isurf].A.keys():
                                self.targets[isurf].A[dataset] = {}
                            self.targets[isurf].A[dataset].update(A[dataset])
                    os.remove(matrix_part_path) # delete file after reading
                    A = None
                    
                # add the datahull and write the module to the disk
                self.targets[isurf].convexhull = swm.create_datahull(chain.gridpoints2d, self.targets[isurf].A)
                swm.dump_surfacewave_module(self.output_location,self.targets[isurf])
                    
            # wait until rank 0 has finished saving the matrix to the harddisk
            mpi_comm.Barrier()
            
            # matrices can be too large, which is a problem for the MPI protocol   
            # A = mpi_comm.bcast(A, root=0) # too large for broadcast
            # let all other processes read the final surfacewave module
            if mpi_rank!=0:
                time.sleep(mpi_rank/10.) # so that not all chains read from the disk at the same time
                self.targets[isurf] = swm.read_surfacewave_module(self.output_location,verbose=False)

            if mpi_rank==0:
                print("# # Finished creating surfacewave matrices (time = %ds)\n" %(int(time.time()-t0)))

                
        if body and self.targets[ibody].isactive:
            
            # remove old matrices from memory to avoid requiring double memory
            for chain in self.chainlist:
                self.targets[ibody].A = None
            
            matrix_dictionary = {}
            path_dictionary = {} # path_dictionary only needed for eikonal paths

            if rays=='straight': # no need to calculate any new paths
                
                for dataset in self.targets[ibody].paths:

                    # bodywaves.paths contains always the straight (TauP) paths
                    all_paths = self.targets[ibody].paths[dataset]
                    
                    startidx = mpi_rank*int(len(self.targets[ibody].data[dataset])/self.no_cores)
                    endidx = (mpi_rank+1)*int(len(self.targets[ibody].data[dataset])/self.no_cores)
                    if (mpi_rank+1) == self.no_cores:
                        endidx += len(self.targets[ibody].data[dataset])%self.no_cores
                    
                    if endidx>0:
                        paths = {dataset:all_paths[startidx:endidx]}
                        gridpoints = np.column_stack((chain.gridpoints[:,:2],
                                                      chain.inverse_z_scaling(chain.gridpoints[:,2])))
                        A = bwm.create_A_matrix(gridpoints,paths,verbose=verbose)
                    else:
                        A = []
                        
                    A_list = []
                    # gather gives a list ordered by rank (which is important!)
                    A_list = mpi_comm.gather(A, root=0)
                    
                    if mpi_rank == 0:
                        A_list = [A_part[dataset] for A_part in A_list if A_part!=[]]
                        matrix_dictionary[dataset] = sparse_vstack(A_list).tocsc()
                        if matrix_dictionary[dataset].shape[0] != len(all_paths):
                            raise Exception("There has been an error. Matrix size is not correct.")
                
            else: # in case new Eikonal paths are calculated
                if mpi_rank==0:
                    print("\n# # # Updating bodywave paths. # # #\n")
                    
                global_average_vs, global_average_vp = (
                    self.get_global_average_model(mpi_rank, mpi_comm))
                
                path_dictionary = {}
                projection_str = self.targets[ibody].projection
                p = pyproj.Proj(projection_str)
                
                # create a regular x,y,z grid in flat earth transformed coordinates
                x = chain.x
                y = chain.y
                z = chain.z
                X,Y,Z = np.meshgrid(x,y,z)
                Vp = np.reshape(global_average_vp, chain.X.shape)
                Vs = np.reshape(global_average_vs, chain.X.shape)
                Z_trans,Vp_trans = bwm.flat_earth_transform(Z, Vp)
                Z_trans,Vs_trans = bwm.flat_earth_transform(Z, Vs)
                z_trans = np.unique(Z_trans)
                vp_func = RegularGridInterpolator((y,x,z_trans), Vp_trans)
                vs_func = RegularGridInterpolator((y,x,z_trans), Vs_trans)
                zmax = 0.
                for dataset in self.targets[ibody].data:
                    zmax = np.max([zmax,np.max(self.targets[ibody].piercepoints[dataset][:,2])])
                z_trans_max,_ = bwm.flat_earth_transform(zmax, 1.0)
                z_regular = np.arange(np.min(z_trans),z_trans_max+10,5)
                if mpi_rank==0:
                    print("z spacing in regular grid is set to 5 km for the bodywave ray tracing.")
                Xr,Yr,Zr = np.meshgrid(x,y,z_regular)
                Vp_reg = vp_func((Yr,Xr,Zr))
                Vs_reg = vs_func((Yr,Xr,Zr))
                                
                # calculate paths between the pierce points and the seismic stations
                for i,dataset in enumerate(self.targets[ibody].data.keys()):
                    
                    if dataset[:2]=="p ":
                        Vr = Vp_reg
                    elif dataset[:2]=="s ":
                        Vr = Vs_reg
                    else:
                        raise Exception("could not recognize whether dataset is S or P (%s)" %dataset)
                    
                    t0 = time.time()                   
                    # make the stations the sources for the path calcultation
                    # because there are less sources than piercepoints
                    sources = self.targets[ibody].receivers[dataset]
                    srcx,srcy = p(sources[:,0],sources[:,1])
                    srcz,_ = bwm.flat_earth_transform(sources[:,2], np.zeros_like(sources[:,0]))
                    sources = np.column_stack((srcx/1000.,srcy/1000.,srcz))
                    # receivers are the piercepoints
                    receivers = self.targets[ibody].piercepoints[dataset].copy()
                    rcvz,_ = bwm.flat_earth_transform(receivers[:,2], np.zeros_like(receivers[:,0]))
                    receivers[:,2] = rcvz
                    
                    # sorting can speed up the calculation
                    sortind = np.lexsort((receivers[:,0],sources[:,1],sources[:,0]))
                    sources = sources[sortind]
                    receivers = receivers[sortind]
    
                    startidx = mpi_rank*int(len(self.targets[ibody].data[dataset])/self.no_cores)
                    endidx = (mpi_rank+1)*int(len(self.targets[ibody].data[dataset])/self.no_cores)
                    if (mpi_rank+1) == self.no_cores:
                        endidx += len(self.targets[ibody].data[dataset])%self.no_cores
       
                    if endidx>0:
                        paths = bwm.calculate_eikonal_paths(
                            Xr,Yr,Zr,Vr,sources[startidx:endidx],
                            receivers[startidx:endidx],verbose=mpi_rank==0,
                            Nprocs=self.no_cores,
                            Nsources=len(self.targets[ibody].data[dataset]))
                        for pi in range(len(paths)):
                            zpath,_ = bwm.flat_earth_transform(
                                paths[pi][:,2],-1e99,inverse=True)
                            paths[pi][:,2] = zpath
                            # save a bit of diskspace and reduce the amount of data shared over MPI
                            paths[pi] = paths[pi].astype('float32')
                    else:
                        paths = np.array([])
                         
                    # gather gives a list ordered by rank (which is important!)
                    paths = mpi_comm.gather(paths, root=0)
                    
                    all_paths = []
                    if mpi_rank == 0:
                        for pathlist in paths:
                            all_paths = all_paths + list(pathlist)
                        # undo sorting and restore original order
                        reverse_sortind = sortind.argsort()
                        all_paths = [all_paths[i] for i in reverse_sortind]
                        for pi,path in enumerate(all_paths):
                            if not np.allclose(path[-1],self.targets[ibody].piercepoints[dataset][pi]):
                                raise Exception("order is wrong!")
                        if self.store_paths:
                            if not os.path.exists(os.path.join(self.output_location,"paths")):
                                os.makedirs(os.path.join(self.output_location,"paths"))
                            np.save(os.path.join(self.output_location,"paths","eikonal_paths_%s.npy" %(dataset)),all_paths)
       
                    # all_paths contains all paths for a certain dataset in correct order
                    all_paths = mpi_comm.bcast(all_paths, root=0)
                    path_dictionary[dataset] = all_paths
                    
                    if mpi_rank == 0:
                        print("Finished calculating bodywave Eikonal paths for dataset %s (time = %ds)" %(dataset,int(time.time()-t0)))
                        t0 = time.time()
                        print("Creating corresponding matrix.")
                        
                    # split up the workload again so that every chain creates
                    # one part of the matrix
                    if endidx>0:
                        paths = {dataset:all_paths[startidx:endidx]}
                        gridpoints = np.column_stack((chain.gridpoints[:,:2],
                                                      chain.inverse_z_scaling(chain.gridpoints[:,2])))
                        A = bwm.create_A_matrix(gridpoints,paths,verbose=verbose)
                    else:
                        A = []
                        
                    A_list = []
                    # gather gives a list ordered by rank (which is important!)
                    A_list = mpi_comm.gather(A, root=0)
                    
                    # join the matrix parts again into one single matrix
                    if mpi_rank == 0:
                        A_list = [A_part[dataset] for A_part in A_list if A_part!=[]]
                        matrix_dictionary[dataset] = sparse_vstack(A_list).tocsc()
                        if matrix_dictionary[dataset].shape[0] != len(all_paths):
                            raise Exception("There has been an error in the MPI communication. Matrix size is not correct.")
                    
            matrix_dictionary = mpi_comm.bcast(matrix_dictionary, root=0)
            self.targets[ibody].A = matrix_dictionary
            self.targets[ibody].paths_eikonal = path_dictionary
            
            if mpi_rank == 0:
                bwm.dump_bodywave_module(self.output_location,self.targets[ibody])
                print("# # Finished creating bodywave matrices (time = %ds)\n" %(int(time.time()-t0)))
                
            # create some example plots of the body-wave Eikonal paths
            if mpi_rank == 0 and rays!='straight' and True:
                print("plotting paths")
                vpfunc = RegularGridInterpolator((chain.y,chain.x,chain.z),Vp,
                                                 bounds_error=False)
                vsfunc = RegularGridInterpolator((chain.y,chain.x,chain.z),Vs,
                                                 bounds_error=False)
                fig_dir = os.path.join(self.output_location,"figures_bodywavepaths")
                if not os.path.exists(fig_dir):
                    os.makedirs(fig_dir)
                for dataset in path_dictionary:
                    for pi,path in enumerate(path_dictionary[dataset]):
                        if pi%1000!=0:
                            continue
                        old_path = self.targets[ibody].paths[dataset][pi]
                        # find a straight line in the x-y plane along the path
                        pnt1 = self.targets[ibody].paths[dataset][pi][0]
                        pnt2 = self.targets[ibody].paths[dataset][pi][-1]
                        m = (pnt2[1] - pnt1[1]) / (pnt2[0] - pnt1[0])
                        t = pnt1[1] - m*pnt1[0]
                        plt.ioff()
                        fig = plt.figure(figsize=(16,6))
                        ax1 = fig.add_subplot(1,3,1)
                        ax2 = fig.add_subplot(1,3,2)
                        ax3 = fig.add_subplot(1,3,3)
                        ax1.plot(old_path[:,0],old_path[:,1],label='TauP path')
                        ax1.plot(path[:,0],path[:,1],label='Eikonal path')
                        Xsec,Zsec = np.meshgrid(chain.x,chain.z)
                        y = m*Xsec.flatten() + t
                        if dataset[:2]=="p ":
                            func = vpfunc
                        else:
                            func = vsfunc
                        Vsec = func(np.column_stack((y,Xsec.flatten(),Zsec.flatten())))
                        Vsec = np.reshape(Vsec,Xsec.shape)
                        dVsec = (Vsec.T-np.nanmean(Vsec,axis=1))/np.nanmean(Vsec,axis=1)*100
                        ax2.pcolormesh(Xsec,Zsec,dVsec.T,vmin=-5,vmax=5,cmap=plt.cm.seismic_r)
                        ax2.plot(old_path[:,0],old_path[:,2],label='TauP path')
                        ax2.plot(path[:,0],path[:,2],label='Eikonal path')
                        Ysec,Zsec = np.meshgrid(chain.y,chain.z)
                        x = (Ysec.flatten() - t ) / m
                        Vsec = vpfunc(np.column_stack((Ysec.flatten(),x,Zsec.flatten())))
                        Vsec = np.reshape(Vsec,Ysec.shape)
                        dVsec = (Vsec.T-np.nanmean(Vsec,axis=1))/np.nanmean(Vsec,axis=1)*100
                        ax3.pcolormesh(Ysec,Zsec,dVsec.T,vmin=-5,vmax=5,cmap=plt.cm.seismic_r)
                        ax3.plot(old_path[:,1],old_path[:,2],label='TauP path')
                        ax3.plot(path[:,1],path[:,2],label='Eikonal path')
                        #ax1.plot(chain.x,m*chain.x+t,'--',lw=0.5,zorder=-2)
                        ax1.plot((chain.y-t)/m,chain.y,'k--',lw=0.5,zorder=-2)
                        ax1.set_xlabel('x')
                        ax1.set_ylabel('y')
                        ax1.set_xlim([np.min(chain.x),np.max(chain.x)])
                        ax1.set_ylim([np.min(chain.y),np.max(chain.y)])
                        ax1.set_aspect('equal')
                        ax1.legend(loc='upper right')
                        ax2.set_xlabel('x')
                        ax2.set_ylabel('z')
                        ax2.set_xlim([np.min(chain.x),np.max(chain.x)])
                        ax2.set_ylim([self.targets[ibody].model_bottom_depth,np.min(chain.z)])
                        ax2.set_aspect('equal')
                        ax2.legend(loc='upper right',fontsize=6)
                        ax3.set_xlabel('y')
                        ax3.set_ylabel('z')
                        ax3.set_xlim([np.min(chain.y),np.max(chain.y)])
                        ax3.set_ylim([self.targets[ibody].model_bottom_depth,np.min(chain.z)])
                        ax3.set_aspect('equal')
                        ax3.legend(loc='upper right',fontsize=6)
                        plt.savefig(os.path.join(fig_dir,"paths_%d %s.jpg" %(pi,dataset)),bbox_inches='tight',dpi=200)
                        plt.close(fig)
                        
            # just to be sure that the information in the self.targets array is correctly updated
            for target in self.targets:
                if target.type=='bodywaves':
                    for dataset in target.A:
                        if not np.array_equal(target.A[dataset].data,matrix_dictionary[dataset].data):
                            raise Exception("list is not updated correctly!")
                        
        # not necessary, but cleaner printout
        mpi_comm.Barrier()
                
    
    

    def swap_temperatures(self,mpi_rank,mpi_comm):
        
        #mpi_comm.Barrier()
        #t0 = t00 = time.time()
        # on my 4 core CPU such a swapping operation took around 0.0003s
        
        # # test with eq. A(10) of Sambridge (2014)
        # for target in self.targets:
        #     if target.type=='bodywaves':
        #         bodywaves=target
        #     elif target.type=='surfacewaves':
        #         surfacewaves=target
        # for chain in self.chainlist:
        #     if surfacewaves.isactive:
        #         phi_sw = swm.get_phi(surfacewaves.data,
        #                              chain.phttime_predictions,
        #                              surfacewaves.path_dists,
        #                              chain.datastd)
        #     if bodywaves.isactive:
        #         phi_bw = bwm.get_phi(bodywaves.data,chain.p_ttimes,
        #                              chain.datastd,bodywaves.M,)
        #     phi = phi_bw+phi_sw
        # # log acceptance prob would then be:
        # acc_prob2 = (phi[pair[1]] - phi[pair[0]])*(1./temps[pair[1]]-1./temps[pair[0]])


        # Based on Blatter et al. (2018), their eq. (5) "Transdimensional inversion..."        
        sendbuf = np.zeros((len(self.chainlist),3))
        for i,chain in enumerate(self.chainlist):
            sendbuf[i] = np.array([chain.chain_no,chain.temperature,chain.loglikelihood_current])

        recvbuf = None
        if mpi_rank == 0:
            recvbuf = np.empty((self.no_chains,3), dtype=float)
        mpi_comm.Gather(sendbuf,recvbuf, root=0)

        if mpi_rank==0:
            
            chain_no = recvbuf[:,0]
            temps = recvbuf[:,1]
            likes = recvbuf[:,2]
            
            chain_indices = np.arange(len(temps))
            np.random.shuffle(chain_indices)
            
            #print("temperatures:",self.temperatures)
            #print("loglikelihoods:",self.loglikelihoods)
        
            for pair in chain_indices[:int(len(chain_indices)/2.)*2].reshape(int(len(chain_indices)/2),2):
                
                if temps[pair[0]] == temps[pair[1]]:
                    continue
                
                # log of eq. (5) of Blatter et al. (2018)
                if np.log(np.random.rand(1)) <= ((likes[pair[1]]    - likes[pair[0]]) *
                                                 (1./temps[pair[0]] - 1./temps[pair[1]])):
                    
                    #swap:
                    temps[pair[0]],temps[pair[1]] = temps[pair[1]],temps[pair[0]]
                    self.accepted_swaps += 1
                    #print("Swapping temperatures:\nchain:",pair[0]+1,"temp:",self.temperatures[pair[1]],"loglikelihood:",self.loglikelihoods[pair[1]],
                    #                            "\nchain:",pair[1]+1,"temp:",self.temperatures[pair[0]],"loglikelihood:",self.loglikelihoods[pair[0]])
                else:
                    self.rejected_swaps += 1
               
            # this is not useful, if we lower the temperatures, the likelihood
            # for a swap is increased, however, the chains will not be any
            # more explorative which may lead to all chains being trapped in 
            # local minima without any swaps
            # # the temperatures are modified so that we get an acceptance rate
            # # of 25% for temperature swaps.
            # if self.accepted_swaps/self.rejected_swaps > 1/4.+0.05:
            #     sortidx = temps.argsort()
            #     reverse_sortidx = sortidx.argsort()
            #     temps = temps[sortidx]
            #     newtemps = np.array(list(np.around(np.logspace(np.log10(1),np.log10(np.max(temps)*1.1),num=len(temps[temps>1.])+1),2)))
            #     temps[temps>1.] = newtemps[1:]
            #     temps = temps[reverse_sortidx]
            # elif self.accepted_swaps/self.rejected_swaps < 1/4.-0.05:
            #     if np.max(temps)*0.9 > 1.:
            #         sortidx = temps.argsort()
            #         reverse_sortidx = sortidx.argsort()
            #         temps = temps[sortidx]
            #         newtemps = np.array(list(np.around(np.logspace(np.log10(1),np.log10(np.max(temps)*0.9),num=len(temps[temps>1.])+1),2)))
            #         temps[temps>1.] = newtemps[1:]
            #         temps = temps[reverse_sortidx]
        
        #self.temperatures = mpi_comm.bcast(self.temperatures, root=0)            
        # this should also in priniple maintain the order
        # temps = mpi_comm.scatter(temps,root=0)
        
        if mpi_rank == 0:
            swaplist = np.column_stack((chain_no,temps))
        else:
            swaplist = None
            #swaplist = np.zeros((self.no_chains,2))
        swaplist = mpi_comm.bcast(swaplist,root=0)
        for chain in self.chainlist:
            #print(chain.temperature)
            chain.temperature = swaplist[swaplist[:,0]==chain.chain_no,1][0]
        
        
        #if mpi_rank==0:
        #    print(time.time()-t0)
        
        # This is slower on my 4 core PC, but maybe faster for more parallel
        # processes?
        """
        mpi_comm.Barrier()
        t0 = time.time()
        pairs = None
        probs = None
        if mpi_rank == 0:
            chain_indices = np.arange(self.no_chains)
            np.random.shuffle(chain_indices)
            pairs = chain_indices[:int(len(chain_indices)/2.)*2].reshape(int(len(chain_indices)/2),2)
            probs = np.random.rand(len(pairs))
        pairs,probs = mpi_comm.bcast((pairs,probs),root=0)
        
        pair_idx,swap_idx = np.where(pairs==mpi_rank)
        
        if len(pair_idx)>0:
            
            pair = pairs[pair_idx][0]
            if swap_idx == 0:
                partner_rank = pair[1]
            else:
                partner_rank = pair[0]
                    
            temp,like = mpi_comm.sendrecv((self.chain.temperature,self.chain.loglikelihood_current),partner_rank)
            
            
            if np.log(probs[pair_idx]) <= ((like-self.chain.loglikelihood_current) * 
                                           (1./temp - 1./self.chain.temperature)):
                
                self.chain.temperature = temp
                self.accepted_swaps += 1     
                
        mpi_comm.Barrier()
        if mpi_rank==0:
            print(time.time()-t0)
            print("")
        """
        return np.max(swaplist[:,1])
                    
    
 
    # def swap_parameterization(self,mpi_rank,mpi_comm):
 
    #     if self.chain.interpolation_type != 'nearest_neighbor':
    #         print("Warning: swap parameterization is not properly implemented for interpolation methods other than nearest neighbor (Voronoi cells).")
               
    #     pairs = None
    #     if mpi_rank == 0:
    #         chain_indices = np.arange(self.no_chains)
    #         np.random.shuffle(chain_indices)
    #         pairs = chain_indices[:int(len(chain_indices)/2.)*2].reshape(int(len(chain_indices)/2),2)
    #     pairs = mpi_comm.bcast(pairs,root=0)
        
    #     pair_idx,swap_idx = np.where(pairs==mpi_rank)
        
    #     if len(pair_idx)>0:
            
    #         pair = pairs[pair_idx][0]
    #         if swap_idx == 0:
    #             partner_rank = pair[1]
    #         else:
    #             partner_rank = pair[0]

            
    #         self.chain.points = mpi_comm.sendrecv(self.chain.points,partner_rank)
    #         self.chain.gpts_idx = mpi_comm.sendrecv(self.chain.gpts_idx,partner_rank)
    #         self.chain.tri = mpi_comm.sendrecv(self.chain.tri,partner_rank)
            
    #         self.chain.prop_model_slowness = np.ones_like(self.chain.model_slowness)
                                               
    #         for i in range(len(self.chain.gpts_idx)):
    #             if len(self.chain.gpts_idx[i])>0:
    #                 mean_slow = np.mean(self.chain.model_slowness[self.chain.gpts_idx[i]])
    #                 self.chain.prop_model_slowness[self.chain.gpts_idx[i]] = mean_slow
    #                 self.chain.points[i,3] = 1./mean_slow
                
    #         self.chain.update_current_likelihood(slowfield = self.chain.prop_model_slowness)
            
    #     mpi_comm.Barrier()
                

    # # this is exactly the same as the swap_parameterization function just
    # # a bit faster
    # def swap_model(self,mpi_rank,mpi_comm):

    #     if self.chain.interpolation_type != 'nearest_neighbor':
    #         print("Warning: swap model is not properly implemented for interpolation methods other than nearest neighbor (Voronoi cells).")

    #     pairs = None
    #     if mpi_rank == 0:
    #         chain_indices = np.arange(self.no_chains)
    #         np.random.shuffle(chain_indices)
    #         pairs = chain_indices[:int(len(chain_indices)/2.)*2].reshape(int(len(chain_indices)/2),2)
    #     pairs = mpi_comm.bcast(pairs,root=0)
        
    #     pair_idx,swap_idx = np.where(pairs==mpi_rank)
        
    #     if len(pair_idx)>0:
            
    #         pair = pairs[pair_idx][0]
    #         if swap_idx == 0:
    #             partner_rank = pair[1]
    #         else:
    #             partner_rank = pair[0]
                
            
    #         self.chain.prop_model_slowness = mpi_comm.sendrecv(self.chain.model_slowness,partner_rank)
                   
    #         for i in range(len(self.chain.gpts_idx)):
    #             if len(self.chain.gpts_idx[i])>0:
    #                 mean_slow = np.mean(self.chain.prop_model_slowness[self.chain.gpts_idx[i]])
    #                 self.chain.model_slowness[self.chain.gpts_idx[i]] = mean_slow
    #                 self.chain.points[i,3] = 1./mean_slow

    #         if np.isnan(self.chain.points[:,3]).any():
    #             raise Exception("MPI rank",mpi_rank,"has a nan in point distribution after model swap and updating")                
                
    #         chain.update_current_likelihood()
            
    #     mpi_comm.Barrier()
        
        

    def plot(self, chainno=None, saveplot=False):

        if chainno is None:
            
            for chain in self.chainlist:
                
                chain.plot(self.targets, saveplot = saveplot,
                           output_location = self.output_location)

        else:
            
            if np.shape(chainno) == ():
                chainno = [chainno]
            
            for chain in self.chainlist:
                
                if chain.chain_no in chainno:
                    
                    chain.plot(self.targets, saveplot = saveplot,
                               output_location = self.output_location)




######################
# # FUNCTIONS

def check_list(variable,index,no_searches):

    if np.shape(variable) == ():
        return variable
    elif len(variable) == no_searches:
        return variable[index]
    elif len(variable) < no_searches:
        if index >= len(variable):  
            print("Warning! More model searches than assigned parameters. "
                  "The same parameter as for model search number 1 will be assigned.")
            return variable[0]
        else:
            return variable[index]
    elif len(variable) > no_searches:
        print("Warning! Less model searches (%d) than assigned parameters (%d). " %(no_searches,len(variable)))
        print("All additional parameters are being ignored.",variable)
        return variable[index]
    else:
        print(variable)
        print(len(variable))
        print(no_searches)
        print(index)
        raise Exception("Unknown error in the check_list function in rjtransdim_helper_classes.")


