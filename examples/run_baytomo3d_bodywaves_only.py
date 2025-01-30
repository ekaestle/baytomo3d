#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import numpy as np

####################################################### 
## USER DEFINED PARAMETERS ## 
#######################################################

plot = True # plot input data and results

# number of searches/chains that will be executed on the number of cores
# defined in mpirun -np no_cores
# If this number is lower than the assigned number of cores, it will be
# automatically increased to the available number of cores so no cores are idle.
# If it is greater than no_cores, the jobs will be split so that the workload
# is equally distributed between no_cores cores.
no_searches = 4

# folder where a separate file for each search is created (careful, older files
# will be overwritten when a new search is started, unless resume_job=True)
output_location = "./modelsearch_bodywaves"

# if resume_job = True, the program will try to read existing files in the
# output_location folder and continue with additional iterations until the
# total number_of_iterations as defined below is reached.
# the follwing parameters can be changed when resuming the job, all others are 
# ignored: priors, proposal_stds, nburnin, update_paths, anisotropic_search,
# collect_step, print_stats_step 
# if output_location does not exists, it is automatically set to False
resume_job = False




# # # # # INPUT DATA SURFACE WAVES # # # # # # #
# # # # SURFACE WAVE PHASE TRAVELTIMES # # #

# The input file should contain (at least) 5 columns
# LAT1 LON1 LAT2 LON2 TRAVELTIME
# the 6th column, if available, is interpreted as traveltime standard deviation
# LAT1 LON1 LAT2 LON2 TRAVELTIME STD
# if the input file contains more than 6 columns, the additional columns are
# interpreted as traveltime values at different periods. If there is no measurement
# for a certain source-receiver pair, give nan in the input file. The traveltime
# standard deviation is always assumed to be in the last column.
# LAT1 LON1 LAT2 LON2 TRAVELTIME_1 TRAVELTIME_2 TRAVELTIME_3 ... TRAVELTIME_N STD
# make sure that the number of traveltime columns matches the number of periods 
# listed in the header (see example files)
input_files = [] # empty, because no surfacewave data is used
# choose between 'latlon' and 'xy' (in km) for the input coordinate system
coordinates = 'latlon'



# # # # # INPUT DATA BODY WAVES # # # # # # #
# # # # BODY WAVE ARRIVAL DATA # # #

# the module expects a folder with xml files that contains
# obspy QuakeML data. Can also be a list with several folders.
xml_folder = ["dataset_bodywaves/p_picks_small_dataset",
              "dataset_bodywaves/s_picks_small_dataset",
]

# give the bottom depth of the model area, relevant only for bodywaves.
# Outside the model area, rays are traced with TauP. If this depths is 
# greater than the maximum z-axis depth, a depthlevel is added to fill the gap.
model_bottom_depth = 600 # km

# if available, give a file where the station coordinates are stored
# station_id latitude longitude (e.g. NET.STA 33.814 104.3)
# otherwise the program tries do download the necessary information automatically
# and a station textfile is created
stationfile = None





# # # # SEARCH PARAMETERS # # # # # 

# desired grid spacing in x and y direction. The point model will be 
# interpolated on a regular grid and the forward problem is solved on this grid.
xgridspacing = 20 #in km
ygridspacing = 20 #in km

# zaxis should have denser sampling at shallow depth for surface waves. 
# Below the last depth level, a halfspace layer is added automatically.
# Bottom depth can correspond to model_bottom_depth, but is not necessary
zaxis = [0,1,2,4,6,8,10,13,16,20,24,28,32,36,40,45,50,60,70,80,90,100,]+list(np.arange(115,600,15))


# ANY OF THE FOLLOWING PARAMETERS CAN ALSO BE A LIST OF LENGTH no_searches
# IN THIS CASE, THE SEARCHES ARE PERFORMED WITH DIFFERENT PARAMETERS
# e.g. no_searches = 3, number_of_burnin_samples = [2000,5000,6000]
# will start 3 searches, with different numbers of burnin samples

# currently this affects the shallow levels down to 100km [more documentation needed]
# see explanation in Kaestle et al. 2024 ("Alpine crust and mantle...", JGR Solid Earth)
z_scale = 10

# total number of iterations
number_of_iterations = 500000

# number of burnin samples, these samples are discarded
number_of_burnin_samples = 200000 # has to be smaller than number_of_iterations

# Model parameterization
parameterization = 'voronoi' # choose from 'voronoi', 'wavelets'

# number of points (i.e. voronoi cells) at the beginning (can also be 'random',
# then for each search, a random number in the prior range (below) is chosen)
init_no_points = np.linspace(5000,8000,no_searches).astype(int)

# initial point velocity (can also be 'random' or a vector of the length
# of init_no_points)
init_vel_points = 'random' # 'random' or filepath to starting model file

# misfit norm, choose between L1 and L2, same for all datasets
misfit_norm = 'L2' # standard is L2, L1 is more resistant to outliers

# Likelihood distribution. Same for all datasets
# Choose from
# 'normal': the probability density function of the data standard deviation is 
# assumed to follow a gaussian distribution.
# 'outliers': approach of Tilmann et al. (2020) ["Another look at the treatment
# of data uncertainty in...]. Including outliers in the gaussian error model so
# that they have a lower impact on the final result. This results in 1 
# additional search parameter, the fraction of outliers (between 0 - 100%).
# 'students_t': use a student's t-distribution, which approaches a gaussian for
# the degree of freedom -> infinity, and otherwise has a longer tail compared
# to a gaussian which reduces the influence of outliers (not yet implemented
# for body waves)
likelihood_distribution = 'normal' #['students_t','outliers','normal']*10

# At each iteration the model is perturbed by one of the following operations:
# Changing the velocity in one cell (velocity update)
# Moving the position of one cell (move)
# Creating or destroying one cell (birth/death)
# For each of these operations, the updated value is drawn from a Gaussian
# distribution, defined by the proposal standard deviation that controls
# how far the updated value deviates from the previous model. These values
# can be fixed (normal case) or the algorithm can try to adapt them automatically
# to obtain a good acceptance rate (optimally around 45%).

# if False, the algorithm will modify the proposal stds. The values below only
# serve as starting values in this case. Otherwise the proposal stds are fixed.
fixed_propvel_std = False # choose from False, True

# standard deviation when proposing a velocity update, large values are more
# explorative, but the model may not converge, low values can cause a very
# slow convergence
proposal_std_velocity_update = 0.1 # recommended 0.1 (units of km/s)
# standard deviation when proposing the velocity of a newborn point or when
# deleting a point. Also used for drawing new vpvs values if ratio not fixed.
proposal_std_birth_death = 0.1 # ['uniform',0.1]*2 
# standard deviation when proposing to move a point to a new location
proposal_std_move = 20. # units are same as x,y units (km)

# If delayed_rejection = True, a bad velocity update or move operation is not
# directly rejected, but a second try with lower std is performed, improving
# the convergence rate. See PhD Thesis of Thomas Bodin for more details.
delayed_rejection = False

# The data standard deviation can also be a search parameter (hierarchical
# algorithm). In this case choose a standard deviation for proposing a std update
proposal_std_data_uncertainty = 0.05 # has no effect if data_std_type = 'fixed'
# The data std can be 'fixed' (not hierarchical), 'absolute' (same std for all data
# points), 'relative' (std follows a linear relationship scaling with distance)
data_std_type = 'absolute' # choose from 'fixed','absolute','relative'

# If the level is 'dataset', then the same hyperparameters are used for all 
# periods within the same dataset. If 'period' is chosen, the hyperparameters
# are different at different periods.
# This has no effect of data_std_type is "fixed"
sw_hyperparameter_level = 'period' # 'dataset' or 'period'

# The hyperparameters for the body-wave datasets can be the same for the entire
# dataset (e.g. for all S-traveltime measurements) or there can be individual
# combinations of hyperparameters for each earthquake event in the dataset
# This has no effect of data_std_type is "fixed"
bw_hyperparameter_level = 'dataset' # 'dataset' or 'event'

# Parallel tempering (temperatures are exchanged between chains)
# If parallel tempering is switched off (False) you can still assign a temp-
# erature to each chain. But there will be no temperature exchanges.
parallel_tempering = False # choose between True, False 

# Chain temperatures. Chains running on higher temperatures are more explorative
# and accept more 'bad' model perturbations
# If 'auto' then chain temperatures are assigned according to Atchadé et al. (2011)
# This results in a logarithmic increase from 1 to 1000 (too high in my opinion)
#temperatures = 1# 'auto' Ti = 10**(3*(i − 1)/(n − 1)), (i = 1, …, n) as proposed by Atchadé et al. (2011)
if no_searches>1 and parallel_tempering:
    # This is my recommendation: 2/3 of the chains run with temperature=1.,
    # the rest has logarithmically increasing temperatures from 1 - 20
    temperatures = list(np.append(np.ones(int(no_searches*(1/4.))),
                                  np.around(np.logspace(np.log10(1),np.log10(50),num=no_searches-int(no_searches*(1/4.))),2)))
else:
    # Or fix all temperatures to 1 (standard case, no temperature values)
    temperatures = 1.

# the ray paths are recalculated using the Fast Marching Method (FMM) every
# update_path_interval iteration (if it is greater than the no of iterations
# it will never update the paths, i.e. use only straight paths)
update_path_interval = int(999999999)

# When Eikonal paths (FMM method) are calculated, it is often good to refine
# the grid. Especially when the grid spacing is very coarse or the velocity
# model very rough. A higher refine_factor will cost more calculation time.
refine_fmm_grid_factor = 2 # can be 1 if x/y-gridspacing is very small, otherwise between 2 and 10

# only every collect_step model (after burnin) is used to calculate the average model
# collect step should be large enough so that models can be considered independent
collect_step = 1000


# parameters that do not influence the result of the model search

# The paths can be stored as an array for plotting purposes or to check that
# the paths actually look correct. If False, the surface wave ray paths are
# discarded after the A matrix has been set up. This saves a bit of disk space.
store_paths = False # choose from True, False

# all models can be stored as complete models interpolated on the xyz grid
# this requires more disk space.
store_models = False

# print status information every print_stats_step iteration to log file
print_stats_step = 10000

# save chains every save_to_disk_step iterations to hard disk
save_to_disk_step = 50000

visualize = False # visualize progress (slower, only for testing)
visualize_step = 100 # visualize every 'visualize_step' iteration



 
# # # # # # PRIORS # # # # # # 

# give a Vs prior model containing the following columns:
# Depth [km] Vsmin [km/s] Vsmax [km/s]
# Note that in the current implementation, Voronoi cells have constant Vs, Vp/Vs
# Therefore, values at individual gridcells can be outside the prior range if
# the cell is large and spans a large depth range.
vs_priors_file = "vs_priors_loose.txt"
# give a file for the conversion of Vs to Vp and to density containing these lines
# Vs[km/s] Vp/Vs Density[g/m³]
vs_vp_conversion_file = "vs_vp_rho_conversion.txt"
# If True, then the Vp/Vs ratio is a search parameter. In this
# case, the Vp values in the vs_vp conversion file are ignored
flexible_vpvs = True # choose between True, False
# the prior on the vpvs ratio is only used when flexible_vpvs=True
min_vpvs = 1.5
max_vpvs = 2.0

# provide a Moho depth file with 3 columns: lat, lon, Moho depth [km] ( if 
# coordinates = 'latlon') or x, y, Moho depth [km] (if coordinates = 'xy')
moho_depth_file = None # 'moho_model_eastalps.txt' # 'filepath' or None

# provide a crustal model with 5 columns: lat, lon, depth, vp, vs
# The crustal model will be interpolated on the regular grid defined above and
# the velocities at the crustal gridpoints will remain fixed and unchanged by
# any model updates.
crustal_model_file = 'crust_from_paffrath2021b.txt' #  'path/to/file.txt' or None

# ANY OF THESE CAN ALSO BE A LIST OF LENGTH no_searches
# IN THIS CASE, THE SEARCHES ARE PERFORMED WITH DIFFERENT PRIORS

# allowed number of points (Voronoi cells in case of nearest neighbor interpolation)
min_no_points = 10
max_no_points = 10000
# the following range is only relevant if the data standard deviation is treated
# as an unknown (hierarchical approach). If data_std_type='fixed', it is ignored.
min_datastd = 0.01 # expected minimum standard deviation of the input data
max_datastd = 2.0 # expected maximum standard deviation of the input data


""" # # # # # ANISOTROPIC SEARCH # # # # # """
# currently not implemented
anisotropic_search = False

####################################################### 
## END OF USER DEFINED PARAMETERS ## 
#######################################################



from mpi4py import MPI
import matplotlib
import os
if not bool(os.environ.get('DISPLAY', None)):
    matplotlib.use('Agg')
# import baytomo3d
from baytomo3d.AverageModels import get_avg_model
import baytomo3d.HelperClasses as rhc
import baytomo3d.BodyWaveModule as bwm
import baytomo3d.SurfaceWaveModule as swm
#from mpl_toolkits.basemap import cm
import pyproj

if __name__ == '__main__': 

    # Initialize MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size() 
    
    if mpi_size > no_searches:
        if mpi_rank == 0:
            print("Warning! The number of searches has been increased to the number of assigned MPI processes")
            print("Initial number of searches: %d, now: %d" %(no_searches,mpi_size))
        no_searches = mpi_size

    if not os.path.exists(output_location):
        mpi_comm.Barrier()
        if resume_job:
            if mpi_rank==0:
                print("Warning: Cannot resume job because output_folder does not exist. Treating as new job.")
            resume_job = False
        if mpi_rank==0:
            os.makedirs(output_location)
            try: # does not work if called from python interpreter
                with open(__file__,"r") as f:
                    lines = f.readlines()
                    outlines = []
                    readline = False
                    for line in lines:
                        if "USER DEFINED PARAMETERS" in line:
                            if readline:
                                break
                            else:
                                readline = True
                        if readline:
                            outlines.append(line)
                with open(os.path.join(output_location,"model_search_setup.txt"),"w") as f:
                    f.writelines(outlines)
            except:
                pass

    # Set up the Parallel Model Search class which will contain the individual
    # chains and take care of communication between mpi processes.
    modelsearch = rhc.parallel_search(output_location,
                                      parallel_tempering,
                                      no_searches,
                                      update_path_interval,
                                      refine_fmm_grid_factor,
                                      anisotropic_search,
                                      store_paths,
                                      save_to_disk_step,
                                      mpi_comm)
    
    if not resume_job:
        
        """ READ INPUT DATA """          
        # Check and read body-wave data
        bodywave_data = bwm.read_data(
            xml_folder,mpi_rank,mpi_comm,model_bottom_depth=model_bottom_depth,
            stationlist=stationfile,plotting=plot,
            output_folder=output_location)
        
        if mpi_rank==0:
            
            # make sure that both datasets use the same projection
            if bodywave_data.isactive:
                projection_str = bodywave_data.projection
            else:
                projection_str = None
            
            surfacewave_data = swm.read_data(
                input_files,coordinates=coordinates,plotting=plot,
                projection = projection_str,
                output_folder=output_location)
                 
            targets =  [bodywave_data,surfacewave_data]
        
            # regular grid in projected coordinates
            minx, miny = 1e99,1e99
            maxx, maxy = -1e99,-1e99
            projection_str = None
            for target in targets:
                if target.isactive:
                    minx = np.min([target.minx - 3*xgridspacing,minx])
                    maxx = np.max([target.maxx + 3*xgridspacing,maxx])
                    miny = np.min([target.miny - 3*ygridspacing,miny])
                    maxy = np.max([target.maxy + 3*ygridspacing,maxy])
                    if projection_str is None:
                        projection_str = target.projection
                    elif projection_str != target.projection:
                        raise Exception("Error! Projections have to be identical.",projection_str,target.projection)
            p = pyproj.Proj(projection_str)
                    
            # free up memory
            print("free up memory")
            bodywave_data.A = None
            surfacewave_data.A = None
            del targets
            del bodywave_data
            del surfacewave_data
            
            if moho_depth_file is not None and crustal_model_file is not None:
                raise Exception("please provide only a moho depth file OR a crustal model file, not both.")
            
            if moho_depth_file is not None:
                moho_depth = np.loadtxt(moho_depth_file)
                if coordinates == 'latlon':
                    moho_x,moho_y = p(moho_depth[:,1],moho_depth[:,0])
                    moho_depth = np.column_stack((moho_x/1000.,moho_y/1000.,
                                                  moho_depth[:,2]))
                if (minx<np.min(moho_depth[:,0]) or 
                    maxx>np.max(moho_depth[:,0]) or 
                    miny<np.min(moho_depth[:,1]) or 
                    maxy>np.max(moho_depth[:,1])):
                    print("Warning: Moho map does not cover the entire study region. Moho is set to 30km outside the defined region.")                    
            else:
                moho_depth = None
                
            if crustal_model_file is not None:
                crust_mod = np.loadtxt(crustal_model_file)
                if coordinates == 'latlon':
                    crust_x,crust_y = p(crust_mod[:,1],crust_mod[:,0])
                    crust_mod[:,0] = crust_x/1000.
                    crust_mod[:,1] = crust_y/1000.
                if (minx<np.min(crust_mod[:,0]) or 
                    maxx>np.max(crust_mod[:,0]) or 
                    miny<np.min(crust_mod[:,1]) or 
                    maxy>np.max(crust_mod[:,1])):
                    print("WARNING: Crustal model does not cover the entire study region. Values outside the region are extrapolated (nearest neighbor).")      
            else:
                crust_mod = None
                
            if type(init_no_points) == type(str):
                if init_no_points == 'random':
                    init_no_points = list(np.random.randint(min_no_points,max_no_points,no_searches))
                else:
                    raise Exception(f"did not understand init_no_points = {init_no_points}. Should be a list of integers or 'random'.")
                
            
    else: # if resuming job these variables aren't needed
        print("Resuming model search")
        minx = maxx = miny = maxy = z = z_scale = None
        xgridspacing = ygridspacing = projection_str = moho_depth = None
        crust_mod = None
        
        
    # Setup the search parameters and the priors
    input_list = []
    if mpi_rank==0:
        
        vs_priors = np.loadtxt(vs_priors_file)
        vs_profile = np.interp(zaxis,vs_priors[:,0],np.mean(vs_priors[:,1:3],axis=1))
        vs_conversion = np.loadtxt(vs_vp_conversion_file)
        
        if temperatures == 'auto':
            temperatures = 10**(3*np.arange(no_searches)/(no_searches - 1)) #, (i = 1, …, n) as proposed by Atchadé et al. (2011)
            print("Temperature levels:",temperatures)
    
        for i in range(no_searches):
            # # # INITIALIZE VARIABLES # # # 
            params = rhc.initialize_params(
                minx=minx, maxx=maxx, miny=miny,maxy=maxy, z=zaxis,
                z_scale=rhc.check_list(z_scale,i,no_searches),
                xgridspacing = xgridspacing,
                ygridspacing = ygridspacing,
                projection = projection_str,
                parameterization = parameterization,
                number_of_iterations = rhc.check_list(number_of_iterations,i,no_searches),
                misfit_norm = rhc.check_list(misfit_norm,i,no_searches),
                likelihood_distribution = rhc.check_list(likelihood_distribution,i,no_searches),
                init_no_points = rhc.check_list(init_no_points,i,no_searches),
                init_vel_points = rhc.check_list(init_vel_points,i,no_searches),
                propvelstd_velocity_update = rhc.check_list(proposal_std_velocity_update,i,no_searches),
                propvelstd_birth_death = rhc.check_list(proposal_std_birth_death,i,no_searches),
                fixed_propvel_std = rhc.check_list(fixed_propvel_std,i,no_searches),
                propmove_std = rhc.check_list(proposal_std_move,i,no_searches),
                delayed_rejection = rhc.check_list(delayed_rejection,i,no_searches),
                temperature = rhc.check_list(temperatures,i,no_searches),
                number_of_burnin_samples = rhc.check_list(number_of_burnin_samples,i,no_searches),
                update_path_interval = rhc.check_list(update_path_interval,i,no_searches),
                propsigma_std = rhc.check_list(proposal_std_data_uncertainty,i,no_searches),
                data_std_type = rhc.check_list(data_std_type,i,no_searches),
                sw_hyperparameter_level = rhc.check_list(sw_hyperparameter_level,i,no_searches),
                bw_hyperparameter_level = rhc.check_list(bw_hyperparameter_level,i,no_searches),
                anisotropic = rhc.check_list(anisotropic_search,i,no_searches),
                collect_step = rhc.check_list(collect_step,i,no_searches),
                store_models = rhc.check_list(store_models,i,no_searches),
                print_stats_step = rhc.check_list(print_stats_step,i,no_searches),
                logfile_path = rhc.check_list(output_location,i,no_searches),
                visualize = rhc.check_list(visualize,i,no_searches),
                visualize_step = rhc.check_list(visualize_step,i,no_searches),
                resume = resume_job)
            prior = rhc.initialize_prior(
                vs_priors = vs_priors,
                vs_conversion = vs_conversion,
                moho_model = moho_depth,
                crust_model = crust_mod,
                flexible_vpvs = rhc.check_list(flexible_vpvs,i,no_searches),
                vpvsmin = rhc.check_list(min_vpvs,i,no_searches),
                vpvsmax = rhc.check_list(max_vpvs,i,no_searches),
                min_no_points=rhc.check_list(min_no_points,i,no_searches),
                max_no_points=rhc.check_list(max_no_points,i,no_searches),
                min_datastd = rhc.check_list(min_datastd,i,no_searches),
                max_datastd = rhc.check_list(max_datastd,i,no_searches))
            
            input_list.append([i+1,prior,params])
                
        if len(input_list) < mpi_size:
            print("mpi size:",mpi_size)
            print("len(input_list):",len(input_list))
            raise Exception("Less chains (%d) than MPI processes (%d), this may cause problems." %(len(input_list),mpi_size))

    
    # share data among processes                    
    input_list = mpi_comm.bcast(input_list,root=0)
    
    # run search
    modelsearch.run_model_search(input_list,
                                 mpi_rank,mpi_size,mpi_comm,
                                 resume=resume_job)
              
    if plot:
        print("plotting models")
        modelsearch.plot(saveplot=True)
        
    if mpi_rank == 0:        
        get_avg_model(output_location,depthlevels='auto',
                       projection_str=None,plotsinglechains=False,
                       plot_avg_model=plot)
        



       
