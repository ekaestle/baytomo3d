#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:52:06 2019

@author: emanuel
"""

import glob, pickle, gzip, os,  sys
# if this script is run from inside the baytomo3d folder, it's necessary to 
# add the parent path to the sys.path list, otherwise it gives an import error
if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.geodesic as cgd
from cmcrameri import cm as cmcram
import matplotlib.gridspec as gridspec
from scipy.io import netcdf_file
from baytomo3d.BodyWaveModule import read_bodywave_module
from baytomo3d.SurfaceWaveModule import read_surfacewave_module
from baytomo3d.SurfaceWaveModule import get_phasevel_maps

# the main function is only executed if the AverageModels.py script
# is called directly
# if it is imported by another script, the main function has no importance
def main():
    #%%
    # folder with chain files
    foldername = "../2024_10_joint_z1"
    # plot also single chain output images
    plotsinglechains = False
    # plot the average models and model statistics
    plot_avg_model = True
    # specify the projection of the X,Y coordinates in the chain files
    projection_str = None # recommended: None, i.e. read from chain file
    # specify at which depth intervals to plot the maps
    depthlevels = 'auto' #[0,12,27,42] or 'auto'
    # create a 3D html plot with plotly (experimental)
    create_3d_plot = False
    
    # the individual chains only store values if the chain temperature is
    # equal to 1. Chains running at a high temperature will have no influence
    # on the global average model
    
    get_avg_model(foldername,depthlevels=depthlevels,
                   projection_str=projection_str,
                   plotsinglechains=plotsinglechains,
                   plot_avg_model=plot_avg_model,
                   create_3d_plot=create_3d_plot)
    #%%


"""
This function reads in the information from each individual chain and creates a global
average model. The final model is stored as a netcdf file and some plots are created.
"""
def get_avg_model(foldername,depthlevels='auto',
                   projection_str=None,plotsinglechains=True,plot_avg_model=True,
                   create_3d_plot=False):
    #%%
    chainfiles = glob.glob(os.path.join(foldername,"rjmcmc_chain*") ,recursive=True)
    if len(chainfiles) == 0:
        raise Exception(f"no files in folder {foldername}")
    chainfiles = np.sort(chainfiles)
    
    bw_module = read_bodywave_module(foldername)
    sw_module = read_surfacewave_module(foldername)
    
    filedict = {}
    for fpath in chainfiles:
        fname = os.path.basename(fpath)
        dirname = os.path.dirname(fpath)
        try:
            filedict[dirname].append(fpath)
        except:
            filedict[dirname] = []
            filedict[dirname].append(fpath)
            
    for dirname in filedict:
        print(len(filedict[dirname]),"chain files in",dirname.split("/")[-1])
        
        
    # empty lists to collect the results from the individual chains:
    average_model_counters = []
    average_vsmods = []
    average_vpmods = []
    vssumsquared = []
    vpsumsquared = []
    average_vpvsmods = []
    vpvssumsquared = []
    no_points = []
    datastd = []
    eventstd = []
    foutlier = []
    outlier_model = False
    likelihoods = []
    residuals = []
    stations = {}
    no_chains = 0
    no_data = {}
    iterations = []
    chain_numbers = []
    datasets = []
    X = None
    
    for dirname in filedict:
        for i,chainpath in enumerate(filedict[dirname]):
            
            if plotsinglechains:
                print("reading and plotting chain",chainpath)
            else:
                print("reading chain",chainpath)
            chain = None # avoid having more than 1 chain in memory
            with gzip.open(chainpath, "rb") as f:
                chain = pickle.load(f)
                chain.prop_vs2vpvs = None
                # update in scipy interpolate, therefore do this test here
                #try:
                #    dump = chain.vs2vpvs(chain.points[:,3])
                #except:
                #    chain.vs2vpvs = interp1d(chain.points[:,3],chain.points[:,4],kind='nearest',fill_value='extrapolate')
                #    vs_vp_conversion_file = "vs_vp_rho_conversion.txt"
                #    vs_conversion = np.loadtxt(vs_vp_conversion_file)
                #    chain.vs2rho = interp1d(vs_conversion[:,0],vs_conversion[:,2],kind='nearest',fill_value='extrapolate')

            if plotsinglechains:
                chain.plot(saveplot=True,output_location=foldername)
                
            if X is None:
                X,Y,Z = np.meshgrid(chain.x,chain.y,chain.z)
                #X = chain.X
                #Y = chain.Y
                #Z = chain.Z
                if isinstance(depthlevels,str):
                    z = np.unique(Z)
                    depthidx = np.unique(np.around(np.linspace(0,len(z),15,endpoint=False)).astype(int))
                    depthlevels = z[depthidx]

                if sw_module.isactive:
                    datasets += list(sw_module.data.keys())
                    for dataset in datasets:
                        stations[dataset] = {}
                        no_data[dataset] = {}
                        for period in sw_module.data[dataset]:
                           stations[dataset][period] = np.unique(np.vstack((sw_module.sources[dataset][period],
                                                                            sw_module.receivers[dataset][period])),axis=0) 
                           no_data[dataset][period] = len(sw_module.data[dataset][period])
                if bw_module.isactive:
                    datasets += list(bw_module.data.keys())
                    
                if projection_str is None:
                    projection_str = chain.projection
                    projection = pyproj.Proj(projection_str)
                    
                valid_gridpoints = chain.para.convexhull.find_simplex(chain.gridpoints)
                valid_gridpoints = (valid_gridpoints>-1).reshape(np.shape(X))
                
                data_coverage_sw = chain.get_data_coverage_sw(sw_module)
                data_coverage_bw = chain.get_data_coverage_bw(bw_module)
                data_coverage_sw = np.reshape(data_coverage_sw,X.shape)
                data_coverage_bw = np.reshape(data_coverage_bw,X.shape)
                
                # get prior boundaries
                #minvs = chain.minvs(chain.z_scaled) # depth dependent
                #maxvs = chain.maxvs(chain.z_scaled) # depth dependent
                minvs = np.min(chain.minvs(chain.z_scaled))
                maxvs = np.max(chain.maxvs(chain.z_scaled))
                bins_vs = np.arange(minvs,maxvs+0.049,0.05)
                vs_histo = np.zeros((len(X.flatten()),len(bins_vs)-1),dtype='int32')
                minvpvs = chain.vpvsmin # can be None if vpvs is not a free parameter
                maxvpvs = chain.vpvsmax # can be None if vpvs is not a free parameter
                if minvpvs is None or maxvpvs is None:
                    minvpvs = np.min(chain.average_vpvs/chain.average_model_counter)
                    maxvpvs = np.max(chain.average_vpvs/chain.average_model_counter)
                bins_vpvs = np.arange(minvpvs,maxvpvs+0.024,0.025)
                vpvs_histo = np.zeros((len(X.flatten()),len(bins_vpvs)-1),dtype='int32')
                bins_vp = np.arange(minvs*minvpvs,maxvs*maxvpvs+0.049,0.05)
                vp_histo = np.zeros((len(X.flatten()),len(bins_vp)-1),dtype='int32')

                # if this is done once, we can remove the matrices from memory
                sw_module.A = None # free up memory space
                bw_module.A = None # free up memory space
                bw_module.paths = None # free up memory
                bw_module.paths_eikonal = None # free up memory
                                
            chain_numbers.append(chain.chain_no)
            average_vsmods.append(chain.average_vs)
            average_vpmods.append(chain.average_vp)
            average_vpvsmods.append(chain.average_vpvs)
            vssumsquared.append(chain.vssumsquared)
            vpsumsquared.append(chain.vpsumsquared)
            vpvssumsquared.append(chain.vpvssumsquared)
            if i>0:
                if np.shape(chain.gridpoints[:,0]) != np.shape(average_vsmods[0]):
                    raise Exception("Chains have different parameterization. Calculating the average model in this case is currently not implemented.")
            average_model_counters.append(chain.average_model_counter)
            no_points.append(np.array(chain.collection_no_points))
            datastd.append(chain.collection_datastd)
            
            # create a historgram
            # keeping all the individual models would use a lot of memory space
            # One 3D model in 16bit float takes up 2MB, so 100 chains with each 600 models result in ~120GB
            if chain.store_models and len(chain.collection_vs_models)>1:
                nmods = len(chain.collection_vs_models)
                if nmods!=chain.average_model_counter-1:
                    print("warning: number of models stored individually is not identical to average model counter",nmods,chain.average_model_counter)
                vsmods = np.vstack(chain.collection_vs_models)
                vpvsmods = np.vstack(chain.collection_vpvs_models)
            else:
                # just use the single average model - add one axis so that the shape is identical
                vsmods = (chain.average_vs/chain.average_model_counter)[None]
                vpvsmods = (chain.average_vpvs/chain.average_model_counter)[None]
            vpmods = vsmods*vpvsmods
            vs_histo += hist_laxis(vsmods.T,bins_vs)
            vp_histo += hist_laxis(vpmods.T,bins_vp)
            vpvs_histo += hist_laxis(vpvsmods.T,bins_vpvs)
            
            if chain.foutlier is not None:
                foutlier.append(chain.foutlier)
                outlier_model = True
            else:
                foutlier.append([])
            # get the standard deviations for the different earthquake events
            if bw_module.isactive:
                eventstd_dict = {}
                for dataset in bw_module.data.keys():
                    events = np.unique(bw_module.event_idx[dataset])
                    eventstds = []
                    for eventidx in events:
                        valid = np.where(bw_module.event_idx[dataset]==eventidx)[0]
                        eventstds.append(np.mean(chain.datastd[dataset][valid]))
                    eventstd_dict[dataset] = eventstds
                eventstd.append(eventstd_dict)
            if chain.total_steps <= chain.nburnin:
                likelihoods.append(np.array(chain.collection_loglikelihood[int(chain.nburnin/4.):]))
                residuals.append(np.array(chain.collection_residuals[int(chain.nburnin/4.):]))
                iterations.append([int(chain.nburnin/4.),chain.total_steps])
            else:
                likelihoods.append(np.array(chain.collection_loglikelihood[chain.nburnin:]))
                residuals.append(np.array(chain.collection_residuals[chain.nburnin:]))
                iterations.append([chain.nburnin,chain.total_steps])
    
            no_chains = i+1

    # reading chain data done

    #%% Join all models and get final average
    
    # keep only a selection of models, e.g. decide to discard 20% of the chains
    # that have the lowest mean likelihoods.
    subselection = 100 # in percent, 100% means keep everything
    if subselection < 100 and no_chains>1:
        print("Taking a subselection of %d percent of the best models" %subselection)
        # sort out the bad chains    
        likelihoodmeans = np.array([np.mean(k) for k in likelihoods])#np.mean(likelihoods,axis=1)
        sortidx = likelihoodmeans.argsort()[::-1]
        good_chains = sortidx[:int(np.around(subselection/100.*len(sortidx)))]
        #good_chains = np.where(likelihoodmeans>=np.median(likelihoodmeans)-1*np.std(likelihoodmeans))[0]
        #good_chains = np.random.choice(good_chains,int(subselection/100*len(good_chains)),replace=False)
        good_chains = np.sort(good_chains)
        print("keeping",len(good_chains),"chains of a total of",no_chains)
    elif no_chains>1: # if 100%, all chains are 'good chains'
        good_chains = np.arange(len(average_model_counters),dtype=int)
    else: # if there is only a single chain
        good_chains = np.array([0])
        
    if len(average_vpvsmods) != len(average_model_counters):
        print("Warning, vpvs ratio may not be correct!")
    
    # keep only [subselection]% of chains
    # average models
    Nmodels = np.sum(np.array(average_model_counters)[good_chains])
    avg_vs_model = np.sum(np.array(average_vsmods)[good_chains],axis=0) / Nmodels
    avg_vp_model = np.sum(np.array(average_vpmods)[good_chains],axis=0) / Nmodels
    avg_vpvs_model = np.sum(np.array(average_vpvsmods)[good_chains],axis=0) / Nmodels

    # average model variance sum(model-avg_model)**2 = sum( model**2 - 2model*avg_model + avg_model**2)
    # model std = sqrt( variance / N)
    vs_variance = (np.sum(np.array(vssumsquared)[good_chains],axis=0) - 
                   2*avg_vs_model * np.sum(np.array(average_vsmods)[good_chains],axis=0) +
                   Nmodels * avg_vs_model**2)
    vp_variance = (np.sum(np.array(vpsumsquared)[good_chains],axis=0) - 
                   2*avg_vp_model * np.sum(np.array(average_vpmods)[good_chains],axis=0) +
                   Nmodels * avg_vp_model**2)
    vpvs_variance = (np.sum(np.array(vpvssumsquared)[good_chains],axis=0) - 
                     2*avg_vpvs_model * np.sum(np.array(average_vpvsmods)[good_chains],axis=0) +
                     Nmodels * avg_vpvs_model**2)            
    # mask regions where no updates happened and avoid sqrt of negative values
    valid = np.where(vs_variance>1e-5)[0]
    modeluncertainties_vs = np.ones_like(vs_variance)*np.nan
    modeluncertainties_vs[valid] = np.sqrt(vs_variance[valid] / Nmodels)
    valid = np.where(vp_variance>1e-5)[0]
    modeluncertainties_vp = np.ones_like(vp_variance)*np.nan
    modeluncertainties_vp[valid] = np.sqrt(vp_variance[valid] / Nmodels)
    valid = np.where(vpvs_variance>1e-5)[0]
    modeluncertainties_vpvs = np.ones_like(vpvs_variance)*np.nan
    modeluncertainties_vpvs[valid] = np.sqrt(vpvs_variance[valid] / Nmodels)
    
    refmodel1dvs = np.mean(avg_vs_model.reshape((len(X[0])*len(X),len(X[0,0,:]))),axis=0)
    refmodel1dvp = np.mean(avg_vp_model.reshape((len(X[0])*len(X),len(X[0,0,:]))),axis=0)

    avg_model_dvs = (avg_vs_model.reshape(np.shape(X))-refmodel1dvs)/refmodel1dvs
    avg_model_dvp = (avg_vp_model.reshape(np.shape(X))-refmodel1dvp)/refmodel1dvp
    if len(datastd) > 0:
        datastd = np.array(datastd,dtype='object')[good_chains]
        if outlier_model:
            foutlier = np.array(foutlier,dtype='object')[good_chains]
    chain_numbers = np.array(chain_numbers,dtype='object')[good_chains]

    # average phase velocity maps
    if sw_module.isactive:
        valid,idx_2d,prop_phslo_ray,prop_phslo_lov,nmods = get_phasevel_maps(
                chain.z,sw_module.periods,chain.shape,
                vs_field=avg_vs_model,vpvs_field=avg_vpvs_model)
    # average Vs model
    avg_vs_model = avg_vs_model.reshape(np.shape(X))
    avg_vs_model = np.ma.masked_where(~valid_gridpoints,avg_vs_model)
    modeluncertainties_vs = modeluncertainties_vs.reshape(np.shape(X))
    modeluncertainties_vs = np.ma.masked_where(~valid_gridpoints,modeluncertainties_vs)
    # average Vp model
    avg_vp_model = avg_vp_model.reshape(np.shape(X))
    avg_vp_model = np.ma.masked_where(~valid_gridpoints,avg_vp_model)    
    modeluncertainties_vp = modeluncertainties_vp.reshape(np.shape(X))
    modeluncertainties_vp = np.ma.masked_where(~valid_gridpoints,modeluncertainties_vp)
    # average vpvs
    avg_vpvs_model = avg_vpvs_model.reshape(np.shape(X))
    avg_vpvs_model = np.ma.masked_where(~valid_gridpoints,avg_vpvs_model)    
    modeluncertainties_vpvs = modeluncertainties_vpvs.reshape(np.shape(X))
    modeluncertainties_vpvs = np.ma.masked_where(~valid_gridpoints,modeluncertainties_vpvs)
    
    # this takes up too much memory!
    # # if individual models are stored, join those in an array
    # if len(all_vs_models) > 0:
    #     all_vs_models = np.array(all_vs_models)[good_chains]
    #     all_vpvs_models = np.array(all_vpvs_models)[good_chains]
    #     nmods = np.sum(list(map(len,all_vs_models))) # total count of models
    #     if nmods!=Nmodels:
    #         print("Warning: the number of individually stored models should be identical to the 'average_model_counter'.")
    #     # shape will be (nmods,shape_of_3D_model), so that all_vs_models[0] is the first model
    #     all_vs_models = np.reshape(np.vstack(all_vs_models),(nmods,)+X.shape)
    #     all_vpvs_models = np.reshape(np.vstack(all_vpvs_models),all_vs_models.shape)
        
    all_stations = np.empty((0,2))
    if sw_module.isactive:
        datastd_sw = []
        for i in range(len(datastd)):
            datastd_sw.append({})
            for dataset in sw_module.data:
                datastd_sw[i][dataset] =  datastd[i][dataset]
        for i,dataset in enumerate(sw_module.data):
            for j,period in enumerate(stations[dataset]):
                if i==0 and j==0:
                    all_stations = stations[dataset][period]
                else:
                    all_stations = np.vstack((all_stations,stations[dataset][period]))
        all_stations = np.unique(all_stations,axis=0)
    if bw_module.isactive:
        for i,dataset in enumerate(bw_module.data):
            all_stations = np.vstack((all_stations,bw_module.receivers[dataset][:,:2]))
        all_stations = np.unique(all_stations,axis=0)
            

    #%%
    # save average phase velocity maps as ascii file
    LON,LAT = projection(X[:,:,0]*1000.,Y[:,:,0]*1000., inverse=True)
    if sw_module.isactive:
        for wavetype in ["Rayleigh","Love"]:
            for pi,period in enumerate(sw_module.periods):
                if wavetype=="Rayleigh":
                    phasevelfield = prop_phslo_ray[period]
                else:
                    phasevelfield = prop_phslo_lov[period]
                phasevelfield = np.ma.masked_where(np.isinf(phasevelfield),phasevelfield)
                phasevelfield = np.reshape(1./phasevelfield,np.shape(X[:,:,0]))
                phasevelfield = np.ma.masked_where(~valid_gridpoints[:,:,0],phasevelfield)
                np.savetxt(os.path.join(foldername,"average_phasevel_%s_%.2fs" %(wavetype,period)),
                           np.column_stack((LON.flatten(),LAT.flatten(),phasevelfield.flatten())))
                
        
    #%%
    # save average 3d model as netcdf file
    f = netcdf_file(os.path.join(foldername,"average_3D_model.nc"),'w')
    f.projection = projection_str
    f.createDimension('x',len(X[0,:,0]))
    f.createDimension('y',len(Y[:,0,0]))
    f.createDimension('z',len(Z[0,0,:]))
    netcdfx = f.createVariable('x','float32',('x',))
    netcdfy = f.createVariable('y','float32',('y',))
    netcdfz = f.createVariable('z','float32',('z',))
    netcdfvs = f.createVariable('vs','float32',('x','y','z',))
    netcdfvp = f.createVariable('vp','float32',('x','y','z',))
    netcdfvpvs = f.createVariable('vpvs','float32',('x','y','z'))
    netcdfvsstd = f.createVariable('vs_std','float32',('x','y','z'))
    netcdfvpstd = f.createVariable('vp_std','float32',('x','y','z'))
    netcdfvpvsstd = f.createVariable('vpvs_std','float32',('x','y','z'))
    netcdfdatacoveragebw = f.createVariable('data_coverage_bw','float32',('x','y','z'))
    netcdfdatacoveragesw = f.createVariable('data_coverage_sw','float32',('x','y','z'))
    netcdfvs1d = f.createVariable('vsref','float32',('z'))
    netcdfvp1d = f.createVariable('vpref','float32',('z'))
    netcdfdvs = f.createVariable('dvs','float32',('x','y','z'))
    netcdfdvp = f.createVariable('dvp','float32',('x','y','z'))
    netcdfx[:] = X[0,:,0]
    netcdfy[:] = Y[:,0,0]
    netcdfz[:] = Z[0,0,:]
    netcdfvs[:,:,:] = np.transpose(avg_vs_model,(1,0,2))
    netcdfvp[:,:,:] = np.transpose(avg_vp_model,(1,0,2))
    netcdfvpvs[:,:,:] = np.transpose(avg_vpvs_model,(1,0,2))
    netcdfvsstd[:,:,:] = np.transpose(modeluncertainties_vs,(1,0,2))
    netcdfvpstd[:,:,:] = np.transpose(modeluncertainties_vp,(1,0,2))
    netcdfvpvsstd[:,:,:] = np.transpose(modeluncertainties_vpvs,(1,0,2))
    netcdfvs1d[:] = refmodel1dvs
    netcdfvp1d[:] = refmodel1dvp
    netcdfdvs[:] = np.transpose(avg_model_dvs,(1,0,2))
    netcdfdvp[:] = np.transpose(avg_model_dvp,(1,0,2))
    netcdfdatacoveragebw[:,:,:] = np.transpose(data_coverage_bw,(1,0,2))
    netcdfdatacoveragesw[:,:,:] = np.transpose(data_coverage_sw,(1,0,2))
    # histograms
    nbins_vs = len(bins_vs)-1
    nbins_vp = len(bins_vp)-1
    nbins_vpvs = len(bins_vpvs)-1
    f.createDimension('hist_size_vs', nbins_vs) # number of bins
    f.createDimension('hist_bins_vs', nbins_vs+1) # bin edges
    f.createDimension('hist_size_vp', nbins_vp) # number of bins
    f.createDimension('hist_bins_vp', nbins_vp+1) # bin edges
    f.createDimension('hist_size_vpvs', nbins_vpvs) # number of bins
    f.createDimension('hist_bins_vpvs', nbins_vpvs+1) # bin endges
    netcdfvsbins = f.createVariable('hist_bins_vs','float32',('hist_bins_vs',))
    netcdfvpbins = f.createVariable('hist_bins_vp','float32',('hist_bins_vp',))
    netcdfvpvsbins = f.createVariable('hist_bins_vpvs','float32',('hist_bins_vpvs',))
    netcdf_vshist = f.createVariable('vs_histogram',    'int32',('x','y','z','hist_size_vs'))
    netcdf_vphist = f.createVariable('vp_histogram',    'int32',('x','y','z','hist_size_vp'))
    netcdf_vpvshist = f.createVariable('vpvs_histogram','int32',('x','y','z','hist_size_vpvs'))
    netcdfvsbins[:] = bins_vs
    netcdfvpbins[:] = bins_vp
    netcdfvpvsbins[:,] = bins_vpvs
    netcdf_vshist[:] = np.transpose(np.reshape(vs_histo,    X.shape+(nbins_vs,)),  (1,0,2,3))
    netcdf_vphist[:] = np.transpose(np.reshape(vp_histo,    X.shape+(nbins_vp,)),  (1,0,2,3))
    netcdf_vpvshist[:] = np.transpose(np.reshape(vpvs_histo,X.shape+(nbins_vpvs,)),(1,0,2,3))
    f.close()
    
    infotext = """Description of average_3D_model.nc file
    This is a netCDF file created with the scipy netCDF module. It can be read in Python with
    
    from scipy.io import netcdf_file
    model = netcdf_file(filepath)
    x = model.variables['x'][:]
    y = model.variables['y'][:]
    projection_str = model.projection.decode()
    etc...
    
    Available independent variables are 'x','y' and 'z' for the model dimensions in kilometers.
    They can be converted to lon/lat/depth values using the projection_str information, e.g.
    
    import pyproj
    projection = pyproj.Proj(projection_str)
    X,Y = np.meshgrid(x,y)
    lons,lats = projection(X*1000,Y*1000,inverse=True)
    
    x and y are on a regular grid, but not lons and lats after inverse projection.
    
    Dependent varaiables are 'vp','vs','vpvs','vs_std','vp_std','vpvs_std','dvs','dvp'.
    They all have shape (x,y,z).
    The 'dvx' variables are calculated based on a reference 1d model, which is a simple average of the 'vp' or 'vs' models
    You can find the reference 1d models in the variables 'vpref' and 'vsref'.
    
    Additinally, you find the data coverage in the variables 'data_coverage_bw' and 'data_coverage_sw'.
    This is simply the summed up number of rays passing through each gridcell.
    
    There are also histograms of the vs, vp and the vpvs values at each gridpoint.
    These are calculated from the individually stored models. If no models are
    stored, there is only one value from the average model for each chain, so
    that the histogram may be not very informative. They are stored under
    'vs_histogram', 'vp_histogram' and 'vpvs_histogram'. 
    The histogram bin edges are stored as 'hist_bins_vs', 'hist_bins_vp' and 'hist_bins_vpvs'
    
    """
    with open(os.path.join(foldername,"average_3D_model_readme.txt"),"w") as f:
        f.write(infotext)
    
        #%%

    if plot_avg_model:

        plt.ioff()
            
        all_periods = []
        labels = []
        lines = []
        if sw_module.isactive:
            datasets_sw = list(sw_module.data.keys())
            if outlier_model:
                fig = plt.figure(figsize=(15,3*len(datasets_sw)))
            else:
                fig = plt.figure(figsize=(12,3*len(datasets_sw)))
            for di,dataset in enumerate(sw_module.data):
                print(dataset)
                periods = list(stations[dataset])
                if outlier_model:
                    ax1 = fig.add_subplot(len(datasets_sw),3,3*di+1)
                    ax2 = fig.add_subplot(len(datasets_sw),3,3*di+2)
                    ax3 = fig.add_subplot(len(datasets_sw),3,3*di+3)
                    for fi,chainoutlier in enumerate(foutlier):
                        outlierfrac = []
                        for period in periods:
                            if len(chainoutlier)>0:
                                outlierfrac.append(100*chainoutlier[dataset][period])
                            else:
                                outlierfrac += []
                        if len(outlierfrac)>0:
                            ax3.plot(periods,outlierfrac,'o')
                        else:
                            ax3.plot([],outlierfrac)
                    ax3.set_ylabel("outlier fraction [%]")
                    
                else:
                    ax1 = fig.add_subplot(len(datasets_sw),2,2*di+1)
                    ax2 = fig.add_subplot(len(datasets_sw),2,2*di+2)
                for ci,chainstd in enumerate(datastd_sw):
                    meanstd = []
                    mean_data_std = np.zeros(len(chainstd[dataset][periods[0]]))
                    total_no_measurements = 0
                    for period in periods:
                        meanstd.append([period,np.mean(chainstd[dataset][period])])
                        if len(chainstd[dataset][period]) > 0:
                            mean_data_std += no_data[dataset][period] * np.array(chainstd[dataset][period])
                            total_no_measurements += no_data[dataset][period]
                    meanstd = np.array(meanstd)
                    ax1.plot(meanstd[:,0],meanstd[:,1],'o')
                    ax1.set_ylabel("mean data std")
                    if ci == 0:
                        ax1.set_title(dataset,loc='left')
                    mean_data_std /= total_no_measurements
                    line = ax2.hist(mean_data_std,25,alpha=0.8)
                    if di == 0:
                        lines.append(line)
                        labels.append("Chain %d" %chain_numbers[ci])
                    ax2.set_ylabel("no of models")
                    all_periods += periods
                
            all_periods = np.unique(all_periods)
            ax1.set_xlabel("period [s]")
            ax2.set_xlabel("mean data std")
            if outlier_model:
                ax3.set_xlabel("period [s]")
            fig.legend(lines,       # List of the line objects
                       labels=labels,       # The labels for each line
                       loc="lower center",  # Position of the legend
                       ncol=6,
                       borderaxespad=0.1,   # Add little spacing around the legend box
                       bbox_to_anchor=(0.25,0.03))
                       #title="Legend Title")      # Title for the legend
            plt.savefig(os.path.join(foldername,'figure_data_std_diagrams_surfwaves.png'),bbox_inches='tight')        
            plt.close(fig)
                
        fig = plt.figure(figsize=(7,5))
        plt.title("loglikelihood")
        for i,ll in enumerate(likelihoods):
            plt.plot(np.arange(iterations[i][0],iterations[i][1]),ll)
        plt.ylabel("loglikelihood")
        plt.xlabel("iteration")
        plt.savefig(os.path.join(foldername,"figure_loglikelihoods.png"), bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(7,5))
        plt.title("residuals")
        for i,ll in enumerate(residuals):
            plt.plot(np.arange(iterations[i][0],iterations[i][1]),ll)
        plt.ylabel("residual")
        plt.xlabel("iteration")
        plt.savefig(os.path.join(foldername,"figure_residuals.png"), bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(7,5))
        plt.title("number of cells")
        for i,pnts in enumerate(no_points):
            plt.plot(pnts)
        plt.ylabel("no of points")
        plt.xlabel("iteration")
        plt.savefig(os.path.join(foldername,"figure_number_of_points.png"), bbox_inches='tight')
        plt.close(fig)
        
        
        LON,LAT = projection(X[:,:,0]*1000.,Y[:,:,0]*1000., inverse=True)
        statlon,statlat = projection(all_stations[:,0]*1000.,all_stations[:,1]*1000,inverse=True)

        # should be possible to directly convert...
        # proj = ccrs.Projection(projection.crs)
        try:
            central_lon = float(projection_str.split()[2].split("=")[-1])
            central_lat = float(projection_str.split()[1].split("=")[-1])
            proj = ccrs.TransverseMercator(central_longitude=central_lon,
                                           central_latitude=central_lat)
        except:
            proj = ccrs.TransverseMercator(central_longitude=np.mean(LON),
                                           central_latitude=np.mean(LAT))
        #X_plot_low,Y_plot_low = X_plot[::reduce_factor,::reduce_factor],Y_plot[::reduce_factor,::reduce_factor]
        
        for depthlevel in depthlevels:
            depthidx = np.abs(Z[0,0,:]-depthlevel).argmin()
            depth = Z[0,0,depthidx]
            
            velfield = avg_vs_model[:,:,depthidx]
            fig = plt.figure(figsize=(14,10))
            axm = plt.axes(projection=proj)
            cbar = axm.pcolormesh(LON,LAT,velfield,cmap=cmcram.roma,
                                  rasterized=True,shading='nearest',
                                  transform=ccrs.PlateCarree())
            axm.coastlines(resolution='50m',linewidth=0.6)
            axm.add_feature(cf.BORDERS.with_scale('50m'),linewidth=0.4)
            lonstep = (np.max(LON)-np.min(LON))/8.        
            decimal_i = 1
            while int(lonstep * decimal_i)==0:
                decimal_i *= 10.
            parallels = np.arange(np.ceil(np.min(LAT)*decimal_i)/decimal_i,
                                  np.ceil(np.max(LAT)*decimal_i)/decimal_i,
                                  np.floor(lonstep*decimal_i)/decimal_i)
            meridians = np.arange(np.ceil(np.min(LON)*decimal_i)/decimal_i,
                                  np.ceil(np.max(LON)*decimal_i)/decimal_i,
                                  np.floor(lonstep*decimal_i)/decimal_i)
            gl = axm.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                   xlocs=meridians,ylocs=parallels,
                   linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            colax = fig.add_axes([0.55,0.05,0.25,0.02])
            plt.colorbar(cbar,label='shear velocity [km/s]',cax=colax,orientation='horizontal')
            #plt.annotate("Rayleigh %.2fs" %period,xycoords='figure fraction',xy=(0.15,0.9),fontsize=16,bbox=dict(boxstyle="square,pad=0.2",fc='white',ec='None', alpha=0.7))
            plt.savefig(os.path.join(foldername,"figure_Vs_map_plot_%.1fkm.png" %depth),
                        bbox_inches='tight',transparent=True)
            plt.close(fig)
        
            vpvsmap = avg_vpvs_model[:,:,depthidx]
            fig = plt.figure(figsize=(14,10))
            axm = plt.axes(projection=proj)
            cbar = axm.pcolormesh(LON,LAT,vpvsmap,rasterized=True,
                                  shading='nearest',
                                  transform=ccrs.PlateCarree())
            axm.coastlines(resolution='50m',linewidth=0.6)
            axm.add_feature(cf.BORDERS.with_scale('50m'),linewidth=0.4)
            gl = axm.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                   xlocs=meridians,ylocs=parallels,
                   linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            colax = fig.add_axes([0.55,0.05,0.25,0.02])
            plt.colorbar(cbar,label='VpVs ratio',cax=colax,orientation='horizontal')
            #plt.annotate("Rayleigh %.2fs" %period,xycoords='figure fraction',xy=(0.15,0.9),fontsize=16,bbox=dict(boxstyle="square,pad=0.2",fc='white',ec='None', alpha=0.7))
            plt.savefig(os.path.join(foldername,"figure_VpVs_ratio_map_%.1fkm.png" %depth),bbox_inches='tight',transparent=True)
            plt.close(fig)
            
            # vp maps
            fig = plt.figure(figsize=(14,10))
            axm = plt.axes(projection=proj)
            vpmap = avg_vp_model[:,:,depthidx]
            vmax = np.max(vpmap)
            vmin = np.min(vpmap)
            cbar = axm.pcolormesh(LON,LAT,vpmap,rasterized=True,
                                  shading='nearest',cmap=cmcram.roma,
                                  vmin=vmin,vmax=vmax,
                                  transform=ccrs.PlateCarree())
            axm.coastlines(resolution='50m',linewidth=0.6)
            axm.add_feature(cf.BORDERS.with_scale('50m'),linewidth=0.4)
            gl = axm.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                   xlocs=meridians,ylocs=parallels,
                   linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            colax = fig.add_axes([0.55,0.05,0.25,0.02])
            plt.colorbar(cbar,label='bulk velocity [km/s]',cax=colax,orientation='horizontal')
            #plt.annotate("Rayleigh %.2fs" %period,xycoords='figure fraction',xy=(0.15,0.9),fontsize=16,bbox=dict(boxstyle="square,pad=0.2",fc='white',ec='None', alpha=0.7))
            plt.savefig(os.path.join(foldername,"figure_Vp_map_plot_%.1fkm.png" %depth),bbox_inches='tight',transparent=True)
            plt.close(fig)
        
            for i in range(3):
                if i==0:
                    model_uncertainty = modeluncertainties_vs[:,:,depthidx]
                    label = "Vs"
                elif i==1:
                    model_uncertainty = modeluncertainties_vp[:,:,depthidx]
                    label = "Vp"
                elif i==2:
                    model_uncertainty = modeluncertainties_vpvs[:,:,depthidx]
                    label = "VpVs"
                fig = plt.figure(figsize=(14,10))
                axm = plt.axes(projection=proj)
                cbar = axm.pcolormesh(LON,LAT,model_uncertainty,cmap=plt.cm.cividis,
                                      rasterized=True,shading='nearest',vmin=0.,
                                      transform=ccrs.PlateCarree())
                axm.coastlines(resolution='50m',linewidth=0.6)
                axm.add_feature(cf.BORDERS.with_scale('50m'),linewidth=0.4)
                gl = axm.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                       xlocs=meridians,ylocs=parallels,
                       linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                colax = fig.add_axes([0.55,0.05,0.25,0.02])
                plt.colorbar(cbar,label='model std [km/s]',cax=colax,orientation='horizontal')
                #plt.annotate("Rayleigh %.2fs" %period,xycoords='figure fraction',xy=(0.15,0.9),fontsize=16,bbox=dict(boxstyle="square,pad=0.2",fc='white',ec='None', alpha=0.7))
                plt.savefig(os.path.join(foldername,"figure_%s_std_map_%.1fkm.png" %(label,depth)),bbox_inches='tight',transparent=True)
                plt.close(fig)            

        # plot phasevel maps
        testidx = np.around(np.linspace(0,len(all_periods)-1,9)).astype(int)
        
        for wavetype in ["Rayleigh","Love"]:
            if len(all_periods)==0:
                continue
            gs = gridspec.GridSpec(3,3,wspace=0.01)
            fig = plt.figure(figsize=(10,10))        
            for pi,period in enumerate(all_periods[testidx]):
                ax = fig.add_subplot(gs[pi],projection=proj)
                if wavetype=="Rayleigh":
                    phasevelfield = prop_phslo_ray[period]
                else:
                    phasevelfield = prop_phslo_lov[period]
                phasevelfield = np.ma.masked_where(np.isinf(phasevelfield),phasevelfield)
                phasevelfield = np.reshape(1./phasevelfield,np.shape(X[:,:,0]))
                phasevelfield = np.ma.masked_where(~valid_gridpoints[:,:,0],phasevelfield)
    
                cbar = ax.pcolormesh(LON,LAT,phasevelfield,cmap=cmcram.roma,
                                     rasterized=True,shading='nearest',
                                     transform=ccrs.PlateCarree())
                ax.coastlines(resolution='50m',linewidth=0.6)
                ax.add_feature(cf.BORDERS.with_scale('50m'),linewidth=0.4)
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                       xlocs=meridians,ylocs=parallels,
                       linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                if pi%3!=0:
                    gl.left_labels = False
                if pi<6:
                    gl.bottom_labels = False
                gl.top_labels = False
                gl.right_labels = False
                ax.text(0.8,0.05,"%.1fs" %period,fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                        transform = ax.transAxes)
                inax = ax.inset_axes([0.05, 0.05, 0.03, 0.3])
                plt.colorbar(cbar,cax=inax)
                #plt.annotate("Rayleigh %.2fs" %period,xycoords='figure fraction',xy=(0.15,0.9),fontsize=16,bbox=dict(boxstyle="square,pad=0.2",fc='white',ec='None', alpha=0.7))
            plt.savefig(os.path.join(foldername,"figure_phasevel_%s.png" %wavetype),
                        bbox_inches='tight',transparent=True,dpi=300)
            plt.close(fig)        
        
        # plot sections
        
        section_x = X[0,:,0][::int(len(X[0])/5)][1:5]
        section_y = Y[:,0,0][::int(len(Y[:,0])/5)][1:5]
        
        fig = plt.figure(figsize=(14,10))
        gs = gridspec.GridSpec(5,2)
        ax0 = fig.add_subplot(gs[0])
        depthidx = np.abs(Z[0,0,:]-depthlevels[0]).argmin()
        depth = Z[0,0,depthidx]
        velfield = avg_vs_model[:,:,depthidx]
        ax0.pcolormesh(X[:,:,0],Y[:,:,0],velfield,cmap=plt.cm.Accent,
                       rasterized=True,vmin=np.min(avg_vs_model),vmax=4.1,
                       shading='nearest')
        for i in range(4):
            ax0.plot([section_x[i],section_x[i]],[np.min(Y),np.max(Y)])
            ax0.text(section_x[i],np.min(Y),"%d" %(i+1))
            ax0.plot([np.min(X),np.max(X)],[section_y[i],section_y[i]])
            ax0.text(np.min(X),section_y[i],"%d" %(i+5))
            ax1 = fig.add_subplot(gs[2*(i+1)])
            profile = avg_vs_model[X==section_x[i]].reshape((np.shape(X)[0],np.shape(X)[2]))
            profile_crust = np.ma.masked_where(profile>4.1,profile)
            ax1.pcolormesh(Y[:,0,0],z,profile_crust.T,cmap=plt.cm.Accent,
                           vmin=np.min(avg_vs_model),vmax=4.1,shading='nearest')
            profile_mantle = np.ma.masked_where(profile<=4.1,profile)
            #ax1.pcolormesh(Y[:,0,0],z,profile_mantle.T,cmap=cmcram.roma,vmin=4.1,vmax=np.max(avg_model))     
            ax1.pcolormesh(Y[:,0,0],z,((profile_mantle-refmodel1dvs)/refmodel1dvs*100).T,
                           cmap=cmcram.roma,vmin=-10,vmax=10,shading='nearest')
            ax1.set_ylim(z[-3],z[0])
            ax1.set_title("%d" %(i+1),loc='left')
            ax2 = fig.add_subplot(gs[2*(i+1)+1])
            profile = avg_vs_model[Y==section_y[i]].reshape((np.shape(X)[1],np.shape(X)[2]))
            profile_crust = np.ma.masked_where(profile>4.1,profile)
            ax2.pcolormesh(X[0,:,0],z,profile_crust.T,cmap=plt.cm.Accent,
                           vmin=np.min(avg_vs_model),vmax=4.1,shading='nearest')
            profile_mantle = np.ma.masked_where(profile<=4.1,profile)
            #ax2.pcolormesh(X[0,:,0],z,profile_mantle.T,cmap=cmcram.roma,vmin=4.1,vmax=np.max(avg_model))
            ax2.pcolormesh(X[0,:,0],z,((profile_mantle-refmodel1dvs)/refmodel1dvs*100).T,
                           cmap=cmcram.roma,vmin=-10,vmax=10,shading='nearest')
            ax2.set_ylim(z[-3],z[0])
            ax2.set_title("%d" %(i+5),loc='left')
            if i==0:
                ax1.set_title("Y-sections",loc='right')
                ax2.set_title("X-sections",loc='right')
        ax0.set_aspect('equal')
        #ax3 = fig.add_subplot(gs[1])
        #plt.colorbar(cbar1,cax=ax3)
        plt.savefig(os.path.join(foldername,"figure_cross_sections_vs.png"),bbox_inches='tight')
        plt.close(fig)
        
        if bw_module.isactive:
            fig = plt.figure(figsize=(14,10))
            isub = len(bw_module.data)
            lon0,lat0 = (np.mean(LON),np.mean(LAT))
            proj_ae = ccrs.LambertAzimuthalEqualArea(central_longitude=lon0,
                                                central_latitude=lat0,
                                                false_easting=0.0, false_northing=0.0, globe=None)
            proj_ae = ccrs.AzimuthalEquidistant(central_longitude=lon0,
                                                central_latitude=lat0,
                                                false_easting=0.0, false_northing=0.0, globe=None)
            for di,dataset in enumerate(bw_module.sources):
                ax = fig.add_subplot(isub,1,di+1,projection=proj_ae)
                events = np.unique(bw_module.event_idx[dataset])
                eventstds = []
                for eventidx in events:
                    valid = np.where(bw_module.event_idx[dataset]==eventidx)[0]
                    evlon,evlat,evdepth = bw_module.sources[dataset][valid[0]]
                    evstd = 0.
                    for cnt in range(len(datastd)):
                        evstd += eventstd[cnt][dataset][eventidx]
                    evstd /= cnt+1
                    eventstds.append([evlon,evlat,evdepth,evstd])
                eventstds = np.array(eventstds)
                cbar = ax.scatter(eventstds[:,0],eventstds[:,1],c=eventstds[:,-1],
                                  transform=ccrs.PlateCarree())
                ax.coastlines(resolution='110m',linewidth=0.6)
                #ax.gridlines(linewidth=0.5,color='grey')
                for radius in np.arange(15,120,15):
                    circle_points = cgd.Geodesic().circle(
                        lon=lon0, lat=lat0, radius=radius*111.19*1000, 
                        n_samples=180, endpoint=True)
                    circ = proj_ae.transform_points(ccrs.PlateCarree(),circle_points[:,0],circle_points[:,1])
                    ax.plot(circ[:,0],circ[:,1],linewidth=0.3,color='k')                
                plt.colorbar(cbar,shrink=0.3,fraction=0.03,pad=0.02,label='data std [s]')
            plt.savefig(os.path.join(foldername,"figure_event_stds.png"),bbox_inches='tight')
            #plt.show()
            plt.close(fig)
                        
        
        #%% create a 3D figure with plotly (optional)
        if create_3d_plot:
            import plotly.offline as pyo
            import plotly.graph_objs as go
            # Set notebook mode to work in offline
            # suppress any messages from init_notebook
            from io import StringIO 
            _stdout = sys.stdout
            sys.stdout = _stringio = StringIO()
            pyo.init_notebook_mode()
            del _stringio # free up variable
            sys.stdout = _stdout
        
            depthlevel = 0
            depthidx = np.where(Z[0,0,:]==depthlevel)[0][0]
            #    plt.figure()
            #    plt.pcolormesh(X[:,:,depthidx],Y[:,:,depthidx],avg_model[:,:,depthidx])
            #    plt.plot(stations[:,0],stations[:,1],'rv')
            #    plt.show()
            #    
            #    fig = plt.figure()
            #    ax = fig.gca(projection='3d')
            #    cset = ax.contourf(X[:,:,depthidx], Y[:,:,depthidx], avg_model[:,:,depthidx], zdir='z', offset=2.5, cmap=cm.coolwarm)
            #    #cset = ax.contourf(X[:,:,depthidx], Y[:,:,depthidx], avg_model[:,:,depthidx], zdir='x', offset=-40, cmap=cm.coolwarm)
            #    #cset = ax.contourf(X[:,:,depthidx], Y[:,:,depthidx], avg_model[:,:,depthidx], zdir='y', offset=40, cmap=cm.coolwarm)
            #    plt.show()
                
                
                
            colorscale=[[0.0, "rgb(165,0,38)"],
                        [0.1111111111111111, "rgb(215,48,39)"],
                        [0.2222222222222222, "rgb(244,109,67)"],
                        [0.3333333333333333, "rgb(253,174,97)"],
                        [0.4444444444444444, "rgb(254,224,144)"],
                        [0.5555555555555556, "rgb(224,243,248)"],
                        [0.6666666666666666, "rgb(171,217,233)"],
                        [0.7777777777777778, "rgb(116,173,209)"],
                        [0.8888888888888888, "rgb(69,117,180)"],
                        [1.0, "rgb(49,54,149)"]]
            
            
            def get_the_slice(x,y,z, surfacecolor,  colorscale=colorscale, cmin=-0.1, cmax=0.1, showscale=True):
                return go.Surface(x=x,  # https://plot.ly/python/reference/#surface
                                  y=y,
                                  z=z,
                                  surfacecolor=surfacecolor,
                                  colorscale=colorscale,
                                  showscale=showscale,
                                  cmin=cmin,
                                  cmax=cmax,
                                  colorbar=dict(thickness=20, ticklen=4))
            
            def get_lims_colors(surfacecolor): # color limits for a slice
                return np.min(surfacecolor), np.max(surfacecolor)
            
            
            z_slices = []
            z_slices_uncertainty = []
            for depthlevel in depthlevels[::2]:
                depthidx = np.abs(Z[0,0,:]-depthlevel).argmin()
                z_slices.append(get_the_slice(X[:,:,depthidx], Y[:,:,depthidx],
                                              -1*depthlevel*np.ones(np.shape(X[:,:,depthidx])),
                                              avg_model_dvs[:,:,depthidx]))
                z_slices_uncertainty.append(get_the_slice(X[:,:,depthidx], Y[:,:,depthidx],
                                                          -1*depthlevel*np.ones(np.shape(X[:,:,depthidx])),
                                                          modeluncertainties_vs[:,:,depthidx],
                                                          colorscale='reds',
                                                          cmin=0.,cmax=np.max(modeluncertainties_vs)))
            
            axis = dict(showbackground=True, 
                        backgroundcolor="rgb(230, 230,230)",
                        gridcolor="rgb(255, 255, 255)",      
                        zerolinecolor="rgb(255, 255, 255)"  
                        )
            
            
            layout = go.Layout(
                      title='Slices in volumetric data',
                      autosize=True,
                      #width=200,
                      #height=800,
                      scene=dict(xaxis=axis,
                                yaxis=axis, 
                                zaxis=dict(axis, range=[-45,0]),
                                aspectratio=dict(x=1, y=1, z=1)
                                )
                    )
                     
            fig = go.Figure(data=z_slices, layout=layout)
            fig.add_trace(go.Scatter3d(
                    x=all_stations[:,0],
                    y=all_stations[:,1],
                    z=np.zeros_like(all_stations[:,0]),
                    mode="markers",
                    marker=dict(
                            color='black',
                            size=4,
                            symbol='diamond')))
            #py.iplot(fig, filename='Slice-volumetric-1')
            #fig.show()
            fig.write_html(os.path.join(foldername,'figure_relative_Vs_3D.html'), auto_open=False)
            #fig.write_image("example3d_slices.png",width=1200,height=1800)
            
            fig = go.Figure(data=z_slices_uncertainty, layout=layout)
            fig.add_trace(go.Scatter3d(
                    x=all_stations[:,0],
                    y=all_stations[:,1],
                    z=np.zeros_like(all_stations[:,0]),
                    mode="markers",
                    marker=dict(
                            color='black',
                            size=4,
                            symbol='diamond')))
            fig.write_html(os.path.join(foldername,'figure_uncertainty_map_Vs_3D.html'), auto_open=False)   
                
                
                
            fig = go.Figure(
                data=[go.Bar(y=[2, 1, 3])],
                layout_title_text="A Figure Displayed with fig.show()"
            )
            fig.show(renderer='svg')
                
              
            import plotly.graph_objects as go
            fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
            fig.write_html('first_figure.html', auto_open=True)
        
    #%%
    print("done creating average model and plotting")

#%%
"""
From https://stackoverflow.com/questions/44152436/calculate-histograms-along-axis
"""
def hist_laxis(data2D, bins):
    
    n_bins = len(bins)-1
    idx = np.searchsorted(bins, data2D,'right')-1

    # We need to use bincount to get bin based counts. To have unique IDs for
    # each row and not get confused by the ones from other rows, we need to 
    # offset each row by a scale (using row length for this).
    scaled_idx = n_bins*np.arange(data2D.shape[0])[:,None] + idx
    
    # Some elements would be off limits, so get a mask for those
    bad_mask = (idx==-1) | (idx==n_bins)

    # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
    limit = n_bins*data2D.shape[0]
    scaled_idx[bad_mask] = limit

    # Get the counts and reshape to multi-dim
    counts = np.bincount(scaled_idx.ravel(),minlength=limit+1)[:-1]
    counts.shape = (data2D.shape[0],n_bins)
    
    return counts

#%%
if __name__ == "__main__":
    main()
    
