# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:40:38 2018

@author: emanuelk
"""
import os
import numpy as np
from scipy.io import netcdf_file
from scipy.interpolate import RegularGridInterpolator,interp1d
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
from matplotlib.colors import LinearSegmentedColormap
import pyproj
from cmcrameri import cm as cmcram
from scipy.spatial import Delaunay

plt.ioff()
  
#load model
modfile = 'modelsearch_bodywaves/average_3D_model.nc'
path = os.path.dirname(modfile)

model = netcdf_file(modfile)
x = model.variables['x'][:]
y = model.variables['y'][:]
try:
    projection_str = model.projection.decode()
except:
    print("Warning: projection not defined! Using lat0/lon0 = 46.4/10.8")
    projection_str = '+proj=tmerc +datum=WGS84 +lat_0=46.400000 +lon_0=10.800000'
projection = pyproj.Proj(projection_str)
X,Y = np.meshgrid(x,y)
mlons,mlats = projection(X*1000,Y*1000,inverse=True)
mdepth = model.variables['z'][:]
mvs = model.variables['vs'][:]
#vref = np.mean(mvs,axis=(0,1))
modelfu = RegularGridInterpolator((x,y,mdepth),mvs,method='nearest',
                                  bounds_error=False, fill_value=np.nan)
sigma_vs_fu = RegularGridInterpolator((x,y,mdepth), model.variables['vs_std'][:],
                                      method='nearest',bounds_error=False, fill_value=np.nan)
sigma_vp_fu = RegularGridInterpolator((x,y,mdepth), model.variables['vp_std'][:],
                                      method='nearest',bounds_error=False, fill_value=np.nan)
raycov_bw = RegularGridInterpolator((x,y,mdepth), model.variables['data_coverage_bw'][:],
                                    bounds_error=False, fill_value = 0)
raycov_sw = RegularGridInterpolator((x,y,mdepth), model.variables['data_coverage_sw'][:],
                                    bounds_error=False, fill_value = 0)
try:
    mvpvs = model.variables['vpvs'][:]
    vpvsfu = RegularGridInterpolator((x,y,mdepth),mvpvs,method='nearest',
                                     bounds_error=False,fill_value=np.nan)
    #vpvsref = np.mean(mvpvs,axis=(0,1))
except:
    print("no vpvs model")
horizons = np.unique(mdepth)

#%% define which regions are subject to large uncertainties
central_region = np.zeros_like(mvs[:,:,0],dtype=bool)
central_region[int(mvs.shape[0]*0.35):int(mvs.shape[0]*0.65),
               int(mvs.shape[1]*0.35):int(mvs.shape[1]*0.65)] = True

valid_regions = np.ones_like(mvs)*np.nan
if np.any(model.variables['data_coverage_sw'][:]>0):
    valid_regions[model.variables['data_coverage_sw'][:]<100000] = 1.
# if np.any(model.variables['data_coverage_bw'][:]>0):
#     valid_regions[model.variables['data_coverage_bw'][:]<100] = 1.

for di,depthlevel in enumerate(mdepth):   
    vs_std = np.std(mvs[central_region,di])
    #print(depthlevel,vs_std)
    #valid_regions[model.variables['vs_std'][:][:,:,di]>3*vs_std,di] = 1
    if depthlevel>60 and np.any(model.variables['data_coverage_bw'][:]>0):
        valid_regions[model.variables['data_coverage_bw'][:][:,:,di]<300,di] = 1.
        valid_regions[model.variables['vs_std'][:][:,:,di]>0.3,di] = 1
    else:
        valid_regions[model.variables['vs_std'][:][:,:,di]>0.4,di] = 1
#valid_regions[model.variables['vp_std'][:]>0.5] = 1.
#valid_regions[model.variables['vs_std'][:]>0.5] = 1.
valregfu = RegularGridInterpolator((x,y,mdepth),valid_regions,method='nearest',
                                   bounds_error=False,fill_value=1.)

# fig = plt.figure()
# ax1 = fig.add_subplot(2,2,1)
# cbar = ax1.pcolormesh(x,y,np.transpose(model.variables['data_coverage_sw'][:][:,:,3]))
# ax1.pcolormesh(x,y,np.transpose(valid_regions[:,:,30]),cmap=plt.cm.gray)
# plt.colorbar(cbar)
# ax2 = fig.add_subplot(2,2,2)
# ax2.pcolormesh(x,y,np.transpose(model.variables['data_coverage_bw'][:][:,:,3]))

# plt.show()


#%%

statnames = []
stats = []
with open("./dataset_bodywaves/p_picks_small_dataset/stationlist.dat","r") as f:
    lines = f.readlines()
    for line in lines:
        statname,lat,lon,elev = line.split()
        statnames.append(statname)
        stats.append([float(lat),float(lon)])
stats = np.vstack(stats)
points = np.vstack((stats+0.5,stats-0.5,stats+np.array([-.5,.5]),stats+np.array([.5,-.5])))
hull = Delaunay(points)
overlay = np.ones_like(mlons)
inhull = np.reshape(hull.find_simplex(np.column_stack((mlats.flatten(),mlons.flatten())))>=0,overlay.shape)
overlay[inhull] = np.nan
# inhull[:] = True
# print("Warning")

vref = np.zeros_like(mdepth)
vpvsref = np.zeros_like(mdepth)
for zi,z in enumerate(mdepth):
    vref[zi] = np.mean(mvs[:,:,zi][inhull.T])
    vpvsref[zi] = np.mean(mvpvs[:,:,zi][inhull.T])
np.savetxt(os.path.join(path,"reference_velocity_1d.txt"),
           np.column_stack((mdepth,vref*vpvsref,vref)),
           header="Depth Vp Vs",fmt="%6.1f %5.2f %5.2f")
# print("using mean prior as reference model.")
# vref = refvsfu(mdepth)
# vpvsref = 1.75

#%% PLOT MAP
from matplotlib.collections import LineCollection

proj = ccrs.TransverseMercator(central_longitude=12,
                               central_latitude=45,
                               approx=False)

extent = [2, 22, 39, 51.5]
    
for param in ['Vs','Vp','VpVs']:
    colmap = plt.cm.coolwarm_r
    label = 'd%s [%%]' %param
    if param=='Vs':
        dv = (mvs-vref)/vref * 100.
    elif param=='Vp':
        dv = (mvs*mvpvs - vref*vpvsref)/(vref*vpvsref) * 100  
    elif param=='VpVs':
        dv = mvpvs
        colmap = cmcram.cork
        label = 'Vp/Vs ratio'
    
    fig = plt.figure(figsize=(14,15))
    depthplot = [60,80,150,200,250,300]
    for di,depth in enumerate(depthplot):
        idx_z = (np.abs(mdepth-depth)).argmin()
        axm = fig.add_subplot(3,2,di+1,projection=proj)
        axm.coastlines(resolution='50m')
        axm.add_feature(cf.BORDERS.with_scale('50m'))
        #axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
        #axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
        dV = dv[:,:,idx_z].T
        valreg = valid_regions[:,:,idx_z].T
        if param=='VpVs':
            vmin = 1.6
            vmax = 1.9
        else:
            vmax = np.max([np.percentile(dV,98),np.percentile(dV,2)])
            vmin = -vmax
        vminmax = np.max([np.percentile(dV,98),np.percentile(dV,2)])
        if True:
            cmap = axm.pcolormesh(mlons,mlats,dV,cmap=colmap,vmin=vmin,vmax=vmax,
                           transform=ccrs.PlateCarree(),shading='nearest')
            plt.colorbar(cmap,fraction=0.04,shrink=0.5,pad=0.02,label=label)
        axm.pcolormesh(mlons,mlats,valreg,cmap=plt.cm.gray,
                       transform=ccrs.PlateCarree(),shading='nearest',alpha=0.4)
        gl = axm.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                       #xlocs=[6,10,14,18],ylocs=[43,45,47,49],
                       linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        axm.set_extent(extent, crs=ccrs.PlateCarree())
        axm.text(0.03,0.03,str(int(mdepth[idx_z]))+" km",fontsize=16,
                 bbox={'facecolor':'white','alpha':0.7,'pad':5},
                 transform=axm.transAxes,
                 horizontalalignment='left',verticalalignment='bottom')
        #plt.savefig("crosssection_figure_section_map_%.1fkm_%.1f-%.1f_%.1f-%.1f.jpg" %(depth,lat1,lon1,lat2,lon2),bbox_inches='tight')
    plt.savefig(os.path.join(path,"figure_%s_maps.jpg" %param),bbox_inches='tight',dpi=300)
    plt.close(fig)
    
#%%
from scipy.interpolate import RectBivariateSpline
import matplotlib.patheffects as path_effects

profiles = {'P1':[3,45.3,17,45],
            'P2':[6,51,14.3,41],
            #'T (TRANSALP)':[11.8,49,13,41],
            #'B (TRANSALP)':[11.9,48.3,12.36,45.17],
            #'E (EASI)':[13.3,50,13.3,41],
            'P3':[17.8,51, 9,42],
            #'A':[8,42,19,47],
            }

topo = netcdf_file('/home/emanuel/PLOTS/ETOPO1_Ice_g_gmt4.grd','r',mmap=False)
tlons = topo.variables['x'][:] # Latitude values in netCDF file
tlats = topo.variables['y'][:] # Longitude ...
elev= topo.variables['z'][:]
topo.close()
latidx = (tlats>extent[2]-1)*(tlats<extent[3]+1)
lonidx = (tlons>extent[0]-5)*(tlons<extent[1]+5)
LONtopo,LATtopo = np.meshgrid(tlons[lonidx],tlats[latidx])
topofu = RectBivariateSpline(tlats[latidx],tlons[lonidx],elev[np.ix_(latidx,lonidx)])

fig = plt.figure()
#axm = plt.axes(projection=proj)
ax = fig.add_subplot(projection=proj)
subsample = 1
ax.pcolormesh(LONtopo[::subsample,::subsample],LATtopo[::subsample,::subsample],
             elev[np.ix_(latidx,lonidx)][::subsample,::subsample],
             cmap=cmcram.bukavu,shading='nearest',
             vmin=-3005,vmax=3000,transform=ccrs.PlateCarree())
#ax.contourf(LONtopo[::subsample,::subsample],LATtopo[::subsample,::subsample],
#             elev[np.ix_(latidx,lonidx)][::subsample,::subsample],
#             cmap=cmcram.bukavu,extend='both',
#             levels=np.linspace(-3005,3000,41),transform=ccrs.PlateCarree())
#ax.plot(stats[:,1],stats[:,0],'kv',ms=3,transform=ccrs.PlateCarree())
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m')
ax.add_feature(cf.BORDERS.with_scale('50m'))
ax.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
ax.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                   #xlocs=[6,10,14,18],ylocs=[43,45,47,49],
                   linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.bottom_labels = True
gl.left_labels = True
gl.right_labels = False
for i,profile_name in enumerate(profiles):
    profile = profiles[profile_name]
    lon1,lat1,lon2,lat2 = profile
    ax.plot([lon1,lon2],[lat1,lat2],'w',linewidth=2,transform=ccrs.PlateCarree())
    ax.plot([lon1,lon2],[lat1,lat2],'k',linewidth=1.5,transform=ccrs.PlateCarree())
    txt = ax.text(lon1-0.1,lat1+0.1,profile_name,fontsize=10,
             transform=ccrs.PlateCarree())
    txt.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'),
                                 path_effects.Normal()])
plt.savefig(os.path.join(path,"profile_locations.png"),bbox_inches='tight',
            transparent=True,dpi=300)
plt.close(fig)        

#%%
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial import KDTree
from obspy.geodetics import gps2dist_azimuth
from matplotlib.ticker import FixedLocator
from scipy.ndimage import gaussian_filter

custom_cmap = LinearSegmentedColormap.from_list('my_list',
        [(1,1,0.7),(0.97,0.92,0.66),(1,0.9,0.04),(0.96,0.75,0.09),
         (0.8,0.55,0.38),(0.77,0.91,0.22),(0.4,0.9,0.2),(0.22,0.6,0.55),
         (0,0.35,1),(0.02,0.2,1),(0,0,0.9)])

#gs = gridspec.GridSpec(nrows=6, ncols=500,hspace=0.05,wspace=0.05)

vsprofiles = np.vstack(np.split(mvs.flatten(),np.product(mvs.shape[:2])))
vpvsprofiles = np.vstack(np.split(mvpvs.flatten(),np.product(mvs.shape[:2])))

for prof_type in ['Vs_crust','Vs_mantle','Vp_crust','Vp_mantle','Vpstd_mantle','Vsstd_mantle','VpVs_crust','VpVs_mantle']:
    print(prof_type)
    
    kdt = KDTree(np.column_stack((mlons.flatten(),mlats.flatten())))
    results = []
    for i,profile_name in enumerate(profiles):
        fig = plt.figure(figsize=(12,8))
        profile = profiles[profile_name]
        lon1,lat1,lon2,lat2 = profile
        dist,az,baz = gps2dist_azimuth(lat1,lon1,lat2,lon2)
        x1,y1 = projection(lon1,lat1)
        x2,y2 = projection(lon2,lat2)
        sample_rate = 10. #every 10km
        num = int(dist/sample_rate/1000.)+1
        modelx = np.linspace(0,dist/1000.,num)
        profile_x = np.linspace(x1,x2,num)
        profile_y = np.linspace(y1,y2,num)
        profile_lons,profile_lats = projection(profile_x,profile_y,inverse=True)
        profile_x /= 1000.
        profile_y /= 1000.
        
        profile_coords = []
        for zp in mdepth:
            for xp,yp in zip(profile_x,profile_y):
                profile_coords.append([xp,yp,zp])
        profile_coords = np.vstack(profile_coords)
        profile_vs = np.reshape(modelfu(profile_coords),(len(mdepth),len(profile_x)))
        profile_dvs = ((profile_vs.T - vref)/vref * 100).T
        profile_vpvs = np.reshape(vpvsfu(profile_coords),(len(mdepth),len(profile_x)))
        profile_dvpvs = ((profile_vpvs.T - vpvsref)/vpvsref * 100).T
        profile_vp = profile_vs*profile_vpvs
        profile_dvp = ((profile_vp.T - vref*vpvsref)/(vref*vpvsref) * 100).T
        profile_sigma_vp = np.reshape(sigma_vp_fu(profile_coords),(len(mdepth),len(profile_x)))
        profile_sigma_vs = np.reshape(sigma_vs_fu(profile_coords),(len(mdepth),len(profile_x)))
        profile_overlay = np.reshape(valregfu(profile_coords),(len(mdepth),len(profile_x)))
        
        profile_fu = interp1d(np.sqrt(profile_lons**2+profile_lats**2),modelx)    
        
        # plot section
        vmin = -5
        vmax = 5
        ticks = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        if prof_type == 'Vs_crust':
            mod = profile_vs
            colmap = custom_cmap
            vmin = 2.5
            vmax = 4.8
            ticks = np.arange(vmin,vmax+0.1,0.3)
        elif prof_type == 'Vs_mantle':
            mod = profile_dvs
            colmap = plt.cm.coolwarm_r
        elif prof_type == 'Vp_crust':
            mod = profile_vp
            colmap = custom_cmap
            vmin = 2.5*1.7
            vmax = 4.8*1.8
            ticks = np.arange(vmin,vmax+0.1,0.5)
        elif prof_type == 'Vp_mantle':
            mod = profile_dvp
            colmap = plt.cm.coolwarm_r
        elif prof_type == 'Vpstd_mantle':
            mod = profile_sigma_vp
            colmap = plt.cm.hot
            vmin=0
            vmax=1
            ticks = [0,0.25,0.5,0.75,1]
        elif prof_type == 'Vsstd_mantle':
            mod = profile_sigma_vs
            colmap = plt.cm.hot
            vmin=0
            vmax=1
            ticks = [0,0.25,0.5,0.75,1]
        else:
            mod = profile_vpvs
            colmap = cmcram.cork
            vmin = 1.6
            vmax = 2.0
            ticks = [1.6,1.7,1.8,1.9,2.0]
            
        cs = None
        ax = fig.add_subplot()
        # cbar = ax.contourf(modelx,depths,section_crust,
        #                    cmap=plt.cm.coolwarm_r,levels=np.linspace(2.5,4.5,25),
        #                    extend='both')
        if 'mantle' in prof_type:
            mod = gaussian_filter(mod,0.6)
        cbar = ax.contourf(modelx,mdepth,mod,cmap=colmap,levels=np.linspace(vmin,vmax,31),extend='both')
        #cbar = ax.pcolormesh(modelx,mdepth,mod,cmap=colmap,vmin=vmin,vmax=vmax)
        if 'mantle' in prof_type:
            cs = ax.contour(modelx,mdepth,mod,cmap=plt.cm.seismic_r,
                            vmin=vmin,vmax=vmax,levels=np.linspace(-10,10,21),linewidths=0.5)
            idx3 = np.where(cs.levels==3)[0]
            ax.clabel(cs, cs.levels[idx3], inline=True, fmt="%.1f", fontsize=8)
            
        if 'Vs_crust' == prof_type:
            cs = ax.contour(modelx,mdepth,profile_vs,levels=[4.15],linewidths=0.7,colors='black')
            ax.clabel(cs, cs.levels, inline=True, fmt="%.2f", fontsize=8)
        elif 'Vp_crust' == prof_type:
            cs = ax.contour(modelx,mdepth,profile_vp,levels=[7.25],linewidths=0.7,colors='black')
            ax.clabel(cs, cs.levels, inline=True, fmt="%.2f", fontsize=8)
            
        ax.pcolormesh(modelx,mdepth,profile_overlay,cmap=plt.cm.gray,alpha=0.5,zorder=2)
        
        #cs = ax.contour(modelx,depths,section,levels=[2.5,3.4,3.6, 3.95],linewidths=0.7,colors='gray')
        #ax.clabel(cs, cs.levels, inline=True, fmt="%.2f", fontsize=8)
        xaxis = modelx
            
        # plot topography
        profile_topo = topofu.ev(profile_lats,profile_lons)
        profile_topo[profile_topo<0.] = 0.
        plt.plot(xaxis,profile_topo/1000.*-5,lw=1,color='black')
        plt.fill_between(xaxis,profile_topo/1000.*-5.,0,color='grey')    
        #plt.plot(np.arange(30)*50.,np.zeros(30),'o',ms=7,color='white',markeredgecolor='black',markeredgewidth=0.5)
        
        profile_label = profile_name.split()[0]
        ax.text(-0.05,1.0,profile_label,fontweight='bold',fontsize=18,transform=ax.transAxes)
        ax.text(-0.05,1.0,"   "+profile_name[2:],fontsize=14,transform=ax.transAxes)
        #ax.text(np.max(modelx),79.5,"km",horizontalalignment='right',fontsize=10)
        #ax.text(np.max(modelx)-5,93,"$^{\circ}$",horizontalalignment='right',fontsize=14)
        if 'crust' in prof_type:
            dy = -0.07
        else:
            dy = 0
        if np.abs(az-90)<45:
            ax.text(0.005,0.95+dy,"W",fontsize=13,transform = ax.transAxes)
            ax.text(0.985,0.95+dy,"E",fontsize=13,transform = ax.transAxes)
        else:
            #if not "(" in profile_name:
            ax.text(0.005,0.95+dy,"N",fontsize=13,transform = ax.transAxes)
            ax.text(0.985,0.95+dy,"S",fontsize=13,transform = ax.transAxes)
        if 'crust' in prof_type:
            ax.set_ylim(70,-18)
            ax.set_yticks([0,10,20,30,40,50,60,70])
            ax.yaxis.set_minor_locator(FixedLocator(np.arange(0,70,2)))
            plt.legend(bbox_to_anchor=(0.99,0.01),loc='lower right')
            colbar = plt.colorbar(cbar,pad=0.02,shrink=0.25,fraction=0.05,
                                  label='%s' %(prof_type.split("_")[0]),ticks = ticks)
            axspacing = -0.2
            ax.set_aspect(2)
        else:
            ax.set_ylim(600,-50)
            colbar = plt.colorbar(cbar,pad=0.02,shrink=0.4,fraction=0.04,
                                  label='%s' %(prof_type.split("_")[0]),ticks = ticks)
            axspacing = -0.05
            ax.set_aspect('equal')
        if cs is not None:
            colbar.add_lines(cs)
        ax.set_xlim(0,modelx[-1])
        ax.yaxis.set_ticks_position('both')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if az>45 and az<135:
            secax = ax.secondary_xaxis(
                axspacing,functions=(interp1d(modelx,profile_lons,fill_value='extrapolate'),
                                interp1d(profile_lons,modelx,fill_value='extrapolate')))
        else:
            secax = ax.secondary_xaxis(
                axspacing,functions=(interp1d(modelx,profile_lats,fill_value='extrapolate'),
                                interp1d(profile_lats,modelx,fill_value='extrapolate')))
        ax.xaxis.set_major_formatter(FormatStrFormatter('$%d~km$'))
        secax.xaxis.set_major_formatter(FormatStrFormatter('$%.1f^{\circ}$'))
            #secax.set_xlabel("latitude")
            #secax.xaxis.set_label_coords(.5,10)
        #ax.set_xlabel("distance [km]")
        #ax.xaxis.set_label_coords(.95,-.02)
        ax.set_ylabel("depth [km]")
        ax.set_rasterization_zorder(1)
        
        # plot colorbar
        if False:#i==2:
            #cax = fig.add_subplot(gs[i,int(np.max(modelx)+65):int(np.max(modelx)+75)])
            plt.colorbar(cbar,pad=0.02,shrink=0.5,fraction=0.05,label='%s' %(prof_type.split("_")[0]))
        # if i==3:
        #     plt.legend(bbox_to_anchor=(1.01,0.5),loc='upper left')
        
        #plt.show()
        #pause
        plt.savefig(os.path.join(path,"profiles_%s_%s.png" %(prof_type,profile_label)),
                    bbox_inches='tight',dpi=300,transparent=True)
        plt.close(fig)
