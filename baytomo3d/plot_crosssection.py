# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:40:38 2018

@author: emanuelk
"""
import sys, os
import numpy as np
from geographiclib.geodesic import Geodesic
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

if len(sys.argv)<2:
    sys.argv=[1,2,3,4,5,6,"mantle"]
    sys.argv[1] = 42.31
    sys.argv[2] =  9.67
    sys.argv[3] = 47.33
    sys.argv[4] = 21.0
    sys.argv[5] = 160.
    azimuth = False


if len(sys.argv) == 7:
    if sys.argv[6].lower()=='mantle':
        mantle = True
    else:
        mantle = False
    if float(sys.argv[4])>25.:
        print("plotting cross section from",sys.argv[1],sys.argv[2],"with azimuth",sys.argv[3],"and length",sys.argv[4],".")
        azimuth = True
    else:    
        print("plotting cross section from",sys.argv[1],sys.argv[2],"to",sys.argv[3],sys.argv[4],".")
        azimuth = False
else:
    print("You have to give 6 input parameters:")
    print("python plot_crosssection.py lat1 lon1 lat2 lon2 depth[km] mantle")
    print("or:")
    print("python plot_crosssection.py lat1 lon1 azimuth length[m] depth[km] crust")
    print("for example:")
    print("python plot_crosssection.py 45.3 4.5 95 800000 100 mantle")
    sys.exit()

#sys.argv=[1,2,3,4,5,6]
#sys.argv[1] = 45.2
#sys.argv[2] =  6
#sys.argv[3]= 45
#sys.argv[4]= 12.9
#azimuth = False

depth = float(sys.argv[5])
if azimuth:
    lat1 = float(sys.argv[1])
    lon1 = float(sys.argv[2])
    azi = float(sys.argv[3])
    length = float(sys.argv[4])
    Line = Geodesic.WGS84.Line(lat1,lon1,azi)
    geoline = Line.Position(length)
    lat2 = geoline['lat2']
    lon2 = geoline['lon2']    
else:
    lat1 = float(sys.argv[1])
    lon1 = float(sys.argv[2])    
    lat2 = float(sys.argv[3])
    lon2 = float(sys.argv[4])
    geoline = Geodesic.WGS84.Inverse(lat1,lon1,lat2,lon2)
    length = geoline['s12']

if (lat1>51 or lat1<40 or lon1>24 or lon1<4 or
    lat2>51 or lat2<40 or lon2>24 or lon2<4):
    print("Warning: section outside model region, filling with NaNs.")
    print("valid latitudes: 40 - 51")
    print("valid longitudes: 4 - 20")
    #sys.exit()
  
#load model
#modfile = '/home/emanuel/Bayesian_tomo/rjmcmc3d_2017/test_bodywaves/test12_bodywavesonly_includingpdiff_maxdepth600km/average_3D_model.nc'
modfile = '/home/emanuel/Bayesian_tomo/rjmcmc3d_2017/joint_2023_10_17_bwS/average_3D_model.nc'
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
try:
    mvpvs = model.variables['vpvs'][:]
    vpvsfu = RegularGridInterpolator((x,y,mdepth),mvpvs,method='nearest',
                                     bounds_error=False,fill_value=np.nan)
    #vpvsref = np.mean(mvpvs,axis=(0,1))
except:
    print("no vpvs model")
horizons = np.unique(mdepth)

statnames = []
stats = []
with open("/home/emanuel/Bayesian_tomo/dataset_teleseismic_traveltimes/p_picks_0.03-0.5_Hz/stationlist.dat","r") as f:
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

vref = np.zeros_like(mdepth)
vpvsref = np.zeros_like(mdepth)
for zi,z in enumerate(mdepth):
    vref[zi] = np.mean(mvs[:,:,zi][inhull.T])
    vpvsref[zi] = np.mean(mvpvs[:,:,zi][inhull.T])

     

 # # LOAD PREM # #
prem_dev = False
prem = np.loadtxt("PREM_1s.csv",delimiter=',')
vsprem = np.sqrt((2*prem[:,5]**2+prem[:,6]**2)/3.) #voigt average
vsprem[vsprem<1.0] = min(vsprem[vsprem>1.0])
premfu = interp1d(prem[:,1],vsprem)

# create map
proj = ccrs.TransverseMercator(central_longitude=12,
                               central_latitude=45,
                               approx=False)
   
x1,y1 = proj.transform_point(lon1,lat1,ccrs.PlateCarree())
x2,y2 = proj.transform_point(lon1,lat1,ccrs.PlateCarree())
sample_rate = 10. #every 10km
num = int(length/sample_rate/1000.)+1
#pts = m.gcpoints(lon1,lat1,lon2,lat2,num)
modelx = np.linspace(0,length/1000.,num)
g = pyproj.Geod(ellps='WGS84')
az12,az21,dist = g.inv(lon1,lat1,lon2,lat2)
del_s = dist/(num+1)
gcpoints = g.fwd_intermediate(lon1, lat1, az12, num, del_s)
profile_lons = np.array(gcpoints.lons)
profile_lats = np.array(gcpoints.lats)
gcpoints_xy = proj.transform_points(ccrs.PlateCarree(),profile_lons,profile_lats)
profile_x = gcpoints_xy[:,0]
profile_y = gcpoints_xy[:,1]
profile_xp,profile_yp = projection(profile_lons,profile_lats)
modelcoords = []                
for xp,yp in zip(profile_xp,profile_yp):
    for z in mdepth:
        modelcoords.append([xp/1000.,yp/1000.,z])
modelcoords = np.array(modelcoords)
modelvals = modelfu(modelcoords)

vmodel = modelvals.reshape(len(modelx),len(mdepth))
mohodepth = []
for line in vmodel:
    if np.isnan(line).any():
        mohodepth.append(np.nan)
    else:
        idxmoho = np.where(line>4.3)[0][0]
        mohodepth.append(mdepth[idxmoho])
vmodel2 = np.copy(vmodel)
vmodel2 = np.ma.masked_less(vmodel2,4.3)
premvels = premfu(mdepth)
premvels[premvels<3.9] = 4.4
premdvsmodel = (vmodel2-premvels)/premvels*100.
dvsmodel = (vmodel2-vref)/vref*100. 

#%% PLOT CROSS SECTION

cmin = 1.5
cmax = 4.3
cdict = {'red':   ((0.0, 1.0, 1.0), #white
                   ((2.3-cmin)/(cmax-cmin), 1.0, 1.0), # red
                   ((3.0-cmin)/(cmax-cmin), 1.0, 1.0), #yellow-orange
                   ((3.4-cmin)/(cmax-cmin), 0.9, 0.9), #yellow
                   ((3.5-cmin)/(cmax-cmin), 0.7, 0.7),
                   ((3.8-cmin)/(cmax-cmin), 0.6, 0.6),
                   ((4.0-cmin)/(cmax-cmin), 0.6, 0.6),
                   (1.0, 0.6, 0.6)),

         'green': ((0.0, 1.0, 1.0),
                   ((2.3-cmin)/(cmax-cmin), 0.0, 0.0),
                   ((3.0-cmin)/(cmax-cmin), 0.9, 0.9),
                   ((3.4-cmin)/(cmax-cmin), 1.0, 1.0),
                   ((3.5-cmin)/(cmax-cmin), 1.0, 1.0),
                   ((3.8-cmin)/(cmax-cmin), 0.8, 0.8),
                   ((4.0-cmin)/(cmax-cmin), 0.8, 0.8),
                   ((4.2-cmin)/(cmax-cmin), 0.6, 0.6),
                   (1.0, 0.5, 0.5)),

         'blue':  ((0.0, 1.0, 1.0),
                  ((2.3-cmin)/(cmax-cmin), 0.0, 0.0),
                   ((3.5-cmin)/(cmax-cmin), 0.0, 0.0),
                   ((3.8-cmin)/(cmax-cmin), 0.0, 0.0),
                   ((4.0-cmin)/(cmax-cmin), 0.0, 0.0),
                   ((4.2-cmin)/(cmax-cmin), 0.0, 0.0),
                   (1.0, 0., 0.))
        }
colmap = LinearSegmentedColormap('colmap', cdict)

vlim = 5
X,Z = np.meshgrid(modelx,mdepth)
levels = np.linspace(-vlim,vlim,50,endpoint=True)
levels2 = np.linspace(1.5,4.3,50,endpoint=True)
if mantle:
    f = plt.figure(figsize=(12,8))
else:
    f=plt.figure(figsize=(12,3))
#cbar1 = plt.pcolormesh(X,Z,vmodel.T,cmap=colmap,zorder=-1)
#cbar1 = plt.contourf(modelx,mdepth,vmodel.T,levels2,cmap=colmap,zorder=-1,extend='both')
#CS=plt.contour(modelx,mdepth,vmodel.T, [2.0,3.0,3.3,3.5,3.77,4.0], colors='grey',linewidths=0.3)
if mantle:
    cbar2 = plt.contourf(modelx,mdepth,dvsmodel.T,levels,cmap=plt.cm.seismic_r,
                         zorder=-1,extend='both')
else:
    #plt.clabel(CS, fontsize=9, inline=0,fmt="%.1f")
    plt.plot(modelx,np.zeros(len(modelx)),'k-',lw=0.3)
plt.plot(modelx,mohodepth,linewidth=3,color='white')
ax = plt.gca()
ax.set_aspect(1)
ax.set_ylim(600,0)
ax.set_xlabel("distance [km]")
ax.set_ylabel("depth [km]")
plt.title("Cross section from %.1f/%.1f to %.1f/%.1f" %(lat1,lon1,lat2,lon2))
if mantle:
    #cax1 = f.add_axes([0.2,0.1,0.1,0.01])
    #plt.colorbar(cbar1,cax=cax1,ticks=[2,3,4],label='Vs crust [km/s]',orientation='horizontal')
    cax2 = f.add_axes([0.2,0.1,0.2,0.02])
    plt.colorbar(cbar2,cax=cax2,extend='neither',ticks=[-5,-2,0,2,5],label='dVs mantle [%]',orientation='horizontal')
else:
    #plt.colorbar(cbar1,ticks=[2,3,4],shrink=0.6,fraction=0.1,label='Vs crust [km/s]')
    ax.set_yticks([0,20,40,60])
    ax.set_ylim(75,-5)
plt.savefig("crosssection_figure_section_%.1f-%.1f_%.1f-%.1f.jpg" %(lat1,lon1,lat2,lon2),bbox_inches='tight')
plt.close(f)

N = int(len(X.flatten())/len(modelx))
section_values = np.column_stack((np.tile(profile_lats,N),np.tile(profile_lons,N),
                                  X.flatten(),Z.flatten(),vmodel.T.flatten()))
np.savetxt("crosssection_values_%.1f-%.1f_%.1f-%.1f.txt" %(lat1,lon1,lat2,lon2),
           section_values,fmt='%.3f\t%.3f\t%.3f\t%.3f\t%.2f',header='LAT LON DIST_ALONG_PROFILE DEPTH VS')

#%% PLOT MAP
import cartopy.io.shapereader as shpreader
from matplotlib.collections import LineCollection
plt.ioff()

extent = [2, 22, 39, 51.5]

reader = shpreader.Reader("/home/emanuel/Maps Alpine Chains & Mediterranean/GIS_digitization/tectonic_maps_4dmb/shape_files/faults_alcapadi")
faults = reader.records()
lines = []
linemarkers = []
linewidths = []
line_labels = []
for fault in faults:
    if fault.attributes["fault_type"] == 1:
        linemarkers.append('solid')
        linewidths.append(1.5)
    elif fault.attributes["fault_type"] == 2:
        linemarkers.append('solid')
        linewidths.append(1.)
    else:
        linemarkers.append('dashed')
        linewidths.append(1.5)
    if fault.geometry.geom_type=='LineString':
        lines.append(np.array(fault.geometry.coords))
    elif fault.geometry.geom_type=='MultiLineString':
        for geo in fault.geometry.geoms:
            lines.append(np.array(geo.coords))
    else:
        print("did not recongize geometry type",fault.geometry.geom_type)
    line_labels.append(fault.attributes['fault_name'])
    
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
    if mantle:
        depthplot = [60,80,150,200,250,300]
    else:
        depthplot = [0,3,8,12,20,30]
    for di,depth in enumerate(depthplot):
        idx_z = (np.abs(mdepth-depth)).argmin()
        axm = fig.add_subplot(3,2,di+1,projection=proj)
        axm.coastlines(resolution='50m')
        axm.add_feature(cf.BORDERS.with_scale('50m'))
        #axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
        #axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
        dV = dv[:,:,idx_z].T
        if param=='VpVs':
            vmin = 1.6
            vmax = 1.9
        else:
            vmax = np.max([np.percentile(dV,98),np.percentile(dV,2)])
            vmin = -vmax
        vminmax = np.max([np.percentile(dV,98),np.percentile(dV,2)])
        if True:#mantle:
            cmap = axm.pcolormesh(mlons,mlats,dV,cmap=colmap,vmin=vmin,vmax=vmax,
                           transform=ccrs.PlateCarree(),shading='nearest')
            plt.colorbar(cmap,fraction=0.04,shrink=0.5,pad=0.02,label=label)
        axm.pcolormesh(mlons,mlats,overlay,cmap=plt.cm.gray,
                       transform=ccrs.PlateCarree(),shading='nearest',alpha=0.4)
        #axm.plot([lon1,lon2],[lat1,lat2],transform=ccrs.PlateCarree(),color='black')
        axm.add_collection(LineCollection(lines,linewidths=linewidths,
                                      linestyles=linemarkers,colors='red',
                                      transform=ccrs.PlateCarree()))
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
ax.add_collection(LineCollection(lines,linewidths=linewidths,
                                  linestyles=linemarkers,colors='red',
                                  transform=ccrs.PlateCarree()))
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
import matplotlib.gridspec as gridspec
from scipy.spatial import KDTree
from obspy.geodetics import gps2dist_azimuth
from shapely.geometry import LineString
import shapely
from matplotlib.ticker import FixedLocator
from scipy.ndimage import gaussian_filter

custom_cmap = LinearSegmentedColormap.from_list('my_list',
        [(1,1,0.7),(0.97,0.92,0.66),(1,0.9,0.04),(0.96,0.75,0.09),
         (0.8,0.55,0.38),(0.77,0.91,0.22),(0.4,0.9,0.2),(0.22,0.6,0.55),
         (0,0.35,1),(0.02,0.2,1),(0,0,0.9)])

spadamoho = []
with open("moho_all_cssrf.dat") as f:
    for line in f.readlines():
        if "# Moho number" in line:
            mohonumber = int(line[-2])
            #print "Moho:",mohonumber
        if len(line.split()) == 5:
            try:
                test = map(float,line.split()[2:])
                #if line.split()[-1]!='NaN':
                spadamoho.append([float(line.split()[2]),float(line.split()[3]),float(line.split()[4]),mohonumber])
            except:
                pass
spadamoho = np.array(spadamoho)
kdtsmoho = KDTree(spadamoho[:,:2])

#gs = gridspec.GridSpec(nrows=6, ncols=500,hspace=0.05,wspace=0.05)

vsprofiles = np.vstack(np.split(mvs.flatten(),np.product(mvs.shape[:2])))
vpvsprofiles = np.vstack(np.split(mvpvs.flatten(),np.product(mvs.shape[:2])))

for prof_type in ['Vs_crust','Vs_mantle','Vp_crust','Vp_mantle','VpVs_crust','VpVs_mantle']:
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
    
        dsmoho,ismoho = kdtsmoho.query(np.column_stack((profile_lons,profile_lats)),k=1)
        smoho_depths = spadamoho[ismoho,2:]
        smoho_depths[dsmoho>0.05] = 999.
        
        profile_fu = interp1d(np.sqrt(profile_lons**2+profile_lats**2),modelx)
        intersections = []
        intersection_labels = []
        profile_line = LineString(np.column_stack((profile_lons,profile_lats)))
        for li,fault in enumerate(lines):
            line = LineString(fault)
            if line.intersects(profile_line):
                int_pt = line.intersection(profile_line)
                if type(int_pt) == shapely.geometry.multipoint.MultiPoint:
                    intersections.append(profile_fu(np.sqrt(int_pt.geoms[-1].x**2+int_pt.geoms[-1].y**2)))
                    intersection_labels.append(line_labels[li])
                    int_pt = int_pt.geoms[0]
                intersections.append(profile_fu(np.sqrt(int_pt.x**2+int_pt.y**2)))
                intersection_labels.append(line_labels[li])
                if np.max(modelx) - intersections[-1] < 30:
                    intersections[-1] -= 20
                if intersection_labels[-1] == "PF" and int_pt.x>10.9 and int_pt.x<11.8:
                    intersection_labels[-1] = "GF"
                if "EASI" in profile_name and intersection_labels[-1]=="KNF":
                    intersection_labels[-1] = ""
        for li in range(len(intersection_labels)):
            if intersection_labels[li] == 'Mölltal Fault':
                intersection_labels[li] = 'MF'
    
        
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
        mod = gaussian_filter(mod,0.8)
        cbar = ax.contourf(modelx,mdepth,mod,cmap=colmap,levels=np.linspace(vmin,vmax,30),extend='both')
        if 'mantle' in prof_type:
            cs = ax.contour(modelx,mdepth,mod,cmap=plt.cm.seismic_r,
                            vmin=vmin,vmax=vmax,levels=np.linspace(-10,10,21),linewidths=0.5)
            idx3 = np.where(cs.levels==3)[0]
            ax.clabel(cs, cs.levels[idx3], inline=True, fmt="%.1f", fontsize=8)
            
        #std_overlay = sectionstd.copy()
        #std_overlay[std_overlay<0.2] = np.nan
        #std_overlay[~np.isnan(std_overlay)] = 1.
        #ax.pcolormesh(modelx,depths,std_overlay,cmap=plt.cm.gray_r,alpha=0.7)
        if 'Vs_crust' == prof_type:
            cs = ax.contour(modelx,mdepth,profile_vs,levels=[4.15],linewidths=0.7,colors='black')
            ax.clabel(cs, cs.levels, inline=True, fmt="%.2f", fontsize=8)
        elif 'Vp_crust' == prof_type:
            cs = ax.contour(modelx,mdepth,profile_vp,levels=[7.25],linewidths=0.7,colors='black')
            ax.clabel(cs, cs.levels, inline=True, fmt="%.2f", fontsize=8)
            
        #cs = ax.contour(modelx,depths,section,levels=[2.5,3.4,3.6, 3.95],linewidths=0.7,colors='gray')
        #ax.clabel(cs, cs.levels, inline=True, fmt="%.2f", fontsize=8)
        xaxis = modelx
        mohocols = {1.: "dashed", 2: "dashdot", 3.: "lightgreen"}
        moholabels = {1.: "Europe", 2: "Adria", 3.: "lightgreen"}
        for im in [1.,2.]:
            ax.plot(modelx[smoho_depths[:,1]==im],smoho_depths[smoho_depths[:,1]==im,0],color='k',
                    ls=mohocols[im],lw=1,label='$Moho~%s$' %moholabels[im])
            
        # plot topography
        profile_topo = topofu.ev(profile_lats,profile_lons)
        profile_topo[profile_topo<0.] = 0.
        plt.plot(xaxis,profile_topo/1000.*-5,lw=1,color='black')
        plt.fill_between(xaxis,profile_topo/1000.*-5.,0,color='grey')    
        #plt.plot(np.arange(30)*50.,np.zeros(30),'o',ms=7,color='white',markeredgecolor='black',markeredgewidth=0.5)
        
        for ii,intsec in enumerate(intersections):
            ax.plot([intsec,intsec],[-5,2],'red',lw=2)
            txt = ax.text(intsec+1,-6,intersection_labels[ii],fontsize=10,
                          rotation=30,color='red',fontweight='bold')
            txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                                  path_effects.Normal()])
        
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