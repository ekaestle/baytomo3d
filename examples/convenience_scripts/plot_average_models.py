#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:52:06 2019

@author: emanuel
"""


from baytomo3d.AverageModels import get_avg_model

# folder with chain files
foldername = "modelsearch_bodywaves"
# plot also single chain output images
plotsinglechains = False
# plot the average models and model statistics
plot_avg_model = True
# specify the projection of the X,Y coordinates in the chain files
projection_str = None # recommended: None, i.e. read from chain file
# specify at which depth intervals to plot the 3D image
depthlevels = 'auto' #[0,12,27,42] or 'auto'

# the individual chains only store values if the chain temperature is
# equal to 1. Chains running at a high temperature will have no influence
# on the global average model

get_avg_model(foldername,depthlevels=depthlevels,
               projection_str=projection_str,
               plotsinglechains=plotsinglechains,plot_avg_model=plot_avg_model)


