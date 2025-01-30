#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:52:06 2019

@author: emanuel
"""

import glob, os, gzip, pickle
import baytomo3d.BodyWaveModule as bwm
import baytomo3d.SurfaceWaveModule as swm


# folder with chain files
foldername = "modelsearch_bodywaves"
chain_no = None


filelist = glob.glob(os.path.join(foldername,"*.pgz"))
if len(filelist)==0:
    raise Exception("No chain files in folder")
    
bw_data = bwm.read_bodywave_module(foldername)
sw_data = swm.read_surfacewave_module(foldername)
    
chainlist = []
for fname_chain in filelist:
    with gzip.open(fname_chain, "rb") as f:
        chain = pickle.load(f)
    chainlist.append(chain)
