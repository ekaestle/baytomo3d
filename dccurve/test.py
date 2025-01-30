import dccurve
import numpy as np

dccurve.init_dccurve(1)

periods = np.array([30.,20,10,5,4,3.,2.,1.]) # numpy array, should be from long to short periods, otherwise you will get a reversed dispersion curve
thickmod = np.ones(5)*5.
vpmod = np.array([3.,4.,5.,6.,7.]) # in km/s
vsmod = vpmod/1.7 # in km/s
densmod = np.linspace(2.2,4.,5) # in g/cm³
nmodes = 1 # only fundamental mode
group = 0 # for group otherwise 0 for phase
ray = 1 # rayleigh dispersion, 0 for love dispersion
# this array will contain the output phase/group slowness values:
slowness = np.zeros(len(periods)) # maybe also np.zeros((nmodes,nperiods)), if you have only one the fundamental mode np.zeros(nperiods) is enough

dccurve.get_disp_curve(nmodes,group,thickmod,vpmod,vsmod,densmod,periods,slowness,ray)

print(np.column_stack((periods,1./slowness)))

#(note that nlayers and nsamples appear in the fortran script but are determined automatically, so they should not be given as input in the function)

