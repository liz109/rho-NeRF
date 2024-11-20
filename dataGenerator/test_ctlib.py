import sys
import os
import time
import numpy as np
from leapctype import *

leapct = tomographicModels()

'''
This demo script shows you how to model the object being imaged as an axially-symmetric object which is essentially
an extension of the Abel Transform.  Only here we allow both the parallel- and cone-beam geometries and allow the user
to specify the axis of symmetry by a rotation angle from the positive z axis around the axis in the same direction as the detector rows.
These types of object models are commonly used in flash x-ray radiography where one is imaging a highly dynamic event
such as an explosion or implosion.  The authors are from LLNL which performs many of these experiments.

To enable this feature, one must specify exactly one projection angle, set the axis of rotation angle between -30 and 30.
Because the object is symmetric, all reconstructions are essentially 2D because numX = 1
'''


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 1024      # data["nDetector"][0]
numAngles = 50      # data["numTrain"]
pixelSize = 1.0        # data["dDetector"][0]
DSD = 1500.0
DSO = 1000.0

# Set the number of detector rows
numRows = numCols

#leapct.set_GPU(-1)

# Set the scanner geometry
#leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0))
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), DSO, DSD)

# Set the rotation angle (degrees) for the axis of symmetry
leapct.set_axisOfSymmetry(0.0)

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# Trouble-Shooting Functions
leapct.print_parameters()
#leapct.sketch_system()

# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f = leapct.allocateVolume()

# Specify simplified FORBILD head phantom
# One could easily do this in Python, but Python is soooooo slow for these types of operations,
# so we implemented this feature with multi-threaded C++
leapct.set_FORBILD(f,False)
#leapct.display(f)


# "Simulate" projection data: g
startTime = time.time()
leapct.project(g,f)
print('Forward Projection Elapsed Time: ' + str(time.time()-startTime))
#leapct.display(g)          # [nViews, col, row]

# Add noise to the data (just for demonstration purposes)
I_0 = 50000.0
#g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)

# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# Reconstruct the data
startTime = time.time()
#leapct.backproject(g,f)
#leapct.FBP(g,f)
#leapct.inconsistencyReconstruction(g,f)
#leapct.print_cost = True
filters = filterSequence(1.0e0)
filters.append(TV(leapct, delta=0.02/20.0))
#leapct.ASDPOCS(g,f,10,10,1,filters)
leapct.SART(g,f,100)
#leapct.OSEM(g,f,10,10)
#leapct.LS(g,f,50,'SQS')
#leapct.RWLS(g,f,50,filters,None,'SQS')
#leapct.RDLS(g,f,50,filters,1.0,True,1)
#leapct.MLTR(g,f,10,10,filters)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))
print(g.shape, f.shape)

# Post Reconstruction Smoothing (optional)
#startTime = time.time()
#leapct.diffuse(f,0.02/20.0,4)
#leapct.MedianFilter(f)
#leapct.BlurFilter(f,2.0)
#print('Post-Processing Elapsed Time: ' + str(time.time()-startTime))

# Display the result with napari        # Failed installed
# leapct.display(f)

