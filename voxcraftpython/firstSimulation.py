import numpy as np

from VoxcraftVXA import VXA
from VoxcraftVXD import VXD

#TODO Use this to generate VXD files for patterns during simulations!
#THE VXD FILES GENERATED FROM HERE ARE USED IN voxcraft-sim

# Generate a Base VXA file
# See here for list of vxa tags: https://gpuvoxels.readthedocs.io/en/docs/
vxa = VXA(EnableExpansion=1, SimTime=5) # pass vxa tags in here

# Create two materials with different properties

mat1 = vxa.add_material(RGBA=(255,0,255), E=5e4, RHO=1e4) # returns the material ID #Solid
mat2 = vxa.add_material(RGBA=(255,0,0), E=1e8, RHO=1e4) #Flexible

# Write out the vxa to data/ directory
vxa.write("base.vxa")

# Create random body array between 0 and maximum material ID

#TODO THIS IS WHERE THE NUMPY ARRAY OF VALUES IS PLACED
body = np.random.randint(0,mat2+1,size=(8,8,7))

# Generate a VXD file
vxd = VXD()
vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
vxd.set_data(body)
# Write out the vxd to data/
vxd.write("robot1.vxd")