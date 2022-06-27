from voxcraftpython.VoxcraftVXA import VXA
from voxcraftpython.VoxcraftVXD import VXD

vxa = VXA(EnableExpansion=1, SimTime=5)

mat1 = vxa.add_material(RGBA=(255,0,255), E=5e4, RHO=1e4) #Solid
mat2 = vxa.add_material(RGBA=(255,0,0), E=1e8, RHO=1e4) #Flexible



#body = 

vxd = VXD()
