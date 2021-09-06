'''
Plotting Code for Diffusion Coefficients
Matt Sampson
2021
'''

############################################
#### Package imports
############################################
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors as colors
import glob
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
from scipy.interpolate import interp2d
Alfven = 1; Mach = 0; Ion = 2; DPar = 4; DPerp = 3; Epar = 5; Eperp = 6; AlphaPar = 8; AlphaPerp = 7
### Correct Scale units



############################################
#### Astro Plot Aesthetics Pre-Amble
############################################
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['ytick.minor.size'] = 5
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
#Direct input 
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params =   {'text.usetex' : True,
            'font.size' : 11,
            'font.family' : 'lmodern'
            }
plt.rcParams.update(params) 

### Setting new default colour cycles
# Set the default color cycle custom colours (Synthwave inspired)
green_m = (110/256, 235/256, 52/256)
purp_m = (210/356, 113/256, 235/256)
blue_m = (50/256, 138/256, 129/256)
dblue_m = (0.1, 0.2, 0.5)
salmon_m = (240/256, 102/256, 43/256)
dgreen_m = (64/256, 128/256, 4/256)
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[dblue_m, green_m, purp_m,  dgreen_m, blue_m]) 
# Dark background
#plt.style.use('dark_background')
##############################################

##############################################
### Data Reading
##############################################
# All files ending with .txt
filenames = glob.glob("*.txt")
ResultsArray = np.loadtxt(filenames[0], dtype=float)
for i in range(len(filenames) - 1):
    DataNew = np.loadtxt(filenames[i+1], dtype=float)
    ResultsArray = np.vstack((ResultsArray, DataNew))

ResultsArray[:,DPerp] = ResultsArray[:,DPerp]**(ResultsArray[:,AlphaPerp])
ResultsArray[:,DPar] = ResultsArray[:,DPar]**(ResultsArray[:,AlphaPar])

################################################
### The Plotting
################################################
### Loop over chi values to make vectors
chi_0 = ResultsArray[ResultsArray[:,Ion] == 1e-0]
chi_1 = ResultsArray[ResultsArray[:,Ion] == 1e-1]
chi_2 = ResultsArray[ResultsArray[:,Ion] == 1e-2]
chi_3 = ResultsArray[ResultsArray[:,Ion] == 1e-3]
chi_4 = ResultsArray[ResultsArray[:,Ion] == 1e-4]
chi_5 = ResultsArray[ResultsArray[:,Ion] == 1e-5]
# Mach
Mach_chi_0 = np.log(chi_0[:,Mach])
Mach_chi_1 = np.log(chi_1[:,Mach])
Mach_chi_2 = np.log(chi_2[:,Mach])
Mach_chi_3 = np.log(chi_3[:,Mach])
Mach_chi_4 = np.log(chi_4[:,Mach])
Mach_chi_5 = np.log(chi_5[:,Mach])
# Alfven 
Alf_chi_0 = np.log(chi_0[:,Alfven])
Alf_chi_1 = np.log(chi_1[:,Alfven])
Alf_chi_2 = np.log(chi_2[:,Alfven])
Alf_chi_3 = np.log(chi_3[:,Alfven])
Alf_chi_4 = np.log(chi_4[:,Alfven])
Alf_chi_5 = np.log(chi_5[:,Alfven])
# Perp
Perp_chi_0 = chi_0[:,DPerp]
Perp_chi_1 = chi_1[:,DPerp]
Perp_chi_2 = chi_2[:,DPerp]
Perp_chi_3 = chi_3[:,DPerp]
Perp_chi_4 = chi_4[:,DPerp]
Perp_chi_5 = chi_5[:,DPerp]
# Par
Par_chi_0 = chi_0[:,DPar]
Par_chi_1 = chi_1[:,DPar]
Par_chi_2 = chi_2[:,DPar]
Par_chi_3 = chi_3[:,DPar]
Par_chi_4 = chi_4[:,DPar]
Par_chi_5 = chi_5[:,DPar]


fig = plt.figure(figsize=(15,10), dpi = 200)
col_map = cmr.ember
#col_map = cmr.toxic
#col_map = cmr.tree
size_f = 20
min_perp = min(0.1 * Par_chi_5)
max_perp = max(Perp_chi_5)
min_par = min(0.1 * Par_chi_5)
max_par = max(Par_chi_5)
contours = 250
plt.subplot(2,3,1)
#########################################################
y_list=Mach_chi_0 ; x_list=Alf_chi_0 ; z_list=Perp_chi_0
f = interp2d(x_list,y_list,z_list,kind="linear")
x_coords = np.arange(min(x_list),max(x_list)+1)
y_coords = np.arange(min(y_list),max(y_list)+1)
Z = f(x_coords,y_coords)
#########################################################
im = plt.tricontourf(x_list,y_list,z_list, contours, cmap=col_map, vmin=min_par, vmax=max_par)
plt.ylabel(r'log($\mathcal{M}$)', fontsize = size_f)
plt.title(r'$\chi = 1$', fontsize = size_f)
#plt.xlabel(r'$\mathcal{M}_{A0}$', fontsize = size_f)
#########################

plt.subplot(2,3,4)
#########################################################
y_list=Mach_chi_3 ; x_list=Alf_chi_3 ; z_list=Perp_chi_3
f = interp2d(x_list,y_list,z_list,kind="linear")
x_coords = np.arange(min(x_list),max(x_list)+1)
y_coords = np.arange(min(y_list),max(y_list)+1)
Z = f(x_coords,y_coords)
#########################################################
im = plt.tricontourf(x_list,y_list,z_list, contours, cmap=col_map, vmin=min_par, vmax=max_par)
plt.ylabel(r'log($\mathcal{M}$)', fontsize = size_f)
plt.title(r'$\chi = 1 \times 10^{-3}$', fontsize = size_f)
plt.xlabel(r'log($\mathcal{M}_{A0}$)', fontsize = size_f)
#########################

plt.subplot(2,3,2)
#########################################################
y_list=Mach_chi_1 ; x_list=Alf_chi_1 ; z_list=Perp_chi_1
f = interp2d(x_list,y_list,z_list,kind="linear")
x_coords = np.arange(min(x_list),max(x_list)+1)
y_coords = np.arange(min(y_list),max(y_list)+1)
Z = f(x_coords,y_coords)
#########################################################
im = plt.tricontourf(x_list,y_list,z_list, contours, cmap=col_map, vmin=min_par, vmax=max_par)
#plt.ylabel(r'$\mathcal{M}$', fontsize = size_f)
plt.title(r'$\chi = 1 \times 10^{-1}$', fontsize = size_f)
#plt.xlabel(r'$\mathcal{M}_{A0}$', fontsize = size_f)
#########################

plt.subplot(2,3,5)
#########################################################
y_list=Mach_chi_4 ; x_list=Alf_chi_4 ; z_list=Perp_chi_4
f = interp2d(x_list,y_list,z_list,kind="linear")
x_coords = np.arange(min(x_list),max(x_list)+1)
y_coords = np.arange(min(y_list),max(y_list)+1)
Z = f(x_coords,y_coords)
#########################################################
im = plt.tricontourf(x_list,y_list,z_list, contours, cmap=col_map, vmin=min_par, vmax=max_par)
#plt.ylabel(r'$\mathcal{M}$', fontsize = size_f)
plt.title(r'$\chi = 1 \times 10^{-4}$', fontsize = size_f)
plt.xlabel(r'log($\mathcal{M}_{A0}$)', fontsize = size_f)
#########################

plt.subplot(2,3,3)
#########################################################
y_list=Mach_chi_2 ; x_list=Alf_chi_2 ; z_list=Perp_chi_2
f = interp2d(x_list,y_list,z_list,kind="linear")
x_coords = np.arange(min(x_list),max(x_list)+1)
y_coords = np.arange(min(y_list),max(y_list)+1)
Z = f(x_coords,y_coords)
#########################################################
im = plt.tricontourf(x_list,y_list,z_list, contours, cmap=col_map, vmin=min_par, vmax=max_par)
#plt.ylabel(r'$\mathcal{M}$', fontsize = size_f)
plt.title(r'$\chi = 1 \times 10^{-2}$', fontsize = size_f)
#plt.xlabel(r'$\mathcal{M}_{A0}$', fontsize = size_f)
#plt.colorbar()
#########################

plt.subplot(2,3,6)
#########################################################
y_list=Mach_chi_5 ; x_list=Alf_chi_5 ; z_list=Perp_chi_5
f = interp2d(x_list,y_list,z_list,kind="linear")
x_coords = np.arange(min(x_list),max(x_list)+1)
y_coords = np.arange(min(y_list),max(y_list)+1)
Z = f(x_coords,y_coords)

#########################################################
#im = plt.imshow(Z,extent=[min(x_list),max(x_list),min(y_list),max(y_list)],
#           origin="lower",cmap=col_map, norm=LogNorm(),aspect = 'auto', vmin=min_par, vmax=max_par)
im = plt.tricontourf(x_list,y_list,z_list, contours, cmap=col_map, vmin=min_par, vmax=max_par)
#plt.ylabel(r'$\mathcal{M}$', fontsize = size_f)
plt.title(r'$\chi = 1 \times 10^{-5}$', fontsize = size_f)
plt.xlabel(r'log($\mathcal{M}_{A0}$)', fontsize = size_f)
#plt.colorbar()
#########################
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.1, 0.04, 0.78])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=22)
cbar.set_label(label = r'$D_{\parallel}$ ($c^{\alpha} \cdot L \cdot c_s \cdot \mathcal{M}$)',size=28)

#plt.suptitle("Perpendicular Diffusion", fontsize = 30)
plt.savefig('3d_Par_toxic.png')
plt.close()



