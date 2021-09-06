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
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors as colors
import glob
import pandas as pd


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

##############################################
### Plotting the Coefficients
##############################################
### Name indexing for code clarity
Alfven = 0; Mach = 1; Ion = 2; DPar = 4; DPerp = 3; Epar = 5; Eperp = 6 ; AlphaPar = 7; AlphaPerp = 8
### Correct Scale units
ResultsArray[:,DPerp] = ResultsArray[:,DPerp]**(ResultsArray[:,AlphaPerp])
ResultsArray[:,DPar] = ResultsArray[:,DPar]**(ResultsArray[:,AlphaPar])
count1 = 0 ; count2 = 0 ; count3 = 0 ; count4 = 0
count5 = 0 ; count6 = 0 ; count7 = 0 ; count8 = 0 
# Plot Easthetic Values
size_marker = 75
alpha_val = 1.0
outline =  'black'
#outline = 'white'
chi_lab = 30
leg_lab = 34
x_limit = [0,12]
chi_x = 0.5 
chi_y_par = 1.5e-1
chi_y_perp = 1.5e-1

fig = plt.figure(figsize=(23.5,14.5), dpi = 200)
plt.rc('font', **{'size':'24'})
#plt.yscale('log')
fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)


axPerp = plt.subplot(2,3,1)
### Reset Count
count1 = 0 ; count2 = 0 ; count3 = 0 ; count4 = 0
count5 = 0 ; count6 = 0 ; count7 = 0 ; count8 = 0 

# Testing
M05 = [] ; M2 = [] ; M4 = []; M6 = []
A_M05 = [] ; A_M2 = [] ; A_M4 = [] ; A_M6 = []

Slope05 = [] ; Slope2 = []; Slope4 = []; Slope6 = []
Error05 = [] ; Error2 = []; Error4 = []; Error6 = []

for i in range(len(filenames)):
        if ResultsArray[i][Ion] == 1e-0:
            # Alfven 0.1
            if ResultsArray[i][Alfven] == 0.5:
               count1 = count1 + 1
               if (count1 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = purp_m, label = r'$\mathcal{M} = 0.5$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = purp_m,zorder=2)
               M05 = np.append(M05,ResultsArray[i][DPerp]); A_M05 = np.append(A_M05,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
               #plt.plot(ResultsArray[i][Mach], ResultsArray[i][DPerp], color = purp_m, linewidth = 1)
            # Alfven 0.5
            if ResultsArray[i][Alfven] == 2:
               count2 = count2 + 1
               if (count2 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 2$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               M2 = np.append(M2,ResultsArray[i][DPerp]); A_M2 = np.append(A_M2,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 1
            if ResultsArray[i][Alfven] == 4:
               count3 = count3 + 1
               if (count3 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = 'cornflowerblue', label = r'$\mathcal{M} = 4$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = 'cornflowerblue',zorder=2)
               M4 = np.append(M4,ResultsArray[i][DPerp]); A_M4 = np.append(A_M4,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 2
            if ResultsArray[i][Alfven] == 6.0:
               count4 = count4 + 1
               if (count4 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 6$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               M6 = np.append(M6,ResultsArray[i][DPerp]); A_M6 = np.append(A_M6,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 4
            if ResultsArray[i][Alfven] == 8.0:
               count5 = count5 + 1
               if (count5 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = 'red', label = r'$\mathcal{M} = 8$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = 'red',zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 6
            if ResultsArray[i][Alfven] == 10.0:
               count6 = count6 + 1
               if (count6 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 10$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 8
            if ResultsArray[i][Alfven] == 12.0:
               count7 = count7 + 1
               if (count7 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 12$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 10
            if ResultsArray[i][Alfven] == 14.0:
               count8 = count8 + 1
               if (count8 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 14$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr= ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            if ResultsArray[i][Alfven] == 16.0:
               count8 = count8 + 1
               if (count8 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 16$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr= ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
###########################################
### Fits
Alf = np.array([0.1 ,0.5, 1 ,2, 4, 6, 8, 10 ])
coef, cov = np.polyfit(np.log(A_M05), np.log(M05),1, cov=True) 
print(f'Mach05: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope05 = np.append(Slope05,round((coef[0]),3))
Error05 = np.append(Error05,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = purp_m)
# M2
coef = np.polyfit(np.log(A_M2), np.log(M2),1)
coef, cov = np.polyfit(np.log(A_M2), np.log(M2),1, cov=True) 
print(f'Mach2: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope2 = np.append(Slope2,round((coef[0]),3))
Error2 = np.append(Error2,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = blue_m)
#M4
coef = np.polyfit(np.log(A_M4), np.log(M4),1)
coef, cov = np.polyfit(np.log(A_M4), np.log(M4),1, cov=True) 
print(f'Mach4: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope4 = np.append(Slope4,round((coef[0]),3))
Error4 = np.append(Error4,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color ='cornflowerblue')
#M6
coef = np.polyfit(np.log(A_M6), np.log(M6),1)
coef, cov = np.polyfit(np.log(A_M6), np.log(M6),1, cov=True) 
print(f'Mach6: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope6 = np.append(Slope6,round((coef[0]),3))
Error6 = np.append(Error6,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = salmon_m)
###########################################
### Full figure aesthetics
###########################################
plt.axhline(y=1, color='r', linestyle='--')
[ylow,yhigh] = plt.gca().get_ylim()
y_limit_perp = [1e-1,yhigh*2]
plt.text(chi_x, chi_y_perp, r"$\chi = 1 $" , fontsize = chi_lab)
plt.ylabel(r'$\frac{D_{\parallel}}{D_{\perp}}$', fontsize = 38)
#plt.xlabel(r'$\mathcal{M}$', fontsize = 28)
plt.yticks(fontsize = 24)
plt.xticks([0,2,4,6,8,10], ['0','2','4','6','8','10'],rotation=0, fontsize = 0.1) 
handles,labels = plt.gca().get_legend_handles_labels()
handles = [handles[0], handles[2], handles[1], handles[3], handles[4]]
labels = [labels[0], labels[2] ,labels[1] ,labels[3],labels[4]]
plt.legend(handles,labels, frameon=False,fontsize=leg_lab, ncol = 5, bbox_to_anchor=(-0.1,1.0,1.1,0.2), loc="upper left")
plt.rc('font', **{'size':'24'})
plt.yscale('log')
#plt.xscale('log')
plt.xlim(x_limit)
plt.ylim(y_limit_perp)


##########################################################################################################################
##########################################################################################################################


axPerp1 = plt.subplot(2,3,2, sharex=axPerp, sharey=axPerp)
### Reset Count
count1 = 0 ; count2 = 0 ; count3 = 0 ; count4 = 0
count5 = 0 ; count6 = 0 ; count7 = 0 ; count8 = 0 

M05 = [] ; M2 = [] ; M4 = []; M6 = []
A_M05 = [] ; A_M2 = [] ; A_M4 = [] ; A_M6 = []

for i in range(len(filenames)):
        if ResultsArray[i][Ion] == 1e-1:
            # Alfven 0.1
            if ResultsArray[i][Alfven] == 0.5:
               count1 = count1 + 1
               if (count1 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = purp_m, label = r'$\mathcal{M} = 0.5$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = purp_m,zorder=2)
               M05 = np.append(M05,ResultsArray[i][DPerp]); A_M05 = np.append(A_M05,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 0.5
            if ResultsArray[i][Alfven] == 2:
               count2 = count2 + 1
               if (count2 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 2$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               M2 = np.append(M2,ResultsArray[i][DPerp]); A_M2 = np.append(A_M2,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 1
            if ResultsArray[i][Alfven] == 4:
               count3 = count3 + 1
               if (count3 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = 'cornflowerblue', label = r'$\mathcal{M} = 4$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = 'cornflowerblue',zorder=2)
               M4 = np.append(M4,ResultsArray[i][DPerp]); A_M4 = np.append(A_M4,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 2
            if ResultsArray[i][Alfven] == 6.0:
               count4 = count4 + 1
               if (count4 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 6$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               M6 = np.append(M6,ResultsArray[i][DPerp]); A_M6 = np.append(A_M6,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 4
            if ResultsArray[i][Alfven] == 8.0:
               count5 = count5 + 1
               if (count5 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = 'red', label = r'$\mathcal{M} = 8$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = 'red',zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 6
            if ResultsArray[i][Alfven] == 10.0:
               count6 = count6 + 1
               if (count6 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 10$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 8
            if ResultsArray[i][Alfven] == 12.0:
               count7 = count7 + 1
               if (count7 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 12$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 10
            if ResultsArray[i][Alfven] == 14.0:
               count8 = count8 + 1
               if (count8 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 14$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr= ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            if ResultsArray[i][Alfven] == 16.0:
               count8 = count8 + 1
               if (count8 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 16$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr= ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
###########################################
### Fits
Alf = np.array([0.1 ,0.5, 1 ,2, 4, 6, 8, 10 ])
coef, cov = np.polyfit(np.log(A_M05), np.log(M05),1, cov=True) 
print(f'Mach05: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope05 = np.append(Slope05,round((coef[0]),3))
Error05 = np.append(Error05,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = purp_m)
# M2
coef = np.polyfit(np.log(A_M2), np.log(M2),1)
coef, cov = np.polyfit(np.log(A_M2), np.log(M2),1, cov=True) 
print(f'Mach2: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope2 = np.append(Slope2,round((coef[0]),3))
Error2 = np.append(Error2,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = blue_m)
#M4
coef = np.polyfit(np.log(A_M4), np.log(M4),1)
coef, cov = np.polyfit(np.log(A_M4), np.log(M4),1, cov=True) 
print(f'Mach4: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope4 = np.append(Slope4,round((coef[0]),3))
Error4 = np.append(Error4,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color ='cornflowerblue')
#M6
coef = np.polyfit(np.log(A_M6), np.log(M6),1)
coef, cov = np.polyfit(np.log(A_M6), np.log(M6),1, cov=True) 
print(f'Mach6: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope6 = np.append(Slope6,round((coef[0]),3))
Error6 = np.append(Error6,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = salmon_m)
###########################################
### Full figure aesthetics
###########################################
plt.axhline(y=1, color='r', linestyle='--')
[ylow,yhigh] = plt.gca().get_ylim()
plt.text(chi_x, chi_y_perp, r"$\chi = 1 \times 10^{-1}$" , fontsize = chi_lab)
#plt.ylabel(r'$D_{\perp}$ ($\frac{cm^2}{s}$)', fontsize = 28)
#plt.xlabel(r'$\mathcal{M}$', fontsize = 28)
plt.yticks(fontsize = 0)
plt.xticks([0,2,4,6,8,10], ['0','2','4','6','8','10'],rotation=0, fontsize = 0.1) 
plt.rc('font', **{'size':'24'})
plt.yscale('log')
#plt.xscale('log')
plt.xlim(x_limit)
plt.ylim(y_limit_perp)

##########################################################################################################################
##########################################################################################################################


axPerp1 = plt.subplot(2,3,3, sharex=axPerp, sharey=axPerp)
### Reset Count
count1 = 0 ; count2 = 0 ; count3 = 0 ; count4 = 0
count5 = 0 ; count6 = 0 ; count7 = 0 ; count8 = 0 

M05 = [] ; M2 = [] ; M4 = []; M6 = []
A_M05 = [] ; A_M2 = [] ; A_M4 = [] ; A_M6 = []

for i in range(len(filenames)):
        if ResultsArray[i][Ion] == 1e-2:
            # Alfven 0.1
            if ResultsArray[i][Alfven] == 0.5:
               count1 = count1 + 1
               if (count1 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = purp_m, label = r'$\mathcal{M} = 0.5$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = purp_m,zorder=2)
               M05 = np.append(M05,ResultsArray[i][DPerp]); A_M05 = np.append(A_M05,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 0.5
            if ResultsArray[i][Alfven] == 2:
               count2 = count2 + 1
               if (count2 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 2$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               M2 = np.append(M2,ResultsArray[i][DPerp]); A_M2 = np.append(A_M2,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 1
            if ResultsArray[i][Alfven] == 4:
               count3 = count3 + 1
               if (count3 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = 'cornflowerblue', label = r'$\mathcal{M} = 4$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = 'cornflowerblue',zorder=2)
               M4 = np.append(M4,ResultsArray[i][DPerp]); A_M4 = np.append(A_M4,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 2
            if ResultsArray[i][Alfven] == 6.0:
               count4 = count4 + 1
               if (count4 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 6$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               M6 = np.append(M6,ResultsArray[i][DPerp]); A_M6 = np.append(A_M6,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 4
            if ResultsArray[i][Alfven] == 8.0:
               count5 = count5 + 1
               if (count5 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = 'red', label = r'$\mathcal{M} = 8$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = 'red',zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 6
            if ResultsArray[i][Alfven] == 10.0:
               count6 = count6 + 1
               if (count6 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 10$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 8
            if ResultsArray[i][Alfven] == 12.0:
               count7 = count7 + 1
               if (count7 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 12$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 10
            if ResultsArray[i][Alfven] == 14.0:
               count8 = count8 + 1
               if (count8 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 14$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr= ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            if ResultsArray[i][Alfven] == 16.0:
               count8 = count8 + 1
               if (count8 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 16$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr= ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
###########################################
### Fits
Alf = np.array([0.1 ,0.5, 1 ,2, 4, 6, 8, 10 ])
coef, cov = np.polyfit(np.log(A_M05), np.log(M05),1, cov=True) 
print(f'Mach05: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope05 = np.append(Slope05,round((coef[0]),3))
Error05 = np.append(Error05,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = purp_m)
# M2
coef = np.polyfit(np.log(A_M2), np.log(M2),1)
coef, cov = np.polyfit(np.log(A_M2), np.log(M2),1, cov=True) 
print(f'Mach2: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope2 = np.append(Slope2,round((coef[0]),3))
Error2 = np.append(Error2,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = blue_m)
#M4
coef = np.polyfit(np.log(A_M4), np.log(M4),1)
coef, cov = np.polyfit(np.log(A_M4), np.log(M4),1, cov=True) 
print(f'Mach4: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope4 = np.append(Slope4,round((coef[0]),3))
Error4 = np.append(Error4,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color ='cornflowerblue')
#M6
coef = np.polyfit(np.log(A_M6), np.log(M6),1)
coef, cov = np.polyfit(np.log(A_M6), np.log(M6),1, cov=True) 
print(f'Mach6: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope6 = np.append(Slope6,round((coef[0]),3))
Error6 = np.append(Error6,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = salmon_m)
###########################################
### Full figure aesthetics
###########################################
plt.axhline(y=1, color='r', linestyle='--')
[ylow,yhigh] = plt.gca().get_ylim()
plt.text(chi_x, chi_y_perp, r"$\chi = 1 \times 10^{-2}$" , fontsize = chi_lab)
#plt.ylabel(r'$D_{\perp}$ ($\frac{cm^2}{s}$)', fontsize = 28)
#plt.xlabel(r'$\mathcal{M}$', fontsize = 28)
plt.yticks(fontsize = 0)
plt.xticks([0,2,4,6,8,10], ['0','2','4','6','8','10'],rotation=0, fontsize = 0.1) 
plt.rc('font', **{'size':'24'})
plt.yscale('log')
#plt.xscale('log')
plt.xlim(x_limit)
plt.ylim(y_limit_perp)




#########################################################################################################################
#########################################################################################################################
#########################################################################################################################


ax1 = plt.subplot(2,3,4)
### Reset Count
count1 = 0 ; count2 = 0 ; count3 = 0 ; count4 = 0
count5 = 0 ; count6 = 0 ; count7 = 0 ; count8 = 0 

M05 = [] ; M2 = [] ; M4 = []; M6 = []
A_M05 = [] ; A_M2 = [] ; A_M4 = [] ; A_M6 = []

Slope05 = [] ; Slope2 = []; Slope4 = []; Slope6 = []
Error05 = [] ; Error2 = []; Error4 = []; Error6 = []

for i in range(len(filenames)):
        if ResultsArray[i][Ion] == 1e-3:
            # Alfven 0.1
            if ResultsArray[i][Alfven] == 0.5:
               count1 = count1 + 1
               if (count1 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = purp_m, label = r'$\mathcal{M} = 0.5$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = purp_m,zorder=2)
               M05 = np.append(M05,ResultsArray[i][DPerp]); A_M05 = np.append(A_M05,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 0.5
            if ResultsArray[i][Alfven] == 2:
               count2 = count2 + 1
               if (count2 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 2$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               M2 = np.append(M2,ResultsArray[i][DPerp]); A_M2 = np.append(A_M2,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 1
            if ResultsArray[i][Alfven] == 4:
               count3 = count3 + 1
               if (count3 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = 'cornflowerblue', label = r'$\mathcal{M} = 4$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = 'cornflowerblue',zorder=2)
               M4 = np.append(M4,ResultsArray[i][DPerp]); A_M4 = np.append(A_M4,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 2
            if ResultsArray[i][Alfven] == 6.0:
               count4 = count4 + 1
               if (count4 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 6$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               M6 = np.append(M6,ResultsArray[i][DPerp]); A_M6 = np.append(A_M6,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 4
            if ResultsArray[i][Alfven] == 8.0:
               count5 = count5 + 1
               if (count5 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = 'red', label = r'$\mathcal{M} = 8$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = 'red',zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 6
            if ResultsArray[i][Alfven] == 10.0:
               count6 = count6 + 1
               if (count6 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 10$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 8
            if ResultsArray[i][Alfven] == 12.0:
               count7 = count7 + 1
               if (count7 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 12$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 10
            if ResultsArray[i][Alfven] == 14.0:
               count8 = count8 + 1
               if (count8 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 14$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr= ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            if ResultsArray[i][Alfven] == 16.0:
               count8 = count8 + 1
               if (count8 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 16$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr= ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
###########################################
### Fits
Alf = np.array([0.1 ,0.5, 1 ,2, 4, 6, 8, 10 ])
coef, cov = np.polyfit(np.log(A_M05), np.log(M05),1, cov=True) 
print(f'Mach05: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope05 = np.append(Slope05,round((coef[0]),3))
Error05 = np.append(Error05,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = purp_m)
# M2
coef = np.polyfit(np.log(A_M2), np.log(M2),1)
coef, cov = np.polyfit(np.log(A_M2), np.log(M2),1, cov=True) 
print(f'Mach2: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope2 = np.append(Slope2,round((coef[0]),3))
Error2 = np.append(Error2,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = blue_m)
#M4
coef = np.polyfit(np.log(A_M4), np.log(M4),1)
coef, cov = np.polyfit(np.log(A_M4), np.log(M4),1, cov=True) 
print(f'Mach4: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope4 = np.append(Slope4,round((coef[0]),3))
Error4 = np.append(Error4,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color ='cornflowerblue')
#M6
coef = np.polyfit(np.log(A_M6), np.log(M6),1)
coef, cov = np.polyfit(np.log(A_M6), np.log(M6),1, cov=True) 
print(f'Mach6: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope6 = np.append(Slope6,round((coef[0]),3))
Error6 = np.append(Error6,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = salmon_m)
###########################################
### Full figure aesthetics
###########################################
plt.axhline(y=1, color='r', linestyle='--')
[ylow,yhigh] = plt.gca().get_ylim()
y_limit_perp = [1e-1,yhigh*2]
plt.text(chi_x, chi_y_perp, r"$\chi = 1 \times 10^{-3}$" , fontsize = chi_lab)
plt.ylabel(r'$\frac{D_{\parallel}}{D_{\perp}}$ ', fontsize = 38)
plt.xlabel(r'$\mathcal{M}_{A0}$', fontsize = 34)
plt.yticks(fontsize = 24)
plt.xticks([0,2,4,6,8,10], ['0','2','4','6','8','10'],rotation=0, fontsize = 24) 
plt.rc('font', **{'size':'24'})
plt.yscale('log')
#plt.xscale('log')
plt.xlim(x_limit)
plt.ylim(y_limit_perp)


##########################################################################################################################
##########################################################################################################################


ax2 = plt.subplot(2,3,5,sharex=ax1, sharey=ax1)
### Reset Count
count1 = 0 ; count2 = 0 ; count3 = 0 ; count4 = 0
count5 = 0 ; count6 = 0 ; count7 = 0 ; count8 = 0 

M05 = [] ; M2 = [] ; M4 = []; M6 = []
A_M05 = [] ; A_M2 = [] ; A_M4 = [] ; A_M6 = []

for i in range(len(filenames)):
        if ResultsArray[i][Ion] == 1e-4:
            # Alfven 0.1
            if ResultsArray[i][Alfven] == 0.5:
               count1 = count1 + 1
               if (count1 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = purp_m, label = r'$\mathcal{M} = 0.5$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = purp_m,zorder=2)
               M05 = np.append(M05,ResultsArray[i][DPerp]); A_M05 = np.append(A_M05,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 0.5
            if ResultsArray[i][Alfven] == 2:
               count2 = count2 + 1
               if (count2 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 2$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               M2 = np.append(M2,ResultsArray[i][DPerp]); A_M2 = np.append(A_M2,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 1
            if ResultsArray[i][Alfven] == 4:
               count3 = count3 + 1
               if (count3 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = 'cornflowerblue', label = r'$\mathcal{M} = 4$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = 'cornflowerblue',zorder=2)
               M4 = np.append(M4,ResultsArray[i][DPerp]); A_M4 = np.append(A_M4,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 2
            if ResultsArray[i][Alfven] == 6.0:
               count4 = count4 + 1
               if (count4 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 6$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               M6 = np.append(M6,ResultsArray[i][DPerp]); A_M6 = np.append(A_M6,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 4
            if ResultsArray[i][Alfven] == 8.0:
               count5 = count5 + 1
               if (count5 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = 'red', label = r'$\mathcal{M} = 8$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = 'red',zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 6
            if ResultsArray[i][Alfven] == 10.0:
               count6 = count6 + 1
               if (count6 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 10$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 8
            if ResultsArray[i][Alfven] == 12.0:
               count7 = count7 + 1
               if (count7 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 12$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 10
            if ResultsArray[i][Alfven] == 14.0:
               count8 = count8 + 1
               if (count8 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 14$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr= ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            if ResultsArray[i][Alfven] == 16.0:
               count8 = count8 + 1
               if (count8 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 16$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr= ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
###########################################
### Fits
Alf = np.array([0.1 ,0.5, 1 ,2, 4, 6, 8, 10 ])
coef, cov = np.polyfit(np.log(A_M05), np.log(M05),1, cov=True) 
print(f'Mach05: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope05 = np.append(Slope05,round((coef[0]),3))
Error05 = np.append(Error05,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = purp_m)
# M2
coef = np.polyfit(np.log(A_M2), np.log(M2),1)
coef, cov = np.polyfit(np.log(A_M2), np.log(M2),1, cov=True) 
print(f'Mach2: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope2 = np.append(Slope2,round((coef[0]),3))
Error2 = np.append(Error2,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = blue_m)
#M4
coef = np.polyfit(np.log(A_M4), np.log(M4),1)
coef, cov = np.polyfit(np.log(A_M4), np.log(M4),1, cov=True) 
print(f'Mach4: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope4 = np.append(Slope4,round((coef[0]),3))
Error4 = np.append(Error4,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color ='cornflowerblue')
#M6
coef = np.polyfit(np.log(A_M6), np.log(M6),1)
coef, cov = np.polyfit(np.log(A_M6), np.log(M6),1, cov=True) 
print(f'Mach6: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope6 = np.append(Slope6,round((coef[0]),3))
Error6 = np.append(Error6,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = salmon_m)
###########################################
### Full figure aesthetics
###########################################
plt.axhline(y=1, color='r', linestyle='--')
[ylow,yhigh] = plt.gca().get_ylim()
plt.text(chi_x, chi_y_perp, r"$\chi = 1 \times 10^{-4}$" , fontsize = chi_lab)
#plt.ylabel(r'$D_{\perp}$ ($\frac{cm^2}{s}$)', fontsize = 28)
plt.xlabel(r'$\mathcal{M}_{A0}$', fontsize = 34)
plt.yticks(fontsize = 0)
plt.xticks([0,2,4,6,8,10], ['0','2','4','6','8','10'],rotation=0, fontsize = 24) 
plt.rc('font', **{'size':'24'})
plt.yscale('log')
#plt.xscale('log')
plt.xlim(x_limit)
plt.ylim(y_limit_perp)

##########################################################################################################################
##########################################################################################################################


ax3 = plt.subplot(2,3,6,sharex=ax1, sharey=ax1)
### Reset Count
count1 = 0 ; count2 = 0 ; count3 = 0 ; count4 = 0
count5 = 0 ; count6 = 0 ; count7 = 0 ; count8 = 0 

M05 = [] ; M2 = [] ; M4 = []; M6 = []
A_M05 = [] ; A_M2 = [] ; A_M4 = [] ; A_M6 = []

for i in range(len(filenames)):
        if ResultsArray[i][Ion] == 1e-5:
            # Alfven 0.1
            if ResultsArray[i][Alfven] == 0.5:
               count1 = count1 + 1
               if (count1 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = purp_m, label = r'$\mathcal{M} = 0.5$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = purp_m,zorder=2)
               M05 = np.append(M05,ResultsArray[i][DPerp]); A_M05 = np.append(A_M05,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 0.5
            if ResultsArray[i][Alfven] == 2:
               count2 = count2 + 1
               if (count2 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 2$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               M2 = np.append(M2,ResultsArray[i][DPerp]); A_M2 = np.append(A_M2,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 1
            if ResultsArray[i][Alfven] == 4:
               count3 = count3 + 1
               if (count3 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = 'cornflowerblue', label = r'$\mathcal{M} = 4$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = 'cornflowerblue',zorder=2)
               M4 = np.append(M4,ResultsArray[i][DPerp]); A_M4 = np.append(A_M4,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 2
            if ResultsArray[i][Alfven] == 6.0:
               count4 = count4 + 1
               if (count4 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 6$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               M6 = np.append(M6,ResultsArray[i][DPerp]); A_M6 = np.append(A_M6,ResultsArray[i][Mach])
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 4
            if ResultsArray[i][Alfven] == 8.0:
               count5 = count5 + 1
               if (count5 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = 'red', label = r'$\mathcal{M} = 8$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp]/ResultsArray[i][DPar], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = 'red',zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 6
            if ResultsArray[i][Alfven] == 10.0:
               count6 = count6 + 1
               if (count6 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 10$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 8
            if ResultsArray[i][Alfven] == 12.0:
               count7 = count7 + 1
               if (count7 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = blue_m, label = r'$\mathcal{M} = 12$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = blue_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr=ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            # Alfven 10
            if ResultsArray[i][Alfven] == 14.0:
               count8 = count8 + 1
               if (count8 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 14$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr= ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
            if ResultsArray[i][Alfven] == 16.0:
               count8 = count8 + 1
               if (count8 < 2):
                   plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
                   edgecolor=outline, facecolor = salmon_m, label = r'$\mathcal{M} = 16$',zorder=2)
               else:
                    plt.scatter(ResultsArray[i][Mach],ResultsArray[i][DPerp], s = size_marker, alpha=alpha_val, 
               edgecolor=outline, facecolor = salmon_m,zorder=2)
               #plt.errorbar(ResultsArray[i][Mach], ResultsArray[i][DPerp], yerr= ResultsArray[i][Eperp],fmt='|', color = outline,zorder=1, capsize=10)
###########################################
### Fits
Alf = np.array([0.1 ,0.5, 1 ,2, 4, 6, 8, 10 ])
coef, cov = np.polyfit(np.log(A_M05), np.log(M05),1, cov=True) 
print(f'Mach05: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope05 = np.append(Slope05,round((coef[0]),3))
Error05 = np.append(Error05,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = purp_m)
# M2
coef = np.polyfit(np.log(A_M2), np.log(M2),1)
coef, cov = np.polyfit(np.log(A_M2), np.log(M2),1, cov=True) 
print(f'Mach2: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope2 = np.append(Slope2,round((coef[0]),3))
Error2 = np.append(Error2,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = blue_m)
#M4
coef = np.polyfit(np.log(A_M4), np.log(M4),1)
coef, cov = np.polyfit(np.log(A_M4), np.log(M4),1, cov=True) 
print(f'Mach4: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope4 = np.append(Slope4,round((coef[0]),3))
Error4 = np.append(Error4,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color ='cornflowerblue')
#M6
coef = np.polyfit(np.log(A_M6), np.log(M6),1)
coef, cov = np.polyfit(np.log(A_M6), np.log(M6),1, cov=True) 
print(f'Mach6: {round(np.exp(coef[0]),2)}')
E = (np.sqrt(np.diag(cov)))
Slope6 = np.append(Slope6,round((coef[0]),3))
Error6 = np.append(Error6,round((E[0]),3))
#plt.plot(Alf, np.exp(coef[0] * np.log(Alf) + coef[1]), '-', color = salmon_m)
###########################################
### Full figure aesthetics
###########################################
plt.axhline(y=1, color='r', linestyle='--')
[ylow,yhigh] = plt.gca().get_ylim()
plt.text(chi_x, chi_y_perp, r"$\chi = 1 \times 10^{-5}$" , fontsize = chi_lab)
#plt.ylabel(r'$D_{\perp}$ ($\frac{cm^2}{s}$)', fontsize = 28)
plt.xlabel(r'$\mathcal{M}_{A0}$', fontsize = 34)
plt.yticks(fontsize = 0)
plt.xticks([0,2,4,6,8,10], ['0','2','4','6','8','10'],rotation=0, fontsize = 24) 
plt.rc('font', **{'size':'24'})
plt.yscale('log')
#plt.xscale('log')
plt.xlim(x_limit)
plt.ylim(y_limit_perp)



#########################################
#########################################
#########################################
plt.savefig('Ratio.png')


