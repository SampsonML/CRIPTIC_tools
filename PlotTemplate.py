'''
Plotting template code for CRIPTIC and MHD simulation data
Matt Sampson
Sept 2021
'''

############################################
#### Package imports
############################################
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator,LogLocator)
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

##################################
### Data reading
##################################
Sim_Data = pd.read_csv('MHD_sim_params_Criptic_params.csv')
Sim_Data = Sim_Data[Sim_Data['mach_rms_50'] > 0.9]
Keys = Sim_Data.keys()
#for i in range(len(Keys)):
#    print(f'Variable: {Keys[i]}')

##################################
### Kappa
# Perpendicular
# chi 1e-5
perp_5 =  Sim_Data['scale_x_perp_1e-05']**Sim_Data['alpha_x_perp_1e-05'] #/ Sim_Data['scale_y_perp_1e-05']**Sim_Data['alpha_y_perp_1e-05']
perp_4 =  Sim_Data['scale_x_perp_0.0001']**Sim_Data['alpha_x_perp_0.0001']# / Sim_Data['scale_y_perp_0.0001']**Sim_Data['alpha_y_perp_0.0001']
perp_3 =  Sim_Data['scale_x_perp_0.001']**Sim_Data['alpha_x_perp_0.001'] #/ Sim_Data['scale_y_perp_0.001']**Sim_Data['alpha_y_perp_0.001']
perp_2 =  Sim_Data['scale_x_perp_0.01']**Sim_Data['alpha_x_perp_0.01']# / Sim_Data['scale_y_perp_0.01']**Sim_Data['alpha_y_perp_0.01']
perp_1 =  Sim_Data['scale_x_perp_0.1']**Sim_Data['alpha_x_perp_0.1'] #/ Sim_Data['scale_y_perp_0.1']**Sim_Data['alpha_y_perp_0.1']
perp_0 =  Sim_Data['scale_x_perp_1']**Sim_Data['alpha_x_perp_1'] #/ Sim_Data['scale_y_perp_1']**Sim_Data['alpha_y_perp_1']
### For y
perp_y_5 =  Sim_Data['scale_y_perp_1e-05']**Sim_Data['alpha_y_perp_1e-05']
perp_y_4 =   Sim_Data['scale_y_perp_0.0001']**Sim_Data['alpha_y_perp_0.0001']
perp_y_3 =   Sim_Data['scale_y_perp_0.001']**Sim_Data['alpha_y_perp_0.001']
perp_y_2 =   Sim_Data['scale_y_perp_0.01']**Sim_Data['alpha_y_perp_0.01']
perp_y_1 =   Sim_Data['scale_y_perp_0.1']**Sim_Data['alpha_y_perp_0.1']
perp_y_0 =   Sim_Data['scale_y_perp_1']**Sim_Data['alpha_y_perp_1']
# chi 1e-5
par_5 = Sim_Data['scale_par_1e-05']**Sim_Data['alpha_par_1e-05']
# chi 1e-4
par_4 = Sim_Data['scale_par_0.0001']**Sim_Data['alpha_par_0.0001']
# chi 1e-3
par_3 = Sim_Data['scale_par_0.001']**Sim_Data['alpha_par_0.001']
# chi 1e-2
par_2 = Sim_Data['scale_par_0.01']**Sim_Data['alpha_par_0.01']
# chi 1e-1
par_1 = Sim_Data['scale_par_0.1']**Sim_Data['alpha_par_0.1']
# chi 1
par_0 = Sim_Data['scale_par_1']**Sim_Data['alpha_par_1']

##################################
### MHD Params
B_cor = Sim_Data['B_cor_scale_50']
B_cor_error = Sim_Data['B_cor_scale_16'] + Sim_Data['B_cor_scale_84'] - 2*B_cor
Mach = Sim_Data['mach_rms_50']
Mach_Error_L = Mach - Sim_Data['mach_rms_16']
Mach_Error_H = Sim_Data['mach_rms_84'] - Mach
MA = Sim_Data['mach_alfven_rms_50']
MA_0 = Sim_Data['mach_alfven_0_rms_50']
MA_0_Error_L = MA_0 - Sim_Data['mach_alfven_0_rms_16']
MA_0_Error_H = Sim_Data['mach_alfven_0_rms_84'] - MA_0
MA_Error_L =  MA - Sim_Data['mach_alfven_rms_16'] 
MA_Error_H = Sim_Data['mach_alfven_rms_84'] - MA
mag_perp = Sim_Data['mag_perp_rms_50']

####################################
def diffusion_uncertainty(alpha, scale):
    """
    Compute the uncertainty assuming Gaussian but correlated
    covariates for a variable,

    kappa = scale ^ alpha,

    from 16th and 84th percentiles.

    """

    alpha_50, alpha_16, alpha_84 = alpha
    scale_50, scale_16, scale_84 = scale
    kappa = scale_50 ** alpha_50
    kappa_16 = np.sqrt( (alpha_50*scale_50**(alpha_50-1)*scale_16)**2 +
                        (scale_50**alpha_50*np.log(scale_50)*alpha_16)**2 +
                        2*alpha_50*scale_50**(alpha_50-1)*scale_16*scale_50**alpha_50*np.log(scale_50)*alpha_16 )
    kappa_84 = np.sqrt( (alpha_50*scale_50**(alpha_50-1)*scale_84)**2 +
                        (scale_50**alpha_50*np.log(scale_50)*alpha_84)**2 +
                        2*alpha_50*scale_50**(alpha_50-1)*scale_84*scale_50**alpha_50*np.log(scale_50)*alpha_84 )
    err = [kappa_16,kappa_84]

    return err

########################################
### Error Calculations 
########################################
kappa_perp_x_err_0    = diffusion_uncertainty([Sim_Data["alpha_x_perp_1"],Sim_Data["alpha_x_error_perp_16_1"],Sim_Data["alpha_x_error_perp_84_1"]],
                                                [Sim_Data["scale_x_perp_1"],Sim_Data["scale_x_error_perp_16_1"],Sim_Data["scale_x_error_perp_84_1"]])
kappa_perp_y_err_0    = diffusion_uncertainty([Sim_Data["alpha_y_perp_1"],Sim_Data["alpha_y_error_perp_16_1"],Sim_Data["alpha_y_error_perp_84_1"]],
                                                [Sim_Data["scale_y_perp_1"],Sim_Data["scale_y_error_perp_16_1"],Sim_Data["scale_y_error_perp_84_1"]])
kappa_par_err_0       = diffusion_uncertainty([Sim_Data["alpha_par_1"],Sim_Data["alpha_error_par_16_1"],Sim_Data["alpha_error_par_84_1"]],
                                                [Sim_Data["scale_par_1"],Sim_Data["scale_error_par_16_1"],Sim_Data["scale_error_par_84_1"]])


kappa_perp_x_err_1    = diffusion_uncertainty([Sim_Data["alpha_x_perp_0.1"],Sim_Data["alpha_x_error_perp_16_0.1"],Sim_Data["alpha_x_error_perp_84_0.1"]],
                                                [Sim_Data["scale_x_perp_0.1"],Sim_Data["scale_x_error_perp_16_0.1"],Sim_Data["scale_x_error_perp_84_0.1"]])
kappa_perp_y_err_1    = diffusion_uncertainty([Sim_Data["alpha_y_perp_0.1"],Sim_Data["alpha_y_error_perp_16_0.1"],Sim_Data["alpha_y_error_perp_84_0.1"]],
                                                [Sim_Data["scale_y_perp_0.1"],Sim_Data["scale_y_error_perp_16_0.1"],Sim_Data["scale_y_error_perp_84_0.1"]])
kappa_par_err_1       = diffusion_uncertainty([Sim_Data["alpha_par_0.1"],Sim_Data["alpha_error_par_16_0.1"],Sim_Data["alpha_error_par_84_0.1"]],
                                                [Sim_Data["scale_par_0.1"],Sim_Data["scale_error_par_16_0.1"],Sim_Data["scale_error_par_84_0.1"]])

kappa_perp_x_err_2    = diffusion_uncertainty([Sim_Data["alpha_x_perp_0.01"],Sim_Data["alpha_x_error_perp_16_0.01"],Sim_Data["alpha_x_error_perp_84_0.01"]],
                                                [Sim_Data["scale_x_perp_0.01"],Sim_Data["scale_x_error_perp_16_0.01"],Sim_Data["scale_x_error_perp_84_0.01"]])
kappa_perp_y_err_2    = diffusion_uncertainty([Sim_Data["alpha_y_perp_0.01"],Sim_Data["alpha_y_error_perp_16_0.01"],Sim_Data["alpha_y_error_perp_84_0.01"]],
                                                [Sim_Data["scale_y_perp_0.01"],Sim_Data["scale_y_error_perp_16_0.01"],Sim_Data["scale_y_error_perp_84_0.01"]])
kappa_par_err_2       = diffusion_uncertainty([Sim_Data["alpha_par_0.01"],Sim_Data["alpha_error_par_16_0.01"],Sim_Data["alpha_error_par_84_0.01"]],
                                                [Sim_Data["scale_par_0.01"],Sim_Data["scale_error_par_16_0.01"],Sim_Data["scale_error_par_84_0.01"]])

kappa_perp_x_err_3    = diffusion_uncertainty([Sim_Data["alpha_x_perp_0.001"],Sim_Data["alpha_x_error_perp_16_0.001"],Sim_Data["alpha_x_error_perp_84_0.001"]],
                                                [Sim_Data["scale_x_perp_0.001"],Sim_Data["scale_x_error_perp_16_0.001"],Sim_Data["scale_x_error_perp_84_0.001"]])
kappa_perp_y_err_3    = diffusion_uncertainty([Sim_Data["alpha_y_perp_0.001"],Sim_Data["alpha_y_error_perp_16_0.001"],Sim_Data["alpha_y_error_perp_84_0.001"]],
                                                [Sim_Data["scale_y_perp_0.001"],Sim_Data["scale_y_error_perp_16_0.001"],Sim_Data["scale_y_error_perp_84_0.001"]])
kappa_par_err_3       = diffusion_uncertainty([Sim_Data["alpha_par_0.001"],Sim_Data["alpha_error_par_16_0.001"],Sim_Data["alpha_error_par_84_0.001"]],
                                                [Sim_Data["scale_par_0.001"],Sim_Data["scale_error_par_16_0.001"],Sim_Data["scale_error_par_84_0.001"]])

kappa_perp_x_err_4    = diffusion_uncertainty([Sim_Data["alpha_x_perp_0.0001"],Sim_Data["alpha_x_error_perp_16_0.0001"],Sim_Data["alpha_x_error_perp_84_0.0001"]],
                                                [Sim_Data["scale_x_perp_0.0001"],Sim_Data["scale_x_error_perp_16_0.0001"],Sim_Data["scale_x_error_perp_84_0.0001"]])
kappa_perp_y_err_4    = diffusion_uncertainty([Sim_Data["alpha_y_perp_0.0001"],Sim_Data["alpha_y_error_perp_16_0.0001"],Sim_Data["alpha_y_error_perp_84_0.0001"]],
                                                [Sim_Data["scale_y_perp_0.0001"],Sim_Data["scale_y_error_perp_16_0.0001"],Sim_Data["scale_y_error_perp_84_0.0001"]])
kappa_par_err_4       = diffusion_uncertainty([Sim_Data["alpha_par_0.0001"],Sim_Data["alpha_error_par_16_0.0001"],Sim_Data["alpha_error_par_84_0.0001"]],
                                                [Sim_Data["scale_par_0.0001"],Sim_Data["scale_error_par_16_0.0001"],Sim_Data["scale_error_par_84_0.0001"]])

kappa_perp_x_err_5    = diffusion_uncertainty([Sim_Data["alpha_x_perp_1e-05"],Sim_Data["alpha_x_error_perp_16_1e-05"],Sim_Data["alpha_x_error_perp_84_1e-05"]],
                                                [Sim_Data["scale_x_perp_1e-05"],Sim_Data["scale_x_error_perp_16_1e-05"],Sim_Data["scale_x_error_perp_84_1e-05"]])
kappa_perp_y_err_5    = diffusion_uncertainty([Sim_Data["alpha_y_perp_1e-05"],Sim_Data["alpha_y_error_perp_16_1e-05"],Sim_Data["alpha_y_error_perp_84_1e-05"]],
                                                [Sim_Data["scale_y_perp_1e-05"],Sim_Data["scale_y_error_perp_16_1e-05"],Sim_Data["scale_y_error_perp_84_1e-05"]])
kappa_par_err_5       = diffusion_uncertainty([Sim_Data["alpha_par_1e-05"],Sim_Data["alpha_error_par_16_1e-05"],Sim_Data["alpha_error_par_84_1e-05"]],
                                                [Sim_Data["scale_par_1e-05"],Sim_Data["scale_error_par_16_1e-05"],Sim_Data["scale_error_par_84_1e-05"]])
#####################################
### 6 Panel Plot
### Make arrarys of y data
kappa_perp_array = [perp_0, perp_1, perp_2, perp_3,perp_4,perp_5]
kappa_perp_array_y = [perp_y_0, perp_y_1, perp_y_2, perp_y_3,perp_y_4,perp_y_5]
kappa_par_array = [par_0, par_1, par_2, par_3,par_4,par_5]
kappa_ratio = [par_0 / perp_0, par_1 / perp_1, par_2 / perp_2,
par_3 / perp_3,par_4 / perp_4,par_5 / perp_5]
kappa_ratio_y = [par_0 / perp_y_0, par_1 / perp_y_1, par_2 / perp_y_2,
par_3 / perp_y_3,par_4 / perp_y_4,par_5 / perp_y_5]
kappa_perp_x_error = [kappa_perp_x_err_0,kappa_perp_x_err_1,kappa_perp_x_err_2,kappa_perp_x_err_3,
kappa_perp_x_err_4,kappa_perp_x_err_5]
kappa_perp_y_error = [kappa_perp_y_err_0,kappa_perp_y_err_1,kappa_perp_y_err_2,kappa_perp_y_err_3,
kappa_perp_y_err_4,kappa_perp_y_err_5]
kappa_par_error = [kappa_par_err_0,kappa_par_err_1,kappa_par_err_2,
kappa_par_err_3,kappa_par_err_4,kappa_par_err_5]

#####################################
### Plot 1 Ratio Plot
#####################################
#################
# Plot aesthetics
marker_size = 55
cm = cmr.fall
fig = plt.figure(figsize=(12.0,7.5), dpi = 200)
plt.rc('font', **{'size':'24'})
plt.subplots_adjust(wspace=0, hspace=0)
### Name data
data_var = kappa_ratio
for i in range(6):
    if i == 0:
        chi_lab = r'$\chi = 1$'
    if i > 0:
        chi_lab = r'$\chi = 1 \times 10^{-' + str(i) + r'}$'
    col_name = data_var[i]
    if i == 0:
        ax = plt.subplot(2,3,i+1)
    if i > 0:
        plt.subplot(2,3,i+1, sharex=ax, sharey=ax)
    ### Errorbars
    plt.errorbar(MA, col_name,
             xerr=[MA_Error_L,MA_Error_H],
             yerr=kappa_perp_x_error[i],
             fmt='none',
             capsize = 2,
             c = 'black',
             zorder = 0) 
    plt.errorbar(MA, kappa_ratio_y[i],
             xerr=[MA_Error_L, MA_Error_H],
             yerr=kappa_perp_y_error[i],
             fmt='none',
             capsize = 2,
             c = 'black',
             zorder = 0) 
    ### Fit lines
    plt.ylim([0.1,5e3])
    coef, cov = np.polyfit(np.log(MA[5:18]), np.log(col_name[5:18]),1, cov=True)
    E = (np.sqrt(np.diag(cov)))
    Label_model = r'$\propto  \mathcal{M}_A^{' + str(round(coef[0],2)) + r' \pm' + str(round(E[0],2)) + r'} $'
    plt.plot(MA[5:18], np.exp(coef[0] * np.log(MA[5:18]) + coef[1]), '--', color = 'black', lw = 2, label = Label_model,zorder = 0)
    ### Chi label
    plt.axhline(y=1, color='r', linestyle='--',zorder = 0)
    plt.text(0.12, 0.26 , chi_lab, size=15, color='black')
    ###
    sc = plt.scatter(MA, col_name, c=Mach,norm=colors.LogNorm(), s=marker_size, cmap=cm, edgecolor = 'black',zorder = 2)
    sc = plt.scatter(MA, kappa_ratio_y[i], c=Mach, s=marker_size, cmap=cm, edgecolor = 'black',zorder = 2)
    # Ticks turn off
    plt.yticks(fontsize = 0)
    plt.xticks(fontsize = 0)
    if i > 2:
        plt.xlabel(r'$\mathcal{M}_{A}$',fontsize = 24)
        plt.xticks(fontsize = 16)    
    if i == 0 or i == 3:
        plt.yticks(fontsize = 16)
        plt.ylabel(r'$\frac{\kappa_{\parallel}}{\kappa_{\perp}}$',fontsize = 34)
    plt.legend(frameon = False, fontsize = 16.5, loc = 'upper right')
    plt.yscale('log')
    plt.xscale('log')
    ax.yaxis.set_minor_locator(LogLocator())

### Set colorbar
from matplotlib import ticker
cbar_ax = fig.add_axes([0.9, 0.11, 0.03, 0.77])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.ax.tick_params(labelsize=22)
cbar.set_label(r'$\mathcal{M}$', rotation=270, fontsize = 24)
cbar.ax.get_yaxis().labelpad = 25
plt.savefig('Ratio_Powerlaw.pdf')


#####################################
### Plot 2 Perp Diffusion
#####################################
#################
# Plot aesthetics
marker_size = 55
cm = cmr.fall
fig = plt.figure(figsize=(12.5,7.5), dpi = 200)
plt.rc('font', **{'size':'24'})
plt.subplots_adjust(wspace=0, hspace=0)
### Name data
data_var = kappa_perp_array
for i in range(6):
    if i == 0:
        chi_lab = r'$\chi = 1$'
    if i > 0:
        chi_lab = r'$\chi = 1 \times 10^{-' + str(i) + r'}$'
    col_name = data_var[i]
    if i == 0:
        ax = plt.subplot(2,3,i+1)
    if i > 0:
        plt.subplot(2,3,i+1, sharex=ax, sharey=ax)
    ### Errorbars
    plt.axvline(x=2, color='r', linestyle='--', label = r'$\mathcal{M}_{A} \approx 2$',zorder = 0)
    plt.errorbar(MA, col_name,
             xerr=[MA_Error_L, MA_Error_H],
             yerr=kappa_perp_x_error[i],
             fmt='none',
             capsize = 2,
             c = 'black',
             zorder = 0) 
    plt.errorbar(MA, kappa_perp_array_y[i],
             xerr=[MA_Error_L, MA_Error_H],
             yerr=kappa_perp_y_error[i],
             fmt='none',
             capsize = 2,
             c = 'black',
             zorder = 0) 
    ### Fit lines
    ### Chi label
    if i < 1:
        plt.text(0.1, 2 , chi_lab, size=16, color='black')
    if i > 0:
        plt.text(0.1, 3 , chi_lab, size=16, color='black')
    ###
    sc = plt.scatter(MA, col_name, c=Mach,  s=marker_size, cmap=cm, edgecolor = 'black',zorder = 2)
    sc = plt.scatter(MA, kappa_perp_array_y[i], c=Mach,  s=marker_size, cmap=cm, edgecolor = 'black',zorder = 2)
    # Ticks turn off
    plt.yticks(fontsize = 0)
    plt.xticks(fontsize = 0)
    if i > 2:
        plt.xlabel(r'$\mathcal{M}_{A}$',fontsize = 24)
        plt.xticks(fontsize = 16)    
    if i == 0 or i == 3:
        plt.yticks(fontsize = 16)
        plt.ylabel(r'$\frac{\kappa_{\perp} }{c_s \mathcal{M}}$',fontsize = 34)
    if i == 0:
        plt.legend(frameon = False, fontsize = 16)
    plt.yscale('log')
    plt.xscale('log')
### Set colorbar
cbar_ax = fig.add_axes([0.9, 0.11, 0.03, 0.77])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.ax.tick_params(labelsize=22)
cbar.set_label(r'$\mathcal{M}$', rotation=270, fontsize = 20)
cbar.ax.get_yaxis().labelpad = 25
plt.savefig('Perp_diffusion_all.pdf')

#####################################
### Plot 2 Perp Diffusion
#####################################
#################

# Plot aesthetics
marker_size = 55
cm = cmr.fall
#cm = cmr.savanna
#cm = cmr.voltage
#cm = cmr.gothic
#cm = cmr.torch
#cm = cmr.horizon
fig = plt.figure(figsize=(10.5,6.5), dpi = 200)
plt.rc('font', **{'size':'24'})
plt.subplots_adjust(wspace=0, hspace=0)
### Name data
data_var = kappa_par_array
for i in range(6):
    if i == 0:
        chi_lab = r'$\chi = 1$'
    if i > 0:
        chi_lab = r'$\chi = 1 \times 10^{-' + str(i) + r'}$'
    col_name = data_var[i]
    if i == 0:
        ax = plt.subplot(2,3,i+1)
    if i > 0:
        plt.subplot(2,3,i+1, sharex=ax, sharey=ax)
    ### Errorbars
    #plt.axvline(x=2, color='r', linestyle='--', label = r'$\mathcal{M}_A \approx 2$')
    plt.errorbar(MA_0, col_name,
             xerr=[MA_Error_L, MA_Error_H],
             yerr=kappa_par_error[i],
             fmt='none',
             capsize = 2,
             c = 'black',
             zorder = 0) 
    ### Fit lines
    ### Chi label
    plt.text(1, 80 , chi_lab, size=15, color='black')
    ###
    sc = plt.scatter(MA_0, col_name, c=Mach, s=marker_size, cmap=cm, edgecolor = 'black')
    # Ticks turn off
    plt.yticks(fontsize = 0)
    plt.xticks(fontsize = 0)
    if i > 2:
        plt.xlabel(r'$\mathcal{M}_{A0}$',fontsize = 24)
        plt.xticks(fontsize = 16)    
    if i == 0 or i == 3:
        plt.yticks(fontsize = 16)
        plt.ylabel(r'$\frac{\kappa_{\parallel}}{ c_s \mathcal{M}}$',fontsize = 34)
    if i == 0:
        plt.legend(frameon = False, fontsize = 16, loc = 'upper left')
    plt.yscale('log')
    plt.xscale('log')

### Set colorbar
cbar_ax = fig.add_axes([0.9, 0.11, 0.03, 0.77])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.ax.tick_params(labelsize=22)
cbar.set_label(r'$\mathcal{M}$', rotation=270, fontsize = 24)
cbar.ax.get_yaxis().labelpad = 25
plt.savefig('Par_diffusion_all.pdf')

#####################################
### Mach Plots
#####################################
#####################################
### Plot 2 Perp Diffusion
#####################################
#################
# Plot aesthetics
marker_size = 55
cm = cmr.cosmic
fig = plt.figure(figsize=(12.5,7.5), dpi = 200)
plt.rc('font', **{'size':'24'})
plt.subplots_adjust(wspace=0, hspace=0)
### Name data
data_var = kappa_perp_array
for i in range(6):
    if i == 0:
        chi_lab = r'$\chi = 1$'
    if i > 0:
        chi_lab = r'$\chi = 1 \times 10^{-' + str(i) + r'}$'
    col_name = data_var[i]
    if i == 0:
        ax = plt.subplot(2,3,i+1)
    if i > 0:
        plt.subplot(2,3,i+1, sharex=ax, sharey=ax)
    ### Errorbars
    #plt.axvline(x=2, color='r', linestyle='--', label = r'$\mathcal{M}_{A0} \approx 2$',zorder = 0)
    plt.errorbar(Mach, col_name,
             xerr=[Mach_Error_L, Mach_Error_H],
             yerr=kappa_perp_x_error[i],
             fmt='none',
             capsize = 2,
             c = 'black',
             zorder = 0) 
    plt.errorbar(Mach, kappa_perp_array_y[i],
             xerr=[Mach_Error_L, Mach_Error_H],
             yerr=kappa_perp_y_error[i],
             fmt='none',
             capsize = 2,
             c = 'black',
             zorder = 0) 
    ### Fit lines
    ### Chi label
    if i < 3:
        plt.text(2, 3 , chi_lab, size=16, color='black')
    if i > 2:
        plt.text(6, 3 , chi_lab, size=16, color='black')
    ###
    sc = plt.scatter(Mach, col_name, c=MA_0, norm=colors.LogNorm(), s=marker_size, cmap=cm, edgecolor = 'black',zorder = 2)
    sc = plt.scatter(Mach, kappa_perp_array_y[i], c=MA_0, norm=colors.LogNorm(), s=marker_size, cmap=cm, edgecolor = 'black',zorder = 2)
    # Ticks turn off
    plt.yticks(fontsize = 0)
    plt.xticks(fontsize = 0)
    if i > 2:
        plt.xlabel(r'$\mathcal{M}$',fontsize = 24)
        plt.xticks(fontsize = 16)    
    if i == 0 or i == 3:
        plt.yticks(fontsize = 16)
        plt.ylabel(r'$\frac{\kappa_{\perp} }{c_s \mathcal{M}}$',fontsize = 34)
    if i == 0:
        plt.legend(frameon = False, fontsize = 16, loc = 'lower left')
    plt.yscale('log')
    #plt.xscale('log')
### Set colorbar
cbar_ax = fig.add_axes([0.9, 0.11, 0.03, 0.77])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.ax.tick_params(labelsize=22)
cbar.set_label(r'$\mathcal{M}_{A0}$', rotation=270, fontsize = 20)
cbar.ax.get_yaxis().labelpad = 10
plt.savefig('Perp_diffusion_Mach.pdf')

#####################################
### Plot Parallel Mach
#####################################
#################
fig = plt.figure(figsize=(12.5,7.5), dpi = 200)
plt.rc('font', **{'size':'24'})
plt.subplots_adjust(wspace=0, hspace=0)
### Name data
data_var = kappa_par_array
for i in range(6):
    if i == 0:
        chi_lab = r'$\chi = 1$'
    if i > 0:
        chi_lab = r'$\chi = 1 \times 10^{-' + str(i) + r'}$'
    col_name = data_var[i]
    if i == 0:
        ax = plt.subplot(2,3,i+1)
    if i > 0:
        plt.subplot(2,3,i+1, sharex=ax, sharey=ax)
    ### Errorbars
    #plt.axvline(x=2, color='r', linestyle='--', label = r'$\mathcal{M}_A \approx 2$')
    plt.errorbar(Mach, col_name,
             xerr=[Mach_Error_L, Mach_Error_H],
             yerr=kappa_par_error[i],
             fmt='none',
             capsize = 2,
             c = 'black',
             zorder = 0) 
    ### Fit lines
    ### Chi label
    if i < 3:
        plt.text(2, 80 , chi_lab, size=15, color='black')
    if i > 2:
        plt.text(2, 1e-1 , chi_lab, size=15, color='black')
    ###
    sc = plt.scatter(Mach, col_name, c=MA_0,norm=colors.LogNorm(), s=marker_size, cmap=cm, edgecolor = 'black')
    # Ticks turn off
    plt.yticks(fontsize = 0)
    plt.xticks(fontsize = 0)
    if i > 2:
        plt.xlabel(r'$\mathcal{M}$',fontsize = 24)
        plt.xticks(fontsize = 16)    
    if i == 0 or i == 3:
        plt.yticks(fontsize = 16)
        plt.ylabel(r'$\frac{\kappa_{\parallel}}{ c_s \mathcal{M}}$',fontsize = 34)
    if i == 0:
        plt.legend(frameon = False, fontsize = 16, loc = 'upper left')
    plt.yscale('log')
    #plt.xscale('log')

### Set colorbar
cbar_ax = fig.add_axes([0.9, 0.11, 0.03, 0.77])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.ax.tick_params(labelsize=22)
cbar.set_label(r'$\mathcal{M}$', rotation=270, fontsize = 24)
cbar.ax.get_yaxis().labelpad = 10
plt.savefig('Par_diffusion_Mach.pdf')

#####################################
### Plot 3 Perp Diffusion FLuccuation
#####################################
#################
mag_perp_error_L = mag_perp - Sim_Data['mag_perp_rms_16']
mag_perp_error_H =  Sim_Data['mag_perp_rms_84'] - mag_perp 
# Plot aesthetics
density = Sim_Data['dens_std_50']
cor_b = Sim_Data['B_cor_scale_50']
cor_error_16 = Sim_Data['B_cor_scale_16'] - Sim_Data['B_cor_scale_50']
cor_error_84 = Sim_Data['B_cor_scale_50'] - Sim_Data['B_cor_scale_84']
marker_size = 55
cm = cmr.cosmic
fig = plt.figure(figsize=(12.5,7.5), dpi = 200)
plt.rc('font', **{'size':'24'})
plt.subplots_adjust(wspace=0, hspace=0)
### Name data
data_var = kappa_perp_array
for i in range(6):
    if i == 0:
        chi_lab = r'$\chi = 1$'
    if i > 0:
        chi_lab = r'$\chi = 1 \times 10^{-' + str(i) + r'}$'
    col_name = data_var[i]
    if i == 0:
        ax = plt.subplot(2,3,i+1)
    if i > 0:
        plt.subplot(2,3,i+1, sharex=ax, sharey=ax)
    ### Errorbars
    #plt.axvline(x=2, color='r', linestyle='--', label = r'$\mathcal{M}_A \approx 2$')
    plt.errorbar(cor_b / 0.5, col_name,
             xerr=[cor_error_16, cor_error_84],
             yerr=kappa_perp_x_error[i],
             fmt='none',
             capsize = 2,
             c = 'black',
             zorder = 0) 
    plt.errorbar(cor_b / 0.5, kappa_perp_array_y[i],
             xerr=[cor_error_16, cor_error_84],
             yerr=kappa_perp_y_error[i],
             fmt='none',
             capsize = 2,
             c = 'black',
             zorder = 0) 
    ### Fit lines
    ### Chi label
    if i < 3:
        plt.text(0.5, 3 , chi_lab, size=17, color='black')
    if i > 2:
        plt.text(0.5, 0.03 , chi_lab, size=17, color='black')
    ###
    sc = plt.scatter(cor_b / 0.5, col_name, c=MA_0, norm=colors.LogNorm(), s=marker_size, cmap=cm, edgecolor = 'black')
    sc = plt.scatter(cor_b / 0.5, kappa_perp_array_y[i], c=MA_0, norm=colors.LogNorm(), s=marker_size, cmap=cm, edgecolor = 'black')
    # Ticks turn off
    plt.yticks(fontsize = 0)
    plt.xticks(fontsize = 0)
    if i > 2:
        plt.xlabel(r'$\frac{\ell_{cor,B}}{\ell_0}$',fontsize = 24)
        plt.xticks(fontsize = 16)    
    if i == 0 or i == 3:
        plt.yticks(fontsize = 16)
        plt.ylabel(r'$\frac{\kappa_{\perp}}{c_s \mathcal{M}}$',fontsize = 34)
    if i == 0:
        plt.legend(frameon = False, fontsize = 16, loc = 'upper left')
    plt.yscale('log')
    #plt.xscale('log')
### Set colorbar
cbar_ax = fig.add_axes([0.9, 0.11, 0.03, 0.77])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.ax.tick_params(labelsize=22)
cbar.set_label(r'$\mathcal{M}_{A0}$', rotation=270, fontsize = 17)
cbar.ax.get_yaxis().labelpad = 10
plt.savefig('Perp_diffusion_correlationB.pdf')


#####################################
###
#####################################
fig = plt.figure(figsize=(12.5,7.5), dpi = 200)
plt.rc('font', **{'size':'24'})
plt.subplots_adjust(wspace=0, hspace=0)
### Name data
data_var = kappa_par_array
for i in range(6):
    if i == 0:
        chi_lab = r'$\chi = 1$'
    if i > 0:
        chi_lab = r'$\chi = 1 \times 10^{-' + str(i) + r'}$'
    col_name = data_var[i]
    if i == 0:
        ax = plt.subplot(2,3,i+1)
    if i > 0:
        plt.subplot(2,3,i+1, sharex=ax, sharey=ax)
    ### Errorbars
    #plt.axvline(x=2, color='r', linestyle='--', label = r'$\mathcal{M}_A \approx 2$')
    plt.errorbar(cor_b / 0.5, col_name,
             xerr=[cor_error_16, cor_error_84],
             yerr=kappa_par_error[i],
             fmt='none',
             capsize = 2,
             c = 'black',
             zorder = 0) 
    ### Fit lines
    ### Chi label
    if i < 3:
        plt.text(0.5, 50 , chi_lab, size=17, color='black')
    if i > 2:
        plt.text(0.5, 50 , chi_lab, size=17, color='black')
    ###
    sc = plt.scatter(cor_b / 0.5, col_name, c=MA_0, norm=colors.LogNorm(), s=marker_size, cmap=cm, edgecolor = 'black')
    # Ticks turn off
    plt.yticks(fontsize = 0)
    plt.xticks(fontsize = 0)
    if i > 2:
        plt.xlabel(r'$\frac{\ell_{cor,B}}{\ell_0}$',fontsize = 24)
        plt.xticks(fontsize = 16)    
    if i == 0 or i == 3:
        plt.yticks(fontsize = 16)
        plt.ylabel(r'$\frac{\kappa_{\parallel}}{c_s \mathcal{M}}$',fontsize = 34)
    if i == 0:
        plt.legend(frameon = False, fontsize = 16, loc = 'upper left')
    plt.yscale('log')
    #plt.xscale('log')

### Set colorbar
cbar_ax = fig.add_axes([0.9, 0.11, 0.03, 0.77])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.ax.tick_params(labelsize=22)
cbar.set_label(r'$\mathcal{M}_{A0}$', rotation=270, fontsize = 17)
cbar.ax.get_yaxis().labelpad = 10
plt.savefig('Par_diffusion_correlationB.pdf')

#####################################
### Plot 1 Ratio Plot
#####################################
#################
# Plot aesthetics
marker_size = 55
cm = cmr.cosmic
fig = plt.figure(figsize=(12.0,7.5), dpi = 200)
plt.rc('font', **{'size':'24'})
plt.subplots_adjust(wspace=0, hspace=0)
### Name data
data_var = kappa_ratio
for i in range(6):
    if i == 0:
        chi_lab = r'$\chi = 1$'
    if i > 0:
        chi_lab = r'$\chi = 1 \times 10^{-' + str(i) + r'}$'
    col_name = data_var[i]
    if i == 0:
        ax = plt.subplot(2,3,i+1)
    if i > 0:
        plt.subplot(2,3,i+1, sharex=ax, sharey=ax)
    ### Errorbars
    plt.errorbar(cor_b/0.5, col_name,
             xerr=[cor_error_16, cor_error_84],
             yerr=kappa_perp_y_error[i],
             fmt='none',
             capsize = 2,
             c = 'black',
             zorder = 0) 
    plt.errorbar(cor_b/0.5, kappa_ratio_y[i],
             xerr=[cor_error_16, cor_error_84],
             yerr=kappa_perp_y_error[i],
             fmt='none',
             capsize = 2,
             c = 'black',
             zorder = 0) 
    ### Fit lines
    plt.axhline(y=1, color='r', linestyle='--',zorder = 0)
    plt.text(0.5, 8e2 , chi_lab, size=15, color='black')
    ###
    sc = plt.scatter(cor_b/0.5, col_name, c=MA_0,norm=colors.LogNorm(), s=marker_size, cmap=cm, edgecolor = 'black',zorder = 2)
    sc = plt.scatter(cor_b/0.5, kappa_ratio_y[i], c=MA_0,norm=colors.LogNorm(), s=marker_size, cmap=cm, edgecolor = 'black',zorder = 2)
    # Ticks turn off
    plt.yticks(fontsize = 0)
    plt.xticks(fontsize = 0)
    if i > 2:
        plt.xlabel(r'$\frac{\ell_{cor,B}}{\ell_0}$',fontsize = 24)
        plt.xticks(fontsize = 16)    
    if i == 0 or i == 3:
        plt.yticks(fontsize = 16)
        plt.ylabel(r'$\frac{\kappa_{\parallel}}{\kappa_{\perp}}$',fontsize = 34)
    plt.legend(frameon = False, fontsize = 16.5, loc = 'upper right')
    plt.yscale('log')
    #plt.xscale('log')
    ax.yaxis.set_minor_locator(LogLocator())

### Set colorbar
from matplotlib import ticker
cbar_ax = fig.add_axes([0.9, 0.11, 0.03, 0.77])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.ax.tick_params(labelsize=22)
cbar.set_label(r'$\mathcal{M}$', rotation=270, fontsize = 24)
cbar.ax.get_yaxis().labelpad = 10
plt.savefig('Ratio_Powerlaw_cor.pdf')


#####################################
### Plot 2 Perp Diffusion
#####################################
#################
# Plot aesthetics
marker_size = 55
cm = cmr.fall
fig = plt.figure(figsize=(12.5,7.5), dpi = 200)
plt.rc('font', **{'size':'24'})
plt.subplots_adjust(wspace=0, hspace=0)
### Name data
data_var = kappa_perp_array
for i in range(6):
    if i == 0:
        chi_lab = r'$\chi = 1$'
    if i > 0:
        chi_lab = r'$\chi = 1 \times 10^{-' + str(i) + r'}$'
    col_name = data_var[i]
    if i == 0:
        ax = plt.subplot(2,3,i+1)
    if i > 0:
        plt.subplot(2,3,i+1, sharex=ax, sharey=ax)
    ### Errorbars
    plt.axvline(x=2, color='r', linestyle='--', label = r'$\mathcal{M}_{A} \approx 2$',zorder = 0)
    plt.errorbar(MA, col_name,
             xerr=[MA_Error_L, MA_Error_H],
             yerr=kappa_perp_x_error[i],
             fmt='none',
             capsize = 2,
             c = 'black',
             zorder = 0) 
    plt.errorbar(MA, kappa_perp_array_y[i],
             xerr=[MA_Error_L, MA_Error_H],
             yerr=kappa_perp_y_error[i],
             fmt='none',
             capsize = 2,
             c = 'black',
             zorder = 0) 
    ### Fit lines
    ### Chi label
    if i < 1:
        plt.text(0.1, 2 , chi_lab, size=16, color='black')
    if i > 0:
        plt.text(0.1, 3 , chi_lab, size=16, color='black')
    ###
    sc = plt.scatter(MA, col_name, c=Mach,  s=marker_size, cmap=cm, edgecolor = 'black',zorder = 2)
    sc = plt.scatter(MA, kappa_perp_array_y[i], c=Mach,  s=marker_size, cmap=cm, edgecolor = 'black',zorder = 2)
    # Ticks turn off
    plt.yticks(fontsize = 0)
    plt.xticks(fontsize = 0)
    if i > 2:
        plt.xlabel(r'$\mathcal{M}_{A}$',fontsize = 24)
        plt.xticks(fontsize = 16)    
    if i == 0 or i == 3:
        plt.yticks(fontsize = 16)
        plt.ylabel(r'$\frac{\kappa_{\perp} }{c_s \mathcal{M}}$',fontsize = 34)
    if i == 0:
        plt.legend(frameon = False, fontsize = 16)
    plt.yscale('log')
    plt.xscale('log')
### Set colorbar
cbar_ax = fig.add_axes([0.9, 0.11, 0.03, 0.77])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.ax.tick_params(labelsize=22)
cbar.set_label(r'$\mathcal{M}$', rotation=270, fontsize = 20)
cbar.ax.get_yaxis().labelpad = 25
plt.savefig('Perp_diffusion_Chi.pdf')