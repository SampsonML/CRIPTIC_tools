"""
Fitting Script for diffusion coefficents CR study
Matt Sampson 2021
"""

import argparse
import numpy as np
from numpy import diff
import astropy.units as u
import astropy.constants as const
from glob import glob
import os.path as osp
from cripticpy import readchk
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib as mpl
from matplotlib import rc_context
import argparse
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors as colors
from scipy.optimize import minimize
from scipy.stats import levy_stable

###################################
# MCMC things
import scipy.stats as st, levy
import emcee
import corner
from multiprocessing import Pool, cpu_count
###################################

############################################
#### Astro Plot Aesthetics Pre-Amble
############################################
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

#plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
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
# Parse arguements
parser = argparse.ArgumentParser(
    description="Analysis script")
parser.add_argument("--testonly",
                    help="report test results only, do not make plots",
                    default=False, action="store_true")
parser.add_argument("-c", "--chk",
                    help="number of checkpoint file to examine (default "
                    "is last checkpoint file in directory)",
                    default=-1, type=int)
parser.add_argument("-d", "--dir",
                    help="directory containing run to analyze",
                    default=".", type=str)
parser.add_argument("-f", "--filenum",
                    help="Amount of chunk files",
                    default="1", type=int)
parser.add_argument("-m", "--Mach",
                    help="Amount of chunk files",
                    default="1", type=float)
parser.add_argument("-a", "--Alfven",
                    help="Amount of chunk files",
                    default="1", type=float)
parser.add_argument("-s", "--c_s",
                    help="Amount of chunk files",
                    default="1", type=float)
parser.add_argument("-i", "--chi",
                    help="Amount of chunk files",
                    default="1", type=float)
parser.add_argument("-r", "--dens",
                    help="Amount of chunk files",
                    default="1", type=float)                  
args = parser.parse_args()

# Extract parameters we need from the input file
fp = open(osp.join(args.dir, 'criptic.in'), 'r')
chkname = 'criptic_'
for line in fp:
    l = line.split('#')[0]    # Strip comments
    s = l.split()
    if len(s) == 0: continue
    if s[0] == 'output.chkname':
        chkname = s[1]
    elif s[0] == 'cr.kPar0':
        kPar0 = float(s[1]) * u.cm**2 / u.s
    elif s[0] == 'cr.kParIdx':
        kParIdx = float(s[1])
    elif s[0] == 'prob.L':
        Lsrc = np.array([ float(s1) for s1 in s[1:]]) * u.erg/u.s
    elif s[0] == 'prob.T':
        Tsrc = np.array([ float(s1) for s1 in s[1:]]) * u.GeV
    elif s[0] == 'prob.r':
        rsrc = float(s[1]) * u.cm
fp.close()

###################################
# Initialise Problem
##################################
# Allocate numpy arrary
variance_cr_perp     = np.zeros(args.filenum)
variance_cr_par     = np.zeros(args.filenum)
chunk_age       = np.zeros(args.filenum)
error_variance_perp  = np.zeros(args.filenum)
error_variance_par  = np.zeros(args.filenum)
displacement = [0] ; displacement_par = [0]
displacement_perp = [0] ; displacement_perp_y = [0]
time_vals = [0]
# Domain Size
L = 3.09e19
# Getting MHD specific parameters
Mach = args.Mach
Alfven = args.Alfven
chi = 1 * 10**(-args.chi)
c_s = args.c_s
rho_0 = 2e-21
B =  c_s * Mach * (1 / Alfven) * np.sqrt(4 * np.pi * rho_0)
# Defining Speeds
V_alfven = B / (np.sqrt(4 * np.pi * rho_0))
Vstr = (1/np.sqrt(chi)) * V_alfven
# Defining timescales
t_cross = 2 * L / Vstr
t_turb = (2 * L) / (c_s * Mach)
t_turnover = t_turb / 2
t_alfven = (2 * L) / V_alfven
###################################
by_num = 100 # Added to speed up small scale local PC testing
if (by_num > args.filenum):
    by_num = 1
    print(f'Chunk gap revert to {1}')
total_load_ins = int(args.filenum / by_num)

for k in range(total_load_ins):  
    #########################################################
    # Figure out which checkpoint file to examine
    args.chk = 1 + k * by_num
    if args.chk >= 0:
        chknum = "{:05d}".format(args.chk)
    else:
        chkfiles = glob(osp.join(args.dir,
                                 chkname+"[0-9][0-9][0-9][0-9][0-9].txt"))
        chknums = [ int(c[-9:-4]) for c in chkfiles ]
        chknum = "{:05d}".format(np.amax(chknums))

    ########################################################
    # Check if file exists
    Path = osp.join(args.dir,chkname+chknum) + '.txt'
    #print(Path)
    file_exist = osp.exists(Path)
    if (file_exist == False):
        #print('Here')
        break
    ########################################################
    # Read the checkpoint
    t, packets, delpackets, sources = readchk(osp.join(args.dir,
                                                       chkname+chknum))
    ########################################################
    
    ########################################################
    ### Extract useful particle information
    ########################################################
    # Get Source
    source = packets['source'][np.argsort(packets['uid'])]
    part_source = sources['x'][np.argsort(sources['uid'])].to_value()
    source_x, source_y, source_z = zip(*part_source)
    # Get Particle Positions
    x = packets['x'][np.argsort(packets['uid'])].to_value()
    x_pos, y_pos, z_pos = zip(*x)
    # Get Particle Age
    particle_age =  t.to_value() - packets['tinj'][np.argsort(packets['uid'])].to_value()
    ## Zip together for sorting
    zipped_lists = zip(particle_age, x_pos, y_pos, z_pos, source)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    particle_age, x_pos, y_pos, z_pos, source = [ list(tuple) for tuple in  tuples]
    particle_age = np.asarray(particle_age)
    #particle_age = particle_age / (60 * 60 * 24 * 365)
    x_pos = np.asarray(x_pos)
    y_pos = np.asarray(y_pos)
    z_pos = np.asarray(z_pos)
    
    ##########################################################
    ### Adjust for Periodicity and Source Location
    ##########################################################
    
    for i in range(len(x_pos)):
        # The bounds adjustment
        if (np.abs(x_pos[i] - source_x[source[i]]) >= L):
            x_pos[i] = x_pos[i] - np.sign(x_pos[i]) * 2 * L
        if (np.abs(y_pos[i] - source_y[source[i]]) >= L):
            y_pos[i] = y_pos[i] - np.sign(y_pos[i]) * 2 * L
        if (np.abs(z_pos[i] - source_z[source[i]]) >= L):
            z_pos[i] = z_pos[i] - np.sign(z_pos[i]) * 2 * L

        # The source adjustment to center all CRs
        x_pos[i] = x_pos[i] - source_x[source[i]]
        y_pos[i] = y_pos[i] - source_y[source[i]]
        z_pos[i] = z_pos[i] - source_z[source[i]] 

    ########################################
    ### Displacement All Data
    ########################################
    displacement_perp = np.append(displacement_perp, (x_pos + x_pos)/2)
    displacement_par = np.append(displacement_par, z_pos)
    time_vals = np.append(time_vals, particle_age)

########################################
### Age Selections
########################################
Data_Combine = np.column_stack((displacement_perp,time_vals, displacement_par))
# Delete old particles
Data_Use = Data_Combine[Data_Combine[:,1] > 1e13]
Data_Use = Data_Use[Data_Use[:,1] < 5e13]

################################################
### Levi Dist
################################################
def distributionLevy(pos,alpha,beta, scale, loc):
    Prob = levy_stable.pdf(pos, alpha, beta, loc, scale)
    return Prob


################################################
#### MCMC 
################################################

###################################
### Define log likelyhood
def log_lik_drift(theta,pos, t):
    L_n =  4
    finite_val = 40
    beta = 0
    mu = 0
    alpha = theta[0]
    scale = theta[1]
    drift = theta[2] 
    drift_val = drift * Vstr 
    ### Account for drift
    pos = pos  -  (drift_val * t) 
    dist_levi = np.zeros(len(pos))
    ### Sum of Levi dist over x + n* L for finite n 
    for i in range(-finite_val, finite_val):
        x_dim = (np.abs(pos + i * L_n) ** (alpha)) / t
        dist_levi += levy.levy(x_dim, alpha=alpha, beta=beta, 
        mu=mu, sigma=scale, cdf=False) # p(theta | x)
    log_sum = sum(np.log(dist_levi))
    return  log_sum

###################################
### Defining prior distribution
def log_prior(theta):
    alpha, scale, drift = theta
    if 0.0 < alpha <= 2 and 0.05 < scale < 10 and 0 <= drift < 50:
        return 0.0
    return -np.inf

###################################
### Define log probability
def log_probability(theta, pos, t):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_lik_drift(theta,pos, t)

#######################################################
'''
   Beginning of MCMC routine, here the function 
   log_lik_drift is the likelyhood function.
   Searching over 3 parameters, alpha (from Levi),
   scale/kappa (from Levi) and drift/gamma which
   is to account for the drift in parallel diffusion.
'''
#######################################################

if __name__ == '__main__':
    

    ###################################
    ''' Non dimensionalise everything'''
    # L = driving scale
    pos = 2 * Data_Use[:,0] / L                       # Divide by Driving scale
    t = Data_Use[:,1] / ( (2 * L) / (c_s * Mach) )    # Divide by turbulent time
    ###################################
    ### Naive intial guess to check values
    # Not used in fitting
    par , log_lik = levy.fit_levy((pos**2 / t),beta=0, mu = 0)
    values = par.get('0') ; scale = values[3] ; alpha = values[0]
    mu = values[2]  ; beta = values[1]
    print(f'scale: {scale}, Mean: {mu}')

    ###################################
    # Set initial drift guess as mean 
    n_cores = 3 # Core num
    num_walkers = n_cores * 3
    num_of_steps = 1000
    alpha = 1.5
    scale = 4
    drift = 1
    print(f'Alpha: {alpha}, Scale: {scale}, Drift: {drift}')
    params = np.array([alpha, scale, drift])
    init_guess = params
    position = init_guess + 1e-2 * np.random.randn(num_walkers, 3) # Add random noise to guess
    nwalkers, ndim = position.shape
    
    ###################################
    # Multiprocessing    
    with Pool(n_cores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(pos, t), pool=pool)
        sampler.run_mcmc(position,
                        num_of_steps,
                        progress=True)
    
    
    ###################################
    ### Running the MCMC steps
    '''
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(pos, t)
    )
    sampler.run_mcmc(position, num_of_steps, progress=True);
    '''
    ###################################
    ### Visualisations
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = [r"$\alpha$", "scale", "drift"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "midnightblue", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    
    axes[-1].set_xlabel("step number")
    name = 'bands_' + str(Mach) + '_' + str(Alfven) + '_' + str(args.chi) + '_par' + '.png'
    plt.savefig(name)
    plt.close()
    flat_samples = sampler.get_chain(discard=600, thin=15, flat=True)
    
    ###################################
    ### Corner plot
    labels          = [r"$\alpha$", r"$\kappa$", r"$\gamma$"]
    # initialise figure
    fig             = corner.corner(flat_samples,
                        labels=labels,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 12},
                        color='midnightblue',
                        truth_color='red',
                        truths=[np.median(flat_samples[:,0]), np.median(flat_samples[:,1]), np.median(flat_samples[:,2])])
    #fig.suptitle(r'$\mathcal{M} = 2 \ \ \ \mathcal{M}_{A0} \approx 2 \ \ \ \chi = 1 \times10^{-4}$',
    #fontsize = 24,y=0.98)
    name = 'corner_' + str(Mach) + '_' + str(Alfven) + '_' + str(args.chi) + '_par' + '.png'
    plt.savefig(name)
    plt.close()
    
    ###################################
    ### Store table data
