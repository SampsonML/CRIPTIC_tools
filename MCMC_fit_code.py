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
from scipy.stats import invgamma
from scipy import stats
from scipy.stats import norm
from scipy.stats import rv_histogram
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
chunk_age       = np.zeros(args.filenum)
displacement = [0] ; displacement_par = [0]
displacement_perp = [0] ; displacement_perp_y = [0]
time_vals = [0]
# 1 /2 Domain Size
L = 3.09e19
# Getting MHD specific parameters
Mach = args.Mach
Alfven = args.Alfven
chi = 1 * 10**(-args.chi)
c_s = args.c_s
rho_0 = 2e-21 # This value for all sims
B =  c_s * Mach * (1 / Alfven) * np.sqrt(4 * np.pi * rho_0)
# Defining Speeds
V_alfven = B / (np.sqrt(4 * np.pi * rho_0))
Vstr = (1/np.sqrt(chi)) * V_alfven
# Defining timescales
t_cross = 2 * L / Vstr
t_turb = (2 * L) / (c_s * Mach)
t_turnover = t_turb / 2
t_alfven = (2 * L) / V_alfven

#############################################################
### Beginning data read in
#############################################################
by_num = 25 # Added to speed up small scale local PC testing set to 1 for GADI
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
    displacement_perp = np.append(displacement_perp, (x_pos + y_pos)/2)
    displacement_par = np.append(displacement_par, z_pos)
    time_vals = np.append(time_vals, particle_age)

########################################
### Age Selections
########################################
Data_Combine = np.column_stack((displacement_perp, time_vals ,displacement_par ))
# Subset data for speed
Data_Use = Data_Combine[Data_Combine[:,1] > 1e1]
Data_Use = Data_Use[Data_Use[:,1] < 1e14]
### Randomly sample data points
number_of_rows = Data_Use.shape[0]
data_points = 5000
random_indices = np.random.choice(number_of_rows, size=data_points, replace=False)
Data_Use = Data_Use[random_indices, :]

################################################
#### MCMC 
################################################
###################################
### Define log likelyhood
def log_lik_drift(theta,pos, t):
    '''
    Log likleyhood function for
    computing a Levy distribution with
    3 parameters, alpha, scale and drift.
    '''
    ### Levy Parameters
    L_n =  2             # Box size
    beta = 0             # Skew
    mu = 0               # Location
    alpha =  theta[0]    # Shape -- superdiffusion
    scale =  theta[1]    # scale -- diffusion coefficient
    drift =  theta[2]    # bulk particle drift speed

    ### Account for drift
    pos_n =  pos  -  (drift * t) % L_n
    for n in range(len(pos_n)):
        while (np.abs(pos_n[n]) > 1):
            pos_n[n] = pos_n[n] - np.sign(pos_n[n]) * 2

    ### Initialising variables
    chi = 0
    tol = 0.1                            # 1% Tolerance
    finite_range = 8
    bin_num = 40
    x_dim = (pos_n) / (t**(1 / alpha))
    dist_levy = levy.levy(x_dim, alpha=alpha, beta=beta, 
            mu=mu, sigma=scale, cdf=False)
    dist_n1   = dist_levy
    err = 5 * tol
    jump_val = 0
    cross_init = 1

    ###############################################
    ### Ewalds summation for periodic bounds
    while  (err > tol):           # Add more terms 
        for i in [x for x in range(finite_range - 1, -finite_range, -1) if x !=0]:
            x_dim = (pos_n +  i * L_n + np.sign(i) * jump_val * L_n) / (t**(1 / alpha))
            dist_n0 = dist_n1
            dist_n1 = levy.levy(x_dim, alpha=alpha, beta=beta, 
            mu=mu, sigma=scale, cdf=False) # p(x| theta) for n + 1
            dist_levy += dist_n1
        err = np.abs(1 - np.sum(dist_n1) / np.sum(dist_n0))
        jump_val += finite_range - 1
    
    ############################################### 
    ### Data PDF
    x_dim = (pos_n) / (t**(1 / alpha))
    sorted_indices = np.argsort(x_dim)
    x_dim = x_dim[sorted_indices]
    dist_levy = dist_levy[sorted_indices]
    r = rv_histogram(np.histogram(x_dim, bins=45))
    data_pdf = r.pdf(x_dim)

    ### Compute chi^2
    chi = np.sum(((data_pdf - dist_levy)**2 )/ dist_levy)  # chi^2 
    return -chi

###################################
### Defining prior distribution
def log_prior(theta):
    alpha, scale, drift = theta
    if 1.0 < alpha <= 2.0 and 0.0 < scale < 1e4 and 0 <= drift:
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
   Searching over 3 parameters, alpha (from Levy),
   scale/kappa (from Levy) and drift/gamma which
   is to account for the drift in parallel diffusion.
'''
#######################################################
if __name__ == '__main__':
    
    ###################################
    ### Perpendicular
    ###################################
    ''' Non dimensionalise everything'''
    # L = driving scale
    pos = Data_Use[:,0] / L                       # Divide by Driving scale
    t = Data_Use[:,1] / ( L / (c_s * Mach) )    # Divide by turbulent time
    perp = Data_Use[:,2] / L 
    ####################################
    print('')
    print('=============================')
    print('         MCMC trials         ')
    print(f'    Using {len(pos)} data points')
    print('=============================')
    print('')
    ####################################
    # Set initial drift guess as mean 
    n_cores = 3 # Core num
    num_walkers = n_cores * 6
    num_of_steps = 250
    burn_num = 175
    alpha = 1.4
    scale = 0.5
    drift = 0.2 
    params = np.array([alpha, scale, drift])
    init_guess = params
    position = init_guess +  2e-1 * np.random.randn(num_walkers, 3) # Add random noise to guess
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
    
    ### Visualisations
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = [r"$\alpha$", r"$\kappa$", r"$\gamma$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "midnightblue", alpha=0.4)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    
    axes[-1].set_xlabel("step number")
    name = 'drift_bands_' + str(Mach) + '_' + str(Alfven) + '_' + str(args.chi) + '_perp' + '.png'
    plt.savefig(name)
    plt.close()
    
    flat_samples = sampler.get_chain(discard=burn_num, thin=15, flat=True)
    flat_samples[:,1] = flat_samples[:,1]**(flat_samples[:,0])

    ################################################
    ### Saving results as txt file
    ################################################
    Filename = 'walkers_perp_' + args.dir + '.txt'
    np.savetxt(Filename, flat_samples, delimiter=',')
    ################################################
    
    ###################################
    ### Corner plot
    labels          = [r"$\alpha$", r"$\kappa$", r"$\gamma$"]
    #labels          = [r"$\kappa$", r"$\gamma$"]
    # initialise figure
    fig             = corner.corner(flat_samples,
                        labels=labels,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 13},
                        color='midnightblue',
                        truth_color='red',
                        truths=[np.median(flat_samples[:,0]),np.median(flat_samples[:,1]),np.median(flat_samples[:,2])])
    #fig.suptitle(r'$\mathcal{M} = 2 \ \ \ \mathcal{M}_{A0} \approx 2 \ \ \ \chi = 1 \times10^{-4}$',
    #fontsize = 24,y=0.98)
    name = 'drift_corner_' + str(Mach) + '_' + str(Alfven) + '_' + str(args.chi) + '_perp' + '.png'
    plt.savefig(name)
    plt.close()
    
    ############################################################
    ### Save data and 1 sigma levels
    ############################################################
    alpha_perp  = np.median(flat_samples[:,0])
    a_perp_lo = alpha_perp - np.quantile(flat_samples[:,0],0.16) 
    a_perp_hi = np.quantile(flat_samples[:,0],0.84) - alpha_perp
    scale_perp = np.median(flat_samples[:,1])
    s_perp_lo = scale_perp - np.quantile(flat_samples[:,1],0.16) 
    s_perp_hi = np.quantile(flat_samples[:,1],0.84) - scale_perp
    drift_perp = np.median(flat_samples[:,2])
    d_perp_lo = drift_perp - np.quantile(flat_samples[:,2],0.16) 
    d_perp_hi = np.quantile(flat_samples[:,2],0.84) - drift_perp
    
    ###################################
    ### Parallel
    ###################################
    # Non dimensionalise everything
    # L = driving scale
    pos = Data_Use[:,2] / L                     # Divide by Driving scale
    t = Data_Use[:,1] / ( L / (c_s * Mach) )    # Divide by turbulent time
    ###################################
    
    ###################################
    # Set initial guess 
    alpha = 1.5
    scale = 1
    drift = 1
    params = np.array([alpha, scale, drift])
    init_guess = params
    position = init_guess + 2e-1 * np.random.randn(num_walkers, 3) # Add random noise to guess
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
    ### Visualisations
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = [r"$\alpha$", r"$\kappa$", r"$\gamma$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "midnightblue", alpha=0.4)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    
    axes[-1].set_xlabel("step number")
    name = 'drift_bands_' + str(Mach) + '_' + str(Alfven) + '_' + str(args.chi) + '_par' + '.png'
    plt.savefig(name)
    plt.close()
    flat_samples = sampler.get_chain(discard=burn_num, thin=15, flat=True)
    flat_samples[:,1] = flat_samples[:,1]**(flat_samples[:,0])

    ################################################
    ### Saving results as txt file
    ################################################
    Filename = 'walkers_par_' + args.dir + '.txt'
    np.savetxt(Filename, flat_samples, delimiter=',')
    ################################################
    
    ###################################
    ### Corner plot
    labels          = [r"$\alpha$", r"$\kappa$", r"$\gamma$"]
    # initialise figure
    fig             = corner.corner(flat_samples,
                        labels=labels,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 13},
                        color='midnightblue',
                        truth_color='red',
                        truths=[np.median(flat_samples[:,0]), np.median(flat_samples[:,1]), np.median(flat_samples[:,2])])
    #fig.suptitle(r'$\mathcal{M} = 2 \ \ \ \mathcal{M}_{A0} \approx 2 \ \ \ \chi = 1 \times10^{-4}$',
    #fontsize = 24,y=0.98)
    name = 'drift_corner_' + str(Mach) + '_' + str(Alfven) + '_' + str(args.chi) + '_par' + '.png'
    plt.savefig(name)
    plt.close()
    
    ############################################################
    ### Save data and 1 sigma levels
    ############################################################
    alpha_par  = np.median(flat_samples[:,0])
    a_par_lo = alpha_par - np.quantile(flat_samples[:,0],0.16) 
    a_par_hi = np.quantile(flat_samples[:,0],0.84) - alpha_par
    scale_par = np.median(flat_samples[:,1])
    s_par_lo = scale_par - np.quantile(flat_samples[:,1],0.16) 
    s_par_hi = np.quantile(flat_samples[:,1],0.84) - scale_par
    drift_par = np.median(flat_samples[:,2])
    d_par_lo = drift_par - np.quantile(flat_samples[:,2],0.16) 
    d_par_hi = np.quantile(flat_samples[:,2],0.84) - drift_par
    ################################################
    ### Plotting fitted Levy distribution
    ################################################
    '''
    #Need to sum analytical fits over "finite range" bounds
    #Need to check the normalisation,
    #int dx (analytic) == \int dx (Hist)
    '''


    ### Get hist data
    fig = plt.figure(figsize=(20.0,12.0), dpi = 150)
    plt.style.use('dark_background')
    ### Plotting Perpendicular
    ### DELETE after ad hoc fitting ###
    #alpha_perp = 1.35
    #alpha_par = 1.5
    #scale_perp = 1.2
    #scale_par = 1
    #drift_perp = 0
    #drift_par = 2
    print('')
    print('====== Parameters =======')
    print(f'Alpha: {alpha_perp}')
    print(f'Scale: {scale_perp}')
    print(f'========================')
    print('')
    ####################################
    ### Perp Ewald Sum
    tol = 0.1
    L_n =  2 
    finite_range = 8
    beta = 0             # Skew
    mu = 0               # Location
    ########################
    ### Account for drift
    pos = Data_Use[:,0] / L 
    pos_n =  pos  -  (drift_perp * t) % L_n
    for n in range(len(pos_n)):
        if (np.abs(pos_n[n]) > 1):
            pos_n[n] = pos_n[n] - np.sign(pos_n[n]) * 2
    # So every single pos is -1 to 1
    #######################
    x_dim = pos / (t**(1 / alpha_perp))
    x_dim = np.sort(x_dim)
    x_diffs = np.diff(x_dim)
    dist_levy = np.zeros(len(x_dim))
    dist_init = levy.levy(x_dim, alpha=alpha_perp, beta=beta, 
    mu=mu, sigma=scale_perp, cdf=False) # p(theta | x)
    init_sum = np.sum(dist_init)
    ewald_sum = init_sum
    x_dim = (pos_n) / (t**(1 / alpha_perp))
    dist_levy = levy.levy(x_dim, alpha=alpha_perp, beta=beta, 
            mu=mu, sigma=scale_perp, cdf=False)
    dist_n1   = dist_levy
    err = 5 * tol
    jump_val = 0
    ###############################################
    ### Ewalds summation for periodic bounds
    while  (err > tol):           # Add more terms 
        for i in [x for x in range(finite_range - 1, -finite_range, -1) if x !=0]:
            x_dim = (pos_n +  i * L_n + np.sign(i) * jump_val * L_n) / (t**(1 / alpha_perp))
            dist_n0 = dist_n1
            dist_n1 = levy.levy(x_dim, alpha=alpha_perp, beta=beta, 
            mu=mu, sigma=scale_perp, cdf=False) # p(x| theta) for n + 1
            dist_levy += dist_n1
        err = np.abs(1 - np.sum(dist_n1) / np.sum(dist_n0))
        jump_val += finite_range - 1
    sorted_indices = np.argsort(x_dim)
    x_dim = x_dim[sorted_indices]
    dist_levy = dist_levy[sorted_indices]
    Prob_perp_analytic = dist_levy
    ################################################
    
    x_dim = pos / (t**(1 / alpha_perp))
    x_dim = np.sort(x_dim)
    Prob_perp_n0 = levy.levy(x_dim, alpha=alpha_perp, beta=beta, 
            mu=mu, sigma=scale_perp, cdf=False)

    vals = stats.gaussian_kde(x_dim)
    KDE_perp = stats.gaussian_kde.pdf(vals,x_dim)
    
    ############################################
    ### Finite val 2
    finite_val_2 = 5
    Prob_perp_analytic_2 = np.zeros(len(x_dim))
    for i in range(-finite_val_2, finite_val_2):
            x_dim = (pos +  i * L_n) / (t**(1 / alpha_perp))
            x_dim = np.sort(x_dim)
            Prob_perp_analytic_2 += levy.levy(x_dim, alpha=alpha_perp, beta=beta, 
            mu=mu, sigma=scale_perp, cdf=False) # p(theta | x)
    x_dim = pos / (t**(1 / alpha_perp))
    x_dim = np.sort(x_dim)

    # Chi^2
    vals = stats.gaussian_kde(x_dim)
    Test = stats.gaussian_kde.pdf(vals,x_dim)
    log_sum = sum(np.log(Prob_perp_analytic))
    chi = np.sum((Test - Prob_perp_analytic)**2 / Prob_perp_analytic)
    chi_perp = r'$\chi^2$ = ' + str(round(chi,2))
    lik_lab = r'$logLik$ $=$ ' + str(round(sum(np.log(Prob_perp_analytic)),2))
    ###############################################################
    ### Parallel ###
    
    pos = Data_Use[:,2] / L 
    Prob_par_analytic = np.zeros(len(pos))
    pos_par = pos  -  (drift_par * t) % L_n
    for n in range(len(pos_par)):
        if np.abs(pos_par[n]) > 1:
                pos_par[n] = pos_par[n] - np.sign(pos_par[n]) * 2
    bin_num =  45
    x_dim_par = pos_par / (t**(1 / alpha_par))
    x_dim_par = np.sort(x_dim_par)
    x_diffs = np.diff(x_dim)
    dist_levy = np.zeros(len(x_dim_par))
    dist_init = levy.levy(x_dim_par, alpha=alpha_par, beta=beta, 
    mu=mu, sigma=scale_par, cdf=False) # p(theta | x)
    init_sum = np.sum(dist_init)
    ewald_sum = init_sum
    x_dim_par = (pos_par) / (t**(1 / alpha_par))
    dist_levy = levy.levy(x_dim, alpha=alpha_perp, beta=beta, 
            mu=mu, sigma=scale_par, cdf=False)
    dist_n1   = dist_levy
    err = 5 * tol
    jump_val = 0
    ###############################################
    ### Ewalds summation for periodic bounds
    while  (err > tol):           # Add more terms 
        for i in [x for x in range(finite_range - 1, -finite_range, -1) if x !=0]:
            x_dim_par = (pos_par +  i * L_n + np.sign(i) * jump_val * L_n) / (t**(1 / alpha_par))
            dist_n0 = dist_n1
            dist_n1 = levy.levy(x_dim, alpha=alpha_par, beta=beta, 
            mu=mu, sigma=scale_par, cdf=False) # p(x| theta) for n + 1
            dist_levy += dist_n1
        err = np.abs(1 - np.sum(dist_n1) / np.sum(dist_n0))
        jump_val += finite_range - 1
    sorted_indices = np.argsort(x_dim_par)
    x_dim_par = x_dim_par[sorted_indices]
    dist_levy = dist_levy[sorted_indices]
    Prob_par_analytic = dist_levy
    ################################################

    x_dim_par = pos_par / (t**(1 / alpha_par))
    x_dim_par = np.sort(x_dim_par)
    Prob_par_n0 = levy.levy(x_dim_par, alpha=alpha_par, beta=beta, 
            mu=mu, sigma=scale_par, cdf=False)

    vals = stats.gaussian_kde(x_dim_par)
    KDE_par = stats.gaussian_kde.pdf(vals,x_dim_par)

    ############################################
    ### Finite val 2
    #finite_val_2 = 20
    Prob_par_analytic_2 = np.zeros(len(x_dim_par))
    for i in range(-finite_val_2, finite_val_2):
            x_dim_par = (pos_par +  i * L_n) / (t**(1 / alpha_par))
            x_dim_par = np.sort(x_dim_par)
            Prob_par_analytic_2 += levy.levy(x_dim_par, alpha=alpha_par, beta=beta, 
            mu=mu, sigma=scale_par, cdf=False) # p(theta | x)
    
    x_dim_par = pos_par / (t**(1 / alpha_par))
    x_dim_par = np.sort(x_dim_par)
    pos_par = x_dim_par

    vals = stats.gaussian_kde(x_dim_par)
    Test = stats.gaussian_kde.pdf(vals,x_dim_par)
    log_sum = sum(np.log(Prob_perp_analytic))
    chi = np.sum((Test - Prob_par_analytic)**2 / Prob_par_analytic)
    chi_par = r'$\chi^2$ = ' + str(round(chi,2))

    
    #############################################################
    lab_a_perp = r'$\alpha = ' + str(round(alpha_perp,3)) + r'^{' + str(round(a_perp_hi,3)) + r'}_{' + str(round(a_perp_lo,3)) + r'}$' 
    lab_s_perp = r'$\kappa = ' + str(round(scale_perp,3)) + r'^{' + str(round(s_perp_hi,3)) + r'}_{' + str(round(s_perp_lo,3)) + r'}$' 
    lab_d_perp = r'$\gamma = ' + str(round(drift_perp,3)) + r'^{' + str(round(d_perp_hi,3)) + r'}_{' + str(round(d_perp_lo,3)) + r'}$' 

    
    lab_a_par = r'$\alpha = ' + str(round(alpha_par,3)) + r'^{' + str(round(a_par_hi,3)) + r'}_{' + str(round(a_par_lo,3)) + r'}$' 
    lab_s_par = r'$\kappa = ' + str(round(scale_par,3)) + r'^{' + str(round(s_par_hi,3)) + r'}_{' + str(round(s_par_lo,3)) + r'}$' 
    lab_d_par = r'$\gamma = ' + str(round(drift_par,3)) + r'^{' + str(round(d_par_hi,3)) + r'}_{' + str(round(d_par_lo,3)) + r'}$' 
    

    ### Colors
    fig = plt.figure(figsize=(18.5,14.5), dpi = 200)
    Hist_col = 'aqua'
    Levy_col = 'darkorange'
    plt.subplot(2,2,1)
    counts, bins = np.histogram(x_dim, bins = bin_num, density = True)
    plt.hist(bins[:-1], bins,weights=counts,histtype='step', 
    edgecolor = Hist_col, label = r'CRIPTIC', linewidth=2, zorder = 1)
    plt.plot(x_dim,Prob_perp_n0 , linestyle='--', c=purp_m,
            alpha=0.99, label=r'Levy n = 0', lw = 3, zorder = 2)
    plt.plot(x_dim,Prob_perp_analytic , linestyle='-', c=Levy_col,
            alpha=0.99, label=f'Levy n = Converged', lw = 3, zorder = 2)
    #plt.plot(x_dim,Prob_perp_analytic_2 , linestyle='-.', c=green_m,
    #        alpha=0.99, label=f'Levy n = {finite_val_2}', lw = 3, zorder = 2)
    plt.xlabel(r'($|\Delta x - \gamma_{\perp}\cdot t|) / t^{1 / \alpha_{\perp}}$', fontsize = 26)
    plt.legend(frameon=False,fontsize=16, loc = 'upper right')
    plt.ylabel(r'PDF', fontsize = 26)
    plt.title(r'Perpendicular $\mathcal{M}_{A0} \approx$ ' + str(Alfven),fontsize = 28)
    # Add labels
    xmin, xmax, ymin, ymax = plt.axis()
    x_text = 0.9*xmin
    y_text_1 = 0.9*ymax ; y_text_2 = 0.8*ymax ; y_text_3 = 0.7*ymax ; y_text_4 = 0.6*ymax
    plt.text(x_text, y_text_1, lab_a_perp, fontsize = 22)
    plt.text(x_text, y_text_2, lab_s_perp, fontsize = 22)
    plt.text(x_text, y_text_3, chi_perp, fontsize = 22)
    #plt.text(x_text, y_text_1, chi_lab, fontsize = 22)

    plt.subplot(2,2,3)
    plt.hist(bins[:-1], bins, weights=counts, density = True, histtype='step', 
    edgecolor = Hist_col, label = r'CRIPTIC', linewidth=2, zorder = 1)
    plt.plot(x_dim,Prob_perp_n0 , linestyle='--', c=purp_m,
            alpha=0.99, label=r'Levy n = 0', lw = 3, zorder = 2)
    plt.plot(x_dim,Prob_perp_analytic , linestyle='-', c=Levy_col,
            alpha=0.99, label=f'Levy n = Converged', lw = 3, zorder = 2)
    #plt.plot(x_dim,Prob_perp_analytic_2 , linestyle='-.', c=green_m,
    #        alpha=0.99, label=f'Levy n = {finite_val_2}', lw = 3, zorder = 2)
    plt.legend(frameon=False,fontsize=16, loc = 'upper right')
    plt.xlabel(r'($|\Delta x - \gamma_{\perp}\cdot t|) / t^{1 / \alpha_{\perp}}$', fontsize = 26)
    plt.ylabel(r'log(PDF)', fontsize = 26)
    plt.yscale('log')
    #plt.savefig('drift_' +  str(Mach) + '_' + str(Alfven) + '_' + str(args.chi)  + "_fitting.png")
    #plt.close()
    
    plt.subplot(2,2,2)
    plt.plot(x_dim_par,Prob_par_n0 , linestyle='--', c=purp_m,
            alpha=0.99, label=r'Levy n = 0', lw = 3, zorder = 2)
    plt.plot(x_dim_par, Prob_par_analytic , linestyle='-', c=Levy_col,
            alpha=0.99, label=f'Levy n = Converged', lw = 2)
    #plt.plot(x_dim_par,Prob_par_analytic_2 , linestyle='-.', c=green_m,
    #        alpha=0.99, label=f'Levy n = {finite_val_2}', lw = 3, zorder = 2)
    counts, bins = np.histogram(x_dim_par, bins = bin_num)
    plt.hist(bins[:-1], bins, weights=counts, density = 'true', histtype='step', 
    edgecolor = Hist_col, label = r'CRIPTIC', linewidth=2)
    plt.legend(frameon=False,fontsize=16, loc = 'upper right')
    plt.title(r'Parallel $\mathcal{M}_{A0} \approx$ ' + str(Alfven),fontsize = 28)
    # Add labels
    xmin, xmax, ymin, ymax = plt.axis()
    x_text = 0.9*xmin
    y_text_1 = 0.9*ymax ; y_text_2 = 0.8*ymax ; y_text_3 = 0.7*ymax
    plt.text(x_text, y_text_1, lab_a_par, fontsize = 24)
    plt.text(x_text, y_text_2, lab_s_par, fontsize = 24)
    plt.text(x_text, y_text_3, chi_par, fontsize = 24)

    plt.subplot(2,2,4)
    #plt.plot(x_dim_par,KDE_par , linestyle='--', c='yellow',
    #        alpha=0.99, label=r'KDE', lw = 3, zorder = 2)
    plt.plot(x_dim_par,Prob_par_n0 , linestyle='--', c=purp_m,
            alpha=0.99, label=r'Levy n = 0', lw = 3, zorder = 2)
    plt.plot(x_dim_par, Prob_par_analytic , linestyle='-', c=Levy_col,
            alpha=0.99, label=f'Levy n = Converged', lw = 2)
    #plt.plot(x_dim_par,Prob_par_analytic_2 , linestyle='-.', c=green_m,
    #        alpha=0.99, label=f'Levy n = {finite_val_2}', lw = 3, zorder = 2)
    plt.hist(bins[:-1], bins, weights=counts, density = 'true', histtype='step', 
    edgecolor = Hist_col, label = r'CRIPTIC', linewidth=2)
    plt.legend(frameon=False,fontsize=16, loc = 'upper right')
    plt.xlabel(r'($|\Delta z - \gamma_{\parallel}\cdot t|) / t^{1 / \alpha_{\parallel}}$', fontsize = 26)
    plt.yscale('log')
    plt.savefig('drift_' +  str(Mach) + '_' + str(Alfven) + '_' + str(args.chi)  + "_fitting.png")
    plt.close()
    

    #################################################
    ### Saving outputs
    ### Making results array
    ############################################################
    '''
    # Results arrary, format is: 
    # Mach || Alfven Mach || Ion Fraction || D_par || D_perp || Error_par || Error_perp || Alpha_par || Alpha_perp
    '''
    APar = alpha_par
    APerp = alpha_perp
    ErrorAPar_L = a_par_lo
    ErrorAPar_H = a_par_hi
    ErrorAPerp_L = a_perp_lo
    ErrorAPerp_H = a_perp_hi
    KPar = scale_perp 
    KPerp = scale_par 
    ErrorKPar_L = s_par_lo
    ErrorKPar_H = s_par_hi
    ErrorKPerp_L = s_perp_lo
    ErrorKPerp_H = s_perp_hi
    IonFrac = 1 * 10**(-args.chi)
    results = np.array([args.Mach, args.Alfven, IonFrac, KPar , KPerp , 
    ErrorKPar_L , ErrorKPar_H, ErrorKPerp_L, ErrorKPerp_H,
    APar, APerp,
    ErrorAPar_L , ErrorAPar_H, ErrorAPerp_L, ErrorAPerp_H,])
    
    ################################################
    ### Saving results as txt file
    ################################################
    Filename = 'coefs_' + args.dir + '.txt'
    np.savetxt(Filename, results, delimiter=',')
    ################################################
   
   
