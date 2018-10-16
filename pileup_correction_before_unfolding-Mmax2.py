from utilities import *
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
import sys, time

from ROOT import gRandom, TH1, TH2, TH1D, TH2D, cout, gROOT, TCanvas, TLegend, RooUnfoldResponse, RooUnfoldBayes

np.random.seed(2)

"""
This script simulates events, folds and plots them 
in histograms as function of Eg1, Eg2 etc.

This version, started 20180910, draws random multiplicities up to 
M=Mt in the true events and M=Mf in the folded (Mf=4 is SuN's 
max detected multiplicity.), and saves both to file.

It attempts to simulate pileup effects by randomly combining gammas 
from the true events. All gammas are folded just once by a
random draw from the response function.

"""

# Global settings:
pileup = True # Choose whether to simulate the effects of detector pileup
p_pile = 0.2 # Pileup probability per gamma ray

N_events = int(3*1e4) # Number of events to simulate

Mt_max = 2#5 # Max true multiplicity
Mf_max = 2 # Max detector multiplicity


customLogNorm = LogNorm(vmin=1e0, vmax=N_events/10)


# == Read and set up response matrix ==

fname_resp = 'resp-SuN2015-20keV-1p0FWHM.dat'
fname_resp_mat = 'response_matrix-SuN2015-20keV-1p0FWHM.m'
# fname_resp = 'resp-SuN2015-50keV-1p0FWHM.dat'
# fname_resp_mat = 'response_matrix-SuN2015-50keV-1p0FWHM.m'
R_2D, cal_resp, E_resp_array, tmp = read_mama_2D(fname_resp_mat)
# R_2D = div0(R_2D , R_2D.sum(rebin_axis=1))

# Read efficiency and other 1-D response variables:
resp = []
with open(fname_resp) as file:
    # Read line by line as there is crazyness in the file format
    lines = file.readlines()
    for i in range(4,len(lines)):
        try:
            row = np.array(lines[i].split(), dtype="double")
            resp.append(row)
        except:
            break

resp = np.array(resp)
# Name the columns for ease of reading
# FWHM = resp[:,1]
eff = resp[:,2]
# pf = resp[:,3]
# pc = resp[:,4]
# ps = resp[:,5]
# pd = resp[:,6]
# pa = resp[:,7]



Emax = 4000 # Crop response matrix (and thus also event generation) to efficiate computation
i_Emax = np.argmin(np.abs(E_resp_array-Emax))
R_2D = R_2D[0:i_Emax,0:i_Emax]
E_resp_array = E_resp_array[0:i_Emax]



# Assumed lower threshold for gammas in response matrix
E_thres = 100
i_thres = np.argmin(np.abs(E_resp_array - E_thres))
R_2D[:,:i_thres] = 0

for i in range(R_2D.shape[0]):
    norm = R_2D[i,:].sum()
    if(norm>0):
        R_2D[i,:] = R_2D[i,:] / norm #* eff[i]
    else:
        R_2D[i,:] = 0





# === Generate events ===
fname_ev_t = "generated_events-true-Emax2MeV-Mtmax2-{:d}_events-pileup_is_{:s}.npy".format(N_events, "on" if pileup else "off")
try:
    events_t = np.load(fname_ev_t)
except:
    events_t = np.zeros((N_events,Mt_max))
    Eg_gaussian_centroids = np.array([1700,1300,500,1000])
    for i_ev in range(N_events):
        # Mt_curr = np.random.randint(low=1,high=(Mt_max+1))
        Mt_curr = 2 # Testing, comparison to the first test we did with 2 gammas for the report
        # Egs_current = np.random.uniform(low=0, high=Emax, size=Mt_curr)
        Egs_current = np.random.normal(loc=Eg_gaussian_centroids[0:Mt_curr], scale=0.5*np.sqrt(Eg_gaussian_centroids[0:Mt_curr]), size=Mt_curr)
        events_t[i_ev,0:Mt_curr] = Egs_current
    np.save(fname_ev_t, events_t)



print("Events true:", flush=True)
print(events_t, flush=True)




# === Fold them with detector response and pileup: ===
def FoldEg(Egs_t, Mf_max, Eg_arr, response, pileup=True, p_pile=0.2):
    """
    This function folds all Egs from a single event.
    """

    
    # print("Folding. True gammas =", Egs_t, flush=True)
    Mt_curr = len(Egs_t[Egs_t>0]) # Number of true gammas

    
    Mf_curr = Mt_curr # Folded multiplicity is the same as true because we do pileup after energy folding

    # Now proceed to fold each gamma with detector response:
    Egs_folded = np.zeros(Mf_curr)
    for i in range(Mf_curr):
        Eg = Egs_t[i]
        index_Eg = np.argmin(np.abs(Eg_arr - Eg))
        if R_2D[index_Eg,:].sum() > 0:
            # choosing rand accounts for the efficiency; As the efficiency read from file currently
            # does not always correspons with the counts in R_2D, see #3, we need two if tests
            rand = np.random.uniform()
            if rand <= eff[index_Eg]:
                # If the gamma is not lost to efficiency, redistribute its energy somewhere in the response:
                Eg_folded = np.random.choice(Eg_arr, p=response[index_Eg,:])
            else: 
                Eg_folded = 0 # Give Energy 0 to events that are not recorded.
        else: 
            Eg_folded = 0 # Give Energy 0 to events that are not recorded (below detector threshold)

        Egs_folded[i] = Eg_folded


    if pileup: # Is pileup desired?
        # For each folded gamma, assign it to pileup with probability p_pile:
        indices_pile = [] # Store indices in Egs_folded
        counter_nopile = 0
        map_to_pileup = {} # Map the indices that are *not* pileup to a new index set so that there are no index holes
                        # For example if Mt = 3 and index 1 is pileup, then map_nopile = {0:0, 2:1}.
                        # Then we decide which of the new indices to add each pileup event to.
        for i_t in range(Mt_curr):
            # Make sure at least one gamma is not assigned to pileup:
            if i_t == Mt_curr-1 and counter_nopile == 0:
                map_to_pileup[i_t] = counter_nopile
                counter_nopile += 1
                break

            # Also make sure no more than 4 gammas remain, by
            # setting pileup probability to 1 if we already have 4 non-piled
            if counter_nopile == 4:
                p_pile = 1

            r_pile = np.random.uniform()
            if r_pile < p_pile:
                # This gamma should be piled.
                indices_pile.append(i_t)
            else:
                # This gamma should not
                map_to_pileup[i_t] = counter_nopile
                counter_nopile += 1

        # Calculate multiplicity of current event after pileup:
        Mf_curr = Mt_curr - len(indices_pile)

        # Decide mapping of pileup gammas:
        for i_t in indices_pile:
            map_to_pileup[i_t] = np.random.randint(0,Mf_curr) if Mf_curr > 0 else 0

        Egs_piled = np.zeros(Mf_curr)
        for i_t in range(Mt_curr):
            Egs_piled[map_to_pileup[i_t]] += Egs_folded[i_t]

    else:
        # Do not include pileup:
        Egs_piled = Egs_folded
        # Mf_curr = Mt_curr

    # print("Piled gammas =", Egs_piled, flush=True)

    return Egs_piled




fname_ev_f = "generated_events-folded-Emax2MeV-Mtmax2-{:d}_events-pileup_is_{:s}.npy".format(N_events, "on" if pileup else "off")
try:
    events_f = np.load(fname_ev_f)
except:
    events_f = np.zeros((N_events, Mf_max))
    for i_ev in range(N_events):
        Egs_folded = FoldEg(events_t[i_ev,:], Mf_max=4, Eg_arr=E_resp_array, response=R_2D, pileup=pileup, p_pile=p_pile)
        events_f[i_ev,0:len(Egs_folded)] = Egs_folded
    np.save(fname_ev_f, events_f)

print("Events folded:", flush=True)
print(events_f, flush=True)


N_Eg = len(E_resp_array)



# # === Plot true and folded matrices as Ex-Eg ===
# f_ExEg, (ax_ExEg_true, ax_ExEg_folded) = plt.subplots(2,1)
# N_Ex = Mt_max*N_Eg
# Ex_array = np.linspace(E_resp_array[0], Mt_max*E_resp_array[-1], N_Ex)

# matrix_ExEg_true = np.zeros((N_Ex,N_Eg))
# matrix_ExEg_folded = np.zeros((N_Ex,N_Eg))
# for i_ev in range(N_events):
#     Egs_t = events_t[i_ev]
#     Ex_t = Egs_t.sum()
#     for Eg_t in Egs_t[Egs_t>0]:
#         matrix_ExEg_true[np.argmin(np.abs(Ex_array-Ex_t)), np.argmin(np.abs(E_resp_array-Eg_t))] += 1

#     Egs_f = events_f[i_ev]
#     Ex_f = Egs_f.sum()
#     for Eg_f in Egs_f[Egs_f>0]:
#         matrix_ExEg_folded[np.argmin(np.abs(Ex_array-Ex_f)), np.argmin(np.abs(E_resp_array-Eg_f))] += 1


# cbar_ExEg_true = ax_ExEg_true.pcolormesh(E_resp_array, Ex_array, matrix_ExEg_true, norm=customLogNorm, cmap="jet")
# cbar_ExEg_folded = ax_ExEg_folded.pcolormesh(E_resp_array, Ex_array, matrix_ExEg_folded, norm=customLogNorm, cmap="jet")
# f_ExEg.colorbar(cbar_ExEg_true, ax=ax_ExEg_true)
# f_ExEg.colorbar(cbar_ExEg_folded, ax=ax_ExEg_folded)









# === Sort events along Eg axes ===

# TODO implement some kind of pileup correction algorithm.
# For now either neglect pileup or keep it turned off.


# We have to sort the data according to Eg0, Eg1 etc along all axes.
# When pileup correction is done, we have to assume some max multiplicity perhaps.
# For now, do the first test with M=4.


# We have the option to try variable bin sizes for the different unfolding axes.
# If we sort the axes so that axis 0 is the most populated (all events), 
# 1 the second most (all events with M>=2), etc., then the axes
# will quickly become more sparsely populated.
# I speculate that this can be amended by rebinning the last axes harder,
# to increase statistics. It is simple enough to try different strategies and see what works best.

# I am also thinking that we should try some kind of "bootstrap"-ish method to
# avoid biasing things by always selecting certain gammas to lie along the first axis, etc.
# If we do several unfoldings with random orderings among the gammas in each event, we can 
# gauge the potential impact of this.



# == Sort data into Mf_max or Mu_max-dimensional array ==
# Choose dimensionality and allocate arrays
dim_folded = (N_Eg, N_Eg)
E0f_array = np.linspace(E_resp_array[0], E_resp_array[-1], dim_folded[0])
E1f_array = np.linspace(E_resp_array[0], E_resp_array[-1], dim_folded[1])
counts_folded = np.zeros(dim_folded)
print("counts_folded.size =", counts_folded.size*8/1024**2, "MB", flush=True)

Mu_max = Mf_max # Max multiplicity of events after unfolding. Set it equal to Mf_max until pileup correction is implemented.
dim_unfolded = (N_Eg, N_Eg)
E0u_array = np.linspace(E_resp_array[0], E_resp_array[-1], dim_unfolded[0])
E1u_array = np.linspace(E_resp_array[0], E_resp_array[-1], dim_unfolded[1])
# Don't allocate this matrix yet, it needs one copy per unfolding axis anyway
# counts_unfolded = np.zeros(dim_unfolded)
# print("counts_unfolded.size =", counts_unfolded.size*8/1024**2, "MB", flush=True)

# Also sort true events with same binning as unfolded for comparison
counts_true = np.zeros(dim_unfolded)

# Sort data:
t_s = time.time()
for i_ev in range(N_events):
    # Mf=2:
    counts_folded[np.argmin(np.abs(E0f_array-events_f[i_ev,0])),np.argmin(np.abs(E1f_array-events_f[i_ev,1]))] += 1
    counts_true[np.argmin(np.abs(E0u_array-events_t[i_ev,0])),np.argmin(np.abs(E1u_array-events_t[i_ev,1]))] += 1
    # Mf=4: (TODO fix this hack to automate handling of all Mt, Mf choices)
    # counts_folded[np.argmin(np.abs(E0f_array-events_f[i_ev,0])),np.argmin(np.abs(E1f_array-events_f[i_ev,1])),np.argmin(np.abs(E2f_array-events_f[i_ev,2])),np.argmin(np.abs(E3f_array-events_f[i_ev,3]))] += 1
    # counts_true[np.argmin(np.abs(E0u_array-events_t[i_ev,0])),np.argmin(np.abs(E1u_array-events_t[i_ev,1])),np.argmin(np.abs(E2u_array-events_t[i_ev,2])),np.argmin(np.abs(E3u_array-events_t[i_ev,3]))] += 1
t_f = time.time()
print("Event sorting took {:.1f} s".format(t_f-t_s), flush=True)





# === Plotting by Eg axes ===
fEgEg, ((axEgEg0, axEgEg1), (axEgEg2, axEgEg3)) = plt.subplots(2,2)
# sum_axes = (2,3) # We can only plot 2D, so we sum the other axes
E_plot_x = E0f_array
E_plot_y = E1f_array

# True spectr:
cbar_EgEg0 = axEgEg0.pcolormesh(E_plot_x, E_plot_y, counts_true, norm=customLogNorm, cmap="jet")
axEgEg0.set_title("True")
fEgEg.colorbar(cbar_EgEg0, ax=axEgEg0)
# Folded spectr:
cbar_EgEg1 = axEgEg1.pcolormesh(E_plot_x, E_plot_y, counts_folded, norm=customLogNorm, cmap="jet")
axEgEg1.set_title("Folded")
fEgEg.colorbar(cbar_EgEg1, ax=axEgEg1)
# # Unfolded axis 0:
# cbar_EgEg2 = axEgEg2.pcolormesh(E_plot_x, E_plot_y, counts_unfolded0, norm=customLogNorm, cmap="jet")
# axEgEg2.set_title("Unfolded axis 0")
# fEgEg.colorbar(cbar_EgEg2, ax=axEgEg2)
# # Unfolded axis 1:
# cbar_EgEg3 = axEgEg3.pcolormesh(E_plot_x, E_plot_y, counts_unfolded1, norm=customLogNorm, cmap="jet")
# axEgEg3.set_title("Unfolded axis 1")
# fEgEg.colorbar(cbar_EgEg3, ax=axEgEg3)
# # # Unfolded axis 2:
# cbar_EgEg4 = axEgEg4.pcolormesh(E_plot_x, E_plot_y, counts_unfolded2.sum(axis=sum_axes), norm=customLogNorm, cmap="jet")
# axEgEg4.set_title("Unfolded axis 2")
# fEgEg.colorbar(cbar_EgEg4, ax=axEgEg4)

plt.show()


# TODO add ExEg plotting of unfolded events.