from utilities import *
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
import sys

np.random.seed(2)

customLogNorm = LogNorm(vmin=1e0, vmax=3e1)

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
pileup = False # Choose whether to simulate the effects of detector pileup
N_events = int(5*1e4) # Number of events to simulate

Mt_max = 4#5 # Max true multiplicity
Mf_max = 4 # Max detector multiplicity



# == Read and set up response matrix ==

fname_resp = 'resp-SuN2015-20keV-1p0FWHM.dat'
fname_resp_mat = 'response_matrix-SuN2015-20keV-1p0FWHM.dat'
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



Emax = 2000 # Crop response matrix (and thus also event generation) to efficiate computation
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
fname_ev_t = "generated_events-true-Emax2MeV-{:d}_events-pileup_is_{:s}.npy".format(N_events, "on" if pileup else "off")
try:
	events_t = np.load(fname_ev_t)
except:
	events_t = np.zeros((N_events,Mt_max))
	for i_ev in range(N_events):
		Mt_curr = np.random.randint(low=1,high=(Mt_max+1))
		Egs_current = np.random.uniform(low=0, high=Emax, size=Mt_curr)
		events_t[i_ev,0:Mt_curr] = Egs_current
	np.save(fname_ev_t, events_t)



print("Events true:", flush=True)
print(events_t, flush=True)




# === Fold them with detector response and pileup: ===
def FoldEg(Egs_t, Mf_max, Eg_arr, response, pileup=True):
	"""
	This function folds all Egs from a single event.
	"""

	
	# print("Folding. True gammas =", Egs_t, flush=True)
	Mt_curr = len(Egs_t[Egs_t>0]) # Number of true gammas

	if pileup: # Is pileup desired?
		# For each true gamma, assign it to pileup with probability p_pile:
		p_pile = 0.2
		indices_pile = [] # Store indices in Egs_true
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
			Egs_piled[map_to_pileup[i_t]] += Egs_t[i_t]

	else:
		# Do not include pileup:
		Egs_piled = Egs

	# print("Piled gammas =", Egs_piled, flush=True)


	# Now proceed to fold each gamma with detector response:
	Egs_folded = np.zeros(Mf_curr)
	for i in range(Mf_curr):
		Eg = Egs_piled[i]
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

	return Egs_folded




fname_ev_f = "generated_events-folded-Emax2MeV-{:d}_events-pileup_is_{:s}.npy".format(N_events, "on" if pileup else "off")
try:
	events_f = np.load(fname_ev_f)
except:
	events_f = np.zeros((N_events, Mf_max))
	for i_ev in range(N_events):
		Egs_folded = FoldEg(events_t[i_ev,:], Mf_max=4, Eg_arr=E_resp_array, response=R_2D)
		events_f[i_ev,0:len(Egs_folded)] = Egs_folded
	np.save(fname_ev_f,	events_f)

print("Events folded:", flush=True)
print(events_f)





# === Plot true and folded matrices as Ex-Eg ===
f_ExEg, (ax_ExEg_true, ax_ExEg_folded) = plt.subplots(2,1)
N_Eg = len(E_resp_array)
N_Ex = Mt_max*N_Eg
Ex_array = np.linspace(E_resp_array[0], Mt_max*E_resp_array[-1], N_Ex)

matrix_ExEg_true = np.zeros((N_Ex,N_Eg))
matrix_ExEg_folded = np.zeros((N_Ex,N_Eg))
for i_ev in range(N_events):
	Egs_t = events_t[i_ev]
	Ex_t = Egs_t.sum()
	for Eg_t in Egs_t[Egs_t>0]:
		matrix_ExEg_true[np.argmin(np.abs(Ex_array-Ex_t)), np.argmin(np.abs(E_resp_array-Eg_t))] += 1

	Egs_f = events_f[i_ev]
	Ex_f = Egs_f.sum()
	for Eg_f in Egs_f[Egs_f>0]:
		matrix_ExEg_folded[np.argmin(np.abs(Ex_array-Ex_f)), np.argmin(np.abs(E_resp_array-Eg_f))] += 1

cbar_ExEg_true = ax_ExEg_true.pcolormesh(E_resp_array, Ex_array, matrix_ExEg_true, norm=customLogNorm)
cbar_ExEg_folded = ax_ExEg_folded.pcolormesh(E_resp_array, Ex_array, matrix_ExEg_folded, norm=customLogNorm)
f_ExEg.colorbar(cbar_ExEg_true, ax=ax_ExEg_true)
f_ExEg.colorbar(cbar_ExEg_folded, ax=ax_ExEg_folded)

plt.show()








# === Unfold the events ===

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
dim_folded = (N_Eg, N_Eg, int(N_Eg/2), int(N_Eg/2))
tensor_folded = np.zeros(dim_folded)
print("tensor_folded.size =", tensor_folded.size*8/1024**2, "MB", flush=True)

Mu_max = Mf_max # Max multiplicity of events after unfolding. Set it equal to Mf_max until pileup correction is implemented.
dim_unfolded = (N_Eg, N_Eg, int(N_Eg/2), int(N_Eg/2))
tensor_unfolded = np.zeros(dim_unfolded)
print("tensor_unfolded.size =", tensor_unfolded.size*8/1024**2, "MB", flush=True)

# Sort data:
for i_ev in range(N_events):
	tensor_folded[np.argmin(np.abs())]

from ROOT import RooUnfold

# == Axis 0 ==





