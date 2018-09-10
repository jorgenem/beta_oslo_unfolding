from utilities import *
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
import sys

np.random.seed(2)

"""
This script simulates events, folds and plots them 
in histograms as function of Eg1, Eg2 etc.

This version from 20180910 draws random multiplicities up to 
M=5 in the true events and M=4 in the folded (this is SuN's 
max detected multiplicity.), and saves both to file.

"""

# Global settings:
pileup = False # Choose whether to simulate the effects of detector pileup
N_events = 5000 # Number of events to simulate

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




# === Unfold the events ===








