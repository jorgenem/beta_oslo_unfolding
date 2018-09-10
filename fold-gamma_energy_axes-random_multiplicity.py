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



Emax = 2000 # Crop response matrix (and thus also event generation) to efficiate computation
i_Emax = np.argmin(np.abs(E_resp_array-Emax))
R_2D = R_2D[0:i_Emax,0:i_Emax]
E_resp_array = E_resp_array[0:i_Emax]


# === Generate events ===
N_events = 5
fname_ev_t = "generated_events-true-Emax2MeV-{:d}_events.npy".format(N_events)
Mt_max = 5 # Max true multiplicity
try:
	events_t = np.load(fname_ev_t)
except:
	events_t = np.zeros((N_events,Mt_max))
	for i_ev in range(N_events):
		Mt_curr = np.random.randint(low=1,high=(Mt_max+1))
		Egs_current = np.random.uniform(low=0, high=Emax, size=Mt_curr)
		events_t[i_ev,0:Mt_curr] = Egs_current
	np.save(fname_ev_t, events_t)



print("events_t =", events_t)




# === Fold them with detector response and pileup: ===
def FoldEg(Egs, Mf_max, Eg_arr, response, pileup=True):
	"""
	This function folds all Egs from a single event.
	"""

	
	print("Folding. True gammas =", Egs, flush=True)
	

	if pileup and Mt_curr > Mf_curr:
		# First decide the detector multiplicity for this event. It's bounded upwards by Mf (4 in case of SuN)
		# A first approx. is a linearly increasing probability from 0 to true multiplicity for current event, 
		# but not higher than Mf_max.
		print("Deciding multiplicity", flush=True)
		Mt_curr = len(Egs[Egs>0])
		Mf_curr = int(min(Mf_max, np.random.triangular(1,Mt_curr,Mt_curr))) if Mt_curr > 1 else 1 # Don't make any events with multiplicity zero. We can model the detector efficiency elsewhere.
		print("Mt_curr =", Mt_curr, "Mf_curr =", Mf_curr, flush=True)

		# Decide which true gammas come in the same detector.
		# Do this by drawing random pairs of indices 
		Mf_tmp = Mt_curr
		indices_combined = {} # Create a dictionary of indices stored as key, value pairs to make mapping easy
		indices_used = []
		print("Deciding gamma combinations", flush=True)
		while Mf_tmp > Mf_curr:
			indices_combined_curr = [0,0]
			# Make sure we combine two different gammas, and that they have not already been combined.
			while indices_combined_curr[0] == indices_combined_curr[1] or indices_combined_curr[0] in indices_used or indices_combined_curr[1] in indices_used:
				indices_combined_curr = np.random.randint(0,Mt_curr+1),np.random.randint(0,Mt_curr+1)
				# Make sure key is lower than value, so that we map both to the lowest index 
				# (it's just easier to code below):
				if indices_combined_curr[0] > indices_combined_curr[1]:
					indices_combined_curr = [indices_combined_curr[1],indices_combined_curr[0]]
			indices_combined[indices_combined_curr[0]] = indices_combined_curr[1]
			Mf_tmp -= 1

		# Combine gammas:
		print("Piling gammas", flush=True)
		Egs_piled = np.zeros(Mf_curr)
		i_piled = 0
		indices_used = []
		while len(indices_used) < Mt_curr:
			if not i_piled in indices_used:
				try: # If index is one of the key, value pairs:
					i1 = i_piled
					i2 = indices_combined[i_piled]
					print("i_piled =", i_piled, "i1 =", i1, "i2 =", i2, flush=True)
					Egs_piled[i_piled] = Egs[i1] + Egs[i2]
					indices_used.append(i1)
					indices_used.append(i2)
					i_piled += 1
				except KeyError: # Not one of the key, value pairs
					print("i_piled =", i_piled, flush=True)
					Egs_piled[i_piled] = Egs[i_piled]
					indices_used.append(i_piled)
					i_piled += 1
			else:
				i_piled += 1
			print("Egs_piled =", Egs_piled, flush=True)
	else:
		# Do not model pileup:
		Egs_piled = Egs



	
	return Egs_piled


	# for i_Eg in range(Mt_curr)
	# if R_2D[index_Eg,:].sum() > 0:
	# index_Eg = np.argmin(np.abs(Eg_arr - Eg))
	# # choosing rand accounts for the efficiency; As the efficiency read from file currently
	# # does not always correspons with the counts in R_2D, see #3, we need two if tests
	# 	rand = np.random.uniform()
	# 	if rand <= eff[index_Eg]:
	# 		Eg_folded = np.random.choice(Eg_arr, size=size, p=response[index_Eg,:])
	# 	else: 
	# 		Eg_folded = 0 # Give Energy 0 to events that are not recorded.
	# else: 
	# 	Eg_folded = 0 # Give Energy 0 to events that are not recorded (below detector threshold)
	# return Egs_folded


Mf_max = 4 # Max detector multiplicity



for i_ev in range(N_events):
	Egs_folded = FoldEg(events_t[i_ev,:], Mf_max=4, Eg_arr=E_resp_array, response=R_2D)
	print("Egs_folded[{:d}] =".format(i_ev), Egs_folded, flush=True)


sys.exit(0)






# plt.pcolormesh(E_resp_array, E_resp_array, R_2D, norm=LogNorm())
# sys.exit(0)

def CalcResponseEgaxes(E1s, E2s=None, E3s=None, E4s=None, E_resp_array=None, N_resp_draws=None, response=None):
	# find the multiplicity
	M = 1
	if E2s is not None: M = 2
	if E3s is not None: M = 3
	if E4s is not None: M = 4

	Nbins = len(E_resp_array)
	matrix = np.zeros(tuple([Nbins for i in range(M)])) # Make an M-dimensional array
	indices_E_resp_array = np.linspace(0,Nbins-1, Nbins).astype(int)

	def FoldEg(Eg, Eg_arr=E_resp_array, size=N_resp_draws, response=response):
		index_Eg = np.argmin(np.abs(Eg_arr - Eg))
		# choosing rand accounts for the efficiency; As the efficiency read from file currently
		# does not always correspons with the counts in R_2D, see #3, we need two if tests
		if R_2D[index_Eg,:].sum() > 0:
			rand = np.random.uniform()
			if rand <= eff[index_Eg]:
				Eg_folded = np.random.choice(Eg_arr, size=size, p=response[index_Eg,:])
			else: 
				Eg_folded=np.nan # Give Energy 0 to events that are not recorded.
		else: 
			Eg_folded = np.nan # Give Energy 0 to events that are not recorded.
		return Eg_folded

	print("working on response matrix")

	# find the response and Ex for each gamma ray in N_draws
	for i_draw in range(N_draws):
		Eg_folded_arr = np.zeros((M,N_resp_draws))
		Eg_folded_arr[0,:] = FoldEg(E1s[i_draw])
		if E2s is not None: Eg_folded_arr[1,:] = FoldEg(E2s[i_draw])
		if E3s is not None: Eg_folded_arr[2,:] = FoldEg(E3s[i_draw])
		if E4s is not None: Eg_folded_arr[3,:] = FoldEg(E4s[i_draw])
		# print(Eg_folded_arr)

		# Ex calculated as the sum over Egs
		# Ex_folded = np.sum(np.nan_to_num(Eg_folded_arr),axis=0)
		# print(Ex_foldZed)

		# fill the matrix
		for i_resp_draws in range(N_resp_draws):
			# i_Ex = np.argmin(np.abs(E_resp_array - Ex_folded[i_resp_draws]))
			Egs = np.nan_to_num(Eg_folded_arr[:,i_resp_draws])
			# Egs[::-1].sort()
			if M == 2:
				i_Eg1 = np.argmin(np.abs(E_resp_array - Egs[0]))
				i_Eg2 = np.argmin(np.abs(E_resp_array - Egs[1]))
				matrix[i_Eg1, i_Eg2] += 1
			# print("Egs =", Egs, flush=True) 
			# indices_Eg = np.argmin(np.abs(np.tile(E_resp_array,M).reshape(Nbins,M) - Egs), axis=0)
			# print("indices_Eg =", indices_Eg, flush=True)
			# matrix[tuple(indices_Eg)] += 1
	print ("Finished repsonse matrix")
	return matrix

Emax = 10*1e3



# writing results to mama
N_draws = 500
N_resp_draws = int(1e3)
defaults = {
	"E_resp_array": E_resp_array,
	"N_resp_draws": N_resp_draws,
	"response": R_2D
}
# Draw some energy pairs:
# E1s = np.random.uniform(low=0, high=Emax, size=N_draws) # uniform distribution of E1s
# E2s = np.random.uniform(low=0, high=Emax, size=N_draws) # uniform distribution of E2s
E1s = np.random.normal(loc=1700, scale=70, size=N_draws) # uniform distribution of E1s
E2s = np.random.normal(loc=1300, scale=50, size=N_draws) # uniform distribution of E2s
N_final = int(len(E_resp_array)/1)
# Calculate, rebin and write folded spectrum:
matrix = CalcResponseEgaxes(E1s,E2s,**defaults)
# matrix_rebinned, E_resp_array_rebinned = rebin_and_shift_memoryguard(rebin_and_shift_memoryguard(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
# write_mama_2D(matrix_rebinned, "folded_2D_2gammas-Egaxes.m", E_resp_array_rebinned, E_resp_array_rebinned)
write_mama_2D(matrix, "folded_2D_2gammas-Egaxes.m", E_resp_array, E_resp_array)
# Write true spectrum (rebinned):
# matrix_true = np.zeros((N_final,N_final))
matrix_true = np.zeros((len(E_resp_array),len(E_resp_array)))

for E1, E2 in zip(E1s, E2s):
	i_Eg1 = np.argmin(np.abs(E_resp_array - E1))
	i_Eg2 = np.argmin(np.abs(E_resp_array - E2))
	matrix_true[i_Eg1,i_Eg2] += 1

write_mama_2D(matrix_true, "truth_2D_2gammas-Egaxes.m", E_resp_array, E_resp_array)
# Plot both matrices
f_mama, (ax_mama1, ax_mama2) = plt.subplots(2,1)
ax_mama1.pcolormesh(E_resp_array, E_resp_array, matrix_true, norm=LogNorm())
ax_mama2.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())



plt.show()



