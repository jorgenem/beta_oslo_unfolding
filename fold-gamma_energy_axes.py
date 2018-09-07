from utilities import *
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm


"""
This script simulates events, folds and plots them 
in histograms as function of Eg1, Eg2 etc.

"""


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
FWHM = resp[:,1]
eff = resp[:,2]
pf = resp[:,3]
pc = resp[:,4]
ps = resp[:,5]
pd = resp[:,6]
pa = resp[:,7]

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

np.random.seed(2)
Emax = 10*1e3



# writing results to mama
N_draws = 30
N_resp_draws = int(1e4)
defaults = {
	"E_resp_array": E_resp_array,
	"N_resp_draws": N_resp_draws,
	"response": R_2D
}
# Draw some energy pairs:
E1s = np.random.uniform(low=0, high=Emax, size=N_draws) # uniform distribution of E1s
E2s = np.random.uniform(low=0, high=Emax, size=N_draws) # uniform distribution of E2s
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



