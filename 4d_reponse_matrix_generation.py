from utilities import *
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm


fname_resp = 'resp-Si28-14keV.dat'
fname_resp_mat = 'response_matrix-Si28-14keV.m'
R_2D, cal_resp, E_resp_array, tmp = read_mama_2D(fname_resp_mat)
# R_2D = div0(R_2D , R_2D.sum(rebin_axis=1))
E_thres = 100
i_thres = np.argmin(np.abs(E_resp_array - E_thres))
R_2D[:,:i_thres] = 0
for i in range(R_2D.shape[0]):
	try:
		R_2D[i,:] = R_2D[i,:] / R_2D[i,:].sum()
	except:
		R_2D[i,:] = 0


# f_cmp, ax_cmp = plt.subplots(1,1)
# ax_cmp.plot(E_resp_array, R_2D[400,:])



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




# plt.pcolormesh(E_resp_array, E_resp_array, R_2D, norm=LogNorm())
# sys.exit(0)


Emin = 100
Emax = 10*1e3

N_draws = 15
N_resp_draws = int(1e4)

np.random.seed(2)

def CalcResponse(E1s, E2s=None, E3s=None, E4s=None, E_resp_array=E_resp_array, N_resp_draws=N_resp_draws, response=R_2D):
	# find the multiplicity
	M = 1
	if E2s is not None: M = 2 
	if E3s is not None: M = 3
	if E4s is not None: M = 4

	Nbins = len(E_resp_array)
	matrix = np.zeros((Nbins, Nbins))
	indices_E_resp_array = np.linspace(0,Nbins-1, Nbins).astype(int)

	def FoldEg(Eg, Eg_arr=E_resp_array, size=N_resp_draws, response=response):
		index_Eg = np.argmin(np.abs(Eg_arr - Eg))
		Eg_folded = np.random.choice(Eg_arr, size=size, p=response[index_Eg,:])
		return Eg_folded

	# find the response and Ex for each gamma ray in N_draws
	for i_draw in range(N_draws):
		Eg_folded_arr = np.zeros((M,N_resp_draws))
		Eg_folded_arr[0,:] = FoldEg(E1s[i_draw])
		if E2s is not None: Eg_folded_arr[1,:] = FoldEg(E2s[i_draw])
		if E3s is not None: Eg_folded_arr[2,:] = FoldEg(E3s[i_draw])
		if E4s is not None: Eg_folded_arr[3,:] = FoldEg(E4s[i_draw])
		# print(Eg_folded_arr)

		# Ex calculated as the sum over Egs
		Ex_folded = np.sum(Eg_folded_arr,axis=0)
		# print(Ex_foldZed)

		# fill the matrix
		for i_resp_draws in range(N_resp_draws):
			i_Ex = np.argmin(np.abs(E_resp_array - Ex_folded[i_resp_draws]))
			for Eg in Eg_folded_arr:
				i_Eg = np.argmin(np.abs(E_resp_array - Eg[i_resp_draws]))
				matrix[i_Ex,i_Eg] += 1

	return matrix

E1s = 1*1e3*np.ones(N_draws) # E1 = 1 MeV
# E1s = np.random.uniform(low=Emin, high=Emax, size=N_draws)
# E1s = np.random.triangular(left=Emin, mode=(Emax+Emin)/2, right=Emax, size=N_draws)
# E2s = np.random.uniform(low=Emin, high=Emax, size=N_draws)
E2s = Emax - E1s
matrix = CalcResponse(E1s,E2s)
write_mama_2D(matrix_rebinned, "folded_2D_2gammas1and9MeV.m", E_resp_array_rebinned, E_resp_array_rebinned)


# print(matrix[matrix>0], flush=True)

N_final = int(len(E_resp_array)/6)
matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)


#write_mama_2D(matrix_rebinned, "folded_2D_2gammas1and9MeV.m", E_resp_array_rebinned, E_resp_array_rebinned)

f_max, ax_mat = plt.subplots(1,1)
ax_mat.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
plt.show()




