from utilities import *
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm


# fname_resp = 'resp-Si28-14keV.dat'
# fname_resp_mat = 'response_matrix-Si28-14keV.m'
fname_resp = "resp-sun2015.dat"
fname_resp_mat = "response_matrix-sun2015.m"
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

Nbins = len(E_resp_array)

N_draws = 1
N_resp_draws = int(1e5)
# M = 2

np.random.seed(2)

indices_E_resp_array = np.linspace(0,len(E_resp_array)-1, len(E_resp_array)).astype(int)




# Simulate multiplicity 2:
for i in range(Nbins): # Ex index
	Ex = E_resp_array[i]
	for j in range(i): # Eg1 index
		# Draw N_resp_draws samples from response for given Ex, Eg1 (Eg2 is also locked for M=2)
		j_E2 = i - j
		Eg1 = E_resp_array[j]
		Eg2 = E_resp_array[j_E2]
		print("Ex=", Ex, "Eg1 =", Eg1, "Eg2 =", Eg2, flush=True)



		matrix = np.zeros((Nbins, Nbins))

		
		
		# print("R_2D[index_E1,:].sum() =", R_2D[index_E1,:].sum())
		print("j =", j, flush=True)
		E1_folded = np.random.choice(E_resp_array, size=N_resp_draws, p=R_2D[j,:])
		E2_folded = np.random.choice(E_resp_array, size=N_resp_draws, p=R_2D[j_E2,:])
	
		Ex_folded = E1_folded + E2_folded
		
		# print("E1_folded =", E1_folded, flush=True)
		
		for i_resp_draws in range(N_resp_draws):
			i_Ex = np.argmin(np.abs(E_resp_array - Ex_folded[i_resp_draws]))
			i_E1 = np.argmin(np.abs(E_resp_array - E1_folded[i_resp_draws]))
			i_E2 = np.argmin(np.abs(E_resp_array - E2_folded[i_resp_draws]))
			matrix[i_Ex,i_E1] += 1
			matrix[i_Ex,i_E2] += 1
		
		
		
		# print(matrix[matrix>0], flush=True)
		
		N_final = int(len(E_resp_array)/6)
		matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
		
		
		write_mama_2D(matrix_rebinned, "grid_response/response-"+str(i)+"-"+str(j)+".m", E_resp_array_rebinned, E_resp_array_rebinned)

f_max, ax_mat = plt.subplots(1,1)
ax_mat.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
plt.show()




