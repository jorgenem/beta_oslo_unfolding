from utilities import *
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm


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

def CalcResponse(E1s, E2s=None, E3s=None, E4s=None, E_resp_array=None, N_resp_draws=None, response=None):
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
		Ex_folded = np.sum(np.nan_to_num(Eg_folded_arr),axis=0)
		# print(Ex_foldZed)

		# fill the matrix
		for i_resp_draws in range(N_resp_draws):
			i_Ex = np.argmin(np.abs(E_resp_array - Ex_folded[i_resp_draws]))
			for Eg in Eg_folded_arr[:,i_resp_draws]:
				if np.isfinite(Eg):
					i_Eg = np.argmin(np.abs(E_resp_array - Eg))
					matrix[i_Ex,i_Eg] += 1
	print ("Finished repsonse matrix")
	return matrix

np.random.seed(2)
Emin = 100
Emax = 10*1e3

DoPlottingM1 = False
DoPlottingM2 = False
DoPlottingM3 = False
DoPlottingM4 = True
write_mama_2D = False

if write_mama_2D:
	# writing results to mama
	N_draws = 1
	N_resp_draws = int(1e4)
	defaults = {
		"E_resp_array": E_resp_array,
		"N_resp_draws": N_resp_draws,
		"response": R_2D
	}
	matrix = CalcResponse(E1s,E2s,**defaults)
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	write_mama_2D(matrix_rebinned, "folded_2D_2gammas1and9MeV.m", E_resp_array_rebinned, E_resp_array_rebinned)


# defaults for the next round of plotting
N_draws = 15 # number of differnt incident gammas dran
N_resp_draws = int(1e4) # number of draws from response function for each incident gamma
defaults = {
	"E_resp_array": E_resp_array,
	"N_resp_draws": N_resp_draws,
	"response": R_2D
}
if DoPlottingM1:
	# create plots
	f_max, ax_mat = plt.subplots(2,3,figsize=(20,15))

	# subplot
	ax = ax_mat[0,0]
	E1s = 1*1e3*np.ones(N_draws) # E1 = xx MeV
	matrix = CalcResponse(E1s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = 1 MeV")

	# subplot
	ax = ax_mat[0,1]
	E1s = 3.5*1e3*np.ones(N_draws) # E1 = xx MeV
	matrix = CalcResponse(E1s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = 3.5 MeV")

	# subplot
	ax = ax_mat[0,2]
	E1s = 5*1e3*np.ones(N_draws) # E1 = xx MeV
	matrix = CalcResponse(E1s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = 5 MeV")

	# subplot
	ax = ax_mat[1,0]
	E1s = np.random.uniform(low=Emin, high=Emax, size=N_draws) # uniform distribution of E1s
	matrix = CalcResponse(E1s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = uniform")

	# subplot
	ax = ax_mat[1,1]
	Emid = (Emax+Emin)/2
	E1s = np.random.triangular(left=Emin, mode=Emid, right=Emax, size=N_draws) # E1 = 1 MeV
	matrix = CalcResponse(E1s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = triangle (Emid={:.2f}) MeV".format(Emid/1e3))

	# subplot
	ax = ax_mat[1,2]
	Emid = (Emax)
	E1s = np.random.triangular(left=Emin, mode=Emid, right=Emax, size=N_draws) # E1 = 1 MeV
	matrix = CalcResponse(E1s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = triangle (Emid={:.2f}) MeV".format(Emid/1e3))

	for ax in ax_mat.flatten():
		ax.set_xlabel("Eg [keV]")
		ax.set_ylabel(r"\sum Eg = Ex [keV]")

	plt.tight_layout()
	plt.subplots_adjust(top=0.93)
	plt.suptitle("Multiplicity 1")
	plt.savefig("resp_M1.png")
	plt.show()


if DoPlottingM2:
	# create plots
	f_max, ax_mat = plt.subplots(2,3,figsize=(20,15))

	# subplot
	ax = ax_mat[0,0]
	E1s = 1*1e3*np.ones(N_draws) # E1 = xx MeV
	E2s = Emax - E1s
	matrix = CalcResponse(E1s,E2s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = 1 MeV")

	# subplot
	ax = ax_mat[0,1]
	E1s = 3.5*1e3*np.ones(N_draws) # E1 = xx MeV
	E2s = Emax - E1s
	matrix = CalcResponse(E1s,E2s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = 3.5 MeV")

	# subplot
	ax = ax_mat[0,2]
	E1s = 5*1e3*np.ones(N_draws) # E1 = xx MeV
	E2s = Emax - E1s
	matrix = CalcResponse(E1s,E2s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = 5 MeV")

	# subplot
	ax = ax_mat[1,0]
	E1s = np.random.uniform(low=Emin, high=Emax, size=N_draws) # uniform distribution of E1s
	E2s = Emax - E1s
	matrix = CalcResponse(E1s,E2s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = uniform")

	# subplot
	ax = ax_mat[1,1]
	Emid = (Emax+Emin)/2
	E1s = np.random.triangular(left=Emin, mode=Emid, right=Emax, size=N_draws) # E1 = 1 MeV
	E2s = Emax - E1s
	matrix = CalcResponse(E1s,E2s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = triangle (Emid={:.2f}) MeV".format(Emid/1e3))

	# subplot
	ax = ax_mat[1,2]
	Emid = (Emax)
	E1s = np.random.triangular(left=Emin, mode=Emid, right=Emax, size=N_draws) # E1 = 1 MeV
	E2s = Emax - E1s
	matrix = CalcResponse(E1s,E2s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = triangle (Emid={:.2f}) MeV".format(Emid/1e3))

	for ax in ax_mat.flatten():
		ax.set_xlabel("Eg [keV]")
		ax.set_ylabel(r"\sum Eg = Ex [keV]")

	plt.tight_layout()
	plt.subplots_adjust(top=0.93)
	plt.suptitle("Multiplicity 2")
	plt.savefig("resp_M2.png")
	plt.show()

if DoPlottingM3:

	# create plots
	f_max, ax_mat = plt.subplots(2,3,figsize=(20,15))

	# subplot
	ax = ax_mat[0,0]
	E1s = 1*1e3*np.ones(N_draws) # E1 = xx MeV
	E2s = 2*1e3*np.ones(N_draws) # E2 = xx MeV
	E3s = Emax - E1s - E2s
	matrix = CalcResponse(E1s,E2s,E3s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = 1 MeV, E2 = 2 MeV")

	# subplot
	ax = ax_mat[0,1]
	E1s = 1*1e3*np.ones(N_draws) # E1 = xx MeV
	E2s = 5*1e3*np.ones(N_draws) # E2 = xx MeV
	E3s = Emax - E1s - E2s
	matrix = CalcResponse(E1s,E2s,E3s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = 1 MeV, E2 = 5 MeV")

	# subplot
	ax = ax_mat[0,2]
	E1s = 3.33*1e3*np.ones(N_draws) # E1 = xx MeV
	E2s = 3.33*1e3*np.ones(N_draws) # E2 = xx MeV
	E3s = Emax - E1s - E2s
	matrix = CalcResponse(E1s,E2s,E3s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1=E2=E3 = 3.33 MeV")

	# subplot
	ax = ax_mat[1,0]
	E1s = np.random.uniform(low=Emin, high=Emax, size=N_draws) # uniform distribution of E1s
	E2s = np.random.uniform(low=Emin, high=Emax-E1s, size=N_draws) # uniform distribution of E1s
	E3s = Emax - E1s - E2s
	matrix = CalcResponse(E1s,E2s,E3s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = E2: uniform")

	# subplot
	ax = ax_mat[1,1]
	Emid = (Emax+Emin)/2
	E1s = np.random.triangular(left=Emin, mode=Emid, right=Emax, size=N_draws) # E1 = 1 MeV
	Emax_2 = Emax-E1s
	Emid_2 = (Emax_2+Emin)/2
	E2s = np.random.triangular(left=Emin, mode=Emid_2, right=Emax_2, size=N_draws) # E1 = 1 MeV
	E3s = Emax - E1s - E2s
	matrix = CalcResponse(E1s,E2s,E3s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1=E2: triangle, top=middle")

	# subplot
	ax = ax_mat[1,2]
	Emid = (Emax+Emin)/2
	E1s = np.random.triangular(left=Emin, mode=Emid, right=Emax, size=N_draws) # E1 = 1 MeV
	Emax_2 = Emax-E1s
	Emid_2 = Emax_2
	E2s = np.random.triangular(left=Emin, mode=Emid_2, right=Emax_2, size=N_draws) # E1 = 1 MeV
	E3s = Emax - E1s - E2s
	matrix = CalcResponse(E1s,E2s,E3s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1=E2: triangle, top=right")

	for ax in ax_mat.flatten():
		ax.set_xlabel("Eg [keV]")
		ax.set_ylabel(r"\sum Eg = Ex [keV]")
	
	plt.tight_layout()
	plt.subplots_adjust(top=0.93)
	plt.suptitle("Multiplicity 3")
	plt.savefig("resp_M3.png")
	plt.show()

if DoPlottingM4:

	# create plots
	f_max, ax_mat = plt.subplots(2,3,figsize=(20,15))

	# subplot
	ax = ax_mat[0,0]
	E1s = 1*1e3*np.ones(N_draws) # E1 = xx MeV
	E2s = 1*1e3*np.ones(N_draws) # E2 = xx MeV
	E3s = 2*1e3*np.ones(N_draws) # E2 = xx MeV
	E4s = Emax - E1s - E2s - E3s
	matrix = CalcResponse(E1s,E2s,E3s,E4s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = 1 MeV, E2 = 1 MeV, E2 = 2 MeV")

	# subplot
	ax = ax_mat[0,1]
	E1s = 1*1e3*np.ones(N_draws) # E1 = xx MeV
	E2s = 2*1e3*np.ones(N_draws) # E2 = xx MeV
	E3s = 3*1e3*np.ones(N_draws) # E2 = xx MeV
	E4s = Emax - E1s - E2s - E3s
	matrix = CalcResponse(E1s,E2s,E3s,E4s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = 1 MeV, E2 = 2 MeV, E2 = 3 MeV")

	# subplot
	ax = ax_mat[0,2]
	E1s = 2.25*1e3*np.ones(N_draws) # E1 = xx MeV
	E2s = 2.25*1e3*np.ones(N_draws) # E2 = xx MeV
	E3s = 2.25*1e3*np.ones(N_draws) # E2 = xx MeV
	E4s = Emax - E1s - E2s - E3s
	matrix = CalcResponse(E1s,E2s,E3s,E4s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1=E2=E3 = 2.25 MeV")

	# subplot
	ax = ax_mat[1,0]
	E1s = np.random.uniform(low=Emin, high=Emax, size=N_draws) # uniform distribution of E1s
	E2s = np.random.uniform(low=Emin, high=Emax-E1s, size=N_draws) # uniform distribution of E1s
	E3s = np.random.uniform(low=Emin, high=Emax-E1s-E2s, size=N_draws) # uniform distribution of E1s
	E4s = Emax - E1s - E2s - E3s
	matrix = CalcResponse(E1s,E2s,E3s,E4s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1 = E2 = E3: uniform")

	# subplot
	ax = ax_mat[1,1]
	Emid = (Emax+Emin)/2
	E1s = np.random.triangular(left=Emin, mode=Emid, right=Emax, size=N_draws) # E1 = 1 MeV
	Emax_2 = Emax-E1s
	Emid_2 = (Emax_2+Emin)/2
	E2s = np.random.triangular(left=Emin, mode=Emid_2, right=Emax_2, size=N_draws) # E1 = 1 MeV
	Emax_3 = Emax_2-E2s
	Emid_3 = (Emax_3+Emin)/2
	E3s = np.random.triangular(left=Emin, mode=Emid_3, right=Emax_3, size=N_draws) # E1 = 1 MeV
	E4s = Emax - E1s - E2s - E3s
	matrix = CalcResponse(E1s,E2s,E3s,E4s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1=E2: triangle, top=middle")

	# subplot
	ax = ax_mat[1,2]
	Emid = (Emax+Emin)/2
	E1s = np.random.triangular(left=Emin, mode=Emid, right=Emax, size=N_draws) # E1 = 1 MeV
	Emax_2 = Emax-E1s
	Emid_2 = Emax_2
	E2s = np.random.triangular(left=Emin, mode=Emid_2, right=Emax_2, size=N_draws) # E1 = 1 MeV
	Emax_3 = Emax_2-E2s
	Emid_3 = Emax_3
	E3s = np.random.triangular(left=Emin, mode=Emid_3, right=Emax_3, size=N_draws) # E1 = 1 MeV
	E4s = Emax - E1s - E2s - E3s
	matrix = CalcResponse(E1s,E2s,E3s,E4s,**defaults)
	# rebin result for plotting
	N_final = int(len(E_resp_array)/6)
	matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)
	ax.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
	ax.set_title("E1=E2: triangle, top=right")

	for ax in ax_mat.flatten():
		ax.set_xlabel("Eg [keV]")
		ax.set_ylabel(r"\sum Eg = Ex [keV]")
	
	plt.tight_layout()
	plt.subplots_adjust(top=0.93)
	plt.suptitle("Multiplicity 4")
	plt.savefig("resp_M4.png")
	plt.show()