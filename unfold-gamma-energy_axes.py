from utilities import *
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# import pyunfold as pu
# from pyunfold.priors import uniform_prior


""" 
This script attempts to unfold the gamma-energy-axes sorted, folded spectra that Fabio invented

First order of business is to find out if two successive 1D unfoldings do the trick or not.
"""


# Read response matrix:
fname_resp_mat = "response_matrix-SuN2015-20keV-1p0FWHM.dat"
fname_resp_dat = "resp-SuN2015-20keV-1p0FWHM.dat"
R_2D, cal_resp, E_array_resp, tmp = read_mama_2D(fname_resp_mat)
# Assumed lower threshold for gammas in response matrix
E_thres = 100
i_thres = np.argmin(np.abs(E_array_resp - E_thres))
R_2D[:,:i_thres] = 0
# Normalize:
for i in range(R_2D.shape[0]):
	norm = R_2D[i,:].sum()
	if(norm>0):
		R_2D[i,:] = R_2D[i,:] / norm #* eff[i]
	else:
		R_2D[i,:] = 0

# Allocate plots and plot response matrix:
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.pcolormesh(E_array_resp, E_array_resp, R_2D, norm=LogNorm())
ax1.set_title("Response matrix")


# Read and plot true and folded M=2 spectra:
matrix_true, tmp, E_array_true, tmp = read_mama_2D("truth_2D_2gammas-Egaxes.m")
ax2.pcolormesh(E_array_true, E_array_true, matrix_true, norm=LogNorm())
ax2.set_title("True (Eg1, Eg2) dist (Not scaled to N_resp_draws)")
matrix_folded, tmp, E_array_folded, tmp = read_mama_2D("folded_2D_2gammas-Egaxes.m")
ax3.pcolormesh(E_array_folded, E_array_folded, matrix_folded, norm=LogNorm())
ax3.set_title("Folded (Eg1, Eg2) dist")



# # Pyunfold:
# It's too slow!

# response_err = np.sqrt(R_2D)

# uni_prior = uniform_prior(len(E_array_folded))

# # Try unfolding the Eg1 axis, Eg2-bin-by-bin:
# # for i_Eg2 in [50]:#range(len(E_array_folded)):
# # print("matrix_folded[:,i_Eg2] =", matrix_folded[:,i_Eg2])
# # print("R_2D.shape =", R_2D.shape)
# data_observed = matrix_folded[:,50]
# print("data_observed.shape = ", data_observed.shape, flush=True)
# data_observed_err = np.sqrt(data_observed)

# # Make up some efficiencies:
# efficiencies = np.ones_like(data_observed, dtype=float)
# efficiencies_err = np.full_like(efficiencies, 0.1, dtype=float)


# unfolded_results = pu.iterative_unfold(data=data_observed,
#                                     data_err=data_observed_err,
#                                     response=R_2D,
#                                     response_err=response_err,
#                                     efficiencies=efficiencies,
#                                     efficiencies_err=efficiencies_err,
#                                     # prior=uni_prior,
#                                     callbacks=[pu.callbacks.Logger()])




# Trying my own unfolding function instead:
# from unfold import *

# matrix_unfolded, E_array_unfolded, tmp = unfold(matrix_folded, E_array_folded, E_array_folded, fname_resp_dat, fname_resp_mat, verbose=True, plot=True)


array_raw = matrix_folded[300,:]

# Run folding iterations:
Nit = 5
for iteration in range(Nit):
    if iteration == 0:
        # matrix_unfolded = matrix_folded
        array_unfolded = array_raw
    else:
        print("array_raw.shape =", array_raw.shape, flush=True)
        print("array_folded.shape =", array_folded.shape, flush=True)
        array_unfolded = array_unfolded + (array_raw.reshape(1,len(array_raw)) - array_folded)
        print("array_unfolded.shape =", array_unfolded.shape, flush=True)

    # Try applying smoothing to the unfolded array before folding:
    # print("array_unfolded.shape =", array_unfolded.shape, flush=True)
    print("array_unfolded.sum() =", array_unfolded.sum())
    array_unfolded = shift_and_smooth3D(array_unfolded.reshape(1,len(array_unfolded)), E_array_folded, FWHM=np.ones(len(array_unfolded)), p=np.ones(len(array_unfolded)), shift=0, smoothing=True)
    array_unfolded.reshape(array_unfolded.size)
    print("array_unfolded.sum() =", array_unfolded.sum())

    array_folded = np.dot(R_2D.T, array_unfolded.T).T # Have to do some transposing to get the axis orderings right for matrix product
    # 20171110: Tried transposing R. Was it that simple? Immediately looks good.
    #           Or at least much better. There is still something wrong giving negative counts left of peaks.
    #           Seems to come from unfolding and not from compton subtraction

    # Calculate reduced chisquare of the "fit" between folded-unfolded matrix and original raw
    # chisquares[iteration] = div0(np.power(foldmat-matrix_folded,2),np.where(matrix_folded>0,matrix_folded,0)).sum() / Ndof
    # if verbose:
        # print("Folding iteration = {}, chisquare = {}".format(iteration,chisquares[iteration]), flush=True)




# ax4.pcolormesh(E_array_folded, E_array_folded, foldmat, norm=LogNorm())
ax4.plot(E_array_folded, array_raw, label="raw")
ax4.plot(E_array_folded, array_folded, label="folded")
ax4.plot(E_array_folded, array_unfolded, label="unfolded")
ax4.legend()


plt.show()