from utilities import *
import matplotlib.pyplot as plt 
import sys


f, (ax_mat, ax_proj) = plt.subplots(2,1)

# fname = "mama-testing/unfolded-si28.m"
# fname = "mama-testing/firstgen-si28.m"
# fname = "data/alfna-Re187.m"
fname = "response_matrix-Oscar2017-20keV-1p0FWHM.m"
# fname = "unfolded-28Si.m"

matrix, cal, Ex_array, Eg_array = read_mama_2D(fname)

from matplotlib.colors import LogNorm
cbar_mat = ax_mat.pcolormesh(Eg_array, Ex_array, matrix, norm=LogNorm(), cmap="jet")

f.colorbar(cbar_mat, ax=ax_mat)
ax_mat.set_title("fname = "+fname)
ax_mat.set_xlabel(r"$E_{\gamma}^{'}$ (keV)")
ax_mat.set_ylabel(r"$E_\gamma$ (keV)")

# Project down on Eg axis for a chosen Ex range
i_proj_low = 255
i_proj_high = 256

ax_proj.plot(Eg_array, matrix[i_proj_low:i_proj_high,:].sum(axis=0), label="proj {:.2f} to {:.2f}".format(Ex_array[i_proj_low],Ex_array[i_proj_high]))
ax_proj.legend()
ax_proj.set_xlabel(r"$E_{\gamma}^{'}$ (keV)")

ax_proj.set_title("Total number of counts in proj. = {:.2f}".format(matrix[i_proj_low:i_proj_high,:].sum(), flush=True))
f.subplots_adjust(hspace=0.3)

plt.show()