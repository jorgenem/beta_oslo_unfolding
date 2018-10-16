from utilities import *
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm

folded1, cal, E_array, tmp = read_mama_2D("folded_2D_2gammas5MeV.m")
folded2, cal, E_array, tmp = read_mama_2D("folded_2D_2gammas1and9MeV.m")

Nbins = len(E_array)


combination = folded1 + folded2

R1 = div0(folded1, folded1.sum(axis=1))
R2 = div0(folded2, folded2.sum(axis=1))


i_1MeV = np.argmin(np.abs(E_array-1000))
i_5MeV = np.argmin(np.abs(E_array-5000))
i_9MeV = np.argmin(np.abs(E_array-9000))

j_10MeV = np.argmin(np.abs(E_array-10000))

unfolded = combination
for iteration in range(15):
	# for i in range(Nbins):
		# for j in range(Nbins):
	folded = np.zeros((Nbins, Nbins))
	folded += R1*unfolded[j_10MeV,i_5MeV]
	# folded += R1*unfolded[j_10MeV,i_5MeV]
	folded += R2*unfolded[j_10MeV,i_1MeV]
	folded += R2*unfolded[j_10MeV,i_9MeV]
	unfolded = unfolded + (combination - folded)
			# if i == i_1MeV and :
	# folded[i_1MeV,j] = 
	# folded = 



f, ((ax_raw, ax_unf), (ax_folded, ax_diff)) = plt.subplots(2,2)
ax_raw.pcolormesh(E_array, E_array, combination, norm=LogNorm())
ax_raw.set_title("raw")
ax_unf.pcolormesh(E_array, E_array, unfolded, norm=LogNorm())
ax_unf.set_title("unf")
ax_folded.pcolormesh(E_array, E_array, folded, norm=LogNorm())
ax_folded.set_title("folded")
ax_diff.pcolormesh(E_array, E_array, np.abs(folded-combination), norm=LogNorm())
ax_diff.set_title("diff")

plt.show()

