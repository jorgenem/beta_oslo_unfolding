from utilities import *
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# import pyunfold as pu
# from pyunfold.priors import uniform_prior


""" 
This script attempts to unfold the gamma-energy-axes sorted, folded spectra that Fabio and I invented

First order of business is to find out if two successive 1D unfoldings do the trick or not.
"""


# Settings for matplotlib and latex:
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'serif':['Computer Modern Roman']})
rc('text', usetex=True)

# Read response matrix:
fname_resp_mat = "response_matrix-SuN2015-20keV-1p0FWHM.m"
fname_resp_dat = "resp-SuN2015-20keV-1p0FWHM.dat"
R_2D, cal_resp, E_resp_array, tmp = read_mama_2D(fname_resp_mat)
# Assumed lower threshold for gammas in response matrix
E_thres = 100
i_thres = np.argmin(np.abs(E_resp_array - E_thres))
R_2D[:,:i_thres] = 0


# Cutoff energy spectra and response at 2 MeV:
Ecut = 2000
Nbins = int((Ecut-cal_resp["a0x"])/cal_resp["a1x"] + 0.5)
R_2D = R_2D[0:Nbins,0:Nbins]
E_resp_array = E_resp_array[0:Nbins]



# Normalize:
for i in range(R_2D.shape[0]):
	norm = R_2D[i,:].sum()
	if(norm>0):
		R_2D[i,:] = R_2D[i,:] / norm #* eff[i]
	else:
		R_2D[i,:] = 0

# Allocate plots and plot response matrix:
f1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
f2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2,2)
# ax1.pcolormesh(E_resp_array, E_resp_array, R_2D, norm=LogNorm())
# ax1.set_title("Response matrix")

customLogNorm = LogNorm(vmin=1e0, vmax=1e4)

# Read and plot true and folded M=2 spectra:
matrix_true, tmp, E_array_true, tmp = read_mama_2D("truth_2D_2gammas-Egaxes.m")
# Multiply by 1e3 because we simulated that many response draws per gamma when generating:
matrix_true = 1e3*matrix_true[0:Nbins,0:Nbins] 
E_array_true = E_array_true[0:Nbins]
# cbar_ax1 = ax1.pcolormesh(E_array_true, E_array_true, matrix_true, norm=LogNorm())
cbar_ax1 = ax1.imshow(matrix_true, norm=customLogNorm, origin="lower", extent=[E_resp_array[0], E_resp_array[-1], E_resp_array[0], E_resp_array[-1]], cmap="jet")
f1.colorbar(cbar_ax1, ax=ax1)
ax1.set_title("True (Eg1, Eg2) distribution")
ax1.set_xlabel("$E_{\gamma,1}\,\mathrm{(keV)}$")
ax1.set_ylabel("$E_{\gamma,2}\,\mathrm{(keV)}$")
matrix_folded, tmp, E_array_folded, tmp = read_mama_2D("folded_2D_2gammas-Egaxes.m")
matrix_folded = matrix_folded[0:Nbins,0:Nbins] 
E_array_folded = E_array_folded[0:Nbins]
# cbar_ax2 = ax2.pcolormesh(E_array_folded, E_array_folded, matrix_folded, norm=LogNorm())
cbar_ax2 = ax2.imshow(matrix_folded, norm=customLogNorm, origin="lower", extent=[E_resp_array[0], E_resp_array[-1], E_resp_array[0], E_resp_array[-1]], cmap="jet")
f1.colorbar(cbar_ax2, ax=ax2)
ax2.set_title("Folded (Eg1, Eg2) distribution")
ax2.set_xlabel("$E_{\gamma,1}\,\mathrm{(keV)}$")
ax2.set_ylabel("$E_{\gamma,2}\,\mathrm{(keV)}$")



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



from ROOT import gRandom, TH1, TH2, TH1D, TH2D, cout, gROOT, TCanvas, TLegend
from ROOT import RooUnfoldResponse
from ROOT import RooUnfoldBayes





Nbins = len(E_resp_array)
Emin = E_resp_array[0]
Emax = E_resp_array[-1]

matrix_unfolded1 = np.zeros((Nbins,Nbins))



hTrue= TH1D ("true", "Test Truth",    Nbins, Emin, Emax);
hMeas= TH1D ("meas", "Test Measured", Nbins, Emin, Emax);

print("==================================== TRAIN ====================================")
response= RooUnfoldResponse (hMeas, hTrue);

for i in range(Nbins): # x_true
    Ei = E_resp_array[i] # x_true
    for j in range(Nbins): # x_measured
        Ej = E_resp_array[j] # x_measured
        mc = R_2D[i,j]
        # response.Fill (x_measured, x_true)
        response.Fill (Ej, Ei, mc);
    # account for eff < 1
    eff_ = R_2D[i,:].sum()
    pmisses = 1-eff_ # probability of misses
    response.Miss(Ei,pmisses)



print("==================================== TEST =====================================")

# # "True" Eg in keV, counts
# Eg_choose = np.array([[4000,2000]])

# Eg_choose = np.array([[4000,2000],
#                       [2000,1000],
#                       [1500,1000],
#                       [3000,500],
#                       ]) 

# Eg_min = 1e3
# i_Eg_choose = np.argmin(np.abs(E_resp_array - Eg_min))
# N_in=40
# Egs_in = E_resp_array[i_Eg_choose:i_Eg_choose+N_in]
# def cnt(E):
#     # some dummy funciton to create a number of counts
#     return (0.2*(E-Egs_in[int(N_in/2)])**2 + 0.05* E)/100
# Eg_choose = np.array([(Eg,cnt(Eg)) for Eg in Egs_in])

# Fill measured hist with row from matrix_folded


Niterations = 10

fname_save_unf1 = "matrix_unfolded1.npy"
try:
    matrix_unfolded1 = np.load(fname_save_unf1)
except:
    matrix_unfolded1 = np.zeros((Nbins,Nbins))
    for i_Eg1 in range(Nbins):#range(60,62):
        hMeas= TH1D ("meas", "Test Measured", Nbins, Emin, Emax);
        print("Now doing i_Eg1 =", i_Eg1, flush=True)
        for i in range(Nbins):
            Ei = E_resp_array[i]
            hMeas.Fill(Ei,matrix_folded[i_Eg1,i])
        
        # hack to recalculate the Uncertainties now, after the histogram is filled
        hMeas.Sumw2(False)
        hMeas.Sumw2(True)
        # hTrue.Sumw2(False) # doesn't work yet?
        # hTrue.Sumw2(True)  # doesn't work yet?
        
        # print("==================================== UNFOLD ===================================")
        unfold= RooUnfoldBayes     (response, hMeas, Niterations);    #  OR
        # unfold= RooUnfoldSvd     (response, hMeas, 20);     #  OR
        #unfold= RooUnfoldTUnfold (response, hMeas);         #  OR
        # unfold= RooUnfoldIds     (response, hMeas, 3);      #  OR
        # unfold= RooUnfoldInvert    (response, hMeas);      #  OR
        
        hReco= unfold.Hreco();
        # unfold.PrintTable (cout, hTrue);
        
        matrix_unfolded1[i_Eg1,:] = np.array(hReco)[0:Nbins]
        

    matrix_unfolded1 = np.nan_to_num(matrix_unfolded1)
    np.save(fname_save_unf1, matrix_unfolded1)



# cbar_ax3 = ax3.pcolormesh(E_array_folded, E_array_folded, matrix_unfolded1, norm=LogNorm(vmin=1e-1, vmax=1e3))
cbar_ax3 = ax3.imshow(matrix_unfolded1, norm=customLogNorm, origin="lower", extent=[E_resp_array[0], E_resp_array[-1], E_resp_array[0], E_resp_array[-1]], cmap="jet")
f1.colorbar(cbar_ax3, ax=ax3)
ax3.set_title("Unfolded first direction")
ax3.set_xlabel("$E_{\gamma,1}\,\mathrm{(keV)}$")
ax3.set_ylabel("$E_{\gamma,2}\,\mathrm{(keV)}$")



# Then unfold the other direction:
fname_save_unf2 = "matrix_unfolded2.npy"
try:
    matrix_unfolded2 = np.load(fname_save_unf2)
except:
    matrix_unfolded2 = np.zeros((Nbins,Nbins))
    for i_Eg2 in range(Nbins):
        hMeas= TH1D ("meas", "Test Measured", Nbins, Emin, Emax);
        print("Now doing i_Eg2 =", i_Eg2, flush=True)
        for i in range(Nbins):
            Ei = E_resp_array[i]
            hMeas.Fill(Ei,matrix_unfolded1[i,i_Eg2])

        # hMeas.Draw()
        # break

        
        # hack to recalculate the Uncertainties now, after the histogram is filled
        hMeas.Sumw2(False)
        hMeas.Sumw2(True)
        # hTrue.Sumw2(False) # doesn't work yet?
        # hTrue.Sumw2(True)  # doesn't work yet?
        
        # print("==================================== UNFOLD ===================================")
        unfold= RooUnfoldBayes     (response, hMeas, Niterations);    #  OR
        # unfold= RooUnfoldSvd     (response, hMeas, 20);     #  OR
        #unfold= RooUnfoldTUnfold (response, hMeas);         #  OR
        # unfold= RooUnfoldIds     (response, hMeas, 3);      #  OR
        # unfold= RooUnfoldInvert    (response, hMeas);      #  OR
        
        hReco= unfold.Hreco();
        # unfold.PrintTable (cout, hTrue);
        
        matrix_unfolded2[:,i_Eg2] = np.array(hReco)[0:Nbins]

    matrix_unfolded2 = np.nan_to_num(matrix_unfolded2)
    np.save(fname_save_unf2, matrix_unfolded2)




# Back to Python: Get the resulting matrix and plot:

# cbar_ax4 = ax4.pcolormesh(E_array_folded, E_array_folded, matrix_unfolded2, norm=LogNorm(vmin=1e-1, vmax=1e3))
cbar_ax4 = ax4.imshow(matrix_unfolded2, norm=customLogNorm, origin="lower", extent=[E_resp_array[0], E_resp_array[-1], E_resp_array[0], E_resp_array[-1]], cmap="jet")
ax4.set_title("Unfolded second direction")
ax4.set_xlabel("$E_{\gamma,1}\,\mathrm{(keV)}$")
ax4.set_ylabel("$E_{\gamma,2}\,\mathrm{(keV)}$")
f1.colorbar(cbar_ax4, ax=ax4)
# ax4.plot(E_array_folded, array_raw, label="raw")
# ax4.plot(E_array_folded, array_folded, label="folded")
# ax4.plot(E_resp_array, array_unfolded[0:Nbins], label="unfolded")



# Recalculate to (Ex, Eg) and plot true vs unfolded:
Nbins_Ex = 2*Nbins
E_array_Ex = np.linspace(E_resp_array[0], 2*E_resp_array[-1], Nbins_Ex)
matrix_ExEg_true = np.zeros((Nbins_Ex,Nbins))
matrix_ExEg_unfolded = np.zeros((Nbins_Ex,Nbins))
matrix_ExEg_folded = np.zeros((Nbins_Ex, Nbins))
for i_Eg1 in range(Nbins):
    for i_Eg2 in range(Nbins):
        i_Ex = i_Eg1 + i_Eg2
        matrix_ExEg_true[i_Ex,i_Eg1] += matrix_true[i_Eg1, i_Eg2]
        matrix_ExEg_true[i_Ex,i_Eg2] += matrix_true[i_Eg1, i_Eg2]

        matrix_ExEg_unfolded[i_Ex,i_Eg1] += matrix_unfolded2[i_Eg1, i_Eg2]
        matrix_ExEg_unfolded[i_Ex,i_Eg2] += matrix_unfolded2[i_Eg1, i_Eg2]

        matrix_ExEg_folded[i_Ex,i_Eg1] += matrix_folded[i_Eg1, i_Eg2]
        matrix_ExEg_folded[i_Ex,i_Eg2] += matrix_folded[i_Eg1, i_Eg2]

cbar_ax5 = ax5.imshow(matrix_ExEg_true, norm=customLogNorm, origin="lower", extent=[E_resp_array[0], E_resp_array[-1], E_array_Ex[0], E_array_Ex[-1]], aspect=Nbins/Nbins_Ex, cmap="jet")
f2.colorbar(cbar_ax5, ax=ax5)
ax5.set_title("True spectrum sorted as Ex vs Eg")
ax5.set_xlabel("$E_{\gamma}\,\mathrm{(keV)}$")
ax5.set_ylabel("$E_x \,\mathrm{(keV)}$")
cbar_ax6 = ax6.imshow(matrix_ExEg_unfolded, norm=customLogNorm, origin="lower", extent=[E_resp_array[0], E_resp_array[-1], E_array_Ex[0], E_array_Ex[-1]], aspect=Nbins/Nbins_Ex, cmap="jet")
f2.colorbar(cbar_ax6, ax=ax6)
ax6.set_title("Unfolded spectrum sorted as Ex vs Eg")
ax6.set_xlabel("$E_{\gamma}\,\mathrm{(keV)}$")
ax6.set_ylabel("$E_x \,\mathrm{(keV)}$")
cbar_ax7 = ax7.imshow(matrix_ExEg_folded, norm=customLogNorm, origin="lower", extent=[E_resp_array[0], E_resp_array[-1], E_array_Ex[0], E_array_Ex[-1]], aspect=Nbins/Nbins_Ex, cmap="jet")
f2.colorbar(cbar_ax7, ax=ax7)
ax7.set_title("Folded (\"detected\") spectrum sorted as Ex vs Eg")
ax7.set_xlabel("$E_{\gamma}\,\mathrm{(keV)}$")
ax7.set_ylabel("$E_x \,\mathrm{(keV)}$")



# Unfold the gamma axis of the ExEg sorted, folded matrix:

fname_save_unf_ExEg = "matrix_unfolded_ExEg.npy"
try:
    matrix_unfolded_ExEg = np.load(fname_save_unf_ExEg)
except:
    matrix_unfolded_ExEg = np.zeros((Nbins_Ex,Nbins))
    for i_Ex in range(Nbins_Ex):#range(60,62):
        hMeas= TH1D ("meas", "Test Measured", Nbins, Emin, Emax);
        print("Now doing i_Ex =", i_Ex, flush=True)
        for i in range(Nbins):
            Ei = E_resp_array[i]
            hMeas.Fill(Ei,matrix_ExEg_folded[i_Ex,i])
        
        # hack to recalculate the Uncertainties now, after the histogram is filled
        hMeas.Sumw2(False)
        hMeas.Sumw2(True)
        # hTrue.Sumw2(False) # doesn't work yet?
        # hTrue.Sumw2(True)  # doesn't work yet?
        
        # print("==================================== UNFOLD ===================================")
        unfold= RooUnfoldBayes     (response, hMeas, Niterations);    #  OR
        # unfold= RooUnfoldSvd     (response, hMeas, 20);     #  OR
        #unfold= RooUnfoldTUnfold (response, hMeas);         #  OR
        # unfold= RooUnfoldIds     (response, hMeas, 3);      #  OR
        # unfold= RooUnfoldInvert    (response, hMeas);      #  OR
        
        hReco= unfold.Hreco();
        # unfold.PrintTable (cout, hTrue);
        
        matrix_unfolded_ExEg[i_Ex,:] = np.array(hReco)[0:Nbins]
        

    matrix_unfolded_ExEg = np.nan_to_num(matrix_unfolded_ExEg)
    np.save(fname_save_unf_ExEg, matrix_unfolded_ExEg)



cbar_ax8 = ax8.imshow(matrix_unfolded_ExEg, norm=customLogNorm, origin="lower", extent=[E_resp_array[0], E_resp_array[-1], E_array_Ex[0], E_array_Ex[-1]], aspect=Nbins/Nbins_Ex, cmap="jet")
f1.colorbar(cbar_ax8, ax=ax8)
ax8.set_title("ExEg sorted folded spec, unfolded along Eg axis")
ax8.set_xlabel("$E_{\gamma}\,\mathrm{(keV)}$")
ax8.set_ylabel("$E_x \,\mathrm{(keV)}$")





# Calculate some integrals:
print("matrix_true.sum() =", matrix_true.sum(), flush=True)
Eg1_low, Eg1_high = 1500,1900
Eg2_low, Eg2_high = 1150,1450
i_Eg1_low, i_Eg1_high = np.argmin(np.abs(E_resp_array - Eg1_low)), np.argmin(np.abs(E_resp_array - Eg1_high))
i_Eg2_low, i_Eg2_high = np.argmin(np.abs(E_resp_array - Eg2_low)), np.argmin(np.abs(E_resp_array - Eg2_high))
print("matrix_unfolded2.sum() =", matrix_unfolded2.sum(), flush=True)
print("matrix_unfolded2[i_Eg1_low:i_Eg1_high,i_Eg2_low:i_Eg2_high].sum() =", matrix_unfolded2[i_Eg1_low:i_Eg1_high,i_Eg2_low:i_Eg2_high].sum(), flush=True)
print("matrix_unfolded2[i_Eg1_low:i_Eg1_high,i_Eg2_low:i_Eg2_high].sum()/matrix_true.sum() =", matrix_unfolded2[i_Eg1_low:i_Eg1_high,i_Eg2_low:i_Eg2_high].sum()/matrix_true.sum(), flush=True)

print("")
print("matrix_unfolded_ExEg.sum() =", matrix_unfolded_ExEg.sum(), flush=True)
# corresponding approximation(!) to Ex
Ex_low = Eg1_low + Eg2_low
Ex_high =  Eg1_high + Eg2_high
i_Ex_low, i_Ex_high = np.argmin(np.abs(E_array_Ex - Ex_low)), np.argmin(np.abs(E_array_Ex - Ex_high))
ncounts = matrix_unfolded_ExEg[i_Ex_low:i_Ex_high,i_Eg1_low:i_Eg1_high].sum()
ncounts += matrix_unfolded_ExEg[i_Ex_low:i_Ex_high,i_Eg2_low:i_Eg2_high].sum()
print("matrix_unfolded_ExEg(Eg1 & Eg2) =", ncounts, flush=True)
print("matrix_unfolded_ExEg(Eg1 & Eg2) / matrix_true.sum() =", ncounts/matrix_true.sum(), flush=True)



# f1.subplots_adjust(wspace=0, hspace=0)
# f2.subplots_adjust(wspace=0, hspace=0)


# == 1-D plotting ==

# f_1d, ax_1d = plt.subplots(1,1)
# E_choose = 1700
# i_choose = np.argmin(np.abs(E_resp_array-E_choose))
# plt.plot(matrix_unfolded2[:,i_choose])


plt.show()