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
R_2D, cal_resp, E_resp_array, tmp = read_mama_2D(fname_resp_mat)
# Assumed lower threshold for gammas in response matrix
E_thres = 100
i_thres = np.argmin(np.abs(E_resp_array - E_thres))
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
ax1.pcolormesh(E_resp_array, E_resp_array, R_2D, norm=LogNorm())
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



from ROOT import gRandom, TH1, TH2, TH1D, TH2D, cout, gROOT, TCanvas, TLegend
from ROOT import RooUnfoldResponse
from ROOT import RooUnfoldBayes





Nbins = len(E_resp_array)
Emin = E_resp_array[0]
Emax = E_resp_array[-1]

matrix_unfolded = np.zeros((Nbins,Nbins))



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




# == This part needs to be looped over the rows i_Eg2, then followed by an opposite loop+unfolding over i_Eg1. ==
# i_Eg2 = 250 # Pick a test row
# ax4.plot(E_array_folded, matrix_folded[i_Eg2,:])
for i_Eg2 in range(Nbins):#range(245,255):
    print("Now doing i_Eg2 =", i_Eg2, flush=True)
    for i in range(Nbins):
        Ei = E_resp_array[i]
        hMeas.Fill(Ei,matrix_folded[i_Eg2,i])
    
    # hack to recalculate the Uncertainties now, after the histogram is filled
    hMeas.Sumw2(False)
    hMeas.Sumw2(True)
    # hTrue.Sumw2(False) # doesn't work yet?
    # hTrue.Sumw2(True)  # doesn't work yet?
    
    # print("==================================== UNFOLD ===================================")
    Niterations = 5
    unfold= RooUnfoldBayes     (response, hMeas, Niterations);    #  OR
    # unfold= RooUnfoldSvd     (response, hMeas, 20);     #  OR
    #unfold= RooUnfoldTUnfold (response, hMeas);         #  OR
    # unfold= RooUnfoldIds     (response, hMeas, 3);      #  OR
    # unfold= RooUnfoldInvert    (response, hMeas);      #  OR
    
    hReco= unfold.Hreco();
    # unfold.PrintTable (cout, hTrue);
    
    matrix_unfolded[i_Eg2,:] = np.array(hReco)[0:Nbins]

# c1 = TCanvas()
# hReco.Draw();
# hMeas.SetLineColor(2)
# hMeas.Draw("same");
# hTrue.SetLineColor(8);
# hTrue.Draw("same");
# # c1.SetLogy()

# legend = TLegend(0.8,0.8,0.9,0.9);
# legend.AddEntry(hReco,"Unfolded","l");
# legend.AddEntry(hMeas,"Measured","l");
# legend.AddEntry(hTrue,"True","l");
# legend.Draw();





# Back to Python: Get the resulting matrix and plot:

ax4.pcolormesh(E_array_folded, E_array_folded, matrix_unfolded, norm=LogNorm())
# ax4.plot(E_array_folded, array_raw, label="raw")
# ax4.plot(E_array_folded, array_folded, label="folded")
# ax4.plot(E_resp_array, array_unfolded[0:Nbins], label="unfolded")
ax4.legend()


plt.show()