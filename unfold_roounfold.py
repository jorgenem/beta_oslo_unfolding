# Trying to unfold gamma-ray spectra via RooUnfold

from utilities import *
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm

from ROOT import gRandom, TH1, TH2, TH1D, TH2D, cout, gROOT, TCanvas, TLegend
from ROOT import RooUnfoldResponse
from ROOT import RooUnfoldBayes
# from ROOT import RooUnfoldSvd
# from ROOT import RooUnfoldTUnfold
# from ROOT import RooUnfoldIds
# from ROOT import RooUnfoldInvert


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

print("Haking an efficiency")
# eff = 0.95
for i in range(R_2D.shape[0]):
	norm = R_2D[i,:].sum()
	# Hack an efficiency
	norm *=  0.95 #eff[i]
	if(norm>0):
		R_2D[i,:] = R_2D[i,:] / (norm)
	else:
		R_2D[i,:] = 0


# f_cmp, ax_cmp = plt.subplots(1,1)
# ax_cmp.plot(E_resp_array, R_2D[400,:])

# fig, ax = plt.subplots(1,1)
# im1 = ax.pcolormesh(E_resp_array, E_resp_array, R_2D, norm=LogNorm(vmin=1e-3, vmax=1e-1))
# plt.colorbar(im1)
# plt.show()


# ==============================================================================
#  Example Unfolding
# ==============================================================================

Nbins = len(E_resp_array)
Emin = E_resp_array[0]
Emax = E_resp_array[-1]
hTrue= TH1D ("true", "Test Truth",    Nbins, Emin, Emax);
hMeas= TH1D ("meas", "Test Measured", Nbins, Emin, Emax);

print("==================================== TRAIN ====================================")
response= RooUnfoldResponse (hMeas, hTrue);

for i in range(Nbins):
	for j in range(Nbins):
		Ei = E_resp_array[i]
		Ej = E_resp_array[j]
		mc = R_2D[i,j]
		response.Fill (Ej, Ei, mc);


print("==================================== TEST =====================================")

# "True" Eg in keV, counts
# Eg_choose = np.array([[4000,2000]])

# Eg_choose = np.array([[4000,2000],
#                       [2000,1000],
#                       [1500,1000],
#                       [3000,500],
#                       ]) 

Eg_min = 1e3
i_Eg_choose = np.argmin(np.abs(E_resp_array - Eg_min))
N_in=40
Egs_in = E_resp_array[i_Eg_choose:i_Eg_choose+N_in]
def cnt(E):
	# some dummy funciton to create a number of counts
	return (0.2*(E-Egs_in[int(N_in/2)])**2 + 0.05* E)/100
Eg_choose = np.array([(Eg,cnt(Eg)) for Eg in Egs_in])

# Fill true and measured histograms
for Eg in Eg_choose:
	i_Eg_choose = np.argmin(np.abs(E_resp_array - Eg[0]))
	# # immitate statistical fluctuations of incident gamma rays
	# ncounts = np.random.normal(loc=Eg[1],scale=np.sqrt(Eg[1])) 
	ncounts = Eg[1]
	for i in range(Nbins):
		Ei = E_resp_array[i]
		ncounts_ = ncounts * R_2D[i_Eg_choose,i]
		# immitate statistical fluctuations of response gamma rays
		ncounts_ = np.random.normal(loc=ncounts_,scale=np.sqrt(ncounts_)) 
		hMeas.Fill(Ei,ncounts_)
	hTrue.Fill(Eg[0],Eg[1])

# hack to recalculate the Uncertainties now, after the histogram is filled
hMeas.Sumw2(False)
hMeas.Sumw2(True)
hTrue.Sumw2(False) # doesn't work yet?
hTrue.Sumw2(True)  # doesn't work yet?

print("==================================== UNFOLD ===================================")
Niterations = 10
unfold= RooUnfoldBayes     (response, hMeas, Niterations);    #  OR
# unfold= RooUnfoldSvd     (response, hMeas, 20);     #  OR
#unfold= RooUnfoldTUnfold (response, hMeas);         #  OR
# unfold= RooUnfoldIds     (response, hMeas, 3);      #  OR
# unfold= RooUnfoldInvert    (response, hMeas);      #  OR

hReco= unfold.Hreco();
unfold.PrintTable (cout, hTrue);

c1 = TCanvas()
hReco.Draw();
hMeas.SetLineColor(2)
hMeas.Draw("same");
hTrue.SetLineColor(8);
hTrue.Draw("same");
# c1.SetLogy()

legend = TLegend(0.8,0.8,0.9,0.9);
legend.AddEntry(hReco,"Unfolded","l");
legend.AddEntry(hMeas,"Measured","l");
legend.AddEntry(hTrue,"True","l");
legend.Draw();