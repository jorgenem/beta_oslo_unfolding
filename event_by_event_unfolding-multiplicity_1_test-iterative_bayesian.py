from utilities import *

import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as st

fname_resp = 'resp-sun2015.dat'
fname_resp_mat = 'response_matrix-sun2015.m'
R_2D, cal_resp, E_resp_array, tmp = read_mama_2D(fname_resp_mat)
N_resp = len(E_resp_array)

# Assumed lower threshold for gammas in response matrix
E_thres = 100
i_thres = np.argmin(np.abs(E_resp_array - E_thres))
R_2D[:,:i_thres] = 0
for i in range(R_2D.shape[0]):
	try:
		R_2D[i,:] = R_2D[i,:] / R_2D[i,:].sum()
	except:
		R_2D[i,:] = 0

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



# Use a seed to reproduce the same events every time
np.random.seed(2)
N_events = 5000
# Imagine that this loop reads a table of events.
# This simplest version assumes only one gamma per event 
# (or, like the Oslo method, unfolding each Eg independently 
# even if they're from the same event.)

# However, to test the approach, we will draw true energies 
# and fold them stochastically.
dist_true = np.zeros(N_resp)
dist_folded = np.zeros(N_resp)
posterior_sum = np.zeros(N_resp)
# prior = np.ones(N_resp)/N_resp # Uniform prior on E_t
prior = np.where(E_resp_array > 700, np.ones(N_resp), np.zeros(N_resp)) # Uniform prior on E_t
# prior /= prior.sum()
# prior = st.triang.pdf(E_resp_array, loc=1000, c=0.5, scale=5000)
N_iter = 1
for i_it in range(N_iter):
	for i_ev in range(N_events):
	    # Draw the true Eg value:
	    # E_t = np.random.uniform(low=4*1e3, high=5*1e3)
	    # E_t = 4500
	    E_t = np.random.normal(loc=4500, scale=100)
	    i_E_t = np.argmin(np.abs(E_resp_array - E_t))
	    print("i_E_t =", i_E_t, flush=True)
	    E_f = np.random.choice(E_resp_array, p=R_2D[i_E_t,:])
	    i_E_f = np.argmin(np.abs(E_resp_array - E_f))
	    print("i_E_f =", i_E_f, flush=True)
	
	    dist_true[i_E_t] += 1
	    dist_folded[i_E_f] += 1
	
	    # Use Bayes theorem to get the probability distribution of E_t given E_f:
	    # This is in fact the column of the response matrix corresponding to E_f
	    posterior_current = R_2D[:,i_E_f].T * prior
	    # posterior_current /= posterior_current.sum() # Normalize current posterior
	
	    # Add to total posterior:
	    posterior_sum += posterior_current

	# Update prior
	# prior = 





f, ax = plt.subplots(1,1)
ax.plot(E_resp_array[0:int(N_resp/2)], dist_true[0:int(N_resp/2)], label="true")
ax.plot(E_resp_array[0:int(N_resp/2)], dist_folded[0:int(N_resp/2)], label="folded")
ax.plot(E_resp_array[0:int(N_resp/2)], posterior_sum[0:int(N_resp/2)], label="posterior")
ax.legend()

plt.show()
