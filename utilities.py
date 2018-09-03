import numpy as np 


def read_mama_2D(filename):
    # Reads a MAMA matrix file and returns the matrix as a numpy array, 
    # as well as a list containing the four calibration coefficients
    # (ordered as [bx, ax, by, ay] where Ei = ai*channel_i + bi)
    # and 1-D arrays of calibrated x and y values for plotting and similar.
    matrix = np.genfromtxt(filename, skip_header=10, skip_footer=1)
    cal = {}
    with open(filename, 'r') as datafile:
        calibration_line = datafile.readlines()[6].split(",")
        # a = [float(calibration_line[2][:-1]), float(calibration_line[3][:-1]), float(calibration_line[5][:-1]), float(calibration_line[6][:-1])]
        # JEM update 20180723: Changing to dict, including second-order term for generality:
        # print("calibration_line =", calibration_line, flush=True)
        cal = {"a0x":float(calibration_line[1]), "a1x":float(calibration_line[2]), "a2x":float(calibration_line[3]), 
             "a0y":float(calibration_line[4]), "a1y":float(calibration_line[5]), "a2y":float(calibration_line[6])}
    # TODO: INSERT CORRECTION FROM CENTER-BIN TO LOWER EDGE CALIBRATION HERE.
    # MAKE SURE TO CHECK rebin_and_shift() WHICH MIGHT NOT LIKE NEGATIVE SHIFT COEFF.
    # (alternatively consider using center-bin throughout, but then need to correct when plotting.)
    Ny, Nx = matrix.shape
    y_array = np.linspace(0, Ny-1, Ny)
    y_array = cal["a0y"] + cal["a1y"]*y_array + cal["a2y"]*y_array**2
    x_array = np.linspace(0, Nx-1, Nx)
    x_array = cal["a0x"] + cal["a1x"]*x_array + cal["a2x"]*x_array**2
    # x_array = np.linspace(cal["a0x"], cal["a0x"]+cal["a1x"]*Nx, Nx) # BIG TODO: This is probably center-bin calibration, 
    # x_array = np.linspace(a[2], a[2]+a[3]*(Ny), Ny) # and should be shifted down by half a bin?
                                                    # Update 20171024: Started changing everything to lower bin edge,
                                                    # but started to hesitate. For now I'm inclined to keep it as
                                                    # center-bin everywhere. 
    return matrix, cal, y_array, x_array # Returning y (Ex) first as this is axis 0 in matrix language


def rebin_and_shift(array, E_range, N_final, rebin_axis=0):
    # Function to rebin an M-dimensional array either to larger or smaller binsize.
    # Written by J{\o}rgen E. Midtb{\o}, University of Oslo, j.e.midtbo@fys.uio.no, github.com/jorgenem
    # Latest change made 20161029.

    # Rebinning is done with simple proportionality. E.g. for down-scaling rebinning (N_final < N_initial): 
    # if a bin in the original spacing ends up between two bins in the reduced spacing, 
    # then the counts of that bin are split proportionally between adjacent bins in the 
    # rebinned array. 
    # Upward binning (N_final > N_initial) is done in the same way, dividing the content of bins
    # equally among adjacent bins.

    # Technically it's done by repeating each element of array N_final times and dividing by N_final to 
    # preserve total number of counts, then reshaping the array from M dimensions to M+1 before summing 
    # along the new dimension of length N_initial, resulting in an array of the desired dimensionality.

    # This version (called rebin_and_shift rather than just rebin) takes in also the energy range array (lower bin edge)
    # corresponding to the counts array, in order to be able to change the calibration. What it does is transform the
    # coordinates such that the starting value of the rebinned axis is zero energy. This is done by shifting all
    # bins, so we are discarding some of the eventual counts in the highest energy bins. However, there is usually a margin.

    if isinstance(array, tuple): # Check if input array is actually a tuple, which may happen if rebin_and_shift() is called several times nested for different axes.
        array = array[0]

    
    N_initial = array.shape[rebin_axis] # Initial number of bins along rebin axis

    # TODO: Loop this part over chunks of the Ex axis to avoid running out of memory.
    # Just take the loop from main program in here. Have some test on the dimensionality 
    # to judge whether chunking is necessary?

    # Repeat each bin of array Nfinal times and scale to preserve counts
    array_rebinned = array.repeat(N_final, axis=rebin_axis)/N_final

    if E_range[0] < 0 or E_range[1] < E_range[0]:
        raise Exception("Error in function rebin_and_shift(): Negative zero energy is not supported. (But it should be relatively easy to implement.)")
    else:
        # Calculate number of extra slices in Nf*Ni sized array required to get down to zero energy
        n_extra = int(np.ceil(N_final * (E_range[0]/(E_range[1]-E_range[0]))))
        # Append this matrix of zero counts in front of the array
        dimensions_append = np.array(array_rebinned.shape)
        dimensions_append[rebin_axis] = n_extra
        array_rebinned = np.append(np.zeros(dimensions_append), array_rebinned, axis=rebin_axis)
        array_rebinned = np.split(array_rebinned, [0, N_initial*N_final], axis=rebin_axis)[1]
        dimensions = np.insert(array.shape, rebin_axis, N_final) # Indices to reshape to
        array_rebinned = array_rebinned.reshape(dimensions).sum(axis=(rebin_axis+1)) 
        E_range_shifted_and_scaled = np.linspace(0, E_range[-1]-E_range[0], N_final)
    return array_rebinned, E_range_shifted_and_scaled