import numpy as np 

def read_mama_2D(filename):
    # Reads a MAMA matrix file and returns the matrix as a numpy array, 
    # as well as a list containing the calibration coefficients
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

def read_mama_1D(filename):
    # Reads a MAMA spectrum file and returns the spectrum as a numpy array, 
    # as well as a list containing the calibration coefficients
    # and 1-D arrays of calibrated x values for plotting and similar.
    with open(filename) as file:
        lines = file.readlines()
        a0 = float(lines[6].split(",")[1]) # calibration
        a1 = float(lines[6].split(",")[2]) # coefficients [keV]
        a2 = float(lines[6].split(",")[3]) # coefficients [keV]
        N = int(lines[8][15:]) +1 # 0 is first index
        cal = {"a0x":a0, "a1x":a1, "a2x":a2}
    x_array = np.linspace(0, N-1, N)
    x_array = cal["a0x"] + cal["a1x"]*x_array + cal["a2x"]*x_array**2
    # Read the rest:
    array = np.genfromtxt(filename, comments="!")
    return array, cal, x_array

def write_mama_2D(matrix, filename, y_array, x_array, comment=""):
    import time
    outfile = open(filename, 'w')

    # Write mandatory header:
    # outfile.write('!FILE=Disk \n')
    # outfile.write('!KIND=Spectrum \n')
    # outfile.write('!LABORATORY=Oslo Cyclotron Laboratory (OCL) \n')
    # outfile.write('!EXPERIMENT=pyma \n')
    # outfile.write('!COMMENT=none|RE:alfna-20FN:RN:UN:FN:RN: \n')
    # outfile.write('!TIME=DATE:'+time.strftime("%d-%b-%y %H:%M:%S", time.localtime())+'   \n')
    # outfile.write('!CALIBRATION EkeV=6, %12.6E, %12.6E, 0.000000E+00, %12.6E, %12.6E, 0.000000E+00 \n' %(Egamma_range[0], (Egamma_range[1]-Egamma_range[0]), Ex_range[0], (Ex_range[1]-Ex_range[0])))
    # outfile.write('!PRECISION=16 \n')
    # outfile.write('!DIMENSION=2,0:%4d,0:%4d \n' %(len(matrix[:,0]), len(matrix[0,:])))
    # outfile.write('!CHANNEL=(0:%4d,0:%4d) \n' %(len(matrix[:,0]), len(matrix[0,:])))
    header_string ='!FILE=Disk \n'
    header_string +='!KIND=Spectrum \n'
    header_string +='!LABORATORY=Oslo Cyclotron Laboratory (OCL) \n'
    header_string +='!EXPERIMENT= pyma \n'
    header_string +='!COMMENT={:s} \n'.format(comment)
    header_string +='!TIME=DATE:'+time.strftime("%d-%b-%y %H:%M:%S", time.localtime())+'   \n'
    header_string +='!CALIBRATION EkeV=6, %12.6E, %12.6E, 0.000000E+00, %12.6E, %12.6E, 0.000000E+00 \n' %(x_array[0], (x_array[1]-x_array[0]), y_array[0], (y_array[1]-y_array[0]))
    header_string +='!PRECISION=16 \n'
    header_string +="!DIMENSION=2,0:{:4d},0:{:4d} \n".format(len(matrix[0,:])-1, len(matrix[:,0])-1)
    header_string +='!CHANNEL=(0:%4d,0:%4d) ' %(len(matrix[0,:])-1, len(matrix[:,0])-1)

    footer_string = "!IDEND=\n"

    # Write matrix:
    # matrix.tofile(filename, sep='       ', format="{:14.8E}")
    # matrix.tofile(filename, sep=' ', format="%-17.8E")
    np.savetxt(filename, matrix, fmt="%-17.8E", delimiter=" ", newline="\n", header=header_string, footer=footer_string, comments="")

    outfile.close()


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


def div0( a, b ):
    """ division function designed to ignore / 0, i.e. div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c


def shift_and_smooth3D(array, Eg_array, FWHM, p, shift, smoothing=True):
    from scipy.stats import norm
    # Updated 201807: Trying to vectorize so all Ex bins are handled simultaneously.
    # Takes a 2D array of counts, shifts it (downward only!) with energy 'shift'
    # and smooths it with a gaussian of specified 'FWHM'.
    # This version is vectorized to shift, smooth and scale all points
    # of 'array' individually, and then sum together and return.

    # The arrays from resp.dat are missing the first channel.
    p = np.append(0, p) 
    FWHM = np.append(0, FWHM)

    a1_Eg = (Eg_array[1]-Eg_array[0]) # bin width
    N_Ex, N_Eg = array.shape
    print("array.shape =", array.shape, flush=True)

    # Shift is the same for all energies 
    if shift == "annihilation":
        # For the annihilation peak, all channels should be mapped on E = 511 keV. Of course, gamma channels below 511 keV,
        # and even well above that, cannot produce annihilation counts, but this is taken into account by the fact that p
        # is zero for these channels. Thus, we set i_shift=0 and make a special dimensions_shifted array to map all channels of
        # original array to i(511). 
        i_shift = 0 
    else:
        i_shift = i_from_E(shift, Eg_array) - i_from_E(0, Eg_array) # The number of indices to shift by


    N_Eg_sh = N_Eg - i_shift
    indices_original = np.linspace(i_shift, N_Eg-1, N_Eg-i_shift).astype(int) # Index array for original array, truncated to shifted array length
    if shift == "annihilation": # If this is the annihilation peak then all counts should end up with their centroid at E = 511 keV
        # indices_shifted = (np.ones(N_Eg-i_from_E(511, Eg_array))*i_from_E(511, Eg_array)).astype(int)
        indices_shifted = (np.ones(N_Eg)*i_from_E(511, Eg_array)).astype(int)
    else:
        indices_shifted = np.linspace(0,N_Eg-i_shift-1,N_Eg-i_shift).astype(int) # Index array for shifted array


    if smoothing:
        # Scale each Eg count by the corresponding probability
        # Do this for all Ex bins at once:
        array = array * p[0:N_Eg].reshape(1,N_Eg)
        # Shift array down in energy by i_shift indices,
        # so that index i_shift of array is index 0 of array_shifted.
        # Also flatten array along Ex axis to facilitate multiplication.
        array_shifted_flattened = array[:,indices_original].ravel()
        # Make an array of N_Eg_sh x N_Eg_sh containing gaussian distributions 
        # to multiply each Eg channel by. This array is the same for all Ex bins,
        # so it will be repeated N_Ex times and stacked for multiplication
        # To get correct normalization we multiply by bin width
        pdfarray = a1_Eg* norm.pdf(
                            np.tile(Eg_array[0:N_Eg_sh], N_Eg_sh).reshape((N_Eg_sh, N_Eg_sh)),
                            loc=Eg_array[indices_shifted].reshape(N_Eg_sh,1),
                            scale=FWHM[indices_shifted].reshape(N_Eg_sh,1)/2.355
                        )
                        
        # Remove eventual NaN values:
        pdfarray = np.nan_to_num(pdfarray, copy=False)
        # print("Eg_array[indices_shifted] =", Eg_array[indices_shifted], flush=True)
        # print("pdfarray =", pdfarray, flush=True)
        # Repeat and stack:
        pdfarray_repeated_stacked = np.tile(pdfarray, (N_Ex,1))

        # Multiply array of counts with pdfarray:
        multiplied = pdfarray_repeated_stacked*array_shifted_flattened.reshape(N_Ex*N_Eg_sh,1)

        # Finally, for each Ex bin, we now need to sum the contributions from the smoothing
        # of each Eg bin to get a total Eg spectrum containing the entire smoothed spectrum:
        # Do this by reshaping into 3-dimensional array where each Eg bin (axis 0) contains a 
        # N_Eg_sh x N_Eg_sh matrix, where each row is the smoothed contribution from one 
        # original Eg pixel. We sum the columns of each of these matrices:
        array_out = multiplied.reshape((N_Ex, N_Eg_sh, N_Eg_sh)).sum(axis=1)
        # print("array_out.shape =", array_out.shape)
        # print("array.shape[0],array.shape[1]-N_Eg_sh =", array.shape[0],array.shape[1]-N_Eg_sh)

    else:
        # array_out = np.zeros(N)
        # for i in range(N):
        #     try:
        #         array_out[i-i_shift] = array[i] #* p[i+1]
        #     except IndexError:
        #         pass

        # Instead of above, vectorizing:
        array_out = p[indices_original].reshape(1,N_Eg_sh)*array[:,indices_original]

    # Append zeros to the end of Eg axis so we match the length of the original array:
    if i_shift > 0:
        array_out = np.concatenate((array_out, np.zeros((N_Ex, N_Eg-N_Eg_sh))),axis=1)


    print("array_out.shape =", array_out.shape, flush=True)
    return array_out





def i_from_E(E, E_array):
    # Function which returns the index of the E_array value closest to given E
    where_array = np.where(E_array > E)[0]
    # print where_array, len(where_array)
    if len(where_array) > 0:
        i = where_array[0]
        if np.abs(E_array[i]-E) > np.abs(E_array[i-1]-E):
            i -= 1
    else:
        i = len(E_array)-1
    return i