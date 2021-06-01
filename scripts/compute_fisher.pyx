"""
@author: Alba Kalaja

"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, log, sqrt
from scipy import integrate

import itertools as it
from itertools import permutations

import py3nj
from py3nj import wigner

DTYPE = np.double
ITYPE = np.int
ctypedef cnp.double_t DTYPE_t
ctypedef cnp.int_t ITYPE_t
#---------------------------------------

# Load pre-computed files: alpha, beta, gamma and delta
# Temperature
root_directory = #insert home directory here
cdef cnp.ndarray alpha_function_temp = np.loadtxt(f'{root_directory}/alphabetagammadelta/alpha_temp_lmax5000.txt')
cdef cnp.ndarray beta_function_temp = np.loadtxt(f'{root_directory}/alphabetagammadelta/beta_temp_lmax5000.txt')
cdef cnp.ndarray gamma_function_temp = np.loadtxt(f'{root_directory}/alphabetagammadelta/gamma_temp_lmax5000.txt')
cdef cnp.ndarray delta_function_temp = np.loadtxt(f'{root_directory}/alphabetagammadelta/delta_temp_lmax5000.txt')
# Polarization
cdef cnp.ndarray alpha_function_pol = np.loadtxt(f'{root_directory}/alphabetagammadelta/alpha_pol_lmax5000.txt')
cdef cnp.ndarray beta_function_pol = np.loadtxt(f'{root_directory}/alphabetagammadelta/beta_pol_lmax5000.txt')
cdef cnp.ndarray gamma_function_pol = np.loadtxt(f'{root_directory}/alphabetagammadelta/gamma_pol_lmax5000.txt')
cdef cnp.ndarray delta_function_pol = np.loadtxt(f'{root_directory}/alphabetagammadelta/delta_pol_lmax5000.txt')

# we add two nan rows in order to loop over the triplet l1,l2,l3
cdef cnp.ndarray alpha_temp = np.r_[np.full((2,602), np.nan),alpha_function_temp]
cdef cnp.ndarray beta_temp = np.r_[np.full((2,602), np.nan),beta_function_temp]
cdef cnp.ndarray gamma_temp = np.r_[np.full((2,602), np.nan),gamma_function_temp]
cdef cnp.ndarray delta_temp = np.r_[np.full((2,602), np.nan),delta_function_temp]

# we add two nan rows in order to loop over the triplet l1,l2,l3
cdef cnp.ndarray alpha_pol_l = np.r_[np.full((2,602), np.nan),alpha_function_pol] 
cdef cnp.ndarray beta_pol_l = np.r_[np.full((2,602), np.nan),beta_function_pol] 
cdef cnp.ndarray gamma_pol_l = np.r_[np.full((2,602), np.nan),gamma_function_pol] 
cdef cnp.ndarray delta_pol_l = np.r_[np.full((2,602), np.nan),delta_function_pol]

# polarization data need to factor out
cdef cnp.ndarray alpha_pol = np.zeros((5001,602),'float64')
cdef cnp.ndarray beta_pol = np.zeros((5001,602),'float64')
cdef cnp.ndarray gamma_pol = np.zeros((5001,602),'float64')
cdef cnp.ndarray delta_pol = np.zeros((5001,602),'float64')
cdef int l
for l in range(0,5001,1):
    alpha_pol[l][:] = np.sqrt((l-1.0)*l*(l+1.0)*(l+2.0))*alpha_pol_l[l][:]
    beta_pol[l][:] = np.sqrt((l-1.0)*l*(l+1.0)*(l+2.0))*beta_pol_l[l][:]
    gamma_pol[l][:] = np.sqrt((l-1.0)*l*(l+1.0)*(l+2.0))*gamma_pol_l[l][:]
    delta_pol[l][:] = np.sqrt((l-1.0)*l*(l+1.0)*(l+2.0))*delta_pol_l[l][:]


# create a single array for temperature and polarization data
cdef cnp.ndarray alpha_temp_pol_data = np.array([alpha_temp,alpha_pol]) # T/E, l, r
cdef cnp.ndarray beta_temp_pol_data = np.array([beta_temp,beta_pol]) # T/E, l, r
cdef cnp.ndarray gamma_temp_pol_data = np.array([gamma_temp,gamma_pol]) # T/E, l, r
cdef cnp.ndarray delta_temp_pol_data = np.array([delta_temp,delta_pol]) # T/E, l, r

# --------------------------------------

# Load pre-computed spectra: TT, EE and TE
cdef cnp.ndarray raw_cmb_power_spectrum_TT = np.loadtxt(f'{root_directory}/power_spectra/raw_cmb_power_spectrum_TT.txt')
cdef cnp.ndarray raw_cmb_power_spectrum_EE = np.loadtxt(f'{root_directory}/power_spectra/raw_cmb_power_spectrum_EE.txt')
cdef cnp.ndarray raw_cmb_power_spectrum_TE = np.loadtxt(f'{root_directory}/power_spectra/raw_cmb_power_spectrum_TE.txt')

cdef cnp.ndarray Cl_data = np.zeros((2,2,5001),dtype = DTYPE)
for l in range(0,5001,1):
    Cl_data[0,0,l] = raw_cmb_power_spectrum_TT[l]
    Cl_data[1,1,l] = raw_cmb_power_spectrum_EE[l]
    Cl_data[1,0,l] = raw_cmb_power_spectrum_TE[l]
    Cl_data[0,1,l] = raw_cmb_power_spectrum_TE[l]

# --------------------------------------
# Compute r_sampling according to Table 1 of Liguori et al. PRD, 76, 105016 (2007)
cdef double tau0 = 14142.0762
cdef double r = tau0 + 500.0
cdef double r_count = 0.0
cdef list r_values_list = [r] #starting from r
cdef list dr = [3.5]
while r > 105.0: #equal to the last r_sample
    r_count+=1.0
    if r_count <= 450.0:
        r_sample = 3.5 #sample densely during recombination
    elif r_count <= 485.0:
        r_sample = 105.0
    elif r_count <= 515.0:
        r_sample = 10.0 #sample densely during reionization
    else:
        r_sample = 105.0
    r-=r_sample
    r_values_list.append(r)
    dr.append(r_sample)
cdef cnp.ndarray r_values_data = np.array(r_values_list)
#---------------------------------------

cpdef double inverse_covariance_matrix(int l,
                                       int X, # T,E
                                       int Y # T,E
                                      ):
    cdef double invCov = 0.0
    cdef cnp.ndarray[DTYPE_t, ndim=3] Cl = Cl_data

    if X == 0 and Y == 0:
        invCov = Cl[1,1,l]/(Cl[0,0,l]*Cl[1,1,l]-Cl[1,0,l]**2)

    elif X == 1 and Y == 1:
        invCov = Cl[0,0,l]/(Cl[0,0,l]*Cl[1,1,l]-Cl[1,0,l]**2)

    else:
        invCov = -Cl[1,0,l]/(Cl[0,0,l]*Cl[1,1,l]-Cl[1,0,l]**2)

    return invCov

# --------------------------------------------------------------------------------

# COMPUTE FISHER MATRIX

# --------------------------------------------------------------------------------

# LOCAL SHAPE 
# - T/E -
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef double compute_fisher_local(int l1,# l-variable for the parallel loop
                                  int lmin,
                                  int lmax,
                                  int source # T = 0, E = 1
                                 ):

    # Fisher variables
    cdef double fisher = 0.0 # initialize
    cdef int l2, l3
    cdef int delta_l = 0
    cdef double bispectrum = 0 # bispectrum 
    cdef cnp.ndarray[DTYPE_t,ndim = 1] r_values = r_values_data

    cdef cnp.ndarray[DTYPE_t,ndim = 2] alpha_temp_pol = alpha_temp_pol_data[source]
    cdef cnp.ndarray[DTYPE_t,ndim = 2] beta_temp_pol = beta_temp_pol_data[source]
    cdef cnp.ndarray[DTYPE_t,ndim = 2] gamma_temp_pol = gamma_temp_pol_data[source]
    cdef cnp.ndarray[DTYPE_t,ndim = 2] delta_temp_pol = delta_temp_pol_data[source]
    
    # Wigner-3j
    cdef int int_l2, int_l1, l3min_index, l1max
    cdef cnp.ndarray[ITYPE_t,ndim = 1] l3_values_w
    cdef cnp.ndarray[DTYPE_t,ndim = 1] wigner3j
    
    # Power spectra for covariance 
    cdef cnp.ndarray[DTYPE_t, ndim=3] Cl = Cl_data

    # l2, l3 ranges are fixed by
    # * symmetry: l3=<l2=<l1
    # * even parity: l1+l2+l3 = even
    # * triangle condition: abs(l2-l3)=<l1=<l2+l3 => abs(l1-l2)<=l3<=l2 and lmin<=l2<=l1

    for l2 in range(lmin,l1+1): # range doesn't include the upper extreme

        # -----------------
        # Wigner-3j, for more info check https://py3nj.readthedocs.io
        int_l2 = int(l2*2)
        int_l1 = int(l1*2)
        l3_values_w, wigner3j = py3nj.wigner._drc3jj(int_l2,int_l1,0,0)
        l3min_index = max(abs(l1-l2), 0) # needed to index wigner3j
        # -----------------

        for l3 in range(l3min_index,l2+1):
            l1max = l2+l3
            if (l1+l2+l3)%2==0 and l3>=lmin and l1<=l1max:

                #------------------------------------------
                if l1==l2 and l2==l3:
                    delta_l = 6
                elif l1==l2 or l1==l3 or l2==l3:
                    delta_l = 2
                else:
                    delta_l = 1
                #------------------------------------------
                # Inverse of the covariance matrix
                invCov = 1.0/(Cl[source,source,l3]*Cl[source,source,l2]*\
                              Cl[source,source,l1])
            
                bispectrum = 6.0/5.0*integrate.trapz(np.power(r_values,2.0)*
                                                         (beta_temp_pol[l3]*beta_temp_pol[l2]*alpha_temp_pol[l1]+
                                                          beta_temp_pol[l1]*beta_temp_pol[l3]*alpha_temp_pol[l2]+
                                                          beta_temp_pol[l2]*beta_temp_pol[l1]*alpha_temp_pol[l3]),
                                                         x = r_values)

                # Compute fisher

                fisher+=(2.0*l1+1.0)*(2.0*l2+1.0)*(2.0*l3+1.0)/(4.0*np.pi)*wigner3j[l3-l3min_index]*\
                        wigner3j[l3-l3min_index]*(bispectrum*bispectrum*invCov)/delta_l

    return fisher
# --------------------------------------------------------------------------
#
# --------------------------------------------------------------------------
# - T+E -
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef double compute_fisher_local_TE(int l1,# l-variable for the parallel loop
                                     int lmin,
                                     int lmax,
                                    ):

    # Fisher variables
    cdef double fisher = 0.0 # initialize
    cdef int l2, l3
    cdef int delta_l = 0
    cdef double invCov = 0
    cdef cnp.ndarray[DTYPE_t,ndim = 3] bispectrum = np.empty((2,2,2),dtype = DTYPE) # <TTT>, <TTE>, <EEE>, ..., etc.
    cdef double bispectrum2 = 0 # bispectrum square
    cdef cnp.ndarray[DTYPE_t,ndim = 1] r_values = r_values_data

    cdef cnp.ndarray[DTYPE_t,ndim = 3] alpha_temp_pol = alpha_temp_pol_data
    cdef cnp.ndarray[DTYPE_t,ndim = 3] beta_temp_pol = beta_temp_pol_data
    cdef cnp.ndarray[DTYPE_t,ndim = 3] gamma_temp_pol = gamma_temp_pol_data
    cdef cnp.ndarray[DTYPE_t,ndim = 3] delta_temp_pol = delta_temp_pol_data
    
    # Wigner-3j
    cdef int int_l2, int_l1, l3min_index, l1max
    cdef cnp.ndarray[ITYPE_t,ndim = 1] l3_values_w
    cdef cnp.ndarray[DTYPE_t,ndim = 1] wigner3j
    
    # loop on T = 0 and E = 1 variables
    cdef list x = [0,1]
    cdef list y = [0,1]
    cdef list z = [0,1]
    cdef int i, j, k, p, q, s
    
    # l2, l3 ranges are fixed by
    # * symmetry: l3=<l2=<l1
    # * even parity: l1+l2+l3 = even
    # * triangle condition: abs(l2-l3)=<l1=<l2+l3 => abs(l1-l2)<=l3<=l2 and lmin<=l2<=l1

    for l2 in range(lmin,l1+1): # range doesn't include the upper extreme

        # -----------------
        # Wigner-3j, for more info check https://py3nj.readthedocs.io
        int_l2 = int(l2*2)
        int_l1 = int(l1*2)
        l3_values_w, wigner3j = py3nj.wigner._drc3jj(int_l2,int_l1,0,0)
        l3min_index = max(abs(l1-l2), 0) # needed to index wigner3j
        # -----------------

        for l3 in range(l3min_index,l2+1):
            l1max = l2+l3
            if (l1+l2+l3)%2==0 and l3>=lmin and l1<=l1max:

                #------------------------------------------
                if l1==l2 and l2==l3:
                    delta_l = 6
                elif l1==l2 or l1==l3 or l2==l3:
                    delta_l = 2
                else:
                    delta_l = 1
                #------------------------------------------

                bispectrum.fill(np.nan)

                for i,j,k,p,q,s in it.product(x,y,z,x,y,z): # loop over T, E

                    # Inverse of the covariance matrix
                    invCov = inverse_covariance_matrix(l1,k,s)*inverse_covariance_matrix(l2,j,q)*\
                             inverse_covariance_matrix(l3,i,p)


                    # In order to avoid repeating calculations
                    # check if the bispectrum for a certain i,j,k has already been computed
                    if np.isnan(bispectrum[i,j,k]): # if not, then compute it
                        bispectrum[i,j,k] = 6.0/5.0*integrate.trapz(np.power(r_values,2.0)*
                                                         (beta_temp_pol[i,l3]*beta_temp_pol[j,l2]*
                                                          alpha_temp_pol[k,l1]+
                                                          beta_temp_pol[k,l1]*beta_temp_pol[i,l3]*
                                                          alpha_temp_pol[j,l2]+
                                                          beta_temp_pol[j,l2]*beta_temp_pol[k,l1]*
                                                          alpha_temp_pol[i,l3]),
                                                         x = r_values)
                    if np.isnan(bispectrum[p,q,s]): # same for p,q,s triplet
                        bispectrum[p,q,s] = 6.0/5.0*integrate.trapz(np.power(r_values,2.0)*
                                                         (beta_temp_pol[p,l3]*beta_temp_pol[q,l2]*
                                                          alpha_temp_pol[s,l1]+
                                                          beta_temp_pol[s,l1]*beta_temp_pol[p,l3]*
                                                          alpha_temp_pol[q,l2]+
                                                          beta_temp_pol[q,l2]*beta_temp_pol[s,l1]*
                                                          alpha_temp_pol[p,l3]),
                                                         x = r_values)

                    bispectrum2 = bispectrum[i,j,k]*bispectrum[p,q,s]

                    # Compute fisher

                    fisher+=(2.0*l1+1.0)*(2.0*l2+1.0)*(2.0*l3+1.0)/(4.0*np.pi)*wigner3j[l3-l3min_index]*\
                        wigner3j[l3-l3min_index]*(bispectrum2*invCov)/delta_l

    return fisher

# --------------------------------------------------------------------------
#
# --------------------------------------------------------------------------

# EQUILATERAL SHAPE
# - T/E -
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef double compute_fisher_equil(int l1,# l-variable for the parallel loop
                                  int lmin,
                                  int lmax,
                                  int source # T = 0, E = 1
                                 ):

    # Fisher variables
    cdef double fisher = 0.0 # initialize
    cdef int l2, l3
    cdef int delta_l = 0
    cdef double bispectrum = 0 # bispectrum 
    cdef cnp.ndarray[DTYPE_t,ndim = 1] r_values = r_values_data

    cdef cnp.ndarray[DTYPE_t,ndim = 2] alpha_temp_pol = alpha_temp_pol_data[source]
    cdef cnp.ndarray[DTYPE_t,ndim = 2] beta_temp_pol = beta_temp_pol_data[source]
    cdef cnp.ndarray[DTYPE_t,ndim = 2] gamma_temp_pol = gamma_temp_pol_data[source]
    cdef cnp.ndarray[DTYPE_t,ndim = 2] delta_temp_pol = delta_temp_pol_data[source]
    
    # Wigner-3j
    cdef int int_l2, int_l1, l3min_index, l1max
    cdef cnp.ndarray[ITYPE_t,ndim = 1] l3_values_w
    cdef cnp.ndarray[DTYPE_t,ndim = 1] wigner3j
    
    # Power spectra for covariance 
    cdef cnp.ndarray[DTYPE_t, ndim=3] Cl = Cl_data

    # l2, l3 ranges are fixed by
    # * symmetry: l3=<l2=<l1
    # * even parity: l1+l2+l3 = even
    # * triangle condition: abs(l2-l3)=<l1=<l2+l3 => abs(l1-l2)<=l3<=l2 and lmin<=l2<=l1

    for l2 in range(lmin,l1+1): # range doesn't include the upper extreme

        # -----------------
        # Wigner-3j, for more info check https://py3nj.readthedocs.io
        int_l2 = int(l2*2)
        int_l1 = int(l1*2)
        l3_values_w, wigner3j = py3nj.wigner._drc3jj(int_l2,int_l1,0,0)
        l3min_index = max(abs(l1-l2), 0) # needed to index wigner3j
        # -----------------

        for l3 in range(l3min_index,l2+1):
            l1max = l2+l3
            if (l1+l2+l3)%2==0 and l3>=lmin and l1<=l1max:

                #------------------------------------------
                if l1==l2 and l2==l3:
                    delta_l = 6
                elif l1==l2 or l1==l3 or l2==l3:
                    delta_l = 2
                else:
                    delta_l = 1
                #------------------------------------------
                # Inverse of the covariance matrix
                invCov = 1.0/(Cl[source,source,l3]*Cl[source,source,l2]*\
                              Cl[source,source,l1])
            
                bispectrum = 18.0/5.0*integrate.trapz(np.power(r_values,2.0)*
                                                    (-beta_temp_pol[l3]*beta_temp_pol[l2]*
                                                     alpha_temp_pol[l1]-
                                                     beta_temp_pol[l1]*beta_temp_pol[l3]*
                                                     alpha_temp_pol[l2]-
                                                     beta_temp_pol[l2]*beta_temp_pol[l1]*
                                                     alpha_temp_pol[l3]-
                                                     2.0*delta_temp_pol[l1]*delta_temp_pol[l2]*
                                                     delta_temp_pol[l3]+
                                                     beta_temp_pol[l1]*gamma_temp_pol[l2]*
                                                     delta_temp_pol[l3]+
                                                     beta_temp_pol[l1]*gamma_temp_pol[l3]*
                                                     delta_temp_pol[l2]+
                                                     beta_temp_pol[l2]*gamma_temp_pol[l1]*
                                                     delta_temp_pol[l3]+
                                                     beta_temp_pol[l2]*gamma_temp_pol[l3]*
                                                     delta_temp_pol[l1]+
                                                     beta_temp_pol[l3]*gamma_temp_pol[l1]*
                                                     delta_temp_pol[l2]+
                                                     beta_temp_pol[l3]*gamma_temp_pol[l2]*
                                                     delta_temp_pol[l1]),
                                                          x = r_values)

                # Compute fisher

                fisher+=(2.0*l1+1.0)*(2.0*l2+1.0)*(2.0*l3+1.0)/(4.0*np.pi)*wigner3j[l3-l3min_index]*\
                        wigner3j[l3-l3min_index]*(bispectrum*bispectrum*invCov)/delta_l

    return fisher

# --------------------------------------------------------------------------
#
# --------------------------------------------------------------------------
# - T+E -
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef double compute_fisher_equil_TE(int l1,# variable for the parallel loop
                                     int lmin,
                                     int lmax,
                                    ):
    
    # Fisher variables
    cdef double fisher = 0.0 # initialize
    cdef int l2, l3
    cdef int delta_l = 0
    cdef double invCov = 0
    cdef cnp.ndarray[DTYPE_t,ndim = 3] bispectrum = np.empty((2,2,2),dtype = DTYPE) # <TTT>, <TTE>, <EEE>, ..., etc.
    cdef double bispectrum2 = 0 # bispectrum square
    cdef cnp.ndarray[DTYPE_t,ndim = 1] r_values = r_values_data

    cdef cnp.ndarray[DTYPE_t,ndim = 3] alpha_temp_pol = alpha_temp_pol_data
    cdef cnp.ndarray[DTYPE_t,ndim = 3] beta_temp_pol = beta_temp_pol_data
    cdef cnp.ndarray[DTYPE_t,ndim = 3] gamma_temp_pol = gamma_temp_pol_data
    cdef cnp.ndarray[DTYPE_t,ndim = 3] delta_temp_pol = delta_temp_pol_data
    
    # Wigner-3j
    cdef int int_l2, int_l1, l3min_index, l1max
    cdef cnp.ndarray[ITYPE_t,ndim = 1] l3_values_w
    cdef cnp.ndarray[DTYPE_t,ndim = 1] wigner3j
    
    # loop on T = 0 and E = 1 variables
    cdef list x = [0,1]
    cdef list y = [0,1]
    cdef list z = [0,1]
    cdef int i, j, k, p, q, s
    
    # l2, l3 ranges are fixed by
    # * symmetry: l3=<l2=<l1
    # * even parity: l1+l2+l3 = even
    # * triangle condition: abs(l2-l3)=<l1=<l2+l3 => abs(l1-l2)<=l3<=l2 and lmin<=l2<=l1

    for l2 in range(lmin,l1+1): # range doesn't include the upper extreme 

        # -----------------
        # wigner 3j, for more info check https://py3nj.readthedocs.io
        int_l2 = int(l2*2)
        int_l1 = int(l1*2)
        l3_values_w, wigner3j = py3nj.wigner._drc3jj(int_l2,int_l1,0,0) 
        l3min_index = max(abs(l1-l2), 0) # needed to index wigner3j
        # -----------------

        for l3 in range(l3min_index,l2+1): 
#         for l3 from l3min_index <= l3 <= l2 by 1:
            l1max = l2+l3
            if (l1+l2+l3)%2==0 and l3>=lmin and l1<=l1max: 

                #-------------------------------------------------------------------------------------
                if l1==l2 and l2==l3:
                    delta_l = 6
                elif l1==l2 or l1==l3 or l2==l3:
                    delta_l = 2
                else:
                    delta_l = 1
                #-------------------------------------------------------------------------------------

                bispectrum.fill(np.nan)

                for i,j,k,p,q,s in it.product(x,y,z,x,y,z): # loop over T, E
                    
                    # Inverse of the covariance matrix
                    invCov = inverse_covariance_matrix(l1,k,s)*inverse_covariance_matrix(l2,j,q)*\
                             inverse_covariance_matrix(l3,i,p)


                    # In order to avoid repeating calculations
                    # check if the bispectrum for a certain i,j,k has already been computed
                    if np.isnan(bispectrum[i,j,k]): # if not, then compute it
                        bispectrum[i,j,k] = 18.0/5.0*integrate.trapz(np.power(r_values,2.0)*
                                                    (-beta_temp_pol[i,l3]*beta_temp_pol[j,l2]*
                                                     alpha_temp_pol[k,l1]-
                                                     beta_temp_pol[k,l1]*beta_temp_pol[i,l3]*
                                                     alpha_temp_pol[j,l2]-
                                                     beta_temp_pol[j,l2]*beta_temp_pol[k,l1]*
                                                     alpha_temp_pol[i,l3]-
                                                     2.0*delta_temp_pol[k,l1]*delta_temp_pol[j,l2]*
                                                     delta_temp_pol[i,l3]+
                                                     beta_temp_pol[k,l1]*gamma_temp_pol[j,l2]*
                                                     delta_temp_pol[i,l3]+
                                                     beta_temp_pol[k,l1]*gamma_temp_pol[i,l3]*
                                                     delta_temp_pol[j,l2]+
                                                     beta_temp_pol[j,l2]*gamma_temp_pol[k,l1]*
                                                     delta_temp_pol[i,l3]+
                                                     beta_temp_pol[j,l2]*gamma_temp_pol[i,l3]*
                                                     delta_temp_pol[k,l1]+
                                                     beta_temp_pol[i,l3]*gamma_temp_pol[k,l1]*
                                                     delta_temp_pol[j,l2]+
                                                     beta_temp_pol[i,l3]*gamma_temp_pol[j,l2]*
                                                     delta_temp_pol[k,l1]),
                                                          x = r_values)
                        
                    if np.isnan(bispectrum[p,q,s]): # same for p,q,s triplet
                        bispectrum[p,q,s] = 18.0/5.0*integrate.trapz(np.power(r_values,2.0)*
                                                    (-beta_temp_pol[p,l3]*beta_temp_pol[q,l2]*
                                                     alpha_temp_pol[s,l1]-
                                                     beta_temp_pol[s,l1]*beta_temp_pol[p,l3]*
                                                     alpha_temp_pol[q,l2]-
                                                     beta_temp_pol[q,l2]*beta_temp_pol[s,l1]*
                                                     alpha_temp_pol[p,l3]-
                                                     2.0*delta_temp_pol[s,l1]*delta_temp_pol[q,l2]*
                                                     delta_temp_pol[p,l3]+
                                                     beta_temp_pol[s,l1]*gamma_temp_pol[q,l2]*
                                                     delta_temp_pol[p,l3]+
                                                     beta_temp_pol[s,l1]*gamma_temp_pol[p,l3]*
                                                     delta_temp_pol[q,l2]+
                                                     beta_temp_pol[q,l2]*gamma_temp_pol[s,l1]*
                                                     delta_temp_pol[p,l3]+
                                                     beta_temp_pol[q,l2]*gamma_temp_pol[p,l3]*
                                                     delta_temp_pol[s,l1]+
                                                     beta_temp_pol[p,l3]*gamma_temp_pol[s,l1]*
                                                     delta_temp_pol[q,l2]+
                                                     beta_temp_pol[p,l3]*gamma_temp_pol[q,l2]*
                                                     delta_temp_pol[s,l1]),
                                                          x = r_values)
                        
                    bispectrum2 = bispectrum[i,j,k]*bispectrum[p,q,s]
                    
                    # Compute fisher

                    fisher+=(2.0*l1+1.0)*(2.0*l2+1.0)*(2.0*l3+1.0)/(4.0*np.pi)*wigner3j[l3-l3min_index]*\
                        wigner3j[l3-l3min_index]*(bispectrum2*invCov)/delta_l

    return fisher

#----------------------------------------------------------------------------------------

# ORTHOGONAL SHAPE
# - T/E -
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef double compute_fisher_ortho(int l1,# l-variable for the parallel loop
                                  int lmin,
                                  int lmax,
                                  int source # T = 0, E = 1
                                 ):

    # Fisher variables
    cdef double fisher = 0.0 # initialize
    cdef int l2, l3
    cdef int delta_l = 0
    cdef double bispectrum = 0 # bispectrum 
    cdef cnp.ndarray[DTYPE_t,ndim = 1] r_values = r_values_data

    cdef cnp.ndarray[DTYPE_t,ndim = 2] alpha_temp_pol = alpha_temp_pol_data[source]
    cdef cnp.ndarray[DTYPE_t,ndim = 2] beta_temp_pol = beta_temp_pol_data[source]
    cdef cnp.ndarray[DTYPE_t,ndim = 2] gamma_temp_pol = gamma_temp_pol_data[source]
    cdef cnp.ndarray[DTYPE_t,ndim = 2] delta_temp_pol = delta_temp_pol_data[source]
    
    # Wigner-3j
    cdef int int_l2, int_l1, l3min_index, l1max
    cdef cnp.ndarray[ITYPE_t,ndim = 1] l3_values_w
    cdef cnp.ndarray[DTYPE_t,ndim = 1] wigner3j
    
    # Power spectra for covariance 
    cdef cnp.ndarray[DTYPE_t, ndim=3] Cl = Cl_data

    # l2, l3 ranges are fixed by
    # * symmetry: l3=<l2=<l1
    # * even parity: l1+l2+l3 = even
    # * triangle condition: abs(l2-l3)=<l1=<l2+l3 => abs(l1-l2)<=l3<=l2 and lmin<=l2<=l1

    for l2 in range(lmin,l1+1): # range doesn't include the upper extreme

        # -----------------
        # Wigner-3j, for more info check https://py3nj.readthedocs.io
        int_l2 = int(l2*2)
        int_l1 = int(l1*2)
        l3_values_w, wigner3j = py3nj.wigner._drc3jj(int_l2,int_l1,0,0)
        l3min_index = max(abs(l1-l2), 0) # needed to index wigner3j
        # -----------------

        for l3 in range(l3min_index,l2+1):
            l1max = l2+l3
            if (l1+l2+l3)%2==0 and l3>=lmin and l1<=l1max:

                #------------------------------------------
                if l1==l2 and l2==l3:
                    delta_l = 6
                elif l1==l2 or l1==l3 or l2==l3:
                    delta_l = 2
                else:
                    delta_l = 1
                #------------------------------------------
                # Inverse of the covariance matrix
                invCov = 1.0/(Cl[source,source,l3]*Cl[source,source,l2]*\
                              Cl[source,source,l1])
            
                bispectrum = 18.0/5.0*integrate.trapz(np.power(r_values,2.0)*
                                                    (-3.0*beta_temp_pol[l3]*beta_temp_pol[l2]*
                                                     alpha_temp_pol[l1]-
                                                     3.0*beta_temp_pol[l1]*beta_temp_pol[l3]*
                                                     alpha_temp_pol[l2]-
                                                     3.0*beta_temp_pol[l2]*beta_temp_pol[l1]*
                                                     alpha_temp_pol[l3]-
                                                     8.0*delta_temp_pol[l1]*delta_temp_pol[l2]*
                                                     delta_temp_pol[l3]+
                                                     3.0*beta_temp_pol[l1]*gamma_temp_pol[l2]*
                                                     delta_temp_pol[l3]+
                                                     3.0*beta_temp_pol[l1]*gamma_temp_pol[l3]*
                                                     delta_temp_pol[l2]+
                                                     3.0*beta_temp_pol[l2]*gamma_temp_pol[l1]*
                                                     delta_temp_pol[l3]+
                                                     3.0*beta_temp_pol[l2]*gamma_temp_pol[l3]*
                                                     delta_temp_pol[l1]+
                                                     3.0*beta_temp_pol[l3]*gamma_temp_pol[l1]*
                                                     delta_temp_pol[l2]+
                                                     3.0*beta_temp_pol[l3]*gamma_temp_pol[l2]*
                                                     delta_temp_pol[l1]),
                                                          x = r_values)

                # Compute fisher

                fisher+=(2.0*l1+1.0)*(2.0*l2+1.0)*(2.0*l3+1.0)/(4.0*np.pi)*wigner3j[l3-l3min_index]*\
                        wigner3j[l3-l3min_index]*(bispectrum*bispectrum*invCov)/delta_l

    return fisher
# --------------------------------------------------------------------------
#
# --------------------------------------------------------------------------
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef double compute_fisher_ortho_TE(int l1,# variable for the parallel loop
                                     int lmin,
                                     int lmax,
                                    ):
    
    # Fisher variables
    cdef double fisher = 0.0 # initialize
    cdef int l2, l3
    cdef int delta_l = 0
    cdef double invCov = 0
    cdef cnp.ndarray[DTYPE_t,ndim = 3] bispectrum = np.empty((2,2,2),dtype = DTYPE) # <TTT>, <TTE>, <EEE>, ..., etc.
    cdef double bispectrum2 = 0 # bispectrum square
    cdef cnp.ndarray[DTYPE_t,ndim = 1] r_values = r_values_data

    cdef cnp.ndarray[DTYPE_t,ndim = 3] alpha_temp_pol = alpha_temp_pol_data
    cdef cnp.ndarray[DTYPE_t,ndim = 3] beta_temp_pol = beta_temp_pol_data
    cdef cnp.ndarray[DTYPE_t,ndim = 3] gamma_temp_pol = gamma_temp_pol_data
    cdef cnp.ndarray[DTYPE_t,ndim = 3] delta_temp_pol = delta_temp_pol_data
    
    # Wigner-3j
    cdef int int_l2, int_l1, l3min_index, l1max
    cdef cnp.ndarray[ITYPE_t,ndim = 1] l3_values_w
    cdef cnp.ndarray[DTYPE_t,ndim = 1] wigner3j
    
    # loop on T = 0 and E = 1 variables
    cdef list x = [0,1]
    cdef list y = [0,1]
    cdef list z = [0,1]
    cdef int i, j, k, p, q, s
    
    # l2, l3 ranges are fixed by
    # * symmetry: l3=<l2=<l1
    # * even parity: l1+l2+l3 = even
    # * triangle condition: abs(l2-l3)=<l1=<l2+l3 => abs(l1-l2)<=l3<=l2 and lmin<=l2<=l1

    for l2 in range(lmin,l1+1): # range doesn't include the upper extreme 

        # -----------------
        # wigner 3j, for more info check https://py3nj.readthedocs.io
        int_l2 = int(l2*2)
        int_l1 = int(l1*2)
        l3_values_w, wigner3j = py3nj.wigner._drc3jj(int_l2,int_l1,0,0) 
        l3min_index = max(abs(l1-l2), 0) # needed to index wigner3j
        # -----------------

        for l3 in range(l3min_index,l2+1): 
#         for l3 from l3min_index <= l3 <= l2 by 1:
            l1max = l2+l3
            if (l1+l2+l3)%2==0 and l3>=lmin and l1<=l1max: 

                #-------------------------------------------------------------------------------------
                if l1==l2 and l2==l3:
                    delta_l = 6
                elif l1==l2 or l1==l3 or l2==l3:
                    delta_l = 2
                else:
                    delta_l = 1
                #-------------------------------------------------------------------------------------

                bispectrum.fill(np.nan)

                for i,j,k,p,q,s in it.product(x,y,z,x,y,z): # loop over T, E
                    
                    # Inverse of the covariance matrix
                    invCov = inverse_covariance_matrix(l1,k,s)*inverse_covariance_matrix(l2,j,q)*\
                             inverse_covariance_matrix(l3,i,p)


                    # In order to avoid repeating calculations
                    # check if the bispectrum for a certain i,j,k has already been computed
                    if np.isnan(bispectrum[i,j,k]): # if not, then compute it
                        bispectrum[i,j,k] = 18.0/5.0*integrate.trapz(np.power(r_values,2.0)*
                                                    (-3.0*beta_temp_pol[i,l3]*beta_temp_pol[j,l2]*
                                                     alpha_temp_pol[k,l1]-
                                                     3.0*beta_temp_pol[k,l1]*beta_temp_pol[i,l3]*
                                                     alpha_temp_pol[j,l2]-
                                                     3.0*beta_temp_pol[j,l2]*beta_temp_pol[k,l1]*
                                                     alpha_temp_pol[i,l3]-
                                                     8.0*delta_temp_pol[k,l1]*delta_temp_pol[j,l2]*
                                                     delta_temp_pol[i,l3]+
                                                     3.0*beta_temp_pol[k,l1]*gamma_temp_pol[j,l2]*
                                                     delta_temp_pol[i,l3]+
                                                     3.0*beta_temp_pol[k,l1]*gamma_temp_pol[i,l3]*
                                                     delta_temp_pol[j,l2]+
                                                     3.0*beta_temp_pol[j,l2]*gamma_temp_pol[k,l1]*
                                                     delta_temp_pol[i,l3]+
                                                     3.0*beta_temp_pol[j,l2]*gamma_temp_pol[i,l3]*
                                                     delta_temp_pol[k,l1]+
                                                     3.0*beta_temp_pol[i,l3]*gamma_temp_pol[k,l1]*
                                                     delta_temp_pol[j,l2]+
                                                     3.0*beta_temp_pol[i,l3]*gamma_temp_pol[j,l2]*
                                                     delta_temp_pol[k,l1]),
                                                          x = r_values)
                        
                    if np.isnan(bispectrum[p,q,s]): # same for p,q,s triplet
                        bispectrum[p,q,s] = 18.0/5.0*integrate.trapz(np.power(r_values,2.0)*
                                                    (-3.0*beta_temp_pol[p,l3]*beta_temp_pol[q,l2]*
                                                     alpha_temp_pol[s,l1]-
                                                     3.0*beta_temp_pol[s,l1]*beta_temp_pol[p,l3]*
                                                     alpha_temp_pol[q,l2]-
                                                     3.0*beta_temp_pol[q,l2]*beta_temp_pol[s,l1]*
                                                     alpha_temp_pol[p,l3]-
                                                     8.0*delta_temp_pol[s,l1]*delta_temp_pol[q,l2]*
                                                     delta_temp_pol[p,l3]+
                                                     3.0*beta_temp_pol[s,l1]*gamma_temp_pol[q,l2]*
                                                     delta_temp_pol[p,l3]+
                                                     3.0*beta_temp_pol[s,l1]*gamma_temp_pol[p,l3]*
                                                     delta_temp_pol[q,l2]+
                                                     3.0*beta_temp_pol[q,l2]*gamma_temp_pol[s,l1]*
                                                     delta_temp_pol[p,l3]+
                                                     3.0*beta_temp_pol[q,l2]*gamma_temp_pol[p,l3]*
                                                     delta_temp_pol[s,l1]+
                                                     3.0*beta_temp_pol[p,l3]*gamma_temp_pol[s,l1]*
                                                     delta_temp_pol[q,l2]+
                                                     3.0*beta_temp_pol[p,l3]*gamma_temp_pol[q,l2]*
                                                     delta_temp_pol[s,l1]),
                                                          x = r_values)
                        
                    bispectrum2 = bispectrum[i,j,k]*bispectrum[p,q,s]
                    
                    # Compute fisher

                    fisher+=(2.0*l1+1.0)*(2.0*l2+1.0)*(2.0*l3+1.0)/(4.0*np.pi)*wigner3j[l3-l3min_index]*\
                        wigner3j[l3-l3min_index]*(bispectrum2*invCov)/delta_l

    return fisher