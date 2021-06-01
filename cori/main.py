from compute_fisher import compute_fisher_equil_TE
# from compute_fisher import compute_fisher_equil
# from compute_fisher import compute_fisher_ortho

import multiprocessing as mp 
pool = mp.Pool(64) 
results = pool.starmap(compute_fisher_equil_TE, [(l1,2,1001) for l1 in range(2,1001,1)]) 
pool.close() 

import numpy
# results_cum = numpy.cumsum(numpy.array(results))
numpy.savetxt('res_equilTE_1.txt',results)