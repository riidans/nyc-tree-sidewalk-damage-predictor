# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 23:59:28 2026

@author: nicol
"""

import os

os.environ["DBOPT_HDBSCAN_CORE_DIST_N_JOBS"] = "-1"
os.environ["DBOPT_HDBSCAN_ALGORITHM"] = "boruvka_kdtree"
os.environ["DBOPT_HDBSCAN_LEAF_SIZE"] = "80"

import DBOpt, pyreadr, numpy as np

from spyder_kernels.utils.iofuncs import save_dictionary

geo_data = pyreadr.read_r('../test_data/staten_geo_data.rds')
df = geo_data[None]
py_geo_data = df.to_numpy()

cluster_sizes = [[2,100],[100,200],[200,300],[300,400],[400,500]]
sample_sizes = [[2,100],[100,200],[200,300],[300,400],[400,500]]

run_num = 1

for c in cluster_sizes:
    for s in sample_sizes:
        model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
                            min_cluster_size = c, min_samples = s, cluster_selection_epsilon = [0.00001,0.005],
                            cluster_selection_method = 'leaf', alpha = 1.0, mem_cutoff = 20.0,
                            hdbscan_n_jobs = 4, random_probe_n_jobs = 1)
        
        model.optimize(py_geo_data)
        
        parameter_sweep_arr = model.parameter_sweep_
        DBOpt_selected_parameters = model.parameters_
        
        parameter_sweep_plot = model.plot_optimization()
        parameter_sweep_plot.savefig(f'staten_run{run_num}_map.png')
        
        model.fit(py_geo_data)
        
        labels = model.labels_
        num_clusters = len(np.unique(labels))
        DBCV_score = model.DBCV_score_
        
        cluster_plot_modified = model.plot_clusters(show_noise = True, ind_cluster_scores = True)
        cluster_plot_modified.savefig(f'staten_run{run_num}_clusters.png')
        
        save_dictionary(globals(), f'staten_run{run_num}_data.spydata')

        run_num += 1

# Notes for 4/30/26: cluster_selection_epsilon shouln't be allowed to be 0, 'leaf' was overwhelmingly the better method, 
# typically favored higher min_cluster_size and lower min_samples

# Notes for 5/1/26: changes from above resulted in slightly lower DBCV score but less noise, which is the main tradeoff
# testing with alpha in [0,2] with large bounds for the others next

# separate alpha to two runs in [0,1] and [1,2]
# maybe do separate runs for 'eom' and 'leaf' methods to definitively rule one out

# min_cluster_size: 63, min_samples: 56, cluster_selection_method: leaf, cluster_selection_epsilon: 1e-05, alpha: 0.7603575426477817
# model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
#                     min_cluster_size = [4,100], min_samples = [4,100], cluster_selection_epsilon = [0.00001,0.005],
#                     cluster_selection_method = ['eom','leaf'], alpha = [0,1], mem_cutoff = 50.0)
# Pretty good results! DCBV ~0.157

# Notes for 5/2/26: look into getting soft clustering using the hdbscan* library
# look into integrating that library into DBOpt for better performance, but this is less of a priority

# min_cluster_size: 17, min_samples: 9, cluster_selection_method: leaf, cluster_selection_epsilon: 1e-05, alpha: 0.7413187939637236
# model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
#                     min_cluster_size = [4,100], min_samples = [4,100], cluster_selection_epsilon = [0.00001,0.005],
#                     cluster_selection_method = ['eom'], alpha = [0,1], mem_cutoff = 50.0)
# Weird results: DBCV ~0.221 (!) but wayyyy too many clusters

# min_cluster_size: 161, min_samples: 108, cluster_selection_method: leaf, cluster_selection_epsilon: 1e-05, alpha: 0.8404760656952529
# model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
#                     min_cluster_size = [50,175], min_samples = [50,175], cluster_selection_epsilon = [0.00001,0.005],
#                     cluster_selection_method = ['eom','leaf'], alpha = [0,1], mem_cutoff = 50.0)
# Solid! DCBV ~0.186 with decent cluster sizes.

# Notes for week of 5/3/26: need to figure out how to report noise percentages!
# probably a good idea to solely use the 'leaf' method from now on

# # min_cluster_size: 129, min_samples: 83, cluster_selection_epsilon: 0.0001, alpha: 1.0
# model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
#                     min_cluster_size = [50,175], min_samples = [50,175], cluster_selection_epsilon = [0.00001,0.0001],
#                     cluster_selection_method = 'leaf', alpha = [0.5,1], mem_cutoff = 50.0)
# DBCV ~0.19, won't plot for some reason
# will test alpha in [0.5,1.5] tomorrow as bounds seemed too tight (based on iterations only without the plotting)

# min_cluster_size: 103, min_samples: 96, cluster_selection_epsilon: 1e-05, alpha: 1.5
# model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
#                     min_cluster_size = [50,175], min_samples = [50,175], cluster_selection_epsilon = [0.00001,0.0001],
#                     cluster_selection_method = 'leaf', alpha = [0.5,1.5], mem_cutoff = 50.0)
# DBCV ~0.18, need to figure out the plotting!

# min_cluster_size: 169, min_samples: 92, cluster_selection_epsilon: 8.252350855852641e-05, alpha: 0.9016820666974613
# model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
#                     min_cluster_size = [150,200], min_samples = [50,100], cluster_selection_epsilon = [0.00001,0.0001],
#                     cluster_selection_method = 'leaf', alpha = [0.5,1.5], mem_cutoff = 50.0)
# DBCV ~ 0.175, ~63% noise
