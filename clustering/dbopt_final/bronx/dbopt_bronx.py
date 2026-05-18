# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:09:15 2026

@author: nicol
"""

import os

os.environ["DBOPT_HDBSCAN_CORE_DIST_N_JOBS"] = "-1"
os.environ["DBOPT_HDBSCAN_ALGORITHM"] = "boruvka_kdtree"
os.environ["DBOPT_HDBSCAN_LEAF_SIZE"] = "80"

import DBOpt, pyreadr, numpy as np

from spyder_kernels.utils.iofuncs import save_dictionary

geo_data = pyreadr.read_r('../test_data/bronx_geo_data.rds')
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
                            hdbscan_n_jobs = 3, random_probe_n_jobs = 1)
        
        model.optimize(py_geo_data)
        
        parameter_sweep_arr = model.parameter_sweep_
        DBOpt_selected_parameters = model.parameters_
        
        parameter_sweep_plot = model.plot_optimization()
        parameter_sweep_plot.savefig(f'bronx_run{run_num}_map.png')
        
        model.fit(py_geo_data)
        
        labels = model.labels_
        num_clusters = len(np.unique(labels))
        DBCV_score = model.DBCV_score_
        
        cluster_plot_modified = model.plot_clusters(show_noise = True, ind_cluster_scores = True)
        cluster_plot_modified.savefig(f'bronx_run{run_num}_clusters.png')
        
        save_dictionary(globals(), f'bronx_run{run_num}_data.spydata')

        run_num += 1

# model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
#                     min_cluster_size = [150,200], min_samples = [50,100], cluster_selection_epsilon = [0.00001,0.0001],
#                     cluster_selection_method = 'leaf', alpha = [0.5,1.5], mem_cutoff = 50.0)
# min_cluster_size: 152, min_samples: 99, cluster_selection_epsilon: 9.290313555608283e-05, alpha: 1.1029553278330853
# DBCV ~0.163, 61% noise

### THE FINALE

# model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
#                     min_cluster_size = [4,200], min_samples = [4,200], cluster_selection_epsilon = [0.00001,0.001],
#                     cluster_selection_method = ['eom','leaf'], mem_cutoff = 50.0,
#                     hdbscan_n_jobs = 3, random_probe_n_jobs = 1)
# min_cluster_size: 23, min_samples: 16, cluster_selection_method: leaf, cluster_selection_epsilon: 0.00015280966847527346
# DBCV ~ 0.171, 619 clusters, 57% noise

# model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
#                     min_cluster_size = [100,300], min_samples = [4,200], cluster_selection_epsilon = [0.00001,0.001],
#                     cluster_selection_method = ['eom','leaf'], mem_cutoff = 50.0,
#                     hdbscan_n_jobs = 3, random_probe_n_jobs = 1)
# min_cluster_size: 249, min_samples: 196, cluster_selection_method: leaf, cluster_selection_epsilon: 0.0003096550231721446
# DBCV ~0.172, 35 clusters, 58% noise