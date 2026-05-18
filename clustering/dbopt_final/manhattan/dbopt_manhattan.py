# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:05:00 2026

@author: nicol
"""

import os

os.environ["DBOPT_HDBSCAN_CORE_DIST_N_JOBS"] = "-1"
os.environ["DBOPT_HDBSCAN_ALGORITHM"] = "boruvka_kdtree"
os.environ["DBOPT_HDBSCAN_LEAF_SIZE"] = "80"

import DBOpt, pyreadr, numpy as np

from spyder_kernels.utils.iofuncs import save_dictionary

geo_data = pyreadr.read_r('../test_data/manhattan_geo_data.rds')
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
        parameter_sweep_plot.savefig(f'manhattan_run{run_num}_map.png')
        
        model.fit(py_geo_data)
        
        labels = model.labels_
        num_clusters = len(np.unique(labels))
        DBCV_score = model.DBCV_score_
        
        cluster_plot_modified = model.plot_clusters(show_noise = True, ind_cluster_scores = True)
        cluster_plot_modified.savefig(f'manhattan_run{run_num}_clusters.png')
        
        save_dictionary(globals(), f'manhattan_run{run_num}_data.spydata')

        run_num += 1

# model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
#                     min_cluster_size = [150,200], min_samples = [50,100], cluster_selection_epsilon = [0.00001,0.0001],
#                     cluster_selection_method = 'leaf', alpha = [0.5,1.5], mem_cutoff = 50.0)
# min_cluster_size: 151, min_samples: 83, cluster_selection_epsilon: 8.555809501880035e-05, alpha: 0.8866758312446313
# DBCV  0.138, 67% noise

# min_cluster_size: 104, min_samples: 44, cluster_selection_epsilon: 0.0003188871729653661, alpha: 1.0448380294893056
# model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
#                     min_cluster_size = [100,150], min_samples = [2,50], cluster_selection_epsilon = [0.00001,0.001],
#                     cluster_selection_method = 'leaf', alpha = [0.5,1.5], mem_cutoff = 50.0)
# DBCV ~0.118, 78 clusters, 69% noise

### THE FINALE

# model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
#                     min_cluster_size = [4,200], min_samples = [4,200], cluster_selection_epsilon = [0.00001,0.001],
#                     cluster_selection_method = ['eom','leaf'], mem_cutoff = 50.0,
#                     hdbscan_n_jobs = 3, random_probe_n_jobs = 1)
# min_cluster_size: 29, min_samples: 25, cluster_selection_method: leaf, cluster_selection_epsilon: 0.0002876192633945196
# DBCV ~0.154, 299 clusters, 65% noise

# model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
#                     min_cluster_size = [100,300], min_samples = [4,200], cluster_selection_epsilon = [0.00001,0.001],
#                     cluster_selection_method = ['eom','leaf'], mem_cutoff = 50.0,
#                     hdbscan_n_jobs = 3, random_probe_n_jobs = 1)
# min_cluster_size: 112, min_samples: 91, cluster_selection_method: leaf, cluster_selection_epsilon: 0.0007582032284119196
# DBCV ~0.128, 57 clusters, 64% noise

