# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:09:02 2026

@author: nicol
"""

import os

os.environ["DBOPT_HDBSCAN_CORE_DIST_N_JOBS"] = "-1"
os.environ["DBOPT_HDBSCAN_ALGORITHM"] = "boruvka_kdtree"
os.environ["DBOPT_HDBSCAN_LEAF_SIZE"] = "80"

import DBOpt, pyreadr, numpy as np

from spyder_kernels.utils.iofuncs import save_dictionary

geo_data = pyreadr.read_r('../test_data/bklyn_queens_geo_data.rds')
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
                            hdbscan_n_jobs = -1, random_probe_n_jobs = 1)
        
        model.optimize(py_geo_data)
        
        parameter_sweep_arr = model.parameter_sweep_
        DBOpt_selected_parameters = model.parameters_
        
        parameter_sweep_plot = model.plot_optimization()
        parameter_sweep_plot.savefig(f'bklyn_queens_run{run_num}_map.png')
        
        model.fit(py_geo_data)
        
        labels = model.labels_
        num_clusters = len(np.unique(labels))
        DBCV_score = model.DBCV_score_
        
        cluster_plot_modified = model.plot_clusters(show_noise = True, ind_cluster_scores = True)
        cluster_plot_modified.savefig(f'bklyn_queens_run{run_num}_clusters.png')
        
        save_dictionary(globals(), f'bklyn_queens_run{run_num}_data.spydata')

        run_num += 1

### THE FINALE

# model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
#                     min_cluster_size = [4,200], min_samples = [4,200], cluster_selection_epsilon = [0.00001,0.001],
#                     cluster_selection_method = ['eom','leaf'], mem_cutoff = 50.0,
#                     hdbscan_n_jobs = 4, random_probe_n_jobs = 1)
# min_cluster_size: 38, min_samples: 37, cluster_selection_method: leaf, cluster_selection_epsilon: 0.00038305576928485813
#  DBCV ~0.127, 1207 clusters, 69% noise

# model = DBOpt.DBOpt(algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
#                     min_cluster_size = [100,300], min_samples = [4,200], cluster_selection_epsilon = [0.00001,0.001],
#                     cluster_selection_method = ['eom','leaf'], mem_cutoff = 50.0,
#                     hdbscan_n_jobs = 4, random_probe_n_jobs = 1)
# min_cluster_size: 198, min_samples: 174, cluster_selection_method: leaf, cluster_selection_epsilon: 0.0002581228572756402
# DBCV ~0.107, 196 clusters, 67% noise