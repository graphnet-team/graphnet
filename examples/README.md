Contained within this folder are example scripts for running the graphnet framework and how to use them.

Local or defined executions:
1) Open the train_model script and choose the feature/truth, detector, ...
2) run the script: 
$ python train_model.py

script execution via clusters {HEP, NPX, ...}:
1) define cluster usage features like job name, num_cpus, num_gpus, etc. in train_model_cluster.py
2) define the environment in train_model_cluster.sh (ensures the envionment is taken into the cluster)
3) run the script:
$ python train_model_cluster.py [right now]
$ python train_model.py --cluster [when we have CLI?]

Conversion of i3 files to sqlite dataformat
$ python i3_to_sqlite.py

--
$ make_pipeline_database.py

plot feature
$ python plot_feature_distributions.py

--
$ python read_sqlite.py

--
$ test_width_plot.py