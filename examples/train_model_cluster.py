"""
The script recognizes .
Tom Stuttard
"""

import os
from graphnet.utils.cluster_params import *

# training example, full variable list available in cluster.py
with ClusterSubmitter(
    job_name="test_job",
    flush_factor=1,  # How many commands to bunch into a single job [batch array later?]
    num_cpus=2,
    # num_gpus=1,
    memory=2
    * 4096,  # most clusters have 4gb memory allocated per cpu; add num_cpus as variable?
    disk_space=1000,
    submit_dir=os.path.join(
        os.path.expanduser("~"), "graphnet", "results", "job"
    ),  # +"./tmp",
    output_dir=os.path.join(
        os.path.expanduser("~"), "graphnet", "results", "job"
    ),
    run_locally=False,
    # cluster_name="icecube_npx" # negates the npx or grid prompt when running on cobalt
) as submitter:

    print("Testing submitter")

    submitter.add(
        command="sh /groups/icecube/qgf305/graphnet/examples/train_model_cluster.sh",
        description="A description",
        allowed_return_status=[5, 0],
    )  # Define acceptable return status values
