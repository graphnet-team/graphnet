'''
Credit to Tom Stuttard for the initial code structure
Notes:
* Print statements cannot be part of the executable script when running on npx
* The script recognizes .
* NPX contains: ~7600 HT CPU cores, ~400 GPUs
'''

#TODO: prompt for python/project directory the first time the script runs
# Start with empty list, prompt user for path, if list is not empty just run script

import os
from graphnet.utilities.cluster_params import ClusterSubmitter

# training example, full variable list available in cluster.py
with ClusterSubmitter(
        job_name="submission_test", # cannot contain space
        flush_factor=1,  # How many commands to bunch into a single job
        num_cpus=1,
        num_gpus=1,
        memory=1*4096, # most clusters have 4gb memory allocated per cpu(n); n*4096
        disk_space=1000,
        submit_dir=os.path.join(os.path.expanduser('~'),'graphnet','results','job'),
        output_dir=os.path.join(os.path.expanduser('~'),'graphnet','results','job'),
        run_locally=False,
	    # if on npx enable cluster_name below to negates the npx or grid prompt when running on cobalt
        #cluster_name="icecube_npx",
		# Type your conda installation path
        start_up_commands=[("source "+os.path.expandvars("/groups/icecube/$USER/anaconda3/etc/profile.d/conda.sh")),
                           ("conda activate graphnet")],
    ) as submitter :
        print("Testing job submitter")

        submitter.add(
            # Type datapath to executable script
            command=[
                ("python "+os.path.expandvars("/groups/icecube/$USER/work/graphnet/examples/train_model.py/train_model.py"))
                ],
			description="submission test",
			allowed_return_status=[5,0] # relates to type of allowed error messages
		) # Define acceptable return status values
