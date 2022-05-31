"""
Tools for running on clusters

Tom Stuttard
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from builtins import open
from builtins import str
from builtins import object
from future import standard_library

standard_library.install_aliases()

import os, socket, subprocess, datetime, shutil

from graphnet.utils.cluster.filesys_tools import TMP_FILE_STRFTIME

from graphnet.utils.cluster.unix_tools import BASH_SHEBANG

from graphnet.utils.cluster.condor import *
from graphnet.utils.cluster.slurm import *
from graphnet.utils.cluster.gridengine import *
from graphnet.utils.cluster.job import *

# py2 vs 3 compatibility
try:
    input = raw_input
except NameError:
    pass

#
# Globals
#

SUBMIT_WRAPPER_SCRIPT = "submit.sh"

#
# User classes for job submission
#


class ClusterSubmitter(object):

    """
    A class for handling cluster submission
    This handles:
      - Checking cluster system on node and providing relevent handling
      - Checking this is a submit node
      - Buffering commands into cluster jobs
      - Creating python scripts to run as jobs that contain the buffered commands
      - Generating submit scripts in whatever the native cluster system is to run all the jobs

    #TODO Describe the connection between this class and the stuff in job.py

    #TODO Add function to query number of running jobs

    Need to respect guidelines on the cluster for where to write data, logs, submission files, etc
    For condor see:
      https://wiki.icecube.wisc.edu/index.php/Condor/BestPractices#DAGMan
      https://wiki.icecube.wisc.edu/index.php/Condor/DAGMan#Using_local_scratch
    """

    def __init__(
        self,
        # Steer the submission
        flush_factor,  # How many commands to bunch into a single job
        job_name,  # A string name for the overall submissions
        run_locally=False,  # Can optionally run locally instead of submitting (many other args can be ignored in this case)
        memory=None,  # Required memory [MB]
        disk_space=None,  # Required disk space [MB]
        wall_time=None,  # Wall time (e.g. when a job will be terminated) [hrs]
        submit_dir=None,  # Where the submission files will be created and run from
        output_dir=None,  # Where the ouput/error files and the job steering file will go
        # cluster_name=None, # Optionally provide cluster name, otherwise will atempt to figure it out
        max_concurrent_jobs=1000,  # Maximum number of concurrent jobs to allow on the cluster
        job_mode="wrapper",  # Various options for which mode to use for generating the job, see "job.py" for details
        dry_run=False,  # Dry run (will not actually submit)
        cluster_name=None,  # Optionally specify the cluster name manually (otherwise it is autmatically determined, which is recommended)
        # Environment setup
        job_env_vars=None,  # Dict of env vars to set on the nodes
        export_env=False,  # Export submission session env. Not recommended to use this, instead pass sectup script to `start_up_commands`
        env_shell=None,  # Specify an IceCube env-shell.sh script to use
        start_up_commands=None,  # Commands to be run at start of job
        tear_down_commands=None,  # Commands to be run at start of job
        # Node requirements
        require_cvmfs=False,  # Nodes must have CVMFS
        require_avx=False,  # Nodes must have AVX
        require_cuda=False,  # Nodes must have CUDA
        require_sl7=False,  # Nodes must have Scientific Linux 7
        choose_sites=None,  # List of sites to choose
        exclude_sites=None,  # List of sites to exclude
        num_cpus=1,  # Request CPUs
        num_gpus=0,  # Request GPUs
        # Special args for SLURM
        partition=None,
        mail_type=None,
        mail_user=None,
        # Special args for Condor
        accounting_group=None,
    ):

        #
        # Store args
        #

        self.flush_factor = flush_factor
        self.job_name = job_name
        self.run_locally = run_locally
        self.memory = memory
        self.disk_space = disk_space
        self.wall_time = wall_time
        self.submit_dir = submit_dir
        self.output_dir = output_dir
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_mode = job_mode
        self.dry_run = dry_run
        self.job_env_vars = job_env_vars
        self.export_env = export_env
        self.env_shell = env_shell
        self.start_up_commands = start_up_commands
        self.tear_down_commands = tear_down_commands
        self.require_cvmfs = require_cvmfs
        self.require_avx = require_avx
        self.require_cuda = require_cuda
        self.require_sl7 = require_sl7
        self.choose_sites = choose_sites
        self.exclude_sites = exclude_sites
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.partition = partition
        self.mail_type = mail_type
        self.mail_user = mail_user
        self.accounting_group = accounting_group

        #
        # Init jobs buffer jobs
        #

        # Handle the buffering of commands into jobs
        self.job_counter = 0
        self.command_counter = 0
        self.commands = []
        self.jobs = []

        #
        # Init submission
        #

        if self.run_locally:

            # Nothing much to prepare when running locally
            print("Submitted jobs will run locally")

        else:

            # Will submit to a cluster, initialise submission
            self._init_cluster_submission(cluster_name)

    def _init_cluster_submission(self, cluster_name=None):

        #
        # Check inputs
        #

        assert (
            self.job_name is not None
        ), "Must specify 'job_name' argument when submitting to cluster"
        # assert isinstance(self.job_name, str) and self.job_name != "", "'job_name' argument must be a valid string"

        assert (
            self.memory is not None
        ), "Must specify 'memory; argument (in [MB]) when submitting to cluster"

        if self.disk_space is None:
            self.disk_space = 1000  # [MB]

        #
        # Figure out what cluster we're on
        #

        self._init_cluster_site(cluster_name)

        #
        # Init directories
        #

        # Check user provided a submission dir
        assert self.submit_dir is not None, "Must provide 'submit_dir"

        # Also check output dir
        # If not specified, default to use the same as submit_dir
        if self.output_dir is None:
            self.output_dir = self.submit_dir

        # Create temporary directories for this particular submission
        tmp_dir_name = os.path.join(
            self.job_name, datetime.datetime.now().strftime(TMP_FILE_STRFTIME)
        )
        self.submit_dir = os.path.abspath(
            os.path.join(self.submit_dir, tmp_dir_name)
        )
        self.output_dir = os.path.abspath(
            os.path.join(self.output_dir, tmp_dir_name)
        )
        for d in set([self.submit_dir, self.output_dir]):
            make_dir(d)

    def _init_cluster_site(self, cluster_name=None):
        """
        This contains all the messy business of handling differences between
        individual clusters used by the collaboration.
        """

        self.cluster = collections.OrderedDict()

        #
        # Determine cluster site
        #

        # Define IceCube clusters (need this later)
        icecube_clusters = {
            "icecube_npx": "submit-1.icecube.wisc.edu",
            "icecube_grid": "sub-1.icecube.wisc.edu",
        }

        # Check where we are running
        self.hostname = socket.getfqdn()  # gethostname

        # Init cluster definitions
        self.cluster["name"] = None  # Identifier of the cluster
        self.cluster[
            "submitter"
        ] = None  # Host for making submissions from (potentially via ssh), only used in some cases

        # If user specified cluster name directly, use this
        if cluster_name is not None:
            self.cluster["name"] = cluster_name

        # Otherwise figure it out for ourselves
        else:

            # NBI
            if self.hostname.startswith("hep") and self.hostname.endswith(
                ".cluster"
            ):
                self.cluster["name"] = "nbi"

            # DESY
            elif self.hostname.endswith("zeuthen.desy.de"):
                self.cluster["name"] = "desy"

            # IceCube (special case as there are two possible clusters)
            elif self.hostname.endswith("icecube.wisc.edu"):

                # IceCube is a special case as there are two possible clusters
                if self.hostname in icecube_clusters.values():
                    for name, sub in icecube_clusters.items():
                        if self.hostname == sub:
                            self.cluster["name"] = name
                else:
                    user_choice = input(
                        "Choose cluster from %s : "
                        % ([str(n) for n in icecube_clusters.keys()])
                    )
                    assert user_choice in icecube_clusters
                    self.cluster["name"] = user_choice

        assert self.cluster["name"] is not None, (
            "Could not determine cluster site : hostname = %s" % self.hostname
        )

        #
        # Init site
        #

        # Set some default values before starting
        self.remote_submission = False
        self.user_proxy = False

        if self.cluster["name"].startswith("icecube"):

            #
            # Init IceCube cluster(s)
            #

            # Note that the IceCube clusters require more complex handling than most
            # There are strict requirements of where things can go, drives missing from nodes, etc
            # Some useful info:
            #  https://wiki.icecube.wisc.edu/index.php/Condor
            # https://wiki.icecube.wisc.edu/index.php/Condor/Grid
            #  https://events.icecube.wisc.edu/event/99/contributions/5949/attachments/5022/5533/20180620_Grid_Tutorial_Madison_Bootcamp.pdf

            self.cluster["system"] = "condor"
            self.cluster["node_scratch"] = "$_CONDOR_SCRATCH_DIR"

            self.cluster["submitter"] = icecube_clusters[self.cluster["name"]]

            # Submit dir MUST be on scratch for IceCube clusters
            if self.submit_dir is not None:
                print(
                    "WARNING : Enforcing use of scratch dir for 'submit_dir'"
                )
            self.submit_dir = os.path.expandvars("/scratch/$USER")

            # Output dir also MUST be on scratch for the case of the IceCube grid
            if self.cluster["name"] == "icecube_grid":
                if self.output_dir is not None:
                    print(
                        "WARNING : Enforcing use of scratch dir for 'output_dir'"
                    )
                self.output_dir = self.submit_dir

            # If not on the submitter node, submit remotely
            if self.hostname != self.cluster["submitter"]:
                self.remote_submission = True

            # Require a user proxy for the grid
            if self.cluster["name"] == "icecube_grid":
                self.user_proxy = True

            # Enforce that args not suitable for NPX are not used in that case
            if self.cluster["name"] == "icecube_npx":
                assert (
                    not self.require_cvmfs
                ), "`require_cvmfs` arg not supported by NPX"
                # TODO what else?

        elif self.cluster["name"] == "nbi":

            #
            # Init NBI cluster
            #

            self.cluster["system"] = "slurm"
            self.cluster["node_scratch"] = "$SCRATCH"

            if self.partition is None:
                self.cluster["partition"] = (
                    "icecube" if self.num_gpus == 0 else "icecube_gpu"
                )  # Need a dedicated partition if want GPUs
            else:
                self.cluster["partition"] = self.partition

        elif self.cluster["name"] == "desy":

            #
            # Init DESY cluster
            #

            self.cluster["system"] = "gridengine"
            self.cluster["node_scratch"] = "$TMPDIR"
            if self.partition is None:
                self.cluster["partition"] = (
                    "icecube" if self.num_gpus == 0 else "icecube_gpu"
                )  # Need a dedicated partition if want GPUs
            else:
                self.cluster["partition"] = self.partition

        else:
            raise Exception("Unknown cluster name : %s" % self.cluster["name"])

        #
        # Done
        #

        print("Cluster details :")
        for k, v in self.cluster.items():
            print("%s : %s" % (k, v))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.submit()
        self.report()

    def add(
        self,
        command,
        description=None,
        allowed_return_status=DEFAULT_ALLOWED_RETURN_STATUS,
    ):

        # Add command to buffer
        self.commands.append(
            ClusterCommand(command, description, allowed_return_status)
        )
        self.command_counter += 1

        # Flush commands into a job if required
        if len(self.commands) == self.flush_factor:
            self.flush()

    def flush(self):

        # Check if there is anything buffered to flush
        if len(self.commands) > 0:

            # Create the job from the buffered commands
            job_index = len(self.jobs) + 1  # job indeces starting with one
            job = ClusterJob(
                job_index,
                self.commands,
                description="",
                env_vars=self.job_env_vars,
            )

            # Add the jobs buffer
            self.jobs.append(
                job
            )  # Store so can generate submit scripts for these jobs later
            self.commands = []

    def submit(self):
        """
        Generate files required for submission
        Then actually submit to the cluster (or run locally depending on options used)
        """

        # Flush any remaining commands
        self.flush()

        # Check if running here locally or submitting to a cluster
        if self.run_locally:
            self._run_jobs_locally()
        else:
            self._submit_to_cluster()

    def _submit_to_cluster(self):
        """
        Submit the buffered jobs to run on the cluster
        """

        # Check that there are any jobs
        num_jobs = len(self.jobs)
        if num_jobs > 0:

            #
            # Initialise the jobs
            #

            # Loop over jobs
            for job in self.jobs:

                # Prepare the job submission
                job.prepare_to_submit(
                    job_dir=self.output_dir,
                    start_up_commands=self.start_up_commands,
                    tear_down_commands=self.tear_down_commands,
                    env_shell=self.env_shell,
                    job_mode=self.job_mode,
                )

            #
            # Handle different cluster systems...
            #

            # Note that some like to handle full commands, whilst some want an executable and a list of arguments
            # Also, some want a list of args for each new command, whereas other just give you a job index

            if self.cluster["system"] == "condor":

                #
                # condor
                #

                # Note that some of this condor stuff isn't that general, e.g. assumes we are using IceCube clsters (NPX, grid, etc)
                # However, there is no non_iceCube cluster running condor that I know of this is is reasonably safe if not toally ideal

                # Create a DAGMan submit file (this will also create the job condor submit file)
                dagman_file_name = self.job_name + ".submit"
                condor_file_name = self.job_name + ".condor"
                dagman_file_path, condor_file_path = create_dagman_submit_file(
                    submit_dir=self.submit_dir,  # Both submit and log files must go to scratch for NPX
                    log_dir=self.submit_dir,
                    dagman_file_name=dagman_file_name,
                    condor_file_name=condor_file_name,
                    jobs=self.jobs,
                    memory_MB=self.memory,
                    disk_space_MB=self.disk_space,
                    num_cpus=self.num_cpus,
                    num_gpus=self.num_gpus,
                    wall_time_hr=self.wall_time,
                    export_env=self.export_env,
                    require_cvmfs=self.require_cvmfs,
                    require_avx=self.require_avx,
                    require_cuda=self.require_cuda,
                    require_sl7=self.require_sl7,
                    choose_sites=self.choose_sites,
                    exclude_sites=self.exclude_sites,
                    user_proxy=self.user_proxy,
                    accounting_group=self.accounting_group,
                )

                # Write a script to do the submission
                # This handles setting up e.g. job arrays
                submit_script_path = os.path.abspath(
                    os.path.join(self.submit_dir, SUBMIT_WRAPPER_SCRIPT)
                )
                with open(submit_script_path, "w") as submit_script:
                    submit_script.write(BASH_SHEBANG + "\n")
                    submit_script.write("\n")
                    submit_script.write(
                        "%s -maxjobs %i %s\n"
                        % (
                            CONDOR_SUBMIT_EXE,
                            self.max_concurrent_jobs,
                            dagman_file_path,
                        )
                    )

                # Done
                print(
                    (
                        "DAGMan/Condor submit script written (%i jobs) : %s"
                        % (num_jobs, submit_script_path)
                    )
                )

            elif self.cluster["system"] == "slurm":
                #
                # SLURM
                #

                # Check for incompatible args
                if self.require_cvmfs:
                    print(
                        "WARNING : `require_cvmfs` arg not currently supported for SLURM submissions, ignoring"
                    )
                if self.require_cuda:
                    print(
                        "WARNING : `require_cuda` arg not currently supported for SLURM submissions, ignoring"
                    )
                if self.require_sl7:
                    print(
                        "WARNING : `require_sl7` arg not currently supported for SLURM submissions, ignoring"
                    )
                assert (
                    self.job_mode != "no_wrapper"
                ), "`job_mode='no_wrapper'` not currently supported for SLURM submissions"  # TODO Fix this, issue is job array indexing

                # SLURM array submission uses an array indes to distriguish between each job, rather than passing it many jobs as is the case for condor
                # Create compatible versions of the various job parameters
                (
                    job_dir,
                    job_out_file,
                    job_err_file,
                    job_wrapper_script,
                ) = self.jobs[0].get_slurm_paths()

                # The command to run is the wrapper script
                exe_command = job_wrapper_script

                # Create a SLURM submit file
                slurm_file_path = create_slurm_submit_file(
                    job_dir=self.submit_dir,
                    job_name=self.job_name,
                    partition=(
                        self.cluster["partition"]
                        if "partition" in self.cluster
                        else None
                    ),
                    mail_type=self.mail_type,
                    mail_user=self.mail_user,
                    exe_commands=[exe_command],
                    working_dir=job_dir,
                    out_file=job_out_file,
                    err_file=job_err_file,
                    memory=self.memory,
                    num_gpus=self.num_gpus,
                    wall_time_hours=self.wall_time,
                    use_array=True,
                    export_env=self.export_env,
                )

                # Write a script to do the submission
                # This handles setting up e.g. job arrays
                submit_script_path = os.path.abspath(
                    os.path.join(self.submit_dir, SUBMIT_WRAPPER_SCRIPT)
                )
                with open(submit_script_path, "w") as submit_script:
                    submit_script.write(BASH_SHEBANG + "\n")
                    submit_script.write("\n")
                    submit_script.write(
                        "%s --array=1-%i %s\n"
                        % (SLURM_SUBMIT_EXE, num_jobs, slurm_file_path)
                    )  # TODO ensure array indices align with job indices

                # Done
                print(
                    "SLURM submit script written (%i jobs) : %s"
                    % (num_jobs, submit_script_path)
                )

            elif self.cluster["system"] == "gridengine":
                #
                # GRIDENGINE
                #

                # Check for incompatible args
                if self.require_cvmfs:
                    print(
                        "WARNING : `require_cvmfs` arg not currently supported for GRIDENGINE submissions, ignoring"
                    )
                if self.require_cuda:
                    print(
                        "WARNING : `require_cuda` arg not currently supported for GRIDENGINE submissions, ignoring"
                    )
                if self.require_sl7:
                    print(
                        "WARNING : `require_sl7` arg not currently supported for GRIDENGINE submissions, ignoring"
                    )

                # GRIDENGINE uses array jobs, where the out and error files follow a naming
                # scheme with an array job index. We need to change the file names accordingly.
                for job in self.jobs:
                    job_submit_dir = os.path.join(
                        self.submit_dir, "job_%d" % job.index
                    )
                    job_output_dir = os.path.join(
                        self.output_dir, "job_%d" % job.index
                    )
                    job.job_dir = job_submit_dir
                    job.out_file = os.path.join(
                        job_output_dir, "job_%d.out" % job.index
                    )
                    job.err_file = os.path.join(
                        job_output_dir, "job_%d.err" % job.index
                    )
                    job.wrapper_script = os.path.join(
                        job_submit_dir, "job_%d.sh" % job.index
                    )

                task_id_string = "`printf %s ${SGE_TASK_ID}`" % JOB_INDEX_FMT
                generic_output_dir = os.path.join(
                    self.output_dir, "job_" + task_id_string
                )
                generic_wrapper_script = os.path.join(
                    self.output_dir,
                    "job_" + task_id_string,
                    "job_" + task_id_string + ".sh",
                )
                # The command to run is the wrapper script
                exe_command = generic_wrapper_script

                # Create a GRIDENGINE submit file
                gridengine_file_path = create_gridengine_submit_file(
                    job_dir=self.submit_dir,
                    job_name=self.job_name,
                    exe_commands=[exe_command],
                    working_dir=generic_output_dir,
                    wall_time_hours=self.wall_time,
                    out_dir=self.output_dir,
                    partition=(
                        self.cluster["partition"]
                        if "partition" in self.cluster
                        else None
                    ),
                    memory=self.memory,
                    num_cpus=self.num_cpus,
                    use_gpu=self.num_gpus > 0,
                    use_array=True,
                    export_env=self.export_env,
                )

                # Write a script to do the submission
                # This handles setting up e.g. job arrays
                submit_script_path = os.path.abspath(
                    os.path.join(self.submit_dir, SUBMIT_WRAPPER_SCRIPT)
                )
                with open(submit_script_path, "w") as submit_script:
                    submit_script.write(BASH_SHEBANG + "\n")
                    submit_script.write("\n")
                    submit_script.write(
                        "%s -t 1-%i %s\n"
                        % (
                            GRIDENGINE_SUBMIT_EXE,
                            num_jobs,
                            gridengine_file_path,
                        )
                    )  # TODO ensure array indices align with job indices

                # Done
                print(
                    (
                        "GRIDENGINE submit script written (%i jobs) : %s"
                        % (num_jobs, submit_script_path)
                    )
                )

            # Complain if can't find cluster system
            else:
                raise Exception(
                    "Cannot submit : Unrecognised cluster system '%s'"
                    % self.cluster["name"]
                )

            #
            # Do the submission
            #

            # Define the command to run
            submit_command = "sh %s" % (submit_script_path)

            if self.dry_run:

                # If dry run, don't actually submit but let the user know
                print("Not submitting, dry run flag enabled")
                print(
                    "Can submit it manually as follows : %s" % submit_command
                )

            else:

                # Check if running on a submitter node
                if self.remote_submission:
                    # Run the generated submission script remotely...

                    # First copy submit dir
                    # This assumes that `submit_dir` exists on both this machine and the remote submit node
                    # Will need to make this smarter if this is not the case in the future
                    print(
                        "Transferring submission dir to the remote submitter (%s)"
                        % self.cluster["submitter"]
                    )
                    remote_rsync_command = "rsync -e ssh -r %s %s:%s" % (
                        self.submit_dir,
                        self.cluster["submitter"],
                        os.path.dirname(self.submit_dir),
                    )  # TODO Handle user name
                    status = subprocess.call(remote_rsync_command, shell=True)
                    assert status == 0, (
                        "Transferring submission dir failed : Error code = %i"
                        % status
                    )

                    # Run the submit script generated above, on the remote submitter
                    print(
                        "Making the remote submission (%s)"
                        % self.cluster["submitter"]
                    )
                    remote_submit_command = "ssh %s 'cd %s && %s'" % (
                        self.cluster["submitter"],
                        self.submit_dir,
                        submit_command,
                    )  # TODO Handle user name
                    print(remote_submit_command)
                    status = subprocess.call(remote_submit_command, shell=True)
                    assert status == 0, (
                        "Remote submission failed : Error code = %i" % status
                    )

                else:
                    # Run the generated submission script here
                    print("Making the submission")
                    status = subprocess.call(submit_command, shell=True)
                    if status != 0:
                        raise Exception(
                            "Submission failed : Error code = %i" % status
                        )

    def _run_jobs_locally(self):
        """
        Run the buffered jobs locally
        """
        # Check that there are any jobs
        num_jobs = len(self.jobs)
        if num_jobs > 0:
            #
            # Loop over commands
            #

            # Loop over buffered jobs
            for job in self.jobs:

                job.start_time = datetime.datetime.now()

                # Set env vars
                if job.env_vars is not None:
                    for k, v in job.env_vars.items():
                        os.environ[k] = v

                # Loop over commands
                for command in job.commands:
                    #
                    # Run the command
                    #

                    print(("\nRunning command : %s" % command.command))

                    command.start_time = datetime.datetime.now()

                    # Run the command
                    try:
                        command.status = subprocess.call(
                            command.command, shell=True
                        )
                    except Exception as e:
                        msg = "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                        msg += "!!! -- ERROR : \n"
                        msg += (
                            "!!! ---- Exception caught running command : %s\n"
                            % str(e)
                        )
                        msg += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        raise Exception(msg)

                    # Check status
                    if command.allowed_return_status is not None:
                        if command.status not in command.allowed_return_status:
                            msg = "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                            msg += "!!! -- ERROR : \n"
                            msg += (
                                "!!! ---- Bad return status : %i\n"
                                % command.status
                            )
                            msg += (
                                "!!! ---- Allow return status values : %s\n"
                                % command.allowed_return_status
                            )
                            msg += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                            raise Exception(msg)

                    command.end_time = datetime.datetime.now()

                    command_time_taken = command.end_time - command.start_time
                    print(("Command took : %s" % command_time_taken))

                job.end_time = datetime.datetime.now()

    def report(self):
        """
        Produce a nice report on what was run
        """

        print("\n>>>>>>>>>> Cluster submitter report >>>>>>>>>>")

        if self.run_locally:
            print("Ran jobs locally")
        else:
            print(
                (
                    "Submitted to '%s' cluster site '%s' via '%s'"
                    % (
                        self.cluster["system"],
                        self.cluster["name"],
                        self.hostname,
                    )
                )
            )

        # General reporting
        print("%i jobs in submission" % len(self.jobs))
        print("<= %i commands in each job" % self.flush_factor)
        print("%i commands in total in submission" % (self.command_counter))
        if len(self.commands) > 0:
            print(
                "WARNING: %i commands still to be flushed"
                % (len(self.commands))
            )

        # Reporting specific to running locally
        if self.run_locally:
            start_times = [
                command.start_time
                for job in self.jobs
                for command in job.commands
                if command.start_time is not None
            ]
            end_times = [
                command.end_time
                for job in self.jobs
                for command in job.commands
                if command.end_time is not None
            ]
            num_commands = len(
                [1 for job in self.jobs for command in job.commands]
            )
            num_completed_jobs = len(end_times)
            if num_completed_jobs < num_commands:
                print(
                    "WARNING : Only %i of %i jobs completed successfully"
                    % (num_completed_jobs, num_completed_jobs)
                )
            if len(end_times) > 0:
                print(
                    "Total command processing time : %s"
                    % (max(end_times) - min(start_times))
                )

        # Reporting specific to submitting to a cluster
        else:
            print("Submit files in : %s" % (self.submit_dir))
            print("Output files in : %s" % (self.output_dir))

        print("<<<<<<<<<< Cluster submitter report <<<<<<<<<<\n")

    def clear_footprint(self):
        """
        Remove the directories/files created during this submission
        """

        # Remove the tmp dirs
        if os.path.exists(self.submit_dir):
            shutil.rmtree(self.submit_dir)

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
