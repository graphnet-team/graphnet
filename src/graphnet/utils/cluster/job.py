"""
Script to define and run a cluster job, using JSON steering files to define what to run, 
store metrics, allow job to be partily or totally re-run, etc.

Note that this script should not depend on anything else in the fridge repo, e.g.
it should be able to run on clusters where the fridge is not installed. 

Tom Stuttard
"""

import os, socket, subprocess, datetime, json, collections, sys, numbers, codecs, stat

#
# Globals
#

JOB_SCRIPT = os.path.abspath(__file__).replace(".pyc", ".py")

JOB_INDEX_FMT = "%i"
JOB_STEM = "job"
JOB_NAME_FMT = JOB_STEM + "_" + JOB_INDEX_FMT
JOB_DIR_FMT = JOB_NAME_FMT
JOB_STEERING_FILE_FMT = JOB_NAME_FMT + ".json"

DEFAULT_ALLOWED_RETURN_STATUS = 0

STATUS_COLORS = collections.OrderedDict(
    [
        ("not_started", "grey"),
        ("started", "blue"),
        ("failure", "red"),
        ("success", "green"),
    ]
)

#
# Cluster job classes
#


class ClusterCommand(object):
    """
    Class defining a command to run
    """

    def __init__(
        self,
        command,
        description=None,
        allowed_return_status=DEFAULT_ALLOWED_RETURN_STATUS,
    ):

        # Store args
        # assert isinstance(command, str), "`command` must be a string"
        self.command = command

        # assert isinstance(description, str) or description is None, "`description` must be a string or None, found %s" % type(description)
        self.description = description

        assert (
            isinstance(allowed_return_status, numbers.Number)
            or isinstance(allowed_return_status, collections.Sequence)
            or allowed_return_status is None
        ), "`allowed_return_status` must be a number, list of numbers, or None"
        if isinstance(allowed_return_status, numbers.Number):
            allowed_return_status = [allowed_return_status]
        if allowed_return_status is not None:
            assert all(
                isinstance(s, int) for s in allowed_return_status
            ), "`allowed_return_status` values must be integers"
        self.allowed_return_status = allowed_return_status

        # Init book keeping variables
        self.start_time = None
        self.end_time = None
        self.status = "not_started"  # THe status as recorded by this code
        self.return_status = None  # The status returned by the command

    def to_dict(self):
        return collections.OrderedDict(
            command=self.command,
            description=self.description,
            allowed_return_status=self.allowed_return_status,
            start_time=self.start_time,
            end_time=self.end_time,
            status=self.status,
            return_status=self.return_status,
        )


class ClusterJob(object):
    """
    Class defining a cluster job, which itself will be a number of commands
    Include function to generate a python scrip to run all the commands in the job
    """

    def __init__(self, job_index, commands, description=None, env_vars=None):

        # Store args
        self.index = job_index
        self.description = (
            description  # A string for identifiying later, if desired
        )
        self.env_vars = env_vars

        # Init some book keeping
        self.start_time = None
        self.end_time = None
        self.status = "not_started"

        # Parse the input commands and convert them to instances of the command class
        assert isinstance(
            commands, collections.Sequence
        ), "Commands must be a list or similar"
        assert len(commands) > 0, "Empty list of commands"
        self.commands = commands
        for c in self.commands:
            assert isinstance(c, ClusterCommand)

    def to_dict(self):
        # TODO Could I use the native __dict__ method instead?
        return collections.OrderedDict(
            index=self.index,
            commands=[c.to_dict() for c in self.commands],
            description=self.description,
            env_vars=self.env_vars,
            start_time=self.start_time,
            end_time=self.end_time,
            status=self.status,
        )

    def create_steering_file(self, output_dir, overwrite=False):

        # Make directories if required
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create file path, and check does not already exist
        steering_file_path = os.path.abspath(
            os.path.join(output_dir, JOB_STEERING_FILE_FMT % (self.index))
        )
        if not overwrite:
            if os.path.exists(steering_file_path):
                raise Exception(
                    "Cannot create steering file for job %i : File '%s' already exists"
                    % (self.index, steering_file_path)
                )

        # Write the file
        # with open(steering_file_path, "w", encoding='utf-8') as steering_file :
        with codecs.open(steering_file_path, "w") as steering_file:
            json.dump(self.to_dict(), steering_file)

        return steering_file_path

    def prepare_to_submit(
        self,
        job_dir,
        env_shell=None,
        start_up_commands=None,
        tear_down_commands=None,
        job_mode=None,
    ):
        """
        Creates the payload to submit.

        This includes the steering file, setting paths to logging files, and the
        final execitable containing both this job command itself + any start-up/
        tear-down, the env-shell, etc.

        Modes:
            "wrapper" - creates a heavy wrapper with lots of robustness, logging, etc - recommended
            "lite_wrapper" - creates a lite wrapper with some useful stuff but crucially no fridge dependence (required on grid)
            "no_wrapper" - no wrapping whatsoever, just the command


        Also features a `lite_mode` where JSON steering files are not used,
        which can be useful for running clusters without shared file systems.
        """

        #
        # Check inputs
        #

        # Check job mode
        if job_mode is None:
            job_mode = "wrapper"
        self.mode = job_mode
        assert self.mode in ["wrapper", "lite_wrapper", "no_wrapper"]

        # Not all behaviour is supported by "lite_wrapper"
        if self.mode == "lite_wrapper":
            assert (self.env_vars is None) or (
                len(self.env_vars) == 0
            ), "Cannot provide job `env_vars` in `lite_mode`"
            # for cmd in self.commands :
            #     assert (cmd.allowed_return_status is None) or (len(cmd.allowed_return_status) == 0), "Cannot provide job `allowed_return_status` in `lite_mode`"

        # Not all behaviour is supported by "no_wrapper"

        #
        # Create a bunch of steering stuff
        #

        # Create a job name string (unique to this job)
        self.name = JOB_DIR_FMT % self.index

        # Create a direcory for this job
        self.dir = os.path.abspath(os.path.join(job_dir, self.name))
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # Define logging files
        self.out_file = os.path.abspath(
            os.path.join(self.dir, self.name + ".out")
        )
        self.err_file = os.path.abspath(
            os.path.join(self.dir, self.name + ".err")
        )

        # Create the steering file
        if self.mode == "wrapper":
            self.steering_file = self.create_steering_file(output_dir=self.dir)

        # Store the env shell
        self.env_shell = env_shell

        #
        # Write the executable (a wrapper script)
        #

        # Check if actually want to do this
        if self.mode == "no_wrapper":

            # In no wrapper mode, just use the command directly
            self.wrapper_script = None

        else:

            # Define the path to the script we are generating here
            self.wrapper_script = os.path.abspath(
                os.path.join(self.dir, JOB_NAME_FMT % self.index + ".sh")
            )

            # Write the script
            with open(self.wrapper_script, "w") as script:

                # Write the shell shebang
                script.write("#!/usr/bin/env bash\n")

                # Write a header
                script.write(
                    "\n# Autogenerated using fridge/utils/cluster/job.Job.prepare_to_submit\n"
                )

                # In lite mode where don't have the full job processor running, add some useful info
                if self.mode == "lite_wrapper":
                    script.write(
                        '\necho ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"\n'
                    )
                    script.write(
                        'echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"\n'
                    )
                    script.write(
                        'echo "--- Job %08i starting at `date`"\n' % self.index
                    )
                    script.write('echo "--- Running on host : $HOSTNAME"\n')
                    script.write('echo ""\n')

                # Write the start up commands
                if start_up_commands is not None:
                    script.write("\n# Start up commands:\n")
                    for c in start_up_commands:
                        script.write(str(c) + "\n")
                    script.write('echo ""\n')

                # Run the commands
                if self.mode == "wrapper":
                    # Run this script with a JSON steering file as input
                    # Prepend the env shell if provided
                    script.write("\n# Job command:\n")
                    if self.env_shell is not None:
                        script.write("%s " % self.env_shell)
                    job_processor_command = "python %s -s %s" % (
                        JOB_SCRIPT,
                        self.steering_file,
                    )
                    script.write(job_processor_command + "\n")

                elif self.mode == "lite_wrapper":
                    # In lite mode, just directly run the commands
                    # Also report status so at least it can be seen in the logs
                    script.write("\n# Job commands...\n")
                    for i, cmd in enumerate(self.commands):
                        script.write(
                            'echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"\n'
                        )
                        script.write(
                            'echo "--- Command %08i starting at `date`"\n' % i
                        )
                        script.write('echo "--- Command : %s"\n' % cmd.command)
                        script.write('echo ""\n')
                        if self.env_shell is not None:
                            script.write("%s " % self.env_shell)
                        script.write(cmd.command + "\n")
                        script.write("STATUS=$?\n")
                        script.write('echo ""\n')
                        script.write(
                            'echo "--- Command %08i finished at `date` : Status = $STATUS"\n'
                            % i
                        )
                        script.write(
                            'echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"\n'
                        )

                # Write the tear down commands
                if tear_down_commands is not None:
                    script.write("\n# Tear down commands:\n")
                    for c in tear_down_commands:
                        script.write(str(c) + "\n")
                    script.write('echo ""\n')

                # In lite mode where don't have the full job processor running, add some useful info
                if self.mode == "lite_wrapper":
                    script.write('echo ""\n')
                    script.write(
                        '\necho "--- Job %08i finished at `date`"\n'
                        % self.index
                    )
                    script.write(
                        'echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"\n'
                    )
                    script.write(
                        'echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"\n'
                    )

            # Make the script executable
            os.chmod(
                self.wrapper_script,
                os.stat(self.wrapper_script).st_mode | stat.S_IEXEC,
            )

        # Return the directory
        return self.dir

    def get_slurm_paths(self):
        """
        Update paths to use SLURM array variables
        Allows the paths to be used by SLURM array submissions

        Using SLURM_ARRAY_TASK_ID and %a respectively instead of an actual number,
        depending on whether used by a shell or #SBATCH statement
        (see https://slurm.schedmd.com/job_array.html)
        """

        job_name_shell = self.name.replace(
            str(self.index), "${SLURM_ARRAY_TASK_ID}"
        )
        job_name_sbatch = self.name.replace(str(self.index), "%a")
        job_dir = self.dir.replace(self.name, job_name_shell)
        job_out_file = self.out_file.replace(self.name, job_name_sbatch)
        job_err_file = self.err_file.replace(self.name, job_name_sbatch)
        job_wrapper_script = self.wrapper_script.replace(
            self.name, job_name_shell
        )

        return job_dir, job_out_file, job_err_file, job_wrapper_script


#
# Steering file dynamic I/O
#


def update_json(file_location, data):
    # TODO Link to stack overflow location
    # TODO Maybe a generic location?
    assert os.path.isfile(file_location), (
        "Input file does not exist : %s" % file_location
    )
    with codecs.open(file_location, "r+") as json_file:
        # I use OrderedDict to keep the same order of key/values in the source file
        json_from_file = json.load(
            json_file
        )  # , object_pairs_hook=collections.OrderedDict)
        for key in data:
            # make modifications here
            assert key in json_from_file, (
                "'%s' does not exist in JSON file" % key
            )
            json_from_file[key] = data[key]
        # rewind to top of the file
        json_file.seek(0)
        # sort_keys keeps the same order of the dict keys to put back to the file
        json.dump(json_from_file, json_file, indent=4, sort_keys=False)
        # just in case your new data is smaller than the older
        json_file.truncate()


def read_job_steering_key(steering_file_path, key):
    assert os.path.isfile(steering_file_path), (
        "Steering file does not exist : %s" % steering_file_path
    )
    with codecs.open(steering_file_path, "r") as steering_file:
        steering_data = json.load(
            steering_file
        )  # , object_pairs_hook=collections.OrderedDict)
        assert key in steering_data, (
            "'%s' does not exist in steering file" % key
        )
        return steering_data[key]


def write_job_steering_key(steering_file_path, key, value):
    update_json(steering_file_path, {key: value})


def read_command_steering_key(steering_file_path, command_index, key):
    commands = read_job_steering_key(steering_file_path, "commands")
    assert command_index < len(
        commands
    ), "Invaid command index %i, only %i commands" % (
        command_index,
        len(commands),
    )
    command = commands[command_index]
    assert key in command, "'%s' does not exist in command %i file" % (
        key,
        command_index,
    )
    return command[key]


def write_command_steering_key(steering_file_path, command_index, key, value):
    commands = read_job_steering_key(steering_file_path, "commands")
    assert command_index < len(
        commands
    ), "Invaid command index %i, only %i commands" % (
        command_index,
        len(commands),
    )
    command = commands[command_index]
    assert key in command, "'%s' does not exist in command %i file" % (
        key,
        command_index,
    )
    command[key] = value
    write_job_steering_key(steering_file_path, "commands", commands)


#
# Running a job
#


def run_job(steering_file, re_run=False):
    """
    Main function for running a job defined in a steering file
    """

    #
    # Start up
    #

    print("")
    print("")
    print("==================================================================")
    print("========================== Start job =============================")
    print("==================================================================")
    print("")

    # Report host
    print("Running on host : %s\n" % socket.gethostname())  # TODO -> json file

    # Check job steering file exists
    assert os.path.isfile(steering_file), (
        "JSON steering file does not exist: %s" % steering_file
    )
    print(("Loading JSON steering file : %s\n" % steering_file))

    # Check if job has already completed
    job_status = read_job_steering_key(steering_file, "status")
    if job_status == "success":
        print("Job has already completed")
        if re_run:
            print("Re-running the job")
        else:
            print("Nothing to do")
            return

    # Report description
    job_description = read_job_steering_key(steering_file, "description")
    if (job_description is not None) and (job_description != ""):
        print(("Job description : %s\n" % job_description))

    # Record the job start time
    write_job_steering_key(
        steering_file, "start_time", str(datetime.datetime.now())
    )

    # Register the job status as "started"
    write_job_steering_key(steering_file, "status", "started")

    #
    # Run job
    #

    # Check number of commands in job
    num_commands = len(read_job_steering_key(steering_file, "commands"))
    assert num_commands > 0, "No commands found in job"

    # Book-keeping
    num_commands_successful = 0

    # Loop over commands
    for command_index in range(0, num_commands):

        # Grab command info
        command_description = read_command_steering_key(
            steering_file, command_index, "description"
        )
        command = read_command_steering_key(
            steering_file, command_index, "command"
        )
        allowed_return_status = read_command_steering_key(
            steering_file, command_index, "allowed_return_status"
        )

        print("")
        print("")
        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        )
        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        )
        print("--- Command info : ")
        print(("---   Number                : %08i" % command_index))
        print(("---   Description           : %s" % command_description))
        print(("---   Command               : %s" % command))
        print(
            (
                "---   Allowed return status : %s"
                % (
                    "Any"
                    if allowed_return_status is None
                    else allowed_return_status
                )
            )
        )

        # Force environment variables
        env_vars = read_job_steering_key(steering_file, "env_vars")
        if (env_vars is not None) and (len(env_vars) > 0):
            print("--- Forcing environment variables :")
            for var_name, var_val in list(env_vars.items()):
                os.environ[var_name] = var_val
                print(("---   %s : %s" % (var_name, var_val)))
                # TODO Warn if overwriting an existing variable

        #
        # Run command
        #

        print("--- Status :")

        # Check if the command has already completed
        command_status = read_command_steering_key(
            steering_file, command_index, "status"
        )
        if command_status == "success":
            print("---   Already completed")
            if re_run:
                print("---   Will re-run the command")
            else:
                print("---   Skipping the command")
                num_commands_successful += 1
                continue
        else:
            print("---   Not yet run")

        # Record the command start time
        start_time = datetime.datetime.now()
        write_command_steering_key(
            steering_file, command_index, "start_time", str(start_time)
        )
        print(("---   Command starting at %s" % start_time))
        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        )
        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        )
        print("")

        # Flush stdout before running the command
        # This is to make the output from this script line up with the output from the command itself
        sys.stdout.flush()

        # Run the command
        try:

            # Init some variables
            return_status = None
            exception = None
            bad_return_status = False
            command_success = False

            # Register the command status as "started"
            write_command_steering_key(
                steering_file, command_index, "status", "started"
            )

            # Run the command
            return_status = subprocess.call(command, shell=True)

            # Record the command return status
            write_command_steering_key(
                steering_file, command_index, "return_status", return_status
            )

            # If make it here, command ran successfully
            command_success = True
            # Check return status
            if allowed_return_status is not None:
                if return_status not in allowed_return_status:
                    command_success = False
                    bad_return_status = True

        except Exception as e:
            exception = e
            command_success = False

        # Record the command end time (even if failed)
        end_time = datetime.datetime.now()
        write_command_steering_key(
            steering_file, command_index, "end_time", str(end_time)
        )

        # Check if command succeeded
        if command_success:

            # Record status
            write_command_steering_key(
                steering_file, command_index, "status", "success"
            )
            num_commands_successful += 1

            # Report
            print("")
            print(
                "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            )
            print(
                "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            )
            print(("--- Command %08i complete :" % command_index))
            print(("---   Return status : %s" % return_status))
            print(("---   Finished at   : %s" % end_time))
            print(("---   Took          : %s" % (end_time - start_time)))
            print(
                "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            )
            print(
                "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            )

        # Otherwise command failed
        else:

            # Record status
            write_command_steering_key(
                steering_file, command_index, "status", "failure"
            )

            # Report error and bail
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            print("!!! -- ERROR : ")
            if exception is not None:
                print(("!!! ---- Exception thrown : %s" % exception))
            if bad_return_status:
                print(("!!! ---- Bad return status : %i" % return_status))
                print(
                    (
                        "!!! ---- Allow return status values : %s"
                        % allowed_return_status
                    )
                )
            print("!!! ---- Exiting job early")
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

    #
    # Tear down
    #

    # Register the job status as "success" or "failure" based on whether commands all ran successfully
    if num_commands_successful < num_commands:
        write_job_steering_key(steering_file, "status", "failure")
    else:
        write_job_steering_key(steering_file, "status", "success")

    # Record the job end time
    write_job_steering_key(
        steering_file, "end_time", str(datetime.datetime.now())
    )

    print("")
    print("")
    print("==================================================================")
    print("=========================== End job ==============================")
    print("==================================================================")
    print("")


#
# Test
#


def test():

    test_dir = "./tmp"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    commands = [
        ClusterCommand("echo 'bar'", description="This will bar"),
        ClusterCommand("echo 'bar'", description="This will bar"),
        # ClusterCommand( "echo %s" % [ i for i in range(10000)], description="A long command" ),
    ]

    job = ClusterJob(
        output_dir=test_dir,
        job_number=123,
        commands=commands,
        description="This is a test",
        env_vars={"FOO": "BAR"},
    )

    steering_file = job.create_steering_file(test_dir)

    run_job(steering_file)


#
# Main
#

# This is what is actually run by cluster jobs

if __name__ == "__main__":

    import argparse

    # Get the inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steering-file", required=True, type=str)
    parser.add_argument("-r", "--re-run", action="store_true")
    args = parser.parse_args()

    run_job(args.steering_file, args.re_run)
