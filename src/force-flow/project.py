"""Define the project's workflow logic and operation functions.
Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:
    $ python src/project.py --help
"""
import sys
import os

# getting the name of the directory
# where the  file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
from trainer import MLTrainer


class MyProject(FlowProject):
    pass


class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
    template = "borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="short",
            help="Specify the partition to submit to."
        )


class R2(DefaultSlurmEnvironment):
    hostname_pattern = "r2"
    template = "r2.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="shortq",
            help="Specify the partition to submit to."
        )


class Fry(DefaultSlurmEnvironment):
    hostname_pattern = "fry"
    template = "fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="batch",
            help="Specify the partition to submit to."
        )


# Definition of project-related labels (classification)
@MyProject.label
def sampled(job):
    return job.doc.get("done")


@directives(executable="python -u")
@directives(ngpu=1)
@MyProject.operation
@MyProject.post(sampled)
def sample(job):
    with job:
        print("-----------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("-----------------------")
        print("----------------------")
        print("Creating the Trainer class...")
        print("----------------------")
        trainer_obj = MLTrainer(job.sp, job.id)
        job.doc["wandb_run_name"] = trainer_obj.wandb_run_name
        job.doc["wandb_run_path"] = trainer_obj.wandb_run_path
        print("Training...")
        print("----------------------")
        trainer_obj.run()
        job.doc["best_val_error"] = trainer_obj.best_val_error
        job.doc["test_error"] = trainer_obj.test_error
        job.doc["done"] = True
        print("-----------------------------")
        print("Training finished")
        print("-----------------------------")


def submit_project():
    MyProject().run()


if __name__ == "__main__":
    MyProject().main()
