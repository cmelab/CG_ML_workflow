#!/usr/bin/env python
"""Initialize the project's data space.
Iterates over all defined state points and initializes
the associated job workspace directories.
The result of running this file is the creation of a signac workspace:
    - signac.rc file containing the project name
    - signac_statepoints.json summary for the entire workspace
    - workspace/ directory that contains a sub-directory of every individual statepoint
    - signac_statepoints.json within each individual statepoint sub-directory.
"""

import logging
from collections import OrderedDict
from itertools import product

import signac


def get_parameters():
    parameters = OrderedDict()

    # project parameters
    parameters["log_dir"] = ["/home/marjanalbooyeh/logs/ML/"]
    parameters["project"] = ["NN_multi_PPS"]
    parameters["group"] = ["10_pps"]
    parameters["notes"] = ["multi N=10, appended input, neighbors in one row"]
    parameters["tags"] = [["PPS", "NN"]]

    # dataset parameters
    parameters["data_path"] = ["/home/marjanalbooyeh/logs/pps_rigid/2023-03-09-18:47:47/dataset"]
    parameters["inp_mode"] = ["append", "stack"]
    parameters["batch"] = [64, 128, 256]

    # model parameters
    parameters["hidden_dim"] = [16, 32, 64, 128]
    parameters["n_layer"] = [2, 3, 4]
    parameters["act_fn"] = ["ReLU", "Tanh"]
    parameters["dropout"] = [0.3, 0.7]

    # optimizer parameters
    parameters["optim"] = ["optim", "Adam"]
    parameters["lr"] = [0.1, 0.01]
    parameters["decay"] = [0.01, 0.0001]

    # run parameters
    parameters["epochs"] = [1000]

    return list(parameters.keys()), list(product(*parameters.values()))


custom_job_doc = {}  # add keys and values for each job document created


def main(root=None):
    project = signac.init_project("ML_CG", root=root)  # Set the signac project name
    param_names, param_combinations = get_parameters()
    # Create jobs
    for params in param_combinations:
        parent_statepoint = dict(zip(param_names, params))
        parent_job = project.open_job(parent_statepoint)
        parent_job.init()
        parent_job.doc.setdefault("done", False)

    project.write_statepoints()
    return project


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
