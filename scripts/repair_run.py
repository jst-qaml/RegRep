import argparse
import json
from pprint import pprint
import re
import sys
import os
import time
import traceback

from repair.semseg.main import main as repair
from repair.semseg.models import SemsegModel, DepthEstModel
from repair.semseg.problems import (
    SemSegRepairProblem, 
    DepthEstRepairProblem, 
    ImprovementType
    )

default_metric_value = 0

class Logger:
    def __init__(self, file_path):
        self.terminal = sys.__stdout__
        self.file_path = file_path
        # Clear the log file if it already exists
        if os.path.exists(self.file_path):
            open(self.file_path, 'w').close()  # Create or clear the log file
        # self.log_file = open(self.file_path, 'a')  # Open the log file in append mode

    def write(self, message):
        self.terminal.write(message)

        # write on log
        # Skip or modify messages here
        # Example: Skip tqdm lines not in 10th percentile
        if re.match(r'(\d{1,3})%\|.*\|', message):  # Pattern for tqdm
            # Check if the tqdm line is not a 10th percentile
            percent_match = re.search(r'(\d+)%', message)
            if percent_match and int(percent_match.group(1)) % 10 != 0:
                return  # Skip this line

        # Reduce multiple newlines to a single newline
        message = re.sub(r'\n+', '\n', message)

        # Write to log file and print to stdout
        with open(self.file_path, "a") as log_file: 
            log_file.write(message)
            log_file.flush()

    def close(self):
        sys.stdout = sys.__stdout__  # Restore original stdout
        sys.stderr = sys.__stderr__  # Restore original stderr


# Function to run a repairing session
# this function runs in a new thread
def repair_session(session_id, logs_path, run_hyperparams, unsearched_parameters):

    # Filepath for output
    output_filepath = f'running_{session_id}.log'

    log_file = logs_path + output_filepath
    open(log_file, "w").close()

    print(f"process {session_id} running!")

    results_path = logs_path + 'results/'
    results_file = f"{session_id}.json"

    if os.path.exists(results_file):
        print(f"Session {session_id} completed with status: Already Run")
        return

    # override certain parameters
    if run_hyperparams["problem"] == "SemSegRepairProblem":
        run_hyperparams["model"] = SemsegModel(
            run_hyperparams["model_path"],
            run_hyperparams["device"],
        )
        run_hyperparams["problem"] = SemSegRepairProblem
        metric_sort_by = "mIoU"
    elif run_hyperparams["problem"] == "DepthEstRepairProblem":
        run_hyperparams["model"] = DepthEstModel(
            run_hyperparams["model_path"],
            run_hyperparams["device"],
        )
        run_hyperparams["problem"] = DepthEstRepairProblem
        metric_sort_by = "mee"
        default_metric_value = 1

    print(
        f"Starting session {session_id} on {run_hyperparams['device']}",
        f"\nwith hyperparameters:",
    )
    pprint(run_hyperparams)

    run_start = time.time()
    try:
        exp_name, original_metrics, repaired_metrics = repair(**run_hyperparams)
        # exp_name, original_metrics, repaired_metrics = ("", {}, {});time.sleep(5)
        run_end  = time.time()
        status = "success"
        exception = ""
    except Exception as exc:
        run_end  = time.time()
        traceback.print_exc()
        exp_name, original_metrics, repaired_metrics = None, {}, {}
        status = "failed"
        exception = str(exc)


    run_duration = run_end - run_start
    run_duration = f"{run_duration//3600}hs {(run_duration%3600)//60}min" if run_duration >= 3600 else f"{run_duration//60}min"

    status_log_file = log_file.replace("running", status)

    os.rename(log_file, status_log_file)

    if run_hyperparams["problem"].improvement_type == ImprovementType.higher_is_better:
        improvements = {
            metric: (repaired_metrics[metric] - original_metrics[metric]) / original_metrics[metric] * 100
            for metric in original_metrics
        }
    else:
        improvements = {
            metric: (1 - (repaired_metrics[metric] / original_metrics[metric])) * 100
            for metric in original_metrics
        }

    if metric_sort_by not in improvements:
        improvements[metric_sort_by] = default_metric_value

    searched_hyperparams = {
        param: run_hyperparams[param]
        for param in run_hyperparams
        if param not in unsearched_parameters
    }

    log_entry = {
        "name": exp_name,
        "status": status,
        "exception": exception,
        "improvements" : improvements,
        "duration": run_duration,
        "logs_path": status_log_file,
        "hyperparameters" : searched_hyperparams,
        "evaluated": False,
    }

    # run repair function with the given args
    print(f"Session {session_id} completed with status: {status}")

    os.makedirs(results_path, exist_ok=True)
    with open(results_path+results_file, mode='w') as f:
        json.dump(log_entry, f, indent=4)

if __name__ == "__main__":
    # get the session id args
    parser = argparse.ArgumentParser(description="""Repair runner from json with file logging""")
    parser.add_argument(
        "--json", type=str, required=True, help="Path to the parameteres json"
    )
    args = parser.parse_args()

    # load parameters
    with open(args.json) as f:
        run_params = json.load(f)

    repair_session(
        run_params["session_id"],
        run_params["logs_path"],
        run_params["run_hyperparameters"],
        run_params["unsearched_parameters"]
    )