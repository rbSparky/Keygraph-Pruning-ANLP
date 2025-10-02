import json
import csv
import os
from datetime import datetime
from typing import Dict,Any


def log_metrics_to_csv(file_path:str,metrics:Dict[str,Any]):
    """Log metrics to a CSV file."""
    file_exists =os.path.exists(file_path)

    with open(file_path,'a',newline ='')as csvfile:
        writer =csv.DictWriter(csvfile,fieldnames =metrics.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics)


def log_metrics_to_jsonl(file_path:str,metrics:Dict[str,Any]):
    """Log metrics to a JSONL file."""
    with open(file_path,'a')as jsonl_file:
        jsonl_file.write(json.dumps(metrics)+'\n')


def create_run_directory(base_dir:str ="runs")->str:
    """Create a timestamped run directory."""
    timestamp =datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir =os.path.join(base_dir,timestamp)
    os.makedirs(run_dir,exist_ok =True)
    return run_dir