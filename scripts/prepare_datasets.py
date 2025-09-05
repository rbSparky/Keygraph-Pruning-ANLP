import os
import sys
import json
import yaml
import argparse


sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from keygraph.data import GovReportAdapter,NarrativeQAAdapter,QasperAdapter


def load_config(config_path ="configs/paths.yaml"):
    """Load configuration from YAML file."""
    with open(config_path,'r')as file:
        config =yaml.safe_load(file)
    return config


def main():
    parser =argparse.ArgumentParser(description ="Prepare and verify datasets")
    parser.add_argument("--gov_dir",help ="Path to GovReport dataset")
    parser.add_argument("--nqa_dir",help ="Path to NarrativeQA dataset")
    parser.add_argument("--qasper_dir",help ="Path to QASPER dataset")
    args =parser.parse_args()


    config =load_config()


    gov_dir =args.gov_dir or os.path.expandvars(config["gov_dir"])
    nqa_dir =args.nqa_dir or os.path.expandvars(config["nqa_dir"])
    qasper_dir =args.qasper_dir or os.path.expandvars(config["qasper_dir"])

    print("Verifying datasets...")


    os.makedirs("runs/dataset_samples",exist_ok =True)


    print(f"\nLoading GovReport from: {gov_dir}")
    gov_adapter =GovReportAdapter(gov_dir)
    gov_samples =gov_adapter.get_samples("test",10)
    print(f"GovReport test samples: {len(gov_samples)}")


    with open("runs/dataset_samples/govreport_samples.jsonl","w")as f:
        for sample in gov_samples:
            f.write(json.dumps(sample)+"\n")


    if gov_samples:
        sample =gov_samples[0]
        print(f"First GovReport sample ID: {sample['id']}")
        print(f"Report length: {len(sample['report'])} characters")
        print(f"Summary length: {len(sample['summary'])} characters")
        print(f"Prompt example:\n{gov_adapter.format_prompt(sample)[:200]}...")


    print(f"\nLoading NarrativeQA from: {nqa_dir}")
    nqa_adapter =NarrativeQAAdapter(nqa_dir)
    nqa_samples =nqa_adapter.get_samples("test",10)
    print(f"NarrativeQA test samples: {len(nqa_samples)}")


    with open("runs/dataset_samples/narrativeqa_samples.jsonl","w")as f:
        for sample in nqa_samples:
            f.write(json.dumps(sample)+"\n")


    if nqa_samples:
        sample =nqa_samples[0]
        print(f"First NarrativeQA sample ID: {sample['id']}")
        print(f"Context length: {len(sample['context'])} characters")
        print(f"Question: {sample['question']}")
        print(f"Number of answers: {len(sample['answers'])}")
        print(f"Prompt example:\n{nqa_adapter.format_prompt(sample)[:200]}...")


    print(f"\nLoading QASPER from: {qasper_dir}")
    qasper_adapter =QasperAdapter(qasper_dir)
    qasper_samples =qasper_adapter.get_samples("test",10)
    print(f"QASPER test samples: {len(qasper_samples)}")


    with open("runs/dataset_samples/qasper_samples.jsonl","w")as f:
        for sample in qasper_samples:
            f.write(json.dumps(sample)+"\n")


    if qasper_samples:
        sample =qasper_samples[0]
        print(f"First QASPER sample ID: {sample['id']}")
        print(f"Context length: {len(sample['context'])} characters")
        print(f"Question: {sample['question']}")
        print(f"Number of answers: {len(sample['answers'])}")
        print(f"Prompt example:\n{qasper_adapter.format_prompt(sample)[:200]}...")

    print("\nDataset verification complete!")


if __name__ =="__main__":
    main()