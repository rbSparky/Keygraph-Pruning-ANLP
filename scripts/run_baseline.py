import os
import sys
import argparse
import json
import yaml
import torch




from keygraph.utils import set_seed,get_device
from keygraph.models import load_model_and_tokenizer
from keygraph.data import GovReportAdapter,NarrativeQAAdapter,QasperAdapter
from keygraph.baseline.full_kv import full_kv_generate
from keygraph.baseline.sliding_window import sliding_window_generate
from keygraph.eval_metrics import evaluate_prediction
from keygraph.logging_utils import log_metrics_to_csv,log_metrics_to_jsonl


def load_config(config_path ="configs/paths.yaml"):
    """Load configuration from YAML file."""
    with open(config_path,'r')as file:
        config =yaml.safe_load(file)
    return config


def get_dataset_adapter(dataset_name,dataset_dir):
    """Get the appropriate dataset adapter."""
    if dataset_name =="govreport":
        return GovReportAdapter(dataset_dir)
    elif dataset_name =="narrativeqa":
        return NarrativeQAAdapter(dataset_dir)
    elif dataset_name =="qasper":
        return QasperAdapter(dataset_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def main():
    parser =argparse.ArgumentParser(description ="Run baseline experiments")
    parser.add_argument("--model_dir",required =True,help ="Path to model directory")
    parser.add_argument("--dataset",required =True,choices =["govreport","narrativeqa","qasper"],
    help ="Dataset to use")
    parser.add_argument("--dataset_dir",required =True,help ="Path to dataset directory")
    parser.add_argument("--num_samples",type =int,default =5,help ="Number of samples to process")
    parser.add_argument("--baseline",choices =["full","window"],required =True,
    help ="Baseline method to use")
    parser.add_argument("--window",type =int,default =1024,help ="Window size for sliding window")
    parser.add_argument("--max_new_tokens",type =int,default =128,help ="Maximum new tokens to generate")
    parser.add_argument("--seed",type =int,default =42,help ="Random seed")
    parser.add_argument("--save_preds",action ="store_true",help ="Save predictions to file")

    args =parser.parse_args()


    set_seed(args.seed)


    device =get_device()


    model,tokenizer =load_model_and_tokenizer(args.model_dir,device)


    dataset_adapter =get_dataset_adapter(args.dataset,args.dataset_dir)


    samples =dataset_adapter.get_samples("test",args.num_samples)


    output_dir =f"runs/baseline_{args.baseline}_{args.dataset}"
    os.makedirs(output_dir,exist_ok =True)


    csv_file =os.path.join(output_dir,"results.csv")


    preds_file =os.path.join(output_dir,"predictions.jsonl")if args.save_preds else None


    all_metrics =[]

    for i,sample in enumerate(samples):
        print(f"\nProcessing sample {i +1}/{len(samples)}")
        print(f"Sample ID: {sample.get('id','N/A')}")


        prompt =dataset_adapter.format_prompt(sample)
        print(f"Prompt length: {len(prompt)} characters")


        if args.baseline =="full":
            generated_text,metrics =full_kv_generate(
            model,tokenizer,prompt,args.max_new_tokens)
        elif args.baseline =="window":
            generated_text,metrics =sliding_window_generate(
            model,tokenizer,prompt,args.max_new_tokens,args.window)


        metrics["sample_id"]=sample.get("id",f"{args.dataset}_{i}")
        metrics["dataset"]=args.dataset
        metrics["baseline"]=args.baseline
        metrics["window_size"]=args.window if args.baseline =="window"else "N/A"
        metrics["max_new_tokens"]=args.max_new_tokens


        if args.dataset =="govreport":
            task_type ="summarization"
            ground_truth =sample["summary"]
        else:
            task_type ="qa"
            ground_truth =sample["answers"]

        if "summary"in sample or "answers"in sample:
            eval_scores =evaluate_prediction(task_type,generated_text,ground_truth)
            metrics.update(eval_scores)


        log_metrics_to_csv(csv_file,metrics)
        if preds_file:
            pred_record ={
            "sample_id":sample.get("id",f"{args.dataset}_{i}"),
            "prompt":prompt,
            "generated":generated_text,
            "ground_truth":ground_truth}
            log_metrics_to_jsonl(preds_file,pred_record)

        all_metrics.append(metrics)

        print(f"Generated {metrics['tokens_generated']} tokens")
        print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
        print(f"Peak VRAM: {metrics['peak_vram_mb']:.2f} MB")
        if "rougeL"in metrics:
            print(f"ROUGE-L: {metrics['rougeL']:.4f}")
        if "f1"in metrics:
            print(f"F1: {metrics['f1']:.4f}")


        print(f"Generated text: {generated_text[:100]}{'...'if len(generated_text)>100 else ''}")


    if all_metrics:
        avg_tokens_per_second =sum(m["tokens_per_second"]for m in all_metrics)/len(all_metrics)
        avg_peak_vram =sum(m["peak_vram_mb"]for m in all_metrics)/len(all_metrics)

        print("\n"+"="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Average tokens/second: {avg_tokens_per_second:.2f}")
        print(f"Average peak VRAM: {avg_peak_vram:.2f} MB")

        if args.dataset =="govreport":
            avg_rougeL =sum(m["rougeL"]for m in all_metrics if "rougeL"in m)/len(all_metrics)
            print(f"Average ROUGE-L: {avg_rougeL:.4f}")
        else:
            avg_f1 =sum(m["f1"]for m in all_metrics if "f1"in m)/len(all_metrics)
            print(f"Average F1: {avg_f1:.4f}")


if __name__ =="__main__":
    main()