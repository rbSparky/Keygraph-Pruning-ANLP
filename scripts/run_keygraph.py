import os
import sys
import argparse
import json
import yaml
import torch
import time


sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from keygraph.utils import set_seed,get_device
from keygraph.models import load_model_and_tokenizer
from keygraph.data import GovReportAdapter,NarrativeQAAdapter,QasperAdapter
from keygraph.eval_metrics import evaluate_prediction
from keygraph.logging_utils import log_metrics_to_csv,log_metrics_to_jsonl
from keygraph.method.keygraph_cache import KeyGraphCache
from keygraph.method.attention_patch import keygraph_attention_patch


def load_config(config_path ="../configs/paths.yaml"):
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


def keygraph_generate(
model,
tokenizer,
prompt:str,
max_new_tokens:int =128,
r_dim:int =32,
knn_k:int =16,
tau:float =0.8,
rescue:bool =True,
rescue_probe_size:int =6,
upper_layers_only:bool =False):
    """
    Generate using KeyGraph pruning method.

    Returns:
        generated_text (str): The generated text
        metrics (dict): Performance metrics
    """
    start_time =time.time()
    torch.cuda.reset_peak_memory_stats()


    keygraph_cache =KeyGraphCache(
    model,tokenizer,prompt,
    r_dim =r_dim,
    tau =tau,
    knn_k =knn_k,
    rescue =rescue,
    rescue_probe_size =rescue_probe_size,
    upper_layers_only =upper_layers_only)


    compression_ratio =keygraph_cache.get_compression_ratio()
    kv_bytes_saved =keygraph_cache.get_kv_bytes_saved()


    with keygraph_attention_patch(model,keygraph_cache)as patched_cache:

        inputs =tokenizer(
        prompt,
        return_tensors ="pt",
        padding =True,
        truncation =True).to(model.device)


        with torch.no_grad():
            outputs =model.generate(
            **inputs,
            max_new_tokens =max_new_tokens,
            do_sample =False,
            return_dict_in_generate =True,
            output_scores =True,
            past_key_values =None,
            use_cache =True)

    end_time =time.time()


    generated_tokens =outputs.sequences.shape[1]-inputs.input_ids.shape[1]
    total_time =end_time -start_time
    tokens_per_second =generated_tokens /total_time if total_time >0 else 0
    peak_memory =torch.cuda.max_memory_allocated()/(1024 **2)


    generated_text =tokenizer.decode(
    outputs.sequences[0][inputs.input_ids.shape[1]:],
    skip_special_tokens =True)

    metrics ={
    "tokens_generated":generated_tokens,
    "total_time_seconds":total_time,
    "tokens_per_second":tokens_per_second,
    "peak_vram_mb":peak_memory,
    "kv_cache_method":"keygraph",
    "compression_ratio":compression_ratio,
    "kv_bytes_saved_mb":kv_bytes_saved /(1024 **2),
    "r_dim":r_dim,
    "knn_k":knn_k,
    "tau":tau}

    return generated_text,metrics


def main():
    parser =argparse.ArgumentParser(description ="Run KeyGraph pruning experiments")
    parser.add_argument("--model_dir",required =True,help ="Path to model directory")
    parser.add_argument("--dataset",required =True,choices =["govreport","narrativeqa","qasper"],
    help ="Dataset to use")
    parser.add_argument("--dataset_dir",required =True,help ="Path to dataset directory")
    parser.add_argument("--num_samples",type =int,default =5,help ="Number of samples to process")
    parser.add_argument("--max_new_tokens",type =int,default =128,help ="Maximum new tokens to generate")
    parser.add_argument("--r_dim",type =int,default =32,help ="Random projection dimension")
    parser.add_argument("--knn_k",type =int,default =16,help ="Number of neighbors for kNN")
    parser.add_argument("--tau",type =float,default =0.8,help ="Cosine similarity threshold")
    parser.add_argument("--no_rescue",action ="store_true",help ="Disable rescue expansion")
    parser.add_argument("--rescue_probe_size",type =int,default =6,help ="Rescue probe size")
    parser.add_argument("--upper_layers_only",action ="store_true",help ="Process only upper layers")
    parser.add_argument("--seed",type =int,default =42,help ="Random seed")
    parser.add_argument("--save_preds",action ="store_true",help ="Save predictions to file")

    args =parser.parse_args()


    set_seed(args.seed)


    device =get_device()


    model,tokenizer =load_model_and_tokenizer(args.model_dir,device)


    dataset_adapter =get_dataset_adapter(args.dataset,args.dataset_dir)


    samples =dataset_adapter.get_samples("test",args.num_samples)


    output_dir =f"runs/keygraph_{args.dataset}"
    os.makedirs(output_dir,exist_ok =True)


    csv_file =os.path.join(output_dir,"results.csv")


    preds_file =os.path.join(output_dir,"predictions.jsonl")if args.save_preds else None


    all_metrics =[]

    for i,sample in enumerate(samples):
        print(f"\nProcessing sample {i +1}/{len(samples)}")
        print(f"Sample ID: {sample.get('id','N/A')}")


        prompt =dataset_adapter.format_prompt(sample)
        print(f"Prompt length: {len(prompt)} characters")


        generated_text,metrics =keygraph_generate(
        model,tokenizer,prompt,
        max_new_tokens =args.max_new_tokens,
        r_dim =args.r_dim,
        knn_k =args.knn_k,
        tau =args.tau,
        rescue =not args.no_rescue,
        rescue_probe_size =args.rescue_probe_size,
        upper_layers_only =args.upper_layers_only)


        metrics["sample_id"]=sample.get("id",f"{args.dataset}_{i}")
        metrics["dataset"]=args.dataset
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
        print(f"Compression ratio: {metrics['compression_ratio']:.4f}")
        print(f"KV bytes saved: {metrics['kv_bytes_saved_mb']:.2f} MB")
        if "rougeL"in metrics:
            print(f"ROUGE-L: {metrics['rougeL']:.4f}")
        if "f1"in metrics:
            print(f"F1: {metrics['f1']:.4f}")


        print(f"Generated text: {generated_text[:100]}{'...'if len(generated_text)>100 else ''}")


    if all_metrics:
        avg_tokens_per_second =sum(m["tokens_per_second"]for m in all_metrics)/len(all_metrics)
        avg_peak_vram =sum(m["peak_vram_mb"]for m in all_metrics)/len(all_metrics)
        avg_compression_ratio =sum(m["compression_ratio"]for m in all_metrics)/len(all_metrics)
        avg_kv_bytes_saved =sum(m["kv_bytes_saved_mb"]for m in all_metrics)/len(all_metrics)

        print("\n"+"="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Average tokens/second: {avg_tokens_per_second:.2f}")
        print(f"Average peak VRAM: {avg_peak_vram:.2f} MB")
        print(f"Average compression ratio: {avg_compression_ratio:.4f}")
        print(f"Average KV bytes saved: {avg_kv_bytes_saved:.2f} MB")

        if args.dataset =="govreport":
            avg_rougeL =sum(m["rougeL"]for m in all_metrics if "rougeL"in m)/len(all_metrics)
            print(f"Average ROUGE-L: {avg_rougeL:.4f}")
        else:
            avg_f1 =sum(m["f1"]for m in all_metrics if "f1"in m)/len(all_metrics)
            print(f"Average F1: {avg_f1:.4f}")


if __name__ =="__main__":
    main()