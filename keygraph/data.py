import os
from typing import Dict,Any,List,Tuple
from datasets import load_dataset,Dataset,DatasetDict
import glob
import pandas as pd

class DatasetAdapter:
    """Base class for dataset adapters."""

    def __init__(self,dataset_path:str):
        self.dataset_path =dataset_path

        try:
            self.dataset =load_dataset(dataset_path,local_files_only =True)
        except:

            try:
                self.dataset ={
                "test":load_dataset(dataset_path,split ="test",local_files_only =True)}
            except:

                self.dataset =self._load_from_parquet()

    def _load_from_parquet(self):
        """Load dataset from parquet files."""
        


        parquet_files =glob.glob(os.path.join(self.dataset_path,"**/*.parquet"),recursive =True)


        splits ={}
        for file_path in parquet_files:
            filename =os.path.basename(file_path)
            if "test"in filename:
                split_name ="test"
            elif "validation"in filename:
                split_name ="validation"
            elif "train"in filename:
                split_name ="train"
            else:
                continue


            df =pd.read_parquet(file_path)
            splits[split_name]=Dataset.from_pandas(df)

        return DatasetDict(splits)if splits else {}

    def get_samples(self,split:str,num_samples:int)->List[Dict[str,Any]]:
        """Get samples from the dataset."""
        raise NotImplementedError

    def format_prompt(self,sample:Dict[str,Any])->str:
        """Format a sample into a prompt."""
        raise NotImplementedError


class GovReportAdapter(DatasetAdapter):
    """Adapter for GovReport dataset."""

    def get_samples(self,split:str ="test",num_samples:int =10)->List[Dict[str,Any]]:
        """Get samples from GovReport dataset."""
        samples =[]
        dataset_split =self.dataset[split]

        for i in range(min(num_samples,len(dataset_split))):
            sample =dataset_split[i]
            samples.append({
            "id":f"govreport_{i}",
            "report":sample["report"],
            "summary":sample["summary"]})

        return samples

    def format_prompt(self,sample:Dict[str,Any])->str:
        """Format GovReport sample into a summarization prompt."""
        return f"Summarize the following government report:\n\n{sample['report']}\n\nSummary:"


class NarrativeQAAdapter(DatasetAdapter):
    """Adapter for NarrativeQA dataset."""

    def get_samples(self,split:str ="test",num_samples:int =10)->List[Dict[str,Any]]:
        """Get samples from NarrativeQA dataset."""
        samples =[]
        dataset_split =self.dataset[split]

        for i in range(min(num_samples,len(dataset_split))):
            sample =dataset_split[i]
            samples.append({
            "id":f"narrativeqa_{i}",
            "context":sample["document"]["summary"]["text"],
            "question":sample["question"]["text"],
            "answers":[answer["text"]for answer in sample["answers"]]})

        return samples

    def format_prompt(self,sample:Dict[str,Any])->str:
        """Format NarrativeQA sample into a QA prompt."""
        return f"Context: {sample['context']}\n\nQuestion: {sample['question']}\n\nAnswer:"


class QasperAdapter(DatasetAdapter):
    """Adapter for Qasper dataset."""

    def get_samples(self,split:str ="test",num_samples:int =10)->List[Dict[str,Any]]:
        """Get samples from Qasper dataset."""
        samples =[]
        dataset_split =self.dataset[split]

        for i in range(min(num_samples,len(dataset_split))):
            sample =dataset_split[i]

            if sample["qas"]:
                qa =sample["qas"][0]
                samples.append({
                "id":f"qasper_{i}",
                "context":sample["abstract"],
                "question":qa["question"],
                "answers":[answer["answer"]["unstructured"]for answer in qa["answers"]]})

        return samples

    def format_prompt(self,sample:Dict[str,Any])->str:
        """Format Qasper sample into a QA prompt."""
        return f"Paper abstract: {sample['context']}\n\nQuestion: {sample['question']}\n\nAnswer:"