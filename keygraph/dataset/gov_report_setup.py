# run_govreport_setup.py

# Import the specific adapter class from your file
from data import GovReportAdapter

def main():
    """
    This function demonstrates how to set up and use the GovReportAdapter.
    """
    # 1. Define the dataset path with the specific summarization subset.
    govreport_path = "ccdv/govreport-summarization" # <-- THIS LINE IS UPDATED
    print(f"Setting up GovReport dataset from: {govreport_path}")

    # 2. Create an instance of your adapter.
    # The first time this runs, it will download the dataset.
    adapter = GovReportAdapter(dataset_path=govreport_path)
    print("-> DatasetAdapter created successfully. Data is loaded/cached.")

    # 3. Get a few samples from the 'test' split.
    print("\nFetching 3 samples from the 'test' split...")
    samples = adapter.get_samples(split="validation", num_samples=3)
    print(f"-> Successfully fetched {len(samples)} samples.")

    # 4. Format the first sample into a prompt to see the output.
    if samples:
        print("\n--- Example Formatted Prompt ---")
        first_sample = samples[0]
        prompt = adapter.format_prompt(first_sample)
        
        # Print the start of the prompt for verification
        print(prompt[:500] + "...")
        print("--- End of Example ---")

if __name__ == "__main__":
    main()


  