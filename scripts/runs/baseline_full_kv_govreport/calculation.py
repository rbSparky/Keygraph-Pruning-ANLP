import pandas as pd

# Define the path to your CSV file
# You can change 'metrics.csv' to the name of your file
file_name = 'key_graph_results.csv'

try:
    # Read the data from the specified CSV file
    df = pd.read_csv(file_name)

    # Select the first 20 samples (or fewer if the file has less)
    num_samples = min(20, len(df))
    samples_df = df.head(num_samples)

    # Select only the numeric columns to average
    numeric_columns = samples_df.select_dtypes(include=['number'])

    # Calculate the average for each numeric metric
    average_metrics = numeric_columns.mean()

    # Print the results
    print(f"Averaging metrics over the first {num_samples} sample(s) from '{file_name}':")
    print(average_metrics)

except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")