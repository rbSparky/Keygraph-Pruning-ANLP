import pandas as pd

# Replace with your CSV file path
file_path = "results.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Calculate the average of each column
column_averages = numeric_df.mean()

# Print the result
print("Average of each numeric column:")
print(column_averages)
