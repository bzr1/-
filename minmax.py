import pandas as pd

# Read the Excel file into a DataFrame
df = pd.read_excel("/Users/zhiranbai/Downloads/工作/数据中心AI/2冷源流量补值代码-3种模式/补充混合+板换+冷机水流量.xlsx")

# Function to handle values with '/'
def handle_slash(value):
    if isinstance(value, str) and '/' in value:
        return [float(part) for part in value.split('/')]
    else:
        return float(value)

# Lists to store minimum and maximum values
min_values = []
max_values = []

# Iterate over each column starting from the second one
for column in df.columns[3:]:
    # Convert values to float, handling '/' separated values
    df[column] = df[column].apply(handle_slash)

    # Find and store the minimum and maximum values of the column
    min_val = df[column].apply(lambda x: min(x) if isinstance(x, list) else x).min()
    max_val = df[column].apply(lambda x: max(x) if isinstance(x, list) else x).max()
    
    min_values.append(min_val)
    max_values.append(max_val)

# Print all the minimum values
print("Minimum values:")
for column, min_val in zip(df.columns[3:], min_values):
    print(f"Minimum value of column '{column}': {min_val}")

# Print all the maximum values
print("\nMaximum values:")
for column, max_val in zip(df.columns[3:], max_values):
    print(f"Maximum value of column '{column}': {max_val}")
