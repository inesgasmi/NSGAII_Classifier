import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch dataset
statlog_german_credit_data = fetch_ucirepo(id=144)

# Extract features and target variable
X = statlog_german_credit_data.data.features
y = statlog_german_credit_data.data.targets

# Manually define column names based on the structure of your data
column_names = ["Feature" + str(i) for i in range(X.shape[1])]
column_names.append("Target")

# Combine features and target variable into a DataFrame
df = pd.DataFrame(data=X, columns=column_names[:-1])  # Assuming the last column is not included in X
df["Target"] = y  # Add the target variable as the last column

# Save DataFrame to a CSV file
csv_file_path = "german_credit_data.csv"
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved successfully at {csv_file_path}")