import pandas as pd

# Load your dataset
df = pd.read_csv("composite_data.csv", encoding='latin1')  # Replace with your actual filename

df = df.drop(columns=['Unnamed: 5'], errors='ignore')
df.columns = (
    df.columns.str.strip()
               .str.replace(' ', '_')
               .str.replace('°', '', regex=False)
               .str.replace('(', '', regex=False)
               .str.replace(')', '', regex=False)
               .str.replace('·', '', regex=False)
               .str.replace('/', '_')
)

# Check the first few rows
print(df.to_string())
print(df.columns.tolist())