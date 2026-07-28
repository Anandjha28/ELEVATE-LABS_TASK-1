import pandas as pd 
df = pd.read_csv("Task-4/car_prices.csv/car_prices.csv")
print(df.head(10))
print(df.isnull().sum())
df.replace("â€”", pd.NA, inplace=True)
df.replace("—", pd.NA, inplace=True)
print(df["interior"].unique())
print(df["color"].unique())
print(df.head(20))
print("Duplicate Rows :", df.duplicated().sum())
categorical_cols = [
    "make",
    "model",
    "trim",
    "body",
    "transmission",
    "color",
    "interior",
    "vin"
]

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")
numerical_cols = [
    "condition",
    "odometer",
    "mmr",
    "sellingprice"
]

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())
df["saledate"] = df["saledate"].ffill()
print("\nNull Values After Cleaning")
print(df.isnull().sum())

df.to_csv("cleaned_car_prices.csv", index=False)

print("\nDataset Cleaned Successfully")