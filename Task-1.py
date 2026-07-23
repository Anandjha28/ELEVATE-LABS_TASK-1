import pandas as pd
df = pd.read_csv("task2\KaggleV2-May-2016.csv")

print("DataSet:-", df)
print("\n")
print("No. of Rows and Columns:",  df.shape)
print("\n")

print("This are the top 10 Data of the Datasets", df.head(10))
print("This are the Bottom 10 Data of the Datasets", df.tail(10))

print(df.duplicated().any())
# By running this function i have received that there is no any duplicate values ---->output(False) that means there is no duplicate values

print("\n")
print("Brief idea about the data: ")
print(df.describe())

print("\n")
print("This is the overview of the Data:- ")
print(df.info())

print("Count of NULL Data in Dataset:- ")
print(df.isnull().sum())
# in my data there is no any null data if null data exist then i could perform two opertions which are dropna() and fillna(), by using this two functions i can handle the null elements 

df.to_csv("Cleaned_Medical_Appointment_data.csv", index=False)