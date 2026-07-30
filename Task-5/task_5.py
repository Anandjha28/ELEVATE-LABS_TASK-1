import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

df = pd.read_csv("Task-5/Titanic-Dataset.csv")
print(df.head())
print(df.tail())
print(df.describe())
print(df.sample(10))
print("Rows and Columns :", df.shape)
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
df.fillna({
    "Age": df["Age"].median(),
    "Embarked": df["Embarked"].mode()[0]
}, inplace=True)
df.drop(columns="Cabin", inplace=True)

print(df.isnull().sum())
print(df["Survived"].value_counts())
print(df["Sex"].value_counts())
print(df["Pclass"].value_counts())
print(df["Embarked"].value_counts())
print(df.isnull().sum())
df.to_csv("CLEANED_DATASET.csv", index=False)

plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df, palette="Set2")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=df, palette="pastel")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()
sns.pairplot(
    df[["Survived","Pclass","Age","Fare"]],
    hue="Survived"
)
plt.show()

sns.set_style("whitegrid")

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("Titanic Dataset - Exploratory Data Analysis", fontsize=18)

sns.countplot(data=df, x="Pclass", palette="viridis", ax=axes[0,0])
axes[0,0].set_title("Passenger Class")
sns.histplot(data=df, x="Age", bins=30, kde=True, color="skyblue", ax=axes[0,1])
axes[0,1].set_title("Age Distribution")
sns.histplot(data=df, x="Fare", bins=30, kde=True, color="orange", ax=axes[0,2])
axes[0,2].set_title("Fare Distribution")
sns.boxplot(data=df, x="Age", color="lightgreen", ax=axes[1,0])
axes[1,0].set_title("Age Boxplot")
sns.boxplot(data=df, x="Fare", color="salmon", ax=axes[1,1])
axes[1,1].set_title("Fare Boxplot")
sns.countplot(data=df, x="Sex", hue="Survived", palette="Set2", ax=axes[1,2])
axes[1,2].set_title("Survival vs Gender")
sns.countplot(data=df, x="Pclass", hue="Survived", palette="Set1", ax=axes[2,0])
axes[2,0].set_title("Survival vs Passenger Class")
sns.scatterplot(data=df, x="Age", y="Fare", hue="Survived", ax=axes[2,1])
axes[2,1].set_title("Age vs Fare")
sns.countplot(data=df, x="Embarked", hue="Survived", palette="coolwarm", ax=axes[2,2])
axes[2,2].set_title("Embarked vs Survival")

plt.tight_layout(pad=2.5)
plt.subplots_adjust(top=0.93)

plt.savefig("Titanic_EDA_Dashboard.png", dpi=300)
plt.show()
plt.figure(figsize=(8,6))

sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.show()