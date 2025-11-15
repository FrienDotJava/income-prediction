import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    return joblib.load("pipeline_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/interim/cleaned_data.csv")

st.header("🔍 Interactive Dataset Explorer")

df = load_data()

# ---- Sidebar Filters ----
st.sidebar.header("Filter the Dataset")

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["sex"].unique(),
    default=df["sex"].unique()
)

race_filter = st.sidebar.multiselect(
    "Race",
    options=df["race"].unique(),
    default=df["race"].unique()
)

workclass_filter = st.sidebar.multiselect(
    "Workclass",
    options=df["workclass"].dropna().unique(),
    default=df["workclass"].dropna().unique()
)

education_filter = st.sidebar.multiselect(
    "Education",
    options=df["education"].unique(),
    default=df["education"].unique()
)

# ---- Apply Filters ----
df_filtered = df[
    (df["sex"].isin(gender_filter)) &
    (df["race"].isin(race_filter)) &
    (df["workclass"].isin(workclass_filter)) &
    (df["education"].isin(education_filter))
]

st.write(f"### Filtered Dataset — {len(df_filtered)} rows")
st.dataframe(df_filtered)

st.markdown("---")


# DATA ANALYSIS SECTION
st.header("📊 Exploratory Data Analysis (Adult Census Income)")

# df = load_data()

categorical_color = "#4C72B0"

# 1. Workclass Distribution
st.subheader("Distribution of Workclass")

workclass_count = df_filtered['workclass'].value_counts()
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(workclass_count.index, workclass_count, color=categorical_color)

for index, value in enumerate(workclass_count):
    ax.text(index, value, str(value), ha='center', va='bottom')

ax.set_title("Distribution of Workclass")
plt.xticks(rotation=45)
st.pyplot(fig)


# 2. Occupation Distribution
st.subheader("Distribution of Occupation")

occupation_count = df_filtered['occupation'].value_counts()
fig, ax = plt.subplots(figsize=(12, 4))
ax.barh(occupation_count.index, occupation_count, color=categorical_color)

for index, value in enumerate(occupation_count):
    ax.text(value, index, str(value), ha='left', va='center')

ax.set_title("Distribution of Occupation")
st.pyplot(fig)


# 3. Education Distribution
st.subheader("Distribution of Education")

edu_labels = [
    'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th',
    'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors',
    'Masters', 'Prof-school', 'Doctorate'
]

df_filtered['education'] = pd.Categorical(df_filtered['education'], categories=edu_labels, ordered=True)
edu_count = df_filtered['education'].value_counts(sort=False)

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(edu_count.index, edu_count, color=categorical_color)

for index, value in enumerate(edu_count):
    ax.text(value, index, str(value), ha='left', va='center')

ax.set_title("Distribution of Education")
st.pyplot(fig)


# 4. Marital Status Distribution
st.subheader("Distribution of Marital Status")

marital_count = df_filtered['marital.status'].value_counts(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(marital_count.index, marital_count, color=categorical_color)

for index, value in enumerate(marital_count):
    ax.text(value, index, str(value), ha='left', va='center')

ax.set_title("Distribution of Marital Status")
st.pyplot(fig)


# 5. Race Distribution
st.subheader("Distribution of Race")

race_count = df_filtered['race'].value_counts()

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(race_count.index, race_count, color=categorical_color)

for index, value in enumerate(race_count):
    ax.text(index, value, str(value), ha='center', va='bottom')

ax.set_title("Distribution of Race")
st.pyplot(fig)


# 6. Gender Distribution
st.subheader("Distribution of Gender")

sex_count = df_filtered['sex'].value_counts()

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(sex_count.index, sex_count, color=categorical_color)

for index, value in enumerate(sex_count):
    ax.text(index, value, str(value), ha='center', va='bottom')

ax.set_title("Distribution of Gender")
st.pyplot(fig)

st.markdown("---")

# 7. Income vs Features
st.header("💰 Income vs Feature Comparison")

data = df_filtered 
high_income_label = ">50K"

# ---------- Income vs Education ----------
st.subheader("Income Proportion by Education")

edu_income = (
    data.groupby("education")["income"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
)

if high_income_label in edu_income.columns:
    high_income_rate_edu = edu_income[high_income_label] * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    high_income_rate_edu.sort_values().plot(kind="barh", ax=ax, color=categorical_color)

    ax.set_xlabel("Percentage of >50K (%)")
    ax.set_ylabel("Education")
    ax.set_title("Percentage of Individuals Earning >50K by Education Level")
    st.pyplot(fig)
else:
    st.warning(f"Column '{high_income_label}' not found in income values. Check your income labels.")


# ---------- Income vs Gender ----------
st.subheader("Income Proportion by Gender")

sex_income = (
    data.groupby("sex")["income"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
)

if high_income_label in sex_income.columns:
    high_income_rate_sex = sex_income[high_income_label] * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    high_income_rate_sex.plot(kind="bar", ax=ax, color=categorical_color)

    ax.set_ylabel("Percentage of >50K (%)")
    ax.set_xlabel("Gender")
    ax.set_title("Percentage of Individuals Earning >50K by Gender")
    st.pyplot(fig)
else:
    st.warning(f"Column '{high_income_label}' not found in income values. Check your income labels.")


# ---------- Income vs Marital Status ----------
st.subheader("Income Proportion by Marital Status")

marital_income = (
    data.groupby("marital.status")["income"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
)

if high_income_label in marital_income.columns:
    high_income_rate_marital = marital_income[high_income_label] * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    high_income_rate_marital.sort_values().plot(kind="barh", ax=ax, color=categorical_color)

    ax.set_xlabel("Percentage of >50K (%)")
    ax.set_ylabel("Marital Status")
    ax.set_title("Percentage of Individuals Earning >50K by Marital Status")
    st.pyplot(fig)
else:
    st.warning(f"Column '{high_income_label}' not found in income values. Check your income labels.")

st.markdown("---")

# PREDICTION SECTION

model = load_model()

st.title('Income Prediction')

age = st.number_input('Age')

education_to_num = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8,
    "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16
}

education = st.selectbox("Education", list(education_to_num.keys()))
education_num = education_to_num[education]

occupation = st.selectbox("Occupation", [
"Tech-support","Craft-repair","Other-service","Sales","Exec-managerial", "Prof-specialty",
"Handlers-cleaners","Machine-op-inspct","Adm-clerical", "Farming-fishing",
"Transport-moving","Priv-house-serv", "Protective-serv","Armed-Forces"])

hours_per_week = st.number_input('Working hours per week')

marital_status = st.selectbox('Marital Status', [
    'Widowed', 'Divorced', 'Separated', 'Never-married',
    'Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'
])

row = {
    "age": age,
    "education.num": education_num,
    "marital.status": marital_status,
    "occupation": occupation,
    "capital.gain": 0,
    "capital.loss": 0,
    "hours.per.week": hours_per_week
}

X_new = pd.DataFrame([row])
st.write(X_new)

if st.button("Predict"):
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0, 1]
    label = ">50K" if pred == 1 else "<=50K"
    st.success(f"Prediction: {label} — Probability: {proba:.3f}")
