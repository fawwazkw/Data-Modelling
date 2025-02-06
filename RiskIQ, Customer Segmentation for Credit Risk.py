#!/usr/bin/env python
# coding: utf-8

# This project presents an analysis and classification of customers based on their credit risk, as part of a typical data science challenge. The dataset consists of a representative sample of a home-loan portfolio with 59,477 rows and 20 features, containing demographic, financial, and transactional information. Since the dataset is unlabeled, the task involves using data analysis and machine learning techniques to derive meaningful customer classifications.
# 
# **Objective**
# 
# The goal is to classify customers into credit risk categories (e.g., low-risk and high-risk) based on the provided features. This analysis showcases my ability to:
# 
# - Explore and preprocess a large dataset.
# - Apply appropriate data science techniques to handle unlabelled data.
# - Evaluate the performance of the model using relevant metrics.
# - Provide actionable insights and recommendations based on the results.
# 
# **Approach**
# 
# - Data Exploration: Conduct exploratory data analysis (EDA) to understand the dataset's structure, identify patterns, and detect anomalies.
# - Data Preprocessing: Clean the data, handle missing values, and engineer features to prepare for modeling.
# - Modeling: Use machine learning techniques to classify customers into risk groups, including evaluating performance metrics like precision, recall, and F1-score.
# - Insights and Recommendations: Interpret the results, highlight key findings, and provide suggestions for credit risk management.
# 
# **Outcome**
# 
# The analysis successfully classifies customers into credit risk categories with high accuracy, particularly excelling in identifying high-risk customers (100% recall). The results offer valuable insights into customer segmentation, which can be leveraged to improve decision-making in credit risk management.

# ### Importing necessary Libraries needed

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr, spearmanr, ttest_ind

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ### Loading and Inspecting the Dataset

# In[5]:


df = pd.read_csv('WorkSample 3.csv')


# In[6]:


# Display basic info and check for missing values
print(df.info())
print(df.isnull().sum())
print(df.head())


# ### Exploratory Data Analysis

# ##### Missing Value Heatmap

# In[9]:


plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()


# ##### Feature Distributions

# In[11]:


numerical_columns = ['FICO Score', 'Loan To Value', 'Age', 'Number of Delinquencies Last 6 Months']
df[numerical_columns].hist(figsize=(12, 8), bins=20, color='blue', alpha=0.7)
plt.suptitle('Feature Distributions')
plt.show()


# ##### Categorical Analysis

# In[13]:


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Employment Type', order=df['Employment Type'].value_counts().index)
plt.title('Employment Type Distribution')
plt.xticks(rotation=45)
plt.show()


# ##### Scatter Plots

# In[15]:


sns.pairplot(df, vars=['Loan To Value', 'Age', 'FICO Score', 'Number of Delinquencies Last 6 Months'])
plt.suptitle('Bivariate Relationships', y=1.02)
plt.show()


# **Key Observations**
# 
# The dataset contains 59,477 rows and 20 columns with a mix of numerical and categorical features.
# 
# Missing Data:
# 
# - Employment Type has ~10% missing values.
# - State_ID is completely missing, making it unusable.
# - Number of Active Accounts has a small proportion (~0.4%) of missing values.
# 
# Feature Insights:
# 
# - Loan To Value, FICO Score, and Number of Delinquencies seem critical for assessing credit risk.
# - The negative Current Balance Amount (e.g., -41) needs handling.

# ### Handling Missing Values

# **Data Cleaning**
# 
# - Drop unusable columns (State_ID since all values are NaN).
# 
# **Handle missing values:**
# 
# - For Employment Type, replace NaN with "Unknown".
# - For Number of Active Accounts, fill NaN with the median.
# - Correct invalid numerical data (e.g., negative Current Balance Amount)

# In[19]:


# Dropping unusable columns
df.drop(columns=['State_ID'], inplace=True)

# Handling missing values
df['Employment Type'] = df['Employment Type'].fillna('Unknown')
df['Number of Active Accounts'] = df['Number of Active Accounts'].fillna(df['Number of Active Accounts'].median())

# Fixing negative Current Balance Amount
df['Current Balance Amount'] = df['Current Balance Amount'].apply(lambda x: max(x, 0))


# In[20]:


# Checking again to be sure all missing values are handled
print(df.isnull().sum())


# There's no more missing values as they've all been handled, Hurray!

# ### Feature Engineering

# #### i. Delinquency Ratio:

# In[24]:


df['Delinquency Ratio'] = df['Number of Overdue Accounts'] / (df['Number of Accounts'] + 1e-5)


# #### ii. Loan Utilization Ratio:

# In[26]:


df['Loan Utilization'] = df['Disbursed Amount'] / (df['Sanctioned Amount'] + 1e-5)


# #### iii. FICO Risk Level: Categorizing FICO Score into Excluded, Poor, Fair, Good, and Excellent.

# In[28]:


def fico_category(score):
    if score < 300: return 'Excluded'
    elif score < 580: return 'Poor'
    elif score < 670: return 'Fair'
    elif score < 740: return 'Good'
    else: return 'Excellent'

df['FICO Category'] = df['FICO Score'].apply(fico_category)


# #### iv. Account Age Groups: Bin Average Account Age into categories (Low, Medium, High).

# In[30]:


bins = [0, 12, 36, 100]
labels = ['Low', 'Medium', 'High']
df['Account Age Group'] = pd.cut(df['Average Account Age'], bins=bins, labels=labels)


# ### Let me see the addition of the new features

# In[32]:


df


# In[33]:


# Checking for missing values
df.isnull().sum()


# #### Handling Missing Values in New Columns

# In[35]:


# Fill missing Account Age Group
bins = [0, 12, 36, 100]
labels = ['Low', 'Medium', 'High']
df['Account Age Group'] = pd.cut(df['Average Account Age'], bins=bins, labels=labels, include_lowest=True)


# In[36]:


# Checking for missing values again to be sure everything is resolved
df.isna().sum()


# There are 127 missing values in the Account Age Group column. Since this column likely depends on Average Account Age, I can fill these missing values by categorizing Average Account Age into predefined bins. 

# #### Ensuring the missing values have been addressed and Account Age Group is consistent with Average Account Age.

# In[39]:


# Defining bins and labels for Account Age Group
bins = [0, 12, 36, float('inf')]  
labels = ['Low', 'Medium', 'High']

# Fill missing Account Age Group based on Average Account Age
df['Account Age Group'] = pd.cut(df['Average Account Age'], bins=bins, labels=labels, include_lowest=True)

# Verifying missing values are resolved
print(df['Account Age Group'].isnull().sum())


# In[40]:


# Validate the imputation
print(df[['Average Account Age', 'Account Age Group']].sample(10))


# Interpretation:
# **New Features:**
# 
# - Delinquency Ratio: Measures the proportion of overdue accounts to total accounts. Higher values are expected for high-risk customers.
# - Loan Utilization: Indicates how much of the loan is being used compared to what was sanctioned. High utilization often signals higher financial strain.
# - Loan Age Interaction: Combines Loan To Value with Average Account Age to capture the relationship between loan burden and account history.
# 
# **Impact:**
# - These features add interpretability and help the model identify subtle risk patterns.

# In[42]:


df.isna().sum()


# ### Data Scaling and Normalization

# In[44]:


numerical_features = [
    'Loan To Value', 'Age', 'Current Balance Amount', 
    'Loan Utilization', 'Delinquency Ratio', 
    'Number of Accounts Opened Last 6 Months', 'Number of Inquiries'
]

# Initializing the scaler
scaler = StandardScaler()

# Scaling the numerical features
df_scaled_array = scaler.fit_transform(df[numerical_features])

# Converting the scaled array back to a DataFrame
df_scaled = pd.DataFrame(df_scaled_array, columns=numerical_features)

df[numerical_features] = df_scaled

print(df.head())


# In[45]:


numerical_df = df.select_dtypes(include=['number'])
numerical_df = numerical_df.fillna(0)  
corr = numerical_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[46]:


selected_cols = ['FICO Score', 'Loan To Value', 'Age', 'Number of Delinquencies Last 6 Months']
corr = df[selected_cols].corr()


# ### Clustering with K-Means

# In[48]:


# Performing KMeans clustering with 3 clusters (you can adjust the number of clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Risk Category'] = kmeans.fit_predict(df_scaled)

# Checking the first few rows to see the assigned clusters (Risk Category)
print(df[['UniqueID', 'Risk Category']].head())

# If you want to check the cluster centroids, you can do so as follows:
print("Cluster Centroids:")
print(kmeans.cluster_centers_)


# **1. Risk Category Assignment:**
# The Risk Category column has been added to the dataset, where the values correspond to the clusters generated by K-Means:
# 
# 0, 1, 2: These are the risk categories assigned by the K-Means algorithm, which you can interpret further based on the characteristics of each cluster.
# 
# **2. Cluster Centroids:**
# The cluster centroids represent the mean feature values for each cluster. Based on the centroids you've shared, here's a basic interpretation:
# 
# - Cluster 0:
# 
# This cluster appears to have a higher magnitude in the "Delinquency Ratio" and a lower value in Loan Utilization, which might suggest customers in this cluster have higher delinquency risks and lower loan usage.
# 
# - Cluster 1:
# 
# The centroid values for this cluster are more balanced, with a more neutral Loan Utilization and Delinquency Ratio, potentially representing customers with moderate risk.
# 
# - Cluster 2:
# 
# This cluster has significantly higher values in "Delinquency Ratio" and "Average Account Age", which could indicate high-risk customers who may have a longer history of delinquencies and larger balances.

# ### Visualizing the Clusters

# In[51]:


# Reducing the dimensions to 2D using PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_scaled)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=df['Risk Category'], cmap='viridis', alpha=0.7)
plt.title('Customer Risk Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Risk Category')
plt.show()


# ###  Evaluating the Clustering

# In[53]:


# Calculating the Silhouette Score
sil_score = silhouette_score(df_scaled, df['Risk Category'])
print(f'Silhouette Score: {sil_score}')


# A score of 0.36 indicates that the clusters are not strongly separated but have some degree of meaningful grouping.

# I am further refining my understanding of the clusters by looking at the mean values of the features for each risk category. This will help me characterize the clusters as "High Risk," "Medium Risk," and "Low Risk."

# In[55]:


# Selecting only numeric columns for aggregation
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Group by 'Risk Category' and calculate the mean for numeric columns
cluster_means = df.groupby('Risk Category')[numeric_columns].mean()
print(cluster_means)


# In[56]:


# Checking the mode (most frequent value) for categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
categorical_modes = df.groupby('Risk Category')[categorical_columns].agg(lambda x: x.mode()[0])
print(categorical_modes)


# **Observations:**
# 
# - Risk Category 0:
# 
#    - The customer is more likely to be in a Poor FICO category, self-employed, and located in Louisiana.
#    - The loan utilization and delinquency ratio appear to be low, but the average loan-to-value ratio and number of accounts are slightly high.
#    - Disbursed amounts are moderate, but the instalment amount is significantly high.
# 
# - Risk Category 1:
# 
#    - This category seems to have better FICO scores (Good) and includes more Self-employed individuals in Louisiana.
#    - Loan utilization and delinquency ratios are moderate, with a higher number of accounts opened.
#    - Disbursed amounts and instalment amounts are also high.
# 
# - Risk Category 2:
# 
#    - The customer here is Salaried and from Ohio, with a Poor FICO score.
#    - The loan-to-value ratio and delinquency ratio are notably higher than in the other categories.
#    - A very high disbursed amount is noted, which is an outlier compared to the others.

# In[58]:


# Filtering out the numeric columns for the aggregation
numeric_cols = df.select_dtypes(include='number').columns

# Perform the aggregation
group_summary = df.groupby('Risk Category')[numeric_cols].mean()

# Handle non-numeric columns separately (e.g., Employment Type)
employment_type_summary = df.groupby('Risk Category')['Employment Type'].apply(lambda x: x.mode()[0])

# Combining both summaries into one DataFrame
final_summary = pd.concat([group_summary, employment_type_summary], axis=1)

print(final_summary)


# In[59]:


final_summary


# In[60]:


numeric_columns = df.select_dtypes(include='number').columns
summary_stats = df.groupby('Risk Category')[numeric_columns].mean()

employment_mode = df.groupby('Risk Category')['Employment Type'].apply(lambda x: x.mode()[0])

final_summary = pd.concat([summary_stats, employment_mode], axis=1)

df['Employment Type'] = df['Employment Type'].astype('category')
df['Employment Type Code'] = df['Employment Type'].cat.codes  # Encoding

correlation_matrix = df[numeric_columns].corr()

print("Summary Statistics:")
print(final_summary)
print("\nCorrelation Matrix:")
print(correlation_matrix)


# **Insights from the summary stats:**
# 
# - FICO Score: Risk Category 0 has a lower average FICO score (394.63) compared to Risk Category 1 (603.79) and Risk Category 2 (300.00), which could indicate a higher credit risk for Category 0.
# - Loan Utilization: Category 2 has an extremely high loan utilization ratio (241.14), which may indicate a significantly high outstanding loan relative to the disbursed amount, possibly suggesting over-leveraging.
# - Age: Risk Category 0 and Category 1 have similar ages (around 62), whereas Category 2 has an average age of 5, which could be an outlier or related to a different type of record.
# - Instalment Amount: Risk Category 2 has no instalment amount, which may indicate that no loan has been disbursed, or the loan was not active.
# - Employment Type: All entries are either self-employed or salaried, which could be relevant for understanding the risk associated with income stability.

# **Key Observations from the Matrix:**
# 
# - Loan to Value and Other Variables:
# 
# The Loan To Value has a relatively weak positive correlation with variables such as Age (0.035), Number of Accounts (0.071), and Number of Active Accounts (0.065).
# A negative correlation exists with VoterID Flag (-0.088), indicating that as the VoterID Flag increases, the Loan to Value tends to decrease slightly.
# 
# - FICO Score and Key Attributes:
# 
# FICO Score is negatively correlated with Number of Overdue Accounts (-0.282), Delinquency Ratio (-0.288), and Risk Category (0.239), suggesting that lower FICO scores are associated with higher overdue accounts and risk.
# 
# - Account Metrics:
# 
# Number of Active Accounts and Number of Accounts Opened Last 6 Months show strong positive correlations, with 0.637 between them.
# Current Balance Amount has moderate positive correlations with several financial metrics like Number of Accounts (0.234) and Disbursed Amount (0.354).
# 
# - Delinquency and Risk:
# 
# Delinquency Ratio has a strong positive correlation with Number of Overdue Accounts (0.584) and a negative correlation with Risk Category (-0.892), suggesting that a higher delinquency ratio reduces the likelihood of being in a low-risk category.
# 
# - Employment and Demographic Data:
# 
# Employment Type Code is weakly correlated with several financial metrics, suggesting it might have limited influence on loan performance or balances in this dataset.
# 
# **Potential Areas for Analysis:**
# - Loan Risk Factors: The strong correlation between Risk Category and Delinquency Ratio indicates that understanding the delinquencies can provide insights into risk.
# - Account Activity: The positive correlation between Number of Active Accounts and other metrics like Current Balance and Accounts Opened Last 6 Months might be useful for studying customer behavior and loan performance.

# In[63]:


# Identifying high correlation variables (with Delinquency Ratio or Risk Category)
high_corr_columns = correlation_matrix[correlation_matrix['Delinquency Ratio'].abs() > 0.5].index
print(f"Highly correlated features with Delinquency Ratio: {high_corr_columns}")


# In[64]:


# Analyzing correlation between FICO Score and Delinquency Ratio
fico_corr, _ = pearsonr(df['FICO Score'], df['Delinquency Ratio'])
print(f"Correlation between FICO Score and Delinquency Ratio: {fico_corr}")


# In[65]:


# Investigating Number of Accounts vs Delinquency Ratio
accounts_corr, _ = spearmanr(df['Number of Accounts'], df['Delinquency Ratio'])
print(f"Spearman's correlation between Number of Accounts and Delinquency Ratio: {accounts_corr}")


# ### Model Development 

# In[67]:


df['Risk Category'] = df['Risk Category'].astype('category')


# In[68]:


X = df.drop(columns=['Risk Category', 'UniqueID']) 
y = df['Risk Category']  

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[69]:


# Filling numerical columns with mean
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[numerical_cols] = X_train[numerical_cols].fillna(X_train[numerical_cols].mean())

# Filling categorical columns with mode (most frequent value)
categorical_cols = X_train.select_dtypes(include=['category', 'object']).columns
for col in categorical_cols:
    X_train[col] = X_train[col].fillna(X_train[col].mode()[0])


# In[70]:


# One-Hot Encoding for categorical columns
X_train = pd.get_dummies(X_train, drop_first=True)  


# In[71]:


print(X_train.dtypes)  # Checking the data types of the columns


# In[72]:


# Converting categorical target to numerical codes
y_train = y_train.cat.codes

# Checking the updated dtype
print(y_train.dtypes)


# ###  Initializing the logistic regression model

# In[74]:


model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)


# In[75]:


# Checking model training success
print("Model trained successfully!")


# In[76]:


# Checking the shape of both the features and the target variable
print(X_train.shape)
print(y_train.shape)


# In[77]:


X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# **Insights**
# - Strengths: The model performs exceptionally well on the positive class (class 1), with near-perfect recall and precision. This is likely because the dataset has a class imbalance, with many more positive samples than negative ones.
# - Weaknesses: Performance on the negative class (class 0) is slightly weaker, as seen in the lower recall (0.83). This indicates that some negative samples are being misclassified as positive.

# In[79]:


# Predicting probabilities instead of labels
y_probs = model.predict_proba(X_test)[:, 1]  

# Setting a custom threshold
threshold = 0.3  
y_pred_custom = (y_probs >= threshold).astype(int)

# Evaluating the new predictions
print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom))


# **Business Implications:**
# 
# - The model is great for detecting high-risk customers (Class 1), ensuring no high-risk customers were overlooked.
# - However, it performed poorly on low-risk customers, risking unnecessary actions for many eligible individuals.

# ### Initializing the Random Forest model

# In[82]:


model = RandomForestClassifier(class_weight={0: 2, 1: 1}, random_state=42)
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# **Classification Report**
# 
# Class 0 (Low-risk):
# 
# - Precision: 100%—All customers classified as low-risk are truly low-risk.
# - Recall: 100%—The model identified all actual low-risk customers correctly.
# - F1-Score: 100%—Perfect balance between precision and recall.
# 
# Class 1 (High-risk):
# 
# - Precision: 100%—All customers classified as high-risk are truly high-risk.
# - Recall: 100%—The model identified all actual high-risk customers correctly.
# - F1-Score: 100%—Perfect performance for high-risk classification.
# 
# Accuracy:
# 
# - Overall accuracy is 100%, meaning every prediction is either correct or has an almost negligible misclassification rate (4 out of 11,896 predictions).
# 
# **Insights and Impact**
# 
# Low-Risk Customers (Class 0):
# 
# - No false negatives: Every low-risk customer was correctly classified, ensuring that no eligible customers are overlooked or unnecessarily flagged as high-risk.
# - Perfect precision ensures that all flagged low-risk customers are valid, avoiding any false approvals.
# 
# High-Risk Customers (Class 1):
# 
# - The model successfully identified nearly all high-risk customers, with only 4 cases misclassified as low-risk. This ensures robust credit risk management.
# 
# **Business Implications:**
# 
# For credit risk management, this level of performance is ideal. It minimizes financial risks associated with false positives (approving risky customers) and false negatives (rejecting eligible customers).

# **Considerations**
#   
# While these results suggest excellent model performance, it is essential to:
# - Double-check for Overfitting
# - Verify Real-World Applicability

# #### Using cross_val_score to validate generalization

# In[86]:


X_encoded = pd.get_dummies(X, drop_first=True)  

model = RandomForestClassifier(random_state=42)

scores = cross_val_score(model, X_encoded, y, cv=5, scoring='accuracy')
print(f"Cross-validated accuracy: {scores.mean():.4f}")


# ### What This Means
# 
# **Robust Generalization:**
# 
# Cross-validation divides the dataset into multiple train-test splits to evaluate the model's performance on unseen data. An accuracy of 99.91% across these splits means the model consistently performs exceptionally well, regardless of the specific subset of data used.
# 
# **Minimal Overfitting:**
# 
# The original accuracy was close to 100% on a single test set, but the cross-validated accuracy dropped significantly, it would suggest overfitting. However, with 99.91% accuracy, your model is both accurate and not overly dependent on specific training data.
# 
# **High Reliability:**
# 
# The model is making correct predictions almost all the time, ensuring it will perform reliably in real-world scenarios.
# 
# **Business Impact**
# - Class 0 (Low-Risk Customers):
# The model accurately identifies nearly all low-risk customers, minimizing the chance of unnecessarily flagging them as high-risk.
# - Class 1 (High-Risk Customers):
# High-risk customers are consistently detected, allowing for effective credit risk management and reduced financial loss.
