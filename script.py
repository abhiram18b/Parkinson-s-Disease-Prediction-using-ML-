# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# %%
df=pd.read_csv('parkinsons.csv')

# %%
df.head()

# %%
df.tail()

# %%
df.info()

# %%
df.describe()

# %%
null_counts=df.isnull().sum()
print(null_counts)

# %%
df.duplicated().sum()

# %%
"""
# Before Capping outilers
"""

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Create subplots
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

# Flatten axes to iterate over them
axes = axes.flatten()

# Iterate over columns
for i, col in enumerate(df.columns):
    if col != 'name' and col != 'status' and i < 25:  # Ensure only the first 25 columns are plotted
        # Plot the distribution
        sns.distplot(df[col], ax=axes[i])
        
        # Calculate statistics
        peak_value = df[col].max()
        mean_value = df[col].mean()
        median_value = df[col].median()
        
        # Annotate the plot with statistics
        axes[i].text(peak_value, 0.03, f'Peak: {peak_value:.2f}', verticalalignment='bottom', horizontalalignment='right')
        axes[i].axvline(mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')
        axes[i].axvline(median_value, color='g', linestyle='-.', label=f'Median: {median_value:.2f}')
        
        # Set plot title and legend
        axes[i].set_title(f'Distribution of {col}')
        axes[i].legend()

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# %%
for col in df.columns:
    if col!='name' and col!='status':
        print(col,":",df[col].skew())

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame
fig, axes = plt.subplots(5, 5, figsize=(15, 15))  # Adjust the number of rows and columns as needed
axes = axes.flatten()

for i, col in enumerate(df.columns):  # Iterate using enumerate
    if col != 'name' and col != 'status':
        sns.boxplot(y=df[col], ax=axes[i])

plt.tight_layout()
plt.show()


# %%
"""
# Function to Cap Outliers
"""

# %%
import numpy as np

def cap_outliers(df, factor=1.5):
    capped_df = df.copy()
    for col in df.columns:
        if col != 'status':  # Assuming 'status' is your target variable and should not be capped
            # Calculate IQR and lower/upper bounds
            Q1 = np.quantile(df[col], 0.25)
            Q3 = np.quantile(df[col], 0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            # Cap outliers
            capped_df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound),
                                      np.clip(df[col], lower_bound, upper_bound),
                                      df[col])
            
    return capped_df


# %%
df_capped=cap_outliers(df.drop('name',axis=1), factor=1.5)

# %%
df_capped

# %%
"""
# After Capping
"""

# %%
fig, axes = plt.subplots(5, 5, figsize=(15, 15))  # Adjust the number of rows and columns as needed
axes = axes.flatten()

for i, col in enumerate(df_capped.columns):  # Iterate using enumerate
    if col != 'status':
        sns.boxplot(y=df_capped[col], ax=axes[i])

plt.tight_layout()
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Create subplots
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

# Flatten axes to iterate over them
axes = axes.flatten()

# Iterate over columns
for i, col in enumerate(df_capped.columns):
    if col != 'status' and i < 25:  # Ensure only the first 25 columns are plotted
        # Plot the distribution
        sns.distplot(df_capped[col], ax=axes[i])
        
        # Calculate statistics
        peak_value = df_capped[col].max()
        mean_value = df_capped[col].mean()
        median_value = df_capped[col].median()
        
        # Annotate the plot with statistics
        axes[i].text(peak_value, 0.03, f'Peak: {peak_value:.2f}', verticalalignment='bottom', horizontalalignment='right')
        axes[i].axvline(mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')
        axes[i].axvline(median_value, color='g', linestyle='-.', label=f'Median: {median_value:.2f}')
        
        # Set plot title and legend
        axes[i].set_title(f'Distribution of {col}')
        axes[i].legend()

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# %%
for col in df_capped.columns:
    if col!='status':
        print(col,":",df_capped[col].skew())

# %%
"""
# To Find Coorelation
"""

# %%
correlation_matrix = df_capped.drop(['status'], axis=1).corr()

# %%
correlation_matrix

# %%
plt.figure(figsize=(15, 15))

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidth=0.5)
plt.title("Correlation Matrix HeatMap")
plt.show()

# %%
correlation_treshold=0.6
columns_to_drop = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] >= correlation_treshold:
            colname = correlation_matrix.columns[j]
            columns_to_drop.add(colname)

# %%
print(columns_to_drop)
print("No.of colums to drop are:",len(columns_to_drop))

# %%
df_filtered=df_capped.drop(columns=columns_to_drop)

# %%
df_filtered

# %%
df_filtered.shape

# %%
correaltion_matrix=df_filtered.drop(['status'],axis=1).corr()

# %%
plt.figure(figsize=(10,10))

sns.heatmap(correaltion_matrix,annot=True,fmt='.2f',linewidth=.5)
plt.show()

# %%
features=df_filtered.drop(['status'],axis=1)
labels=df_filtered['status']

# %%
features

# %%
labels

# %%
scaler=MinMaxScaler(feature_range=(-1,1))

scaler.fit(features)

X=scaler.transform(features)

X.shape

# %%
X

# %%
y=labels.values

# %%
y


# %%
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=7)

# %%
print("X_train shape:",X_train.shape)
print("y_train shape:", y_train.shape)

# %%
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# %%
from imblearn.over_sampling import SMOTE
smote = SMOTE()

# Generate synthetic samples
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


print("X_train_resampled shape:", X_train_resampled.shape)
print("y_train_resampled shape:", y_train_resampled.shape)

# %%
"""
# XGBClassifier
"""

# %%
xgb_model=XGBClassifier()

xgb_model.fit(X_train_resampled,y_train_resampled)

# %%
xgb_train_predictions = xgb_model.predict(X_train_resampled)
xgb_test_predictions = xgb_model.predict(X_test)

# %%
xgb_train_accuracy = accuracy_score(y_train_resampled, xgb_train_predictions)
xgb_test_accuracy = accuracy_score(y_test, xgb_test_predictions)
xgb_precision = precision_score(y_test, xgb_test_predictions)
xgb_recall = recall_score(y_test, xgb_test_predictions)
xgb_conf_matrix = confusion_matrix(y_test, xgb_test_predictions)

# %%
print("\nXGBoost Classifier:")
print("Training Accuracy:", xgb_train_accuracy)
print("Test Accuracy:", xgb_test_accuracy)
print("Precision:", xgb_precision)
print("Recall:", xgb_recall)
print("Confusion Matrix:")
print(xgb_conf_matrix)

# %%
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(xgb_conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# %%
"""
# RandomForest Classifier
"""

# %%
rf_model=RandomForestClassifier()
rf_model.fit(X_train_resampled,y_train_resampled)

# %%
rf_train_predictions = rf_model.predict(X_train_resampled)
rf_test_predictions = rf_model.predict(X_test)

# %%
rf_train_accuracy = accuracy_score(y_train_resampled, rf_train_predictions)
rf_test_accuracy = accuracy_score(y_test, rf_test_predictions)
rf_precision = precision_score(y_test, rf_test_predictions)
rf_recall = recall_score(y_test, rf_test_predictions)
rf_conf_matrix = confusion_matrix(y_test, rf_test_predictions)

# %%
print("Random Forest Classifier:")
print("Training Accuracy:", rf_train_accuracy)
print("Test Accuracy:", rf_test_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("Confusion Matrix:")
print(rf_conf_matrix)


# %%
plt.figure(figsize=(8, 6))
sns.heatmap(rf_conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# %%
"""
# Logistic Regression
"""

# %%
logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train_resampled,y_train_resampled)

# %%
logistic_train_predictions = logistic_classifier.predict(X_train_resampled)
logistic_test_predictions = logistic_classifier.predict(X_test)

# %%
# Metrics
logistic_train_accuracy = accuracy_score(y_train_resampled, logistic_train_predictions)
logistic_test_accuracy = accuracy_score(y_test, logistic_test_predictions)
logistic_precision = precision_score(y_test, logistic_test_predictions)
logistic_recall = recall_score(y_test, logistic_test_predictions)
logistic_conf_matrix = confusion_matrix(y_test, logistic_test_predictions)


# %%
# Print Metrics
print("Logistic Regression Classifier:")
print("Training Accuracy:", logistic_train_accuracy)
print("Test Accuracy:", logistic_test_accuracy)
print("Precision:", logistic_precision)
print("Recall:", logistic_recall)
print("Confusion Matrix:")
print(logistic_conf_matrix)

# %%
plt.figure(figsize=(8, 6))
sns.heatmap(logistic_conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# %%
import numpy as np

def cap_row_with_df_iqr(row, df, factor=1.5):
    capped_row = row.copy()
    for i, val in enumerate(row):
        col = df.columns[i]  # Assuming the order of columns in df matches the order in row
        if col != 'status':  # Assuming 'status' is your target variable and should not be capped
            # Calculate IQR from the entire DataFrame
            Q1 = np.quantile(df[col], 0.25)
            Q3 = np.quantile(df[col], 0.75)
            col_iqr = Q3 - Q1
            
            # Calculate lower and upper bounds based on IQR and factor
            lower_bound = Q1 - factor * col_iqr
            upper_bound = Q3 + factor * col_iqr

            # Cap the value if it's outside the bounds
            if val < lower_bound:
                capped_row[i] = lower_bound
            elif val > upper_bound:
                capped_row[i] = upper_bound
    return capped_row


# %%
input_row=np.array([119.992,74.997,0.00784,0.00007,0.00554,0.04374,0.4266,0.2182,0.0313,0.02971,0.6545,0.02211,21.033,0.414783,0.815285,2.301442])

# %%
capped_input_row = cap_row_with_df_iqr(input_row,df.drop('name',axis=1),1.5)

# %%
capped_input_row

# %%
sacled_input=scaler.transform(np.array(capped_input_row,ndmin=2))

# %%
rf_model.predict(sacled_input)

# %%
input_row2=np.array([197.076,192.055,0.00289,0.00001,0.00168,0.01098,0.097,0.00563,0.0068,0.00802,0.01689,0.00339,26.775,0.422229,0.741367,1.743867])

# %%
capped_input_row2= cap_row_with_df_iqr(input_row2,df.drop('name',axis=1),1.5)

# %%
capped_input_row2

# %%
sacled_input2=scaler.transform(np.array(capped_input_row2,ndmin=2))

# %%
rf_model.predict(sacled_input2)

# %%
