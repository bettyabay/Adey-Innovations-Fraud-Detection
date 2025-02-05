import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from datetime import datetime

# Handling Missing Values
def handle_missing_values(df, strategy='drop'):
    if strategy == 'drop':
        return df.dropna()  # Drop missing values
    elif strategy == 'mean':
        return df.fillna(df.mean())  # Fill missing values with mean
    elif strategy == 'median':
        return df.fillna(df.median())  # Fill missing values with median
    else:
        return df  # Add other strategies as needed

# Data Cleaning
def remove_duplicates(df):
    return df.drop_duplicates()

def correct_data_types(df):
    if 'signup_time' in df.columns and 'purchase_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        df['ip_address'] = df['ip_address'].astype(int)

    return df

# Exploratory Data Analysis (EDA)
def univariate_analysis(df, column, bins=30):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column], kde=True, bins=bins)
    plt.title(f'Univariate Analysis of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def bivariate_analysis(df, col1, col2):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col1], y=df[col2])
    plt.title(f'Bivariate Analysis of {col1} vs {col2}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()

def correlation_heatmap(df):
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

def pairplot_analysis(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    sns.pairplot(df[columns])
    plt.suptitle('Pair Plot Analysis', y=1.02)
    plt.show()

def scatter_plot(df, col1, col2):
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=df[col1], y=df[col2])
    plt.title(f'Scatter Plot: {col1} vs {col2}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()

# Merge Datasets for Geolocation Analysis
def ip_to_int(ip):
    parts = list(map(int, ip.split('.')))
    return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]

def merge_ip_data(fraud_df, ip_df):
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
    merged_df = fraud_df.merge(ip_df, left_on='ip_int', right_on='lower_bound_ip_address', how='left')
    return merged_df

# Feature Engineering
def transaction_frequency(df):
    df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
    return df

def add_time_features(df):
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.weekday
    df['month'] = df['purchase_time'].dt.month
    df['year'] = df['purchase_time'].dt.year
    return df

def scale_data(df, method='standard'):
    scaler_std = StandardScaler()
    scaler_minmax = MinMaxScaler()
    
    # Fraud_Data.csv: Standard Scaling for purchase_value and age
    if 'purchase_value' in df.columns and 'age' in df.columns:
        df[['purchase_value', 'age']] = scaler_std.fit_transform(df[['purchase_value', 'age']])
    
    # creditcard.csv: Standard Scaling for V1 - V28, Amount | MinMax Scaling for Time
    creditcard_features = [f'V{i}' for i in range(1, 29)] + ['Amount']
    if all(feature in df.columns for feature in creditcard_features):
        df[creditcard_features] = scaler_std.fit_transform(df[creditcard_features])
    if 'Time' in df.columns:
        df[['Time']] = scaler_minmax.fit_transform(df[['Time']])

    return df

# Encode Categorical Features
def encode_categorical(df, columns):
    encoder = LabelEncoder()
    for col in columns:
        df[col] = encoder.fit_transform(df[col])
    return df

# Additional Visualizations
def distribution_plot(df, column):
    plt.figure(figsize=(8, 4))
    sns.distplot(df[column], kde=True)
    plt.title(f'Distribution Plot of {column}')
    plt.show()

def countplot_analysis(df, column):
    plt.figure(figsize=(8, 4))
    sns.countplot(x=df[column])
    plt.title(f'Count Plot of {column}')
    plt.show()

def violin_plot(df, column, category_column):
    plt.figure(figsize=(8, 4))
    sns.violinplot(x=df[category_column], y=df[column])
    plt.title(f'Violin Plot of {column} by {category_column}')
    plt.show()

def heatmap_missing_values(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()




    # Visualizations
    distribution_plot(fraud_df, 'amount')
    countplot_analysis(fraud_df, 'user_id')
    violin_plot(fraud_df, 'amount', 'user_id')
    heatmap_missing_values(fraud_df)
