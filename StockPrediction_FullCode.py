import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose as sd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tabulate import tabulate
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import random
import os
from tensorflow.keras.utils import plot_model
from io import StringIO

def fil(path,dataset):
    df = pd.read_csv(path)
    print(df.head(10))
    if dataset=='AAPL':
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df = df[(df['Date'] >= '2000-01-01') & (df['Date'] <= '2019-12-31')]
    return df

def clean(df):
    columns_to_fill = ['TY','FFR','CPI','UNRATE','USDX','UMCSENT','crude_oil','natural_gas','Close','RSI','MOM','ATR']
    df[columns_to_fill] = df[columns_to_fill].replace('.', None)
    df[columns_to_fill] = df[columns_to_fill].apply(pd.to_numeric, errors='coerce')
    df[columns_to_fill] = df[columns_to_fill].fillna(method='ffill')
    df[columns_to_fill] = df[columns_to_fill].fillna(method='bfill')
    return df


def dist(df, column):
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(y=df[column], color='skyblue', ax=axs[0])
    axs[0].set_title('Box Plot of Stock ' + column + ' Price')
    axs[0].set_ylabel(f'Stock {column} Price')
    axs[0].grid(True)
    sns.kdeplot(x=df[column], shade=True, ax=axs[1], color='green')
    axs[1].set_title('KDE Plot of Stock ' + column + ' Price')
    axs[1].set_xlabel(f'Stock {column} Price')
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()

def ts(df, monthly_avg):
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    axs[0].plot(df['Date'], df['Close'], label='Daily Close Price', color='blue')
    axs[0].set_title('Daily Time Series')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Stock Close Price')
    axs[0].grid(True)
    axs[0].legend()
    axs[1].plot(monthly_avg['Month'], monthly_avg['Close'], label='Monthly Average Close Price', color='orange', linewidth=2)
    axs[1].set_title('Monthly Average Close Price')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Stock Close Price')
    axs[1].grid(True)
    axs[1].legend()
    plt.tight_layout()
    plt.show()

def corr_func(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                linewidths=0.5,
                linecolor='gray',
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                annot_kws={"size": 10},
                vmin=-1, vmax=1)
    plt.title("Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.show()

def boxsubyear(df_before, df_after):
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=False)
    with plt.style.context('fivethirtyeight'):
        sns.boxplot(x=df_before['Year'], y=df_before['Close'], palette='RdBu', ax=axes[0])
        axes[0].set_title('Box Plots Year Wise - Stock Price (Up to 2011)', fontsize=18)
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Close Price')
        sns.boxplot(x=df_after['Year'], y=df_after['Close'], palette='RdBu', ax=axes[1])
        axes[1].set_title('Box Plots Year Wise - Stock Price (After 2011)', fontsize=18)
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Close Price')
    plt.tight_layout()
    plt.show()

def month_year(before, after):
    custom_palette = sns.color_palette("tab20", 12)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 14))
    sns.barplot(x='Year', y='Close', hue='MonthName', data=before,
                palette=custom_palette, ax=axes[0])
    axes[0].set_title("Stock Prices Year & Month Wise (Before 2011)", fontsize=15)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Average Close Price")
    axes[0].legend(loc='upper left', title='Month')
    sns.barplot(x='Year', y='Close', hue='MonthName', data=after,
                palette=custom_palette, ax=axes[1])
    axes[1].set_title("Stock Prices Year & Month Wise (After 2011)", fontsize=15)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Average Close Price")
    axes[1].legend(loc='upper left', title='Month')
    plt.tight_layout()
    plt.show()

def avgstock_plots(df):
    plt.style.use('fivethirtyeight')
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 16))
    group_cols = ['Year', 'Month', 'DayOfWeek', 'Quarter']
    for ax, col in zip(axes, group_cols):
        data_agg = df.groupby(col)['Close'].mean().reset_index()
        data_agg.sort_values('Close', inplace=True)
        sns.barplot(x=col, y='Close', data=data_agg, ax=ax)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Mean Close', fontsize=12)
        ax.set_title(f"Average Stock Price By {col}", fontsize=15)
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

def scaling(df,tar_column):
    x =  df[['TY','FFR','CPI','UNRATE','USDX','UMCSENT','crude_oil','natural_gas','RSI','MOM','ATR']]
    y = df[[tar_column]]

    scaler_X = StandardScaler()
    x_scaled = scaler_X.fit_transform(x)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    df_scaled = pd.DataFrame(x_scaled, columns=x.columns)
    df_scaled[tar_column] = y_scaled
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, shuffle=False)
    print(f"\nTraining set size: {x_train.shape}")
    print(f"Training set size: {y_train.shape}")
    print(f"Test set size: {x_test.shape}")
    print(f"Test set size: {y_test.shape}")
    return x,y,x_scaled,y_scaled,x_train, x_test, y_train, y_test ,df_scaled,scaler_X,scaler_y

def perform_PCA(x_scaled):
    pca = PCA(n_components=0.95) 
    X_pca = pca.fit_transform(x_scaled)
    print(f"Number of components selected: {pca.n_components_}")
    X = pd.DataFrame(x_scaled, columns=[f'feature{i}' for i in range(1, x_scaled.shape[1] + 1)])
    u, s, v = np.linalg.svd(X.values)
    print("\nSingular values of the original feature matrix:")
    print(s)
    cond_d = np.linalg.cond(X.values)
    print(f'Condition number for X is: {cond_d}')
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', label="Cumulative Explained Variance")
    plt.axhline(y=0.95, color='r', linestyle='-', label="95% Variance Threshold")
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Scree Plot of PCA - Explained Variance vs. Components')
    plt.legend()
    plt.grid(True)
    plt.show()

def modelsum(x):
    X1 = sm.add_constant(x)
    model = sm.OLS(y_train,X1).fit()
    print((model.summary()))
    aic = model.aic
    bic = model.bic
    adj_r2 = model.rsquared_adj
    features = list(x.columns)
    return aic, bic, adj_r2 , features

'''def linear_pred(xfin,x,y_train,x_test,scaler_y,selected_features):
    X1 = sm.add_constant(xfin)
    model = sm.OLS(y_train,X1).fit()
    model.summary()
    y_train_pred = model.predict(X1)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    print(f"Linear Regression Train MSE: {mse_train}, Linear Regression Train RMSE: {rmse_train}")
    print('\n')
    selected_features = selected_features
    x_test_df = pd.DataFrame(x_test, columns=x.columns)
    x_test_df = x_test_df[selected_features]  
    x_test_n = sm.add_constant(x_test_df) 
    if 'const' in x_test_n.columns and model.params.shape[0] != x_test_n.shape[1]:
        x_test_n = x_test_n.drop(columns=['const']) 

    y_pred = model.predict(x_test_n)
    y_pred_rescaled = scaler_y.inverse_transform(y_pred.values.reshape(-1, 1))  
    y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))  
    y_train_rescaled = scaler_y.inverse_transform(y_train.reshape(-1, 1))
 
    train_indices = np.arange(len(y_train_rescaled))
    test_indices = np.arange(len(y_train_rescaled), len(y_train_rescaled) + len(y_test_rescaled))

    plt.figure(figsize=(12, 8))
    plt.plot(train_indices, y_train_rescaled, label='Train Data', color='blue')
    plt.plot(test_indices, y_test_rescaled, label='Test Data', color='green')   
    plt.plot(test_indices, y_pred_rescaled, label='Predicted Data', color='red', linestyle='dashed')

    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.title('Train, Test, and Predicted Price')
    plt.legend()
    plt.show()
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    print(f"Linear Regression Test Mean Squared Error (MSE): {mse}")
    print(f"Linear Regression Test Root Mean Squared Error (RMSE): {rmse}")'''

def linear_pred(xfin, x, y_train, x_test, scaler_y, selected_features, log_transform=False):   
    X1 = sm.add_constant(xfin)
    model = sm.OLS(y_train, X1).fit()
    print(model.summary())
    y_train_pred = model.predict(X1)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    print(f"Linear Regression Train MSE: {mse_train} \n")
    print(f"Linear Regression Train RMSE: {rmse_train}")
    x_test_df = pd.DataFrame(x_test, columns=x.columns)
    x_test_df = x_test_df[selected_features]
    x_test_n = sm.add_constant(x_test_df)
    if 'const' in x_test_n.columns and model.params.shape[0] != x_test_n.shape[1]:
        x_test_n = x_test_n.drop(columns=['const'])
    y_pred = model.predict(x_test_n)
    y_pred_rescaled = scaler_y.inverse_transform(y_pred.values.reshape(-1, 1))
    y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_train_rescaled = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    if log_transform:
        y_pred_final = np.exp(y_pred_rescaled)
        y_test_final = np.exp(y_test_rescaled)
        y_train_final = np.exp(y_train_rescaled)
    else:
        y_pred_final = y_pred_rescaled
        y_test_final = y_test_rescaled
        y_train_final = y_train_rescaled

    train_indices = np.arange(len(y_train_final))
    test_indices = np.arange(len(y_train_final), len(y_train_final) + len(y_test_final))
    
    plt.figure(figsize=(12, 8))
    plt.plot(train_indices, y_train_final, label='Train Data', color='blue')
    plt.plot(test_indices, y_test_final, label='Test Data', color='green')
    plt.plot(test_indices, y_pred_final, label='Predicted Data', color='red', linestyle='dashed')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    if log_transform:
        plt.title('Train, Test, and Predicted Price (Log-Transformed OLS Model)')
    else:
        plt.title('Train, Test, and Predicted Price')
    plt.legend()
    plt.show()
    
    mse_test = mean_squared_error(y_test_final, y_pred_final)
    rmse_test = np.sqrt(mse_test)
    mape_test = np.mean(np.abs((y_test_final - y_pred_final) / y_test_final)) * 100
    print(f"Linear Regression Test Mean Squared Error (MSE): {mse_test}")
    print(f"Linear Regression Test Root Mean Squared Error (RMSE): {rmse_test}")
    print(f"Linear Regression Test Mean Absolute Percentage Error (MAPE): {mape_test}%")

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_dict = dict(zip(vif_data["feature"], vif_data["VIF"]))
    return vif_dict

def decomposition(df):
    plt.figure(figsize=(20,16))
    decomposed_series = sd(df['Close'],model='multiplicative',period=252)
    decomposed_series.plot()
    plt.show()
    trend = decomposed_series.trend
    seasonality = decomposed_series.seasonal
    residuals = decomposed_series.resid
    trend_clean = trend.dropna()
    residuals_clean = residuals.dropna()
    trend_strength = 1 - (np.var(residuals_clean) / np.var(trend_clean + residuals_clean))

    seasonality_clean = seasonality.dropna()
    residuals_clean = residuals.dropna()
    seasonality_strength =  1 - (np.var(residuals_clean) / np.var(seasonality_clean + residuals_clean))

    print(f"Trend Strength: {trend_strength:.4f}")
    print(f"Seasonality Strength: {seasonality_strength:.4f}")

'''def hw(df,hw_column,trend='add',seasonal='add'):
    train = df[df['Date'] < '2016-01-01']
    test  = df[df['Date'] >= '2016-01-01']

    seasonal_periods = 252
    model = ExponentialSmoothing(train[hw_column], trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    hw_fit = model.fit()

    y_pred_log = hw_fit.forecast(len(test))
    y_pred = np.exp(y_pred_log)
    y_test_actual = np.exp(test[hw_column])
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    print(f"Holt Winters Mean Squared Error: {mse:.3f}")
    print(f"Holt Winters Root Mean Squared Error: {rmse:.3f}")
    train_indices = np.arange(len(train))
    test_indices  = np.arange(len(train), len(train) + len(test))
    plt.figure(figsize=(12, 8))
    plt.plot(train_indices, np.exp(train[hw_column]), label='Train Data', color='blue')
    plt.plot(test_indices, y_test_actual, label='Test Data', color='green')
    plt.plot(test_indices, y_pred, label='Predicted Data', color='red', linestyle='dashed')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.title('Train, Test, and Predicted Price')
    plt.legend()
    plt.show()'''

def hw(df, hw_column, trend='add', seasonal='add'):
    train = df[df['Date'] < '2016-01-01']
    test  = df[df['Date'] >= '2016-01-01']
    
    seasonal_periods = 252 
    model = ExponentialSmoothing(train[hw_column], trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    hw_fit = model.fit()
    forecast = hw_fit.forecast(steps=len(test))
    if hw_column == 'Close_log':
        y_pred = np.exp(forecast)
        y_test_actual = np.exp(test[hw_column])
        train_plot = np.exp(train[hw_column])
    else:
        y_pred = forecast
        y_test_actual = test[hw_column]
        train_plot = train[hw_column]
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
    print(f"Holt Winters Mean Squared Error: {mse:.3f}")
    print(f"Holt Winters Root Mean Squared Error: {rmse:.3f}")
    print(f"Holt Winters Mean Absolute Percentage Error: {mape:.3f}%")
    train_indices = np.arange(len(train))
    test_indices  = np.arange(len(train), len(train) + len(test))
    plt.figure(figsize=(12, 8))
    plt.plot(train_indices, train_plot, label='Train Data', color='blue')
    plt.plot(test_indices, y_test_actual, label='Test Data', color='green')
    plt.plot(test_indices, y_pred, label='Predicted Data', color='red', linestyle='dashed')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.title('Train, Test, and Predicted Price')
    plt.legend()
    plt.show()

def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()

def diff2(data, order=1, seasonal_period=1, seasonal=False):
    differenced_data = data.copy()

    if seasonal and seasonal_period > 1:
        differenced_data = differenced_data.diff(seasonal_period)

    for _ in range(order):
        differenced_data = differenced_data.diff()

    return differenced_data.dropna()

def generate_gpac_table(autocovariances, j=7, k=7):
    gpac_table = np.zeros((j, k))
    mid_index = len(autocovariances) // 2 

    for row in range(j): 
        for col in range(1, k + 1): 
            denom_matrix = np.zeros((col, col))
            num_matrix = np.zeros((col, col))
            
            for i in range(col):
                for m in range(col):
                    lag_index = row + i - m
                    adjusted_index = mid_index + lag_index
                    if 0 <= adjusted_index < len(autocovariances):
                        denom_matrix[i, m] = autocovariances[adjusted_index]
                    else:
                        denom_matrix[i, m] = 0 
            
            num_matrix[:, :-1] = denom_matrix[:, :-1]
            for i in range(col):
                last_column_lag_index = row + i + 1 
                adjusted_index = mid_index + last_column_lag_index
                if 0 <= adjusted_index < len(autocovariances):
                    num_matrix[i, -1] = autocovariances[adjusted_index] 
                else:
                    num_matrix[i, -1] = 0  
            
            try:
                denom_det = np.linalg.det(denom_matrix)
                if np.abs(denom_det) < 1e-10:
                    gpac_value = np.nan
                else:
                    gpac_value = np.linalg.det(num_matrix) / denom_det
            except np.linalg.LinAlgError:
                gpac_value = np.nan

            gpac_table[row, col - 1] = gpac_value

    plt.figure(figsize=(10, 8))
    sns.heatmap(gpac_table, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
                xticklabels=[str(i) for i in range(1, k + 1)],  # k from 1 to k
                yticklabels=[str(i) for i in range(j)])  # j from 0 to j-1
    plt.title("GPAC Table")
    plt.xlabel("k (order, starting from 1)")
    plt.ylabel("j (starting from 0)")
    plt.show()


def cal_auto(series, lags):
    series_clean = series.dropna() if hasattr(series, 'dropna') else series

    if len(series_clean) < 2:
        raise ValueError("Error: Series is too short for ACF calculation.")

    lags = min(lags, len(series_clean) - 1)
    use_fft = False if len(series_clean) < 50 else True
    ry = acf(series_clean, nlags=lags, fft=use_fft)
    ry_symmetric = np.concatenate((ry[:0:-1], ry))

    return ry_symmetric

def sarimax_forecast(df,p, d, q, P, D, Q, seasonal_period):
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    model = SARIMAX(train['Close'],
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    result = model.fit(disp=False)
    
    forecast = result.get_forecast(steps=len(test))
    conf_int = forecast.conf_int()
    forecast_index = test.index
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Close'], label='Train', color='blue')
    plt.plot(test.index, test['Close'], label='Test', color='green')
    plt.plot(forecast_index, forecast.predicted_mean, label='Forecast', color='red')
    plt.fill_between(forecast_index,
                     conf_int.iloc[:, 0],
                     conf_int.iloc[:, 1],
                     color='pink', alpha=0.3)
    plt.title("SARIMAX Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
    mse = mean_squared_error(test['Close'], forecast.predicted_mean)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test['Close'] - forecast.predicted_mean) / test['Close'])) * 100
    print(f"Sarimax Mean Squared Error (MSE) on original scale: {mse:.4f}")
    print(f"Sarimax Root Mean Squared Error (RMSE) on original scale: {rmse:.4f}") 
    print(f"Sarimax Mean Absolute Percentage Error (MAPE) on original scale: {mape:.4f}%") 

def select_features_list(dataset_name,df):
    dataset_name = dataset_name.upper()
    if dataset_name == "AAPL":
        df = df[['Date','FFR', 'UNRATE', 'crude_oil', 'natural_gas', 'MOM', 'RSI', 'ATR','Close']]
    elif dataset_name == "JPM":
        df = df[['Date','CPI', 'FFR', 'UMCSENT', 'USDX', 'natural_gas', 'MOM', 'ATR','Close']]
    elif dataset_name == "XOM":
        df = df[['Date','TY', 'FFR', 'UMCSENT', 'crude_oil', 'natural_gas', 'ATR', 'RSI','Close']]
    else:
        return []
    return df 

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass
    print(f"Random seeds set to {seed}")
    if tf.config.list_physical_devices('GPU'):
        gpus = tf.config.list_physical_devices('GPU')
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
def prepare_data(df, target_col='Close', sequence_length=60, dataset_type='UNKNOWN'):
    np.random.seed(42)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    dates = df['Date'].values
    df_for_scaling = df.drop('Date', axis=1)
    
    eps = 1e-8  
    if dataset_type == 'JPM':
        print(f"Applying log transformation to {target_col}")
        df_for_scaling[target_col] = np.log(df_for_scaling[target_col] + eps)
    
    feature_columns = df_for_scaling.columns.drop(target_col)
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = feature_scaler.fit_transform(df_for_scaling[feature_columns])
    scaled_target = target_scaler.fit_transform(df_for_scaling[[target_col]])
    scaled_data = np.hstack((scaled_features, scaled_target))
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length, :-1])  
        y.append(scaled_data[i+sequence_length, -1])      
    X, y = np.array(X), np.array(y)
    
    return X, y, feature_scaler, target_scaler, dates, feature_columns, eps


def create_lstm_model(input_shape, lstm_units=128, dropout_rate=0.3, learning_rate=0.001):
    tf.random.set_seed(42) 
    kernel_regularizer = l2(1e-4)
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=input_shape, kernel_regularizer=kernel_regularizer))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=32, activation='relu', kernel_regularizer=kernel_regularizer))
    model.add(Dropout(dropout_rate/2))
    model.add(Dense(units=1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model



def split_data(X, y, validation_split=0.2, test_split=0.1):
    n_samples = len(X)
    test_size = int(n_samples * test_split)
    valid_size = int((n_samples - test_size) * validation_split)
    train_size = n_samples - test_size - valid_size

    train_indices = list(range(0, train_size))
    valid_indices = list(range(train_size, train_size + valid_size))
    test_indices  = list(range(train_size + valid_size, n_samples))
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[valid_indices], y[valid_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, train_indices, valid_indices, test_indices

def hyperparameter_tuning(X_train, y_train, X_val, y_val, input_shape,hyperparams_grid, epochs=100, patience=15):
    best_val_loss = float('inf')
    best_params = None
    best_model = None
    tuning_results = []
    
    for lstm_units in hyperparams_grid['lstm_units']:
        for dropout_rate in hyperparams_grid['dropout_rate']:
            for learning_rate in hyperparams_grid['learning_rate']:
                for batch_size in hyperparams_grid['batch_size']:
                    set_seeds(42)
                    model = create_lstm_model(
                        input_shape=input_shape,
                        lstm_units=lstm_units,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate
                    )
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=patience,
                        restore_best_weights=True,
                        verbose=1
                    )
                    history = model.fit(
                        X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=1
                    )
                    current_val_loss = min(history.history['val_loss'])
                    tuning_results.append({
                        'lstm_units': lstm_units,
                        'dropout_rate': dropout_rate,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'val_loss': current_val_loss,
                        'history': history.history
                    })
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        best_params = {
                            'lstm_units': lstm_units,
                            'dropout_rate': dropout_rate,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size
                        }
                        best_model = model
                    print(f"Tested params: lstm_units={lstm_units}, dropout_rate={dropout_rate}, "
                          f"learning_rate={learning_rate}, batch_size={batch_size} => val_loss={current_val_loss:.6f}")
    
    print(f"\nBest params: {best_params} with val_loss={best_val_loss:.6f}")
    return best_model, best_params, tuning_results


def predict_future_prices(model, last_sequence, feature_scaler, target_scaler, n_steps=30, dataset_type='UNKNOWN', eps=1e-8):
    tf.random.set_seed(42)
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_steps):
        predicted = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
        future_predictions.append(predicted[0, 0])
        last_features = current_sequence[-1, :].copy()
        new_row = np.zeros_like(last_features)
        new_row[:-1] = last_features[:-1]
        new_row[-1] = predicted[0, 0]
        current_sequence = np.vstack([current_sequence[1:], new_row.reshape(1, -1)])
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_prices = target_scaler.inverse_transform(future_predictions)
    
    if dataset_type == 'JPM':
        future_prices = np.exp(future_prices) - eps
    
    return future_prices.flatten()

def visualize_results(df, results, future_dates, future_predictions, target_scaler, dataset_type='UNKNOWN', eps=1e-8, target_col='Close'):
    actual_prices = df[target_col].values
    actual_dates = df['Date'].values
    test_predictions = target_scaler.inverse_transform(results['test_predictions'])
    y_test_original = target_scaler.inverse_transform(results['y_test'].reshape(-1, 1))
    
    if dataset_type == 'JPM':
        test_predictions = np.exp(test_predictions) - eps
        y_test_original = np.exp(y_test_original) - eps
    
    test_indices = results['test_indices']
    sequence_length = results['X_test'].shape[1]
    test_dates = actual_dates[sequence_length + np.array(test_indices)]
    
    plt.figure(figsize=(16, 8))
    plt.plot(actual_dates, actual_prices, label='Actual Prices', color='blue')
    plt.plot(test_dates, test_predictions, label='Test Predictions', color='orange')
    plt.plot(future_dates, future_predictions, label='Future Predictions', color='red')
    plt.title(f'{target_col} Price Prediction', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    
    mse = ((test_predictions.flatten() - y_test_original.flatten())**2).mean()
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_original.flatten() - test_predictions.flatten()) / y_test_original.flatten())) * 100
    text = f'Test MSE: {mse:.4f}\nTest RMSE: {rmse:.4f}\nTest MAPE: {mape:.2f}%'
    plt.figtext(0.15, 0.8, text, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_original, label='Actual Test Values', color='blue')
    plt.plot(test_dates, test_predictions, label='Test Predictions', color='orange')
    plt.plot(future_dates, future_predictions, label='Future Predictions', color='red')
    plt.title(f'{target_col} Price Prediction (Test Period and Future)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.show()


def run_stock_prediction(df, dataset_type, target_col='Close', sequence_length=60, future_days=30,validation_split=0.2, test_split=0.1, epochs=100, patience=15,save_path=None):
    X, y, feature_scaler, target_scaler, dates, feature_columns, eps = prepare_data(
        df, target_col=target_col, sequence_length=sequence_length, dataset_type=dataset_type
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test, train_indices, valid_indices, test_indices = split_data(
        X, y, validation_split, test_split
    )
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    hyperparams_grid = {
        'lstm_units': [128,256],
        'dropout_rate': [0.3,0.4],
        'learning_rate': [0.001,0.0001],
        'batch_size': [32]
    }
    
    print("Starting hyperparameter tuning...")
    best_model, best_params, tuning_results = hyperparameter_tuning(
        X_train, y_train, X_val, y_val,
        input_shape, hyperparams_grid, epochs=epochs, patience=patience
    )
    
    best_run_history = None
    for result in tuning_results:
        if (result['lstm_units'] == best_params['lstm_units'] and
            result['dropout_rate'] == best_params['dropout_rate'] and
            result['learning_rate'] == best_params['learning_rate'] and
            result['batch_size'] == best_params['batch_size']):
            best_run_history = result['history']
            break

    if best_run_history is not None and 'val_loss' in best_run_history:
        plt.figure(figsize=(12, 6))
        plt.plot(best_run_history['loss'], label='Training Loss')
        plt.plot(best_run_history['val_loss'], label='Validation Loss')
        plt.title('Training & Validation Loss (Best Tuning Run)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("Validation loss was not captured during hyperparameter tuning.")

    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    set_seeds(42)
    final_model = create_lstm_model(
        input_shape=input_shape,
        lstm_units=best_params['lstm_units'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate']
    )
    stream = StringIO()
    final_model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_string = stream.getvalue()
    with open("model_summary_aapl.txt", "w") as f:
        f.write(summary_string)
    #plot_model(final_model, to_file='lstm_model_architecture.png', show_shapes=True, show_layer_names=True)
    final_history = final_model.fit(
        X_train_val, y_train_val,
        batch_size=best_params['batch_size'],
        epochs=epochs,
        verbose=1
    )
    
    test_loss = final_model.evaluate(X_test, y_test, verbose=0)
    test_predictions = final_model.predict(X_test, verbose=0)
    
    results = {
        'test_loss': test_loss,
        'test_predictions': test_predictions,
        'y_test': y_test,
        'X_test': X_test,
        'train_indices': train_indices,
        'valid_indices': valid_indices,
        'test_indices': test_indices
    }
    
    if save_path:
        final_model.save(save_path)
        print(f"Model saved to: {save_path}")
    
    last_sequence = X[-1]
    future_predictions = predict_future_prices(
        final_model, last_sequence, feature_scaler, target_scaler,
        n_steps=future_days, dataset_type=dataset_type, eps=eps
    )
    
    last_date = pd.Timestamp(dates[-1])
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
    
    print("Visualizing results...")
    visualize_results(df, {**results, 'X_test': X_test}, future_dates, future_predictions,
                      target_scaler, dataset_type, eps, target_col)
    
    y_test_scaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    test_predictions_scaled = target_scaler.inverse_transform(test_predictions).flatten()
    if dataset_type == 'JPM':
        y_test_original = np.exp(y_test_scaled) - eps
        test_predictions_original = np.exp(test_predictions_scaled) - eps
    else:
        y_test_original = y_test_scaled
        test_predictions_original = test_predictions_scaled
    
    mse = ((y_test_original - test_predictions_original) ** 2).mean()
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_original - test_predictions_original) / y_test_original)) * 100
    
    final_train_mse = final_history.history['loss'][-1]
    final_train_rmse = np.sqrt(final_train_mse)
    print(f"Final Training MSE: {final_train_mse:.4f}")
    print(f"Final Training RMSE: {final_train_rmse:.4f}")
    
    print("\n===== Performance Metrics =====")
    print(f"LSTM Test MSE (original scale): {mse:.4f}")
    print(f"LSTM Test RMSE (original scale): {rmse:.4f}")
    print(f"LSTM Test MAPE: {mape:.2f}%")
    
    plt.figure(figsize=(12, 6))
    plt.plot(final_history.history['loss'], label='Training Loss', color='blue')
    plt.title('Training Loss (Final Model)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()   
    
    return final_model, final_history, results, future_predictions, future_dates

def load_model_and_predict(dataset_name,df):
    dataset_type = dataset_name.strip().upper()
    if dataset_name == 'AAPL':
        weights_path = './model_finalweights/model_weights_aapl.h5'
        df = select_features_list('AAPL',df)
    elif dataset_name == 'JPM':
        weights_path = './model_finalweights/model_weights_jpm.h5'
        df = select_features_list('JPM',df)
    elif dataset_name == 'XOM':
        weights_path = './model_finalweights/model_weights_xom.h5'
        df = select_features_list('XOM',df)
    else:
        print("Invalid dataset name. Please enter AAPL, JPM, or XOM.")
        return
    
    if os.path.exists(weights_path):
        print(f"Model weights found at {weights_path}. Loading model and running predictions...")
        final_model = tf.keras.models.load_model(weights_path)
        X, y, feature_scaler, target_scaler, dates, feature_columns, eps = prepare_data(
            df, target_col='Close', sequence_length=60, dataset_type=dataset_type
        )
        X_train, y_train, X_val, y_val, X_test, y_test, train_indices, valid_indices, test_indices = split_data(
            X, y, validation_split=0.2, test_split=0.1
        )
        test_loss = final_model.evaluate(X_test, y_test, verbose=0)
        test_predictions = final_model.predict(X_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        results = {
            'test_loss': test_loss,
            'test_predictions': test_predictions,
            'y_test': y_test,
            'X_test': X_test,
            'train_indices': train_indices,
            'valid_indices': valid_indices,
            'test_indices': test_indices
        }
        last_sequence = X[-1]
        future_predictions = predict_future_prices(final_model, last_sequence, feature_scaler, target_scaler,
                                                    n_steps=30, dataset_type=dataset_type, eps=eps)
        last_date = pd.Timestamp(dates[-1])
        future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
        
        visualize_results(df, {**results, 'X_test': X_test}, future_dates, future_predictions,
                          target_scaler, dataset_type, eps, target_col='Close')
        y_test_scaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        test_predictions_scaled = target_scaler.inverse_transform(test_predictions).flatten()
        if dataset_type == 'JPM':
            y_test_original = np.exp(y_test_scaled) - eps
            test_predictions_original = np.exp(test_predictions_scaled) - eps
        else:
            y_test_original = y_test_scaled
            test_predictions_original = test_predictions_scaled
        mse = ((y_test_original - test_predictions_original) ** 2).mean()
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test_original - test_predictions_original) / y_test_original)) * 100
        
        print("\nTest Performance Metrics:")
        print(f"Test MSE (original scale): {mse:.4f}")
        print(f"Test RMSE (original scale): {rmse:.4f}")
        print(f"Test MAPE: {mape:.2f}%")
    else:
        print("No saved model weights found. Proceeding with model training...")
        run_stock_prediction(df, dataset_type, target_col='Close', sequence_length=60, future_days=30,
                             validation_split=0.2, test_split=0.1, epochs=100, patience=15, save_path=weights_path)


if __name__ == "__main__":
    dataset_name = input("Enter dataset name (AAPL, JPM, or XOM): ").strip().upper()
    if dataset_name == "AAPL":
        file_path = "/Users/bhoomikan/Documents/Capstone_Project/Datasets/AAPL_Data.csv"
        gpac_j = 27
        p, d, q = 0, 2, 1
        P, D, Q = 0, 1, 1
        seasonal_period = 26
        tar_scale = 'Close'
        trend_hw = 'add'
        seasonal_hw = 'add'
        hw_column = 'Close_log'
    elif dataset_name == "JPM":
        file_path = "/Users/bhoomikan/Documents/Capstone_Project/Datasets/JPM_Dataset.csv"
        tar_scale = 'Close_log'
        trend_hw = 'mul'
        seasonal_hw = 'mul'
        hw_column = 'Close_log'
        gpac_j = 53
        p, d, q = 0, 1, 0
        P, D, Q = 0, 2, 2
        seasonal_period = 26
    elif dataset_name == "XOM":
        file_path = "/Users/bhoomikan/Documents/Capstone_Project/Datasets/XOM_Dataset.csv"
        tar_scale = 'Close'
        trend_hw = 'mul'
        seasonal_hw = 'mul'
        hw_column = 'Close'
        p, d, q = 0, 2, 1
        P, D, Q = 0, 1, 1
        seasonal_period = 26
    else:
        print("Invalid dataset name. Please enter AAPL, JPM, or XOM.")
        exit()
    
    df = fil(file_path,dataset_name)
    print(df.info())
    print(df.head(10))
    df = clean(df)
    print(df.info())
    df.info()
    df['Close_log'] = np.log(df['Close'] + 1e-6)
    dist(df, 'Close')
    dist(df, 'Close_log')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_avg = df.groupby('Month')['Close'].mean().reset_index()
    monthly_avg['Month'] = monthly_avg['Month'].dt.to_timestamp()
    ts(df, monthly_avg)
    df['Year'] = df['Date'].dt.year
    df_before = df[df['Date'] <= '2011-01-01']
    df_after = df[df['Date'] > '2011-01-01']
    boxsubyear(df_before, df_after)
    monthly_data_before = df_before.groupby(['Year', 'Month'], as_index=False)['Close'].mean()
    monthly_data_after = df_after.groupby(['Year', 'Month'], as_index=False)['Close'].mean()
    monthly_data_before['MonthName'] = monthly_data_before['Month'].apply(lambda x: calendar.month_abbr[x.month])
    monthly_data_after['MonthName'] = monthly_data_after['Month'].apply(lambda x: calendar.month_abbr[x.month])
    month_year(monthly_data_before, monthly_data_after)
    df['Month'] = df['Date'].dt.month_name()
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Quarter'] = df['Date'].dt.quarter
    avgstock_plots(df)
    corrdf = df.copy()
    corrdf.drop(['Year','Quarter'],axis=1,inplace=True)
    corr_func(corrdf)
    x,y,x_scaled,y_scaled,x_train, x_test, y_train, y_test,df_scaled,scaler_X,scaler_y = scaling(df,tar_scale)
    perform_PCA(x_scaled)
    x_train_df = pd.DataFrame(x_train, columns=x.columns)
    if dataset_name == "AAPL":
        "Considering all features"
        x1 = x_train_df
        aic1,bic1,adj_r21,features1 = modelsum(x1)
        print(aic1,bic1,adj_r21,features1)
        vif1 = calculate_vif(x1)
        print(vif1)
        x2 = x_train_df.drop(columns=['CPI'])
        aic2, bic2, adj_r22,features2= modelsum(x2)
        print(aic2,bic2,adj_r22,features2)
        vif2 = calculate_vif(x2)
        print(vif2)
        x3 = x2.drop(columns=['TY'])
        aic3, bic3, adj_r23,features3= modelsum(x3)
        print(aic3,bic3,adj_r23,features3)
        vif3 = calculate_vif(x3)
        print(vif3)
        x4 = x3.drop(columns=['UMCSENT'])
        aic4, bic4, adj_r24,features4= modelsum(x4)
        print(aic4,bic4,adj_r24,features4)
        vif4 = calculate_vif(x4)
        print(vif4)
        x5 = x4.drop(columns=['USDX'])
        aic5, bic5, adj_r25,features5= modelsum(x5)
        print(aic5, bic5, adj_r25,features5)
        vif5 = calculate_vif(x5)
        print(vif5)
        x6 = x5.drop(columns=['RSI'])
        aic6, bic6, adj_r26,features6= modelsum(x6)
        print(aic6, bic6, adj_r26,features6)
        vif6 = calculate_vif(x6)
        print(vif6)
        
        summary_data = []

        summary_data.append({
            "Iteration": 1,
            "Description": "Considering all features",
            "Adjusted R²": adj_r21,
            "AIC": aic1,
            "BIC": bic1,
            "Remaining Features": ", ".join(features1)
        })

        summary_data.append({
            "Iteration": 2,
            "Description": f"Dropping CPI with VIF = {vif1['CPI']:.2f}",
            "Adjusted R²": adj_r22,
            "AIC": aic2,
            "BIC": bic2,
            "Remaining Features": ", ".join(features2)
        })

        summary_data.append({
            "Iteration": 3,
            "Description": f"Dropping TY with VIF = {vif2['TY']:.2f}",
            "Adjusted R²": adj_r23,
            "AIC": aic3,
            "BIC": bic3,
            "Remaining Features": ", ".join(features3)
        })

        summary_data.append({
            "Iteration": 4,
            "Description": f"Dropping UMCSENT with VIF = {vif3['UMCSENT']:.2f}",
            "Adjusted R²": adj_r24,
            "AIC": aic4,
            "BIC": bic4,
            "Remaining Features": ", ".join(features4)
        })

        summary_data.append({
            "Iteration": 5,
            "Description": f"Dropping USDX with VIF = {vif4['USDX']:.2f}",
            "Adjusted R²": adj_r25,
            "AIC": aic5,
            "BIC": bic5,
            "Remaining Features": ", ".join(features5)
        })

        summary_data.append({
            "Iteration": 6,
            "Description": f"Dropping RSI with VIF = {vif5['RSI']:.2f}",
            "Adjusted R²": adj_r26,
            "AIC": aic6,
            "BIC": bic6,
            "Remaining Features": ", ".join(features6)
        })

        summary_table = pd.DataFrame(summary_data)
        print(summary_table.to_markdown())
        selected_features = ['FFR','UNRATE', 'crude_oil','natural_gas','MOM','RSI','ATR']
        linear_pred(x6,x,y_train,x_test,scaler_y,x6.columns)
        differenced_data = diff2(df['Close'], order=2,seasonal_period=26,seasonal=True)

    elif dataset_name == "JPM":
        "Considering all features"
        x1 = x_train_df
        aic1,bic1,adj_r21,features1 = modelsum(x1)
        print(aic1,bic1,adj_r21,features1)

        vif1 = calculate_vif(x1)
        print(vif1)

        x2 = x_train_df.drop(columns=['TY'])
        aic2, bic2, adj_r22,features2= modelsum(x2)
        print(aic2,bic2,adj_r22,features2)

        vif2 = calculate_vif(x2)
        print(vif2)

        x3 = x2.drop(columns=['UNRATE'])
        aic3, bic3, adj_r23,features3= modelsum(x3)
        print(aic3,bic3,adj_r23,features3)

        vif3 = calculate_vif(x3)
        print(vif3)

        x4 = x3.drop(columns=['RSI'])
        aic4, bic4, adj_r24,features4= modelsum(x4)
        print(aic4,bic4,adj_r24,features4)

        vif4 = calculate_vif(x4)
        print(vif4)

        x5 = x4.drop(columns=['crude_oil'])
        aic5, bic5, adj_r25,features5= modelsum(x5)
        print(aic5, bic5, adj_r25,features5)

        vif5 = calculate_vif(x5)
        print(vif5)

        summary_data = []

        # Iteration 1: Considering all features
        summary_data.append({
            "Iteration": 1,
            "Description": "Considering all features",
            "Adjusted R²": adj_r21,
            "AIC": aic1,
            "BIC": bic1,
            "Remaining Features": ", ".join(features1)
        })

        # Iteration 2: Dropping CPI
        summary_data.append({
            "Iteration": 2,
            "Description": f"Dropping TY with VIF = {vif1['TY']:.2f}",
            "Adjusted R²": adj_r22,
            "AIC": aic2,
            "BIC": bic2,
            "Remaining Features": ", ".join(features2)
        })

        # Iteration 3: Dropping TY
        summary_data.append({
            "Iteration": 3,
            "Description": f"Dropping UNRATE with VIF = {vif2['UNRATE']:.2f}",
            "Adjusted R²": adj_r23,
            "AIC": aic3,
            "BIC": bic3,
            "Remaining Features": ", ".join(features3)
        })

        # Iteration 4: Dropping UMCSENT
        summary_data.append({
            "Iteration": 4,
            "Description": f"Dropping RSI with VIF = {vif3['RSI']:.2f}",
            "Adjusted R²": adj_r24,
            "AIC": aic4,
            "BIC": bic4,
            "Remaining Features": ", ".join(features4)
        })

        # Iteration 5: Dropping USDX
        summary_data.append({
            "Iteration": 5,
            "Description": f"Dropping crudeoil with VIF = {vif4['crude_oil']:.2f}",
            "Adjusted R²": adj_r25,
            "AIC": aic5,
            "BIC": bic5,
            "Remaining Features": ", ".join(features5)
        })

        summary_table = pd.DataFrame(summary_data)
        print(summary_table.to_markdown())
        selected_features = ['CPI','FFR','UMCSENT','USDX','natural_gas','MOM','ATR']
        linear_pred(x5,x,y_train,x_test,scaler_y,x5.columns,log_transform=True)
        jpm_differenced_data = diff2(df['Close'], order=0,seasonal_period=26,seasonal=True)
        jpm_differenced_data2 = diff2(jpm_differenced_data, order=0,seasonal_period=26,seasonal=True)
        differenced_data = diff2(jpm_differenced_data2, order=1,seasonal_period=0,seasonal=False)
    else:
        "Considering all features"
        x1 = x_train_df
        aic1,bic1,adj_r21,features1 = modelsum(x1)
        print(aic1,bic1,adj_r21,features1)
        vif1 = calculate_vif(x1)
        print(vif1)
        x2 = x_train_df.drop(columns=['CPI'])
        aic2, bic2, adj_r22,features2= modelsum(x2)
        print(aic2,bic2,adj_r22,features2)
        vif2 = calculate_vif(x2)
        print(vif2)
        x3 = x2.drop(columns=['UNRATE'])
        aic3, bic3, adj_r23,features3= modelsum(x3)
        print(aic3,bic3,adj_r23,features3)
        vif3 = calculate_vif(x3)
        print(vif3)
        x4 = x3.drop(columns=['USDX'])
        aic4, bic4, adj_r24,features4= modelsum(x4)
        print(aic4,bic4,adj_r24,features4)
        vif4 = calculate_vif(x4)
        print(vif4)
        x5 = x4.drop(columns=['MOM'])
        aic5, bic5, adj_r25,features5= modelsum(x5)
        print(aic5, bic5, adj_r25,features5)
        vif5 = calculate_vif(x5)
        print(vif5)
        summary_data = []

        # Iteration 1: Considering all features
        summary_data.append({
            "Iteration": 1,
            "Description": "Considering all features",
            "Adjusted R²": adj_r21,
            "AIC": aic1,
            "BIC": bic1,
            "Remaining Features": ", ".join(features1)
        })

        # Iteration 2: Dropping CPI
        summary_data.append({
            "Iteration": 2,
            "Description": f"Dropping CPI with VIF = {vif1['CPI']:.2f}",
            "Adjusted R²": adj_r22,
            "AIC": aic2,
            "BIC": bic2,
            "Remaining Features": ", ".join(features2)
        })

        # Iteration 3: Dropping TY
        summary_data.append({
            "Iteration": 3,
            "Description": f"Dropping UNRATE with VIF = {vif2['UNRATE']:.2f}",
            "Adjusted R²": adj_r23,
            "AIC": aic3,
            "BIC": bic3,
            "Remaining Features": ", ".join(features3)
        })

        # Iteration 4: Dropping UMCSENT
        summary_data.append({
            "Iteration": 4,
            "Description": f"Dropping USDX with VIF = {vif3['USDX']:.2f}",
            "Adjusted R²": adj_r24,
            "AIC": aic4,
            "BIC": bic4,
            "Remaining Features": ", ".join(features4)
        })

        # Iteration 5: Dropping USDX
        summary_data.append({
            "Iteration": 5,
            "Description": f"Dropping MOM with VIF = {vif4['MOM']:.2f}",
            "Adjusted R²": adj_r25,
            "AIC": aic5,
            "BIC": bic5,
            "Remaining Features": ", ".join(features5)
        })

        summary_table = pd.DataFrame(summary_data)
        print(summary_table.to_markdown())
        selected_features = ['TY', 'FFR', 'UMCSENT', 'crude_oil', 'natural_gas', 'ATR', 'RSI', 'Close']
        linear_pred(x5,x,y_train,x_test,scaler_y,x5.columns)
        differenced_data = diff2(df['Close'], order=2,seasonal_period=26,seasonal=True)

    decomposition(df)
    hw(df,hw_column,trend_hw,seasonal_hw)
    ACF_PACF_Plot(df['Close'],100)
    ACF_PACF_Plot(differenced_data,100)
    ry_differenced = cal_auto(differenced_data, lags=30)
    sarimax_forecast(df,p, d, q, P, D, Q, seasonal_period)
    load_model_and_predict(dataset_name,df)



        



