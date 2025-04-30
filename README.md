# Prerequisites
1. Install all the packages required to run StockPrediction_FullCode.py
2. Replace the  dataset filepaths in the code  to your systems filepaths where datasets are stored.
   
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
   
3. Replace the path of the model weights similarly

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

Finally , run the code using Pycharm/VSCode, which prompts the user to enter the dataset name.
Enter dataset name (AAPL, JPM, or XOM): AAPL
Post this the code generates relevant graphs and model predictions accordingly.
