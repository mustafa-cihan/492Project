import numpy as np
import pandas as pd
import sklearn as sk
import xgboost as xgb
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn import metrics 
import torch.optim as optim
from torch import nn, Tensor
import torch
import warnings

warnings.filterwarnings("ignore")

class Transformer(nn.Module):
    def __init__(self, feature_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        src = self.transformer_encoder(src)
        output = self.decoder(src[-1, :, :])  # only take the output from the last time step
        return output

def create_inout_sequences(input_data: np.array, tw: int):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

def split_dataset_by_date(dataset, split_date):

    dataset.loc[:,'first_day_of_month'] = pd.to_datetime(dataset.loc[:,'first_day_of_month'], format='%Y-%m-%d').copy()
    split_date = pd.to_datetime(split_date, format='%Y-%m-%d')
    train = dataset.loc[dataset['first_day_of_month'] <= split_date].copy()
    test = dataset.loc[dataset['first_day_of_month'] > split_date].copy()
    return train, test

def smape(y_true, y_pred):
    # CONVERT TO NUMPY
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # WHEN BOTH EQUAL ZERO, METRIC IS ZERO
    both = np.abs(y_true) + np.abs(y_pred)
    idx = np.where(both==0)[0]
    y_true[idx]=1; y_pred[idx]=1

    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def total_score(data):
    result = pd.DataFrame(columns=['first_day_of_month','mae', 'smape'])
    
    firstDay = data.first_day_of_month.unique()
    for first in firstDay:
        maeVals = []
        smapeVals = []
        ids = data.cfips.unique()
        for id in ids:
            id_data = data[data.cfips == id]
            id_data = id_data[id_data.first_day_of_month == first]
            maeVals.append(min(id_data.mae_linear.values[0], id_data.mae_lasso.values[0], id_data.mae_ridge.values[0], id_data.mae_xgb.values[0], id_data.mae_lstm.values[0], id_data.mae_gru.values[0], id_data.mae_transformer.values[0]))
            # maeVals.append(min(id_data.mae_linear.values[0], id_data.mae_lasso.values[0], id_data.mae_ridge.values[0], id_data.mae_xgb.values[0]))
            smapeVals.append(min(id_data.smape_linear.values[0], id_data.smape_lasso.values[0], id_data.smape_ridge.values[0], id_data.smape_xgb.values[0], id_data.smape_lstm.values[0], id_data.smape_gru.values[0], id_data.smape_transformer.values[0]))
            # smapeVals.append(min(id_data.smape_linear.values[0], id_data.smape_lasso.values[0], id_data.smape_ridge.values[0], id_data.smape_xgb.values[0]))
            
        temp_df = pd.DataFrame({
            'first_day_of_month': [first],
            'mae': sum(maeVals)/len(ids),
            'smape': sum(smapeVals)/len(ids)
        })
        result = result.append(temp_df, ignore_index=True)
    return result
    
def linear_regression(x_test, x_train, y_train):
    linearRegressionModel = LinearRegression()
    linearRegressionModel.fit(x_train, y_train)
    return linearRegressionModel.predict(x_test) 

def lasso(x_test, x_train, y_train):
    lassoModel = Lasso()
    lassoModel.fit(x_train, y_train)
    return lassoModel.predict(x_test)

def ridge(x_test, x_train, y_train):
    ridgeModel = Ridge()
    ridgeModel.fit(x_train, y_train)
    return ridgeModel.predict(x_test)

def xgboost(x_test, x_train, y_train):
    xgbModel = xgb.XGBRegressor(
            objective ='reg:squarederror',
            learning_rate = 0.1,
            max_depth = 5,
            n_estimators = 100
    )
    xgbModel.fit(x_train, y_train)
    return xgbModel.predict(x_test)

def create_dataset(dataset, features, n_steps=1, n_future=1):
    X, Y = [], []
    for i in range(len(dataset) - n_steps - n_future + 1):
        x_seq = dataset[i : (i + n_steps)].reshape(-1, 1)  # Reshape x_seq to a 2D array with a single column
        y_seq = dataset[(i + n_steps) : (i + n_steps + n_future)]
        X.append(np.hstack([x_seq, np.tile(features[i], (n_steps, 1))]))
        Y.append(y_seq)
    return np.array(X, dtype=np.float32).reshape(-1, n_steps, 1 + features.shape[1]), np.array(Y, dtype=np.float32)

def lstm(x_test, x_train, y_train, n_steps, n_features, n_future):
    # Reshape the input to 3D (samples, timesteps, features)
    x_train = x_train.reshape((x_train.shape[0], n_steps, n_features))
    x_test = x_test.reshape((x_test.shape[0], n_steps, n_features))

    # Create the LSTM model
    lstmModel = Sequential()
    lstmModel.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, n_features)))
    lstmModel.add(Dropout(0.2))
    lstmModel.add(LSTM(units=50))
    lstmModel.add(Dropout(0.2))
    lstmModel.add(Dense(units=n_future))

    lstmModel.compile(optimizer='adam', loss='mean_squared_error')
    lstmModel.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

    return lstmModel.predict(x_test)

def gru(x_test, x_train, y_train, n_steps, n_features, n_future):
    # Reshape the input to 3D (samples, timesteps, features)
    x_train = x_train.reshape((x_train.shape[0], n_steps, n_features))
    x_test = x_test.reshape((x_test.shape[0], n_steps, n_features))

    # Create the GRU model
    gruModel = Sequential()
    gruModel.add(GRU(units=50, return_sequences=True, input_shape=(n_steps, n_features)))
    gruModel.add(Dropout(0.2))
    gruModel.add(GRU(units=50))
    gruModel.add(Dropout(0.2))
    gruModel.add(Dense(units=n_future))

    gruModel.compile(optimizer='adam', loss='mean_squared_error')
    gruModel.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

    return gruModel.predict(x_test)

def transformer(data, trainSize, testSize):
    months = [month for month in data.first_day_of_month.unique()]
    data['first_day_of_month'] = data['first_day_of_month'].replace(to_replace=months, value=np.arange(1,40))
    densities = data["microbusiness_density"].values
    
    # use the first 36 and use time series forecasting transformers to precict the next 3
    densities_train = densities[:trainSize]
    densities_test = densities[trainSize:]

    # Initialize the transformer model, loss function and optimizer
    model = Transformer(feature_size=1, num_layers=1, dropout=0.1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare data for the transformer
    train_data = create_inout_sequences(densities_train, 4)  # 12 can be adjusted depending on the periodicity in the data

    # Train the model
    for epoch in range(100):  # 100 epochs, adjust as needed
        model.train()
        for seq, labels in train_data:
            seq = torch.FloatTensor(seq).view(-1, 1, 1)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, torch.FloatTensor(labels))
            loss.backward()
            optimizer.step()

    # Test the model
    model.eval()

    if testSize == 3:
        with torch.no_grad():
            seq = torch.FloatTensor(densities_train).view(-1, 1, 1)   
            predictions = model(seq)
            item = predictions.item()
            arr = np.array([item])
            d= np.append(densities_train, arr, axis=0)

            seq = torch.FloatTensor(d).view(-1, 1, 1)   
            predictions = model(seq)
            item = predictions.item()
            arr = np.array([item])
            b = np.append(d, arr, axis=0)

            seq = torch.FloatTensor(b).view(-1, 1, 1)   
            predictions = model(seq)
            item = predictions.item()
            arr = np.array([item])
            result= np.append(densities_train, arr, axis=0)
            return result[-3:]

    elif testSize == 6:
        with torch.no_grad():
            seq = torch.FloatTensor(densities_train).view(-1, 1, 1)   
            predictions = model(seq)
            item = predictions.item()
            arr = np.array([item])
            d= np.append(densities_train, arr, axis=0)

            seq = torch.FloatTensor(d).view(-1, 1, 1)   
            predictions = model(seq)
            item = predictions.item()
            arr = np.array([item])
            b = np.append(d, arr, axis=0)

            seq = torch.FloatTensor(b).view(-1, 1, 1)   
            predictions = model(seq)
            item = predictions.item()
            arr = np.array([item])
            c = np.append(b, arr, axis=0)

            seq = torch.FloatTensor(c).view(-1, 1, 1)   
            predictions = model(seq)
            item = predictions.item()
            arr = np.array([item])
            e = np.append(c, arr, axis=0)

            seq = torch.FloatTensor(e).view(-1, 1, 1)   
            predictions = model(seq)
            item = predictions.item()
            arr = np.array([item])
            f = np.append(e, arr, axis=0)

            seq = torch.FloatTensor(f).view(-1, 1, 1)   
            predictions = model(seq)
            item = predictions.item()
            arr = np.array([item])
            result= np.append(f, arr, axis=0)
            return result[-6:]

def predict_microbusiness_density(data, split_date, trainSize, testSize):

    result = pd.DataFrame(columns=["cfips", "first_day_of_month", "linear", "lasso", "ridge", "xgb","lstm", "gru","transformer","mae_linear", "mae_lasso", "mae_ridge", "mae_xgb", "mae_lstm", "mae_gru", "mae_transformer", "smape_linear", "smape_lasso", "smape_ridge", "smape_xgb", "smape_lstm", "smape_gru", "smape_transformer"])

    ids = data.cfips.unique()

    print("Number of CFIPS: ", len(ids))

    n_steps = 3
    count = 0
    for cfips in ids:

        print("CFIPS: ", cfips)
        print("Count: ", count)
        count += 1

        cfips_data = data[data.cfips == cfips]
        train, test = split_dataset_by_date(cfips_data, split_date)

        months = [month for month in train.first_day_of_month.unique()]
        train['first_day_of_month'] = train['first_day_of_month'].replace(to_replace=months, value=np.arange(1,trainSize))

        months = [month for month in test.first_day_of_month.unique()]
        test['first_day_of_month'] = test['first_day_of_month'].replace(to_replace=months, value=np.arange(1,testSize))

        x_train = train.drop(["microbusiness_density"], axis=1, inplace=False)
        y_train = train.microbusiness_density

        x_test = test.drop(["microbusiness_density"], axis=1, inplace=False)
        y_test = test.microbusiness_density
            
        extra_features_train = train.drop(["first_day_of_month", "microbusiness_density"], axis=1, inplace=False).values
        extra_features_test = test.drop(["first_day_of_month", "microbusiness_density"], axis=1, inplace=False).values

        x_train_lstm, y_train_lstm = create_dataset(y_train.values, extra_features_train, n_steps, n_future=testSize-1)
        x_test_lstm, _ = create_dataset(np.concatenate((y_train[-n_steps:].values, y_test.values)), extra_features_test, n_steps, n_future=testSize-1)

        linearRegressionModel_pred = linear_regression(x_test, x_train, y_train)
        lassoModel_pred = lasso(x_test, x_train, y_train)
        ridgeModel_pred = ridge(x_test, x_train, y_train)
        xgbModel_pred = xgboost(x_test, x_train, y_train)
        lstmModel_pred = lstm(x_test_lstm, x_train_lstm, y_train_lstm, n_steps, n_features=1 + extra_features_train.shape[1], n_future=testSize-1)
        gruModel_pred = gru(x_test_lstm, x_train_lstm, y_train_lstm, n_steps, n_features=1 + extra_features_train.shape[1], n_future=testSize-1)
        transformerModel_pred = transformer(cfips_data, trainSize - 1, testSize - 1)


        mae_linear = [0] * y_test.shape[0]
        mae_lasso = [0] * y_test.shape[0]
        mae_ridge = [0] * y_test.shape[0]
        mae_xgb = [0] * y_test.shape[0]
        mae_lstm = [0] * y_test.shape[0]
        mae_gru = [0] * y_test.shape[0]
        mae_transformer = [0] * y_test.shape[0]

        smape_linear = [0] * y_test.shape[0]
        smape_lasso = [0] * y_test.shape[0]
        smape_ridge = [0] * y_test.shape[0]
        smape_xgb = [0] * y_test.shape[0]
        smape_lstm = [0] * y_test.shape[0]
        smape_gru = [0] * y_test.shape[0]
        smape_transformer = [0] * y_test.shape[0]

        for a, y in enumerate(y_test):
            mae_linear[a] = metrics.mean_absolute_error([y], [linearRegressionModel_pred[a]])
            mae_lasso[a] = metrics.mean_absolute_error([y], [lassoModel_pred[a]])
            mae_ridge[a] = metrics.mean_absolute_error([y], [ridgeModel_pred[a]])
            mae_xgb[a] = metrics.mean_absolute_error([y], [xgbModel_pred[a]])
            mae_lstm[a] = metrics.mean_absolute_error([y], [lstmModel_pred[0][a]])
            mae_gru[a] = metrics.mean_absolute_error([y], [gruModel_pred[0][a]])
            mae_transformer[a] = metrics.mean_absolute_error([y], [transformerModel_pred[a]])
            smape_linear[a] = smape([y], [linearRegressionModel_pred[a]])
            smape_lasso[a] = smape([y], [lassoModel_pred[a]])
            smape_ridge[a] = smape([y], [ridgeModel_pred[a]])
            smape_xgb[a] = smape([y], [xgbModel_pred[a]])
            smape_lstm[a] = smape([y], [lstmModel_pred[0][a]])
            smape_gru[a] = smape([y], [gruModel_pred[0][a]])
            smape_transformer[a] = smape([y], [transformerModel_pred[a]])


        temp_df = pd.DataFrame({
            'cfips': [cfips] * len(test),
            'first_day_of_month': test['first_day_of_month'],
            'linear': linearRegressionModel_pred,
            'lasso': lassoModel_pred,
            'ridge': ridgeModel_pred,
            'xgb': xgbModel_pred,
            'lstm': lstmModel_pred[0],
            'gru': gruModel_pred[0],
            'transformer': transformerModel_pred,
            'mae_linear': mae_linear,
            'mae_lasso': mae_lasso,
            'mae_ridge': mae_ridge,
            'mae_xgb': mae_xgb,
            'mae_lstm': mae_lstm,
            'mae_gru': mae_gru,
            'mae_transformer': mae_transformer,
            'smape_linear': smape_linear,
            'smape_lasso': smape_lasso,
            'smape_ridge': smape_ridge,
            'smape_xgb': smape_xgb,
            'smape_lstm': smape_lstm,
            'smape_gru': smape_gru,
            'smape_transformer': smape_transformer
        })
        result = pd.concat([result, temp_df], ignore_index=True)
    
    return result

command = input("3 months prediction or 6 months prediction? (3/6) ")
mode = input("Fair or not fair? (fair = 0 /not fair = 1) ")

data = pd.read_csv("data/data.csv")
data.drop(['row_id','county', 'state', 'Neighbour county code'], axis = 1, inplace=True)
data.dropna(inplace=True)

if command == '3':

    if mode == '0':

        data.drop(['active', 'unemployment_rate', 'unemployment', 'employment', 'labor_force'], axis = 1, inplace=True)
        print("Predicting 3 months fair mode")

        predicts = predict_microbusiness_density(data, '2022-07-01', 37, 4)
        print("Calculating score")

        score = total_score(predicts)
        print("Saving results")

        predicts.to_csv("results/3_months_fair_result.csv", index=False)
        print("Saving score")

        score.to_csv("results/3_months_fair_score.csv", index=False)

    elif mode == '1':
        predicts = predict_microbusiness_density(data, '2022-07-01', 37, 4)
        score = total_score(predicts)

        predicts.to_csv("results/3_months_not_fair_result.csv", index=False)
        score.to_csv("results/3_months_not_fair_score.csv", index=False)

elif command == '6':

    if mode == '0':
        data.drop(['active', 'unemployment_rate', 'unemployment', 'employment', 'labor_force'], axis = 1, inplace=True)

        predicts  = predict_microbusiness_density(data, '2022-04-01', 34, 7)
        score = total_score(predicts)

        predicts.to_csv("results/6_months_fair_result.csv", index=False)
        score.to_csv("results/6_months_fair_score.csv", index=False)
        
    elif mode == '1':
        predicts  = predict_microbusiness_density(data, '2022-04-01', 34, 7)
        score = total_score(predicts)

        predicts.to_csv("results/6_months_not_fair_result.csv", index=False)
        score.to_csv("results/6_months_not_fair_score.csv", index=False)