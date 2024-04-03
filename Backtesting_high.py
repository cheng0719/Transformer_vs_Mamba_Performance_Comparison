import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import Transformer_based_model as transformer
import Mamba_based_model as mamba

def Backtesting():
    tsmc_data = pd.read_csv('./Datasets/tsmc_stock_prices_INT_high_only_datetime_backtesting.csv')
    tsmc_data.index = tsmc_data["date"]
    tsmc_data = tsmc_data.drop(columns=["date"])
    # print(tsmc_data.head())

    input_length = 30
    output_length = 1
    dataset = tsmc_data['high'].to_numpy()

    scaler = MinMaxScaler()
    dataset_norm = scaler.fit_transform(dataset.reshape(-1, 1)).flatten()
    dataset_list = []
    x = []
    y = []

    x.append(dataset_norm[0 : input_length])
    y.append(dataset_norm[input_length])
    x = np.array(x)
    y = np.array(y)

    y = np.expand_dims(y, axis=1)

    # print('x.shape:', x.shape)
    # print('y.shape:', y.shape)

    transformer_args = transformer.ModelArgs()
    transformerModel = transformer.init_model(transformer_args)

    mamba_args = mamba.ModelArgs(
        model_input_dims=3,
        model_states=32,
        num_layers=5,
        dropout_rate=0.1
    )
    mambaModel = mamba.init_model(mamba_args)

    ### Model training ###
    # transformerHistory = transformerModel.fit(x_train, y_train,
    #                     batch_size=256,
    #                     epochs=200,
    #                     validation_data=(x_val, y_val),
    #                     callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    # mambaHistory = mambaModel.fit(x_train, y_train,
    #                     batch_size=256,
    #                     epochs=200,
    #                     validation_data=(x_val, y_val),
    #                     callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    transformerModel.load_weights('./Weights/transformer_high_checkpoint.weights.h5')
    mambaModel.load_weights('./Weights/mamba_high_checkpoint.weights.h5')

    transformer_pred = transformerModel.predict(x)
    mamba_pred = mambaModel.predict(x)
    y_real = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    transformer_pred = scaler.inverse_transform(transformer_pred.reshape(-1, 1)).flatten()
    mamba_pred = scaler.inverse_transform(mamba_pred.reshape(-1, 1)).flatten()

    transformer_rmse = np.sqrt(mean_squared_error(y_real, transformer_pred))
    mamba_rmse = np.sqrt(mean_squared_error(y_real, mamba_pred))

    return y_real, transformer_pred, mamba_pred, transformer_rmse, mamba_rmse

# y_real, transformer_pred, mamba_pred, transformer_rmse, mamba_rmse = Backtesting()
# print('TSMC 2024/2/23 High Price Prediction: ' + str(y_real[-1]))
# print('Transformer Predicted High Price: ' + str(transformer_pred[-1]))
# print('Mamba Predicted High Price: ' + str(mamba_pred[-1]))
# print('Transformer RMSE: ' + str(transformer_rmse))
# print('Mamba RMSE: ' + str(mamba_rmse))