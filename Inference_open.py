import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import Transformer_based_model as transformer
import Mamba_based_model as mamba

def Inference():
    tsmc_data = pd.read_csv('./Datasets/tsmc_stock_prices_INT_open_only_datetime_2010-2023.csv')
    tsmc_data.index = tsmc_data["date"]
    tsmc_data = tsmc_data.drop(columns=["date"])
    # print(tsmc_data.head())

    input_length = 30
    output_length = 1
    dataset = tsmc_data['open'].to_numpy()

    scaler = MinMaxScaler()
    dataset_norm = scaler.fit_transform(dataset.reshape(-1, 1)).flatten()
    dataset_list = []
    for i in range(len(dataset) - input_length - output_length):
        dataset_list.append(dataset_norm[i:i + input_length + output_length])
    dataset_list = np.array(dataset_list)

    split_idx_train = int(len(dataset_list) * 0.8)
    split_idx_val = int(len(dataset_list) * 0.9)

    x_train = dataset_list[:split_idx_train, :-1]
    y_train = dataset_list[:split_idx_train, -1:]

    x_val = dataset_list[split_idx_train:split_idx_val, :-1]
    y_val = dataset_list[split_idx_train:split_idx_val, -1:]

    x_test = dataset_list[split_idx_val:, :-1]
    y_test = dataset_list[split_idx_val:, -1:]

    # print('x_train.shape:', x_train.shape)
    # print('y_train.shape:', y_train.shape)
    # print('x_val.shape:', x_val.shape)
    # print('y_val.shape:', y_val.shape)
    # print('x_test.shape:', x_test.shape)
    # print('y_test.shape:', y_test.shape)


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

    transformerModel.load_weights('./Weights/transformer_open_checkpoint.weights.h5')
    mambaModel.load_weights('./Weights/mamba_open_checkpoint.weights.h5')

    transformer_pred = transformerModel.predict(x_test)
    mamba_pred = mambaModel.predict(x_test)
    y_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    transformer_pred = scaler.inverse_transform(transformer_pred.reshape(-1, 1)).flatten()
    mamba_pred = scaler.inverse_transform(mamba_pred.reshape(-1, 1)).flatten()
    transformer_rmse = np.sqrt(mean_squared_error(y_real, transformer_pred))
    mamba_rmse = np.sqrt(mean_squared_error(y_real, mamba_pred))

    return y_real, transformer_pred, mamba_pred, transformer_rmse, mamba_rmse

# y_real, transformer_pred, mamba_pred, transformer_rmse, mamba_rmse = Inference()
# plt.figure(1)
# plt.plot(y_real, label='real')
# plt.plot(transformer_pred, label='transformer')
# plt.plot(mamba_pred, label='mamba')
# transformer_rmse = np.sqrt(mean_squared_error(y_real, transformer_pred))
# mamba_rmse = np.sqrt(mean_squared_error(y_real, mamba_pred))
# plt.xlabel('Transformer RMSE Error: {}'.format(transformer_rmse) + "\n" + 'Mamba RMSE Error: {}'.format(mamba_rmse))
# plt.legend()
# plt.title('Open price prediction result')
# plt.show()
