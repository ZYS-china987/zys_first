import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVR
from warnings import filterwarnings
from matplotlib import pyplot as plt
import joblib


# TODO 跑100次的结果  MSE
def main():
    # 1. 数据集选择
    data = np.loadtxt('../../data1.txt')
    data1 = np.loadtxt('../../data2.txt')
    dim = data.shape  # (500, 19)
    data_input = data1[:, :11]  # (500, 16)
    data_output = data[:, -1:]  # (500, 1)
    n_list = list(range(int(dim[0])))
    random.shuffle(n_list)  # 将0-499的列表顺序随机化
    # 训练集选择
    x_train = data_input[n_list[:400], :]  # (400, 16)
    y_train = data_output[n_list[:400], :]  # (400, 1)
    x_test = data_input[n_list[-100:], :]
    y_test = data_output[n_list[-100:], :]

    # 标准化
    # transfer = StandardScaler()
    # transfer.fit(x_train)
    # x_train_Scaler = transfer.transform(x_train)
    # x_test_Scaler = transfer.transform(x_test)

    # 最大最小归一化
    transfer = MinMaxScaler()
    transfer.fit(x_train)
    x_train_Scaler = transfer.transform(x_train)
    x_test_Scaler = transfer.transform(x_test)

    # [-1,1]归一化
    # transfer = MaxAbsScaler()
    # transfer.fit(x_train)
    # x_train_Scaler = transfer.transform(x_train)
    # x_test_Scaler = transfer.transform(x_test)

    svm = SVR(kernel="poly",C=10000,gamma=0.01,epsilon=0.01)
    svm.fit(x_train_Scaler, y_train.ravel())

    pre = svm.predict(x_test_Scaler)
    MSE = mean_squared_error(y_test, pre)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(y_test, pre)
    MAPE = mean_absolute_percentage_error(y_test, pre)

    print("MSE:", MSE)
    print("MAE:", MAE)
    print("RMSE:", RMSE)
    print("MAPE", MAPE)
    return y_test.ravel(), pre, RMSE, MAE, MAPE


if __name__ == '__main__':
    pre = np.zeros([30, 100])
    true = np.zeros([30, 100])
    RMSE = np.zeros([30, 1])
    MAE = np.zeros([30, 1])
    MAPE = np.zeros([30, 1])
    for i in range(30):
        filterwarnings('ignore')
        results = main()
        true[i, :] = results[0]
        pre[i, :] = results[1]
        RMSE[i, :] = results[2]
        MAE[i, :] = results[3]
        MAPE[i, :] = results[4]
    np.savetxt('true.txt', true)
    np.savetxt('pre.txt', pre)
    np.savetxt('RMSE.txt', RMSE)
    np.savetxt('MAE.txt', MAE)
    np.savetxt('MAPE.txt', MAPE)
    print("----", RMSE.sum(axis=0) / 30)
