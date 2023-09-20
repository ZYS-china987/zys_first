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

def main():
    # 1. 数据集选择
    data = np.loadtxt('../../data1.txt')
    data1 = np.loadtxt('../../data2.txt')
    dim = data.shape  # (500, 19)
    data_input = data1[:, :11]  # (500, 11)
    data_output = data[:, -1:]  # (500, 1)
    n_list = list(range(int(dim[0])))
    random.shuffle(n_list)  # 将0-499的列表顺序随机化
    # 训练集选择
    x_train = data_input[n_list[:400], :]  # (400, 11)
    y_train = data_output[n_list[:400], :]  # (400, 1)
    x_test = data_input[n_list[-100:], :]
    y_test = data_output[n_list[-100:], :]

    # 用more和less两个模型
    x_train_more = x_train[:330, :]
    y_train_more = y_train[:330, :]
    x_train_less = x_train[-70:, :]
    y_train_less = y_train[-70:, :]

    # 标准化
    transfer_more = MaxAbsScaler()
    transfer_more.fit(x_train_more)
    x_train_more_Scaler = transfer_more.transform(x_train_more)
    x_test_Scaler = transfer_more.transform(x_test)
    transfer_less = MaxAbsScaler()
    transfer_less.fit(x_train_less)
    x_train_less_Scaler = transfer_less.transform(x_train_less)
    svm_more = SVR(kernel='poly',C=10000,gamma=0.01,epsilon=0.01)
    svm_more.fit(x_train_more_Scaler, y_train_more.ravel())

    svm_less = SVR(kernel='poly',C=10000,gamma=0.01,epsilon=0.01)
    svm_less.fit(x_train_less_Scaler, y_train_less.ravel())

    # TODO 计算新数据对这两个模型的欧式距离
    x_train_more_mean = x_train_more_Scaler.sum(axis=0) / 330
    distance_more = np.zeros([100, 1])
    pre_more = np.zeros([330,1])
    pre_less = np.zeros([70,1])

    for p in range(100):
        distance_more[p, :] = np.sqrt(np.sum(np.square(x_test_Scaler[p, :] - x_train_more_mean)))

    x_train_less_mean = x_train_less_Scaler.sum(axis=0) / 70
    distance_less = np.zeros([100, 1])
    for j in range(100):
        distance_less[j, :] = np.sqrt(np.sum(np.square(x_test_Scaler[j, :] - x_train_less_mean)))
    pre_train_more = svm_more.predict(x_train_more_Scaler)
    pre_train_less = svm_less.predict(x_train_less_Scaler)
    pre_more = svm_more.predict(x_test_Scaler)
    pre_less = svm_less.predict(x_test_Scaler)
    w_more = np.ones([1, 100])
    w_less = np.ones([1, 100])
    pre = np.zeros([1, 100])
    for k in range(100):
        if distance_less[k, :] < distance_more[k, :]:
            w_less[0, k] = 1 - ((distance_more[k, :] - distance_less[k, :]) / distance_more[k, :])
            w_more[0, k] = (distance_more[k, :] - distance_less[k, :]) / distance_more[k, :]
            pre[0, k] = pre_more[k] * w_more[0, k] + pre_less[k] * w_less[0, k]
        elif distance_less[k, :] > distance_more[k, :]:
            w_more[0, k] = 1 - ((distance_less[k, :] - distance_more[k, :]) / distance_less[k, :])
            w_less[0, k] = (distance_less[k, :] - distance_more[k, :]) / distance_less[k, :]
            pre[0, k] = pre_more[k] * w_more[0, k] + pre_less[k] * w_less[0, k]
        else:
            w_more[0, k] = 0.5
            w_less[0, k] = 0.5
            pre[0, k] = pre_more[k] * w_more[0, k] + pre_less[k] * w_less[0, k]

    # pre = pre_more * w_1 + pre_less * w_2
    MSE = mean_squared_error(y_test, pre.reshape(-1, 1))
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(y_test, pre.reshape(-1, 1))
    MAPE = mean_absolute_percentage_error(y_test, pre.reshape(-1, 1))
    print("MSE:", MSE)
    print("RMSE:", RMSE)
    print("MAE:", MAE)
    print("MAPE", MAPE)
    np.savetxt("w_more",w_more)
    np.savetxt("w_less",w_less)
    np.savetxt("pre_train_more",pre_train_more)
    np.savetxt("pre_train_less",pre_train_less)
    np.savetxt("x_train",x_train)
    np.savetxt("x_test",x_test)
    return y_test.ravel(), pre, RMSE, MAE, MAPE, w_more, w_less


if __name__ == '__main__':
    pre = np.zeros([30, 100])
    true = np.zeros([30, 100])
    RMSE = np.zeros([30, 1])
    MAE = np.zeros([30, 1])
    MAPE = np.zeros([30, 1])
    w_more = np.zeros([30, 100])
    w_less = np.zeros([30, 100])
    for i in range(30):
        filterwarnings('ignore')
        results = main()
        true[i, :] = results[0]
        pre[i, :] = results[1]
        RMSE[i, :] = results[2]
        MAE[i, :] = results[3]
        MAPE[i, :] = results[4]
        w_more[i, :] = results[5]
        w_less[i, :] = results[6]
    np.savetxt('true.txt', true)
    np.savetxt('pre.txt', pre)
    np.savetxt('RMSE.txt', RMSE)
    np.savetxt('MAE.txt', MAE)
    np.savetxt('MAPE.txt', MAPE)
    np.savetxt("w_less", w_less)
    np.savetxt("w_more", w_more)
    print("----", RMSE.sum(axis=0) / 30)
