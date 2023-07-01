import os
import numpy as np
from scipy.io import loadmat, savemat
from missile import Missile, RAD
from multiprocessing import Pool

g = 9.81
cpu_counts = 10  # 并行采样使用的进程数(需小于电脑支持的最大进程数)
samples_num = 10000  # 采样弹道数


def generate(miss):
    h = 0.01  # 仿真步长
    while True:
        miss.modify()
        done = False
        while done is False:
            done = miss.step(h)
        if miss.R < 200 * h and miss.t > 1:
            break
    missile = np.array(miss.record["state"])
    s = np.concatenate([np.array([miss.record["ad"]]).T, missile[:, 1:]], axis=1)  # v, theta, x, y
    x = np.dot(s, np.diag([10, 0.05, 10, 1e-3, 1e-3]))  # 输入向量
    y = missile[-1, 0] - missile[:, 0]  # 输出向量 tgo
    return x, y


def process(count):
    miss = Missile()  # 创建导弹对象
    x, y = generate(miss)  # 生成随机样本
    for itr in range(int(samples_num // cpu_counts)):
        print("==========迭代次数 {}==========".format(itr + 1))
        batch_x, batch_y = generate(miss)  # 生成随机样本
        x = np.concatenate([x, batch_x])
        y = np.concatenate([y, batch_y])
    flight_data = {"x": x, "y": y}
    if not os.path.exists('mats'):
        os.makedirs('mats')
    savemat('mats/anti_flight_data_{}.mat'.format(count), flight_data)


def collect_data():
    # pool = Pool(cpu_counts)
    # pool.map(process, list(range(cpu_counts)))  # 创建多个线程

    data_raw = loadmat('./mats/anti_flight_data_0.mat')
    print("simulate data collect done!")

    x = data_raw["x"]
    y = data_raw["y"].T

    for i in range(1, cpu_counts):
        data_raw = loadmat('./mats/anti_flight_data_{}.mat'.format(i))
        x = np.concatenate([x, data_raw["x"]])
        y = np.concatenate([y, data_raw["y"].T])
    flight_data = {"x": x, "y": y}
    savemat('./mats/itacg_train_data.mat', flight_data)
    return x, y


if __name__ == "__main__":
    collect_data()
