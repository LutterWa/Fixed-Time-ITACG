import os
import keras
import keras.optimizers.optimizer_v2.adam as adam
from keras.layers import Input, Dense, add
from keras.activations import relu
from keras.callbacks import TensorBoard
from scipy.io import loadmat, savemat


def load_data(file):
    data_raw = loadmat(file)
    x = data_raw["x"]
    y = data_raw["y"]
    print("training data load done!")
    return x, y


def init_network(layers, units):
    # 创建网络
    x = Input(shape=[5], name="input")
    l1 = Dense(units=units, activation='relu')(x)
    for i in range((layers - 1) // 2):  # 神经网络层数
        l2 = Dense(units=units, activation='relu')(l1)
        l1 = relu(add([Dense(units=units)(l2), l1]))  # 残差神经网络
    y = Dense(1, name='outputs')(l1)
    model = keras.Model(inputs=x, outputs=y)
    return model


def res():
    try:
        model = keras.models.load_model("itacg_model/res_model.h5")
    except OSError:
        if not os.path.exists("itacg_model"):
            os.makedirs("itacg_model")
        model = init_network(layers=10, units=100)
        model.compile(loss="mse", optimizer=adam.Adam(learning_rate=0.001))

        # 训练网络
        x, y = load_data('mats/itacg_train_data.mat')
        tb = TensorBoard(log_dir='logs/res', write_images=True)
        model.fit(x, y, batch_size=500, epochs=30, validation_split=0.02, verbose=1, callbacks=tb)
        model.save("itacg_model/res_model.h5")
    return model


def load_train(x, y, h5file):  # 加载性能最好的模型进一步训练
    model = keras.models.load_model(h5file)
    for lr in [0.0005, 0.0002, 0.0001, 0.00005]:
        model.compile(loss="mse", optimizer=adam.Adam(learning_rate=lr))
        model.fit(x, y, batch_size=500, epochs=2, validation_split=0.02, verbose=1)
    model.save("itacg_model/res_model.h5")


if __name__ == "__main__":
    model = res()  # 第一次训练神经网络

    # 加载性能最好的模型进一步训练
    # x, y = load_data('mats/itacg_train_data.mat')
    # load_train(x, y, "itacg_model/res_model.h5")
