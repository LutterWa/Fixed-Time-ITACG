import math
import os
import sys
import random
import keras
import numpy as np
from scipy.io import savemat, loadmat
from resnet import load_data, res
from missile import Missile, RAD

if not os.path.exists('itacg_figs'):
    os.makedirs('itacg_figs')

task = ["monte", "single", "compare", "paras", "res_test"][0]


class ITACG(Missile):
    def __init__(self, k=(1.0, 1.0, 1.0)):
        super().__init__(k=k)
        self.res = keras.models.load_model("itacg_model/res_model.h5")
        x = np.dot([self.ad, self.v, self.theta, self.x, self.y], np.diag([10, 0.05, 10, 1e-3, 1e-3]))
        self.tgo = float(self.res.predict(x[np.newaxis, :], verbose=0, use_multiprocessing=True))

    def get_tgo(self):
        x = np.dot([self.ad, self.v, self.theta, self.x, self.y], np.diag([10, 0.05, 10, 1e-3, 1e-3]))
        self.tgo = float(self.res.predict(x[np.newaxis, :], verbose=0, use_multiprocessing=True))
        return self.tgo

    def itacg_step(self, h=0.001, td=80.):
        t, v, theta, R, q, ad = self.t, self.v, self.theta, self.R, self.q, self.ad
        tgo = env.get_tgo()

        eta = theta - q  # 速度前置角
        eta_e = ad - q  # 速度角误差
        Lgo = R * (1 + (2 * eta ** 2 + 2 * eta_e ** 2 - eta * eta_e) / 30)  # 预测剩余轨迹长度
        va = Lgo / tgo  # 平均速度
        e = max(td - t - tgo, 0)  # 飞行时间误差

        def fl(e, p):
            return abs(e) ** p * np.sign(e)

        k1, k2, k3 = 5.7708, 4, 4  # 4/ln2约等于5.7708
        p1, p2 = 0.5, 1.5
        fixe = k1 * fl(e, p1) + k2 * e + k3 * fl(e, p2)
        ab = 30 * v * va / (R * tgo * (4 * eta - eta_e)) * fixe
        return self.step(h=h, ab=ab)


def progress_bar(i, t):
    i *= 100  # 转换为百分比
    print("\r", end="")
    print("{:.2f} Data in progress: {:.2f}% ".format(t, i),
          "[" + "=" * (int(min(i, 100) / 2)) + ">" + "." * (50 - int(min(i, 100) / 2)) + "]", end="")
    sys.stdout.flush()


env = ITACG()

if task == "monte":
    # result = loadmat('itacg_figs/sim_fixe_monte_euler.mat')
    result = {}
    N = 500
    for i in range(1, N + 1):
        while True:
            env.modify()
            td = env.get_tgo() * random.uniform(1.0, 1.2)
            done = False
            while done is False:
                if (td - env.t - env.tgo) < -1 or env.t / td > 1.01:
                    done = True
                elif env.tgo < 2:
                    done = env.itacg_step(h=0.01, td=td)
                else:
                    done = env.itacg_step(h=0.2, td=td)
                progress_bar(env.t / td, td - env.t - env.tgo)
            if env.R < 5 and abs(env.ad - env.theta) * RAD < 1 and abs(td - env.t) < 1:
                break
        print("{}/{} 脱靶量={:.4f} 角度误差={:.4f} 时间误差={:.4f}".format(
            i, N, env.R, (env.ad - env.theta) * RAD, td - env.t))
        result["sim_{}".format(i)] = env.record
        result["err_{}".format(i)] = [env.ad * RAD, env.theta * RAD, (env.ad - env.theta) * RAD, td, env.t, td - env.t]
        savemat('itacg_figs/sim_fixe_monte_euler.mat', result)
elif task == "single":
    tds = [60, 70, 80]
    for td in tds:
        ad = -90
        env.modify(state=[0., 600, 0, -20000, 10000], ad=ad)
        result = {"td": [], "tgo": []}
        done = False
        while done is False:
            done = env.itacg_step(h=0.01, td=td)
            progress_bar(env.t / td, td - env.t - env.tgo)
            result["td"].append(td - env.t)
            result["tgo"].append(env.tgo)
        print("脱靶量={:.4f} 角度误差={:.4f} 时间误差={:.4f}".format(env.R, (env.ad - env.theta) * RAD, td - env.t))
        savemat('itacg_figs/sim_fixe_ad_{}_td_{}.mat'.format(-ad, td), dict(env.record, **result))

    ads = [-45, -60]
    for ad in ads:
        td = 80
        env.modify(state=[0., 600, 0, -20000, 10000], ad=ad)
        print(env.get_tgo())
        result = {"td": [], "tgo": []}
        done = False
        while done is False:
            done = env.itacg_step(h=0.01, td=td)
            progress_bar(env.t / td, td - env.t - env.tgo)
            result["td"].append(td - env.t)
            result["tgo"].append(env.tgo)
        print("脱靶量={:.4f} 角度误差={:.4f} 时间误差={:.4f}".format(env.R, (env.ad - env.theta) * RAD, td - env.t))
        savemat('itacg_figs/sim_fixe_ad_{}_td_{}.mat'.format(-ad, td), dict(env.record, **result))
elif task == "compare":
    td = 80
    ad = -90
    adr = ad / RAD
    for compared_itacg in ["li", "chen"]:
        env.modify(state=[0., 600, 0, -20000, 10000], ad=ad)
        result = {"td": [], "tgo": []}
        done = False
        while done is False:
            eta = env.theta - env.q
            eta_e = adr - env.q
            P = env.R / env.v * (1 + (2 * eta ** 2 + 2 * eta_e ** 2 - eta * eta_e) / 30)  # 预测剩余飞行时间
            D = td - env.t  # 期望剩余飞行时间
            E = max(D - P, 0)  # 剩余飞行时间误差
            if compared_itacg == "li":  # 最优误差动力学推导的ITACG 李斌
                k_li = 6
                ab = 30 * k_li * env.v ** 2 * E / (env.R * P * (4 * eta - eta_e))
            elif compared_itacg == "chen":  # 非线性控制推导的ITACG chen
                k_chen, p_chen, q_chen, mu_chen, epsilon_chen = 150, 2, 1, 0.1, 0.2


                def m(phi):
                    return 1 if phi >= epsilon_chen else (phi / epsilon_chen) ** 2


                def sgn(x):
                    return 1 if x >= 0 else -1


                def sat(x):
                    return np.sign(x) if abs(x) >= 1 else x


                phi_chen = env.q - adr + 4 * eta
                ab = (m(phi_chen) * k_chen / phi_chen + p_chen * np.sign(phi_chen) + q_chen * sgn(eta) * sgn(-E)) * sat(
                    E / mu_chen)

            done = env.step(h=0.01, ab=ab)
            result["td"].append(td - env.t)
            result["tgo"].append(env.get_tgo())
            progress_bar(env.t / td, td - env.t - env.tgo)
        print("脱靶量={:.4f} 角度误差={:.4f} 时间误差={:.4f}".format(env.R, (env.ad - env.theta) * RAD, td - env.t))
        savemat('itacg_figs/sim_{}_ad_{}_td_{}.mat'.format(compared_itacg, -ad, td), dict(env.record, **result))
elif task == "res_test":
    model = res()
    x, y = load_data("mats/itacg_train_data.mat")
    y_hat = model.predict(x[-int(15e4):, :])
    savemat("itacg_figs/res_test.mat", {"y": y[-int(15e4):], "y_hat": y_hat})
