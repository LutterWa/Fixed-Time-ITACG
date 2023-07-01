import math
import numpy as np
from random import randint, uniform, choice
from scipy import interpolate  # 导入插值函数

pi = 3.141592653589793
RAD = 180 / pi  # 弧度转角度
g = 9.81

max_time = 300

Ma2 = np.array([[0.4, 39.056, 0.4604, 39.072],
                [0.6, 39.468, 0.4635, 39.242],
                [0.8, 40.801, 0.4682, 40.351],
                [0.9, 41.372, 0.4776, 41.735],
                [1.0, 41.878, 0.4804, 43.014],
                [1.2, 42.468, 0.4797, 42.801],
                [1.4, 41.531, 0.4784, 42.656],
                [1.6, 41.224, 0.4771, 42.593],
                [1.8, 40.732, 0.4768, 42.442],
                [2.0, 40.321, 0.4761, 42.218],
                [2.2, 40.033, 0.4756, 42.034],
                [2.4, 39.912, 0.4751, 41.977],
                [2.6, 39.756, 0.4748, 41.893],
                [2.8, 39.501, 0.4743, 41.808],
                [3.0, 39.344, 0.4739, 41.754]])


class Missile:  # 常规制导武器
    def __init__(self, state=None, k=(1.0, 1.0, 1.0)):  # 构造函数
        if state is None:
            state = [0., 600., 0. / RAD, -20000., 10000]

        """导弹自身状态"""
        self.S = 0.0572555  # 参考面积
        self.m = 200  # 重量kg

        self.state = np.array(state)
        self.t = state[0]  # 时间
        self.v = state[1]  # 速度
        self.theta = state[2]  # 弹道倾角
        self.x = state[3]  # 横向位置
        self.y = state[4]  # 高度

        """弹目相对状态"""
        Rx, Ry = self.x, self.y
        self.q = math.atan2(Ry, Rx)  # 弹目视线角
        self.R = np.linalg.norm([Rx, Ry], ord=2)  # 弹目距离
        self.Rdot = 0.
        self.qdot = 0.
        self.am = 0.  # 制导指令
        self.ad = -60. / RAD  # 期望的终端角度

        def load_atm(path):  # 大气数据载入
            file = open(path)
            atm_str = file.read().split()
            atm = []
            for _ in range(0, len(atm_str), 3):
                atm.append([float(atm_str[_]), float(atm_str[_ + 1]), float(atm_str[_ + 2])])
            return np.array(atm)

        """环境气象参数"""
        atm = load_atm('atm2.txt')  # 大气参数
        self.f_rho = interpolate.interp1d(atm[:, 0], atm[:, 1], 'linear')
        self.f_ma = interpolate.interp1d(atm[:, 0], atm[:, 2], 'linear')
        self.f_clalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 1] * k[0], 'cubic')  # 升力系数
        self.f_cd0 = interpolate.interp1d(Ma2[:, 0], Ma2[:, 2] * k[1], 'cubic')  # 零升阻力
        self.f_cdalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 3] * k[2], 'cubic')  # 攻角阻力

        self.record = {"state": [], "R": [], "q": [], "am": [], "alpha": [], "ad": []}  # 全弹道历史信息

    def refresh(self):  # 更新系统状态
        self.t = self.state[0]  # 时间
        self.v = self.state[1]  # 速度
        self.theta = self.state[2]  # 弹道倾角
        self.x = self.state[3]  # 横向位置
        self.y = self.state[4]  # 高度

    def modify(self, state=None, ad=None):  # 修改导弹初始状态
        self.terminate()
        if state is None:
            state = [0., uniform(400, 800), uniform(-15, 30) / RAD, uniform(-40000, -5000), uniform(5000, 20000)]
        self.state = np.array(state)
        if ad is None:
            ad = -uniform(0, 90)
        self.ad = ad / RAD
        self.seeker()  # 更新弹体状态和弹目相对运动学关系

    def guidance(self):  # 制导功能
        R, Rdot, q, qdot = self.seeker()  # 计算弹目相对信息
        ac = 3 * self.v * qdot + g * math.cos(self.theta)
        tsg = self.v * self.qdot + 2 * self.v ** 2 * (self.q - self.ad) / self.R  # 弹道成型的偏置项形式
        return ac + tsg

    def seeker(self):  # 计算弹目相对信息
        self.refresh()
        Rx = -self.x
        Ry = -self.y
        self.q = q = math.atan2(Ry, Rx)  # 弹目视线角
        self.R = R = np.linalg.norm([Rx, Ry], ord=2)  # 弹目距离
        vx = -self.v * math.cos(self.theta)  # x向速度
        vy = -self.v * math.sin(self.theta)  # y向速度
        self.Rdot = Rdot = (Rx * vx + Ry * vy) / R
        self.qdot = qdot = (Rx * vy - Ry * vx) / R ** 2
        return R, Rdot, q, qdot

    def dery(self, state, X, L):  # 右端子函数
        v = state[1]
        theta = state[2]
        m = self.m
        dy = np.array([1,
                       -X / m - g * math.sin(theta),
                       (L - m * g * math.cos(theta)) / (v * m),
                       v * math.cos(theta),
                       v * math.sin(theta)])
        return dy

    def step(self, h=0.001, ab=0.):  # 单步运行
        self.refresh()  # 更新系统状态
        if self.t < max_time and self.R > 50 * h and self.y > 0:
            rho = self.f_rho(np.clip(self.y, 0., 86000))  # 大气密度
            ma = np.clip(self.v / self.f_ma(self.y), Ma2[0, 0], Ma2[-1, 0])  # 马赫数
            Q = 0.5 * rho * (self.v ** 2)  # 动压

            am_bound = 5 * g  # 制导指令限幅
            alpha_bound = 90 / RAD  # 攻角限幅

            self.am = np.clip(self.guidance() + ab, -am_bound, am_bound)  # 过载指令
            cl_alpha = self.f_clalpha(ma)
            alpha = np.clip((self.m * self.am) / (Q * self.S * cl_alpha), -alpha_bound, alpha_bound)  # 平衡攻角

            cd = self.f_cd0(ma) + self.f_cdalpha(ma) * alpha ** 2  # 阻力系数
            cl = cl_alpha * alpha  # 升力系数

            X = cd * Q * self.S  # 阻力
            L = cl * Q * self.S  # 升力

            # 四阶龙格库塔
            def rk4(state=self.state):
                k1 = h * self.dery(state, X, L)
                k2 = h * self.dery(state + 0.5 * k1, X, L)
                k3 = h * self.dery(state + 0.5 * k2, X, L)
                k4 = h * self.dery(state + k3, X, L)
                output = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                return output

            # 欧拉法
            def euler(state=self.state):
                output = state + h * self.dery(state, X, L)
                return output

            # self.state = rk4()
            self.state = euler()

            if self.state[2] > pi:
                self.state[2] = self.state[2] - 2 * pi
            if self.state[2] < -pi:
                self.state[2] = self.state[2] + 2 * pi

            self.record["state"].append(self.state)
            self.record["R"].append(self.R)
            self.record["q"].append(self.q)
            self.record["am"].append(self.am)
            self.record["alpha"].append(alpha)
            self.record["ad"].append(self.ad)
            return False
        else:
            return True

    def terminate(self):  # 终止弹道并清空状态
        self.R, self.q = 0., 0.  # 弹目相对关系
        self.am = 0.  # 制导指令
        self.record = {"state": [], "R": [], "q": [], "am": [], "alpha": [], "ad": []}  # 全弹道历史信息


if __name__ == '__main__':
    miss = Missile()
    for _ in range(100):
        miss.modify([0., 600, 0, -20000, 10000], -90)
        done = False
        h = 0.001
        while done is False:
            done = miss.step(h)
        print("脱靶量={:.4f} 飞行时间={:.4f} 终端角度={:.4f}".format(miss.R, miss.t, -miss.q * RAD))
