''''
Generate Data of    dX_t = drift(X_t) dt + diffusion(X_t) dB_t,  0<=t<=1
time_instants: E.g. torch.tensor([0, 0.2, 0.5, 1])
samples_num: E.g. 10000
dim: E.g. 1
drift_term: E.g. torch.tensor([0, 1, 0, -1]) -- that means drift = x - x^3
diffusion_term: E.g. torch.tensor([1, 0, 0, 0]) -- that means diffusion = 1
return data: [time, samples, dim]
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
from scipy import integrate, stats


class DataSet(object):
    def __init__(self, time_instants, dt, dim, drift_term, diffusion_term,
                 initialization, sampling, explosion_prevention=False):
        self.time_instants = time_instants
        self.dt = dt
        self.dim = dim
        self.drift_term = drift_term
        self.diffusion_term = diffusion_term
        self.initialization = initialization
        self.sampling = sampling
        self.explosion_prevention = explosion_prevention

        self.explosion_prevention_N = 0

    def drift(self, x):
        y = 0
        for i in range(self.drift_term.shape[0]):
            y = y + self.drift_term[i] * x ** i
        return y

    def diffusion(self, x):
        y = 0
        for i in range(self.diffusion_term.shape[0]):
            y = y + self.diffusion_term[i] * x ** i
        return y

    def subSDE(self, t0, t1, x):
        if t0 == t1:
            return x
        else:
            t = torch.arange(t0, t1 + self.dt, self.dt)
            y = x
            for i in range(t.shape[0] - 1):
                y = y + self.drift(y) * self.dt + self.diffusion(y) * torch.sqrt(torch.tensor(self.dt)) * torch.randn(self.dim)
                if self.explosion_prevention:
                    if any(y < 0):
                        y[y < 0] = 0
                        self.explosion_prevention_N = self.explosion_prevention_N + 1
            return y

    @utils.timing
    def get_data(self, sample_num, sub_length, sample_type, plot_hist=False):
        data = torch.zeros(self.time_instants.shape[0], self.dim)
        data[0, :] = self.subSDE(0, self.time_instants[0], self.initialization)  # self.initialization
        for i in range(self.time_instants.shape[0] - 1):
            data[i + 1, :] = self.subSDE(self.time_instants[i], self.time_instants[i + 1], data[i, :])
            if 10 * i % self.time_instants.shape[0] == 0:
                print("Generate data %s%% finished..." % (100 * i // self.time_instants.shape[0]))

        data_process = self.sampling(data, self.time_instants, sample_num, sub_length)
        data0 = data_process.get_sub_data(sample_way=sample_type)

        if self.explosion_prevention:
            print("explosion_prevention * %s" % self.explosion_prevention_N)
        if plot_hist:
            for i in range(self.dim):
                plt.figure()
                plt.hist(x=data0[:, :, i].reshape(-1).numpy(), bins=60, range=[data.min().numpy(), data.max().numpy()], density=True)
                plt.title("t=All")
                plt.figure()
                plt.hist(x=data0[0, :, i].numpy(), bins=60, range=[data.min().numpy(), data.max().numpy()], density=True)
                plt.title("t=0")
                plt.figure()
                plt.hist(x=data0[-1, :, i].numpy(), bins=60, range=[data.min().numpy(), data.max().numpy()], density=True)
                plt.title("t=T")
            plt.show()
        return data0


class Sampling(object):
    def __init__(self, data, time, sample_num, sub_length):
        self.data = data
        self.time = time
        self.sample_num = sample_num
        self.sub_length = sub_length

        self.time_total, self.dim = self.data.shape
        self.index_all = torch.arange(0, self.time_total - self.sub_length)

    def no_intersection(self):
        if self.time_total >= self.sub_length * self.sample_num:
            self.init_snapshots = torch.arange(0, self.sub_length * self.sample_num, self.sub_length)
        else:
            raise RuntimeError("The number of samples no_intersection exceeds the number of data!")

    def random(self):
        pro = np.zeros(self.time_total - self.sub_length)
        pro = None
        self.init_snapshots = torch.tensor(np.random.choice(self.index_all, size=self.sample_num, replace=True, p=pro))

    def gauss(self, mu, sigma, interval_num):
        time_interval = torch.linspace(mu - 3 * sigma, mu + 3 * sigma, interval_num + 1)
        probability = np.zeros(interval_num)
        p = stats.norm(loc=mu, scale=sigma)
        for i in range(interval_num):
            probability[i] = integrate.quad(p.pdf, time_interval[i], time_interval[i + 1])[0]
        print("probability:", probability, "sum:", probability.sum())
        if self.dim == 1:
            data2 = self.data.squeeze()[self.index_all]
            snapshots = []
            for i in range(interval_num):
                index = ((time_interval[i] <= data2) & (data2 < time_interval[i + 1])).nonzero().squeeze()
                index_select = torch.tensor(np.random.choice(index, size=int(self.sample_num*probability[i]), replace=True, p=None))
                snapshots.append(index_select)
            select_num = len(snapshots)
            snapshots.append(torch.tensor(np.random.choice(
                self.index_all, size=self.sample_num-select_num, replace=True, p=None)))
            self.init_snapshots = torch.cat(snapshots, dim=0)
        else:
            raise RuntimeError("Higher dimensional Gaussian sampling method has not bee implemented!")

    def data_detach(self):
        self.new_data = torch.zeros(self.sub_length, self.sample_num, self.dim)
        for i in range(self.sample_num):
            self.new_data[:, i, :] = self.data[self.init_snapshots[i] + torch.arange(0, self.sub_length), :]

    @utils.timing
    def get_sub_data(self, sample_way):
        if sample_way == "no_intersection":
            self.no_intersection()
        elif sample_way == "random":
            self.random()
        elif sample_way == "gauss":
            self.gauss(mu=0, sigma=0.1, interval_num=30)
        else:
            raise RuntimeError("Other sampling method has not bee implemented!")
        self.data_detach()
        return self.new_data


if __name__ == '__main__':
    print(callable(Sampling))
    drift = torch.tensor([0, 1, 0, -1])
    diffusion = torch.tensor([1])
    dataset = DataSet(torch.linspace(1, 10, 10), dt=0.01, dim=1,
                      drift_term=drift, diffusion_term=diffusion, initialization=torch.rand(1) + 1,
                      sampling=Sampling, explosion_prevention=False)
    data = dataset.get_data(sample_num=4, sub_length=2, sample_type="gauss", plot_hist=True)
    print("data.size: ", data.size())
    print("data.max: ", data.max(), "data.min: ", data.min())