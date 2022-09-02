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


class DataSet(object):
    def __init__(self, time_instants, dt, true_dt, samples_num, dim, drift_term, diffusion_term,
                 initialization, explosion_prevention=False, trajectory_information=True):
        self.time_instants = time_instants
        self.dt = dt
        self.true_dt = true_dt
        self.trajectory_information = trajectory_information
        if self.trajectory_information:
            self.samples_num = samples_num
        else:
            self.samples_num_true, self.samples_num = samples_num
        self.dim = dim
        self.drift_term = drift_term
        self.diffusion_term = diffusion_term
        self.initialization = initialization
        self.explosion_prevention = explosion_prevention

        self.shape_t = self.time_instants.shape[0]
        self.t_diff = torch.from_numpy(np.diff(self.time_instants.numpy()))
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

    @utils.timing
    def get_data(self, plot_hist=False):
        data = torch.zeros(self.shape_t, self.samples_num, self.dim)
        data[0, :, :] = self.initialization   # torch.randn(self.samples_num, self.dim) + 1     # initial condition
        for i in range(self.shape_t - 1):
            data[i + 1, :, :] = data[i, :, :] + self.drift(data[i, :, :]) * self.t_diff[i]
            data[i + 1, :, :] = data[i + 1, :, :] + self.diffusion(data[i, :, :]) * torch.sqrt(self.t_diff[i]) * torch.randn(self.samples_num, self.dim)
            if self.explosion_prevention:
                if any(data[i + 1, :, :] < 0):
                    data[i + 1, :, :][data[i + 1, :, :] < 0] = 0
                    self.explosion_prevention_N = self.explosion_prevention_N + 1
        print("explosion_prevention * %s" % self.explosion_prevention_N)
        if plot_hist:
            for i in range(self.dim):
                plt.figure()
                plt.hist(x=data[-1, :, i].numpy(), bins=100, range=[data.min().numpy(), data.max().numpy()], density=True)
            plt.show()
        index0 = np.arange(0, self.shape_t, int(self.true_dt // self.t_diff[0]))
        data0 = data[index0, :, :]
        if self.trajectory_information:
            return data0
        else:
            data1 = torch.zeros(data0.shape[0], self.samples_num_true, data0.shape[2])
            for i in range(data0.shape[0]):
                index1 = np.arange(self.samples_num)
                np.random.shuffle(index1)
                data1[i, :, :] = data0[i, index1[0: self.samples_num_true], :]
            return data1


if __name__ == '__main__':
    # drift = torch.tensor([0, -24, 50, -35, 10, -1])
    # diffusion = torch.tensor([1, 0, 0, 0, 0, 0])
    drift = torch.tensor([0, 1, 0, -1])
    diffusion = torch.tensor([1, 0, 0, 0])
    dataset = DataSet(torch.linspace(0, 10, 10001), dt=0.001, true_dt=0.1, samples_num=[2000, 5000], dim=1,
                      drift_term=drift, diffusion_term=diffusion, initialization=torch.rand(5000, 1) + 1,
                      explosion_prevention=False, trajectory_information=True)
    data = dataset.get_data()
    print(data.size())
    print(data.max(), data.min())