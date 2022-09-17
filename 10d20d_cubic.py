"""
Function: data-driven reveal SDE by the weak form of FKE: 1. linear regression 2. adversarial update
@author pi square
@email: hpp1681@gmail.com
created in Oct 11, 2021
update log:
    0. Oct 11, 2021: created
    1. Oct 15, 2021: change time index loop to Tensor form including the time index as the first dimension
    2. Oct 28, 2021: fix bugs of self.t_number -> self.bash_size
    3. Dec 9, 2021: modifies the adversarial.py to weak_gaussian_sampling for sampling in the test function
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from data.GenerateData_n import DataSet
from pyDOE import lhs
import time
import utils
import scipy.io

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# change to high dimensional saving cost version by using diagnol and repeat use functions 2022/06/01 by pi ==> speed up 10X for forward function
class Gaussian(torch.nn.Module):
    def __init__(self, mu, sigma, device):
        super(Gaussian, self).__init__()
        self.mu = mu.to(device)
        self.sigma = sigma.to(device)
        self.dim = mu.shape[0]
        self.device = device
        self.g0 = None
        self.g1 = None
        self.g2 = None
        self.funcd = None

    def gaussZero(self, x):
        func = 1
        for d in range(self.dim):
            func = func * 1 / (self.sigma * torch.sqrt(2 * torch.tensor(torch.pi))) * torch.exp(
                -0.5 * (x[:, :, d] - self.mu[d]) ** 2 / self.sigma ** 2)
        self.g0 = func
        return self.g0

    def gaussFirst(self, x):
        self.g1 = torch.zeros([x.shape[0], x.shape[1], x.shape[2]]).to(self.device)
        self.funcd = torch.zeros([x.shape[0], x.shape[1], x.shape[2]]).to(self.device)
        for k in range(self.dim):
           self.funcd[:, :, k] = -(x[:, :, k] - self.mu[k]) / self.sigma ** 2
           self.g1[:, :, k] = self.funcd[:, :, k] * self.g0
        return self.g1

    def gaussSecond(self, x):
        self.g2 = torch.zeros([x.shape[0], x.shape[1], x.shape[2]]).to(self.device)
        for k in range(self.dim):
           self.g2[:, :, k] = (-1 / self.sigma ** 2 + self.funcd[:, :, k] * self.funcd[:, :, k]) * self.g0
        return self.g2

    # @utils.timing
    def forward(self, x, diff_order=0):
        g0 = self.gaussZero(x)
        if diff_order == 0:
            return g0
        elif diff_order == 1:
            return self.gaussFirst(x)
        elif diff_order == 2:
            return self.gaussSecond(x)
        else:
            raise RuntimeError("higher order derivatives of the gaussian has not bee implemented!")

class Model(object):
    """A ``Model`` solve the true coefficients of the basis on the data by the outloop for linear regression and 
    and the inner loop of increasing the parameters in the test function TestNet.
    Args:
        t : `` t'' vector read from the file
        data: ``data`` matrix read from the file.
        testFunc: ``DNN`` instance.
    """
    def __init__(self, t, data, testFunc, device):
        self.device = device
        self.t = t.to(self.device)
        self.itmax = len(t)
        self.data = data.to(self.device)
        self.net = testFunc
        self.basis = None # given by build_basis
        self.A = None # given by build_A
        self.b = None # given by build_b
        self.dimension = None
        self.basis_number = None
        self.basis_order = None
        self.bash_size = data.shape[1]

        self.zeta = None # coefficients of the unknown function
        self.error_tolerance = None
        self.max_iter = None
        self.loss = None

        # self.batch_size = None # tbd
        # self.train_state = TrainState() # tbd
        # self.losshistory = LossHistory() # tbd


    def _get_data_t(self, it):
        X = self.data[it,:,:]
        return X
    
    @utils.timing # decorator
    @torch.no_grad()
    def build_basis(self): # \Lambda matrix
        """build the basis list for the different time snapshot 
        """
        self.basis_number = int(np.math.factorial(self.dimension+self.basis_order)
                                /(np.math.factorial(self.dimension)*np.math.factorial(self.basis_order)))
        basis = []

        for it in range(self.t_number):
            X = self._get_data_t(it)
            basis_count = 0
            Theta = torch.zeros(X.size(0),self.basis_number)
            Theta[:,0] = 1
            basis_count += 1
            for ii in range(0,self.dimension):
                Theta[:,basis_count] = X[:,ii]
                basis_count += 1

            if self.basis_order >= 2:
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        Theta[:,basis_count] = torch.mul(X[:,ii],X[:,jj])
                        basis_count += 1

            if self.basis_order >= 3:
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        for kk in range(jj,self.dimension):
                            Theta[:,basis_count] = torch.mul(torch.mul(X[:,ii],
                                X[:,jj]),X[:,kk])
                            basis_count += 1

            if self.basis_order >= 4:
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        for kk in range(jj,self.dimension):
                            for ll in range(kk,self.dimension):
                                Theta[:,basis_count] = torch.mul(torch.mul(torch.mul(X[:,ii],
                                    X[:,jj]),X[:,kk]),X[:,ll])
                                basis_count += 1

            if self.basis_order >= 5:
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        for kk in range(jj,self.dimension):
                            for ll in range(kk,self.dimension):
                                for mm in range(ll,self.dimension):
                                    Theta[:,basis_count] = torch.mul(torch.mul(torch.mul(torch.mul(
                                        X[:,ii],X[:,jj]),X[:,kk]),
                                            X[:,ll]),X[:,mm])
                                    basis_count += 1
            assert basis_count == self.basis_number
            basis.append(Theta)
            # print("X", X)
            # print("theta", Theta)

        self.basis = torch.stack(basis).to(self.device)
        # print("self.basis.shape", self.basis.shape)

    def computeAb(self, gauss):
        H_number = self.dimension * self.basis_number
        F_number = self.dimension if self.diffusion_independence else 1

        A = torch.zeros([self.t_number, H_number+F_number]).to(self.device)

        # ##########################################################
        #  Tensor form of computing A and b for parallel computing
        # ##########################################################

        TX = self.data
        TX.requires_grad = True
        # Phi = self.net(TX)
        gauss0 = gauss(TX, diff_order=0)
        gauss1 = gauss(TX, diff_order=1)
        gauss2 = gauss(TX, diff_order=2)
        # print("gauss0: ", gauss0.device)
        # print("gauss1: ", gauss1.device)
        # print("gauss2: ", gauss2.device)
        # print("self.data: ", self.data.device)

        # print("self.basis_number", self.basis_number)
        # print("self.dimension", self.dimension)
        for kd in range(self.dimension):
            for jb in range(self.basis_number):
                if self.drift_independence:
                    H = torch.mean(gauss1[:, :, kd] * self.data[:, :, kd] ** jb, dim=1)
                else:                    
                    H = torch.mean(gauss1[:, :, kd] * self.basis[:, :, jb], dim=1)
                A[:, kd*self.basis_number+jb] = H

        # compute A by F_lkj
        if self.diffusion_independence:
            for ld in range(self.dimension):
                F = torch.mean(gauss2[:, :, ld], dim=1)
                A[:, H_number+ld] = F
        else:
            F = np.sum([torch.mean(gauss2[:, :, i], dim=1) for i in range(self.dimension)])
            A[:, H_number] = F
        rb = 1/self.bash_size * torch.sum(gauss0, dim=1).squeeze()
        dt = (torch.max(self.t)-torch.min(self.t)) / (self.t_number - 1)
        # print("b", rb)

        # b = torch.tensor(torch.enable_grad()(utils.compute_b)(rb, dt, time_diff='Tik'))
        # print("b.shape", b.shape)

        # plt.clf()
        # plt.plot(rb.detach().numpy(),'-*')
        # plt.plot(b.detach().numpy(),'-o')
        # plt.draw()
        # plt.pause(1)

        # print("b", b)
        # print("A.shape", A.shape)
        if self.type == 'PDEFind':
            b = torch.tensor(torch.enable_grad()(utils.compute_b)(rb, dt, time_diff='Tik'))
            return A, b
        if self.type == 'LMM_2':
            AA = torch.ones(A.size(0) - 1, A.size(1)).to(self.device)
            bb = torch.ones(A.size(0) - 1).to(self.device)
            for i in range(AA.size(0)):
                AA[i, :] = (A[i, :] + A[i + 1, :]) * dt / 2
                bb[i] = rb[i + 1] - rb[i]
            return AA, bb
        if self.type == 'LMM_3':
            AA = torch.ones(A.size(0) - 2, A.size(1)).to(self.device)
            bb = torch.ones(A.size(0) - 2).to(self.device)
            for i in range(AA.size(0)):
                AA[i, :] = (A[i, :] + 4*A[i + 1, :] + A[i + 2, :]) * dt / 3
                bb[i] = rb[i + 2] - rb[i]
            return AA, bb
        if self.type == 'LMM_6':
            AA = torch.ones(A.size(0) - 5, A.size(1)).to(self.device)
            bb = torch.ones(A.size(0) - 5).to(self.device)
            for i in range(AA.size(0)):
                AA[i, :] = (
                        A[i+1, :] +
                        1/2 * (A[i+2, :] + A[i+1, :]) +
                        5/12 * A[i+3, :] + 8/12 * A[i+2, :] - 1/12 * A[i+1, :] +
                        9/24 * A[i+4, :] + 19/24 * A[i+3, :] - 5/24 * A[i+2, :] + 1/24 * A[i+1, :] +
                        251/720 * A[i + 5, :] + 646/720 * A[i + 4, :] - 264/720 * A[i + 3, :] + 106/720 * A[i + 2, :] - 19/720 * A[i + 1, :]
                            ) * dt
                bb[i] = rb[i + 5] - rb[i]
            return AA, bb

    def sampleTestFunc(self, samp_number):
        # for i in range(self.sampling_number):
        if self.gauss_samp_way == 'lhs':
            lb = torch.tensor([self.data[:, :, i].min() for i in range(self.dimension)]).to(self.device)
            ub = torch.tensor([self.data[:, :, i].max() for i in range(self.dimension)]).to(self.device)
            mu_list = lb + self.lhs_ratio * (ub - lb) * torch.tensor(lhs(self.dimension, samp_number), dtype=torch.float32).to(self.device)
        if self.gauss_samp_way == 'SDE':
            if samp_number <= self.bash_size:
                index = np.arange(self.bash_size)
                np.random.shuffle(index)
                mu_list = data[-1, index[0: samp_number], :]
            else:
                print("The number of samples shall not be less than the number of tracks!")
        print("mu_list", mu_list.shape)
        sigma_list = torch.ones(samp_number).to(self.device)*self.variance
        print("sigma_list", sigma_list.shape)
        return mu_list, sigma_list

    @utils.timing
    def buildLinearSystem(self, samp_number):
        mu_list, sigma_list = self.sampleTestFunc(samp_number)
        # print("mu_list: ", mu_list.device)
        # print("sigma_list: ", sigma_list.device)
        A_list = []
        b_list = []
        for i in range(mu_list.shape[0]):
            if i % 20 == 0:
                print('buildLinearSystem:', i)
            mu = mu_list[i]
            sigma = sigma_list[i]
            gauss = self.net(mu, sigma, self.device)
            A, b = self.computeAb(gauss)
            A_list.append(A)
            b_list.append(b)
        # print("A_list", A_list)
        # print("b_list", b_list)
        self.A = torch.cat(A_list, dim=0) # 2-dimension
        self.b = torch.cat(b_list, dim=0).unsqueeze(-1) # 1-dimension


    @torch.no_grad()
    def solveLinearRegress(self):
        self.zeta = torch.tensor(np.linalg.lstsq(self.A.detach().numpy(), self.b.detach().numpy())[0])
        # TBD sparse regression

    def STRidge(self, X0, y, lam, maxit, tol, normalize=0, print_results = False):
        """
        Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
        approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

        This assumes y is only one column
        """
        n,d = X0.shape
        X = np.zeros((n,d), dtype=np.complex64)
        # First normalize data
        if normalize != 0:
            Mreg = np.zeros((d,1))
            for i in range(0,d):
                Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
                X[:,i] = Mreg[i]*X0[:,i]
        else: X = X0

        # Get the standard ridge esitmate
        if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y))[0]
        else: w = np.linalg.lstsq(X,y)[0]
        num_relevant = d
        biginds = np.where(abs(w) > tol)[0]

        # Threshold and continue
        for j in range(maxit):
            # Figure out which items to cut out
            smallinds = np.where(abs(w) < tol)[0]
            print("STRidge_j: ", j)
            print("smallinds", smallinds)
            new_biginds = [i for i in range(d) if i not in smallinds]

            # If nothing changes then stop
            if num_relevant == len(new_biginds):
                print("here1")
                break
            else: num_relevant = len(new_biginds)

            # Also make sure we didn't just lose all the coefficients
            if len(new_biginds) == 0:
                if j == 0:
                    print("here2")
                    #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                    return w
                else:
                    print("here3")
                    break
            biginds = new_biginds

            # Otherwise get a new guess
            w[smallinds] = 0
            if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]
            else: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

        # Now that we have the sparsity pattern, use standard least squares to get w
        if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

        if normalize != 0: return np.multiply(Mreg,w)
        else: return w
    
    # @utils.timing
    def compile(self, basis_order, gauss_variance, type, drift_term, diffusion_term,
                drift_independence, diffusion_independence, gauss_samp_way, lhs_ratio):
        self.dimension = self.data.shape[-1]
        self.t_number = len(self.t)
        self.basis_order = basis_order
        self.variance = gauss_variance
        self.type = type
        self.drift = drift_term
        self.diffusion = diffusion_term
        self.drift_independence = drift_independence
        self.diffusion_independence = diffusion_independence
        if self.drift_independence:
            self.basis_number = self.basis_order + 1
        else:
            self.build_basis()
        self.gauss_samp_way = gauss_samp_way
        self.lhs_ratio = lhs_ratio if self.gauss_samp_way == 'lhs' else 1

    @utils.timing
    @torch.no_grad()
    def train(self, gauss_samp_number, lam, STRidge_threshold, only_hat_13=False):
        self.buildLinearSystem(samp_number=gauss_samp_number)
        if only_hat_13:
            I = torch.tensor([1, 2, 6, 7, 8, 9, 11, 12, 16, 17, 18, 19, 20, 21])
            self.A = self.A[:, I]
        self.A = self.A.to("cpu")
        self.b = self.b.to("cpu")
        AA = torch.mm(torch.t(self.A), self.A)
        Ab = torch.mm(torch.t(self.A), self.b)
        print("A.max: ", self.A.max(), "b.max: ", self.b.max())
        print("ATA.max: ", AA.max(), "ATb.max: ", Ab.max())
        self.zeta = torch.tensor(self.STRidge(self.A.detach().numpy(), self.b.detach().numpy(), lam, 100, STRidge_threshold)).to(torch.float)
        # print("zeta: ", self.zeta.size(), self.zeta)

        if self.drift_independence:
            for i in range(self.dimension):
                drift = [self.zeta[i*self.basis_number].numpy()]
                for j in range(self.basis_number - 1):
                    drift.extend([" + ", self.zeta[i*self.basis_number + j + 1].numpy(), 'x_', i + 1, '^', j + 1])
                print("Drift term: ", "".join([str(_) for _ in drift]))
        else:
            if only_hat_13 :
                self.basis_number = 6
            for i in range(self.dimension):
                drift = [self.zeta[i*self.basis_number].numpy()]
                for j in range(self.basis_number - 1):
                    drift.extend([" + ", self.zeta[i*self.basis_number + j + 1].numpy(), 'x_', j + 1])
                print("Drift term : ", i+1, "".join([str(_) for _ in drift]))
        if self.diffusion_independence:
            self.zeta.squeeze()[self.dimension * self.basis_number:] = torch.sqrt(2*self.zeta.squeeze()[self.dimension * self.basis_number:])
            print("Diffusion term: ", "diag ", self.zeta.squeeze()[self.dimension*self.basis_number:].numpy())
        else:
            print("Diffusion term: ", np.sqrt(2*self.zeta.squeeze()[self.dimension * self.basis_number].numpy()))

        true = torch.cat((self.drift.view(-1), self.diffusion))
        if only_hat_13:
            true = true[I]
        index = torch.nonzero(true).squeeze()
        relative_error = torch.abs((self.zeta.squeeze()[index] - true[index]) / true[index])
        print("Maximum relative error: ", relative_error.max().numpy())
        print("Maximum index: ", torch.argmax(relative_error))


if __name__ == '__main__':
    np.random.seed(100)
    torch.manual_seed(100)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # device = torch.device('mps')
    print("device: ", device)

    T, dt, true_dt = 1, 0.001, 0.1
    t = np.linspace(0, T, int(T/dt) + 1).astype(np.float32)
    t = torch.tensor(t)
    dim = 10
    # data = scipy.io.loadmat('./data/data3d.mat')['bb'].astype(np.float32)
    # data = torch.tensor(data)
    # drift = torch.tensor([[0, -0.5, 0, 0], [0, 0, -0.7, 0], [0, 0, 0, -1]])
    # drift = torch.cat((torch.zeros(10, 1), torch.diag(torch.arange(-0.1, -1.1, -0.1))), dim=1)
    # drift = torch.tensor([[0, 1, 0, -1], [0, 1, 0, -1.3], [0, 1, 0, -1], [0, 1, 0, -1], [0, 1, 0, -1]])
    drift = torch.tensor([0, 1, 0, -1]).repeat(dim, 1)
    # drift = torch.tensor([[0, 10, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, -4, 0, -4, 0, 0, 0, 0],
    #                          [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, -4, 0, -4, 0],
    #                          [0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, -4, 0, -4]])
    # drift = torch.tensor([[0, 10, 0, 0, 0, 0, -4, 0, -4, 0], [0, 0, 10, 0, 0, 0, 0, -4, 0, -4]])
    # diffusion = torch.tensor([1, 1])
    diffusion = torch.ones(dim)
    sample, dim = 100000, dim
    dataset = DataSet(t, true_dt=true_dt, samples_num=sample, dim=dim, drift_term=drift, diffusion_term=diffusion,
                      initialization=torch.normal(mean=0., std=0.1, size=(sample, dim)),
                      drift_independence=True, explosion_prevention=False, trajectory_information=True)
    data = dataset.get_data(plot_hist=False)
    data = data * (1 + 0.*torch.rand(data.shape))
    torch.save(data, "data.pt")
    data = torch.load("data.pt")
    print("data: ", data.shape, data.max(), data.min())
    # print(data[0, 1, :])

    testFunc = Gaussian
    model = Model(torch.linspace(0, T, int(T/true_dt) + 1), data, testFunc, device)
    model.compile(basis_order=3, gauss_variance=1, type='LMM_3', drift_term=drift, diffusion_term=diffusion,
                  drift_independence=True, diffusion_independence=True, gauss_samp_way='lhs', lhs_ratio=1)
    # For 1 dimension, "drift_independence" and "diffusion_independence"  make no difference
    model.train(gauss_samp_number=10000, lam=0.0, STRidge_threshold=0.1, only_hat_13=False)



# dim = 10, 100000, 2000
# Drift term:  [0.] + [1.0828791]x_1^1 + [0.]x_1^2 + [-1.0689465]x_1^3
# Drift term:  [0.] + [0.9765123]x_2^1 + [0.]x_2^2 + [-0.9755056]x_2^3
# Drift term:  [0.] + [0.9529692]x_3^1 + [0.]x_3^2 + [-0.970244]x_3^3
# Drift term:  [0.] + [1.1462214]x_4^1 + [0.]x_4^2 + [-1.1299471]x_4^3
# Drift term:  [0.] + [0.98774177]x_5^1 + [0.]x_5^2 + [-0.98394936]x_5^3
# Drift term:  [0.] + [1.0351396]x_6^1 + [0.]x_6^2 + [-1.0144199]x_6^3
# Drift term:  [0.] + [1.0700953]x_7^1 + [0.]x_7^2 + [-1.0756891]x_7^3
# Drift term:  [0.] + [0.91275865]x_8^1 + [0.]x_8^2 + [-0.91309655]x_8^3
# Drift term:  [0.] + [1.0291636]x_9^1 + [0.]x_9^2 + [-1.0209548]x_9^3
# Drift term:  [0.] + [1.0673567]x_10^1 + [0.]x_10^2 + [-1.0448446]x_10^3
# Diffusion term:  diag  [0.9977212  0.9983615  1.0075259  0.99480945 0.99786764 0.99413294
#  1.0011218  1.008092   0.9972691  0.99181   ]
# Maximum relative error:  0.1462214
# Maximum index:  tensor(6)
# 'train' took 231.522776 s

# dim = 10, 10000, 1000
# Drift term:  [0.] + [1.0079209]x_1^1 + [0.]x_1^2 + [-1.000232]x_1^3
# Drift term:  [0.] + [1.0252012]x_2^1 + [0.]x_2^2 + [-1.0233462]x_2^3
# Drift term:  [0.] + [0.8816458]x_3^1 + [0.]x_3^2 + [-0.9016202]x_3^3
# Drift term:  [0.] + [0.96515214]x_4^1 + [0.]x_4^2 + [-0.9632371]x_4^3
# Drift term:  [0.] + [0.74840844]x_5^1 + [0.]x_5^2 + [-0.8436009]x_5^3
# Drift term:  [0.] + [1.2134445]x_6^1 + [0.]x_6^2 + [-1.1232896]x_6^3
# Drift term:  [0.] + [1.3809309]x_7^1 + [0.]x_7^2 + [-1.3047208]x_7^3
# Drift term:  [0.] + [0.9546177]x_8^1 + [0.]x_8^2 + [-0.9887492]x_8^3
# Drift term:  [0.] + [0.7454044]x_9^1 + [0.]x_9^2 + [-0.81488854]x_9^3
# Drift term:  [0.] + [0.6563197]x_10^1 + [0.]x_10^2 + [-0.6862587]x_10^3
# Diffusion term:  diag  [0.9744072  1.0064926  1.0208662  0.9929923  1.0244726  0.9553542
#  0.95008284 1.0075322  1.0374355  1.0406317 ]
# Maximum relative error:  0.3809309
# Maximum index:  tensor(12)
# 'train' took 19.214371 s


# dim = 10, 10000, 10000
# Drift term:  [0.] + [1.1443638]x_1^1 + [0.]x_1^2 + [-1.1268948]x_1^3
# Drift term:  [0.] + [1.1643473]x_2^1 + [0.]x_2^2 + [-1.1263916]x_2^3
# Drift term:  [0.] + [0.9344389]x_3^1 + [0.]x_3^2 + [-0.9386132]x_3^3
# Drift term:  [0.] + [0.8267552]x_4^1 + [0.]x_4^2 + [-0.90594566]x_4^3
# Drift term:  [0.] + [0.9149755]x_5^1 + [0.]x_5^2 + [-0.94768333]x_5^3
# Drift term:  [0.] + [1.1315198]x_6^1 + [0.]x_6^2 + [-1.0775677]x_6^3
# Drift term:  [0.] + [0.9819692]x_7^1 + [0.]x_7^2 + [-0.9610379]x_7^3
# Drift term:  [0.] + [0.9920275]x_8^1 + [0.]x_8^2 + [-1.0032072]x_8^3
# Drift term:  [0.] + [0.68968487]x_9^1 + [0.]x_9^2 + [-0.7570637]x_9^3
# Drift term:  [0.] + [0.9806246]x_10^1 + [0.]x_10^2 + [-0.9611879]x_10^3
# Diffusion term:  diag  [0.97895396 0.97214997 1.0042064  1.0277948  1.0039356  0.9758804
#  0.9826347  1.0077187  1.0442824  1.0036548 ]
# Maximum relative error:  0.31031513
# Maximum index:  tensor(16)
# 'train' took 187.840324 s

# dim = 10, 100000, 10000
# Drift term:  [0.] + [1.0369135]x_1^1 + [0.]x_1^2 + [-1.0247759]x_1^3
# Drift term:  [0.] + [1.0727105]x_2^1 + [0.]x_2^2 + [-1.0780263]x_2^3
# Drift term:  [0.] + [1.0402976]x_3^1 + [0.]x_3^2 + [-1.0394964]x_3^3
# Drift term:  [0.] + [1.0602895]x_4^1 + [0.]x_4^2 + [-1.0437388]x_4^3
# Drift term:  [0.] + [1.0530245]x_5^1 + [0.]x_5^2 + [-1.0399375]x_5^3
# Drift term:  [0.] + [1.0284597]x_6^1 + [0.]x_6^2 + [-1.0125804]x_6^3
# Drift term:  [0.] + [1.0170561]x_7^1 + [0.]x_7^2 + [-1.0193262]x_7^3
# Drift term:  [0.] + [0.9392866]x_8^1 + [0.]x_8^2 + [-0.92351246]x_8^3
# Drift term:  [0.] + [1.0583261]x_9^1 + [0.]x_9^2 + [-1.0407009]x_9^3
# Drift term:  [0.] + [1.0220228]x_10^1 + [0.]x_10^2 + [-1.0104069]x_10^3
# Diffusion term:  diag  [0.998788   0.99694383 0.9957617  0.99700874 0.99772525 0.99702525
#  0.99982816 0.999082   0.9928979  0.99679804]
# Maximum relative error:  0.078026295
# Maximum index:  tensor(3)
# 'train' took 1123.643677 s

# dim = 20, 100000, 10000
# Drift term:  [0.] + [1.2134612]x_1^1 + [0.]x_1^2 + [-1.1048486]x_1^3
# Drift term:  [0.] + [1.0830477]x_2^1 + [0.]x_2^2 + [-1.0934552]x_2^3
# Drift term:  [0.] + [1.0417438]x_3^1 + [0.]x_3^2 + [-1.0402415]x_3^3
# Drift term:  [0.] + [1.1524194]x_4^1 + [0.]x_4^2 + [-1.1420962]x_4^3
# Drift term:  [0.] + [0.7333361]x_5^1 + [0.]x_5^2 + [-0.78140736]x_5^3
# Drift term:  [0.] + [1.5205417]x_6^1 + [0.]x_6^2 + [-1.5444709]x_6^3
# Drift term:  [0.] + [1.3111937]x_7^1 + [0.]x_7^2 + [-1.3373351]x_7^3
# Drift term:  [0.] + [0.9338893]x_8^1 + [0.]x_8^2 + [-0.8867003]x_8^3
# Drift term:  [0.] + [1.195055]x_9^1 + [0.]x_9^2 + [-1.1412553]x_9^3
# Drift term:  [0.] + [0.93111134]x_10^1 + [0.]x_10^2 + [-0.8301321]x_10^3
# Drift term:  [0.] + [1.1561071]x_11^1 + [0.]x_11^2 + [-1.0148734]x_11^3
# Drift term:  [0.] + [1.0763218]x_12^1 + [0.]x_12^2 + [-1.0753682]x_12^3
# Drift term:  [0.] + [0.8033595]x_13^1 + [0.]x_13^2 + [-0.8654067]x_13^3
# Drift term:  [0.] + [1.0027046]x_14^1 + [0.]x_14^2 + [-1.0330116]x_14^3
# Drift term:  [0.] + [0.9699889]x_15^1 + [0.]x_15^2 + [-0.9049834]x_15^3
# Drift term:  [0.] + [1.024053]x_16^1 + [0.]x_16^2 + [-0.9607201]x_16^3
# Drift term:  [0.] + [1.210988]x_17^1 + [0.]x_17^2 + [-1.1655047]x_17^3
# Drift term:  [0.] + [1.2895465]x_18^1 + [0.]x_18^2 + [-1.2354982]x_18^3
# Drift term:  [0.] + [1.4205891]x_19^1 + [0.]x_19^2 + [-1.3055811]x_19^3
# Drift term:  [0.] + [0.95513755]x_20^1 + [0.]x_20^2 + [-1.0227251]x_20^3
# Diffusion term:  diag  [0.9599684  1.0064831  1.0111227  0.9811345  1.0449649  0.97538584
#  0.98331517 0.99163127 0.991372   0.9811756  0.9582765  0.9927545
#  1.0366073  1.0034341  0.99205405 0.99053127 0.98531026 1.0015638
#  0.9543664  1.0219405 ]
# Maximum relative error:  0.5444709
# Maximum index:  tensor(11)
# 'train' took 2770.313326 s

# dim = 20, 100000, 100000
# Drift term:  [0.] + [1.0575641]x_1^1 + [0.]x_1^2 + [-1.0534418]x_1^3
# Drift term:  [0.] + [1.1948632]x_2^1 + [0.]x_2^2 + [-1.18301]x_2^3
# Drift term:  [0.] + [1.2112317]x_3^1 + [0.]x_3^2 + [-1.1958923]x_3^3
# Drift term:  [0.] + [0.77928126]x_4^1 + [0.]x_4^2 + [-0.77646714]x_4^3
# Drift term:  [0.] + [1.3589128]x_5^1 + [0.]x_5^2 + [-1.3185179]x_5^3
# Drift term:  [0.] + [1.1171635]x_6^1 + [0.]x_6^2 + [-1.098339]x_6^3
# Drift term:  [0.] + [0.73656356]x_7^1 + [0.]x_7^2 + [-0.75131166]x_7^3
# Drift term:  [0.] + [0.9826067]x_8^1 + [0.]x_8^2 + [-0.99235743]x_8^3
# Drift term:  [0.] + [1.1341841]x_9^1 + [0.]x_9^2 + [-1.1104031]x_9^3
# Drift term:  [0.] + [0.8545066]x_10^1 + [0.]x_10^2 + [-0.86164504]x_10^3
# Drift term:  [0.] + [1.0481594]x_11^1 + [0.]x_11^2 + [-1.0190942]x_11^3
# Drift term:  [0.] + [0.95597464]x_12^1 + [0.]x_12^2 + [-0.9506377]x_12^3
# Drift term:  [0.] + [1.3912469]x_13^1 + [0.]x_13^2 + [-1.3281534]x_13^3
# Drift term:  [0.] + [1.1105962]x_14^1 + [0.]x_14^2 + [-1.095604]x_14^3
# Drift term:  [0.] + [0.8180783]x_15^1 + [0.]x_15^2 + [-0.8169279]x_15^3
# Drift term:  [0.] + [0.8790658]x_16^1 + [0.]x_16^2 + [-0.8746517]x_16^3
# Drift term:  [0.] + [0.9205012]x_17^1 + [0.]x_17^2 + [-0.9209385]x_17^3
# Drift term:  [0.] + [1.0218757]x_18^1 + [0.]x_18^2 + [-1.0272515]x_18^3
# Drift term:  [0.] + [1.0702751]x_19^1 + [0.]x_19^2 + [-1.0546889]x_19^3
# Drift term:  [0.] + [0.9742499]x_20^1 + [0.]x_20^2 + [-0.94962674]x_20^3
# Diffusion term:  diag  [0.9891241 0.9881352 0.9832464 1.015648  0.9702128 0.9947744 1.0112026
#  1.0063944 0.9850211 1.0125747 0.9972094 1.0064305 0.9617753 0.9979025
#  1.0205034 1.0045714 1.0084716 1.0074805 0.9841694 0.9976742]
# Maximum relative error:  0.3912469
# Maximum index:  tensor(24)
# 'train' took 12718.607842 s

# 20, 500000, 10000
# Drift term:  [0.] + [1.1993715]x_1^1 + [0.]x_1^2 + [-1.2104814]x_1^3
# Drift term:  [0.] + [0.8743425]x_2^1 + [0.]x_2^2 + [-0.8868175]x_2^3
# Drift term:  [0.] + [1.0212557]x_3^1 + [0.]x_3^2 + [-1.0471622]x_3^3
# Drift term:  [0.] + [0.80469376]x_4^1 + [0.]x_4^2 + [-0.8655785]x_4^3
# Drift term:  [0.] + [1.10553]x_5^1 + [0.]x_5^2 + [-1.0859576]x_5^3
# Drift term:  [0.] + [0.692159]x_6^1 + [0.]x_6^2 + [-0.748522]x_6^3
# Drift term:  [0.] + [1.1067381]x_7^1 + [0.]x_7^2 + [-1.1030906]x_7^3
# Drift term:  [0.] + [1.2772129]x_8^1 + [0.]x_8^2 + [-1.2017493]x_8^3
# Drift term:  [0.] + [1.1858177]x_9^1 + [0.]x_9^2 + [-1.1674691]x_9^3
# Drift term:  [0.] + [0.8462725]x_10^1 + [0.]x_10^2 + [-0.8513291]x_10^3
# Drift term:  [0.] + [0.68655664]x_11^1 + [0.]x_11^2 + [-0.64444005]x_11^3
# Drift term:  [0.] + [0.8877618]x_12^1 + [0.]x_12^2 + [-0.9304214]x_12^3
# Drift term:  [0.] + [1.2451679]x_13^1 + [0.]x_13^2 + [-1.2089671]x_13^3
# Drift term:  [0.] + [0.9266026]x_14^1 + [0.]x_14^2 + [-0.9552419]x_14^3
# Drift term:  [0.] + [1.2502381]x_15^1 + [0.]x_15^2 + [-1.2274537]x_15^3
# Drift term:  [0.] + [1.1378006]x_16^1 + [0.]x_16^2 + [-1.0874512]x_16^3
# Drift term:  [0.] + [1.0856326]x_17^1 + [0.]x_17^2 + [-1.0614915]x_17^3
# Drift term:  [0.] + [0.68631583]x_18^1 + [0.]x_18^2 + [-0.71467113]x_18^3
# Drift term:  [0.] + [1.1776267]x_19^1 + [0.]x_19^2 + [-1.1402587]x_19^3
# Drift term:  [0.] + [1.1826924]x_20^1 + [0.]x_20^2 + [-1.1952404]x_20^3
# Diffusion term:  diag  [0.98494947 1.0137924  0.99926704 1.0299505  0.98770696 1.0302154
#  0.9957345  0.9742521  0.9804623  1.0056248  1.0145321  1.0150443
#  0.9885298  1.0120976  0.9758958  0.9829955  0.986879   1.0338583
#  0.97371036 0.99457765]
# Maximum relative error:  0.35555995
# Maximum index:  tensor(21)
# 'train' took 5731.549085 s