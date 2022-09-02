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
    @utils.timing
    def solveLinearRegress(self):
        self.zeta = torch.tensor(np.linalg.lstsq(self.A.detach().numpy(), self.b.detach().numpy())[0])
        # TBD sparse regression

    @utils.timing
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
    dim = 100
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
    # dataset = DataSet(t, true_dt=true_dt, samples_num=sample, dim=dim, drift_term=drift, diffusion_term=diffusion,
    #                   initialization=torch.rand(sample, dim) + 1,   # torch.normal(mean=0., std=0.1, size=(sample, dim)),
    #                   drift_independence=True, explosion_prevention=False, trajectory_information=True)
    dataset = DataSet(t, true_dt=true_dt, samples_num=sample, dim=dim, drift_term=drift, diffusion_term=diffusion,
                      initialization=torch.normal(mean=0., std=0.2, size=(sample, dim)),   # torch.normal(mean=0., std=0.1, size=(sample, dim)),
                      drift_independence=True, explosion_prevention=False, trajectory_information=True)
    data = dataset.get_data(plot_hist=False)
    data = data * (1 + 0.*torch.rand(data.shape))
    print("data: ", data.shape, data.max(), data.min())

    testFunc = Gaussian
    model = Model(torch.linspace(0, T, int(T/true_dt) + 1), data, testFunc, device)
    model.compile(basis_order=3, gauss_variance=1, type='LMM_3', drift_term=drift, diffusion_term=diffusion,
                  drift_independence=True, diffusion_independence=True, gauss_samp_way='lhs', lhs_ratio=1)
    # For 1 dimension, "drift_independence" and "diffusion_independence"  make no difference
    model.train(gauss_samp_number=100, lam=0.0, STRidge_threshold=0.2, only_hat_13=False)




# dim = 10, 10000, 1000
# Drift term:  [0.] + [1.0595657]x_1^1 + [0.]x_1^2 + [-1.0308417]x_1^3
# Drift term:  [0.] + [0.94064754]x_2^1 + [0.]x_2^2 + [-0.94613767]x_2^3
# Drift term:  [0.] + [0.680325]x_3^1 + [0.]x_3^2 + [-0.715011]x_3^3
# Drift term:  [0.] + [1.1368537]x_4^1 + [0.]x_4^2 + [-1.123517]x_4^3
# Drift term:  [0.] + [0.7537237]x_5^1 + [0.]x_5^2 + [-0.82927203]x_5^3
# Drift term:  [0.] + [0.98286986]x_6^1 + [0.]x_6^2 + [-0.94031614]x_6^3
# Drift term:  [0.] + [1.5521599]x_7^1 + [0.]x_7^2 + [-1.4487752]x_7^3
# Drift term:  [0.] + [0.8920566]x_8^1 + [0.]x_8^2 + [-0.9148894]x_8^3
# Drift term:  [0.] + [0.8919276]x_9^1 + [0.]x_9^2 + [-0.9378]x_9^3
# Drift term:  [0.] + [0.9295914]x_10^1 + [0.]x_10^2 + [-0.9204409]x_10^3
# Diffusion term:  diag  [0.9665317  1.0158957  1.0402669  0.9803336  1.0216801  0.9881562
#  0.92535764 1.0107666  1.0229598  1.0090044 ]
# Maximum relative error:  0.5521599
# Maximum index:  tensor(12)
# 'train' took 19.548051 s


# dim = 10, 10000, 10000
# Drift term:  [0.] + [1.0715154]x_1^1 + [0.]x_1^2 + [-1.0593119]x_1^3
# Drift term:  [0.] + [1.1940501]x_2^1 + [0.]x_2^2 + [-1.1478386]x_2^3
# Drift term:  [0.] + [0.8946942]x_3^1 + [0.]x_3^2 + [-0.9029954]x_3^3
# Drift term:  [0.] + [0.8871966]x_4^1 + [0.]x_4^2 + [-0.94989216]x_4^3
# Drift term:  [0.] + [0.98494464]x_5^1 + [0.]x_5^2 + [-1.0030527]x_5^3
# Drift term:  [0.] + [1.1603328]x_6^1 + [0.]x_6^2 + [-1.1206105]x_6^3
# Drift term:  [0.] + [1.1268816]x_7^1 + [0.]x_7^2 + [-1.0933751]x_7^3
# Drift term:  [0.] + [0.9385868]x_8^1 + [0.]x_8^2 + [-0.9471763]x_8^3
# Drift term:  [0.] + [0.6427385]x_9^1 + [0.]x_9^2 + [-0.70372826]x_9^3
# Drift term:  [0.] + [1.007449]x_10^1 + [0.]x_10^2 + [-0.9770405]x_10^3
# Diffusion term:  diag  [0.9833021  0.9724209  1.0079197  1.0211995  0.99600154 0.9799199
#  0.9711033  1.0105146  1.0458313  0.9957073 ]
# Maximum relative error:  0.35726148
# Maximum index:  tensor(16)
# 'train' took 185.248455 s

# dim = 10, 100000, 1000
# Drift term:  [0.] + [0.88438815]x_1^1 + [0.]x_1^2 + [-0.8947174]x_1^3
# Drift term:  [0.] + [1.0002885]x_2^1 + [0.]x_2^2 + [-1.0042301]x_2^3
# Drift term:  [0.] + [0.93102485]x_3^1 + [0.]x_3^2 + [-0.95854807]x_3^3
# Drift term:  [0.] + [0.95046175]x_4^1 + [0.]x_4^2 + [-0.9380948]x_4^3
# Drift term:  [0.] + [1.0931412]x_5^1 + [0.]x_5^2 + [-1.0769056]x_5^3
# Drift term:  [0.] + [1.0819291]x_6^1 + [0.]x_6^2 + [-1.0562521]x_6^3
# Drift term:  [0.] + [0.9445285]x_7^1 + [0.]x_7^2 + [-0.9396141]x_7^3
# Drift term:  [0.] + [1.0426557]x_8^1 + [0.]x_8^2 + [-1.0221556]x_8^3
# Drift term:  [0.] + [1.1281002]x_9^1 + [0.]x_9^2 + [-1.110421]x_9^3
# Drift term:  [0.] + [1.0862463]x_10^1 + [0.]x_10^2 + [-1.0856614]x_10^3
# Diffusion term:  diag  [1.0008442  0.99994177 1.0089513  0.9931957  0.9906971  0.98999316
#  1.0073184  0.993294   0.995657   0.9940986 ]
# Maximum relative error:  0.12810016
# Maximum index:  tensor(16)
# 'train' took 116.714351 s

# dim = 10, 100000, 10000
# Drift term:  [0.] + [1.012197]x_1^1 + [0.]x_1^2 + [-1.004239]x_1^3
# Drift term:  [0.] + [1.0772319]x_2^1 + [0.]x_2^2 + [-1.0788848]x_2^3
# Drift term:  [0.] + [0.98084617]x_3^1 + [0.]x_3^2 + [-0.9904502]x_3^3
# Drift term:  [0.] + [1.052063]x_4^1 + [0.]x_4^2 + [-1.0368673]x_4^3
# Drift term:  [0.] + [1.0693477]x_5^1 + [0.]x_5^2 + [-1.0589726]x_5^3
# Drift term:  [0.] + [0.9754462]x_6^1 + [0.]x_6^2 + [-0.9667391]x_6^3
# Drift term:  [0.] + [0.98528475]x_7^1 + [0.]x_7^2 + [-0.99109507]x_7^3
# Drift term:  [0.] + [0.9688166]x_8^1 + [0.]x_8^2 + [-0.95515496]x_8^3
# Drift term:  [0.] + [1.070916]x_9^1 + [0.]x_9^2 + [-1.0579861]x_9^3
# Drift term:  [0.] + [1.0192893]x_10^1 + [0.]x_10^2 + [-1.0103362]x_10^3
# Diffusion term:  diag  [1.0005744  0.99489206 1.0010949  0.9967245  0.9968781  1.0006505
#  1.0020194  0.99781036 0.9925565  0.99695003]
# Maximum relative error:  0.07888484
# Maximum index:  tensor(3)
# 'train' took 1122.320055 s

# dim = 20, 10000, 10000
# Drift term:  [0.] + [1.1647022]x_1^1 + [0.]x_1^2 + [-0.99317515]x_1^3
# Drift term:  [0.] + [1.8593854]x_2^1 + [0.]x_2^2 + [-1.6564704]x_2^3
# Drift term:  [0.] + [1.2521828]x_3^1 + [0.]x_3^2 + [-1.1931735]x_3^3
# Drift term:  [0.] + [0.68556917]x_4^1 + [0.]x_4^2 + [-0.6003768]x_4^3
# Drift term:  [0.] + [1.4960207]x_5^1 + [0.]x_5^2 + [-1.5709757]x_5^3
# Drift term:  [0.] + [2.7557385]x_6^1 + [0.]x_6^2 + [-2.3793914]x_6^3
# Drift term:  [0.] + [0.7966737]x_7^1 + [0.]x_7^2 + [-0.78391254]x_7^3
# Drift term:  [0.] + [1.5589547]x_8^1 + [0.]x_8^2 + [-1.4361829]x_8^3
# Drift term:  [0.] + [0.8493152]x_9^1 + [0.]x_9^2 + [-0.835422]x_9^3
# Drift term:  [0.] + [1.9925809]x_10^1 + [0.]x_10^2 + [-1.8053694]x_10^3
# Drift term:  [0.] + [1.6002327]x_11^1 + [0.]x_11^2 + [-1.3944683]x_11^3
# Drift term:  [0.] + [1.7029676]x_12^1 + [0.]x_12^2 + [-1.330901]x_12^3
# Drift term:  [0.] + [1.0247062]x_13^1 + [0.]x_13^2 + [-0.89432824]x_13^3
# Drift term:  [0.] + [0.41045865]x_14^1 + [0.]x_14^2 + [-0.40067533]x_14^3
# Drift term:  [0.] + [1.3565534]x_15^1 + [0.]x_15^2 + [-1.1776927]x_15^3
# Drift term:  [0.] + [2.040773]x_16^1 + [0.]x_16^2 + [-1.9214375]x_16^3
# Drift term:  [0.] + [0.6105626]x_17^1 + [0.]x_17^2 + [-0.6890639]x_17^3
# Drift term:  [0.] + [1.9462731]x_18^1 + [0.]x_18^2 + [-1.6737919]x_18^3
# Drift term:  [0.] + [1.6701978]x_19^1 + [0.]x_19^2 + [-1.5452036]x_19^3
# Drift term:  [0.] + [0.9014032]x_20^1 + [0.]x_20^2 + [-1.0127585]x_20^3
# Diffusion term:  diag  [0.9156878  0.9055983  0.9849663  1.000534   1.0022058  0.8404614
#  0.9962087  0.95627934 1.0559005  0.9083299  0.9171698  0.8558494
#  1.048113   1.0738444  0.9465337  0.9035097  1.063468   0.9106108
#  0.91336805 1.0381655 ]
# Maximum relative error:  1.7557385
# Maximum index:  tensor(10)
# 'train' took 356.716357 s



# dim = 20, 100000, 10000
# Drift term:  [0.] + [1.2965428]x_1^1 + [0.]x_1^2 + [-1.1971459]x_1^3
# Drift term:  [0.] + [1.0707076]x_2^1 + [0.]x_2^2 + [-1.0642772]x_2^3
# Drift term:  [0.] + [0.90906835]x_3^1 + [0.]x_3^2 + [-0.9212286]x_3^3
# Drift term:  [0.] + [1.1777157]x_4^1 + [0.]x_4^2 + [-1.1806903]x_4^3
# Drift term:  [0.] + [0.86061925]x_5^1 + [0.]x_5^2 + [-0.8759751]x_5^3
# Drift term:  [0.] + [1.2733783]x_6^1 + [0.]x_6^2 + [-1.2989466]x_6^3
# Drift term:  [0.] + [1.2554698]x_7^1 + [0.]x_7^2 + [-1.3279897]x_7^3
# Drift term:  [0.] + [0.91214746]x_8^1 + [0.]x_8^2 + [-0.8508773]x_8^3
# Drift term:  [0.] + [1.1428977]x_9^1 + [0.]x_9^2 + [-1.1123488]x_9^3
# Drift term:  [0.] + [1.3286101]x_10^1 + [0.]x_10^2 + [-1.1966629]x_10^3
# Drift term:  [0.] + [1.2184552]x_11^1 + [0.]x_11^2 + [-1.062825]x_11^3
# Drift term:  [0.] + [1.0718924]x_12^1 + [0.]x_12^2 + [-1.0367743]x_12^3
# Drift term:  [0.] + [0.79269457]x_13^1 + [0.]x_13^2 + [-0.82967985]x_13^3
# Drift term:  [0.] + [1.2124543]x_14^1 + [0.]x_14^2 + [-1.223214]x_14^3
# Drift term:  [0.] + [0.83680934]x_15^1 + [0.]x_15^2 + [-0.80874753]x_15^3
# Drift term:  [0.] + [1.1436517]x_16^1 + [0.]x_16^2 + [-1.0915296]x_16^3
# Drift term:  [0.] + [1.1510905]x_17^1 + [0.]x_17^2 + [-1.0967455]x_17^3
# Drift term:  [0.] + [1.2050061]x_18^1 + [0.]x_18^2 + [-1.1574419]x_18^3
# Drift term:  [0.] + [1.3162247]x_19^1 + [0.]x_19^2 + [-1.2280293]x_19^3
# Drift term:  [0.] + [0.84241104]x_20^1 + [0.]x_20^2 + [-0.90673053]x_20^3
# Diffusion term:  diag  [0.9528751  1.0049958  1.0231199  0.99163544 1.019474   0.99063957
#  1.0010692  1.0002908  0.9918071  0.93942434 0.96035844 0.9873085
#  1.0319908  0.9829528  1.0096761  0.9832944  0.9834493  1.0077132
#  0.9611907  1.034153  ]
# Maximum relative error:  0.32861006
# Maximum index:  tensor(18)
# 'train' took 2800.502069 s