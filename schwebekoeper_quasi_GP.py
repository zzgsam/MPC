from casadi import SX, DM, vertcat, reshape, Function, nlpsol, inf, norm_2
import matplotlib.pyplot as plt
import random

import numpy as np
from numpy.linalg import inv


def posterior_predictive(
        X_s,
        X_train,
        Y_train,
        l=1.0,
        sigma_f=1.0,
        sigma_y=2e-1):
    ''' Computes the suffifient statistics of the GP posterior predictive distribution from m training data X_train and Y_train and n new inputs X_s. Args: X_s: New input locations (n x d). X_train: Training locations (m x d). Y_train: Training targets (m x 1). l: Kernel length parameter. sigma_f: Kernel vertical variation parameter. sigma_y: Noise parameter. Returns: Posterior mean vector (n x d) and covariance matrix (n x n). '''
    K = kernel(X_train, X_train, l, sigma_f) + \
        sigma_y ** 2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    # Equation (4)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (5)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


def kernel(X1, X2, l=1.0, sigma_f=1.0):
    ''' Isotropic squared exponential kernel. Computes a covariance matrix from points in X1 and X2. Args: X1: Array of m points (m x d). X2: Array of n points (n x d). Returns: Covariance matrix (m x n). '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
        np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def data_string2float(data_matrix):
    data_matrix = data_matrix.strip("[]")
    data_matrix = data_matrix.split(',')
    # Map applies a function to all the items in an input_list. Here is the
    # blueprint:
    data_matrix = list(map(float, data_matrix))
    return data_matrix


if __name__ == "__main__":
    T = 0.1
    # get training data
    train_u_file_name = './data/training_u.txt'
    train_x_file_name = './data/training_x.txt'
    train_x_pre_file_name = './data/training_x_pre.txt'
    with open(train_u_file_name, 'r') as f:
        u_train_matrix = f.read()
    with open(train_x_file_name, 'r') as f:
        x_train_matrix = f.read()
    with open(train_x_pre_file_name, 'r') as f:
        x_train_pre_matrix = f.read()

    u_train_matrix = data_string2float(u_train_matrix)
    # print(u_train_matrix)
    x_train_matrix = data_string2float(x_train_matrix)
    # print(x_train_matrix)
    x_train_pre_matrix = data_string2float(x_train_pre_matrix)
    # print(x_train_pre_matrix)

    x_prior_train_matrix = x_train_matrix[0:-3]
    x_post_train_matrix = x_train_matrix[3::]
    # transformed into numpy array
    x_prior_train_matrix_np = np.array(x_prior_train_matrix).reshape(-1, 3)
    x_post_train_matrix_np = np.array(x_post_train_matrix).reshape(-1, 3)
    x_train_pre_matrix_np = np.array(x_train_pre_matrix).reshape(-1, 3)
    u_train_matrix_np = np.array(u_train_matrix).reshape(-1, 1)

    z_np_train = np.c_[x_prior_train_matrix_np, u_train_matrix_np]
    y_np_train = (x_post_train_matrix_np -
                  x_train_pre_matrix_np).reshape(-1, 3)
    z = np.array([[1.96598163e-02,
                   3.04763486e-02,
                   5.50964910e+01,
                   2.54999997e+02],
                  [3.84996369e-02,
                   1.53654459e-01,
                   5.92988609e+01,
                   2.54999999e+02]])
    # print(z_np_train)
    print(x_post_train_matrix)
    print(x_train_pre_matrix)
    # print(y_np_train)

    train_set_input_z = [z_np_train[0:34, :],
                         z_np_train[34:67, :], z_np_train[67:101, :]]
    train_set_output_y = [y_np_train[0:34, :],
                          y_np_train[34:67, :], y_np_train[67:101, :]]
    #mu_s, cov_s = posterior_predictive(z, z_np_train, y_np_train)
    # for i in range(3):
    # print(train_set_input_z)
    train_set_input_temp = np.r_[train_set_input_z[2], train_set_input_z[1]]
    train_set_output_temp = np.r_[train_set_output_y[2], train_set_output_y[1]]

    mu_s0, cov_s0 = posterior_predictive(
        train_set_input_z[0], train_set_input_temp, train_set_output_temp)

    train_set_input_temp = np.r_[train_set_input_z[0], train_set_input_z[2]]
    train_set_output_temp = np.r_[train_set_output_y[0], train_set_output_y[2]]

    mu_s1, cov_s1 = posterior_predictive(
        train_set_input_z[1], train_set_input_temp, train_set_output_temp)

    train_set_input_temp = np.r_[train_set_input_z[0], train_set_input_z[1]]
    train_set_output_temp = np.r_[train_set_output_y[0], train_set_output_y[1]]

    mu_s2, cov_s2 = posterior_predictive(
        train_set_input_z[2], train_set_input_temp, train_set_output_temp)

    # out=np.r_[mu_s0-train_set_output_y[0],mu_s1-train_set_output_y[1],mu_s2-train_set_output_y[2]]
    out = np.r_[mu_s0, mu_s1, mu_s2]
    y_out = out[:, 0].reshape(-1, 1)
    f_out = x_train_pre_matrix_np.reshape(-1, 3)[:, 0].reshape(-1, 1)
    print(y_out.shape)
    print(f_out)
    x_next = y_out + f_out
    print(x_next - x_post_train_matrix_np[:, 0].reshape(-1, 1))
    t = [T * i for i in range(101)]
    plt.plot(t, x_next, "b")
    plt.xlabel('t')
    # plt.ylabel('eta,h')
    plt.ylabel('h')
    plt.show()

    t = [T * i for i in range(101)]
    plt.plot(t, f_out, "b")
    plt.xlabel('t')
    # plt.ylabel('eta,h')
    plt.ylabel('h')
    plt.show()
    # print(z_np_train,y_np_train)
    #[0.0304763, 1.23178, -14.9723]
