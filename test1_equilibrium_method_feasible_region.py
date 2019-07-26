from ipydex import IPS, activate_ips_on_exception
from sys import path

from casadi import SX,DM, vertcat, reshape, Function, nlpsol, inf, norm_2
import matplotlib.pyplot as plt
import numpy as np
import random
from NMPC_methods import J_fh as J

def plant_init():
    #Ruhelage eta=48,753
    #u_PWM=157,2688
    # Parameter konfiguration
    global A_B,A_SP,m,g,T_M,k_M,k_V,k_L,eta_0
    A_B= 2.8274e-3  # [m**2]
    A_SP = 0.4299e-3  # [m**2]
    m = 2.8e-3  # [kg]
    g = 9.81  # [m/(s**2)]
    T_M = 0.57  # [s]
    k_M = 0.31  # [s**-1]
    k_V = 6e-5  # [m**3]
    k_L = 2.18e-4  # [kg/m]
    eta_0 = 1900 / 60  # / 60 * 2 * pi
def NMPC_init():
    global h_ub, h_lb,eta_ub,eta_lb,T,N_pred,state_r,input_r
    h_ub = 2
    h_lb = 0
    eta_ub = 200
    eta_lb = 0

    # Abtastzeit und Prediction horizon
    T = 0.1
    N_pred = 3
    # Zielruhelage von den Zustaenden
    # Referenz von h, h_p, eta
    state_r = [0.8, 0, 48.753]
    input_r = [157.29]
if __name__ == '__main__':
    plant_init()
    NMPC_init()
    N_fe_range=10
    N_feasible_region=np.zeros([N_fe_range,N_fe_range])

    h_test_start=0.8-0.001
    h_test_end=0.8+0.001
    eta_test_start=48.753-0.001
    eat_test_end=48.753+0.001
    # Zustaende als Symbol definieren
    h = SX.sym('h')
    h_p = SX.sym('h_p')
    eta = SX.sym('eta')
    # Zustandsvektor
    states = vertcat(h, h_p, eta)
    # print(states.shape)

    # Eingang symbolisch definieren
    u_pwm = SX.sym('u_pwm')
    # Eingangsverktor
    controls = vertcat(u_pwm)

    # kriegen die Anzahl der Zeiler("shape" ist ein Tuple)
    n_states = states.shape[0]  # p 22.

    # Eingangsanzahl
    n_controls = controls.shape[0]

    # Zustandsdarstellung
    rhs = vertcat(h_p, k_L / m * ((k_V * (eta + eta_0) - A_B * h_p) /
                                  A_SP)**2 - g, -1 / T_M * eta + k_M / T_M * u_pwm)
    # system r.h.s eta_p h_p h_pp
    print(rhs)

    Q = SX.zeros(3, 3)
    Q[0, 0] = 1
    Q[1, 1] = 1
    Q[2, 2] = 1

    R = SX.zeros(1, 1)
    R[0, 0] = 1





    state_0 = [h_test_start, 0, 50]
    u0 = [0 for i in range(n_controls)]
    pred_counter = 0
    u_matrix = []
    state_matrix = []
    state_matrix += state_0
    print(state_0 + state_r)
    state_pre_matrix = []

    for k in range(N_fe_range):
        state_0[2] = eta_test_start
        for j in range(N_fe_range):
            for n in range(2,100):
                # specific nonlinear optimal problem
                nl = {}

                nl['lbx'] = [0 for i in range(n)]
                nl['ubx'] = [0 for i in range(n)]
                # ??? better method?
                for i in range(n):
                    nl['lbx'][i] = 0
                    nl['ubx'][i] = 255
                # multi-constraints defined by list
                # initialisation
                nl['lbg'] = [0 for i in range(3 * (n + 1))]
                nl['ubg'] = [0 for i in range(3 * (n + 1))]
                #
                nl['lbg'][::3] = [h_lb] * (n + 1)
                nl['ubg'][::3] = [h_ub] * (n + 1)
                nl['lbg'][1::3] = [-inf] * (n + 1)
                nl['ubg'][1::3] = [inf] * (n + 1)
                nl['lbg'][2::3] = [eta_lb] * (n + 1)
                nl['ubg'][2::3] = [eta_ub] * (n + 1)
                nl['lbg'][-3] = 0.8
                nl['ubg'][-3] = 0.8
                nl['lbg'][-2] = 0
                nl['ubg'][-2] = 0
                nl['lbg'][-1] = 48.753
                nl['ubg'][-1] = 48.753

                f, solver = J(T, Q, R, n, states, controls, rhs)
                #print(solver)
                nl['p'] = state_0 + state_r + input_r
                nl['x0'] = [0] * n
                sol = solver(**nl)
                print(n)
                if (solver.stats())['success'] == True:
                    N_feasible_region[k][j] = n
                    break
                if n==100:
                    N_feasible_region[k][j] = 10000
                # print(N_feasible_region[k][j])
            state_0[2]=state_0[2]+(eat_test_end-eta_test_start)/N_fe_range
        state_0[0] = state_0[0] + (h_test_end-h_test_start) / N_fe_range
    print(N_feasible_region)
    X, Y = np.meshgrid(np.linspace(h_test_start, h_test_end, N_fe_range), np.linspace(eta_test_start, eat_test_end, N_fe_range))


    # plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    c = ax.pcolormesh(X, Y, N_feasible_region, cmap='Spectral_r')
    cb = fig.colorbar(c)
    cb.set_label('Prediction Horizon')
    plt.xlabel("h [m]")
    plt.ylabel("eta")
    plt.show()