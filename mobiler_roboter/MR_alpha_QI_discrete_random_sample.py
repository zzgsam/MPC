from ipydex import IPS, activate_ips_on_exception
from sys import path

from casadi import sin,cos,SX, DM, vertcat, reshape, Function, nlpsol, inf, norm_2,sqrt,gradient,dot
import matplotlib.pyplot as plt
import random
if __name__ == '__main__':
    # Parameter konfiguration


    A_d=SX([[1,0,0],[0,1,0],[0,0,1]])
    B_d=SX([[0,0],[1,0],[0,1]])
    T=0.1

    temp_x1= 0.2
    temp_x2 =5
    temp_x3 =0
    N_fe_range=10

    # Zustaende als Symbol definieren
    x = SX.sym('x')
    y = SX.sym('y')
    theta = SX.sym('theta')
    # ZustandsvekFtor
    states = vertcat(x, y, theta)
    # print(states.shape)

    # Eingang symbolisch definieren
    u_1 = SX.sym('u_1')
    u_2= SX.sym("u_2")
    # Eingangsverktor
    controls = vertcat(u_1,u_2)

    # kriegen die Anzahl der Zeiler("shape" ist ein Tuple)
    n_states = states.shape[0]  # p 22.

    # Eingangsanzahl
    n_controls = controls.shape[0]

    rhs = vertcat(sin(theta)*u_1,cos(theta)*u_1,u_2)

    f = Function('f', [states, controls], [rhs*T+states])


    K = SX([[0.99581, 0.086879, 0.1499]])
    # K = SX([[31.2043,  2.7128, 0.4824]])
    phi_c=A_d-B_d@K


    P_f=SX([[5959.1,517.92,65.058],[517.92,51.198,5.6392],[65.058,5.6392,641.7]])
    # P_f=SX([[68639,5891.4,616.42],[5891.4,519.84,53.599],[616.42,53.599,641.7]])
    state_r =SX ([[0.8], [0], [48.760258862]])
    u_r = [157.291157619]
    Q = SX.zeros(3, 3)
    # Q[0, 0] = 1e3
    Q[0, 0] = 1
    Q[1, 1] = 1
    Q[2, 2] = 1

    R = SX.zeros(1, 1)
    R[0, 0] = 1

    Q_s=Q+K.T@R@K
    # delta_Q=SX([[5000,0,0],[0,5,0],[0,0,5]])
    delta_Q=SX([[50,0,0],[0,5,0],[0,0,5]])
    K=reshape(K,1,3)
    X = SX.sym('X', 3, 1)

    psi=f(X,-K@(X-state_r)+u_r)-((phi_c)@(X-state_r)+state_r)
    psi_f=Function("psi_f",[X],[psi])
    i_n=10
    j_n=10
    k_n=10
    for i in range(i_n+1):#h
        temp_x1_loop = -temp_x1 + temp_x1 * 2 * i / i_n
        for j in range(j_n+1): #h_p
            temp_x2_loop = -temp_x2 + temp_x2 * 2 * j / j_n

            for k in range (k_n+1): #eta
                temp_x3_loop=-temp_x3+ temp_x3*2*k/k_n
                print("temp_x1_loop",temp_x1_loop,"temp_x2_loop",temp_x2_loop,"temp_x3_loop",temp_x3_loop)
                state_act=SX ([[0.8+temp_x1_loop], [0+temp_x2_loop], [48.760258862+temp_x3_loop]])
                xi = -(X - state_r).T @ (delta_Q) @ (X - state_r) + 2 * psi.T @ P_f @ phi_c @ (X - state_r) + psi.T @ P_f @ psi
                # xi_1=-(X - state_r).T @ (delta_Q) @ (X - state_r)
                # xi_2=2 * psi.T @ P_f @ phi_c @ (X - state_r)
                # xi_3= psi.T @ P_f @ psi
                xi_f=Function("xi_f",[X],[xi])
                # xi1_f=Function("xi1_f",[X],[xi_1])
                # xi2_f=Function("xi2_f",[X],[xi_2])
                # xi3_f=Function("xi3_f",[X],[xi_3])
                # print(state_act-state_r)
                # print("psi",psi_f(state_act))
                print("xi_f",xi_f(state_act))
                # print("xi1_f",xi1_f(state_act))
                # print("xi2_f",xi2_f(state_act))
                # print("xi3_f",xi3_f(state_act))
                if k == k_n-1:
                    temp_x3_loop=temp_x3
            if j == j_n - 1:
                temp_x2_loop = temp_x2
        if i == i_n - 1:
            temp_x1_loop = temp_x1

"""
    {'f': DM(2.40313e-009), 'g': DM([1, 2.33839e-007]), 'lam_g': DM([-0.00173027, -0.0102769]), 'lam_p': DM([]),
     'lam_x': DM([-2.81628e-010, 0, -3.11122e-011]), 'x': DM([0.943983, 16.5829, 52.141])}
    cost = 49343.5
"""

"""
{'f': DM(2.37397e-008), 'g': DM([1, 255]), 'lam_g': DM([-768.682, 0]), 'lam_p': DM([]), 'lam_x': DM([-5.06284e-008, 0, -1.61154e-010]), 'x': DM([0.710579, -10.3015, 46.6602])}
    cost = 19041
"""

