from casadi import SX,DM, vertcat, reshape, Function, nlpsol, inf, norm_2,sin,cos,pi
import matplotlib.pyplot as plt
import numpy as np
from NMPC_methods import J_fh as J


def NMPC_init():
    global x_ub, x_lb,y_ub,y_lb,T,N_pred,state_r,input_r
    x_ub = 10
    x_lb = -10
    y_ub = 10
    y_lb = -10

    # Abtastzeit und Prediction horizon
    T = 0.1
    N_pred = 9
    # Zielruhelage von den Zustaenden
    # Referenz von h, h_p, eta
    state_r = [0, 0, 0]
    input_r = [0, 0]
if __name__ == '__main__':
    NMPC_init()
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

    # Zustandsdarstellung
    rhs = vertcat(sin(theta)*u_1,cos(theta)*u_1,u_2)
    print(rhs)

    Q = SX.zeros(3, 3)
    Q[0, 0] = 1
    Q[1, 1] = 1
    Q[2, 2] = 1

    R = SX.zeros(2, 2)
    R[0, 0] = 1
    R[1,1]=1

    f, solver = J(T,Q,R,N_pred,states,controls,rhs)

    #specific nonlinear optimal problem
    nl = {}
    nl['lbx'] = [0 for i in range(n_controls*N_pred)]
    nl['ubx'] = [0 for i in range(n_controls*N_pred)]
    # ??? better method?
    for i in range(N_pred):
        nl['lbx'][:2*(N_pred):2] = [-10] * (N_pred)
        nl['ubx'][:2*(N_pred):2] =  [10] * (N_pred)
        nl['lbx'][1:2*(N_pred):2] = [-2] * (N_pred)
        nl['ubx'][1:2*(N_pred):2] =  [2] * (N_pred)
    # multi-constraints defined by list
    # initialisation
    nl['lbg'] = [0 for i in range(3 * (N_pred + 1) )]
    nl['ubg'] = [0 for i in range(3 * (N_pred + 1) )]
    nl['lbg'][:3 * (N_pred + 1):3] = [x_lb] * (N_pred + 1)
    nl['ubg'][:3 * (N_pred + 1):3] = [x_ub] * (N_pred + 1)
    nl['lbg'][1:3 * (N_pred + 1):3] = [y_lb] * (N_pred + 1)
    nl['ubg'][1:3 * (N_pred + 1):3] = [y_ub] * (N_pred + 1)
    nl['lbg'][2:3 * (N_pred + 1):3] = [-inf] * (N_pred + 1)
    nl['ubg'][2:3 * (N_pred + 1):3] = [inf] * (N_pred + 1)
    nl['lbg'][-3] = state_r[0]
    nl['ubg'][-3] =state_r[0]
    nl['lbg'][-2] = state_r[1]
    nl['ubg'][-2] =state_r[1]
    print(nl['lbg'])


    state_0 = [5, 5, 0]
    u0 = [0 for i in range(n_controls)]
    pred_counter = 0
    u_matrix = []
    state_matrix = []
    state_matrix += state_0
    print(state_0 + state_r)
    state_pre_matrix = []

    while pred_counter <= 100:
        nl['p'] = state_0 + state_r + input_r
        nl['x0'] = [0] * N_pred*n_controls
        sol = solver(**nl)
        print(sol)
        print(solver.stats()['success'] == True)
        u0 = [float(sol['x'][i]) for i in range(n_controls)]
        state_0_DM = DM(state_0)
        state_0_DM = reshape(state_0, n_states, 1)
        u0_DM = DM(u0)
        u0_DM = reshape(u0, n_controls, 1)
        f_value_DM = f(state_0_DM, u0_DM)
        state_0_DM = state_0_DM + (T * f_value_DM)
        # state_0 = [float(state_0_DM[i])+random.gauss(0, 0.05) for i in
        # range(n_states)] #with random sensor noise
        state_0 = [float(state_0_DM[i]) for i in range(n_states)]
        state_pre_matrix += state_0
        # with random sensor noise on position
        pred_counter += 1
        u_matrix += u0
        state_matrix += state_0

    t = [T * i for i in range(pred_counter)]
    x_out = []
    y_out = []
    u1_out = []
    u2_out =[]
    for i in range(pred_counter):
        x_out.append(state_matrix[3 * i])  # extend
        y_out.append(state_matrix[3 * i + 1])
    # plt.plot(t, eta_out, "r")
    plt.plot(x_out, y_out, "b")
    # plt.plot(t, eta_out, "r")
    # plt.plot(t, u_matrix,"g")
    plt.xlabel('x')
    # plt.ylabel('eta,h')
    plt.ylabel('y')

    plt.show()