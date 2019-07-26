from casadi import SX, DM, vertcat, reshape, Function, nlpsol, inf, norm_2
import matplotlib.pyplot as plt
import random
from NMPC_methods import J_qi as J

def plant_init():
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
    state_r = [0.8, 0, 48]
    input_r = [157.29]
if __name__ == '__main__':
    plant_init()
    NMPC_init()

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
    Q[0, 0] = 1e7
    Q[1, 1] = 1
    Q[2, 2] = 1

    R = SX.zeros(1, 1)
    R[0, 0] = 1

    P_f = SX.zeros(3, 3)
    P_f[0, 0] = 1.0439e+007
    P_f[0, 1] = -3.0878e+007
    P_f[0, 2] = -1.6277e+009
    P_f[1, 0] = -3.0878e+007
    P_f[1, 1] = 1.6483e+008
    P_f[1, 2] = 6.4157e+009
    P_f[2, 0] = -1.6277e+009
    P_f[2, 1] = 6.4157e+009
    P_f[2, 2] = 2.9567e+011

    f, solver = J(T,Q,R,P_f,N_pred,states,controls,rhs)
    #specific nonlinear optimal problem
    nl = {}
    nl['lbx'] = [0 for i in range(N_pred)]
    nl['ubx'] = [0 for i in range(N_pred)]
    # ??? better method?
    for i in range(N_pred):
        nl['lbx'][i] = 0
        nl['ubx'][i] = 255
    # multi-constraints defined by list
    # initialisation
    nl['lbg'] = [0 for i in range(3 * (N_pred + 1) + 1)]
    nl['ubg'] = [0 for i in range(3 * (N_pred + 1) + 1)]
    nl['lbg'][:3 * (N_pred + 1):3] = [h_lb] * (N_pred + 1)
    nl['ubg'][:3 * (N_pred + 1):3] = [h_ub] * (N_pred + 1)
    nl['lbg'][1:3 * (N_pred + 1):3] = [-inf] * (N_pred + 1)
    nl['ubg'][1:3 * (N_pred + 1):3] = [inf] * (N_pred + 1)
    nl['lbg'][2:3 * (N_pred + 1):3] = [eta_lb] * (N_pred + 1)
    nl['ubg'][2:3 * (N_pred + 1):3] = [eta_ub] * (N_pred + 1)
    nl['lbg'][-1] = -inf
    #nl['ubg'][-1] = 500
    nl['ubg'][-1] = 0.25
    print(nl['lbg'])

    state_0 = [0, 0, 50]
    u0 = [0 for i in range(n_controls)]
    pred_counter = 0
    u_matrix = []
    state_matrix = []
    state_matrix += state_0
    print(state_0 + state_r)
    state_pre_matrix = []
    # while norm_2(DM(state_0) - DM(state_r)) >= 1:
    while pred_counter <= 100:

        nl['p'] = state_0 + state_r + input_r
        nl['x0'] = [0] * N_pred
        sol = solver(**nl)
        print(sol)
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
        state_0[0] = state_0[0] + random.gauss(0, 0.02)
        print(state_0)
        pred_counter += 1
        u_matrix += u0
        state_matrix += state_0

    t = [T * i for i in range(pred_counter)]
    eta_out = []
    h_out = []
    u_out = []
    for i in range(pred_counter):
        h_out.append(state_matrix[3 * i])  # extend
        eta_out.append(state_matrix[3 * i + 2])
    # plt.plot(t, eta_out, "r")
    plt.plot(t, h_out, "b")
    # plt.plot(t, eta_out, "r")
    # plt.plot(t, u_matrix,"g")
    plt.xlabel('t')
    # plt.ylabel('eta,h')
    plt.ylabel('h')

    plt.show()
