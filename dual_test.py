from ipydex import IPS, activate_ips_on_exception
from sys import path

from casadi import SX, DM, vertcat, reshape, Function, nlpsol, inf, norm_2,integrator,MX
import matplotlib.pyplot as plt
import random
if __name__ == '__main__':
    train_u_file_name = './data/training_u.txt'
    train_x_file_name = './data/training_x.txt'
    train_x_pre_file_name = './data/training_x_pre.txt'
    # Parameter konfiguration
    A_B = 2.8274e-3  # [m**2]
    A_SP = 0.4299e-3  # [m**2]
    m = 2.8e-3  # [kg]
    g = 9.81  # [m/(s**2)]
    T_M = 0.57  # [s]
    k_M = 0.31  # [s**-1]
    k_V = 6e-5  # [m**3]
    k_L = 2.18e-4  # [kg/m]
    eta_0 = 1900 / 60  # / 60 * 2 * pi

    h_ub = 2
    h_lb = 0
    eta_ub = 200
    eta_lb = 0

    # Abtastzeit und Prediction horizon
    T = 0.1
    N_pred = 3
    # Zielruhelage von den Zustaenden
    # Referenz von h, h_p, eta
    cons_sublevel=1.5862
    state_0 = [0, 0, 50]
    state_r = [0.8, 0, 48.760258862]
    u_r=[157.291157619]
    # K = -SX([[1, 0.0874, 0.1630]])
    K=-SX([[100,8.6681,1.1539]])
    # activate_ips_on_exception()

    # IPS()

    # Zustaende als Symbol definieren
    h = SX.sym('h')
    h_p = SX.sym('h_p')
    eta = SX.sym('eta')
    # Zustandsvektor
    states = vertcat(h, h_p, eta)
    # print(states.shape)
    # kriegen die Anzahl der Zeiler("shape" ist ein Tuple)
    n_states = states.shape[0]  # p 22.

    # Eingang symbolisch definieren
    u_pwm = SX.sym('u_pwm')
    # Eingangsverktor
    controls = vertcat(u_pwm)
    # Eingangsanzahl
    n_controls = controls.shape[0]
    # Zustandsdarstellung
    rhs = vertcat(h_p, k_L / m * ((k_V * (eta + eta_0) - A_B * h_p) /
                  A_SP)**2 - g, -1 / T_M * eta + k_M / T_M * u_pwm)
    # system r.h.s eta_p h_p h_pp
    print(rhs)

    # nonlinear mapping function f(x,u)
    f = Function('f', [states, controls], [rhs])
    # Decision variables (controls) u_pwm
    U = SX.sym('U', n_controls, N_pred)
    # parameters (which include the !!!initial and the !!!reference state of
    # the robot)
    P = SX.sym('P', n_states + n_states)
    # number of x is always N+1   -> 0..N
    X = SX.sym('X', n_states, (N_pred + 1))

    X[:, 0] = P[0:3]
    obj = 0
    g = []  # constraints vector
    Q = SX.zeros(3, 3)
    Q[0, 0] = 1e4
    Q[1, 1] = 1
    Q[2, 2] = 1

    R = SX.zeros(1, 1)
    R[0, 0] = 1

    # P_f = SX([[159.781, 13.8916, 1.8387], [13.8916, 1.2518, 0.16074], [1.8387, 0.16074, 0.2998]])
    P_f = SX([[21506, 1795.4, 183.87], [1795.4, 152.95, 15.938], [183.87, 15.938, 2.1217]])
    print(P_f)

    f_q=(states-state_r).T@Q@(states-state_r)+(u_pwm-u_r).T@R@(u_pwm-u_r)
    dae = {'x': states, 'p': u_pwm, 'ode': rhs, 'quad': f_q}
    print(dae)
    opts = {"tf": 0.5}
    I = integrator('I', "idas", dae, opts)

    X_temp=state_0
    J = 0
    for k in range(N_pred):
        Ik = I(x0=X_temp, p=U[k][0])
        X_temp = Ik['xf']
        J += Ik['qf']  # Sum up quadratures
    print("J",J)

    print(J)
    # compute predicted states and cost function
    for k in range(N_pred):
        st = X[:, k]
        con = U[:, k]
        f_value = f(st, con)
       # print(f_value)
        st_next = st + (T * f_value)
        X[:, k + 1] = st_next
        obj = obj + ((st - P[3:6]).T@Q@(st - P[3:6]) + (con-u_r).T@R@(con-u_r))*T
    # print(obj)
    st = X[:, N_pred]
    print(st)
    obj = obj + (st - P[3:6]).T @ P_f @ (st - P[3:6])
    # create a dense matrix of state constraints
    g = SX.zeros(3 * (N_pred + 1), 1)
    g = SX.zeros(3 * (N_pred + 1) + 1, 1)

    print(g.size())

    # Constrainsvariable spezifizieren
    for k in range(N_pred + 1):
        # h
        g[3 * k] = X[0, k]
    #    h_p

        g[3 * k + 1] = X[1, k]
    #   eta
        g[3 * k + 2] = X[2, k]
    g[3 * (N_pred + 1)] = (X[:, N_pred] - P[3:6]
                           ).T@ P_f @ (X[:, N_pred] - P[3:6])
    print(g)

    OPT_variables = reshape(U, N_pred, 1)

    nlp_prob = {'f': J, 'x': OPT_variables, 'g': g, 'p': P}
    opts = {}
    opts['print_time'] = False
    opts['ipopt'] = {
        'max_iter': 100,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6}
    # print(opts)
    solver = nlpsol("solver", "ipopt", nlp_prob, opts)
    print(solver)

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
    nl['ubg'][-1] = cons_sublevel
    print(nl['lbg'])



    u0 = [0 for i in range(n_controls)]
    pred_counter = 0
    u_matrix = []
    state_matrix = []
    state_matrix += state_0
    print(state_0 + state_r)
    state_pre_matrix = []
    # while norm_2(DM(state_0) - DM(state_r)) >= 1:
    while pred_counter <= 100:

        if (SX(state_0) - SX(state_r)).T@P_f@ (SX(state_0) - SX(state_r)) >= cons_sublevel:
            nl['p'] = state_0 + state_r
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
            #state_0[0] = state_0[0] + random.gauss(0, 0.02)
            print(state_0)
            pred_counter += 1
            u_matrix += u0
            state_matrix += state_0
        else:
            print("switch to linear controller")
            u0=K@(SX(state_0) - SX(state_r)) + SX(u_r)
            u0_DM = DM(u0)
            u0_DM = reshape(u0, n_controls, 1)
            f_value_DM = f(state_0_DM, u0_DM)
            state_0_DM = state_0_DM + (T * f_value_DM)
            # state_0 = [float(state_0_DM[i])+random.gauss(0, 0.05) for i in
            # range(n_states)] #with random sensor noise
            state_0 = [float(state_0_DM[i]) for i in range(n_states)]
            state_pre_matrix += state_0
            # with random sensor noise on position
            #state_0[0] = state_0[0] + random.gauss(0, 0.02)
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

