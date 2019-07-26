from ipydex import IPS, activate_ips_on_exception
from sys import path

from casadi import SX, DM, vertcat, reshape, Function, nlpsol, inf, norm_2
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

    h_ub = 1.9
    h_lb = 0
    eta_ub = 200
    eta_lb = 0

    # Abtastzeit und Prediction horizon
    T = 0.1
    N_pred = 3
    # Zielruhelage von den Zustaenden
    # Referenz von h, h_p, eta
    state_r = [0.8, 0, 48]
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
    V = SX.sym('V', n_controls, N_pred)
    # parameters (which include the !!!initial and the !!!reference state of
    # the robot)
    P = SX.sym('P', n_states + n_states)
    # number of x is always N+1   -> 0..N
    Z = SX.sym('Z', n_states, (N_pred + 1))


    Z[:, 0] = P[0:3]
    obj_nom = 0

    g_nom = []  # constraints vector
    Q_nom = SX.zeros(3, 3)
    Q_nom[0, 0] = 1e7
    Q_nom[1, 1] = 1
    Q_nom[2, 2] = 1

    R_nom = SX.zeros(1, 1)
    R_nom[0, 0] = 1

    P_f = SX.zeros(3, 3)
    # compute predicted states and cost function
    for k in range(N_pred):
        st = Z[:, k]
        con = V[:, k]
        f_value = f(st, con)
       # print(f_value)
        st_next = st + (T * f_value)
        Z[:, k + 1] = st_next
        obj_nom = obj_nom + \
            (st - P[3:6]).T@Q_nom@(st - P[3:6]) + (con - 157.29).T@R_nom@(con - 157.29)
    # print(obj)
    st = Z[:, N_pred]
    print(st)
    obj_nom = obj_nom + (st - P[3:6]).T @ P_f @ (st - P[3:6])
    # create a dense matrix of state constraints
    #g = SX.zeros(3 * (N_pred + 1), 1)
    g_nom = SX.zeros(3 * (N_pred + 1), 1)

    print(g_nom.size())

    # Constrainsvariable spezifizieren
    for k in range(N_pred):
        # h
        g_nom[3 * k] = Z[0, k]
    #    h_p

        g_nom[3 * k + 1] = Z[1, k]
    #   eta
        g_nom[3 * k + 2] = Z[2, k]
    g_nom[3 * N_pred:3 * (N_pred + 1)] = Z[:, N_pred] - P[3:6]
    print(g_nom)

    OPT_variables_nom = reshape(V, N_pred, 1)

    nlp_prob_nom = {'f': obj_nom, 'x': OPT_variables_nom, 'g': g_nom, 'p': P}
    opts = {}
    opts['print_time'] = False
    opts['ipopt'] = {
        'max_iter': 100,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6}
    # print(opts)
    solver_nom = nlpsol("solver_nom", "ipopt", nlp_prob_nom, opts)
    print(solver_nom)

    nl_nom = {}

    nl_nom['lbx'] = [0 for i in range(N_pred)]
    nl_nom['ubx'] = [0 for i in range(N_pred)]
    # ??? better method?
    for i in range(N_pred):
        nl_nom['lbx'][i] = 0
        nl_nom['ubx'][i] = 255
    # multi-constraints defined by list
    # initialisation
    nl_nom['lbg'] = [0 for i in range(3 * (N_pred + 1))]
    nl_nom['ubg'] = [0 for i in range(3 * (N_pred + 1))]
    nl_nom['lbg'][:3 * N_pred:3] = [h_lb] * N_pred
    nl_nom['ubg'][:3 * N_pred:3] = [h_ub] * N_pred
    nl_nom['lbg'][1:3 * N_pred:3] = [-inf] * N_pred
    nl_nom['ubg'][1:3 * N_pred:3] = [inf] * N_pred
    nl_nom['lbg'][2:3 * N_pred:3] = [eta_lb] * N_pred
    nl_nom['ubg'][2:3 * N_pred:3] = [eta_ub] * N_pred
    #terminal constraint
    nl_nom['lbg'][-3:]=[0]*n_states
    nl_nom['ubg'][-3:]=[0]*n_states

    print(nl_nom['lbg'])

    #Ab hier definieren wir den Ancilliary-Regler
    P_anc = SX.sym('P_anc', n_states + n_states)
    X = SX.sym('X', n_states, (N_pred + 1))
    Z_opt = SX.sym('Z_opt', n_states, (N_pred + 1))
    Z_opt = reshape(Z_opt,-1,1)
    #print("Z_opt",Z_opt)
    U = SX.sym('U', n_controls, N_pred)
    V_opt = SX.sym('V_opt', n_controls, N_pred)
    P_anc = vertcat(P_anc,Z_opt)
    #print("P_anc",P_anc)
    X[:, 0] = P_anc[0:3]
    obj_anc = 0
    g_anc = []  # constraints vector
    Q_anc = SX.zeros(3, 3)
    Q_anc[0, 0] = 1e7
    Q_anc[1, 1] = 1
    Q_anc[2, 2] = 1

    R_anc = SX.zeros(1, 1)
    R_anc[0, 0] = 1

    P_f_anc=SX.zeros(3,3)
    for k in range(N_pred):
        st = X[:, k]
        con = U[:, k]
        f_value = f(st, con)
       # print(f_value)
        st_next = st + (T * f_value)
        X[:, k + 1] = st_next
        obj_anc = obj_anc + \
            (st - P_anc[2*n_states+k*n_states:2*n_states+(k+1)*n_states]).T@Q_anc@(st - P_anc[2*n_states+k*n_states:2*n_states+(k+1)*n_states]) + (con-U[:,k]).T@R_anc@(con-U[:,k])
    st = Z[:, N_pred]
    obj_anc = obj_anc + (st - P_anc[3:6]).T @ P_f_anc @ (st - P_anc[3:6])
    print(obj_anc)

    g_anc = SX.zeros(3 * (N_pred + 1), 1)

    print(g_anc.size())

    # Constrainsvariable spezifizieren
    for k in range(N_pred):
        # h
        g_anc[3 * k] = X[0, k]
    #    h_p

        g_anc[3 * k + 1] = X[1, k]
    #   eta
        g_anc[3 * k + 2] = X[2, k]
    print(X[:,N_pred])
    g_anc[-3:]= X[:,N_pred]-P_anc[-3:]

    print("g_anc",g_anc[-3:])
    print("P_anc",P_anc)
    OPT_variables_anc = reshape(U, N_pred, 1)

    nlp_prob_anc = {'f': obj_anc, 'x': OPT_variables_anc, 'g': g_anc, 'p': P_anc}
    opts = {}
    opts['print_time'] = False
    opts['ipopt'] = {
        'max_iter': 100,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6}
    # print(opts)
    solver_anc = nlpsol("solver_anc", "ipopt", nlp_prob_anc, opts)
    print(solver_anc)

    nl_anc = {}

    nl_anc['lbx'] = [0 for i in range(N_pred)]
    nl_anc['ubx'] = [0 for i in range(N_pred)]
    # ??? better method?
    for i in range(N_pred):
        nl_anc['lbx'][i] = 0
        nl_anc['ubx'][i] = 255
    # multi-constraints defined by list
    # initialisation
    nl_anc['lbg'] = [-inf for i in range(3 * (N_pred + 1))]
    nl_anc['ubg'] = [inf for i in range(3 * (N_pred + 1))]
    #terminal constraint
    nl_anc['lbg'][-3:]=[0] * n_states
    nl_anc['ubg'][-3:]=[0] * n_states

    print(nl_anc['ubg'])


    state_0 = [0, 0, 50]
    state_0_nom = state_0
    state_0_anc =  state_0
    u0 = [0 for i in range(n_controls)]
    pred_counter = 0
    u_matrix = []
    state_matrix = []
    state_matrix += state_0
    print(state_0 + state_r)
    state_matrix_nom = []
    state_matrix_nom += state_0
    # while norm_2(DM(state_0) - DM(state_r)) >= 1:
    while pred_counter <= 100:
        # if pred_counter % 20 ==0 :
        #     state_r[0] +=0.1
        nl_nom['p'] = state_0_nom + state_r
        nl_nom['x0'] = [0] * N_pred
        sol_nom = solver_nom(**nl_nom)
        #z+=z_opt(1;z)
        state_0_nom = [float(sol_nom['g'][i+n_states]) for i in range(n_states)]
        state_matrix_nom +=state_0_nom
        #calculate the last state z_opt(N;z)
        u_nom_last_temp=sol_nom['x'][-1 - (n_controls - 1):]
        state_last_minus1_temp = sol_nom['g'][n_states*(N_pred-1):n_states*N_pred]
        state_last_DM = f(state_last_minus1_temp,u_nom_last_temp)*T+state_last_minus1_temp
        print(sol_nom)
        print(state_0_nom)
        Z_opt_anc = [float(sol_nom['g'][i]) for i in range(n_states*(N_pred))] + [float(state_last_DM[i]) for i in range(n_states)]
        # print("u_nom_last_temp",u_nom_last_temp)
        # print("state_last_plus1_temp", state_last_minus1_temp)
        # print("state_last_DM", state_last_DM)
        # print("nl_anc['p']",nl_anc['p'])

        #solve ancilliary controller problem
        nl_anc['p'] = state_0_anc + state_r + Z_opt_anc
        nl_anc['x0'] = [0] * N_pred
        sol_anc=solver_anc(**nl_anc)
        print("sol_anc",sol_anc)
        u0 = [float(sol_anc['x'][i]) for i in range(n_controls)]
        state_0_DM = DM(state_0_anc)
        state_0_DM = reshape(state_0_anc, n_states, 1)
        u0_DM = DM(u0)
        u0_DM = reshape(u0, n_controls, 1)
        #get next state
        f_value_DM = f(state_0_DM, u0_DM)
        state_0_DM = state_0_DM + (T * f_value_DM)
        # state_0 = [float(state_0_DM[i])+random.gauss(0, 0.05) for i in
        # range(n_states)] #with random sensor noise
        state_0_anc = [float(state_0_DM[i]) for i in range(n_states)] #measurement
        state_0_anc[0]= state_0_anc[0]+random.gauss(0,0.01)
        # with random sensor noise on position
        #state_0[0] = state_0[0] + random.gauss(0, 0.02)
        print(state_0_anc)
        pred_counter += 1
        u_matrix += u0
        state_matrix += state_0_anc


    t = [T * i for i in range(pred_counter)]
    eta_out_real = []
    h_out_real = []
    h_out_id = []
    u_out = []
    for i in range(pred_counter):
        h_out_real.append(state_matrix[3 * i])  # extend
        eta_out_real.append(state_matrix[3 * i + 2])
        h_out_id.append(state_matrix_nom[3 * i])
    # plt.plot(t, eta_out, "r")
    plt.plot(t, h_out_real, "b")
    plt.plot(t,h_out_id,"r")
    # plt.plot(t, eta_out, "r")
    # plt.plot(t, u_matrix,"g")
    plt.xlabel('t')
    # plt.ylabel('eta,h')
    plt.ylabel('h')

    plt.show()
