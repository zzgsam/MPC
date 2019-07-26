from ipydex import IPS, activate_ips_on_exception
from sys import path

from casadi import SX, DM, vertcat, reshape, Function, nlpsol, inf, norm_2,gradient,jacobian,inv
import matplotlib.pyplot as plt
import random
import time


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
    N_pred = 4
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

    obj = 0
    g = []  # constraints vector
    Q = SX.zeros(3, 3)
    Q[0, 0] = 1e8
    Q[1, 1] = 1
    Q[2, 2] = 0.5

    R = SX.zeros(1, 1)
    R[0, 0] = 0

    P_f=SX.zeros(3,3)

    #Lagrange Function definieren
    L = SX.sym('L')
    Lambda = SX.sym('Lambda', n_states, (N_pred + 1)) #Lagrangesche Multiplikator


    # Constrainsvariable spezifizieren
    g = SX.zeros(3 * (N_pred +1) , 1)
    g[0:3] =Z[:,0] - P[0:3]


    # compute predicted states and cost function
    for k in range(N_pred):
        st = Z[:, k]
        con = V[:, k]
        obj = obj + (st - P[3:6]).T@Q@(st - P[3:6]) + (con.T-157.29)@R@(con-157.29)
        g[(k+1)*3:(k+2)*3] =Z[:,k+1] - (f(st,con) * T+Z[:,k])
    st = Z[:, N_pred]
    print(st)
    obj = obj + (st - P[3:6]).T @ P_f @ (st - P[3:6])
    # create a dense matrix of state constraints
    #g = SX.zeros(3 * (N_pred + 1), 1)
    for i in range(Lambda.size(2)):
        L=L+Lambda[:,i].T @ g[3*i:3*(i+1)]
    L=L+obj
    print(L)
    #hinkreigen phi und Solution S
    phi=SX.sym('phi')
    S=SX.sym('S')

    phi=gradient(L,Lambda[:,0])
    S=Lambda[:,0]
    print("phi",phi)

    for i in range(N_pred):
        phi=vertcat(phi,gradient(L,Z[:,i]))
        phi = vertcat(phi, gradient(L, V[:, i]))
        phi = vertcat(phi, gradient(L, Lambda[:, i+1]))
        S=vertcat(S,Z[:,i])
        S=vertcat(S,V[:,i])
        S=vertcat(S,Lambda[:,i+1])

    phi = vertcat(phi, gradient(L, Z[:, N_pred]))
    S = vertcat(S,Z[:,N_pred])
    phi_func = Function("phi_func",[P,S],[phi])
    print(phi_func)
    #print(g[0:3])
    #print("s size",S.size())

   #KKT matrix
    KKT_M=SX.sym("KKT_M")
    KKT_M=jacobian(phi,S)
    print(KKT_M.size())

    KKT_M_func = Function('KKT_M_func', [S], [KKT_M])
    #KKT_M_func(SX.zeros(34,1))
    # OPTIMIZATION Variable
    OPT_variables_Z=reshape(Z,3*( N_pred+1) , 1)
    OPT_variables_V= reshape(V, N_pred, 1)

    OPT_variables =vertcat(OPT_variables_Z,OPT_variables_V)
    #print(OPT_variables.size())





    nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}
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

    nl['lbx'] = [0 for i in range(OPT_variables.size(1))]
    nl['ubx'] = [0 for i in range(OPT_variables.size(1))]
    # ??? better method?
    for i in range(OPT_variables.size(1)):
        nl['lbx'][i] = -inf
        nl['ubx'][i] = inf
    # multi-constraints defined by list
    # initialisation
    nl['lbg'] = [0 for i in range(g.size(1))]
    nl['ubg'] = [0 for i in range(g.size(1))]

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

        nl['p'] = state_0 + state_r
        nl['x0'] = [0] * OPT_variables.size(1)
        #start = time.clock()
        sol = solver(**nl)
        #elapsed = (time.clock() - start)
        #print("Time used:", elapsed)
        s_opt = SX.sym("s_opt")
        s_opt=sol['lam_g'][0:n_states]
        for i in range(N_pred):
            s_opt=vertcat(s_opt,sol['x'][n_states*i:n_states*(i+1)])
            s_opt = vertcat(s_opt, sol['x'][i*n_controls-n_controls*N_pred:(i+1)*n_controls-n_controls*N_pred])
            s_opt = vertcat(s_opt,sol['lam_g'][(i+1)*n_states:(i+2)*n_states])
        s_opt=vertcat(s_opt,sol['x'][N_pred*n_states:(N_pred+1)*n_states])
        #print(type(SX(s_opt)))
        print(sol)



        KKT_M_temp=KKT_M_func(s_opt)

        #print(KKT_M_temp)
        #u0 = [float(sol['x'][i-n_controls*N_pred]) for i in range(n_controls)]
        state_0_DM = DM(state_0)
        state_0_DM = reshape(state_0, n_states, 1)
        state_0_DM = DM(state_0)
        state_0_DM[1]=state_0_DM[1]+random.gauss(0, 0.1)
        P_new=vertcat(state_0_DM,DM(state_r))
        phi_temp=phi_func(P_new,s_opt)
        #delta s for input u0
        start = time.clock()
        delta_s=-inv(KKT_M_temp)@phi_temp
        elapsed = (time.clock() - start)
        print("Time used:", elapsed)
        print(delta_s.size())
        u0 = [float(delta_s[2*n_states:2*n_states+n_controls]+float(sol['x'][i-n_controls*N_pred])) for i in range(n_controls)]
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
