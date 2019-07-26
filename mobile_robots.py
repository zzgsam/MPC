from ipydex import IPS, activate_ips_on_exception
from sys import path

from casadi import SX,DM, vertcat, reshape, Function, nlpsol, inf, norm_2
import matplotlib.pyplot as plt

if __name__ == '__main__':

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
    N_pred = 90

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
    Q[0, 0] = 10
    Q[1, 1] = 1
    Q[2, 2] = 1e-5

    R = SX.zeros(1, 1)
    R[0, 0] = 1

    # compute predicted states and cost function
    for k in range(N_pred):
        st = X[:, k]
        con = U[:, k]
        f_value = f(st, con)
       # print(f_value)
        st_next = st + (T * f_value)
        X[:, k + 1] = st_next
        obj = obj + (st - P[3:6]).T@Q@(st - P[3:6]) + (con-157).T@R@(con-157)
    # print(obj)

    # create a dense matrix of state constraints
    g = SX.zeros(3 * (N_pred + 1), 1)
    print(g.size())

    # Constrainsvariable spezifizieren
    for k in range(N_pred + 1):
        # h
        g[3 * k] = X[0, k]
    #    print(g[k])

        #h_p
        g[3 * k + 1] = X[1, k]
        # eta
        g[3 * k + 2] = X[2, k]
    print(g)
    OPT_variables = reshape(U, N_pred, 1)

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

    nl['lbx'] = [0 for i in range(N_pred)]
    nl['ubx'] = [0 for i in range(N_pred)]
    # ??? better method?
    for i in range(N_pred):
        nl['lbx'][i] = 0
        nl['ubx'][i] = 255
    # multi-constraints defined by list
    # initialisation
    nl['lbg'] = [0 for i in range(3 * (N_pred + 1))]
    nl['ubg'] = [0 for i in range(3 * (N_pred + 1))]
    #
    nl['lbg'][::3] = [h_lb] * (N_pred + 1)
    nl['ubg'][::3] = [h_ub] * (N_pred + 1)
    nl['lbg'][1::3] = [-inf] * (N_pred + 1)
    nl['ubg'][1::3] = [inf] * (N_pred + 1)
    nl['lbg'][2::3] = [eta_lb] * (N_pred + 1)
    nl['ubg'][2::3] = [eta_ub] * (N_pred + 1)

    print(nl['ubg'])

    nl['p'] = [0, 0, 70, 0.9, 0, 48]
    nl['x0'] = [0 for i in range(N_pred)]

    sol = solver(**nl)

    # print(type(float((sol['g'][2]))))

    print(sol)
    # print(sol['x'].shape)
    eta_out = []
    h_out = []
    t = [T * i for i in range(N_pred)]
    for i in range(N_pred):
        h_out.append(float(sol['g'][3 * i]))  # extend
        #eta_out.append(float(sol['g'][3 * i + 2]))
    #plt.plot(t, eta_out, "r")
    plt.plot(t, h_out, "b")
    plt.xlabel('t')
    # plt.ylabel('eta,h')
    plt.ylabel('h')

    plt.show()
