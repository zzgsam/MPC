from ipydex import IPS, activate_ips_on_exception
from sys import path

from casadi import SX, DM, vertcat, reshape, Function, nlpsol, inf, norm_2,sqrt,gradient,dot
import matplotlib.pyplot as plt
import random
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

    A_d=SX([[1,0.1,0],[0,-0.14957,0.024395],[0,0,0.82456]])
    B_d=SX([[0],[0],[0.054386]])
    T=0.1


    h = SX.sym('h')
    h_p = SX.sym('h_p')
    eta = SX.sym('eta')
    states = vertcat(h, h_p, eta)
    u_pwm = SX.sym('u_pwm')
    controls = vertcat(u_pwm)
    rhs = vertcat(h_p, k_L / m * ((k_V * (eta + eta_0) - A_B * h_p) /
                  A_SP)**2 - g, -1 / T_M * eta + k_M / T_M * u_pwm)

    f = Function('f', [states, controls], [rhs*T+states])

    # K = SX([[0.99581, 0.086879, 0.1499]])
    K = SX([[ 31.2043, 2.7128, 0.4824]])

    phi_c=A_d-B_d@K

    P_f=SX([[5959.1,517.92,65.058],[517.92,51.198,5.6392],[65.058,5.6392,641.7]])
    # P_f=SX([[2.4944e+005,20941,1932.4],[20941,1839.4,168.05],[1932.4,168.05,641.7]])
    state_r =SX ([[0.8], [0], [48.760258862]])
    u_r = [157.291157619]
    Q = SX.zeros(3, 3)
    # Q[0, 0] = 1e4
    Q[0, 0] = 1
    Q[1, 1] = 1
    Q[2, 2] = 1

    R = SX.zeros(1, 1)
    R[0, 0] = 1

    Q_s=Q+K.T@R@K
    # delta_Q=SX([[5000,0,0],[0,5,0],[0,0,5]])
    delta_Q=SX([[5,0,0],[0,5,0],[0,0,5]])
    K=reshape(K,1,3)
    X = SX.sym('X', 3, 1)

    psi=f(X,-K@(X-state_r)+u_r)-((phi_c)@(X-state_r)+state_r)



    V_inf=(X-state_r).T@P_f@(X-state_r)
    Dis=norm_2(-K@(X-state_r)+u_r[0]-255)/sqrt(K[0]**2+K[1]**2+K[2]**2)
    print("dis",Dis)
    print("K",K,K.shape)
    obj=Dis
    OPT_variables=reshape(X,3,1)

    g=SX.zeros(2,1)
    g[0]=K@gradient(V_inf,X)/(norm_2(K)*norm_2(gradient(V_inf,X)))
    g[1]=-K@(X-state_r)+u_r[0]
    nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g}
    opts = {}
    opts['print_time'] = False
    opts['ipopt'] = {
        'max_iter': 1000,
        'print_level': 3,
        #'acceptable_tol': 1e-8,
        #'acceptable_obj_change_tol': 1e-6
        }
    # print(opts)
    solver = nlpsol("solver", "ipopt", nlp_prob, opts)
    nl = {}
    nl['lbx'] = [0 for i in range(3)]
    nl['ubx'] = [0 for i in range(3)]
    nl['lbx'][0] = 0
    nl['ubx'][0] = 2
    nl['lbx'][1] = -inf
    nl['ubx'][1] = inf
    nl['lbx'][2] = 0
    nl['ubx'][2] = 200
    nl['lbg'] = [0 for i in range(2)]
    nl['ubg'] = [0 for i in range(2)]
    nl['lbg'][0] = 1    ##+-1
    nl['ubg'][0] = 1    ##+-1
    nl['lbg'][1] = 0
    nl['ubg'][1] = 255

    nl['x0'] = [1] * 3
    sol = solver(**nl)
    print(sol)


    print((sol['x']-state_r).T@P_f@(sol['x']-state_r))

    xi=-(X-state_r).T@(delta_Q)@(X-state_r)+2*psi.T@P_f@phi_c@(X-state_r)+psi.T@P_f@psi

    OPT_variables=reshape(X,3,1)
    g=SX.zeros(1,1)
    g[0]=(X-state_r).T@P_f@(X-state_r)
    #g[1]=-(X-state_r).T@(delta_Q)@(X-state_r)
    #g[2]=2*psi.T@P_f@phi_c@(X-state_r) #
    #g[3]=psi.T@P_f@psi
    nlp_prob = {'f': -xi, 'x': OPT_variables, 'g': g}

    opts = {}
    opts['print_time'] = False
    opts['ipopt'] = {
        'max_iter': 1000,
        'print_level': 3,
        # 'acceptable_tol': 1e-8,
        # 'acceptable_obj_change_tol': 1e-6
    }
    # print(opts)
    solver = nlpsol("solver", "ipopt", nlp_prob, opts)
    nl = {}
    nl['lbx'] = [0 for i in range(3)]
    nl['ubx'] = [0 for i in range(3)]
    nl['lbx'][0] = 0
    nl['ubx'][0] = 2
    nl['lbx'][1] = -inf
    nl['ubx'][1] = inf
    nl['lbx'][2] = 0
    nl['ubx'][2] = 200
    nl['lbg'] = [0 for i in range(1)]
    nl['ubg'] = [0 for i in range(1)]
    nl['lbg'][0] = -inf
    nl['ubg'][0] =590
    nl['lbg'][1] = -inf
    # nl['ubg'][1] = inf
    # nl['lbg'][2] = -inf
    # nl['ubg'][2] = inf
    # nl['lbg'][3] = -inf
    # nl['ubg'][3] = inf

    nl['x0'] = [0.8,0,48.7603]
    sol = solver(**nl)
    print(sol)
    print((sol['x']-state_r).T@P_f@(sol['x']-state_r))


"""
    {'f': DM(2.40313e-009), 'g': DM([1, 2.33839e-007]), 'lam_g': DM([-0.00173027, -0.0102769]), 'lam_p': DM([]),
     'lam_x': DM([-2.81628e-010, 0, -3.11122e-011]), 'x': DM([0.943983, 16.5829, 52.141])}
    cost = 49343.5
"""

"""
{'f': DM(2.37397e-008), 'g': DM([1, 255]), 'lam_g': DM([-768.682, 0]), 'lam_p': DM([]), 'lam_x': DM([-5.06284e-008, 0, -1.61154e-010]), 'x': DM([0.710579, -10.3015, 46.6602])}
    cost = 19041
"""

