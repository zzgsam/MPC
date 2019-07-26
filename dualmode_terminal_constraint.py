from ipydex import IPS, activate_ips_on_exception
from sys import path

from casadi import SX,DM, vertcat, reshape, Function, nlpsol, inf, norm_2,jacobian,horzcat
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
    N_pred = 100
    # Zielruhelage von den Zustaenden
    # Referenz von h, h_p, eta
    # state_r=[0.5, 0, 48]
    # activate_ips_on_exception()
    state_r = [0.8, 0, 48.760258862]
    u_r=[157.291157619]

    h_ub = 2
    h_lb = 0
    eta_ub = 200
    eta_lb = 0

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
                              A_SP) ** 2 - g, -1 / T_M * eta + k_M / T_M * u_pwm)
    # system r.h.s eta_p h_p h_pp

    P = SX.sym('P',n_states,1)
    print(rhs)
    K=-SX([[1,0.0874,0.1630]])

    print(K)
    # nonlinear mapping function f(x,u)
    epsilo=0.00031
    f = Function('f', [states, controls], [rhs])
    #P_cost=SX([[162.5528,-1,-47.6879],[-1,0.0503,0.3040],[-47.6879,0.304,14.3427]])
    P_cost=SX([[159.781,13.8916,1.8387],[13.8916,1.2518,0.16074],[1.8387,0.16074,0.2998]])
    A=SX([[0,1,0],[0,-11.4957,0.2439],[-0.54386,-0.04744,-1.8431]])
    Q_s=SX([[2,0.0874,0.163],[0.0874,1.0076,0.0143],[0.163,0.0143,1.0266]])


    obj=(states-P).T@P_cost@(f((states-P),K@(states-P)+u_r)-A@(states-P))-1/2*(states-P).T@Q_s@(states-P)+1/2*epsilo*(states-P).T@P_cost@(states-P)
    phi=Function('phi',[states],[((f((states),K@(states-state_r)+u_r)-A@(states-state_r)).T@P_cost@(f((states),K@(states-state_r)+u_r)-A@(states-state_r)))])
    temp=Function('temp',[states],[f((states),K@(states-state_r)+u_r)-A@(states-state_r)])
    print("f",f(state_r,u_r))
    state_temp=SX([0.8, 0, 48.760258862])
    print("phi_norm", phi(state_temp))
    print("x_norm",(state_temp-state_r).T@P_cost@(state_temp-state_r))
    print("phi/x", phi(state_temp)/((state_temp-state_r).T@P_cost@(state_temp-state_r)))
    print("K(X-X_r)+u_r",K@(state_temp-state_r)+u_r)
    print("f(x,kx)",f(state_temp,K@(state_temp-state_r)+u_r))
    print("A(X-X_r)",A@(state_temp-state_r))
    print("phi",temp(state_temp))
    g = SX.zeros(1, 1)
    g=norm_2(states-P)
    nlp_prob = {'f': -obj, 'x': states, 'g': g, 'p': P}
    opts = {}
    opts['print_time'] = False
    opts['ipopt'] = {
        'max_iter': 200,
        'print_level': 0,
        'acceptable_tol': 1e-4,
        # 'fixed_variable_treatment':'make_constraint',
        'acceptable_constr_viol_tol': 1e-4,
        'print_info_string': 'no',
        'acceptable_obj_change_tol': 1e-5}
    # print(opts)
    solver = nlpsol("solver", "ipopt", nlp_prob, opts)
    print(solver)

    nl = {}

    nl['lbx'] = [0 for i in range(n_states)]
    nl['ubx'] = [0 for i in range(n_states)]
    # ??? better method?
    # nl['lbx'][0] = h_lb
    # nl['ubx'][0] = h_ub
    # nl['lbx'][0] = 0.8
    # nl['ubx'][0] = 0.8
    # nl['lbx'][1] = 0
    # nl['ubx'][1] = 0
    # nl['lbx'][2] = eta_lb
    # nl['ubx'][2] = eta_ub
    # nl['lbx'][2] = 48.753
    # nl['ubx'][2] = 48.753
    nl['lbx'][0] = -inf
    nl['ubx'][0] = inf
    nl['lbx'][1] = -inf
    nl['ubx'][1] = inf
    nl['lbx'][2] = -inf
    nl['ubx'][2] = inf

    # multi-constraints defined by list
    # initialisation
    nl['lbg'] = [0 for i in range(1)]
    nl['ubg'] = [0 for i in range(1)]
    nl['lbg'][0] = -inf
    nl['ubg'][0] = 0.015863


    nl['p'] = [0.8, 0, 48.760258862]
    nl['x0'] = [0 for i in range(n_states)]

    sol = solver(**nl)
    print(sol)