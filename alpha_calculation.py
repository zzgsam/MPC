from casadi import SX,DM, vertcat, reshape, Function, nlpsol, inf, norm_2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #alpha_1 calculation
    n_states=3

    h_ub = 2
    h_lb = 0
    eta_ub = 200
    eta_lb = 0

    X_alpha1 = SX.sym('X', n_states,1 )
    P_f = SX.zeros(3, 3)
    P_f[0,0]=1.0439e+007
    P_f[0,1] =-3.0878e+007
    P_f[0,2] =-1.6277e+009
    P_f[1,0]=-3.0878e+007
    P_f[1,1] =1.6483e+008
    P_f[1,2] = 6.4157e+009
    P_f[2,0]=-1.6277e+009
    P_f[2,1] =6.4157e+009
    P_f[2,2] = 2.9567e+011

    obj_alpha1 = -X_alpha1.T@P_f@X_alpha1
    K=SX.zeros(1,3)
    K[0,0]=3162.3
    K[0,1]=256.11
    K[0,2]=12.303

    g=SX.zeros(1,1)
    g = -K@X_alpha1
    P=[]
    OPT_variables = X_alpha1

    nlp_prob = {'f': obj_alpha1, 'x': OPT_variables, 'g': g, 'p': P}
    print(nlp_prob)
    opts = {}
    opts['print_time'] = False
    opts['ipopt'] = {
        'max_iter': 100,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6}
    # print(opts)
    solver = nlpsol("solver", "ipopt", nlp_prob, opts)
    nl = {}
    nl['lbx'] = [0 for i in range(n_states)]
    nl['ubx'] = [0 for i in range(n_states)]
    nl['lbx'][0:3] = [h_lb,-inf,eta_lb]
    nl['ubx'][0:3] = [h_ub, inf, eta_ub]

    nl['lbg'] = [0]
    nl['ubg'] = [255]
    sol = solver(**nl)
    print(sol)

    #alpha calculation
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
    N_pred = 50
    # Zielruhelage von den Zustaenden
    # Referenz von h, h_p, eta
    state_r = [0.9, 0, 48]
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
                                  A_SP) ** 2 - g, -1 / T_M * eta + k_M / T_M * u_pwm)
    # system r.h.s eta_p h_p h_pp
    # nonlinear mapping function f(x,u)
    f = Function('f', [states, controls], [rhs])
    print(rhs)
    X_alpha = SX.sym('X_alpha', n_states, 1)
    A_k = SX.zeros(3,3)
    A_k[0,0] = 0
    A_k[0,1] = 1
    A_k[0,2] = 0
    A_k[1,0] = 0
    A_k[1,1] = -11.496
    A_k[1,2] = 0.24395
    A_k[2,0] = -1719.8
    A_k[2,1] = -139.29
    A_k[2,2] = -8.4457
    #print(A_k)
    phi= f(X_alpha, K@X_alpha) - A_k@X_alpha
    obj_alpha = -(X_alpha.T @ P_f @ phi-2*X_alpha.T @ P_f @ X_alpha)
    g_alpha=SX.zeros(1,1)
    g_alpha = X_alpha.T @ P_f @ X_alpha
    OPT_alpha = X_alpha
    nlp_prob_alpha = {'f': obj_alpha, 'x': OPT_alpha, 'g': g_alpha, 'p': P}
    opts_alpha = {}
    opts_alpha['print_time'] = False
    opts_alpha['ipopt'] = {
        'max_iter': 100,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6}
    solver_alpha = nlpsol("solver_alpha", "ipopt", nlp_prob_alpha, opts_alpha)
    nl_alpha = {}
    nl_alpha['lbx'] = [0 for i in range(n_states)]
    nl_alpha['ubx'] = [0 for i in range(n_states)]
    nl_alpha['lbx'][0:3] = [h_lb,-inf,eta_lb]
    nl_alpha['ubx'][0:3] = [h_ub, inf, eta_ub]
    print(g_alpha)
    nl_alpha['lbg'] = [-inf]
    nl_alpha['ubg'] = [float(-sol['f'][0])-1e16-1802159683811731.874]
    sol_alpha = solver_alpha(**nl_alpha)
    print(nl_alpha['ubg'])
    print(sol_alpha)
