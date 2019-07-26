from casadi import SX, DM, vertcat, reshape, Function, nlpsol


#func_type: 0 discrete, 1 continous
def J_qi(T,Q,R,P_f,N_pred,states,controls,rhs,func_type=1):

    # kriegen die Anzahl der Zeiler("shape" ist ein Tuple)
    n_states = states.shape[0]  # p 22.

    # Eingangsanzahl
    n_controls = controls.shape[0]

    # nonlinear mapping function f(x,u)
    f = Function('f', [states, controls], [rhs])
    # Decision variables (controls) u_pwm
    U = SX.sym('U', n_controls, N_pred)
    # parameters (which include the !!!initial and the !!!reference state of
    # the robot!!!reference input)
    P = SX.sym('P', n_states + n_states + n_controls)
    # number of x is always N+1   -> 0..N
    X = SX.sym('X', n_states, (N_pred + 1))

    X[:, 0] = P[0:n_states]
    obj = 0

    # compute predicted states and cost function
    for k in range(N_pred):
        st = X[:, k]
        con = U[:, k]
        f_value = f(st, con)
       # print(f_value)
        st_next = st + (T * f_value)
        X[:, k + 1] = st_next
        obj = obj + (st - P[n_states:2*n_states]).T@Q@(st - P[n_states:2*n_states]) + (con-P[2*n_states:2*n_states+n_controls]).T@R@(con-P[2*n_states:2*n_states+n_controls])
    # print(obj)
    st = X[:, N_pred]
    obj = obj + (st - P[n_states:2*n_states]).T @ P_f @ (st - P[n_states:2*n_states])
    # create a dense matrix of state constraints. Size of g: n_states * (N_pred+1) +1
    g = SX.zeros(n_states * (N_pred + 1) + 1, 1)

    # Constrainsvariable spezifizieren
    for i in range(N_pred + 1):
        for j in range(n_states):
            g[n_states * i + j] = X[j, i]

    g[n_states * (N_pred + 1)] = (X[:, N_pred] - P[n_states:2*n_states]
                           ).T@ P_f @ (X[:, N_pred] - P[n_states:2*n_states])
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
    return f, nlpsol("solver", "ipopt", nlp_prob, opts)

def J_fh(T,Q,R,N_pred,states,controls,rhs,func_type=1):

    # kriegen die Anzahl der Zeiler("shape" ist ein Tuple)
    n_states = states.shape[0]  # p 22.

    # Eingangsanzahl
    n_controls = controls.shape[0]

    # nonlinear mapping function f(x,u)
    f = Function('f', [states, controls], [rhs])
    # Decision variables (controls) u_pwm
    U = SX.sym('U', n_controls, N_pred)
    print((U))
    # parameters (which include the !!!initial and the !!!reference state of
    # the robot!!!reference input)
    P = SX.sym('P', n_states + n_states + n_controls)
    # number of x is always N+1   -> 0..N
    X = SX.sym('X', n_states, (N_pred + 1))

    X[:, 0] = P[0:n_states]
    obj = 0

    # compute predicted states and cost function
    for k in range(N_pred):
        st = X[:, k]
        con = U[:, k]
        f_value = f(st, con)
       # print(f_value)
        st_next = st + (T * f_value)
        X[:, k + 1] = st_next
        obj = obj + (st - P[n_states:2*n_states]).T@Q@(st - P[n_states:2*n_states]) + (con-P[2*n_states:2*n_states+n_controls]).T@R@(con-P[2*n_states:2*n_states+n_controls])
    # create a dense matrix of state constraints. Size of g: n_states * (N_pred+1)
    g = SX.zeros(n_states * (N_pred + 1), 1)

    # Constrainsvariable spezifizieren
    for i in range(N_pred + 1):
        for j in range(n_states):
            g[n_states * i + j] = X[j, i]

    OPT_variables = reshape(U, N_pred*n_controls, 1)
    print((OPT_variables))
    nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}
    opts = {}
    opts['print_time'] = False
    opts['ipopt'] = {
        'max_iter': 100,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6}
    # print(opts)
    return f, nlpsol("solver", "ipopt", nlp_prob, opts)

