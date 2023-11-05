def rk4_step(f, x, u, dt):
    """
        Input:
            xdot = f(x) - function to be integrated, passed as f
            x - initial condition to the function 
            dt - time step 
        Output: 
            x[t+dt] 
    """
    # one step of runge-kutta integration
    k1 = f(x, u)
    k2 = f(x + dt*k1/2, u)
    k3 = f(x + dt*k2/2, u)
    k4 = f(x + dt*k3, u)
    xdot = x + 1/6*(k1+2*k2+2*k3+k4)*dt
    return xdot 

def get_state_dict(x):
    R = x[:9].reshape((3,3))
    p = x[9:12]
    w = x[12:15]
    v = x[15:]
    return {
        'R': R,
        'p': p,
        'w': w,
        'v': v
    }