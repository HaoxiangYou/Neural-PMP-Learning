from functools import partial
from matplotlib.pyplot import step
import numpy as onp
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as np

from jax import jacfwd, grad, value_and_grad, jit
from jax.lax import scan
from jax.ops import index_update, index

from integrators import rk4, euler

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

class MPC(object):
    """Create MPC controller"""
    def __init__(self, dyn, ell, 
                T=200, _dt=0.01, _max_iter=100, _step_size=1e-3):
        def F(x, u):
            x = euler(dyn.f, x, u, _dt)
            return x, ell(x, u)
        self.F = jit(F)
        def J(u, x0):
            xf, Jt = scan(F, x0, u)
            return np.mean(Jt)
        self.J = J
        dJ = jit(value_and_grad(J))
        self.dJ = dJ
        self.u = np.ones((T, dyn._m))
        def _update_plan(u, x0):
            for k in range(_max_iter):
                _J, _du = dJ(u, x0)
                u = u - _step_size * _du
            return u
        self._update_plan = _update_plan
    def plan(self, x0):
        self.u = self._update_plan(self.u, x0)
        u_now = self.u[0].copy()
        self.u = index_update(self.u, index[:-1,:], self.u[1:,:])
        self.u = index_update(self.u, index[-1,:], 1.)
        return u_now

if __name__=='__main__':
    from drone.drone_dynamics import DroneDynamics, g_vec
    from drone.transform_lib import euler2mat
    drone = DroneDynamics()
    R = euler2mat([0.2,0.4,0.5])
    p = np.ones(3)*1.
    w = np.ones(3)*0.
    v = np.ones(3)*0.
    x = np.concatenate([R.ravel(), p, w, v])

    Rd = onp.eye(3)
    Q = onp.diag([1.0,1.0,100.])
    def ell(x, u):
        R = x[:9].reshape((3,3))
        p = x[9:12]
        w = x[12:15]
        v = x[15:]
        # tr_RtR = np.trace(R.T@Rd)
        # _arc_c_arg = (tr_RtR - 1.)/2.0
        # _th = np.arccos(_arc_c_arg)
        R_err = np.linalg.norm(np.eye(3)-R)
        _p_err = np.sum(Q@np.square(p - np.array([0.,0.,0.])))
        return _p_err + R_err\
                    + 1e-4*np.sum(np.square(v)) \
                    + 1e-4*np.sum(np.square(w)) 
    _dt = 0.01
    mpc = MPC(drone, ell, _dt=_dt)
    sim_step = jit(lambda x, u: rk4(drone.f, x, u, _dt))
    traj = []
    for t in range(1000):
        u_now = mpc.plan(x)
        x = sim_step(x, u_now)
        traj.append(x.copy())
        print(get_state_dict(x))
    import matplotlib.pyplot as plt
    plt.plot(onp.stack(traj)[:,9:12])
    plt.show()
