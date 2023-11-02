import numpy as onp
import jax.numpy as np

# Drone constants 
_n = 18
_m = 4
# _dt = 1.0/200.0
# J = onp.diag([0.04, 0.0375, 0.0675])
J = onp.diag([0.025, 0.021, 0.043])

INV_J = onp.linalg.inv(J)

KT = 0.6
KM = 0.022
OMEGA_MAX = 10.
ARM_LENGTH = 0.15
MASS = 0.85
INV_MASS = 1.0/MASS
SQRT2 = onp.sqrt(2)

e_z = onp.array([0.,0.,1.])

g_vec = -9.81 * e_z

D = onp.diag([0.01,0.01,0.1])*0.

def hat(w):
    return np.array([
        [0.,      -w[2], w[1]],
	    [w[2],    0.,   -w[0]],
	    [-w[1],   w[0],   0.]]
    )

class DroneDynamics(object):

    def __init__(self) -> None:
        self._n = _n 
        self._m = _m
        def f(x, u):
            u = 2*np.clip(u, 0., 7.)
            # g = x[0:16].reshape((4,4))
            # R, p = trans_to_Rp(g)
            R = x[:9].reshape((3,3))
            p = x[9:12]
            w = x[12:15]
            v = x[15:]
            # twist   = x[16:]

            f_t = u[0] + u[1] + u[2] + u[3]
            tau = np.array([
                            ARM_LENGTH * (u[0] + u[1] - u[2] - u[3])/SQRT2,
                            ARM_LENGTH * (-u[0] + u[1] + u[2] - u[3])/SQRT2,
                            KM * (u[0] - u[1] + u[2] - u[3])
            ])

            wdot = INV_J @ (tau - np.cross(w, J@w))
            vdot = g_vec + INV_MASS * R @ (e_z*f_t) #- R@D@R.T@v 
            pdot = v 
            dR = (R @ hat(w)).ravel()
            # dg = (g @ hat(twist)).ravel()
            return np.concatenate([dR, pdot, wdot, vdot])
        self.f = f
        # self.fdx = jit(jacfwd(self.f, argnums=0))
        # self.fdu = jit(jacfwd(self.f, argnums=1))

if __name__=='__main__':
    drone = DroneDynamics()

    R = np.eye(3)
    p = np.ones(3)*2
    w = np.ones(3)*3
    v = np.ones(3)*4
    x = np.concatenate([R.ravel(), p, w, v])
    print(x)
    R = x[:9].reshape((3,3))
    p = x[9:12]
    omega   = x[12:15]
    v       = x[15:]
    print(R, p, omega, v)
    u = np.zeros(4)

    drone.f(x, u)