import jax
from functools import partial
from jax import grad, jacfwd, vmap, jit, hessian
from jax.lax import scan, stop_gradient
from jax.ops import index_update, index
import jax.random as jnp_random
import jax.numpy as np
from jax.nn.initializers import glorot_normal, normal, uniform

from jax.example_libraries import stax, optimizers
from jax.example_libraries.stax import elementwise, Tanh, Selu, Relu
from jax.flatten_util import ravel_pytree

import numpy as onp

class NeuralControlOptimizer(object):
    def __init__(self, n_var, m_var, n_const, loss, eq_constr, step_size=1e-3):
        self.loss = loss 
        self.eq_constr = eq_constr
        x_init, x_net = stax.serial(
            stax.Dense(64),Tanh,
            stax.Dense(64),Tanh,
            stax.Dense(n_var)
        )
        u_init, x_net = stax.serial(
            stax.Dense(64),Tanh,
            stax.Dense(64),Tanh,
            stax.Dense(m_var)
        )
        rho_init, rho_net = stax.serial(
            stax.Dense(64),Tanh,
            stax.Dense(64),Tanh,
            stax.Dense(n_const)
        )

        '''
        min_x,u max_lam = \int_t0^tf \ell(x,u) - \lam @ (\dot{x} - f(x,u))
        '''

        self.theta = {'x' : x0, 'lam' : lam}
        def lagrangian(theta):
            x = theta['x']
            lam = theta['lam']
            _eq_constr = eq_constr(x)
            return loss(x, u) + np.sum(lam * _eq_constr + 0.5 * (_eq_constr)**2)
        dldth = jit(grad(lagrangian))
        @jit
        def step(theta):
            _dldth = dldth(theta)
            theta['x'] = theta['x'] - step_size * _dldth['x']
            # theta['lam'] = theta['lam'] + eq_constr(theta['x'])
            theta['lam'] = theta['lam'] + step_size * _dldth['lam']
            return theta 
        self.lagrangian = lagrangian
        self.step = step
    def solve(self, max_iter=10000):
        for k in range(max_iter):
            self.theta = self.step(self.theta)