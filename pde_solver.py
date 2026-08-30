from typing import Iterable, Callable, Tuple

from dataclasses import dataclass
import numpy as np 

from numerics import solve_tridiag


@dataclass
class TriagMatrix:
    diag: np.ndarray = None
    lower: np.ndarray = None
    upper: np.ndarray = None


@dataclass
class PDESolver:
    """
    Solves the PDE with constant coefficients:

    u_t = a_0 u + a_1 u_x + a_2 u_xx

    on domain [x_min...x_max, 0...T] 
    with inital condition               u(x, 0) = initial(x)
    and Dirichlet boundary conditions   u(x_min, t) = lower_boundary(t)
                                        u(x_min, t) = upper_boundary(t)

    """

    # Required arguments
    a: Tuple[float, float, float]   # Tuple of coefficients.
    x_domain: Tuple[float, float]   # Tuple of x_min, x_max
    T: float                        # Time to which to solve the PDE
    
    initial: Callable               # Function of x
    lower_boundary: Callable        # Function of x, t
    upper_boundary: Callable        # Function of x, t

    # Optional grid set up
    t_steps: int = 500
    x_steps: int = 500
    euler_half_steps: int = 4

    # Internal variables

    # Regular grid of x points.
    # Should be initialiazed during the base solve.
    # Keep fixed for greeks.
    _x_grid: np.ndarray | None = None

    # Hold base run solution at T after solve()
    _u: np.ndarray | None = None

    # Parameters of the tri-diagonal matrix
    _m: TriagMatrix = None


    def __post__init__(self):
        assert(len(self.a) == 3)
        assert(self.x_domain[0] < self.x_domain[1])
        assert(self.T > 0)
        assert(self.euler_half_steps % 2 == 0)

    def _setup(self):
        self._x_grid = np.linspace(self.x_domain[0], self.x_domain[1], self.x_steps + 1)
        # Space step
        h = self._x_grid[1] - self._x_grid[0]

        # Time step
        k = self.T / self.t_steps

        kah2 = k * self.a[2] / h / h
        kb4h = 0.25 * k * self.a[1] / h

        mat_size = self.t_steps - 1
        self._m = TriagMatrix(
            diag=np.full(mat_size, 1 + kah2 - 0.5 * k * self.a[0]),
            lower=np.full(mat_size, -0.5 * kah2 + kb4h),
            upper=np.full(mat_size, -0.5 * kah2 - kb4h)
        )

    def _init_next(self, t: float) -> np.ndarray:
        u = np.zeros_like(self._u)
        u[0]  = self.lower_boundary(self._x_grid[0], t)
        u[-1] = self.upper_boundary(self._x_grid[-1], t)
        return u

    def _matrix_rhs(self):
        rhs = (2 - self._m.diag[0]) * self._u[1:-1] - self._m.lower[0] * self._u[:-2] - self._m.upper[0] * self._u[2:]
        return rhs        

    def solve(self): 
        if self._x_grid is None:
            self._setup()
        
        dt = self.T / self.t_steps
        self._u = self.initial(self._x_grid)

        tau = 0
        # Euler steps first
        for _ in range(self.euler_half_steps):
            tau += 0.5 * dt
            u_next = self._init_next(tau)
            u_next[1:-1] = self._matrix_rhs()
            self._u = u_next

        # Crank-Nicolson steps
        for _ in range(self.t_steps - self.euler_half_steps // 2):
            tau += dt

            u_next = self._init_next(tau)
            rhs = self._matrix_rhs()
            rhs[0]  -= self._m.lower[0] * u_next[0]
            rhs[-1] -= self._m.upper[0] * u_next[-1]

            u_next[1:-1] = solve_tridiag(self._m.lower, self._m.diag, self._m.upper, rhs)
            self._u = u_next
    
    def solution(self):
        return self._u
