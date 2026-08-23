import numpy as np
import matplotlib.pyplot as plt

from instr import Option
from market import Market
from bsm import bsm_value

from pde_solver import PDESolver



def main():
    # Define the option and market parameters
    opt = Option(exercise='E', type='C', ttm=1, strike=90.0)
    mkt = Market(spot=100, r=0.05, q=0.02, vol=0.4)

    # Define log S PDE coefficients
    pde_a = 0.5 * mkt.vol * mkt.vol
    pde_b = mkt.r - mkt.q - pde_a
    pde_c = -mkt.r

    # x0 = np.log(mkt.spot)
    x0 = 0
    dx = 5 * mkt.vol * np.sqrt(opt.ttm)

    def euro_log_payoff(logS):
        return opt.payoff(np.exp(logS))
    
    # def lower_b():

    def lower_b(logS, t):
        return opt.strike * np.exp(-mkt.r * t) - mkt.spot * np.exp(logS - mkt.q * t) if opt.type == 'P' else 0.0
    
    def upper_b(logS, t):
        return mkt.spot * np.exp(logS - mkt.q * t) - opt.strike * np.exp(-mkt.r * t) if opt.type == 'C' else 0.0


    solver = PDESolver(
        a=(pde_c, pde_b, pde_a),        # Tuple of coefficients.
        x_domain=(x0 - dx, x0 + dx),    # Tuple of x_min, x_max
        T=opt.ttm,                      # Time to which to solve the PDE
        
        initial=euro_log_payoff,        # Function of x
        lower_boundary=lower_b,         # Function of t
        upper_boundary=upper_b          # Function of t
    )

    solver.solve()
    u = solver.solution()
    x = solver._x_grid

    scaler = (x[1] - x[0]) ** 2

    u_xx = (u[:-2] - 2 * u[1:-1] + u[2:]) / scaler
    u_x = (u[2:] - u[:-2]) / 2 / (x[1] - x[0])

    plt.figure()
    # plt.plot(mkt.spot * np.exp(solver._x_grid), u)
    plt.plot(mkt.spot * np.exp(solver._x_grid[1:-1]), (u_x))
    # plt.plot(mkt.spot * np.exp(solver._x_grid[1:-1]), (u_xx - u_x))
    plt.show()


if __name__ == "__main__":
    main()
