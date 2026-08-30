import numpy as np
import matplotlib.pyplot as plt

from instr import Option
from market import Market
from bsm import bsm_value

from pde_solver import PDESolver


def plot(spots, pvs, payoff, delta, gamma, max_spot=None):

    max_ind = len(spots) if max_spot is None else np.searchsorted(spots, max_spot, side='right')
        
    _, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].plot(spots[:max_ind], pvs[:max_ind])
    ax[0, 0].plot(spots[:max_ind], payoff[:max_ind], '--')
    ax[0, 0].set_title('PV and Payoff')

    ax[0, 1].plot(spots[1:max_ind-1], delta[:max_ind-2])
    ax[0, 1].set_title('Delta')
    ax[1, 1].plot(spots[1:max_ind-1], gamma[:max_ind-2])
    ax[1, 1].set_title('Gamma')
    plt.show()


def main():
    # Define the option and market parameters
    opt = Option(exercise='A', type='P', ttm=1, strike=90.0)
    mkt = Market(spot=100, r=0.05, q=0.02, vol=0.4)

    # Define log S PDE coefficients
    pde_a = 0.5 * mkt.vol * mkt.vol
    pde_b = mkt.r - mkt.q - pde_a
    pde_c = -mkt.r

    x0 = 0
    dx = 5 * mkt.vol * np.sqrt(opt.ttm)

    def euro_log_payoff(x):
        return opt.payoff(np.exp(x) * mkt.spot)

    def lower_b(x, t):
        return opt.strike * np.exp(-mkt.r * t) - mkt.spot * np.exp(x - mkt.q * t) if opt.type == 'P' else 0.0
    
    def upper_b(x, t):
        return mkt.spot * np.exp(x - mkt.q * t) - opt.strike * np.exp(-mkt.r * t) if opt.type == 'C' else 0.0

    def early_exercise(t):
        return opt.can_exercise(t)

    solver = PDESolver(
        a=(pde_c, pde_b, pde_a),        # Tuple of coefficients.
        x_domain=(x0 - dx, x0 + dx),    # Tuple of x_min, x_max
        T=opt.ttm,                      # Time to which to solve the PDE
        
        initial=euro_log_payoff,        # Function of x
        lower_boundary=lower_b,         # Function of t
        upper_boundary=upper_b,         # Function of t
        is_exer_time=early_exercise     # Function of t
    )

    solver.solve()
    u = solver.solution()
    x = solver._x_grid
    S = mkt.spot * np.exp(x)

    denom = x[1] - x[0]
    u_xx = (u[:-2] - 2 * u[1:-1] + u[2:]) / denom / denom
    u_x = (u[2:] - u[:-2]) / 2 / denom

    u_S = u_x / (S[1:-1])
    u_SS = (u_xx - u_x) / (S[1:-1])**2

    bsm_val, bsm_greeks = bsm_value(opt, mkt, calc_greeks=True)
    print(f"Value = {u[len(u)//2]}, BSM = {bsm_val}")
    print(f"Delta = {u_S[len(u_S)//2]}, BSM = {bsm_greeks['delta']}")
    print(f"Gamma = {u_SS[len(u_SS)//2]}, BSM = {bsm_greeks['gamma']}")

    plot(spots=S, pvs=u, payoff=opt.payoff(S), delta=u_S, gamma=u_SS, max_spot=mkt.spot * 3)

if __name__ == "__main__":
    main()
