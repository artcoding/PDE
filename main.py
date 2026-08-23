import numpy as np
import matplotlib.pyplot as plt

from instr import Option
from market import Market
from bsm import bsm_value
from crank import crank_nicolson_value, crank_nicolson_bump


def plot(spots, payoff, bsm, bsm_vega, bsm_rho, cn_spots, cn_values, cn_vega, cn_rho):
    plt.figure()
    # plt.xscale('log')
    plt.plot(spots, payoff, label='Asymptotic', color='red', linestyle='--')
    plt.scatter(spots, bsm, label='BSM Price', color='blue', facecolors='none')
    plt.plot(cn_spots, cn_values, label='Crank-Nicolson Price', color='orange')
    plt.xlim(np.min(spots), np.max(spots))
    plt.ylim(-0.1 * np.max(bsm), np.max(bsm) * 1.1)
    plt.legend()
    plt.title("Option Price vs Spot Price")
    plt.show()

    plt.figure()
    plt.plot(cn_spots, cn_vega, label='Crank-Nicolson Vega', color='green')
    plt.scatter(spots, bsm_vega, label='BSM Vega', color='blue', facecolors='none')
    plt.legend()
    plt.title("Vega vs Spot")
    plt.show()

    plt.figure()
    plt.plot(cn_spots, cn_rho, label='Crank-Nicolson Rho', color='purple')
    plt.scatter(spots, bsm_rho, label='BSM Rho', color='blue', facecolors='none')
    plt.legend()
    plt.title("Rho vs Spot")
    plt.show()


def plot_delta_gamma(x_spots, cn_spots, cn_values):
    # x = np.diff(x_spots)
    x = x_spots[2:] - x_spots[:-2]
    u_x = (cn_values[2:] - cn_values[:-2]) / x
    u_S = u_x / cn_spots[1:-1]

    den = (x_spots[1] - x_spots[0]) ** 2
    u_xx = (cn_values[:-2] - 2 * cn_values[1:-1] + cn_values[2:]) / den

    u_SS = (u_xx - u_x) / (cn_spots[1:-1] ** 2)

    # x = cn_spots[:-1] + 0.5 * x
    plt.figure()
    plt.plot(cn_spots[1:-1], u_S)
    # plt.plot(cn_spots[1:-1], u_SS)
    # plt.xlim(0, 200)
    plt.show()



def assymptote(opt: Option, mkt: Market, spots):

    values = np.array([s * np.exp(-mkt.q * opt.ttm) - opt.strike * np.exp(-mkt.r * opt.ttm) for s in spots])
    if opt.type == 'P':
        values = -values

    return np.maximum(values, 0.0)

def main():
    # Define the option and market parameters
    opt = Option(exercise='E', type='C', ttm=1, strike=90.0)
    mkt = Market(spot=100, r=0.05, q=0.02, vol=0.4)

    # spots = np.linspace(10, 250, 40)

    # cn_values, cn_spots, cn_greeks = crank_nicolson_value(opt, mkt, calc_greeks=True)
    cn_values, cn_spots, cn_greeks = crank_nicolson_bump(opt, mkt, calc_greeks=True)

    payoffs = assymptote(opt, mkt, cn_spots)

    bsm_vector = np.zeros_like(cn_spots)
    vega_vector = np.zeros_like(cn_spots)
    rho_vector = np.zeros_like(cn_spots)

    # bump_vega = np.zeros_like(cn_spots)
    # bump = 0.001
    # vals_p, _, _ = crank_nicolson_value(opt, Market(spot=mkt.spot, r=mkt.r, q=mkt.q, vol=mkt.vol + bump), calc_greeks=False)
    # # val_p = np.interp(s, spots, vals_p)
    # vals_m, _, _ = crank_nicolson_value(opt, Market(spot=mkt.spot, r=mkt.r, q=mkt.q, vol=mkt.vol - bump), calc_greeks=False)
    # # val_m = np.interp(s, spots, vals_p)
    # bump_vega = (vals_p - vals_m) / 2 / bump

    # for i, s in enumerate(cn_spots):
    #     bump = 0.0001
    #     vals_p, spots, _ = crank_nicolson_value(opt, Market(spot=s, r=mkt.r, q=mkt.q, vol=mkt.vol + bump), calc_greeks=False)
    #     val_p = np.interp(s, spots, vals_p)
    #     vals_m, spots, _ = crank_nicolson_value(opt, Market(spot=s, r=mkt.r, q=mkt.q, vol=mkt.vol - bump), calc_greeks=False)
    #     val_m = np.interp(s, spots, vals_p)
    #     bump_vega[i] = (val_p - val_m) / 2 / bump

    for i, s in enumerate(cn_spots):
        b, greeks = bsm_value(opt, Market(spot=s, r=mkt.r, q=mkt.q, vol=mkt.vol), calc_greeks=True)
        bsm_vector[i] = b
        vega_vector[i] = greeks['vega']
        rho_vector[i] = greeks['rho']


    print("BSM Value:", bsm_value(opt, mkt))
    cn_interpolated_value = np.interp(mkt.spot, cn_spots, cn_values)
    print("Crank-Nicolson Value:", cn_interpolated_value)

    plot(cn_spots, payoffs, bsm_vector, vega_vector, rho_vector, cn_spots, cn_values, cn_greeks['vega'], cn_greeks['rho'])
    # plot(cn_spots, payoffs, bsm_vector, vega_vector, rho_vector, cn_spots, cn_values, bump_vega, cn_greeks['rho'])

    x_spots = np.log(cn_spots / mkt.spot)

    # plot_delta_gamma(x_spots, cn_spots, cn_values)

if __name__ == "__main__":
    main()
