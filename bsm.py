from instr import Option
from market import Market

from math import sqrt, log, exp
from scipy.stats import norm


def bsm_value(opt: Option, mkt: Market, calc_greeks: bool = False):

    f = mkt.spot * exp((mkt.r - mkt.q) * opt.ttm)  # forward
    disc = exp(-mkt.r * opt.ttm)

    sign = 1 if opt.type == 'C' else -1

    period_vol = mkt.vol * sqrt(opt.ttm)
    # Check if very low vol or close to expiry
    if period_vol < 0.0001:
        moneyness = sign * (f - opt.strike)
        if moneyness > 0.0:     # option in the money
            return moneyness * disc
        return 0.0

    d1 = (log(mkt.spot / opt.strike) + (mkt.r-mkt.q) * opt.ttm) / period_vol + 0.5 * period_vol
    d2 = d1 - period_vol

    fv = sign * (f * norm.cdf(sign * d1) - opt.strike * norm.cdf(sign * d2))

    greeks = {}
    if calc_greeks:
        delta = sign * norm.cdf(sign * d1) * exp(-mkt.q * opt.ttm)
        vega = mkt.spot * exp(-mkt.q * opt.ttm) * norm.pdf(d1) * sqrt(opt.ttm)
        rho = sign * opt.ttm * opt.strike * exp(-mkt.r * opt.ttm) * norm.cdf(sign * d2)
        greeks = {'delta': delta, 'vega': vega, 'rho': rho}

    return fv * disc, greeks
    