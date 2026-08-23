import numpy as np
from numerics import solve_tridiag
from instr import Option
from market import Market

TIME_STEPS = 400
SPACE_STEPS = 400


def boundary(opt: Option, mkt: Market, x_min: float, x_max: float, tau: float):
    """
    Define the boundary conditions for the Crank-Nicolson method.

    Returns:
    tuple: A tuple containing the lower and upper boundary conditions.
    """

    if opt.type == 'C':
        lower = 0.0
        upper = mkt.spot * np.exp(x_max - mkt.q * tau) - opt.strike * np.exp(-mkt.r * tau)
    elif opt.type == 'P':
        lower = opt.strike * np.exp(-mkt.r * tau) - mkt.spot * np.exp(x_min - mkt.q * tau)
        upper = 0.0

    return lower, upper


def _calc_once(opt: Option, mkt: Market, x_grid):

    h = x_grid[1] - x_grid[0]
    k = opt.ttm / TIME_STEPS  # time step size

    s = mkt.spot * np.exp(x_grid)

    pde_a = 0.5 * mkt.vol * mkt.vol
    pde_b = mkt.r - mkt.q - pde_a
    pde_c = -mkt.r

    # Half steps for the Euler scheme, hence k := k / 2
    Euler_im1 = np.full(SPACE_STEPS - 1, 0.5 * k / h * (pde_a / h - 0.5 * pde_b))
    Euler_i   = np.full(SPACE_STEPS - 1, 1 + 0.5 * k * (pde_c - 2 * pde_a / h / h))
    Euler_ip1 = Euler_im1 + 0.5 * k * pde_b / h

    fill_a = 0.5 * k * (pde_a - 0.5 * pde_b * h)
    fill_b = h * h * (0.5 * pde_c * k - 1) - k * pde_a
    fill_b1 = fill_b + 2 * h * h
    fill_c = 0.5 * k * (pde_a + 0.5 * pde_b * h)

    A = np.full(SPACE_STEPS - 1, fill_a)
    B = np.full(SPACE_STEPS - 1, fill_b)
    C = np.full(SPACE_STEPS - 1, fill_c)

    u = opt.payoff(s)  # initial condition at maturity

    tau = 0.0

    # Do four Euler half steps
    for _ in range(4):
        tau += 0.5 * k
        u_new = np.zeros(SPACE_STEPS + 1)
        u_new[0], u_new[-1] = boundary(opt, mkt, x_grid[0], x_grid[-1], tau)

        u_new[1:-1] = Euler_im1 * u[:-2] + Euler_i * u[1:-1] + Euler_ip1 * u[2:]
        u = u_new

    for _ in range(3, TIME_STEPS + 1):
        # Crank-Nicolson steps
        tau += k
        u_new = np.zeros(SPACE_STEPS + 1)
        u_new[0], u_new[-1] = boundary(opt, mkt, x_grid[0], x_grid[-1], tau)

        D = -fill_a * u[:-2] - fill_b1 * u[1:-1] - fill_c * u[2:]        
        D[0] += -fill_a * u_new[0]
        D[SPACE_STEPS - 2] += -fill_c * u_new[-1]

        # Solve the tridiagonal system
        u_new[1:-1] = solve_tridiag(A, B, C, D)

        # Update the solution
        u = u_new
    
    return u, s


def crank_nicolson_bump(opt: Option, mkt: Market, calc_greeks: bool = False):
    period_vol = mkt.vol * np.sqrt(opt.ttm)

    x = np.linspace(-5 * period_vol, 5 * period_vol, SPACE_STEPS + 1) + np.log(opt.strike / mkt.spot)  # center grid at ATM

    bump = 0.1
    base_value, spots = _calc_once(opt, mkt, x)

    greeks = {}
    if calc_greeks:
        print("In bump")
        up_value, _   = _calc_once(opt, Market(spot=mkt.spot, r=mkt.r, q=mkt.q, vol=mkt.vol + bump), x)
        down_value, _ = _calc_once(opt, Market(spot=mkt.spot, r=mkt.r, q=mkt.q, vol=mkt.vol - bump), x)

        greeks['vega'] = 0.5 * (up_value - down_value) / bump
        greeks['vega_conv'] = (up_value - 2 * base_value + down_value) / bump / bump

        up_value, _   = _calc_once(opt, Market(spot=mkt.spot, r=mkt.r + bump, q=mkt.q, vol=mkt.vol), x)
        down_value, _ = _calc_once(opt, Market(spot=mkt.spot, r=mkt.r - bump, q=mkt.q, vol=mkt.vol), x)

        greeks['rho'] = 0.5 * (up_value - down_value) / bump
        greeks['rho_conv'] = (up_value - 2 * base_value + down_value) / bump / bump

    return base_value, spots, greeks


def crank_nicolson_value(opt: Option, mkt: Market, calc_greeks: bool = False):

    period_vol = mkt.vol * np.sqrt(opt.ttm)

    h = 10 * period_vol / SPACE_STEPS  # log-space step size
    k = opt.ttm / TIME_STEPS  # time step size

    x = np.linspace(-5 * period_vol, 5 * period_vol, SPACE_STEPS + 1) + np.log(opt.strike / mkt.spot)  # center grid at ATM
    s = mkt.spot * np.exp(x)

    pde_a = 0.5 * mkt.vol * mkt.vol
    pde_b = mkt.r - mkt.q - pde_a
    pde_c = -mkt.r

    # Half steps for the Euler scheme, hence k := k / 2
    Euler_im1 = np.full(SPACE_STEPS - 1, 0.5 * k / h * (pde_a / h - 0.5 * pde_b))
    Euler_i   = np.full(SPACE_STEPS - 1, 1 + 0.5 * k * (pde_c - 2 * pde_a / h / h))
    Euler_ip1 = Euler_im1 + 0.5 * k * pde_b / h

    fill_a = 0.5 * k * (pde_a - 0.5 * pde_b * h)
    fill_b = h * h * (0.5 * pde_c * k - 1) - k * pde_a
    fill_b1 = fill_b + 2 * h * h
    fill_c = 0.5 * k * (pde_a + 0.5 * pde_b * h)

    A = np.full(SPACE_STEPS - 1, fill_a)
    B = np.full(SPACE_STEPS - 1, fill_b)
    C = np.full(SPACE_STEPS - 1, fill_c)

    u = opt.payoff(s)  # initial condition at maturity

    greeks = {}
    if calc_greeks:
        vega = np.zeros(SPACE_STEPS + 1, dtype=np.float64)
        rho = np.zeros(SPACE_STEPS + 1, dtype=np.float64)

    tau = 0.0
    for j in range(1, TIME_STEPS + 1):

        u_new = np.zeros(SPACE_STEPS + 1)
        u_new[0], u_new[-1] = boundary(opt, mkt, x[0], x[-1], tau)

        if j < 5:
            # Do four Euler half steps
            tau += 0.5 * k
            u_new[1:-1] = Euler_im1 * u[:-2] + Euler_i * u[1:-1] + Euler_ip1 * u[2:]
            u = u_new

        else:
            # Crank-Nicolson steps
            tau += k
            D = -fill_a * u[:-2] - fill_b1 * u[1:-1] - fill_c * u[2:]
            
            D[0] += -fill_a * u_new[0]
            D[SPACE_STEPS - 2] += -fill_c * u_new[-1]

            # Solve the tridiagonal system
            u_new[1:-1] = solve_tridiag(A, B, C, D)

        if calc_greeks:
            v_new = np.zeros(SPACE_STEPS + 1, dtype=np.float64)
            rho_new = np.zeros(SPACE_STEPS + 1, dtype=np.float64)

            if opt.type == 'C':
                rho_new[0] = 0.0
                rho_new[-1] = tau * opt.strike * np.exp(-mkt.r * tau)
            else:
                rho_new[0] = -tau * opt.strike * np.exp(-mkt.r * tau)
                rho_new[-1] = 0.0

            if j < 5:
                # Do four Euler half steps
                uu = u[1:-1]
                u_x = 0.5 * (u[2:] - u[:-2]) / h
                u_xx = (u[2:] - 2 * u[1:-1] + u[:-2]) / h / h

                v_new[1:-1] = Euler_im1 * vega[:-2] + Euler_i * vega[1:-1] + Euler_ip1 * vega[2:] - mkt.vol * (u_x - u_xx)
                rho_new[1:-1] = Euler_im1 * rho[:-2] + Euler_i * rho[1:-1] + Euler_ip1 * rho[2:] - (uu - u_x)
                vega = v_new
                rho = rho_new

            u_x = h * k / 4 * (u[2:] - u[:-2] + u_new[2:] - u_new[:-2])
            u_xx = k / 2 * (u[2:] - 2 * u[1:-1] + u[:-2] + u_new[2:] - 2 * u_new[1:-1] + u_new[:-2])


            D = -fill_a * vega[:-2] - fill_b1 * vega[1:-1] - fill_c * vega[2:]
            D += mkt.vol * (u_x - u_xx)
            
            v_new[1:-1] = solve_tridiag(A, B, C, D)
            vega = v_new

            uu = h * h * k / 2 * (u[1:-1] + u_new[1:-1])
    
            D = -fill_a * rho[:-2] - fill_b1 * rho[1:-1] - fill_c * rho[2:]
            D += uu - u_x

            D[0] += -fill_a * rho_new[0]
            D[SPACE_STEPS - 2] += -fill_c * rho_new[-1]

            rho_new[1:-1] = solve_tridiag(A, B, C, D)
            rho = rho_new

        # Update the solution
        u = u_new
    
    if calc_greeks:
        greeks['vega'] = vega
        greeks['rho'] = rho

    return u, s, greeks


