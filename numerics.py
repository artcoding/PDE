import numpy as np
from scipy.linalg import solve_banded


def solve_tridiag(a, b, c, rhs):
    """
    Solve a tridiagonal system of equations Ax = rhs, where A is a tridiagonal matrix

        [b0, c0,  0,  0,  0]
        [a1, b1, c1,  0,  0]
    A = [ 0, a2, b2, c2,  0]
        [ 0,  0, a3, b3, c3]
        [ 0,  0,  0, a4, b4]

    with sub-diagonal a, main diagonal b, and super-diagonal c.

    Parameters:
    a : array_like
        Sub-diagonal elements of the tridiagonal matrix (length n-1).
    b : array_like
        Main diagonal elements of the tridiagonal matrix (length n).
    c : array_like
        Super-diagonal elements of the tridiagonal matrix (length n-1).
    rhs : array_like
        Right-hand side vector (length n).

    Returns:
    x : ndarray
        Solution vector (length n).
    """
    n = len(rhs)

    if len(b) != n or len(a) < n or len(c) < n - 1:
        raise ValueError("Invalid input lengths.")
    
    # Create the banded matrix representation for solve_banded
    ab = np.zeros(shape=(3, n))
    
    ab[0, 1:] = c[:n-1]  # Super-diagonal
    ab[1] = b   # Main diagonal
    ab[2, :n-1] = a[1:n]  # Sub-diagonal

    # Solve the system using scipy's solve_banded function
    x = solve_banded((1, 1), ab, rhs, overwrite_ab=True, overwrite_b=True)
    
    return x


if __name__ == "__main__":
    # Example usage
    a = np.array([np.nan, 1, 1, 1, 1])  # Sub-diagonal
    b = np.array([5, 4, 3, 2, 1])       # Main diagonal
    c = np.array([2, 2, 2, -1])         # Super-diagonal
    
    rhs = np.array([3, -1, 4, 2, 2])    # Right-hand side

    solution = solve_tridiag(a, b, c, rhs)
    print("Solution:", solution)
