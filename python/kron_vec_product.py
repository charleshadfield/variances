# Taken from this https://gist.github.com/ahwillia/f65bc70cb30206d4eadec857b98c4065
import numpy as np
import numpy.random as npr
from functools import reduce

# Goal
# ----
# Compute (As[0] kron As[1] kron ... As[-1]) @ v

# ==== HELPER FUNCTIONS ==== #

def unfold(tens, mode, dims):
    """
    Unfolds tensor into matrix.
    Parameters
    ----------
    tens : ndarray, tensor with shape == dims
    mode : int, which axis to move to the front
    dims : list, holds tensor shape
    Returns
    -------
    matrix : ndarray, shape (dims[mode], prod(dims[/mode]))
    """
    if mode == 0:
        return tens.reshape(dims[0], -1)
    else:
        return np.moveaxis(tens, mode, 0).reshape(dims[mode], -1)


def refold(vec, mode, dims):
    """
    Refolds vector into tensor.
    Parameters
    ----------
    vec : ndarray, tensor with len == prod(dims)
    mode : int, which axis was unfolded along.
    dims : list, holds tensor shape
    Returns
    -------
    tens : ndarray, tensor with shape == dims
    """
    if mode == 0:
        return vec.reshape(dims)
    else:
        # Reshape and then move dims[mode] back to its
        # appropriate spot (undoing the `unfold` operation).
        tens = vec.reshape(
            [dims[mode]] +
            [d for m, d in enumerate(dims) if m != mode]
        )
        return np.moveaxis(tens, 0, mode)

# ==== KRON-VEC PRODUCT COMPUTATIONS ==== #

def kron_vec_prod(As, v):
    """
    Computes matrix-vector multiplication between
    matrix kron(As[0], As[1], ..., As[N]) and vector
    v without forming the full kronecker product.
    """
    dims = [A.shape[0] for A in As]
    vt = v.reshape(dims)
    for i, A in enumerate(As):
        vt = refold(A @ unfold(vt, i, dims), i, dims)
    return vt.ravel()


def kron_brute_force(As, v):
    """
    Computes kron-matrix times vector by brute
    force (instantiates the full kron product).
    """
    return reduce(np.kron, As) @ v


# Quick demonstration.
if __name__ == "__main__":

    # Create random problem.
    _dims = [3, 3, 3, 3, 3, 3, 3, 3]
    As = [npr.randn(d, d) for d in _dims]
    v = npr.randn(np.prod(_dims))

    # Test accuracy.
    actual = kron_vec_prod(As, v)
    expected = kron_brute_force(As, v)
    print(np.linalg.norm(actual - expected))