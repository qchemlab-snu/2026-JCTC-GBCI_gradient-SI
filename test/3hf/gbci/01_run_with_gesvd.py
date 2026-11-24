"""Run 01_gbci_s1_grad_check_631g.py with a gesdd -> gesvd SVD fallback.

numpy's default divide-and-conquer driver fails to converge on the
biorthogonalization overlap of one displaced geometry.  gesvd is the older,
slower, QR-based driver and does not have that failure mode.  The fallback
fires only on LinAlgError, so every SVD that gesdd handles is untouched --
which is what the control task is there to confirm.
"""
import runpy
import sys

import numpy
import numpy.linalg as nl
from scipy.linalg import svd as scipy_svd

_orig = nl.svd


def svd(a, *args, **kwargs):
    try:
        return _orig(a, *args, **kwargs)
    except nl.LinAlgError:
        sys.stderr.write("[gesvd fallback] gesdd failed on a %s matrix\n"
                         % (numpy.shape(a),))
        sys.stderr.flush()
        full = kwargs.get("full_matrices", args[0] if args else True)
        return scipy_svd(a, full_matrices=full, lapack_driver="gesvd")


nl.svd = svd
numpy.linalg.svd = svd

sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
