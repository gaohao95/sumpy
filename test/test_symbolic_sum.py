import pyopencl as cl
import numpy as np
import scipy as sp
import scipy.fftpack
import logging

from sumpy.kernel import HelmholtzKernel
from sumpy.expansion.local import H2DLocalExpansion, H2DLocalExpansionSymbolicSum
from sumpy.qbx import LayerPotential

aspect_ratio = 1
nsrc = 500
novsmp = 4 * nsrc
helmholtz_k = (35+4j)*0.3
what_operator = "D"
what_operator_lpot = "D"
force_center_side = 1
order = 5

logging.basicConfig(level=logging.INFO)


class CurveGrid:
    def __init__(self, x, y):
        self.pos = np.vstack([x,y]).copy()
        xp = self.xp = sp.fftpack.diff(x, period=1)
        yp = self.yp = sp.fftpack.diff(y, period=1)
        xpp = self.xpp = sp.fftpack.diff(xp, period=1)
        ypp = self.ypp = sp.fftpack.diff(yp, period=1)
        self.mean_curvature = (xp*ypp-yp*xpp)/((xp**2+yp**2)**(3/2))

        speed = self.speed = np.sqrt(xp**2+yp**2)
        self.normal = (np.vstack([yp, -xp])/speed).copy()

    def __len__(self):
        return len(self.pos)

    def plot(self):
        import matplotlib.pyplot as pt
        pt.plot(self.pos[:, 0], self.pos[:, 1])


def process_kernel(knl, what_operator):
    if what_operator == "S":
        pass
    elif what_operator == "S0":
        from sumpy.kernel import AxisTargetDerivative
        knl = AxisTargetDerivative(0, knl)
    elif what_operator == "S1":
        from sumpy.kernel import AxisTargetDerivative
        knl = AxisTargetDerivative(1, knl)
    elif what_operator == "D":
        from sumpy.kernel import DirectionalSourceDerivative
        knl = DirectionalSourceDerivative(knl)
    # DirectionalTargetDerivative (temporarily?) removed
    # elif what_operator == "S'":
    #     from sumpy.kernel import DirectionalTargetDerivative
    #     knl = DirectionalTargetDerivative(knl)
    else:
        raise RuntimeError("unrecognized operator '%s'" % what_operator)

    return knl


if novsmp is None:
    novsmp = 4 * nsrc

if what_operator_lpot is None:
    what_operator_lpot = what_operator

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# {{{ make p2p kernel calculator

if isinstance(helmholtz_k, complex):
    knl = HelmholtzKernel(2, allow_evanescent=True)
else:
    knl = HelmholtzKernel(2)

knl_kwargs = {"k": helmholtz_k}

lpot_knl = process_kernel(knl, what_operator_lpot)

lpot_unroll = LayerPotential(
    ctx,
    [H2DLocalExpansion(lpot_knl, order=order)],
    value_dtypes=np.complex128
)

lpot_symbolic_sum = LayerPotential(
    ctx,
    [H2DLocalExpansionSymbolicSum(lpot_knl, order=order)],
    value_dtypes=np.complex128
)

# }}}

# {{{ set up geometry

# r,a,b match the corresponding letters from G. J. Rodin and O. Steinbach,
# Boundary Element Preconditioners for problems defined on slender domains.
# http://dx.doi.org/10.1137/S1064827500372067

a = 1
b = 1 / aspect_ratio


def map_to_curve(t):
    t = t * (2 * np.pi)

    x = a * np.cos(t)
    y = b * np.sin(t)

    w = (np.zeros_like(t) + 1) / len(t)

    return x, y, w


native_t = np.linspace(0, 1, nsrc, endpoint=False)
native_x, native_y, native_weights = map_to_curve(native_t)
native_curve = CurveGrid(native_x, native_y)

ovsmp_t = np.linspace(0, 1, novsmp, endpoint=False)
ovsmp_x, ovsmp_y, ovsmp_weights = map_to_curve(ovsmp_t)
ovsmp_curve = CurveGrid(ovsmp_x, ovsmp_y)

curve_len = np.sum(ovsmp_weights * ovsmp_curve.speed)
hovsmp = curve_len / novsmp
center_dist = 5 * hovsmp

if force_center_side is not None:
    center_side = force_center_side * np.ones(len(native_curve))
else:
    center_side = -np.sign(native_curve.mean_curvature)

centers = (native_curve.pos
           + center_side[:, np.newaxis]
           * center_dist * native_curve.normal)

# native_curve.plot()
# pt.show()

volpot_kwargs = knl_kwargs.copy()
lpot_kwargs = knl_kwargs.copy()

if what_operator == "D":
    volpot_kwargs["src_derivative_dir"] = native_curve.normal

if what_operator_lpot == "D":
    lpot_kwargs["src_derivative_dir"] = ovsmp_curve.normal

if what_operator_lpot == "S'":
    lpot_kwargs["tgt_derivative_dir"] = native_curve.normal

# }}}

# {{{ compute potentials

mode_nr = 0
density = np.cos(mode_nr*2*np.pi*native_t).astype(np.complex128)
ovsmp_density = np.cos(mode_nr*2*np.pi*ovsmp_t).astype(np.complex128)

_, (curve_pot_unroll,) = lpot_unroll(
    queue, native_curve.pos, ovsmp_curve.pos,
    centers,
    [ovsmp_density * ovsmp_curve.speed * ovsmp_weights],
    expansion_radii=np.ones(centers.shape[1]),
    **lpot_kwargs
)

_, (curve_pot_symbolic_sum,) = lpot_symbolic_sum(
    queue, native_curve.pos, ovsmp_curve.pos,
    centers,
    [ovsmp_density * ovsmp_curve.speed * ovsmp_weights],
    expansion_radii=np.ones(centers.shape[1]),
    **lpot_kwargs
)


error = np.linalg.norm(curve_pot_unroll - curve_pot_symbolic_sum, ord=np.inf) / np.linalg.norm(curve_pot_unroll, ord=np.inf)
print(error)
assert error < 1e-12

# }}}
