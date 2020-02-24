from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from six.moves import range, zip
import sumpy.symbolic as sym

from sumpy.expansion import (
    ExpansionBase, VolumeTaylorExpansion, LaplaceConformingVolumeTaylorExpansion,
    HelmholtzConformingVolumeTaylorExpansion)


class LocalExpansionBase(ExpansionBase):
    pass


import logging
logger = logging.getLogger(__name__)

__doc__ = """

.. autoclass:: VolumeTaylorLocalExpansion
.. autoclass:: H2DLocalExpansion
.. autoclass:: Y2DLocalExpansion
.. autoclass:: LineTaylorLocalExpansion

"""


# {{{ line taylor

class LineTaylorLocalExpansion(LocalExpansionBase):

    def get_storage_index(self, k):
        return k

    def get_coefficient_identifiers(self):
        return list(range(self.order+1))

    def coefficients_from_source(self, avec, bvec, rscale):
        # no point in heeding rscale here--just ignore it
        if bvec is None:
            raise RuntimeError("cannot use line-Taylor expansions in a setting "
                    "where the center-target vector is not known at coefficient "
                    "formation")

        tau = sym.Symbol("tau")

        avec_line = avec + tau*bvec

        line_kernel = self.kernel.get_expression(avec_line)

        from sumpy.symbolic import USE_SYMENGINE

        if USE_SYMENGINE:
            from sumpy.tools import MiDerivativeTaker, my_syntactic_subs
            deriv_taker = MiDerivativeTaker(line_kernel, (tau,))

            return [my_syntactic_subs(
                        self.kernel.postprocess_at_target(
                            self.kernel.postprocess_at_source(
                                deriv_taker.diff(i),
                                avec), bvec),
                        {tau: 0})
                    for i in self.get_coefficient_identifiers()]
        else:
            # Workaround for sympy. The automatic distribution after
            # single-variable diff makes the expressions very large
            # (https://github.com/sympy/sympy/issues/4596), so avoid doing
            # single variable diff.
            #
            # See also https://gitlab.tiker.net/inducer/pytential/merge_requests/12

            return [self.kernel.postprocess_at_target(
                        self.kernel.postprocess_at_source(
                            line_kernel.diff("tau", i), avec),
                        bvec)
                    .subs("tau", 0)
                    for i in self.get_coefficient_identifiers()]

    def evaluate(self, coeffs, bvec, rscale):
        # no point in heeding rscale here--just ignore it
        from pytools import factorial
        return sym.Add(*(
                coeffs[self.get_storage_index(i)] / factorial(i)
                for i in self.get_coefficient_identifiers()))

# }}}


# {{{ volume taylor

class VolumeTaylorLocalExpansionBase(LocalExpansionBase):
    """
    Coefficients represent derivative values of the kernel.
    """

    def coefficients_from_source(self, avec, bvec, rscale):
        from sumpy.tools import MiDerivativeTaker
        ppkernel = self.kernel.postprocess_at_source(
                self.kernel.get_expression(avec), avec)

        taker = MiDerivativeTaker(ppkernel, avec)
        return [
                taker.diff(mi) * rscale ** sum(mi)
                for mi in self.get_coefficient_identifiers()]

    def evaluate(self, coeffs, bvec, rscale):
        from sumpy.tools import mi_power, mi_factorial
        evaluated_coeffs = (
            self.derivative_wrangler.get_full_kernel_derivatives_from_stored(
                coeffs, rscale))
        bvec = bvec * rscale**-1
        result = sum(
                coeff
                * mi_power(bvec, mi, evaluate=False)
                / mi_factorial(mi)
                for coeff, mi in zip(
                        evaluated_coeffs, self.get_full_coefficient_identifiers()))
        return result

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale):
        logger.info("building translation operator: %s(%d) -> %s(%d): start"
                % (type(src_expansion).__name__,
                    src_expansion.order,
                    type(self).__name__,
                    self.order))

        if not self.use_rscale:
            src_rscale = 1
            tgt_rscale = 1

        from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansionBase
        if isinstance(src_expansion, VolumeTaylorMultipoleExpansionBase):
            # We know the general form of the multipole expansion is:
            #
            #    coeff0 * diff(kernel, mi0) + coeff1 * diff(kernel, mi1) + ...
            #
            # To get the local expansion coefficients, we take derivatives of
            # the multipole expansion.
            #
            # This code speeds up derivative taking by caching all kernel
            # derivatives.

            taker = src_expansion.get_kernel_derivative_taker(dvec)

            from sumpy.tools import add_mi

            result = []
            for deriv in self.get_coefficient_identifiers():
                local_result = []
                for coeff, term in zip(
                        src_coeff_exprs,
                        src_expansion.get_coefficient_identifiers()):

                    kernel_deriv = (
                            src_expansion.get_scaled_multipole(
                                taker.diff(add_mi(deriv, term)),
                                dvec, src_rscale,
                                nderivatives=sum(deriv) + sum(term),
                                nderivatives_for_scaling=sum(term)))

                    local_result.append(
                            coeff * kernel_deriv * tgt_rscale**sum(deriv))
                result.append(sym.Add(*local_result))
        else:
            from sumpy.tools import MiDerivativeTaker
            expr = src_expansion.evaluate(src_coeff_exprs, dvec, rscale=src_rscale)
            taker = MiDerivativeTaker(expr, dvec)

            # Rscale/operand magnitude is fairly sensitive to the order of
            # operations--which is something we don't have fantastic control
            # over at the symbolic level. The '.expand()' below moves the two
            # canceling "rscales" closer to each other in the hope of helping
            # with that.
            result = [
                    (taker.diff(mi) * tgt_rscale**sum(mi)).expand()
                    for mi in self.get_coefficient_identifiers()]

        logger.info("building translation operator: done")
        return result


class VolumeTaylorLocalExpansion(
        VolumeTaylorExpansion,
        VolumeTaylorLocalExpansionBase):

    def __init__(self, kernel, order, use_rscale=None):
        VolumeTaylorLocalExpansionBase.__init__(self, kernel, order, use_rscale)
        VolumeTaylorExpansion.__init__(self, kernel, order, use_rscale)


class LaplaceConformingVolumeTaylorLocalExpansion(
        LaplaceConformingVolumeTaylorExpansion,
        VolumeTaylorLocalExpansionBase):

    def __init__(self, kernel, order, use_rscale=None):
        VolumeTaylorLocalExpansionBase.__init__(self, kernel, order, use_rscale)
        LaplaceConformingVolumeTaylorExpansion.__init__(
                self, kernel, order, use_rscale)


class HelmholtzConformingVolumeTaylorLocalExpansion(
        HelmholtzConformingVolumeTaylorExpansion,
        VolumeTaylorLocalExpansionBase):

    def __init__(self, kernel, order, use_rscale=None):
        VolumeTaylorLocalExpansionBase.__init__(self, kernel, order, use_rscale)
        HelmholtzConformingVolumeTaylorExpansion.__init__(
                self, kernel, order, use_rscale)

# }}}


# {{{ 2D Bessel-based-expansion

class _FourierBesselLocalExpansion(LocalExpansionBase):
    def get_storage_index(self, k):
        return self.order+k

    def get_coefficient_identifiers(self):
        return list(range(-self.order, self.order+1))

    def coefficients_from_source(self, avec, bvec, rscale):
        if not self.use_rscale:
            rscale = 1

        from sumpy.symbolic import sym_real_norm_2
        hankel_1 = sym.Function("hankel_1")

        arg_scale = self.get_bessel_arg_scaling()

        # The coordinates are negated since avec points from source to center.
        source_angle_rel_center = sym.atan2(-avec[1], -avec[0])
        avec_len = sym_real_norm_2(avec)
        return [self.kernel.postprocess_at_source(
                    hankel_1(l, arg_scale * avec_len)
                    * rscale ** abs(l)
                    * sym.exp(sym.I * l * source_angle_rel_center), avec)
                    for l in self.get_coefficient_identifiers()]

    def evaluate(self, coeffs, bvec, rscale):
        if not self.use_rscale:
            rscale = 1

        from sumpy.symbolic import sym_real_norm_2
        bessel_j = sym.Function("bessel_j")
        bvec_len = sym_real_norm_2(bvec)
        target_angle_rel_center = sym.atan2(bvec[1], bvec[0])

        arg_scale = self.get_bessel_arg_scaling()

        return sum(coeffs[self.get_storage_index(l)]
                   * self.kernel.postprocess_at_target(
                       bessel_j(l, arg_scale * bvec_len)
                       / rscale ** abs(l)
                       * sym.exp(sym.I * l * -target_angle_rel_center), bvec)
                for l in self.get_coefficient_identifiers())

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale):
        from sumpy.symbolic import sym_real_norm_2

        if not self.use_rscale:
            src_rscale = 1
            tgt_rscale = 1

        arg_scale = self.get_bessel_arg_scaling()

        if isinstance(src_expansion, type(self)):
            dvec_len = sym_real_norm_2(dvec)
            bessel_j = sym.Function("bessel_j")
            new_center_angle_rel_old_center = sym.atan2(dvec[1], dvec[0])
            translated_coeffs = []
            for l in self.get_coefficient_identifiers():
                translated_coeffs.append(
                    sum(src_coeff_exprs[src_expansion.get_storage_index(m)]
                        * bessel_j(m - l, arg_scale * dvec_len)
                        / src_rscale ** abs(m)
                        * tgt_rscale ** abs(l)
                        * sym.exp(sym.I * (m - l) * -new_center_angle_rel_old_center)
                    for m in src_expansion.get_coefficient_identifiers()))
            return translated_coeffs

        if isinstance(src_expansion, self.mpole_expn_class):
            dvec_len = sym_real_norm_2(dvec)
            hankel_1 = sym.Function("hankel_1")
            new_center_angle_rel_old_center = sym.atan2(dvec[1], dvec[0])
            translated_coeffs = []
            for l in self.get_coefficient_identifiers():
                translated_coeffs.append(
                    sum(
                        (-1) ** l
                        * hankel_1(m + l, arg_scale * dvec_len)
                        * src_rscale ** abs(m)
                        * tgt_rscale ** abs(l)
                        * sym.exp(sym.I * (m + l) * new_center_angle_rel_old_center)
                        * src_coeff_exprs[src_expansion.get_storage_index(m)]
                        for m in src_expansion.get_coefficient_identifiers()))
            return translated_coeffs

        raise RuntimeError("do not know how to translate %s to %s"
                           % (type(src_expansion).__name__,
                               type(self).__name__))


class _FourierBesselLocalExpansionSymbolicSum(_FourierBesselLocalExpansion):

    def coefficients_from_source(self, avec, bvec, rscale):
        if not self.use_rscale:
            rscale = 1

        # "hankel_1n" differs from "hankel_1" by precomputing Hankel function inside
        # a loop instead of recursively unrolling it
        hankel_1 = sym.Function("hankel_1n")

        arg_scale = self.get_bessel_arg_scaling()

        # The coordinates are negated since avec points from source to center.
        from sumpy.symbolic import sym_real_norm_2
        source_angle_rel_center = sym.atan2(-avec[1], -avec[0])
        avec_len = sym_real_norm_2(avec)

        import sympy
        from sumpy.expansion import OrderInameGenerator
        order_iname = OrderInameGenerator.get_next_order_iname()
        l = sympy.var(order_iname)

        return (
            self.kernel.postprocess_at_source(
                hankel_1(l, arg_scale * avec_len, order_iname)
                * rscale ** abs(l)
                * sym.exp(sym.I * l * source_angle_rel_center), avec
            ), order_iname
        )

    def evaluate(self, coeffs, bvec, rscale):
        coeffs_sym, order_iname = coeffs

        if not self.use_rscale:
            rscale = 1

        # "bessel_jn" differs from "bessel_j" by precomputing Bessel function inside
        # a loop instead of recursively unrolling it
        bessel_j = sym.Function("bessel_jn")

        from sumpy.symbolic import sym_real_norm_2
        bvec_len = sym_real_norm_2(bvec)
        target_angle_rel_center = sym.atan2(bvec[1], bvec[0])

        arg_scale = self.get_bessel_arg_scaling()

        import sympy
        l = sympy.symbols(order_iname)

        return sympy.Sum(
            coeffs_sym * self.kernel.postprocess_at_target(
                bessel_j(l, arg_scale * bvec_len, order_iname)
                / rscale ** abs(l)
                * sym.exp(sym.I * l * -target_angle_rel_center), bvec
            ), (l, -self.order, self.order+1))


class H2DLocalExpansion(_FourierBesselLocalExpansion):
    def __init__(self, kernel, order, use_rscale=None):
        from sumpy.kernel import HelmholtzKernel
        assert (isinstance(kernel.get_base_kernel(), HelmholtzKernel)
                and kernel.dim == 2)

        super(H2DLocalExpansion, self).__init__(kernel, order, use_rscale)

        from sumpy.expansion.multipole import H2DMultipoleExpansion
        self.mpole_expn_class = H2DMultipoleExpansion

    def get_bessel_arg_scaling(self):
        return sym.Symbol(self.kernel.get_base_kernel().helmholtz_k_name)


class H2DLocalExpansionSymbolicSum(H2DLocalExpansion, _FourierBesselLocalExpansionSymbolicSum):
    pass


class Y2DLocalExpansion(_FourierBesselLocalExpansion):
    def __init__(self, kernel, order, use_rscale=None):
        from sumpy.kernel import YukawaKernel
        assert (isinstance(kernel.get_base_kernel(), YukawaKernel)
                and kernel.dim == 2)

        super(Y2DLocalExpansion, self).__init__(kernel, order, use_rscale)

        from sumpy.expansion.multipole import Y2DMultipoleExpansion
        self.mpole_expn_class = Y2DMultipoleExpansion

    def get_bessel_arg_scaling(self):
        return sym.I * sym.Symbol(self.kernel.get_base_kernel().yukawa_lambda_name)

# }}}

# vim: fdm=marker
