import numpy as np
from sparculing.helper_functions import *
import dynpssimpy.dynamic as dps


class GenSensDispatchUnconstrained:
    def __init__(self, model):
        self.model = model
        self.p0 = get_gen_power_vector(model)
        self.p = np.copy(self.p0)
        self.ratings = get_gen_ratings(model)
        self.ps_lin_0 = get_lin_sys(dps.PowerSystemModel(model=model))

        # Initial state
        self.min_mode = np.argmin(self.ps_lin_0.damping)
        self.zeta = self.ps_lin_0.damping[self.min_mode]
        self.eigs_0 = self.ps_lin_0.eigs
        self.eig = self.eigs_0[self.min_mode]
        self.gen_sens = self.get_gen_sens()

        # Initial guesses
        self.l1 = 1
        self.l2 = 1

    def _lagrangian_value(self):
        x = np.zeros(len(self.p) + 2)
        kappa = self.zeta / np.sqrt(1 - self.zeta ^ 2)
        for i, p in enumerate(self.p):
            x[i] = (
                2 * (p - self.p0[i])
                + self.l1
                * (np.real(self.gen_sens[i]) + np.imag(self.gen_sens[i]) * kappa)
                + self.l2
            )

        x[-2] += np.real(self.eig) + kappa * np.imag(self.eig)
        x[-1] = np.sum(self.p) - np.sum(self.p0)

        return x

    def _set_jacobian_vals(self, jac, p, p0, k, i):
        jac[i, i] = 2 * (p - p0)
        jac[i, -2] = k
        jac[i, -1] = 1
        jac[-2, i] = k
        jac[-1, i] = 1

    def _make_jacobian(self):
        jac = np.zeros((len(self.p) + 2, len(self.p) + 2))
        kappa = self.zeta / np.sqrt(1 - self.zeta ^ 2)
        for i, p in enumerate(self.p):
            k = np.real(self.gen_sens[i]) + kappa * np.imag(self.gen_sens[i])
            self._set_jacobian_vals(jac, p, self.p0[i], k, i)
        return jac

    def _do_newton_step(self):
        f = self._lagrangian_value()
        jac = self._make_jacobian()
        return np.linalg.solve(-jac, f)

    def find_dispatch(self, zeta_1=0.1, dP_max=0.05, max_iter=100):
        zeta_b = self.zeta  # Best damping so far
        p_b = self.p  # Best power dispatch so far
        i = 0  # For controlling max iteration

        gen_sensed = True

        while self.zeta < zeta_1 and i < max_iter:
            dx = self._do_newton_step()

            self.l1 += dx[-2]
            self.l2 += dx[-1]

            for i in np.arange(len(self.p)):
                self.p[i] -= dx[i]

            change_all_gen_powers(self.model, self.p)

            lin_sys = get_lin_sys(dps.PowerSystemModel(model=self.model))
            self.zeta = lin_sys.damping[self.min_mode]

            if self.zeta < zeta_b:
                zeta_b = self.zeta
                p_b = self.p
                gen_sensed = False
            else:
                if not gen_sensed:
                    print("Damping not improving, recalculating gen_sens")
                    self.gen_sens = self.get_gen_sens()
                    gen_sensed = True
                else:
                    print("Damping not improving, giving up.")
                    return p_b, zeta_b
            i += 1
        return p_b, zeta_b

    def get_gen_sens(self, dP=1e-3):
        sens = np.zeros((len(self.eigs_0), len(self.p)), dtype=complex)

        for gen_i, rating in enumerate(self.ratings):
            change = rating * dP

            powers = self._change_power_with_distributed_slack(change, gen_i)
            change_all_gen_powers(self.model, powers)

            sens[:, gen_i] = get_lin_sys(dps.PowerSystemModel(model=self.model)).eigs

            powers = self._change_power_with_distributed_slack(-2 * change, gen_i)

            change_all_gen_powers(self.model, powers)

            sens[:, gen_i] -= get_lin_sys(dps.PowerSystemModel(model=self.model)).eigs
            sens[:, gen_i] = sens[:, gen_i] / (self.ratings[gen_i] * 2 * dP)

        return sens

    def _change_power_with_distributed_slack(self, change, gen_i):
        slack_ratings = self.ratings[np.arange(len(self.ratings)) != gen_i]
        slack = slack_ratings / np.sum(slack_ratings) * change

        powers = np.copy(self.p)
        powers[np.arange(len(powers)) != gen_i] = (
            powers[np.arange(len(powers)) != gen_i] + slack
        )
        powers[gen_i] = powers[gen_i] + change

        return powers


# def _variable_transform(powers, ratings):
# """Do the variable transform from the paper.
# For simplicity I assume the upper limit to be 95% of rating
# and the lower limit to be 0."""

# f_1 = np.zeros(len(powers))
# f_2 = np.zeros(len(powers))
# a = np.zeros(len(powers))

# for i, power in enumerate(powers):
# ul = ratings[i] * 0.95  # hard coded upper limit
# ll = 0  # hard coded lower limit
# a[i] = np.arcsin((power + (ul + ll) / 2) / (ul - ll) / 2)
# f_1[i] = (ul - ll) / 2 * np.cos(a[i])
# f_2[i] = -(ul - ll) / 2 * np.sin(a[i])
# return a, f_1, f_2
