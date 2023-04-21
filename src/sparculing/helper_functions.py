import dynpssimpy.dynamic as dps
import dynpssimpy.modal_analysis as dps_mdl
import numpy as np


def change_gen_power(model, gen_i, power):
    model["generators"]["GEN"][gen_i + 1][4] = power


def change_all_gen_powers(model, powers):
    """Change the generation in a case.

    Changes the set point of generators.
    """
    for i, power in enumerate(powers):
        change_gen_power(model, i, power)


def get_gen_power_vector(model):
    """Returns the powers of the generators in a case."""
    return np.array([gen[4] for gen in model["generators"]["GEN"][1:]])


def get_gen_ratings(model):
    """Returns the ratings of the generators in the case model."""
    return np.array([gen[2] for gen in model["generators"]["GEN"][1:]])


def get_gen_names(model):
    """Returns the names of the machines in the case model."""
    return [gen[0] for gen in model["generators"]["GEN"][1:]]


def reschedule_and_get_min_damping(model, powers):
    """Change the power and calculate minimum damping."""
    ps = change_all_gen_powers(model, powers)
    return get_min_damping(ps)


def get_lin_sys(ps):
    "Returns the linerisation of the system"
    ps.init_dyn_sim()
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()
    return ps_lin


def get_min_damping(ps_lin):
    """Calculate minimum damping."""
    zero_eig_idx = abs(ps_lin.eigs) < 1e-4
    assert sum(zero_eig_idx) == 1
    ps_lin.eigs[zero_eig_idx] = 0
    ps_lin.damping[zero_eig_idx] = np.inf
    return np.min(ps_lin.damping)


def get_random_dispatch(model, std=0.01):
    """Caclulate a new random balanced dispatch.

    The methods draws generator powers from a normal distribution centered,
    around the current dispatch. The change in power compared to the base
    case is distributed based on generator ratings.
    """
    old_powers = get_gen_power_vector(model)
    new_powers = [np.random.normal(power, power * std) for power in old_powers]
    slack = sum(old_powers) - sum(new_powers)
    ratings = get_gen_ratings(model)
    total_rating = sum(ratings)
    new_powers += slack * ratings / total_rating  # I should consider limits later
    return new_powers
