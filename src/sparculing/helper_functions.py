import dynpssimpy.modal_analysis as dps_mdl
import numpy as np
from scipy import optimize


def match_eigenvalues(v1, v2, dist=lambda x1, x2: abs(x1 - x2), return_idx=False):
    # From https://stackoverflow.com/questions/54183370/how-to-sort-vector-to-have-minimal-distance-to-another-vector-efficiently
    assert v1.ndim == v2.ndim == 1
    assert v1.shape[0] == v2.shape[0]
    n = v1.shape[0]
    t = np.dtype(dist(v1[0], v2[0]))
    dist_matrix = np.fromiter((dist(x1, x2) for x1 in v1 for x2 in v2),
                              dtype=t, count=n*n).reshape(n, n)
    row_ind, col_ind = optimize.linear_sum_assignment(dist_matrix)
    return (v2[col_ind], col_ind) if return_idx else v2[col_ind]



def change_gen_power(ps, gen_i, power):
    ps.gen["GEN"].par['P'][gen_i] = power


def change_all_gen_powers(ps, powers):
    """Change the generation in a case.

    Changes the set point of generators.
    """
    for i, power in enumerate(powers):
        change_gen_power(ps, i, power)

def change_load_power(ps, load_i, power):
    ps.loads['Load'].par['P'][load_i] = power

def change_all_load_powers(ps, powers):
    for i, power in enumerate(powers):
        change_load_power(ps, i, power)


def get_gen_power_vector(ps):
    """Returns the powers of the generators in a case."""
    return ps.gen['GEN'].par['P']  #  np.array([gen[4] for gen in model["generators"]["GEN"][1:]])

def get_load_power_vector(ps):
    """Returns the powers of the generators in a case."""
    return ps.loads['Load'].par['P']  # np.array([load[2] for load in model["loads"][1:]])

def get_gen_ratings(ps):
    """Returns the ratings of the generators in the case model."""
    return ps.gen['GEN'].par['S_n']  # np.array([gen[2] for gen in model["generators"]["GEN"][1:]])


def get_gen_names(ps):
    """Returns the names of the machines in the case model."""
    return ps.gen['GEN'].par['name']  # [gen[0] for gen in model["generators"]["GEN"][1:]]


def reschedule_and_get_min_damping(ps, powers):
    """Change the power and calculate minimum damping."""
    ps = change_all_gen_powers(ps, powers)
    return get_min_damping(ps)


def get_lin_sys(ps):
    "Returns the linerisation of the system"
    ps.power_flow()
    ps.init_dyn_sim()
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()
    return ps_lin


def remove_inaccurate_zero(ps_lin):
    zero_eig_idx = abs(ps_lin.eigs) < 1e-4
    assert sum(zero_eig_idx) == 1
    ps_lin.eigs[zero_eig_idx] = 0
    ps_lin.damping[zero_eig_idx] = np.inf


def get_random_dispatch(ps, std=0.01):
    """Caclulate a new random balanced dispatch.

    The methods draws generator powers from a normal distribution centered,
    around the current dispatch. The change in power compared to the base
    case is distributed based on generator ratings.
    """
    old_powers = get_gen_power_vector(ps)
    new_powers = [np.random.normal(power, power * std) for power in old_powers]
    slack = sum(old_powers) - sum(new_powers)
    ratings = get_gen_ratings(ps)
    total_rating = sum(ratings)
    new_powers += slack * ratings / total_rating  # I should consider limits later
    return new_powers
