import statistics

from evaluation.evaluators.constants import Constants


def has_converged(loss_list: [float], convergence_check_timespan: int) -> bool:
    """checks a list of float if its values have converged"""
    if len(loss_list) < convergence_check_timespan * 2:
        return False

    mean_losses_old = statistics.mean(loss_list[-convergence_check_timespan * 2:-convergence_check_timespan])
    mean_losses_new = statistics.mean(loss_list[-convergence_check_timespan:])
    epsilon = Constants.convergence_check_epsilon

    return (mean_losses_old - epsilon) <= mean_losses_new
