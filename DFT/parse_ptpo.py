import albano_params as ap
import python_util

'''Loads and prints data for PTPO calculations'''

ptpo_results = python_util.open_object_file("ptpo_tddft_bare_results_internal.pkl")
ptpo_results.print_info()

