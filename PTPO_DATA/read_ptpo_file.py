import numpy as np
import python_util
ptpo_tddft_results = python_util.open_object_file("ptpo_tddft_bare_results_internal")
print("All states info:")
ptpo_tddft_results.print_info()
select_set = np.array([6, 13, 24, 26, 27, 36, 38]) #states with large dipole moments-->most are forbidden transitions
ptpo_tddft_truncated = ptpo_tddft_results.truncate_selection(select_set)
print("Selected states (highest dipole magnitudes) info:")
ptpo_tddft_truncated.print_info()