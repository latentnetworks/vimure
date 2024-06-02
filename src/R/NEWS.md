<!-- NEWS.md is maintained by https://cynkra.github.io/fledge, do not edit -->

# vimure 0.2.0 (2024-06-02)

Changes the way reporters' mask is handled

- [MAJOR CHANGE] [PYTHON] [R] Instead of creating a custom reporter's mask, users are now encouraged to choose from three preset reporting scenarios that are hardcoded into the code:
  - `reporting_scenario = 1` (single-sampling, excluding self-loops)
  - `reporting_scenario = 2` (double-sampling, excluding self-loops)
  - `reporting_scenario = 3` (nodes can report any ties in the network, excluding self-loops)

- [MAJOR CHANGE] [R] Adds `reporting_scenario` (int) parameter to `vimure()` function.
- [MAJOR CHANGE] [PYTHON] Adds `reporting_scenario` (int) parameter to `VIMureModel()` constructor.

- [MAJOR CHANGE] [R] Adds `reporting_scenario` (int) parameter to `generate_X()` function, used when creating synthetic X data.
- [MAJOR CHANGE] [PYTHON] Adds `reporting_scenario` (int) parameter to `_build_X()` method, used when creating synthetic X data.

- [MAJOR CHANGE] [R] Removes `flag_self_reporter` parameter from `generate_X()` function, as this functionality is now handled by the `reporting_scenario` parameter.
- [MAJOR CHANGE] [PYTHON] Removes `flag_self_reporter` parameter from `_build_X()` method, as this functionality is now handled by the `reporting_scenario` parameter.

- [REFACTORING] [PYTHON] Updated the internal logic of the Python function 'read_from_edgelist()' to properly enforce the new parameter `reporting_scenario` or a custom reporter mask (if informed).

- [MAJOR CHANGE] [PYTHON] When reading data frames with the `read_from_edgelist()` function, the logic behind reporters' mask is now handled in the following priority order:
  1. For backward compatibility, the function first checks if the parameter `reporters` (a list) was passed to the function (i.e. `read_from_edgelist(edgelist, reporters=reporting_nodes)`). If it was, the reporters' mask is set to double-sampling (i.e. `reporting_scenario = 2`), but applied only to the subset of nodes passed as reporters.
  2. If no `reporters` parameter is passed, the function checks if the parameter `R` was passed to the function (i.e., `read_from_edgelist(edgelist, R=custom_mask)`). If it was, then this custom mask is checked for validity and used as the reporters' mask.
  3. If neither `reporters` nor `R` parameters are passed, the `reporting_scenario` parameter is used to determine the reporting scenario.

- [WARNINGS] [PYTHON] [R] Remove a warning message that was printed when no custom reporter mask was passed to VIMureModel#fit() method (Python code). By default, the model will use the double-sampling scenario and does not expect a custom reporter mask.
- [TESTS] [R] Adds two new unit tests to address GitHub Issue '#92 [BUG] Custom `reporter_mask` being ignored by the R package': 1) Check that custom reporters' mask are enforced 2) Check that custom reporters' mask are enforced when input is edgelist.
- [TESTS] [PYTHON] Adds new unit tests within the class `TestSyntheticWithReportingScenarios` to ensure the new reporting scenarios work as expected.
- [TESTS] [PYTHON] Add a new test `test_data_as_edgelist_reporting_scenario_2`, using Karnataka data, to check if the `reporting_scenario` parameter is working as expected when the real data is passed as an edgelist.

- [REFACTORING] [PYTHON] There were multiple functions, in the Python code, that created a reporters' mask. These functions were refactored to a single function called `build_self_reporters_mask(reporting_scenario, L, N, M)` hosted in the `utils.py` file. This function is now used by the `read_from_edgelist()` function to create the reporters' mask, by the `generate_X()` function to create the synthetic X data and by the `VIMureModel#fit()` method to create the reporters' mask for the model.


# vimure 0.1.3 (2023-08-18)

(R)

- Users can now install the package with a single devtools::install_github() call instead of a cumbersome set of steps

(Python)

- No changes

(Docs)

- Updated the installation instructions for the R package and changed references from a previous version (v0.1.1) to the current one (v0.1.3)

(DEV changes)

- Sync Python and R package versions. 
- Modify .github/workflows to avoid running the CI all the time

# vimure 0.1.2.9000 (2023-08-16)

- (R) Experimental stage: embedding the Python installation and full setup directly into the R package, so that the user does not need to install Python separately.

# vimure 0.1.2 (2023-08-16)

- (R) Add minimal vignettes for synthetic models
- (R) Synthetic network models have better names + modify unit tests
- (Python) no changes


# vimure 0.1.1 (2023-03-27)

- Added documentation and tutorials in both Python and R, in preparation for the Sunbelt workshop (https://www.insna.org/events/latent-network-modelling-for-networks-based-on-multiple-imperfect-reports-an-introduction-to-the-vimure-package-in-r-and-python)


# vimure 0.1 (2021-12-27)

- Added Python notebooks to replicate the JRSSS-A paper (https://doi.org/10.1093/jrsssa/qnac004)
- Preliminary version of the vimure package for general use (Pytnon and R)

