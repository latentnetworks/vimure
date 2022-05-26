#' Install Vimure and its dependencies
#'
#' `install_vimure()` installs just the vimure python package and it's
#' direct dependencies.
#'
#' @details You may be prompted to download and install
#'   miniconda if reticulate did not find a non-system installation of python.
#'   Miniconda is the recommended installation method for most users, as it
#'   ensures that the R python installation is isolated from other python
#'   installations. All python packages will by default be installed into a
#'   self-contained conda or venv environment named "r-reticulate". Note that
#'   "conda" is the only supported method on M1 Mac.
#'
#'   If you initially declined the miniconda installation prompt, you can later
#'   manually install miniconda by running [`reticulate::install_miniconda()`].
#'
#' @section Custom Installation: `install_vimure()` isn't required to use vimure with the package.
#'   If you manually configure a python environment with the required
#'   dependencies, you can tell R to use it by pointing reticulate at it,
#'   commonly by setting an environment variable:
#'
#'   ``` R
#'   Sys.setenv("RETICULATE_PYTHON" = "~/path/to/python-env/bin/python")
#'   ```
#'
#' @md
#'
#' @inheritParams reticulate::py_install
#'
#' @param version Vimure version to install. Valid values include:
#'
#'   +  `"default"` installs  `develop branch`
#'
#'   + A version specification like `"2.4"` or `"2.4.0"`. Note that if the patch
#'   version is not supplied, the latest patch release is installed (e.g.,
#'   `"2.4"` today installs version "2.4.2")
#'
#'   + The full URL or path to a installer binary or python *.whl file.
#'
#' @param restart_session Restart R session after installing (note this will
#'   only occur within RStudio).
#'
#' @param python_version,conda_python_version Pass a string like "3.8" to
#'   request that conda install a specific Python version. This is ignored when
#'   attempting to install in a Python virtual environment. Note that the Python
#'   version must be compatible with the requested Vimure version, documented
#'   here: <https://github.com/latentnetworks/vimure/blob/develop/src/python/setup.py>
#'
#' @param pip_ignore_installed Whether pip should ignore installed python
#'   packages and reinstall all already installed python packages. This defaults
#'   to `TRUE`, to ensure that Vimure dependencies like NumPy are compatible
#'   with the prebuilt Vimure binaries.
#'
#' @param ... other arguments passed to [`reticulate::conda_install()`] or
#'   [`reticulate::virtualenv_install()`], depending on the `method` used.
#'
#' @export
install_vimure <- function(method = c("auto", "virtualenv", "conda"),
           conda = "auto",
           version = "develop",
           envname = NULL,
           restart_session = TRUE,
           conda_python_version = NULL,
           ...,
           pip_ignore_installed = TRUE,
           python_version = conda_python_version){

    method <- match.arg(method)

    if(file.exists(version)){
      package = version
    } else {
      package <- paste0(
        "git+https://github.com/latentnetworks/vimure.git@", version, "#egg=vimure&subdirectory=src/python/")
    }

    reticulate::py_install(
      packages       = package,
      envname        = envname,
      method         = method,
      conda          = conda,
      python_version = python_version,
      pip            = TRUE,
      pip_ignore_installed = pip_ignore_installed,
      ...
    )

    cat("\nInstallation complete.\n\n")

    if (restart_session &&
        requireNamespace("rstudioapi", quietly = TRUE) &&
        rstudioapi::hasFun("restartSession"))
      rstudioapi::restartSession()

    invisible(NULL)
}
