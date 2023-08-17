vimureP <- NULL

# An adaptation of https://github.com/rstudio/reticulate/blob/fd9301ff77575227336d421f099c580ed68ab1b2/R/miniconda.R#L302 # nolint
miniconda_conda <- function(path = reticulate::miniconda_path()) {
  exe <- if (Sys.info()["sysname"] == "Windows") {
    "condabin/conda.bat"
  } else {
    "bin/conda"
  }
  file.path(path, exe)
}

conda_path <- miniconda_conda()
env_name <- "r-vimure"

github_repo <- "https://github.com/latentnetworks/vimure.git"
py_pkg_suffix <- "#egg=vimure&subdirectory=src/python/"

.onLoad <- function(libname, pkgname) {

  reticulate::use_condaenv(env_name, required = TRUE, conda = conda_path)

  vimureP <<- reticulate::import("vimure", delay_load = list(
    on_load = function() {
      print("Loaded")
      pkg_version <- get_pkg_version()
      emit <- get("packageStartupMessage") # R CMD check
      emit("Loaded vimure version ", pkg_version)
    },
    on_error = function(e) {
      config <- vm_config()
      stop(config$error_message, call. = FALSE)
    }
  ))
}



check_installation <- function() {
  if (is.null(vimureP)) {
    config <- reticulate::py_config()
    if (length(config$python_versions) > 0) {
      message <- paste0(
        "Valid installation of `vimure` not found.\n\n",
        "Python environments searched for 'vimure' package:\n"
      )
      python_versions <- paste0(" ", normalizePath(config$python_versions, mustWork = FALSE),
        collapse = "\n"
      )
      message <- paste0(message, python_versions, sep = "\n")
    } else {
      message <- "Valid installation of `vimure` not found."
    }
    stop(message)
  }
}

get_pkg_version <- function() {
  # To understand how this works, see:
  # https://r-pkgs.org/data.html#sec-data-system-file
  descpath <- system.file("DESCRIPTION", package = "vimure")
  pkg_version <- read.dcf(descpath, all = TRUE)$Version
  return(pkg_version)
}

#' Vimure configuration information
#'
#' @return List with information on the current configuration of Vimure
#'   You can determine whether Vimure was found using the `available`
#'   member,
#'
#' @keywords internal
#' @export
vm_config <- function() {
  config <- reticulate::py_config()
  pkg_version <- get_pkg_version()

  structure(class = "vimure_config", list(
    available = TRUE,
    version = pkg_version,
    version_str = pkg_version,
    location = config$required_module_path,
    python = config$python,
    python_version = config$version
  ))
}

#' @export
print.vimure_config <- function(x, ...) {
  if (x$available) {
    aliased <- function(path) sub(Sys.getenv("HOME"), "~", path)
    cat("Python version: v", x$python_version, " (", aliased(x$python), ")\n", sep = "")
    cat("Python version of vimure: ", x$version_str, "\n", sep = "")
  } else {
    cat(x$error_message, "\n")
  }
}

# Build error message for Vimure configuration errors
vm_config_error_message <- function() {
  message <- "Valid installation of `vimure` not found."
  config <- reticulate::py_config()
  if (!is.null(config)) {
    if (length(config$python_versions) > 0) {
      message <- paste0(
        message,
        "\n\nPython environments searched for 'vimure' package:\n"
      )
      python_versions <- paste0(" ", normalizePath(config$python_versions, mustWork = FALSE),
        collapse = "\n"
      )
      message <- paste0(message, python_versions, sep = "\n")
    }
  }

  python_error <- tryCatch(
    {
      reticulate::import("vimure")
      list(message = NULL)
    },
    error = function(e) {
      on.exit(reticulate::py_clear_last_error())
      reticulate::py_last_error()
    }
  )

  message <- paste0(
    message,
    "\nPython exception encountered:\n ",
    python_error$message, "\n"
  )

  message <- paste0(
    message,
    "\nYou can install `vimure` using the install_vimure() function.\n"
  )
  message
}
