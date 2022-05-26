NULL

# globals
.globals <- new.env(parent = emptyenv())
.globals$tensorboard <- NULL

.onLoad <- function(libname, pkgname) {
  vimureP <<- reticulate::import("vimure", delay_load = list(
    priority = 5,
    environment = "r-reticulate",
    on_load = function(){
      version <- vm_version()
      packageStartupMessage("Loaded vimure version ", version)
    },
    on_error = function(e) {
      stop(vm_config_error_message(), call. = FALSE)
    }
  ))
}

#' Vimure configuration information
#'
#' @return List with information on the current configuration of Vimure
#'   You can determine whether Vimure was found using the `available`
#'   member,
#'
#' @keywords internal
#' @export
vm_config <- function(){
  # first check if we found vimure
  have_vimure <- reticulate::py_module_available("vimure")

  # get py config
  config <- reticulate::py_config()

  # found it!
  if (have_vimure) {
    # get version
    if (reticulate::py_has_attr(vimureP, "version"))
      version_raw <- vimureP$version$VERSION
    else
      version_raw <- ""

    structure(class = "vimure_config", list(
      available = TRUE,
      version = version_raw,
      version_str = version_raw,
      location = config$required_module_path,
      python = config$python,
      python_version = config$version
    ))
  } else {
    structure(class = "vimure_config", list(
      available = FALSE,
      python_versions = config$python_versions,
      error_message = vm_config_error_message()
    ))
  }
}

#' @rdname vm_config
#' @keywords internal
#' @export
vm_version <- function() {
  config <- vm_config()
  if (config$available)
    config$version
  else
    NULL
}

#' @export
print.vimure_config <- function(x, ...) {
  if (x$available) {
    aliased <- function(path) sub(Sys.getenv("HOME"), "~", path)
    cat("Vimure v", x$version_str, " (", aliased(x$location), ")\n", sep = "")
    cat("Python v", x$python_version, " (", aliased(x$python), ")\n", sep = "")
  } else {
    cat(x$error_message, "\n")
  }
}

# Build error message for Vimure configuration errors
vm_config_error_message <- function(){
  message <- "Valid installation of `vimure` not found."
  config <- reticulate::py_config()
  if (!is.null(config)) {
    if (length(config$python_versions) > 0) {
      message <- paste0(message,
                        "\n\nPython environments searched for 'vimure' package:\n")
      python_versions <- paste0(" ", normalizePath(config$python_versions, mustWork = FALSE),
                                collapse = "\n")
      message <- paste0(message, python_versions, sep = "\n")
    }
  }

  python_error <- tryCatch({
    reticulate::import("vimure")
    list(message = NULL)
  },
  error = function(e) {
    on.exit(reticulate::py_clear_last_error())
    reticulate::py_last_error()
  })

  message <- paste0(message,
                    "\nPython exception encountered:\n ",
                    python_error$message, "\n")

  message <- paste0(message,
                    "\nYou can install `vimure` using the install_vimure() function.\n")
  message
}



