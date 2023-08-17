require(reticulate)

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
py_pkg_version <- "0.1.2"

github_repo <- "https://github.com/latentnetworks/vimure.git"
py_pkg_suffix <- "#egg=vimure&subdirectory=src/python/"

tryCatch(
  {
    reticulate::install_miniconda()
  },
  error = function(e) {
    emit <- get("packageStartupMessage") # R CMD check
    msg <- paste0(
      "Tried to install Miniconda but it looks like it is already installed.\n",
      "See the following message:\n\n",
      e$message,
      "\n\n"
    )
    emit(msg)
  }
)


tryCatch(
  {
    reticulate::use_condaenv(env_name, required = TRUE, conda = conda_path)
  },
  error = function(e) {
    reticulate::conda_create(env_name, conda = conda_path)
  }
)


if (!reticulate::py_module_available("vimure")) {
  # To understand how this works, see:
  # https://r-pkgs.org/data.html#sec-data-system-file
  #   descpath <- system.file("DESCRIPTION", package = "vimure")
  #   pkg_version <- read.dcf(descpath, all = TRUE)$Version
  py_pkg_url <- paste0("git+", github_repo, "@v", py_pkg_version, py_pkg_suffix)

  emit <- get("packageStartupMessage") # R CMD check
  emit("Installing vimure version ", py_pkg_version, " from ", py_pkg_url)
  reticulate::conda_install(py_pkg_url, pip = TRUE, envname = env_name, conda = conda_path)
} else {
  print("vimure already installed")
}
