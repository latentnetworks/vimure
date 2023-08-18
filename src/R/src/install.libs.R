require(reticulate)

#### CONSTANTS ####

env_name <- "r-vimure"
github_repo <- "https://github.com/latentnetworks/vimure.git"
py_pkg_suffix <- "#egg=vimure&subdirectory=src/python/"


#### FUNCTIONS ####

miniconda_conda <- function(path = reticulate::miniconda_path()) {
  exe <- if (Sys.info()["sysname"] == "Windows") {
    "condabin/conda.bat"
  } else {
    "bin/conda"
  }
  file.path(path, exe)
}

get_current_module_version <- function(module_name) {
  all_pkgs <- reticulate::py_list_packages()
  all_pkgs[all_pkgs["package"] == module_name][2]
}

install_python_vimure <- function() {
  # To understand how this works, see:
  # https://r-pkgs.org/data.html#sec-data-system-file
  descpath <- system.file("DESCRIPTION", package = "vimure")
  pkg_version <- read.dcf(descpath, all = TRUE)$Version
  py_pkg_url <- paste0("git+", github_repo, "@v", pkg_version, py_pkg_suffix)

  emit <- get("packageStartupMessage") # R CMD check
  emit("Installing vimure version ", pkg_version, " from ", py_pkg_url)
  reticulate::conda_install(py_pkg_url, pip = TRUE, envname = env_name,
                            conda = conda_path)
}

#### MINICONDA AND CONDA ENV SETUP ####

conda_path <- miniconda_conda()

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
    reticulate::use_condaenv(env_name, required = TRUE, conda = conda_path)
  }
)


#### INSTALL VIMURE ####

if (!reticulate::py_module_available("vimure")) {
  install_python_vimure()
} else if (get_current_module_version("vimure") != pkg_version) {
  install_python_vimure()
} else {
  emit <- get("packageStartupMessage") # R CMD check
  emit("VIMuRe already installed")
}
