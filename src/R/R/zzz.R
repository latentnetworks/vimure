vimureP <- NULL

.onLoad <- function(libname, pkgname) {
  vimureP <<- reticulate::import("vimure", delay_load = T)
}

.onAttach <- function(libname, pkgname) {
  if(!reticulate::virtualenv_exists("r-vimure")){
    packageStartupMessage("Creating a virtualenv (r-vimure)")
    reticulate::virtualenv_create("r-vimure")
    reticulate::virtualenv_install("r-vimure",
                                   "git+https://github.com/latentnetworks/vimure.git#egg=vimure&subdirectory=src/python/")
  }else{
    # Use developing version of Python package
    packageStartupMessage("Updating vimure to developing version (r-vimure)")
    reticulate::virtualenv_remove("r-vimure", "vimure")
    reticulate::virtualenv_install("r-vimure",
                                   "git+https://github.com/latentnetworks/vimure.git@25-vimure-v01-r-implement-vimuremodel#egg=vimure&subdirectory=src/python/")
  }

  reticulate::use_virtualenv("r-vimure", required = T)
  py_path <- reticulate::py_config()$python
  packageStartupMessage("PYTHON_PATH=", py_path)
}
