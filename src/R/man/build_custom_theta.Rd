% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/synthetic.R
\name{build_custom_theta}
\alias{build_custom_theta}
\title{Customize 'Reliability' parameter theta}
\usage{
build_custom_theta(
  synthetic,
  theta_ratio = 0.5,
  exaggeration_type = c("over", "under"),
  seed = NULL
)
}
\arguments{
\item{synthetic}{A synthetic model}

\item{theta_ratio}{Percentage of reporters who exaggerate [0,1]}

\item{exaggeration_type}{"over" or "under"}

\item{seed}{Pseudo random generator seed to use}
}
\value{
A L x M matrix for theta
}
\description{
Instead of the regular generative model for theta (theta ~ Gamma(sh, sc)),
create a more extreme scenario where some percentage of reporters exaggerates
}
