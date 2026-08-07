library(sf)
library(dplyr)
library(purrr)
library(tidyr)



build_probability_map <- function(samples){
  
  mats <- lapply(
    samples,
    function(x) x$aligned_density
  )
  
  out<-Reduce("+", mats) / length(mats)
}
