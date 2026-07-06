load_polygons <- function(rda_dir){
  
  files <- list.files(
    rda_dir,
    pattern="\\.RDS$",
    recursive=TRUE,
    full.names=TRUE
  )
  
  polys <- list()
  
  for(i in seq_along(files)){
  
    
    polys[[i]] <- readRDS(files[[i]])
  }
  return(polys)
}
rda_dir<-"E:/Stomata_maize/all_images/consensus_and_inference_rda3"
polys<-load_polygons(rda_dir)
