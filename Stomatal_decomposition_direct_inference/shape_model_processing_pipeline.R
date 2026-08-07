source("D:/Stomatal_analysis_project/Stomatal_decomposition_direct_inference/load_polygons.R")
#build informative stomata
rda_dir<-"E:/Stomata_maize/all_images/consensus_and_inference_rda3"
#Stomata<-load_polygons(rda_dir)
Stomata2<-load_polygons_only(rda_dir)
#save(Stomata#,file = "#Maize_direct_inferances.RDA")
load("D:/Stomatal_analysis_project/Maize_direct_inferances.RDA")
 #crop_locations<-read.csv(file = "cropped_obbs.csv")
 #crops_with_rotation<-build_chain_angles(crop_locations)
 #save(crops_with_rotation,file = "crops_with_angles.RDA")
load("crops_with_angles.RDA")

idx <- match(
  sapply(Stomata, `[[`, "image_path"),
  crops_with_rotation$crop
)

for (i in seq_along(Stomata)) {
  Stomata[[i]]$chain_angle <- crops_with_rotation$chain_angle[idx[i]]
}
source("D:/Stomatal_analysis_project/Stomatal_decomposition_direct_inference/align_polygons.R")
source("D:/Stomatal_analysis_project/Stomatal_decomposition_direct_inference/build_probabiltiy_matrix.R")
##build probability map 
aligned_Stomata<-prepare_aligned_shapes(Stomata)
#save(aligned_Stomata,file = "aligned_Stomata.RDA")
# load("aligned_Stomata.RDA")
aligned_Stomata <- Filter(function(x) {
  !(
    is.null(x[["chain_angle"]]) ||
      is.null(x[["aligned_density"]]) ||
      any(is.na(x[["chain_angle"]])) ||
      any(is.na(x[["aligned_density"]]))
  )
}, aligned_Stomata)
probability_map<-build_probability_map(aligned_Stomata)
image(probability_map)
source("D:/Stomatal_analysis_project/Stomatal_decomposition_direct_inference/build_shape_model.R")
## build shape model  

model <- build_shape_model(aligned_Stomata)

plot_shape(model$mean_shape)


plot_mode(model, mode = 1)

##save full model
stomata_model <- list(
  shape_model = model,
  probability_map = probability_map,
  n_landmarks = 200,
  template_size = dim(probability_map),
  version = "1.0"
)

saveRDS(stomata_model, "stomata_model.rds")



