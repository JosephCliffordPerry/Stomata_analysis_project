source("Stomata_only_pipeline/crop_to_stomata_outline.R")
source("Stomata_only_pipeline/Individual_stomata_crop.R")
source("Stomata_only_pipeline/stomata_dataframe_builder.R")
source("Stomata_only_pipeline/stomata_stats_helper.R")
###
image_dir <- "D:/stomata/November_images/november_toy_data"
model_path <- "Stomata_obbox.pt"
crops_df <- individual_stomata_crop(image_dir, model_path)
crop_dir<-paste0(image_dir,"/crops")
stomata_list<-yolo_seg_inference(image_dir = crop_dir,model_path = "single_stomata_internal.pt")  
stomata_df <- interpolate_stomata_list(stomata_list, n_points = 100)
stomata_df<-add_stomata_geometry_columns(stomata_df)
#plan was then to produce profiles from the stomata for comparison and analysis
#
