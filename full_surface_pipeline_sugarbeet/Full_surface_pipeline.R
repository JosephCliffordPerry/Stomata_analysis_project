#full_surface_pipeline
source("D:/Stomatal_analysis_project/full_surface_pipeline/stitch_and_inference_wrapper.R")

polys_all <- batch_yolo_stitch(
  image_dir = "E:/Stomata/Sugarbeet_stomata_imaging/sugarbeet_all_mips",
  model_path = "E:/Stomata/Sugarbeet_stomata_imaging/beetmip_model2/beetmip_model2.pt"
)
source("D:/Stomatal_analysis_project/full_surface_pipeline/Poly_list_to_morphometric_structure.R")

all_poly_stats<-data.frame()
for (i in seq_along(polys_all)) {
cell_list<-polys_all[[i]][[1]]
metrics_list<-polys_all[[i]][[2]]
names(cell_list) <- rep(names(polys_all), length.out = length(cell_list))
image_df<-interpolate_cell_list(cell_list,metrics_list)
all_poly_stats<-rbind(all_poly_stats,image_df)
}

all_poly_stats_with_geometry<-add_cell_geometry_columns(all_poly_stats)
# all_poly_stats

all_poly_stats_with_geometry$Sample_id <- sub("-mip.*", "", all_poly_stats_with_geometry$image_id)

all_poly_stats_with_geometry$Sample_id <- sub(".nd2*", "", all_poly_stats_with_geometry$Sample_id)
all_poly_stats_with_geometry <- all_poly_stats_with_geometry %>%
  filter(area >= 500 & area <= 3000)

cell_geometry_umap<-umap::umap(all_poly_stats_with_geometry[,204:503])
library(ggplot2)

plot_df <- data.frame(
  UMAP1 = cell_geometry_umap$layout[,1],
  UMAP2 = cell_geometry_umap$layout[,2],
  Sample_id = all_poly_stats_with_geometry$Sample_id
)


ggplot(plot_df, aes(UMAP1, UMAP2, colour = Sample_id)) +
  geom_point(size = 1.5, alpha = 0.7) +
  theme_minimal()


NMAcompanion::make_NMA_profile_graphs(data = all_poly_stats_with_geometry,groups =all_poly_stats_with_geometry$Sample_id )

library(dplyr)
library(ggplot2)

plot_data <- all_poly_stats_with_geometry %>%
  mutate(
    circularity_group = ifelse(circularity > 0.95, "stomataoid", "pavementoid")
  )

ggplot(plot_data, aes(x = area)) +
  geom_histogram(bins = 40) +
  facet_wrap(~ circularity_group, scales = "free_y") +
  theme_minimal() +
  labs(
    x = "Cell Area",
    y = "Count",
    title = "Area Distribution by Circularity Group"
  )



ggplot(plot_data, aes(x = area)) +
  geom_density() +
  facet_wrap(~ circularity_group, scales = "free_y") +
  theme_minimal()
NMAcompanion::make_NMA_profile_graphs(data = plot_data,groups =plot_data$circularity_group )

