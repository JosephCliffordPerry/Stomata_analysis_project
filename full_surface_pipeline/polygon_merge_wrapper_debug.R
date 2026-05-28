# --------------------------------
plot_sf_overlay(
  img = img,
  sf_polys = polys,
  alpha = 0.4,
  line_col = "red"
)

axis_edges <- axis_merge_vectorised(polys, distance_tol = 5,
                                    min_shared_length = 5)

stage1_sf <- merge_polygons_convex_hull(
  polygons = polys,
  axis_edges = axis_edges
)
plot_sf_overlay(
  img = img,
  sf_polys = stage1_sf,
  alpha = 0.4,
  line_col = "yellow"
)
# --------------------------------
# Stage 2: bbox/iou merge
# --------------------------------
bbox_edges <- build_bbox_iou_edges(
  stage1_sf,
  centroid_dist_tol = 64,
  bbox_iou_threshold = 0.8
)


result <- merge_polygons_convex_hull(
  polygons = stage1_sf,
  axis_edges = bbox_edges)
plot_sf_overlay(
  img = img,
  sf_polys = result,
  alpha = 0.4,
  line_col = "yellow"
)
result_polys <- split_isolated_polygons(
  polygons = result,
  original_polygons = polys,
  k = 10,
  overlap_tol = 20,
  axis_length_threshold = 30,
  area_min = 500
)
plot_sf_overlay(
  img = img,
  sf_polys = result_polys$isolated_polygons,
  alpha = 0.4,
  line_col = "yellow"
)
plot_sf_overlay(
  img = img,
  sf_polys = result_polys$remaining_original_polygons,
  alpha = 0.4,
  line_col = "yellow"
)

### Filter round 2 

polys_2 <-result_polys$remaining_original_polygons
axis_edges2 <- axis_merge_vectorised(polys_2, distance_tol = 30,
                                    min_shared_length = 5)

stage2_sf <- merge_polygons_convex_hull(
  polygons = polys_2,
  axis_edges = axis_edges2
)
plot_sf_overlay(
  img = img,
  sf_polys = stage2_sf,
  alpha = 0.4,
  line_col = "yellow"
)
# --------------------------------
# Stage 2: bbox/iou merge
# --------------------------------
bbox_edges2 <- build_bbox_iou_edges(
  stage2_sf,
  centroid_dist_tol = 64,
  bbox_iou_threshold = 0.5
)


result2 <- merge_polygons_convex_hull(
  polygons = stage2_sf,
  axis_edges = bbox_edges2)

plot_sf_overlay(
  img = img,
  sf_polys = result2,
  alpha = 0.4,
  line_col = "yellow"
)
 result_polys_2<- split_isolated_polygons(
  polygons = result2,
  original_polygons = polys_2,
  k = 10,
  overlap_tol = 300,
  axis_length_threshold = 30,
  area_min = 500
)
plot_sf_overlay(
  img = img,
  sf_polys = result_polys_2$isolated_polygons,
  alpha = 0.4,
  line_col = "yellow"
)
plot_sf_overlay(
  img = img,
  sf_polys = result_polys_2$remaining_original_polygons,
  alpha = 0.4,
  line_col = "yellow"
)

plot_sf_overlay(
  img = img,
  sf_polys = rbind(polys_1$isolated_polygons,polys_2$isolated_polygons),
  alpha = 0.4,
  line_col = "yellow"
)


plot_isolation_debug(
  img = img,
  original_polygons = polys,
  used_original_polygons = polys_2$used_original_polygons,
  remaining_original_polygons = polys_2$remaining_original_polygons,
  isolated_polygons = polys_2$isolated_polygons
)
