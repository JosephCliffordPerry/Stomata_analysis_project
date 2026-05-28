# --------------------------------
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
  polygons = stage1_sf$geometry,
  axis_edges = bbox_edges)
plot_sf_overlay(
  img = img,
  sf_polys = result,
  alpha = 0.4,
  line_col = "yellow"
)
polys_2<-split_isolated_polygons(
  result,
  k = 10,
  overlap_tol = 20,
  axis_length_threshold = 30,
  area_min = 500,
axis_edges = bbox_edges)
plot_sf_overlay(
  img = img,
  sf_polys = polys_2$isolated_polygons,
  alpha = 0.4,
  line_col = "yellow"
)
