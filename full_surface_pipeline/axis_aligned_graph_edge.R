axis_merge_vectorised <- function(
    polygons,
    distance_tol = 5,
    min_shared_length = 10
){
  suppressPackageStartupMessages({
    library(data.table)
  })
  
  # ===============================
  # EXTRACT + CLEAN COORDS
  # ===============================
  cleaned <- lapply(seq_along(polygons), function(id){
    
    p <- polygons[[id]]
    
    # ---- HANDLE sfg ----
    if (inherits(p, "sfg")) {
      if (length(p) == 0) return(NULL)
      p <- p[[1]]
    }
    
    # ---- GUARD ----
    if (
      is.null(p) ||
      !is.numeric(p) ||
      is.null(dim(p)) ||
      nrow(p) < 3 ||
      ncol(p) < 2
    ) return(NULL)
    
    p <- p[complete.cases(p), , drop = FALSE]
    if (nrow(p) < 3) return(NULL)
    
    # close ring
    if (sqrt(sum((p[1,] - p[nrow(p),])^2)) > 1e-6)
      p <- rbind(p, p[1,])
    
    if (nrow(p) < 4) return(NULL)
    
    list(id = id, coords = p)
  })
  
  cleaned <- cleaned[!sapply(cleaned, is.null)]
  if (length(cleaned) == 0) return(NULL)
  
  # ===============================
  # EXTRACT SEGMENTS
  # ===============================
  seg_dt <- rbindlist(lapply(cleaned, function(obj){
    
    coords <- obj$coords
    id <- obj$id
    
    out <- list()
    k <- 1
    
    for(i in 1:(nrow(coords)-1)){
      
      x1 <- coords[i,1]; y1 <- coords[i,2]
      x2 <- coords[i+1,1]; y2 <- coords[i+1,2]
      
      if (abs(x1 - x2) < 1e-6){
        out[[k]] <- list(id=id, orient=1, coord=x1,
                         start=min(y1,y2), end=max(y1,y2))
        k <- k + 1
        
      } else if (abs(y1 - y2) < 1e-6){
        out[[k]] <- list(id=id, orient=0, coord=y1,
                         start=min(x1,x2), end=max(x1,x2))
        k <- k + 1
      }
    }
    
    if (length(out)) rbindlist(out) else NULL
  }), fill = TRUE)
  
  if (is.null(seg_dt) || nrow(seg_dt) == 0) return(NULL)
  
  setDT(seg_dt)
  
  # ===============================
  # SELF JOIN
  # ===============================
  setkey(seg_dt, orient, coord)
  
  pairs <- seg_dt[
    seg_dt,
    allow.cartesian = TRUE,
    nomatch = 0
  ][
    id < i.id
  ]
  
  if (nrow(pairs) == 0) return(NULL)
  
  # ===============================
  # OVERLAP
  # ===============================
  overlap_1d <- function(a1,a2,b1,b2){
    pmax(0, pmin(a2,b2) - pmax(a1,b1))
  }
  
  pairs[, overlap := overlap_1d(start, end, i.start, i.end)]
  pairs <- pairs[overlap > min_shared_length]
  
  if (nrow(pairs) == 0) return(NULL)
  
  # ===============================
  # ADJACENCY FILTER
  # ===============================
  pairs[, gap := pmin(
    abs(start - i.start),
    abs(end - i.end)
  )]
  
  pairs <- pairs[gap <= distance_tol]
  
  if (nrow(pairs) == 0) return(NULL)
  
  # ===============================
  # OUTPUT EDGES
  # ===============================
  edges <- unique(pairs[, .(id1 = id, id2 = i.id)])
  
  as.matrix(edges)
}

#axis_edges<-axis_merge_vectorised(output$polygons)

