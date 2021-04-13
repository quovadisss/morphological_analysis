library('igraph')
library('dplyr')
library('ggplot2')
library('readxl')

org_xlsx <- read.csv('fintech_news.csv')
A <- as.matrix(org_xlsx)
diag(A) <- 0

g1 = graph.adjacency(A, mode='undirected', weighted = TRUE, diag=FALSE)

# Degree Centrality
deg_g1 <- degree(g1)
D1 <- as.matrix(deg_g1)

# Betweeness Centrality
btw1 <- betweenness(g1)
B1 <- as.matrix(btw1)

# Eigen centrality
eigen1 <- eigen_centrality(g1)
E1 <- as.matrix(eigen1)

# Closeness centrality
close1 <- closeness(g1)
C1 <- as.matrix(close1)

# Constraint
const_doc <- constraint(g1, nodes=V(g1), weights=NULL)
const_doc_m <- as.matrix(const_doc)

#Effective Size
# definite a function like this
ego.effective.size <- function(g, ego, ...) {
  egonet <- induced.subgraph(g, neighbors(g, ego, ...))
  n <- vcount(egonet)
  t <- ecount(egonet)
  return(n - (2 * t) / n)
}

effective.size <- function(g, ego=NULL, ...) {
  if(!is.null(ego)) {
    return(ego.effective.size(g, ego, ...))
  }
  return (sapply(V(g), function(x) {ego.effective.size(g,x, ...)}))
}
effective_doc <- effective.size(g1, mode='all')
effective_doc_m <- as.matrix(effective_doc)


# save dataframe as csv
write.csv(effective_doc_m, 'effective_fintech_news.csv')
write.csv(const_doc_m, 'constraint_fintech_news.csv')
write.csv(E1, 'eigen_fintech_news.csv')
write.csv(B1, 'between_fintech_news.csv')
write.csv(C1, 'closeness_fintech.csv')

