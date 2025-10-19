library(cluster)
library(readr)

iris <- read_csv("data/files/iris.csv")
iris$variety <- factor(iris$variety)

d <- daisy(iris, metric = "gower")
D <- as.matrix(d)
block <- D[1:10, 6:15]

print(round(block, 8))