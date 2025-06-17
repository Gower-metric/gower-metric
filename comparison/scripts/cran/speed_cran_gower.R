library(gower)
library(readr)

data <- read.csv("your_path/adult_reduced.csv")
d1 <- data[1:32558,]
d2 <- data[2:32559,]

n_runs <- 100
times <- numeric(n_runs)

for (i in seq_len(n_runs)) {
  start.time <- Sys.time()
  d <- gower_dist(d1, d2)
  times[i] <- as.numeric(difftime(Sys.time(), start.time, units = "secs"))
}