library(cluster)
library(readr)

data <- read_csv("your_path/adult_reduced.csv")

data$race <- factor(data$race)
data$sex <- factor(data$sex)

data <- data[1:1000,]

n_runs <- 100
times <- numeric(n_runs)

for (i in seq_len(n_runs)) {
  start.time <- Sys.time()
  d <- daisy(data, metric = "gower")
  times[i] <- as.numeric(difftime(Sys.time(), start.time, units = "secs"))
}