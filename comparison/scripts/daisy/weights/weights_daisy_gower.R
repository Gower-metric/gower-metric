library(cluster)
library(readr)

data <- read_csv("/Users/bk/Documents/Zajęcia (październik - luty 2025)/Uczenie Maszynowe/gower-similarity/comparison/data/weights_custom.csv")

data$age <- as.numeric(data$age)                
data$hours_per_week <- as.numeric(data$hours_per_week)

data$gender <- factor(data$gender)
edu_levels <- c("low", "medium", "high")
data$education <- factor(data$education, levels  = edu_levels, ordered = TRUE)

data$income <- factor(data$income,  levels = c(0, 1))
data$infected <- factor(data$infected, levels = c(0, 1))

w <- c(age = 1, gender = 2, education = 3, hours_per_week = 4, income = 5, infected = 6)

d <- daisy(data, metric = "gower", type = list(symm = "income", asymm = "infected"), weights = w)
write.table(round(as.matrix(d), digits = 6), file="/Users/bk/Documents/Zajęcia (październik - luty 2025)/Uczenie Maszynowe/gower-similarity/comparison/scripts/daisy/weights/results/weights_daisy.txt", row.names=FALSE, col.names=FALSE)