library(gower)
dat1 <- iris[1:10,]
dat2 <- iris[6:15,]
result <- gower_dist(dat1, dat2)

write.table(format(result, digits = 5),
            file = "comparison/scripts/cran/cran_results/cran_gower.txt",
            row.names = FALSE,
            col.names = FALSE,
            quote = FALSE)