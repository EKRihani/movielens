##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

#Chrono A ENLEVER
#start.time <- Sys.time()

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
#download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl) # Download Remote File

#dl <- "~/projects/movielens/ml-10m.zip"    # Use Local File (faster)
#ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
#                 col.names = c("userId", "movieId", "rating", "timestamp"))
#movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

##### DEBUT A ENLEVER ####
#dl <- "~/projects/movielens/ml-20m.zip"   # /!\ GROS dataset
#ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-20m/ratings.csv"))), col.names = c("userId", "movieId", "rating", "timestamp"))
#movies <- str_split_fixed(readLines(unzip(dl, "ml-20m/movies.csv")), "\\::", 3)

dl <- "~/projects/movielens/ml-1m.zip"   # /!\ MINI dataset
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-1m/ratings.dat"))), col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-1m/movies.dat")), "\\::", 3)
##### FIN A ENLEVER ####

colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
   semi_join(edx, by = "movieId") %>%
   semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# Begin ANALYSIS
##########################################################

# Chrono A ENLEVER
#end.time <- Sys.time()
#time.taken <- end.time - start.time
#time.taken
#rm(end.time, time.taken, start.time)
# Chrono A ENLEVER

# Installing required packages/libraries
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
#if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
#if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(recommenderlab)) install.packages("recommenderlab", repos = "http://cran.us.r-project.org")

# Loading required packages/libraries
library(dplyr)
#library(lubridate)
#library(stringr)
library(reshape2)
library(ggplot2)
library(recommenderlab)


# Basic data to introduce the global dataset (training + validation)
total_number_ratings <- nrow(union(edx,validation))
total_number_movies <- n_distinct(union(edx$movieId, validation$movieId))
total_number_users <-n_distinct(union(edx$userId, validation$userId))
column_names <- colnames(edx)

# First remove movies/users not found in validation set ?


# Converting edx and validation sets to matrix used by RecommanderLab
EDX <- acast(edx, userId ~ movieId, value.var = "rating")
#VALIDATION <- as(acast(validation, userId ~ movieId, value.var = "rating"), "realRatingMatrix")
EDX <- as(EDX, "realRatingMatrix")

Edx <- EDX
VALIDATION <- Edx[5401:6040]
EDX <- Edx[1:5401]


#hist(getRatings(train))

# Normalized distribution of ratings
#hist(getRatings(normalize(train, method ="Z-score"))) 
# Number of movies rated by each user
#hist(rowCounts(train))
# Mean rating for each movie
#hist(colMeans(train))

#CHRONO
start.time <- Sys.time()
recom <- Recommender(EDX, "UBCF")
#recom <- Recommender(getData(e, "train"), "UBCF")
#UBCF, IBCF, POPULAR, RANDOM, ALS, ALS_implicit, SVD, SVDF

predi <- predict(recom, VALIDATION, type = "ratingMatrix")
#predi <- predict(recom, getData(e,"known"), type = "ratings")

calcPredictionAccuracy(VALIDATION,predi)
#Accuracy <- calcPredictionAccuracy(predi,getData(e, "unknown"))
#Accuracy["RMSE"]
#RMSE(predi,getData(e, "unknown"))

# CHRONO
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
#rm(end.time, time.taken, start.time)
# Chrono A ENLEVER

save.image(file = "EKR-MovieLens.RData")