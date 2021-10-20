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

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
#download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl) # FICHIER A DISTANCE

#dl <- "~/projects/movielens/ml-10m.zip"    # Use Local File (faster)
#ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
#                 col.names = c("userId", "movieId", "rating", "timestamp"))
#movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

##### DEBUT A ENLEVER ####
#dl <- "~/projects/movielens/ml-1m.zip"   # /!\ MINI dataset
#ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-1m/ratings.dat"))), col.names = c("userId", "movieId", "rating", "timestamp"))
#movies <- str_split_fixed(readLines(unzip(dl, "ml-1m/movies.dat")), "\\::", 3)

dl <- "~/projects/movielens/ml-latest-small.zip"   # /!\ MICRO dataset
ratings <- fread(text = gsub(",", "\t", readLines(unzip(dl, "ml-latest-small/ratings.csv"))), col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-latest-small/movies.csv")), "\\,", 3)
##### FIN A ENLEVER ####

colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
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

######################
#   Begin ANALYSIS   #
######################

######### MULTITHREAD LIBRARY (needs OpenBLAS)
library(RhpcBLASctl)
Ncores <- get_num_procs()
blas_set_num_threads(Ncores)
omp_set_num_threads(Ncores)
###########################

## Set-up
   # Check/install required packages/libraries
   if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
   if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
   if(!require(recommenderlab)) install.packages("recommenderlab", repos = "http://cran.us.r-project.org")
   # Load required packages/libraries
   library(reshape2)     # For acast function
   library(ggplot2)     # For pretty graphics
   library(recommenderlab)     # For data analysis
   # Raise R memory limit size (Windows-only), or won't be able to allocate vector of size 5+Gb...
   memory.limit(size = 50000)

## Basic data to introduce the whole dataset (training + validation)
total_dataset <- full_join(edx,validation)
total_number_ratings <- nrow(total_dataset)
total_number_movies <- n_distinct(total_dataset$movieId)
total_number_users <- n_distinct(total_dataset$userId)
column_names <- colnames(total_dataset)
rm(total_dataset)

## Prepare training dataset : adapt the "edx" set for a recommenderlab analysis
   # 10-star scale + integer conversion (uses less RAM = less swapping = improved performance)
   edx$rating <- edx$rating*2
   edx$rating <- as.integer(edx$rating)
   edx$movieId <- as.integer(edx$movieId)
   edx$userId <- as.integer(edx$userId)
   # Save the edx and validation datasets to an external file (will be used later for our validation set final preparation)
   save(edx,validation, file = "EdxVal.RData")

##### Suppression valeurs inutilisées #####
#MissingVal <- anti_join(edx, validation, by = "movieId")
#MissingVal <- MissingVal %>% select(movieId) %>% group_by(movieId) %>% slice(1)
#edx <- anti_join(edx, MissingVal, by = "movieId")
###########################################


   # Convert training set to a matrix, then a realRatingMatrix (class used by recommenderlab)
   gc(verbose = FALSE)     # Free as much memory as possible
   Edx <- acast(edx, userId ~ movieId, value.var = "rating")
   Edx <- as(edx, "realRatingMatrix")
   gc(verbose = FALSE)     # Free memory


## Prepare the validation dataset for RMSE computation
   # Load edx and validation datasets
  load("EdxVal.RData")    ###
  # 10-star scale + integer conversion (uses less RAM = less swapping = improves performance)
  validation$rating <- validation$rating*2
  validation$rating <- as.integer(validation$rating)
  validation$movieId <- as.integer(validation$movieId)
  validation$userId <- as.integer(validation$userId)
   # Remove data that weren't used in this study
   validation <- validation %>% select(userId,movieId,rating)
   edx <- edx %>% select(userId,movieId,rating)
   # Detect missing movies in the validation set (present in the training but not in the validation set)
   MissingVal <- anti_join(edx, validation, by = "movieId")
   MissingVal <- MissingVal %>% group_by(movieId) %>% slice(1)
   # Fill these missing lines with empty (NA) ratings
   MissingVal$userId <- NA   ###
   MissingVal$rating <- NA   ###
   MissingVal <- as.data.frame(MissingVal)
   # Integrate empty rows after the validation set
   validation <- rbind(validation, MissingVal)    ###


   
   # Converts validation set to matrix, then realRatingMatrix (class used by recommenderlab)
  rm(edx)     # Free memory (recommenderlab will be using the "Edx" realRatingMatrix, not the edx dataframe)
   gc(verbose = FALSE)     # Free memory
   validation <- acast(validation, userId ~ movieId, value.var = "rating")
   validation <- as(validation, "realRatingMatrix")  
   gc(verbose = FALSE)     # Free memory

###### Enregistrement SETS #####
save(Edx, validation, file = "EdxVal.RData")
#load("EdxVal.RData")

#hist(getRatings(train))
# Normalized distribution of ratings
#hist(getRatings(normalize(train, method ="Z-score"))) 
# Number of movies rated by each user
#hist(rowCounts(train))
# Mean rating for each movie
#hist(colMeans(train))

##########################
#    TRAINING METHODS    #
##########################
#UBCF, IBCF, POPULAR, RANDOM, ALS, ALS_implicit, SVD, SVDF
   
##### SVD System #####
SVD.K <- 200 # Défaut = 10 (meilleur RMSE >= 500)
SVD.M <- 100 # Défaut = 100 (pas d'effet???)
SVD.N <- "Z-score" # center, Z-Score (pas d'effet???)
#recommend <- Recommender(data= Edx, method= "SVD", 
#   param= list(k = SVD.K, maxiter = SVD.M, normalize = SVD.N))

##### POPULAR System #####
POP.N <- "Z-score" # center, Z-Score (Z plus rapide ?)
recommend <- Recommender(data= Edx, method= "POPULAR", 
   param= list(normalize = POP.N))


##### UBCF System ##### ????? NE MARCHE PAS ?????
UBCF.M <- "cosine"
UBCF.N <- 10
UBCF.S <- FALSE
UBCF.W <- TRUE
UBCF.N <- "center"
UBCF.MM <- 0
UBCF.MP <- 0
#recommend <- Recommender(data= edx, method= "UBCF", 
#   param= list(method = UBCF.M, nn = UBCF.N, sample = UBCF.S, weighted = UBCF.W, normalize= UBCF.N, min_matching_items= UBCF.MM, min_predictive_items = UBCF.MP))
#recommend <- Recommender( Edx,"UBCF")

##### ALS System #####
ALS.L <- 0.001  # 0.1 par défaut (meilleur RMSE < 0.02)
ALS.F <- 50  # 10 par défaut (+ précision)
ALS.I <- 10  # 10 par défaut (++ temps, + précision)
ALS.M <- 1
#recommend <- Recommender(data= Edx, method= "ALS", 
#   param= list(lambda = ALS.L, n_factors = ALS.F, n_iterations = ALS.I, min_item_nr = ALS.M))

######CHRONO#####
start.time <- Sys.time()

##### SVDF System #####
SVDF.K <- 10  #10 par défaut
SVDF.G <- 0.015  #0,015 par défaut
SVDF.L <- 0.001  #0,001 par défaut (meilleur RMSE 0,01)
SVDF.minE <- 50  #50 par défaut
SVDF.MaxE <- 400 #200 par défaut (++temps, + précision)
SVDF.I <- 0.000001
SVDF.N <- "center"
SVDF.V <- FALSE
#recommend <- Recommender(data=Edx, method= "SVDF", 
#   param= list(k=SVDF.K, gamma=SVDF.G, lambda=SVDF.L, min_epochs=SVDF.minE, max_epochs=SVDF.MaxE, min_improvement=SVDF.I, normalize=SVDF.N, verbose=SVDF.V))


predictions <- predict(recommend, validation, type = "ratingMatrix")
Accuracy <- calcPredictionAccuracy(validation,predictions)
gc(verbose = FALSE)

rm(predictions)  # Freeing memory
gc(verbose = FALSE)

##### CHRONO#####
end.time <- Sys.time()
time.pred <- end.time - start.time
time.matrix
time.pred
##########################"
Accuracy["RMSE"]

save.image(file = "EKR-MovieLens.RData")

