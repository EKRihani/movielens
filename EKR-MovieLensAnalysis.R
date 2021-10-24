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

dl <- "~/projects/movielens/ml-10m.zip"    # Use Local File (faster)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

##### DEBUT A ENLEVER ####
#dl <- "~/projects/movielens/ml-1m.zip"   # /!\ MINI dataset
#ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-1m/ratings.dat"))), col.names = c("userId", "movieId", "rating", "timestamp"))
#movies <- str_split_fixed(readLines(unzip(dl, "ml-1m/movies.dat")), "\\::", 3)

#dl <- "~/projects/movielens/ml-latest-small.zip"   # /!\ MICRO dataset
#ratings <- fread(text = gsub(",", "\t", readLines(unzip(dl, "ml-latest-small/ratings.csv"))), col.names = c("userId", "movieId", "rating", "timestamp"))
#movies <- str_split_fixed(readLines(unzip(dl, "ml-latest-small/movies.csv")), "\\,", 3)
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

#################################################################

##########################
#   BEGINNING ANALYSIS   #
##########################

######### MULTITHREAD LIBRARY (needs OpenBLAS)
#library(RhpcBLASctl)
#Ncores <- get_num_procs()
#blas_set_num_threads(Ncores)
#omp_set_num_threads(Ncores)
###########################

## Set-up
# Check/install required packages/libraries
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(ggrepel)) install.packages("ggrepel", repos = "http://cran.us.r-project.org")
if(!require(recommenderlab)) install.packages("recommenderlab", repos = "http://cran.us.r-project.org")
if(!require(parallel)) install.packages("parallel", repos = "http://cran.us.r-project.org")
# Load required packages/libraries
library(reshape2)       # For acast function
library(ggplot2)        # For pretty graphics
library(ggrepel)        # For repelled labels on graphics
library(recommenderlab) # For data analysis
library(parallel)       # For parSapply function (multi-thread sapply)
# Raise R memory limit size (Windows-only), or won't be able to allocate vector of size 5+Gb during our Matrix/realRatingMatrix conversion...
memory.limit(size = 50000)

## Basic data to introduce the whole dataset (training + validation)
total_dataset <- full_join(edx,validation)
total_number_ratings <- nrow(total_dataset)
total_number_movies <- n_distinct(total_dataset$movieId)
total_number_users <- n_distinct(total_dataset$userId)
column_names <- colnames(total_dataset)
rm(total_dataset)    # Free some memory
save(edx,validation, file = "edxval.RData")

load(file = "edxval.RData")
## Prepare training dataset : adapt the "edx" set for a recommenderlab analysis
# 10-star scale + integer conversion (uses less RAM = less swapping = improved performance)
edx$rating <- edx$rating*2
edx$rating <- as.integer(edx$rating)
edx$movieId <- as.integer(edx$movieId)
edx$userId <- as.integer(edx$userId)
# Save the edx and validation datasets to an external file (will be used later for our validation set final preparation)
save(edx,validation, file = "edxval.RData")
##### Suppression valeurs inutilisées = GAIN PERFS #####
missing_movieId <- anti_join(edx, validation, by = "movieId")
missing_movieId <- missing_movieId %>% select(movieId) %>% group_by(movieId) %>% slice(1)
edx <- anti_join(edx, missing_movieId, by = "movieId")
################   A ENLEVER +++++   ##################

# Convert data set to matrix, then realRatingMatrix (class used by recommenderlab)
gc(verbose = FALSE)     # Free as much memory as possible
edx_rrm <- acast(edx, userId ~ movieId, value.var = "rating")
edx_rrm <- as(edx, "realRatingMatrix")
gc(verbose = FALSE)     # Free memory
rm(edx)  # Free memory

#######################################
#    BENCHMARKING TRAINING METHODS    #
#######################################

## Build the benchmarking data set
# Reduce set size
train_size <- 0.01
set.seed(1234, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1234)`
reduction_index <- sample(x = seq(1, nrow(edx_rrm)), size = nrow(edx_rrm) * train_size, replace = FALSE)
edx_rrm_small <- edx_rrm[reduction_index]
# Build the training and evaluation sets
train_ratio <- 0.9
test_index <- sample(x = seq(1, nrow(edx_rrm_small)), size = nrow(edx_rrm_small)*train_ratio, replace = FALSE)
edx_rrm_train <- edx_rrm_small[test_index]
edx_rrm_test <- edx_rrm_small[-test_index]

## Run the first benchmark, all methods, default parameters (no IBCF: abysmal performance)
list_methods <- c("RANDOM", "POPULAR", "LIBMF", "SVD", "SVDF", "ALS", "ALS_implicit", "UBCF") # For 1%
#list_methods <- c("RANDOM", "POPULAR", "LIBMF", "SVD", "UBCF") # Fpr 2-5%
#list_methods <- c("RANDOM", "POPULAR","LIBMF", "SVD") # Above 5%

# Benchmark (time and RMSE) for each method
benchmark <- function(model){
   start_time <- Sys.time()
   recommend <- Recommender(data = edx_rrm_train, method = model)  # Set recommendation parameters
   prediction <- predict(recommend, edx_rrm_test, type = "ratingMatrix")  # Run prediction
   accuracy <- calcPredictionAccuracy(edx_rrm_test,prediction) # Compute accuracy
   end_time <- Sys.time()
   running_time <- round(difftime(end_time, start_time, units = "secs"),2) # Time difference, unit forced (so mins and secs aren't mixed...)
   rmse <- as.numeric(round(accuracy["RMSE"]/2,4)) # Convert 10-stars RMSE to 5-stars RMSE
   c(rmse, running_time)
}
start_time <- Sys.time() ### A enlever
# Set multithreading
#n_threads <- detectCores()
#cluster <- makeCluster(n_threads)
#clusterExport(cluster, varlist=c("edx_rrm_test","edx_rrm_train"))
#clusterEvalQ(cluster, library(recommenderlab))

# Report and plot benchmarking results
benchmark_result <- as.data.frame(t(sapply(X = list_methods, FUN = benchmark)))
#benchmark_result <- as.data.frame(t(parSapply(cl = cluster, X = list_methods, FUN = benchmark))) # Parallel sapply
#stopCluster(cluster) # Stop multithread
colnames(benchmark_result) <- c("RMSE", "time")
benchmark_result$RMSE <- as.numeric(benchmark_result$RMSE)
benchmark_result$time <- as.numeric(benchmark_result$time)
benchmark_result
end_time <- Sys.time() ### A enlever
end_time - start_time ### A enlever
benchmark_result %>%
   ggplot(aes(x = time, y = RMSE, label = row.names(.))) +
   geom_point() +
   geom_text_repel() +
   ggtitle("Recommanderlab Models Performance")

gc(verbose = FALSE)     # Free memory
###############################################
#    FINE-TUNING SELECTED TRAINING METHODS    #
###############################################

# Fitting function that measures time and RMSE for each function and parameter

##### A MODIFIER #####
fitting <- function(model, parameter, value){
   start_time <- Sys.time()
   recommend <- Recommender(data = edx_rrm_train, model)  # Set recommendation parameters
   prediction <- predict(recommend, edx_rrm_test, type = "ratingMatrix")  # Run prediction
   accuracy <- calcPredictionAccuracy(edx_rrm_test,prediction) # Compute accuracy
   end_time <- Sys.time()
   running_time <- difftime(end_time, start_time, units = "secs") # Time difference, unit forced (so mins and secs aren't mixed...)
   rmse <- as.numeric(round(accuracy["RMSE"]/2,4)) # Convert 10-stars RMSE to 5-stars RMSE
   c(rmse, running_time)
}
##### SVD System #####
SVD.K <- 10 # Défaut = 10 (meilleur RMSE >= 500)
SVD.M <- 100 # Défaut = 100 (pas d'effet???)
SVD.N <- "center" # center, Z-Score (pas d'effet???)
#recommend <- Recommender(data=edx_rrm, method="SVD", param=list(k=SVD.K, maxiter=SVD.M, normalize=SVD.N))



##### POPULAR Method #####
POP.N <- "center" # center, Z-Score (Z plus rapide ?)
#recommend <- Recommender(data=edx_rrm, method="POPULAR", param=list(normalize=POP.N))

##### LIBMF Method #####
LIBMF.D <- 100  # 10 par défaut (+++ précision)
LIBMF.P <- 0.01   # 0.01 par défaut
LIBMF.Q <- 0.01  #0.01 par défaut
LIBMF.T <- 16   # 1 par défaut
#recommend <- Recommender(data=edx_rrm,method="LIBMF", param=list(dim=LIBMF.D,costp_l2=LIBMF.P, costq_l2=LIBMF.Q,  nthread=LIBMF.T))
#recommend <- Recommender(data=edx_rrm,method="LIBMF")

#####################################################
#    CALCULATING RMSE AGAINST THE VALIDATION SET    #
#####################################################

## Prepare the validation dataset for RMSE computation
# Load edx and validation datasets
#load("edxval.RData")    ### A remettre
# 10-star scale + integer conversion (uses less RAM = less swapping = improves performance)
validation$rating <- validation$rating*2
validation$rating <- as.integer(validation$rating)
validation$movieId <- as.integer(validation$movieId)
validation$userId <- as.integer(validation$userId)

# Remove data that weren't used in this study
validation <- validation %>% select(userId,movieId,rating)
edx <- edx %>% select(userId,movieId,rating)
# Detect missing movies in the validation set (present in the training but not in the validation set), keeping 1 movieId for each
missing_movieId <- anti_join(edx, validation, by = "movieId")
missing_movieId <- missing_movieId %>% group_by(movieId) %>% slice(1)
# Fill these missing lines with empty (NA) ratings
#missing_movieId$userId <- NA   ### A remettre
#missing_movieId$rating <- NA   ### A remettre
#missing_movieId <- as.data.frame(missing_movieId)  ### A remettre
# Integrate empty rows after the validation set
#validation <- rbind(validation, missing_movieId)    ###  A remettre

# Convert validation set to matrix, then realRatingMatrix (class used by recommenderlab)
gc(verbose = FALSE)
validation_rrm <- acast(validation, userId ~ movieId, value.var = "rating")
validation_rrm <- as(validation, "realRatingMatrix")
gc(verbose = FALSE)
rm(edx, validation)     # edx/validation won't be used anymore ; keeping only edx_rrm/validation_rrm
   
###### Enregistrement SETS #####
save(edx_rrm, validation_rrm, file = "edxval_rrm.RData")
#load("edxval.RData")
######## A ENLEVER +++++ ######

#hist(getRatings(train))
# Normalized distribution of ratings
#hist(getRatings(normalize(train, method ="Z-score"))) 
# Number of movies rated by each user
#hist(rowCounts(train))
# Mean rating for each movie
#hist(colMeans(train))

predictions <- predict(recommend, validation_rrm, type = "ratingMatrix") #type = realRatingMatrix ?
accuracy <- calcPredictionAccuracy(validation_rrm, predictions)
gc(verbose = FALSE)

rm(predictions)  # Free memory
gc(verbose = FALSE)
accuracy["RMSE"]/2

save.image(file = "EKR-MovieLens.RData")

