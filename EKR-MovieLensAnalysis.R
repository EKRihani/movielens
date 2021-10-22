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
dl <- "~/projects/movielens/ml-1m.zip"   # /!\ MINI dataset
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-1m/ratings.dat"))), col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-1m/movies.dat")), "\\::", 3)

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
library(reshape2)       # For acast function
library(ggplot2)        # For pretty graphics
library(recommenderlab) # For data analysis
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


## Build the training and prevalidation sets, on a smaller dataset
# Reduce set size
train_size <-0.5
index <- sample(x = seq(1, nrow(edx_rrm)), size = nrow(edx_rrm) * train_size, replace = FALSE)
small_edx_rrm <- edx_rrm[index]
# Building the training and evaluation sets with Recommenderlab
evaluation <- evaluationScheme(data = small_edx_rrm, method = "split", train = 0.9, given = 1)

# Run the first benchmark, all methods, default parameters
#list_methods <- c("RANDOM", "POPULAR", "IBCF", "UBCF", "SVD", "SVDF", "ALS", "ALS_implicit", "LIBMF")
list_methods <- c("RANDOM", "POPULAR","LIBMF") ### Liste Courte

# Benchmark (time and RMSE) each method
benchmark <- function(model){
   start_time <- Sys.time()
   recommend <- Recommender(getData(evaluation, "train"), model)  # Set recommendation parameters
   predict <- predict(recommend, getData(evaluation, "known"), type = "ratingMatrix")  # Run prediction
   accuracy <- calcPredictionAccuracy(predict,getData(evaluation, "unknown")) # Compute accuracy
   end_time <- Sys.time()
   running_time <- end_time - start_time
   rmse <- as.numeric(accuracy["RMSE"]/2) # Convert 10-stars RMSE to 5-stars RMSE
   c(model, rmse, running_time)
}

# Report and plot benchmarking results
benchmark_result <- as.data.frame(t(sapply(X = list_methods, FUN = benchmark)))
colnames(benchmark_result) <- c("model", "RMSE", "time")
benchmark_result
benchmark_result %>%
   ggplot(aes(x = RMSE, y = time)) +
   geom_point() +
   ggtitle("Recommanderlab Model Performance") 
   #geom_text()

######################################
#    FINE-TUNING TRAINING METHODS    #
######################################   

##### Section à réinsérer ici #####


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


######################################
#    FINE-TUNING TRAINING METHODS    #
######################################

##### RANDOM Method #####
#recommend <- Recommender(data=edx_rrm, method="RANDOM")

##### SVD System #####
SVD.K <- 10 # Défaut = 10 (meilleur RMSE >= 500)
SVD.M <- 100 # Défaut = 100 (pas d'effet???)
SVD.N <- "center" # center, Z-Score (pas d'effet???)
#recommend <- Recommender(data=edx_rrm, method="SVD", param=list(k=SVD.K, maxiter=SVD.M, normalize=SVD.N))

##### POPULAR Method #####
POP.N <- "center" # center, Z-Score (Z plus rapide ?)
#recommend <- Recommender(data=edx_rrm, method="POPULAR", param=list(normalize=POP.N))

##### IBCF Method #####
IBCF.K <- 30         # k (default = 30)
IBCF.M <- "Cosine"   # Method (default = Cosine)
IBCF.N <- "center"   # Normalize (default = center)
IBCF.NSM <- FALSE    # Normalize Sim Matrix (default = FALSE)
IBCF.A <-0.5         # Alpha (default = 0.5)
IBCF.NAZ <- FALSE    #Na as Zero (default = FALSE)
#recommend <- Recommender(data = edx_rrm, method = "IBCF", param = list(k=IBCF.K, method=IBCF.M, normalize=IBCF.N, normalize_sim_matrix=IBCF.NSM, alpha=IBCM.A, na_as_zero=IBCM.NAZ))

##### UBCF Method ##### ????? NE MARCHE PAS ?????
UBCF.M <- "cosine"
UBCF.N <- 25
UBCF.S <- FALSE
UBCF.W <- TRUE
UBCF.N <- "center"
UBCF.MM <- 0
UBCF.MP <- 0
#recommend <- Recommender(data=edx_rrm, method="UBCF", param=list(method=UBCF.M, nn=UBCF.N, sample=UBCF.S, weighted=UBCF.W, normalize=UBCF.N, min_matching_items=UBCF.MM, min_predictive_items=UBCF.MP))

##### LIBMF Method #####
LIBMF.D <- 100  # 10 par défaut (+++ précision)
LIBMF.P <- 0.01   # 0.01 par défaut
LIBMF.Q <- 0.01  #0.01 par défaut
LIBMF.T <- 16   # 1 par défaut
#recommend <- Recommender(data=edx_rrm,method="LIBMF", param=list(dim=LIBMF.D,costp_l2=LIBMF.P, costq_l2=LIBMF.Q,  nthread=LIBMF.T))
recommend <- Recommender(data=edx_rrm,method="LIBMF")

##### ALS Method #####
ALS.L <- 0.001  # 0.1 par défaut (meilleur RMSE < 0.02)
ALS.F <- 50  # 10 par défaut (+ précision)
ALS.I <- 10  # 10 par défaut (++ temps, + précision)
ALS.M <- 1
#recommend <- Recommender(data=edx_rrm, method="ALS", param=list(lambda=ALS.L, n_factors=ALS.F, n_iterations=ALS.I, min_item_nr=ALS.M))

##### SVDF Method #####
SVDF.K <- 10  #10 par défaut
SVDF.G <- 0.015  #0,015 par défaut
SVDF.L <- 0.001  #0,001 par défaut (meilleur RMSE 0,01)
SVDF.minE <- 50  #50 par défaut
SVDF.MaxE <- 400 #200 par défaut (++temps, + précision)
SVDF.I <- 0.000001
SVDF.N <- "center"
SVDF.V <- FALSE
#recommend <- Recommender(data=edx_rrm, method= "SVDF", param= list(k=SVDF.K, gamma=SVDF.G, lambda=SVDF.L, min_epochs=SVDF.minE, max_epochs=SVDF.MaxE, min_improvement=SVDF.I, normalize=SVDF.N, verbose=SVDF.V))

######CHRONO#####
start.time <- Sys.time()
predictions <- predict(recommend, validation_rrm, type = "ratingMatrix") #type = realRatingMatrix ?
accuracy <- calcPredictionAccuracy(validation_rrm, predictions)
gc(verbose = FALSE)

rm(predictions)  # Free memory
gc(verbose = FALSE)

##### CHRONO#####
end.time <- Sys.time()
time.pred <- end.time - start.time
time.pred
##########################"
accuracy["RMSE"]/2

save.image(file = "EKR-MovieLens.RData")

