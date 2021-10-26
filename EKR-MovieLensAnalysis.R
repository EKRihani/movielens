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


####################
#   SYSTEM SETUP   #
####################

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

######### MULTITHREAD LIBRARY ??? (needs OpenBLAS)
#library(RhpcBLASctl)
#Ncores <- get_num_procs()
#blas_set_num_threads(Ncores)
#omp_set_num_threads(Ncores)
###########################

######################
#   BASIC ANALYSIS   #
######################

# Introduce the whole dataset (training + validation)
total_dataset <- full_join(edx,validation)
total_number_ratings <- nrow(total_dataset)
total_number_movies <- n_distinct(total_dataset$movieId)
total_number_users <- n_distinct(total_dataset$userId)
column_names <- colnames(total_dataset)
rm(total_dataset)    # Free some memory

#Save the edx and validation datasets to an external file (will be used later for our validation set final preparation)
save(edx,validation, file = "edxval.RData")
#load(file = "edxval.RData")
# Free some memory
rm(validation)    # Won't be needed until the final RMSE computation
gc(verbose = FALSE)     # Free as much memory as possible

## Prepare training dataset : adapt the "edx" set for a recommenderlab analysis
# 10-star scale + integer conversion (uses less RAM = less swapping = improved performance)
#edx$rating <- edx$rating*2
#edx$rating <- as.integer(edx$rating)
#edx$movieId <- as.integer(edx$movieId)
#edx$userId <- as.integer(edx$userId)

# Convert data set to matrix, then realRatingMatrix (class used by recommenderlab)

edx_rrm <- acast(edx, userId ~ movieId, value.var = "rating")
edx_rrm <- as(edx_rrm, "realRatingMatrix")
gc(verbose = FALSE)     # Free memory
rm(edx)  # Free memory

#######################################
#    BENCHMARKING TRAINING METHODS    #
#######################################
set.seed(1234, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1234)`

# Create empty objects for our training and data sets
edx_rrm_train = NULL
edx_rrm_test = NULL

# Define dataset size-reducing and splitting (train/evaluation) function
dataset_build <- function(train_size){
   reduction_index <- sample(x = seq(1, nrow(edx_rrm)), size = nrow(edx_rrm) * train_size, replace = FALSE)
   edx_rrm_small <- edx_rrm[reduction_index]
   test_index <- sample(x = seq(1, nrow(edx_rrm_small)), size = nrow(edx_rrm_small)*0.1, replace = FALSE)
   edx_rrm_train <<- edx_rrm_small[-test_index]
   edx_rrm_test <<- edx_rrm_small[test_index]
}

# Define benchmark function (time and RMSE)
bench <- function(model){
   start_time <- Sys.time()   # Start chronometer
   recommend <- Recommender(data = edx_rrm_train, method = model)  # Set recommendation parameters
   prediction <- predict(recommend, edx_rrm_test, type = "ratingMatrix")  # Run prediction
   accuracy <- calcPredictionAccuracy(edx_rrm_test,prediction) # Compute accuracy
   end_time <- Sys.time()     # Stop chronometer
   running_time <- round(difftime(end_time, start_time, units = "secs"),2) # Time difference, unit forced (so mins and secs aren't mixed...)
   rmse <- as.numeric(round(accuracy["RMSE"],4))   # Compute RMSE with 4 digits
   c(rmse, running_time)   # Reports RMSE and running time
   #gc(verbose = FALSE)     # Free memory
}

# Define report function
run_bench <- function(model_list){
   result <- as.data.frame(t(sapply(X = model_list, FUN = bench)))
   colnames(result) <- c("RMSE", "time")  # Add column names
   result$RMSE <- as.numeric(result$RMSE) # Convert factors to numeric values
   result$time <- as.numeric(result$time)
   result}

# Define plot function
plot_bench <- function(benchresult){
   benchresult %>%
      ggplot(aes(x = time, y = RMSE, label = row.names(.))) +
         geom_point() +
         scale_x_continuous(limits = c(0,NA)) +
         geom_text_repel() +  # Add labels
         geom_hline(yintercept = 0.9, linetype = "dashed", color = "darkred", alpha = 0.4) + # Minimal objective
         geom_hline(yintercept = 0.865, linetype = "dashed", color = "darkgreen", alpha = 0.4)  # Optimal objective 
}

# Define tested models and dataset sizes (no IBCF)
list_methods1 <- c("RANDOM", "POPULAR", "LIBMF", "SVD", "SVDF", "ALS", "ALS_implicit", "UBCF")
list_methods2 <- c("POPULAR", "LIBMF", "SVD", "UBCF")
list_methods3 <- c("POPULAR","LIBMF", "SVD")
train_size1 <- 0.005
train_size2 <- 0.02
train_size3 <- 0.1

start.time <- Sys.time()  ### A SUPPRIMER
# Build the 3 sets, run the 3 benchmarks, create the 3 plots
dataset_build(train_size1)
benchmark_result1 <- run_bench(list_methods1)
plot_result1 <- plot_bench(benchmark_result1) +
   ggtitle("Recommanderlab Benchmark (0.5 % subset)")
dataset_build(train_size2)
benchmark_result2 <- run_bench(list_methods2)
plot_result2 <- plot_bench(benchmark_result2) +
   ggtitle("Recommanderlab Benchmark (2 % subset)")

dataset_build(train_size3)
benchmark_result3 <- run_bench(list_methods2)
plot_result3 <- plot_bench(benchmark_result3) +
   ggtitle("Recommanderlab Benchmark (10 % subset)")


end.time <- Sys.time()  ### A SUPPRIMER
end.time - start.time   ### A SUPPRIMER
# Facultatif : affichage graphique
plot_result1
gc(verbose = FALSE)     # Free memory
save.image(file = "EKR-MovieLens.RData")
###############################################
#    FINE-TUNING SELECTED TRAINING METHODS    #
###############################################

# Fitting function (RMSE vs time) for each model and parameters
fitting <- function(model, config){
   start_time <- Sys.time()
   recommend <- Recommender(data = edx_rrm_train, method = model, param = config)  # Set recommendation parameters
   prediction <- predict(recommend, edx_rrm_test, type = "ratingMatrix")  # Run prediction
   accuracy <- calcPredictionAccuracy(edx_rrm_test,prediction) # Compute accuracy
   end_time <- Sys.time()
   running_time <- difftime(end_time, start_time, units = "secs") # Time difference, unit forced (so mins and secs aren't mixed...)
   rmse <- as.numeric(round(accuracy["RMSE"],4)) # Convert 10-stars RMSE to 5-stars RMSE
   c(rmse, running_time)
}

# Fitting k parameter      !!! A FINIR !!!
parameter <- "k = 10"
fitting("SVD", parameter)

plot_benchmark
##### SVD Method #####
SVD.K <- 10 # Défaut = 10 (meilleur RMSE >= 500)
SVD.M <- 100 # Défaut = 100 (pas d'effet???)
SVD.N <- "center" # center, Z-Score (pas d'effet???)
#recommend <- Recommender(data=edx_rrm, method="SVD", param=list(k=SVD.K, maxiter=SVD.M, normalize=SVD.N))

# Set multithreading
#n_threads <- detectCores()
#cluster <- makeCluster(n_threads)
#clusterExport(cluster, varlist=c("edx_rrm_test","edx_rrm_train"))
#clusterEvalQ(cluster, library(recommenderlab))
# Run multithreading
#benchmark_result <- as.data.frame(t(parSapply(cl = cluster, X = list_methods, FUN = benchmark))) # Parallel sapply
#stopCluster(cluster) # Stop multithread

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
#validation$rating <- validation$rating*2
#validation$rating <- as.integer(validation$rating)
#validation$movieId <- as.integer(validation$movieId)
#validation$userId <- as.integer(validation$userId)

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

