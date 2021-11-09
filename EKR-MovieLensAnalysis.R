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
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")

# Load required packages/libraries
library(reshape2)       # For acast function
library(ggplot2)        # For pretty graphics
library(ggrepel)        # For repelled labels on graphics
library(recommenderlab) # For data analysis
library(parallel)       # For parSapply function (multi-thread sapply)
library(stringr)        # For parameters text (tuning stage)

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

##### A SUPPRIMER####
save(edx_rrm, file = "edxRRM.RData")
load(file = "edxRRM.RData")


###########################################
#    BENCHMARKING THE TRAINING METHODS    #
###########################################

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
   result <- cbind(model_list,result)  # Add model column
   colnames(result) <- c("model", "RMSE", "time")  # Add column names
   result$RMSE <- as.numeric(result$RMSE) # Convert factors to numeric values
   result$time <- as.numeric(result$time)
   result
}

# Define plot function
plot_bench <- function(benchresult){
   benchresult %>%
      ggplot(aes(x = time, y = RMSE, label = model)) +
      xlab("Time (s)") +
      ylab("Error (RMSE)") +
      geom_point() +
      scale_x_continuous(limits = c(0,NA)) +
      geom_text_repel() +
      geom_hline(yintercept = 0.9, linetype = "dashed", color = "darkred", alpha = 0.4) + # Minimal objective
      geom_hline(yintercept = 0.865, linetype = "dashed", color = "darkgreen", alpha = 0.4)  # Optimal objective 
}

# Define tested models (no IBCF)
#list_methods_1 <- c("RANDOM", "POPULAR", "LIBMF", "SVD", "SVDF", "ALS", "ALS_implicit", "UBCF")
#list_methods_2 <- c("POPULAR", "LIBMF", "SVD", "UBCF")
#list_methods_3 <- c("POPULAR","LIBMF", "SVD")

# LISTES LIGHT, A SUPPRIMER
list_methods_1 <- c("RANDOM", "POPULAR","LIBMF", "SVD", "UBCF", "ALS")
list_methods_2 <- c("RANDOM", "POPULAR", "LIBMF", "SVD")
list_methods_3 <- c("POPULAR","LIBMF")
###############################

# Define the method lists and corresponding dataset sizes (for 9 runs)
methods_sizes <- data.frame(
   method = c(1, rep(2,4), rep(3,4)), # Method list numbers
   size = c(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 1)   # Training set sizes
)

# Build the 9 datasets, run the corresponding  benchmarks
for (n in 1:9){
   size <- methods_sizes$size[n] # Select the size given in the 'n' line
   dataset_build(methods_sizes$size[n]) # Build the dataset of the selected size
   method_name <- paste0("list_methods",methods_sizes$method[n]) # Concatenate "list_methods" and the number of the method in the [n] line
   method <- get(method_name) # Get the actual list of methods in the method_list'n'
   benchname <- paste0("benchmark_result", n)   # Concatenate benchmark_result with the n
   assign(benchname, run_bench(method))  # Report the results in the benchmark_result'n' dataframe
   assign(benchname, cbind(get(benchname), size)) # Add a size column with the selected size
}

# Create 3 time/RMSE plots
plot_time_rmse1 <- plot_bench(benchmark_result1) +
   ggtitle("Recommanderlab Benchmark (0.5 % subset)")
plot_time_rmse2 <- plot_bench(benchmark_result3) +
   ggtitle("Recommanderlab Benchmark (2 % subset)")
plot_time_rmse3 <- plot_bench(benchmark_result5) +
   ggtitle("Recommanderlab Benchmark (10 % subset)")

# Build the size vs time/rmse base for our best models
time_result <- rbind(benchmark_result1, benchmark_result2, benchmark_result3, benchmark_result4, benchmark_result5) %>%
   filter(model %in% c("SVD", "POPULAR", "LIBMF", "UBCF")) %>%
   arrange(.,model)

plot_time_size1 <- time_result %>%
   ggplot(aes(x = size, y = time, color = model)) +
   ggtitle("Computing time of the 4 best models") +
   xlab("Dataset size") +
   scale_x_continuous(labels = scales::percent) +
   ylab("Time (s)") +
   geom_point() +
   geom_line() +
   theme_bw()

plot_time_size2 <- time_result %>%
   ggplot(aes(x = size, y = time, color = model)) +
   ggtitle("Computing time of the 4 best models") +
   xlab("Dataset size") +
   scale_x_continuous(labels = scales::percent) +
   ylab("Time (s)") +
   geom_point() +
   geom_line() +
   scale_y_sqrt() +
   theme_bw()

plot_time_size3 <- time_result %>%
   filter(model != "UBCF") %>%
   ggplot(aes(x = size, y = time, color = model)) +
   ggtitle("Computing time of the 3 best models") +
   xlab("Dataset size") +
   scale_x_continuous(labels = scales::percent) +
   ylab("Time (s)") +
   geom_point() +
   geom_line() +
   theme_bw()

gc(verbose = FALSE)     # Free memory

rmse_result <- rbind(time_result, benchmark_result6, benchmark_result7) %>%
   filter(model %in% c("SVD", "POPULAR", "LIBMF")) %>%
   arrange(.,model)

plot_rmse_size <- rmse_result %>%
   ggplot(aes(x = size, y = RMSE, color = model)) +
   ggtitle("Stability of the 3 best models") +
   xlab("Dataset size") +
   ylab("Error (RMSE)") +
   scale_x_continuous(labels = scales::percent) +
   geom_point() +
   geom_line() +
   geom_hline(yintercept = 0.9, linetype = "dotted", color = "darkred", alpha = 0.5) + # Minimal objective
   geom_hline(yintercept = 0.865, linetype = "dotted", color = "darkgreen", alpha = 0.5) +  # Optimal objective 
   geom_vline(xintercept = 0.2, linetype = "dashed", color = "royalblue4", alpha = 0.7) +  # Optimal dataset size
   theme_bw()

### Affichage graphiques (FACULTATIF) ###
plot_time_size2
plot_rmse_size

###############################################
#    FINE-TUNING SELECTED TRAINING METHODS    #
###############################################
gc(verbose = FALSE)     # Free memory

# Build the training dataset with the most representative size (20%)
dataset_build(0.2)
save.image(file = "EKR-MovieLens.RData")
load(file = "EKR-MovieLens.RData")

gc(verbose = FALSE)     # Free memory

# Fitting function (RMSE vs time) for each model and parameters
#fitting <- function(model, config){
#   start_time <- Sys.time() # [OK]
#   recommend <- Recommender(data = edx_rrm_train, method = model, param = config)  # Set recommendation parameters
#   prediction <- predict(recommend, edx_rrm_test, type = "ratingMatrix")  # [OK] Run prediction
#   accuracy <- calcPredictionAccuracy(edx_rrm_test,prediction) # [OK] Compute accuracy
#   end_time <- Sys.time()  # [OK]
#   running_time <- difftime(end_time, start_time, units = "secs") # [OK] Time difference, unit forced (so mins and secs aren't mixed...)
#   rmse <- as.numeric(round(accuracy["RMSE"],4)) # [OK] Compute RMSE with 4 digits
#   c(rmse, running_time) # [OK]
#}

##### POPULAR Method #####
values <- c("center","Z-Score")
parameter <- "normalize = "
tuning <- str_c(parameter, values)

# RMSE and time measuring function for POPULAR model
fit_pop <- function(config){
   start_time <- Sys.time()   # Start chronometer
   parameters <- str_c("list(", config, ")") # Convert parameters in appropriate form for "param = list(parameter=value)"
   recommend <- Recommender(data = edx_rrm_train, method = "POPULAR", param = parameters)  # Set recommendation parameters
   prediction <- predict(recommend, edx_rrm_test, type = "ratingMatrix")  # Run prediction
   accuracy <- calcPredictionAccuracy(edx_rrm_test,prediction) # Compute accuracy
   end_time <- Sys.time()     # Stop chronometer
   running_time <- difftime(end_time, start_time, units = "secs") # Time difference, unit forced (so mins and secs aren't mixed...)
   running_time <- round(running_time,2)  # Rounding to 2 decimals
   rmse <- round(accuracy["RMSE"],4) # Compute RMSE with 4 digits
   c(rmse, running_time)
}

# Fitting function
run_fit_pop <- function(parameter){
   result <- as.data.frame(t(sapply(X = parameter, FUN = fit_pop)))
   parameter <- str_remove (parameter, "[a-z]+ = ")
   result <- cbind(parameter,result)  # Add parameter column
   colnames(result) <- c("Value","RMSE", "Time")  # Add column names
   rownames(result) <- NULL # Remove row names
   result
}

fit_pop_result <- run_fit_pop(tuning)
fit_pop_result
save.image(file = "EKR-MovieLens.RData")

##### LIBMF Method #####
ramp <- c(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20)
libmf_d <- str_c("dim = ", 10*ramp) # default = 10
libmf_p <- str_c("costp_l2 = ", 0.01*ramp) # Regularization parameter for user factor (default = 0.01)
libmf_q <- str_c("costq_l2 = ", 0.01*ramp) # Regularization parameter for item factor (default = 0.01)
libmf_t <- str_c("nthread = ", c(1, 2, 4, 8, 16, 32))   # Number of threads (default = 1)
#recommend <- Recommender(data=edx_rrm,method="LIBMF", param=list(dim=LIBMF.D,costp_l2=LIBMF.P, costq_l2=LIBMF.Q,  nthread=LIBMF.T))


##### SVD Method #####
svd_k <- str_c("k = ", 10*ramp) # Rank of the SVD approximation ? (default = 10)
svd_m <- str_c("maxiter = ", 100*ramp) # Maximum number of iterations (default = 100)
svd_n <- str_c("normalize =", c("center", "Z-Score")) # Normalization method (default = center)
#recommend <- Recommender(data=edx_rrm, method="SVD", param=list(k=SVD.K, maxiter=SVD.M, normalize=SVD.N))

start.time <- Sys.time()  ### A SUPPRIMER
end.time <- Sys.time()  ### A SUPPRIMER
end.time - start.time   ### A SUPPRIMER

##### Méthode Alternative ####

# Setting parameters and value for popular method
model <- "POPULAR"
pop <- data.frame(parameter = "normalize", value = c("center", "Z-score")) # Normalization parameter (default = center)
popular_settings <- data.frame(model, pop)  # Get all POPULAR settings (parameter, values) together

# Setting parameters and value for LIBMF method
ramp <- c(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20)
model <- "LIBMF"
libmf_d <- data.frame(parameter = "dim", value = c(10*ramp)) # default = 10
libmf_p <- data.frame(parameter = "costp_l2", value = c(0.01*ramp)) # Regularization parameter for user factor (default = 0.01)
libmf_q <- data.frame(parameter = "costq_l2", value = c(0.01*ramp)) # Regularization parameter for item factor (default = 0.01)
libmf_t <- data.frame(parameter = "nthread", value = c(1, 2, 4, 8, 16, 32))   # Number of threads (default = 1)
limbf_settings <- data.frame(model, rbind(libmf_d, libmf_p, libmf_q, libmf_t)) # Get all LIBMF settings (parameters, values) together

# Setting parameters and value for SVD method
model <- "SVD"
svd_k <- data.frame(parameter = "k", value = c(10*ramp)) # Rank of the SVD approximation ? (default = 10)
svd_m <- data.frame(parameter = "maxiter", value = c(100*ramp)) # Maximum number of iterations (default = 100)
svd_n <- data.frame(parameter = "normalize", value = c("center", "Z-Score")) # Normalization method (default = center)
svd_settings <- data.frame(model, rbind(svd_k, svd_m, svd_n)) # Get all SVD settings (parameters, values) together

model_settings <- rbind(popular_settings, limbf_settings, svd_settings) # Get all settings together


# Set multithreading
#n_threads <- detectCores()
#cluster <- makeCluster(n_threads)
#clusterExport(cluster, varlist=c("edx_rrm_test","edx_rrm_train"))
#clusterEvalQ(cluster, library(recommenderlab))
# Run multithreading
#benchmark_result <- as.data.frame(t(parSapply(cl = cluster, X = list_methods, FUN = benchmark))) # Parallel sapply
#stopCluster(cluster) # Stop multithread

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

predictions <- predict(recommend, validation_rrm, type = "ratingMatrix")
accuracy <- calcPredictionAccuracy(validation_rrm, predictions)
gc(verbose = FALSE)

rm(predictions)  # Free memory
gc(verbose = FALSE)
accuracy["RMSE"]/2

save.image(file = "EKR-MovieLens.RData")
