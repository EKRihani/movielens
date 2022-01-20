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
if(!require(ggrepel)) install.packages("ggrepel", repos = "http://cran.us.r-project.org")
if(!require(devtools)) install.packages("devtools", repos = "http://cran.us.r-project.org") #
if(!require(arules)) install_version("arules", version = "1.6-8", repos = "http://cran.us.r-project.org")  # Install arules package for R 3.6
#if(!require(arules)) install.packages("arules", repos = "http://cran.us.r-project.org")  # Install arules package for R >= 4.0
if(!require(recommenderlab)) install.packages("recommenderlab", repos = "http://cran.us.r-project.org")

# Load required packages/libraries
library(reshape2)       # For acast function
library(ggrepel)        # For repelled labels on graphics
library(devtools)       # For legacy atools package (v. 1.6-8) if using R 3.6
library(recommenderlab) # For data analysis

# Raise R memory limit size (Windows-only), or won't be able to allocate vector of size 5+Gb during our Matrix/realRatingMatrix conversion...
memory.limit(size = 50000)

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

# Save the validation dataset to an external file (will be used later for our validation set final preparation)
save(edx, validation, file = "edxval.RData")
rm(validation)     # Won't be needed until the final RMSE computation
gc(verbose = FALSE)   # Free as much memory as possible

# Prepare training dataset for recommenderlab
edx_rrm <- acast(edx, userId ~ movieId, value.var = "rating")   # Convert data to matrix
edx_rrm <- as(edx_rrm, "realRatingMatrix")     # Convert matrix to realRatingMatrix
rm(edx)     # Free memory
gc(verbose = FALSE)     # Free memory

##### A SUPPRIMER####
save(edx_rrm, file = "edxRRM.RData")
load(file = "edxRRM.RData")
#########################

###########################################
#    BENCHMARKING THE TRAINING METHODS    #
###########################################

set.seed(1234, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1234)`

# Create empty objects for training and test sets
edx_rrm_train = NULL
edx_rrm_test = NULL

# Define function : dataset size-reducing and splitting (train/evaluation)
dataset_build <- function(train_size){
  reduction_index <- sample(x = seq(1, nrow(edx_rrm)), size = nrow(edx_rrm) * train_size, replace = FALSE) # Build a train_size sized index (training + test)
  edx_rrm_small <- edx_rrm[reduction_index]
  test_index <- sample(x = seq(1, nrow(edx_rrm_small)), size = nrow(edx_rrm_small)*0.1, replace = FALSE) # Build a 10% sized test index
  edx_rrm_train <<- edx_rrm_small[-test_index]  # Split into a 90%-sized training set
  edx_rrm_test <<- edx_rrm_small[test_index]    # Split into a 10%-sized testing set
}

# Define function : benchmark (time and RMSE)
bench <- function(model){
  start_time <- Sys.time()     # Start chronometer
  recommend <- Recommender(data = edx_rrm_train, method = model)   # Set recommendation parameters
  prediction <- predict(recommend, edx_rrm_test, type = "ratingMatrix")   # Run prediction
  accuracy <- calcPredictionAccuracy(edx_rrm_test,prediction)   # Compute accuracy
  end_time <- Sys.time()     # Stop chronometer
  time <- round(difftime(end_time, start_time, units = "secs"),2)   # Time difference, unit forced (or will mix mins and secs)
  rmse <- as.numeric(round(accuracy["RMSE"],4))   # Compute RMSE with 4 digits
  c(rmse, time)     # Reports RMSE and running time
}

# Define function : report results
run_bench <- function(model_list){
  result <- as.data.frame(t(sapply(X = model_list, FUN = bench)))
  result <- cbind(model_list,result)   # Add model column
  colnames(result) <- c("model", "RMSE", "time")   # Add column names
  result$RMSE <- as.numeric(result$RMSE)   # Convert factors to numeric values
  result$time <- as.numeric(result$time)
  result
}

# Define function : plot (time v. RMSE) 
plotting_time_rmse <- function(benchresult){
  benchresult %>%
    ggplot(aes(x = time, y = RMSE, label = model)) +
      xlab("Time (s)") +
      ylab("Error (RMSE)") +
      scale_x_continuous(limits = c(0,NA)) +
      geom_hline(yintercept = 0.9, linetype = "dashed", color = "darkred", alpha = 0.4) +   # Minimal objective
      geom_hline(yintercept = 0.865, linetype = "dashed", color = "darkgreen", alpha = 0.4)   # Optimal objective 
}

# Create list of tested models
list_methods_1 <- c("RANDOM", "POPULAR", "LIBMF", "SVD", "SVDF", "ALS", "ALS_implicit", "UBCF") # No IBCF
# list_methods_1 <- c("IBCF", "RANDOM", "POPULAR", "LIBMF", "SVD", "SVDF", "ALS", "ALS_implicit", "UBCF") # With IBCF, for VERY powerful computers
list_methods_2 <- c("POPULAR", "LIBMF", "SVD", "UBCF")
list_methods_3 <- c("POPULAR","LIBMF", "SVD")

# Lighter lists, for slower computers
#list_methods_1 <- c("RANDOM", "POPULAR","LIBMF", "SVD", "UBCF")
#list_methods_2 <- c("POPULAR", "LIBMF","SVD")
#list_methods_3 <- c("POPULAR","LIBMF")

# Define the dataset sizes and corresponding method lists (for 9 runs)
methods_sizes <- data.frame(
  method = c(1, rep(2,4), rep(3,4)),     # Method list numbers, to be concatenated with "list_methods_"
  size = c(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 1)   # List of training set sizes
)

# Build the 9 datasets, run the corresponding benchmarks
l <- nrow(methods_sizes)
for (n in 1:l){
  size <- methods_sizes$size[n]     # Select the size given in the 'n' line
  dataset_build(methods_sizes$size[n])   # Build the dataset of the selected size
  method_name <- paste0("list_methods_",methods_sizes$method[n])   # Concatenate "list_methods" and the number of the method in the [n] line
  method <- get(method_name)     # Get the actual list of methods in the method_list'n'
  benchname <- paste0("benchmark_result", n)   # Concatenate benchmark_result with the n
  assign(benchname, run_bench(method))   # Report the results in the benchmark_result'n' dataframe
  assign(benchname, cbind(get(benchname), size))   # Add a size column with the selected size
}

# Create 3 RMSE vs time plots
plot_time_rmse1 <- plotting_time_rmse(benchmark_result1) +
  geom_point(data = benchmark_result1 %>% filter(model != "IBCF")) + # Don't display IBCF (very high time)
  geom_text_repel(data = benchmark_result1 %>% filter(model != "IBCF")) +
  ggtitle("Benchmark (0.5 % subset)")
plot_time_rmse2 <- plotting_time_rmse(benchmark_result3) +
  geom_point() +
  geom_text_repel() +
  ggtitle("Benchmark (2 % subset)")
plot_time_rmse3 <- plotting_time_rmse(benchmark_result5) +
  geom_point() +
  geom_text_repel() +
  ggtitle("Benchmark (10 % subset)")

# Build the size vs time/rmse base for our best models
time_result <- rbind(benchmark_result1, benchmark_result2, benchmark_result3, benchmark_result4, benchmark_result5) %>%
  filter(model %in% c("SVD", "POPULAR", "LIBMF", "UBCF")) %>%
  arrange(.,model)

# Draw time vs size plots
plot_time_size1 <- time_result %>%
  ggplot(aes(x = size, y = time, color = model)) +
  #ggtitle("Computing time of the 4 best models") +
  xlab("Dataset size") +
  scale_x_continuous(labels = scales::percent) +
  ylab("Time (s)") +
  geom_point() +
  geom_line() +
  theme_bw()

plot_time_size2 <- plot_time_size1 +
  scale_y_continuous(trans = "sqrt")   # Show quadratic behavior

plot_time_size3 <- time_result %>%
  filter(model != "UBCF") %>%
  ggplot(aes(x = size, y = time, color = model)) +
  #ggtitle("Computing time of the 3 best models") +
  xlab("Dataset size") +
  scale_x_continuous(labels = scales::percent) +
  ylab("Time (s)") +
  geom_point() +
  geom_line() +
  theme_bw()

# Model time vs size behavior

tvs_popular <- time_result %>% filter(model == "POPULAR") %>% lm(formula = time ~ size)  # Compute linear model
tvs_popular_pred <- predict.lm(tvs_popular, newdata = data.frame(1))  # Predict time for size = 1 (full sized dataset)
rsq_popular <- summary(tvs_popular)[["r.squared"]]  # Extract R squared from summary

tvs_libmf <- time_result %>% filter(model == "LIBMF") %>% lm(formula = time ~ size)
tvs_libmf_pred <- predict.lm(tvs_libmf, newdata = data.frame(1))
rsq_libmf <- summary(tvs_libmf)[["r.squared"]]

tvs_svd <- time_result %>% filter(model == "SVD") %>% lm(formula = time ~ size)
tvs_svd_pred <- predict.lm(tvs_svd, newdata = data.frame(1))
rsq_svd <- summary(tvs_svd)[["r.squared"]]

time_result_sq <- time_result %>% mutate(sqrt_time = sqrt(time)) # Compute sqrt(time) for UBCF quadratic model
tvs_ubcf <- time_result_sq %>% filter(model == "UBCF") %>% lm(formula = sqrt_time ~ size)
tvs_ubcf_pred <- predict.lm(tvs_ubcf, newdata = data.frame(1))^2  # Prediction is squared (quadratic model)
rsq_ubcf <- summary(tvs_ubcf)[["r.squared"]]

tsv_models <- tibble(
  Method = c("Popular", "LIBMF", "SVD", "UBCF"),
  Model = c("Linear", "Linear", "Linear", "Quadratic"),
  Intercept = c(tvs_popular[["coefficients"]][1], tvs_libmf[["coefficients"]][1], tvs_svd[["coefficients"]][1], tvs_ubcf[["coefficients"]][1]),
  Slope = c(tvs_popular[["coefficients"]][2], tvs_libmf[["coefficients"]][2], tvs_svd[["coefficients"]][2], tvs_ubcf[["coefficients"]][2]),
  "Pred. Time (s)" = c(tvs_popular_pred, tvs_libmf_pred, tvs_svd_pred, tvs_ubcf_pred),
  "R²" = c(rsq_popular, rsq_libmf, rsq_svd, rsq_ubcf)
)

gc(verbose = FALSE)   # Free memory

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
  geom_hline(yintercept = 0.9, linetype = "dotted", color = "darkred", alpha = 0.5) +  # Minimal objective
  geom_hline(yintercept = 0.865, linetype = "dotted", color = "darkgreen", alpha = 0.5) +  # Optimal objective 
  geom_vline(xintercept = 0.2, linetype = "dashed", color = "royalblue4", alpha = 0.7) +  # Optimal dataset size
  theme_bw()

###############################################
#    FINE-TUNING SELECTED TRAINING METHODS    #
###############################################

# Build the training dataset with the most representative size (20%)
dataset_build(0.2)
save.image(file = "EKR-MovieLens.RData")
load(file = "EKR-MovieLens.RData")

gc(verbose = FALSE)     # Free memory

# Set parameters and values for popular method
model <- "POPULAR"
pop <- data.frame(parameter = "normalize", value = c("'center'", "'Z-score'"))   # Normalization parameter (default = center)
popular_settings <- data.frame(model, pop)     # Get all POPULAR settings together

# Set parameters and values for LIBMF method
ramp <- c(0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50)
model <- "LIBMF"
libmf_d <- data.frame(parameter = "dim", value = c(20*ramp))   # Number of latent features (default = 10)
libmf_p <- data.frame(parameter = "costp_l2", value = as.character(c(0.01*ramp)))   # Regularization parameter for user factor (default = 0.01)
libmf_q <- data.frame(parameter = "costq_l2", value = as.character(c(0.01*ramp)))   # Regularization parameter for item factor (default = 0.01)
libmf_t <- data.frame(parameter = "nthread", value = as.character(c(1, 2, 4, 8, 16)))  # Number of threads (default = 1)
libmf_settings <- data.frame(model, rbind(libmf_d, libmf_p, libmf_q, libmf_t))   # Get all LIBMF settings together

# Set parameters and values for SVD method
smallramp <- c(0.2, 0.5, 1, 2, 5, 10)
model <- "SVD"
svd_k <- data.frame(parameter = "k", value = c(10*smallramp))     # Rank of the SVD approximation ? (default = 10)
svd_m <- data.frame(parameter = "maxiter", value = c(100*smallramp))   # Maximum number of iterations (default = 100)
svd_n <- data.frame(parameter = "normalize", value = c("'center'", "'Z-Score'"))   # Normalization method (default = center)
svd_settings <- data.frame(model, rbind(svd_k, svd_m, svd_n))   # Get all SVD settings together

model_settings <- rbind(popular_settings, libmf_settings, svd_settings)   # Get all models settings together (popular + LIBMF + SVD)

# Run benchmark for all models and all parameters
l <- nrow(model_settings)
#l <- which(model_settings$model == "SVD") - 1   # For slow computers : skips the SVD fitting (long, memory-heavy)
results_fitting <- NULL
for (n in 1:l){
  start_time <- Sys.time()     # Start chronometer
  testparam <- str_c("list(", model_settings$parameter[n], " = ", model_settings$value[n], ", verbose = TRUE)")   # Convert parameters in appropriate form for "param = list(parameter=value)"
  testparam <- eval(parse(text=testparam))     # Evaluate the result of the character string
  recommend <- Recommender(data = edx_rrm_train, method = model_settings$model[n], param = testparam)   # Set recommendation parameters
  prediction <- predict(recommend, edx_rrm_test, type = "ratingMatrix")   # Run prediction
  accuracy <- calcPredictionAccuracy(edx_rrm_test,prediction)     # Compute accuracy
  end_time <- Sys.time()     # Stop chronometer
  time <- difftime(end_time, start_time, units = "secs")   # Time difference, unit forced (so mins and secs aren't mixed...)
  time <- round(time,2)   # Rounding to 2 decimals
  rmse <- as.numeric(round(accuracy["RMSE"],4))   # Compute RMSE with 4 digits
  result <- data.frame(rmse, time)   # Combine RMSE and computing time
  results_fitting <- rbind(results_fitting, cbind(model_settings[n,],result))   # Put the new results below the old ones
}
results_fitting

# Plot fitting results
plot_criteria <- results_fitting %>% select(model, parameter) %>% unique() %>% filter(parameter != "normalize")  # Get all models/parameters, except normalize (no plot : only 2 values)
l <- nrow(plot_criteria)

for (n in 1:l){
  plot_title <- paste("Fitting :", plot_criteria$model[n], "model,", plot_criteria$parameter[n], "parameter")
  plot <- results_fitting %>%
    filter(model == plot_criteria$model[n], parameter == plot_criteria$parameter[n]) %>%
    ggplot(aes(x = time, y = rmse, label = value)) +
    ggtitle(plot_title) +
    ylab("Error (RMSE)") +
    xlab("Time (s)") +
    scale_x_continuous() +     # Manually sets scale for difftime objects
    geom_point() +
    geom_text_repel()
  plotname <- paste0("plot_fitting", n)   # Concatenate plot_fitting with the n
  assign(plotname, plot)     # Assign the plot to the plot_fitting'n' name
}

# Time/RMSE LIBMF optimization plot
plot_fitting1b <- results_fitting %>%
    filter(model == "LIBMF", parameter == "dim") %>%
    ggplot(aes(x = as.numeric(paste(value)), y = as.numeric(time)*rmse^2)) +
    ggtitle("Fitting : Time.RMSE² optimization") +
    ylab("Time.RMSE²") +
    xlab("dim factor") +
    scale_x_continuous(trans="log10") +     # Manually sets scale
    scale_y_continuous() +     # Manually sets scale 
    geom_point()

# Build report tables for the normalize parameters
table_pop_normalize <- results_fitting %>% filter(model == "POPULAR") %>% select(value, rmse, time)
table_svd_normalize <- results_fitting %>% filter(model == "SVD", parameter == "normalize") %>% select(value, rmse, time)

# Extract/compute some interesting fine-tuning values that are used in the report
SVD_optimal_k <-results_fitting %>%   # Find all k parameters that give an optimal mark (RMSE < 0.865), on SVD model
  filter(model == "SVD" & parameter == "k" & rmse <= 0.865)
SVD_maxiter_timespan <- results_fitting %>%   # Calculate the computing time span for SVD model and maxiter parameter
  filter(model == "SVD" & parameter == "maxiter") %>%
  summarize(span = round(max(time) - min(time),1))
LIBMF_best_dim <- results_fitting %>%   # Find the best (lowest) RMSE for LIBMF model and dim parameter
  filter(model == "LIBMF" & parameter == "dim") %>%
  filter(rmse == min(rmse))
LIBMF_best_dim_composite <- results_fitting %>%   # Find the lowest time*rmse² value for LIBMF model and dim parameter
  filter(model == "LIBMF" & parameter == "dim") %>%
  mutate(comp = time * rmse^2) %>%
  filter(comp == min(comp))
LIBMF_costp_CI <- results_fitting %>%   # Calculate the computing time 95% confidence interval (total width) for LIBMF model and costP parameter
  filter(model == "LIBMF" & parameter == "costp_l2") %>%
  summarize(CI = 2*sd(time)/sqrt(length(time))*qnorm(.975))
LIBMF_costq_CI <- results_fitting %>%   # Calculate the computing time 95% confidence interval (total width) for LIBMF model and costQ parameter
  filter(model == "LIBMF" & parameter == "costq_l2") %>%
  summarize(CI = 2*sd(time)/sqrt(length(time))*qnorm(.975))
LIBMF_nthread_rmse_span <- results_fitting %>%   # Calculate the total RMSE spread for LIBMF model and nthread parameter
  filter(model == "LIBMF" & parameter == "nthread") %>%
  summarize(max(rmse) - min(rmse))

save.image(file = "EKR-MovieLens.RData")

#####################################################
#    CALCULATING RMSE AGAINST THE VALIDATION SET    #
#####################################################

# Load edx and validation datasets
load("edxval.RData")

# Remove movies that aren't used in the validation set (adding lines to the validation set is forbidden)
edx <- semi_join(edx, validation, by = "movieId")

# Convert validation set to matrix, then realRatingMatrix (class used by recommenderlab)
gc(verbose = FALSE)
validation_rrm <- acast(validation, userId ~ movieId, value.var = "rating")
validation_rrm <- as(validation_rrm, "realRatingMatrix")
gc(verbose = FALSE)

# Convert edx set to matrix, then realRatingMatrix (class used by recommenderlab)
gc(verbose = FALSE)
edx_rrm <- acast(edx, userId ~ movieId, value.var = "rating")
edx_rrm <- as(edx_rrm, "realRatingMatrix")
gc(verbose = FALSE)

rm(edx, validation)     # edx/validation won't be used anymore ; keep only edx_rrm/validation_rrm

# Run the final validation benchmark
start_time <- Sys.time()     # Start chronometer
recommend <- Recommender(data = edx_rrm, method = "LIBMF", param = list(dim = 400, costp_l2 = 0.01, costq_l2 = 0.01, nthread = 16))  # Set recommendation parameters
prediction <- predict(recommend, validation_rrm, type = "ratingMatrix")   # Run prediction
final_accuracy <- calcPredictionAccuracy(validation_rrm, prediction)   # Compute accuracy
end_time <- Sys.time()     # Stop chronometer
final_time <- difftime(end_time, start_time, units = "secs")   # Time difference, unit forced (so mins and secs aren't mixed...)
final_time <- round(final_time,2)   # Rounding to 2 decimals
final_rmse <- as.numeric(round(final_accuracy["RMSE"],4))
final_time
final_accuracy

rm(recommend, prediction,edx_rrm, validation_rrm) # Clean memory
gc(verbose = FALSE)
save.image(file = "EKR-MovieLens.RData")
