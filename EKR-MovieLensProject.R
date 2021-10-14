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

# Downloading and loading the packages used in this study
if(!require(dplyr)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")

library(dplyr)
library(lubridate)
library(stringr)
library(ggplot2)

# Basic data to introduce the global dataset (training + validation)
total_number_ratings <- nrow(union(edx,validation))
total_number_movies <- n_distinct(union(edx$movieId, validation$movieId))
total_number_users <-n_distinct(union(edx$userId, validation$userId))
column_names <- colnames(edx)

# Conversion of timestamp in comprehensible format


# Root-Mean-Squared Error calculation function
RMSE <- function(true_rating, predicted_rating){
   round(sqrt(mean((true_rating - predicted_rating)^2)),6)
}


######################################################################
# Multiple Linear Regression Model
######################################################################

# Naive prediction, that applies the mean rating (beta) to all predictions
mean_rating <- mean(edx$rating)
rmse_naive <- RMSE(validation$rating, mean_rating)

# Linear Model that takes into account the movie-effect
   # Calculating alpha_movie by computing the difference between the rating of each movie and the average rating
   movie_avg <- edx %>% 
         group_by(movieId) %>% 
         summarize(alpha_movie = mean(rating - mean_rating))
   # Prediction using movie-effect model
   predictedM <- mean_rating + validation %>% 
      left_join(movie_avg, by='movieId') %>%
      .$alpha_movie
   # RMSE of the movie-effect model
   rmse_movie <- RMSE(validation$rating, predictedM)

# Model that takes into account the movie/user-effect
   # Calculating alpha_user by computing the difference between the rating of each user and the average rating of each movie
   user_avg <- edx %>% 
      left_join(movie_avg, by='movieId') %>%
      group_by(userId) %>%
      summarize(alpha_user = mean(rating - mean_rating - alpha_movie))
   #Prediction using movie/user-effect model
   predictedMU <- validation %>% 
      left_join(movie_avg, by='movieId') %>%
      left_join(user_avg, by='userId') %>%
      mutate(prediction = mean_rating + alpha_movie + alpha_user) %>%
      .$prediction
   # RMSE of the movie/user-effect model
   rmse_movie_user <- RMSE(validation$rating, predictedMU)

# Model that takes into account the year effect
   # Converting rating timestamp to "y-m-d h:min:s" format on training set
   edx <- edx %>% 
      mutate(rating_time = as.POSIXct(timestamp, origin="1970-01-01")) %>%
      # Extracting year/month of the rating  
      mutate(rating_year = year(rating_time), rating_month = month(rating_time))
      # Extracting year of the movie : " (19YY)" or " (20YY)"
      year_pattern <- "\\s\\(\\d{4}\\)"
      edx <- edx %>% 
      mutate(movie_year = str_extract(title,year_pattern)) %>%
      # Writing correctly the year of the movie
      mutate(movie_year = as.numeric(str_extract(movie_year, "\\d{4}")))

      # Converting rating timestamp to "y-m-d h:min:s" format on validation set
      validation <- validation %>% 
         mutate(rating_time = as.POSIXct(timestamp, origin="1970-01-01")) %>%
         # Extracting year/month of the rating  
         mutate(rating_year = year(rating_time), rating_month = month(rating_time))
      # Extracting year of the movie : " (19YY)" or " (20YY)"
      validation <- validation %>% 
         mutate(movie_year = str_extract(title,year_pattern)) %>%
         # Writing correctly the year of the movie
         mutate(movie_year = as.numeric(str_extract(movie_year, "\\d{4}")))  
            
      # Calculating alpha_movie_year by computing the difference between the rating of year and the average rating of each movie
      year_avg <- edx %>% 
         left_join(movie_avg, by='movieId') %>%
         left_join(user_avg, by='userId') %>%
         group_by(movie_year) %>%
         summarize(alpha_movie_year = mean(rating - mean_rating - alpha_movie - alpha_user))
      #Prediction using movie/user-effect model
      predictedMUY <- validation %>% 
         left_join(movie_avg, by='movieId') %>%
         left_join(user_avg, by='userId') %>%
         left_join(year_avg, by='movie_year') %>%
         mutate(prediction = mean_rating + alpha_movie + alpha_user + alpha_movie_year) %>%
         .$prediction
      # RMSE of the movie/user/year-effect model
      rmse_movie_user_year <- RMSE(validation$rating, predictedMUY)

      # Calculating alpha_rating_month by computing the difference between the rating of year and the average rating of each movie
      month_avg <- edx %>% 
         left_join(movie_avg, by='movieId') %>%
         left_join(user_avg, by='userId') %>%
         group_by(rating_month) %>%
         summarize(alpha_rating_month = mean(rating - mean_rating - alpha_movie - alpha_user))
      #Prediction using movie/user/month-effect model
      predictedMUM <- validation %>% 
         left_join(movie_avg, by='movieId') %>%
         left_join(user_avg, by='userId') %>%
         left_join(month_avg, by='rating_month') %>%
         mutate(prediction = mean_rating + alpha_movie + alpha_user + alpha_rating_month) %>%
         .$prediction
      # RMSE of the movie/user/month-effect model
      rmse_movie_user_month <- RMSE(validation$rating, predictedMUM)

######################################################################
#     Non Linear Models
######################################################################

# Adding non-linear models
#edx <- edx %>% mutate (SquareRating = rating^2, 
#                       SqrtRating = sqrt(rating), 
#                       ExpRating = exp(rating), 
#                       LogRating = log(rating), 
#                       LogisticRating = 1/(1+exp(- rating)))
# Non-linear Model that takes into account the movie-effect
   # Calculating average
   mean_SQrating <- mean(edx$rating^2)
   # Calculating alpha_movie by computing the difference between the rating of each movie and the average rating
   movie_avgSQ <- edx %>% 
       group_by(movieId) %>% 
      summarize(alpha_movie = mean(mean_SQrating - mean_rating^2))
   # Prediction using movie-effect model
   predictedMSQ <- mean_SQrating + validation^2 %>% 
      left_join(movie_avgSQ, by='movieId') %>%
      .$alpha_movie
   predictedMSQ <- sqrt(predictedMSQ)
   # RMSE of the movie-effect models
   rmse_movieSQ <- RMSE(validation$rating, sqrt(predictedMSQ))
   #rmse_movieSQRT <- RMSE(validation$rating, predictedMSQRT)
   #rmse_movieEXP <- RMSE(validation$rating, predictedMEXP)
   #rmse_movieLOG <- RMSE(validation$rating, predictedMLOG)
   #rmse_movieLOGISTIC <- RMSE(validation$rating, predictedMLOGISTIC)


# Plotting year effect
   # Mean rating by movie year
#   edx <- edx %>% group_by(movie_year) %>% 
#      mutate(RmoyMY = mean(rating))        
   # Plot
#   plot_movie_year <- ggplot(edx, aes(y=RmoyMY, x=movie_year)) + 
#      geom_point() +
#      geom_smooth()
#   plot_movie_year

#Clear large data sets
rm(predictedM, predictedMU, predictedMUM, predictedMUY)      
# Save ALL information, for the Rmd report
save.image (file = "EKR-MovieLens.RData")

