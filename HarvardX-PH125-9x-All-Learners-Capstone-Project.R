library(tidyverse)
library(caret)
options(timeout = 120, digits=5, max.print=400, width = 160)

# File preparation
dl <- "ml-10M100K.zip"
if(!file.exists(dl)) download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings_file <- "ml-10M100K/ratings.dat" 
if(!file.exists(ratings_file)) unzip(dl, ratings_file)
movies_file <- "ml-10M100K/movies.dat" 
if(!file.exists(movies_file)) unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), 
                                   fixed("::"), simplify = TRUE), 
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>% mutate(userId = as.integer(userId), movieId = as.integer(movieId),
                              rating = as.numeric(rating), timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), 
                                  fixed("::"), simplify = TRUE), 
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>% mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

final_holdout_test <- temp %>% semi_join(edx, by = "movieId") %>% semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
edx <- rbind(edx, anti_join(temp, final_holdout_test))

rm(dl, ratings, movies, test_index, temp, movies_file, ratings_file) 

# Preliminary assessment
# Number of training reviews
nrow(edx)

# Number of movies
n_distinct(edx$movieId)

# Number of users
n_distinct(edx$userId)

# The test set
head(final_holdout_test, n=5)

# A plot of the ratings distribution
ggplot(edx, aes(x = rating, y=..count../1000000)) + 
  geom_histogram() + xlab("rating") + ylab("millions of reviews")

# Average rating
mu <- mean(edx$rating)

# Naive root mean square error (nRMSE) 
nRMSE <- sqrt(mean((final_holdout_test$rating - mu)^2))

# Subtracting the mean score from all ratings to normalize the scores around the mean and hence
# work on minimizing the residual values only.
edx$rating = edx$rating - mu

# *************************************************************************************************

# Movie effect

# Average rating of each movie is computed in the table mov_avg, with its distribution plotted.
mov_avg <- edx %>% 
  group_by(movieId) %>% 
  summarize(mov_avg = mean(rating), count=n()) %>%
  arrange (movieId)

ggplot(mov_avg, aes(mov_avg)) + geom_density() +
  xlab("mean rating of movies") + ylab("relative frequency")

# The number of reviews for each movie varies extremely, with most movie reviews receiving only 
# several ratings while the most popular ones receiving tens of thousands.

ggplot(mov_avg, aes(count, ..count..)) + geom_histogram(binwidth=20) + 
  xlab("no. of  reviews") + ylab("count") + 
  xlim(0,max(mov_avg$count)) + ylim(0,700)

# Adding a penalty term in the mean movie ratings to reduce the deviation by rarely seen movies. 
# The weighted mean ratings with an added penalty coefficient lambda for a movie is given by 
# dividing the sum of ratings by the sum of lambda and the number of reviews for that movie:
#
# weighted average rating = (sum of ratings of the user) /  (lambda + no. of ratings of the user)
#
# Trying different values of lambda to find its optimal value by cross-validation. First divide 
# the training set into 10 equal subsets with i containing the corresponding indexes of each.

i <- createDataPartition(edx$rating, 10, 0.1, FALSE)

# With each subset k1 to k10, its counterpart records are used to determine the weighted mean ratings.
# Then the validation subset is joined to the counterpart records by movieId, and MSE in each subset
# is computed. The average MSE across all 10 subsets are in turn computed. Finally, we will feed
# lambda values between 0 and 4 into the function to see which one gives the lowest RSME.

avg_mov_MSE <- function (lambda) mean(
  sapply(1:10, function(k)
    edx[setdiff(1:nrow(edx),i[,k]),2:3] %>%
      group_by(movieId) %>%
      summarize(mov_avg = sum(rating)/(lambda + n())) %>%
      inner_join(edx[i[,k],2:3], "movieId")%>%
      group_by(movieId) %>% 
      summarize(SSE = sum((rating - mov_avg)^2), penalty= sum(mov_avg^2)/n() * lambda, count=n()) %>%
      summarize(MSE = sum(SSE + penalty)/sum(count)) %>% pull(MSE)))

lambda<-seq(0,4)

sapply(lambda, avg_mov_MSE)

# MSE is the least when lambda is zero. In other words, the original mean rating is the best estimate.
# With this, the RMSE in the test set is re-evaluated. Note mu is to be subtracted from the original 
# ratings in the test set first.

RMSE <- final_holdout_test %>% 
  select(movieId, rating) %>% 
  inner_join(mov_avg, "movieId") %>%
  mutate(SE = (rating - mu - mov_avg)^2) %>% 
  pull(SE) %>% mean() %>% sqrt()

# This RMSE is significantly better than the naive RMSE. 

# The movie averages are now subtracted from the edx ratings to further analyze the residual errors. 
edx <- edx %>% inner_join(mov_avg, "movieId") %>% mutate(rating = rating - mov_avg)

# *************************************************************************************************

# User effect

# An average for the user effect, user_avg, similar to that of movie average is also determined.

user_avg <- edx[,c(1,2,3)] %>% 
  group_by(userId) %>% 
  summarize(user_avg = mean(rating), count=n()) %>%
  arrange(userId)

# Again cross-validation is carried out to see if the RMSE can be better estimated by adding a penalty term.

i <- createDataPartition(edx$rating, 10, 0.1, FALSE)

user_avg_MSE <- function (lambda) mean(
  sapply(1:10, function (k)
    edx[setdiff(1:nrow(edx),i[,k]),1:3] %>%
      group_by(userId) %>% 
      summarize(user_avg = sum(rating)/(lambda + n())) %>%
      inner_join(edx[i[,k],1:3], "userId") %>%
      group_by(userId) %>%
      summarize(SSE = sum((rating - user_avg)^2), penalty = sum(user_avg^2)/n() * lambda, count=n()) %>% 
      summarize(MSE = sum(SSE + penalty)/sum(count)) %>% pull(MSE)))

lambda <- c(0:4)

sapply(lambda, user_avg_MSE)

# Again, the average MSE is the least when lambda is zero, so the original user average is still the best estimate. 
# With this the RMSE in the test set is re-evaluated. Note both mu and movie mean are to be subtracted from the 
# original ratings in the test set first.

RMSE <- final_holdout_test %>% 
  select(movieId, userId, rating) %>% 
  inner_join(mov_avg, "movieId") %>%
  inner_join(user_avg, "userId") %>%
  mutate(SE = (rating - mu - mov_avg - user_avg)^2) %>% 
  pull(SE) %>% mean() %>% sqrt()

# This brings another significant improvement, but even better estimates can be made by differentiating the genre
# preference of users instead of just estimating with one user average.

# *************************************************************************************************

# Genre effect

# As each movie may contain a few genres separated by the symbol |, the first action is to determine the maximum 
# number of genres a movie will have by counting the | symbols.

max(str_count(edx$genres, "[|]"))

# The next step is to separate the genres column into a maximum of 8 different columns and gather each under a 
# separate row. The empty columns and the genre with the entry "(no genres listed)" are removed. A total of 19 
# genres are present. 

data.frame(genres = edx[,6]) %>% 
  separate(genres, as.character(seq(1:8)),"[|]") %>% 
  gather(genre_no, genre, 1:8) %>% 
  filter(!is.na(genre) & genre !="(no genres listed)") %>% 
  group_by(genre) %>% summarize(count=n())


# A movie genre data frame mov_gen is created by separating the genres of each movie.
mov_gen <- edx[,c(2,6)] %>% 
  group_by(movieId) %>% slice(1) %>%
  separate(genres, paste("genre", seq(1:8), sep=""),"[|]") %>%
  gather(count, genre, genre1:genre8) %>% 
  filter(!is.na(genre) & genre !="(no genres listed)") %>%
  select(movieId, genre) %>% 
  arrange(movieId, genre)

# By adding 1 to each genre present for each movie, spreading out the genres and filling the NA spaces filled by 0, 
# a genre chart with each movie on a row is created.
mutate(mov_gen, review = 1) %>% spread(genre, review, 0)

# A user genre chart (user_gen) is created in a similar manner to mov_gen, but it computes the average rating of the
# reviews within a given genre for each user. 

user_gen <- edx[,c(1,2,3,6)] %>% 
  separate(genres, paste("genre", seq(1:8), sep=""),"[|]") %>%
  gather(count, genre, genre1:genre8) %>% 
  filter(!is.na(genre) & genre !="(no genres listed)")%>% 
  group_by(userId, genre) %>% 
  summarize(gen_avg = mean(rating)) %>%
  arrange(userId, genre)

# With this genre preference information, the RMSE in the test set is re-evaluated. If the user has no ratings for  
# any of the genres of the movie in question, the overall rating (user_avg) will be used instead. This improves the 
# RMSE to below 0.85000.

RMSE <- final_holdout_test %>% 
  inner_join(mov_avg, "movieId") %>%
  inner_join(user_avg, "userId") %>%
  inner_join(mov_gen, "movieId") %>%
  left_join(user_gen, c("userId", "genre")) %>% 
  group_by(movieId, userId, rating, mov_avg, user_avg) %>% 
  summarize(gen_avg = mean(gen_avg, na.rm = T)) %>% 
  mutate(SE = (rating - mu - mov_avg - ifelse(is.na(gen_avg),user_avg,gen_avg))^2) %>% 
  pull(SE) %>% mean() %>% sqrt()

# The user/genre averages are now subtracted from the training set ratings to further analyze the residual errors. 
# Note 7 reviews with the genre "(no genres listed)" are removed from the training set from here onward.

edx <- edx %>%
  select(movieId, userId, rating, timestamp) %>%
  inner_join(mov_gen, "movieId") %>%
  inner_join(user_gen, c("userId", "genre")) %>% 
  group_by(movieId, userId, rating, timestamp) %>% 
  summarize(gen_avg = mean(gen_avg, na.rm = T)) %>%
  mutate(rating = rating - gen_avg)

# *************************************************************************************************

# The average residual ratings for each movie and the no. of reviews are saved in data frame rsd_mov_avg. 
rsd_mov_avg <- edx %>% group_by(movieId) %>% 
  summarize(rsd_mov_avg=mean(rating), no_of_reviews=n()) 

# Correlation between the two variables is computed.
cor(rsd_mov_avg$no_of_reviews,rsd_mov_avg$rsd_mov_avg)

# A regression tree model is created to predict the residual ratings by partitioning the movies review count. 
# The complexity parameter is tuned through through cross-validation.

rgtr <- train(rsd_mov_avg~no_of_reviews, 
              data = rsd_mov_avg,
              method = "rpart",
              tuneGrid = data.frame(cp = seq(0, 0.01, 0.0001)),
              trControl = trainControl(method = "cv", p = 0.9))

# The resulting regression tree
plot(rgtr$finalModel, margin=0.1)
text(rgtr$finalModel, cex=1)

# The residual rating estimates as per range of reviews suggested by the model
filter(rgtr$finalModel$frame, var=="<leaf>") %>% 
  select(yval) %>% rename(rsd_estimate=yval) %>%
  cbind("reviews fr" = c(sort(rgtr$finalModel$splits[,4]+0.5, decreasing = T),1),
        "reviews to" = c(max(rsd_mov_avg$no_of_reviews), sort(rgtr$finalModel$splits[,4]-0.5, decreasing = T)))

# The residual rating estimates are plugged back into the rsd_mov_avg using the predict function. Then include  
# this this term for subtraction in the final RMSE evaluation.

rsd_mov_avg <- rsd_mov_avg %>% 
  select(movieId, rsd_mov_avg, no_of_reviews) %>% 
  cbind(rev_cnt_rsd = predict(rgtr))

RMSE <- final_holdout_test %>% 
  select(movieId, userId, rating) %>%
  inner_join(user_avg, "userId") %>%
  inner_join(mov_gen, "movieId") %>%
  left_join(user_gen, c("userId", "genre")) %>% 
  group_by(movieId, userId, rating, user_avg) %>% 
  summarize(gen_avg = mean(gen_avg, na.rm = T)) %>%
  inner_join(mov_avg, "movieId") %>%
  inner_join(rsd_mov_avg, "movieId") %>%
  mutate(SE = (rating - mu - mov_avg
               - ifelse(is.na(gen_avg),user_avg,gen_avg)
               - rev_cnt_rsd)^2) %>% 
  pull(SE) %>% mean() %>% sqrt()

# End of the algorithm 
