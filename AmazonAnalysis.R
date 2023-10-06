##
## Amazon Employee Access
##

library(tidymodels)
library(vroom)

amazon_train <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/AmazonEmployeeAccess/train.csv")
amazon_test <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/AmazonEmployeeAccess/test.csv")
