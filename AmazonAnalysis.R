##
## Amazon Employee Access
##

library(tidyverse)
library(tidymodels)
library(vroom)
library(ggmosaic)

amazon_train <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/AmazonEmployeeAccess/train.csv")
amazon_test <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/AmazonEmployeeAccess/test.csv")

vars_to_convert <- c("RESOURCE", "MGR_ID", "ROLE_ROLLUP_1",
                     "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_TITLE",
                     "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE")

amazon_train[vars_to_convert] <- lapply(amazon_train[vars_to_convert], as.factor)
amazon_test[vars_to_convert] <- lapply(amazon_test[vars_to_convert], as.factor)

##### EDA #####
ggplot(data=amazon_train) + 
  geom_boxplot(aes(x=RESOURCE, y=ACTION))



##### Recipe making #####

my_recipe <- recipe(ACTION~., data=amazon_train) %>%
              step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
              step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value
              step_dummy(all_nominal_predictors()) # %>% # dummy variable encoding
              # step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding
              # also step_lencode_glm() and step_lencode_bayes()


prepped_recipe <- prep(my_recipe)
baked_recipe <- bake(prepped_recipe, amazon_train)
