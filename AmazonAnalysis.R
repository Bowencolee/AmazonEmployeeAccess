##
## Amazon Employee Access
##

library(tidymodels)
library(vroom)
library(embed) # target encoding

library(glmnet) # penalized log regression
library(ranger) # random forests
library(discrim) # naive bayes
library(kknn) # KNN
library(kernlab) # SVM

# setwd("C:/Users/bowen/Desktop/Stat348/AmazonEmployeeAccess")
amazon_train <- vroom::vroom("train.csv") %>%
  mutate(ACTION = as.factor(ACTION)) # need to fix ^^ for Becker/BATCH
amazon_test <- vroom::vroom("test.csv")


##### EDA #####
# vars_to_convert <- c("RESOURCE", "MGR_ID", "ROLE_ROLLUP_1",
#                      "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_TITLE",
#                      "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE")
# 
# amazon_train[vars_to_convert] <- lapply(amazon_train[vars_to_convert], as.factor)
# amazon_test[vars_to_convert] <- lapply(amazon_test[vars_to_convert], as.factor)
# 
# ggplot(data=amazon_train) + 
#   geom_boxplot(aes(x=RESOURCE, y=ACTION))



##### Recipe making #####

my_recipe <- recipe(ACTION~., data=amazon_train) %>%
              step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
              step_normalize(all_numeric_predictors()) %>%
              # step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
              # step_dummy(all_nominal_predictors()) # %>% # dummy variable encoding
              step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% #target encoding
              step_normalize(all_numeric_predictors()) %>%
              step_pca(all_predictors(),threshold = .9) # principle component reduction
              # also step_lencode_glm(), step_lencode_bayes(), and step_lencode_mixed()


prepped_recipe <- prep(my_recipe)
baked_recipe <- bake(prepped_recipe, amazon_test)


##### Logistic Regression #####
# 
# logReg_mod <- logistic_reg() %>% #Type of model
#                 set_engine("glm")
# 
# logReg_wf <- workflow() %>%
#               add_recipe(my_recipe) %>%
#               add_model(logReg_mod) %>%
#               fit(data = amazon_train) # Fit the workflow
# 
# logReg_preds <- predict(logReg_wf, new_data=amazon_test,type="prob") %>%
#   bind_cols(., amazon_test) %>% #Bind predictions with test data
#   select(id, .pred_1) %>% #Just keep resource and predictions
#   rename(Action=.pred_1)

#vroom_write(x=logReg_preds, file="./amazon_logReg.csv", delim=",")


##### Penalized Logistic Regression #####
# 
# penLog_mod <- logistic_reg(mixture = tune(),
#                            penalty = tune()) %>% #Type of model
#                 set_engine("glmnet")
# 
# penLog_wf <- workflow() %>%
#               add_recipe(my_recipe) %>%
#               add_model(penLog_mod) %>%
#               fit(data = amazon_train)
# 
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 5)
# 
# folds <- vfold_cv(amazon_train, v = 5, repeats = 1)
# 
# CV_results <- penLog_wf %>%
#                 tune_grid(resamples=folds,
#                           grid=tuning_grid,
#                           metrics=metric_set(roc_auc)) #f_meas,sens, recall,spec, precision, accuracy
# 
# bestTune <- CV_results %>%
#               select_best("roc_auc")
# 
# final_wf <- penLog_wf %>%
#               finalize_workflow(bestTune) %>%
#               fit(data=amazon_train)
# 
# penLog_preds <- predict(final_wf, new_data=amazon_test,type="prob") %>%
#   bind_cols(., amazon_test) %>% #Bind predictions with test data
#   select(id, .pred_1) %>% #Just keep resource and predictions
#   rename(Action=.pred_1)

#vroom_write(x=penLog_preds, file="./amazon_penLog.csv", delim=",")

# save(file="filename.RData", list=c("logReg_wf"))
# load("filename.RData")

##### Classification Random Forests #####
# 
# classForest_model <- rand_forest(mtry = tune(), # how many var are considered
#                             min_n=tune(), # how many observations per leaf
#                             trees=500) %>% #Type of model
#   set_engine("ranger") %>% # What R function to use
#   set_mode("classification")
# 
# ## Set Workflow
# classForest_wf <- workflow() %>%
#                   add_recipe(my_recipe) %>%
#                   add_model(classForest_model)
# 
# ## Grid of values to tune over
# tuning_grid <- grid_regular(mtry(range =c(1,5)),
#                             min_n(),
#                             levels = 6) ## L^2 total tuning possibilities
# 
# ## Split data for CV
# folds <- vfold_cv(amazon_train, v = 6, repeats=1)
# 
# CV_results <- classForest_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc)) #f_meas,sens, recall,spec, precision, accuracy
# 
# bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# final_wf <- classForest_wf %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=amazon_train)
# 
# classForest_preds <- predict(final_wf, new_data=amazon_test,type="prob") %>%
#   bind_cols(., amazon_test) %>% #Bind predictions with test data
#   select(id, .pred_1) %>% #Just keep resource and predictions
#   rename(Action=.pred_1)

#vroom_write(x=classForest_preds, file="./amazon_classForest.csv", delim=",")

##### Naive Bayes #####
# 
# nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
#              set_mode("classification") %>%
#              set_engine("naivebayes")
# 
# nb_wf <- workflow() %>%
#          add_recipe(my_recipe) %>%
#          add_model(nb_model)
# 
# tuning_grid <- grid_regular(Laplace(),
#                             smoothness(),
#                             levels = 10)
# 
# folds <- vfold_cv(amazon_train, v = 10, repeats = 1)
# 
# CV_results <- nb_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc)) #f_meas,sens, recall,spec, precision, accuracy
# 
# bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# final_wf <- nb_wf %>%
#  finalize_workflow(bestTune) %>%
#  fit(data=amazon_train)
# 
# nb_preds <- predict(final_wf, new_data=amazon_test,type="prob") %>%
#   bind_cols(., amazon_test) %>% #Bind predictions with test data
#   select(id, .pred_1) %>% #Just keep resource and predictions
#   rename(Action=.pred_1)
# 
# vroom_write(x=nb_preds, file="./amazon_naiveBayes.csv", delim=",")

##### K-Nearest Neighbors #####
# 
# knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
#               set_mode("classification") %>%
#               set_engine("kknn")
# 
# knn_wf <- workflow() %>%
#           add_recipe(my_recipe) %>%
#           add_model(knn_model)
# 
# tuning_grid <- grid_regular(neighbors(),
#                             levels = 10)
# 
# folds <- vfold_cv(amazon_train, v = 10, repeats = 1)
# 
# CV_results <- knn_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc)) #f_meas,sens, recall,spec, precision, accuracy
# 
# bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# final_wf <- knn_wf %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=amazon_train)
# 
# knn_preds <- predict(final_wf, new_data=amazon_test,type="prob") %>%
#   bind_cols(., amazon_test) %>% #Bind predictions with test data
#   select(id, .pred_1) %>% #Just keep resource and predictions
#   rename(Action=.pred_1)
# 
# vroom_write(x=knn_preds, file="./amazon_KNN.csv", delim=",")
# 



##### Support Vector Machines #####

## SVM models
# svmLinear <- svm_linear(cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
# set_engine("kernlab")
# 
# svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
# set_engine("kernlab")

svmRadial_model <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
              set_mode("classification") %>%
              set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial_model)

tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 3)

folds <- vfold_cv(amazon_train, v = 3, repeats = 1)

CV_results <- svm_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #f_meas,sens, recall,spec, precision, accuracy

bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <- svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

svm_preds <- predict(final_wf, new_data=amazon_test,type="prob") %>%
  bind_cols(., amazon_test) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep resource and predictions
  rename(Action=.pred_1)

vroom_write(x=svm_preds, file="./amazon_SVM.csv", delim=",")