##
## Amazon Employee Access
##

library(tidymodels)
library(vroom)
#library(embed) # target encoding
#library(glmnet) # penalized log regression

amazon_train <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/AmazonEmployeeAccess/train.csv") %>%
  mutate(ACTION = as.factor(ACTION))
amazon_test <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/AmazonEmployeeAccess/test.csv")


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
              # step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
              step_dummy(all_nominal_predictors()) # %>% # dummy variable encoding
              # step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding
              # also step_lencode_glm() and step_lencode_bayes()


prepped_recipe <- prep(my_recipe)
baked_recipe <- bake(prepped_recipe, amazon_test)


##### Logistic Regression #####

logReg_mod <- logistic_reg() %>% #Type of model
                set_engine("glm")

logReg_wf <- workflow() %>%
              add_recipe(my_recipe) %>%
              add_model(logReg_mod) %>%
              fit(data = amazon_train) # Fit the workflow

logReg_preds <- predict(logReg_wf, new_data=amazon_test,type="prob") %>%
  bind_cols(., amazon_test) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep resource and predictions
  rename(Action=.pred_1)

vroom_write(x=logReg_preds, file="./amazon_logReg.csv", delim=",")


##### Penalized Logistic Regression #####

penLog_mod <- logistic_reg(mixture = tune(),
                           penalty = tune()) %>% #Type of model
                set_engine("glmnet")

penLog_wf <- workflow() %>%
              add_recipe(my_recipe) %>%
              add_model(penLog_mod) %>%
              fit(data = amazon_train)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(amazon_train, v = 5, repeats = 1)

CV_results <- penLog_wf %>%
                tune_grid(resamples=folds,
                          grid=tuning_grid,
                          metrics=metric_set(roc_auc)) #f_meas,sens, recall,spec, precision, accuracy

bestTune <- CV_results %>%
              select_best("roc_auc")

final_wf <- penLog_wf %>%
              finalize_workflow(bestTune) %>%
              fit(data=amazon_train)

penLog_preds <- predict(final_wf, new_data=amazon_test,type="prob") %>%
  bind_cols(., amazon_test) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep resource and predictions
  rename(Action=.pred_1)

vroom_write(x=penLog_preds, file="./amazon_penLog.csv", delim=",")

# save(file="filename.RData", list=c("logReg_wf"))
# load("filename.RData")