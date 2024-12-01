# OTTO PRODUCT CLASSIFICATION #
library(tidyverse)
library(tidymodels)
library(vroom)
library(lightgbm)
library(bonsai)
library(ranger)

#### READ IN DATA ####
otto_test = vroom("./test.csv")
otto_train = vroom("./train.csv")

## CLEAN THE DATA
otto_train$target = factor(otto_train$target)


#### EDA ####
glimpse(otto_train)
#summary(otto_train)

ggplot(otto_train, aes(x=target), stat=count) +
  geom_bar()

#### RECIPE ####
otto_recipe <- recipe(target~., data=otto_train) %>% 
  step_rm(id) %>% 
  step_normalize(all_numeric_predictors())

#### RANDOM FOREST ####
otto_rf <- rand_forest(mtry = tune(),
                       min_n = tune(),
                       trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

## WORKFLOW
rf_workflow <- workflow() %>% 
  add_recipe(otto_recipe) %>% 
  add_model(otto_rf)

## Tuning Grid
tuning_grid_rf <- grid_regular(mtry(range = c(1,10)),
                            min_n(),
                            levels=5)

## CV 
folds <- vfold_cv(otto_train, v=5, repeats=1)

## Run the CV
CV_results_rf <- rf_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid_rf,
            metrics=metric_set(mn_log_loss))
# metric_set(f_meas, sens, recall, spec)

## Find best tuning parameters
best_tune_rf <- CV_results_rf %>% 
  select_best(metric="mn_log_loss")
print(best_tune_rf)


#### LIGHT GBM ####
otto_lgbm <- boost_tree(tree_depth = tune(),  #15
                       learn_rate = tune(),  #0.1
                       trees = 500) %>% 
  set_engine("lightgbm") %>% 
  set_mode("classification")

## CV
CV_results_lgbm <- rf_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid_lgbm,
            metrics=metric_set(mn_log_loss))

## Find best tuning parameters
best_tune_lgbm <- CV_results_lgbm %>% 
  select_best(metric="mn_log_loss")
print(best_tune_lgbm)

## WORKFLOW
#lgbm_workflow <- workflow() %>% 
#  add_recipe(otto_recipe) %>% 
#  add_model(otto_lgbm) %>% 
#  fit(data = otto_train)


#### XGBOOST ####
otto_xgb <- boost_tree(learn_rate = tune(), #.3
                       tree_depth = tune(), #6
                       trees = 500) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

## WORKFLOW
xgb_workflow <- workflow() %>% 
  add_recipe(otto_recipe) %>% 
  add_model(otto_xgb)

## Tuning Grid
tuning_grid_xgb <- grid_regular(learn_rate(),
                                tree_depth(),
                                levels=5)

## CV
CV_results_xgb <- xgb_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid_xgb,
            metrics=metric_set(mn_log_loss))

best_tune_xgb <- CV_results_xgb %>% 
  select_best(metric="mn_log_loss")
print(best_tune_xgb)

#### COMPARE OUTPUT ####
collect_metrics(CV_results_rf)
collect_metrics(CV_results_xgb)
collect_metrics(CV_results_lgbm)


######################################

#### FINALIZE WF ####
#final_wf <- rf_workflow %>% 
#  finalize_workflow(best_tune) %>% 
#  fit(data=otto_train)

#### MAKE PREDICTIONS ####
#otto_preds <- final_wf %>% 
#  predict(new_data = otto_test, type="prob")

## Format predictions for kaggle upload
#otto_kaggle_submission <- otto_preds %>% 
#  bind_cols(otto_te$id) %>%
#  rename("id" = "...10",
#         "Class_1"= ".pred_Class_1",
#         "Class_2"= ".pred_Class_2",
#         "Class_3"= ".pred_Class_3",
#         "Class_4"= ".pred_Class_4",
#         "Class_5"= ".pred_Class_5",
#         "Class_6"= ".pred_Class_6",
#         "Class_7"= ".pred_Class_7",
#         "Class_8"= ".pred_Class_8",
#         "Class_9"= ".pred_Class_9") %>%
#  select(id, everything())

## Write out file
#vroom_write(x=otto_kaggle_submission, file="./OttoClassPreds.csv", delim=",")







