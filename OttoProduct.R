# OTTO PRODUCT CLASSIFICATION #
library(tidyverse)
library(tidymodels)
library(vroom)
library(lightgbm)
library(bonsai)
library(ranger)
library(discrim)

#### READ IN DATA ####
otto_test = vroom("./test.csv")
otto_train = vroom("./train.csv")

## CLEAN THE DATA
otto_train$target = factor(otto_train$target)

#
#### EDA ####
glimpse(otto_train)
#summary(otto_train)

ggplot(otto_train, aes(x=target), stat=count) +
  geom_bar()

#### RECIPE ####
otto_recipe <- recipe(target~., data=otto_train) %>% 
  step_rm(id) %>% 
  step_normalize(all_numeric_predictors())

#### LIGHT GBM ####
otto_lgbm <- boost_tree(tree_depth = 4,  #4
                       learn_rate = 0.1,  #0.1
                       trees = 1000) %>% 
  set_engine("lightgbm") %>% 
  set_mode("classification")

## CV 
folds <- vfold_cv(otto_train, v=5, repeats=1)
CV_results_lgbm <- fit_resamples(otto_lgbm,
                                 otto_recipe,
                                 folds,
                                 metrics = metric_set(mn_log_loss),
                                 control = control_resamples(save_pred = T))
collect_metrics(CV_results_lgbm)
# mean log loss: 0.491

#### RANDOM FOREST ####
otto_rf <- rand_forest(mtry = 10,
                       min_n = 2,
                       trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

## Run the CV
CV_results_rf <- fit_resamples(otto_rf,
                               otto_recipe,
                               folds,
                               metrics = metric_set(mn_log_loss),
                               control = control_resamples(save_pred = T))
collect_metrics(CV_results_rf)
# mean log loss: 0.567

#### NAIVE BAYES ####
otto_nb <- naive_Bayes(Laplace=0,
                       smoothness=1.5) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

CV_results_nb <- fit_resamples(otto_nb,
                               otto_recipe,
                               folds,
                               metrics = metric_set(mn_log_loss),
                               control = control_resamples(save_pred = T))
collect_metrics(CV_results_nb)
# mean log loss: 21.2


#### COMPARE OUTPUT ####
collect_metrics(CV_results_lgbm) # best
collect_metrics(CV_results_rf)
collect_metrics(CV_results_nb) # worst


######################################

#### FINALIZE WF ####
lgbm_workflow <- workflow() %>% 
  add_recipe(otto_recipe) %>% 
  add_model(otto_lgbm) %>% 
  fit(data = otto_train)

#### MAKE PREDICTIONS ####
otto_preds <- lgbm_workflow %>% 
  predict(new_data = otto_test, type="prob")

## Format predictions for kaggle upload
otto_kaggle_submission <- otto_preds %>% 
  bind_cols(otto_test$id) %>%
  rename("id" = "...10",
         "Class_1"= ".pred_Class_1",
         "Class_2"= ".pred_Class_2",
         "Class_3"= ".pred_Class_3",
         "Class_4"= ".pred_Class_4",
         "Class_5"= ".pred_Class_5",
         "Class_6"= ".pred_Class_6",
         "Class_7"= ".pred_Class_7",
         "Class_8"= ".pred_Class_8",
         "Class_9"= ".pred_Class_9") %>%
  select(id, everything())

## Write out file
vroom_write(x=otto_kaggle_submission, file="./OttoClassPreds.csv", delim=",")




