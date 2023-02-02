### Example script for Göteson et al, Transl Psych (2022)
### https://doi.org/10.1038/s41398-022-01819-y
### This document only describes the ML-model as the other analyses contain no custom code. 
### Questions? Email andreas.goteson@gu.se

set.seed(123)

# Load packages
require(tidyverse)
require(tidymodels)
require(furrr)

# Create nested data
A3 = nested_cv(A, outside = vfold_cv(v=5, repeats=5, strata=y), 
                    inside = bootstraps(times = 25))

### ------------------------------- ###
### Define model (in several steps) ###
### ------------------------------- ###

# 1) Define `makePredictions`. Includes model specifications and takes 
### the model and makes predictions to the analysis set.
makePredictions = function(object, model_type="rf", 
                           param1=NULL, param2=NULL){
  
  if (model_type=="rf"){
    mod = rand_forest(mode = "classification", trees=500, min_n=param1, mtry=param2) %>% 
      set_engine("ranger")
  }
  
  if (model_type=="reg"){
    mod = logistic_reg(mode = "classification", mixture=param1, penalty=param2) %>% 
      set_engine("ranger")
  }
  
  # Fit the models
  mod = mod %>% fit(y ~ ., data = analysis(object))
  
  # Predict
  holdout_pred <- 
    predict(mod, assessment(object) %>% dplyr::select(-y), type="class") %>% 
    bind_cols(predict(mod, assessment(object) %>% dplyr::select(-y), type="prob")) %>% 
    bind_cols(assessment(object) %>% dplyr::select(y))
  
  return(holdout_pred)
}

# 2) Define `deriveMetrics`. Takes a dataframe with predictions and 
### This function takes a dataframe with predictions (both class and prob)
### and derives user specified metrics. It requires the colnames for 
### truth and estimate to be specified. Output can be done as columns 
### or as a rowbind dataframe
deriveMetrics = function(df, truth="y", estimate=".pred_class", 
                          as_cols=F, two_class=F,
                          # Specify functions to derive metrics from
                          funs = list(class = list(accuracy, mcc, kap),
                                      prob = list())
){
  # List arguments to pass to each function
  class_args = list(data=df, truth=truth, estimate=estimate)
  if (two_class==T) prob_args = list(data=df, truth=truth, estimate=paste(".pred", levels(df[[truth]])[1], sep="_"))
  else prob_args = list(data=df, truth=truth, estimate=paste(".pred", levels(df[[truth]]), sep="_"))
  
  # Execute each function using the arguments and bind the results row-wise
  out = bind_rows(
    map_dfr(funs$class, exec, !!!class_args),
    map_dfr(funs$prob, exec, !!!prob_args)
  )
  
  # if as_cols=T --> spread to columns and drop ".estimator"
  if (as_cols==T) out = out[,-2] %>% spread(.metric, .estimate)
  
  return(out)
}

# 3) define `runWrapper`. A simple wrapper to combine ´makePredictions´ 
### and ´deriveMetrics´. Tunes across two parameters (e.g., lambda and mixture)
runWrapper = function(object, param1, param2, 
                      funs=slim_funs, two_class=F, as_cols=F){
  makePredictions(object, param1=param1, param2=param2) %>% 
    deriveMetrics(funs=funs, two_class=two_class, as_cols=as_cols)
}

# 4) define `tuneModel`. Runs the models across a grid. The grid is 
### defined inside the function for proper inconvenience :)
tuneModel <- function(object, model_type="rf", two_class=F) {
  
  if (model_type=="reg"){
    grid = grid_regular(penalty(), mixture(), levels = 20) %>%   
      rename("param1" = mixture, "param2" = penalty)
  }
  
  if (model_type=="rf"){
    grid = expand.grid(
      mtry = seq(5,50, by=5),
      min_n = seq(5, 50, by=5)) %>% 
      rename("param1" = min_n, "param2" = mtry)
  }
  
  # Map across grid and derive metrics from inner loop
  grid %>% bind_cols(map2_dfr(.$param1, .$param2, runWrapper, 
                              object=object, two_class=two_class, as_cols=T))
}

# 5) define `summarize_tune_results`. Runs the tuning and computes 
### mean metrics across inner loops
summarize_tune_results <- function(object, model_type="rf", two_class=F) {
  
  # Return row-bound tibble of the inner loop results
  out = map_df(object$splits, tuneModel, 
               model_type=model_type, two_class=two_class) %>% 
    # For each value of the tuning parameter, compute the 
    # average metric which is the inner bootstrap estimate. 
    group_by(param1, param2) %>%
    summarize_all(mean, na.rm=T, .groups="drop")
  
  if (model_type=="rf") out = rename(out, "min_n" = param1, "mtry" = param2)
  if (model_type=="reg") out = rename(out, "mixture" = param1, "penalty" = param2)
  
  return(out)
}


### ---------------- ###
### Tune and evalute ###
### ---------------- ###

# Run the tuning (using parallel processing)
A3_tuning_rf <- future_map(A3$inner_resamples, summarize_tune_results,
                         model_type = "rf",
                        .options = furrr_options(seed = TRUE)) 

# Evaluate tuning with a plot
A3_pooled_inner <- A3_tuning_rf %>% bind_rows(.id="outer_loop") 

plot_data=A3_pooled_inner %>% 
  pivot_longer(cols = c("accuracy", "kap"), 
               names_to = "metric",
               values_to = "metric_val") %>% 
  group_by(outer_loop, mtry, min_n, metric) %>% 
  summarize(metric_val = mean(metric_val))

ggarrange(
  # Accuracy
  ggplot(plot_data %>% filter(metric=="accuracy"), 
         aes(min_n, mtry, fill=metric_val)) + 
    geom_point(shape=21, size=3) + 
    scale_fill_viridis_c(guide = guide_colorbar("")) + 
    theme(legend.position="top"),
  # Kappa
  ggplot(plot_data %>% filter(metric=="kap"), 
         aes(min_n, mtry, fill=metric_val)) + 
    geom_point(shape=21, size=3) + 
    scale_fill_viridis_c(guide = guide_colorbar("")) + 
    theme(legend.position="top")
  )


# Select the best parameters acc to Cohen's kappa and finalize model
extractParams = function(tune_res, metric="kap", model_type="rf"){
  if (model_type=="rf") params = c("min_n", "mtry")
  if (model_type=="reg") params = c("mixture", "penalty")
  
  tune_res %>% 
    map_df(.f = function(x) arrange_at(x, metric, .by_group = F) %>% tail(1)) %>% 
  select(params)
}

A3 = bind_cols(A3, extractParams(A3_tuning_rf, model_type="rf") 

### ------------------------------- ###
### Finalize model and plot metrics ###
### ------------------------------- ###

### pmap maps data and hyperparameters from `results` to runWrapper
A3$rf_metrics = pmap_dfr(list(A3$splits, A3$min_n, A3$mtry), 
                 .f = runWrapper, funs=maxxed_funs,
                 as_cols=T, two_class=T)

# Plot the metrics per outer segment
bind_cols(A3[,c(2,3)], A3$rf_metrics) %>% 
    gather(metric, metric_val, -id, -id2) %>% 
    ggplot(., aes(metric, metric_val)) + 
    geom_boxplot(outlier.shape = NA, alpha=.3) + 
    geom_point(aes(col=paste(id, id2, sep="_"))) + 
    geom_line(aes(group=id, col=id)) + 
    coord_flip() +
    theme_minimal_grid(font_size=12) + 
    labs(x="", y="") + 
    guides(col=F)


### ----------------------- ###
### Per class metrics & VIP ###
### ----------------------- ###

# Function to change predictions to two-class format
predByClass = function(pred, class, truth="y"){

  # 1) Change hard-called predictions to be class vs. all other
  tmp_pred = pred %>%
    mutate_at(vars(.pred_class, truth),
              function(x) {
                factor(ifelse(x == class, class, "other"), levels=c(class, "other"))
                })

  # 2) Change probabilities to two-level format
  ### Sum up probabilities for all classes except `class` and join
  .pred_other = tmp_pred %>% select(!matches(class)) %>% select(where(is.numeric)) %>% 
    mutate(.pred_other = rowSums(.)) %>% select(.pred_other)
  
  out = tmp_pred %>% select(.pred_class, matches(class), y) %>% 
    bind_cols(.pred_other)
  
  return(out)
}

# Pick one prediction output
pred = pmap_dfr(list(A3$splits, A3$min_n, A3$mtry), 
                .f = makePredictions,
                model_type="rf")

metrics_by_class = 
  lapply(levels(pred$y), function(x){
  # Convert to two-class
  predByClass(pred, x) %>%
    # Derive metrics
    deriveMetrics(two_class=T, funs=maxxed_funs)
  }) %>% 
  # Bind all rows together
  bind_rows(.id="ind") %>% 
  inner_join(
    tibble(class = levels(pred$y),
           ind = as.character(1:length(class)))
  ) %>% select(-ind)

metrics_by_class %>% 
  filter(.metric %in% c("accuracy", "kap", "roc_auc", "classification_cost")) %>% 
  mutate_at(".estimate", round, 2) 

# Derive VIP-scores
deriveVip = function(splits, min_n, mtry){
  rand_forest(mode="classification", trees=500,
              min_n=min_n, mtry=mtry) %>% 
    set_engine("ranger", importance = "permutation") %>% 
    fit(y ~ ., data = analysis(splits)) %>% 
    vi()
}

VIPscores = pmap(list(A3$splits, A3$min_n, A3$mtry), .f=deriveVip) %>% 
  bind_rows(.id="outer")