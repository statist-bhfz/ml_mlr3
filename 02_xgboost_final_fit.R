library(data.table)
library(mlr3verse)
library(mlr3learners)
library(mlr3tuning)
library(xgboost)

dt <- fread("data/train.csv")

# Усредняем по категориям предикторов
dt <- dt[, .(price = mean(price)), by = c("x1", "x2", "x3", "x4")]

# Конвертируем в факторы
cols <- setdiff(names(dt), "price")
dt[, (cols) := lapply(.SD, as.factor), .SDcols = cols]

# Подбор оптимального числа деревьев при помощи early stopping
# 10% оставляем для валидации при использовании early stopping
set.seed(42)
split <- sample(
  c("train", "val", "val_early_stopping"), 
  size = dt[, .N], 
  prob = c(0.8, 0.1, 0.1), 
  replace = TRUE
)

# Предварительная трансформация данных для early stopping
task_train <- TaskRegr$new(
  id = "dprice", 
  backend = dt[split %in% c("train", "val")], 
  target = "price"
)
task_valid_early_stopping <- TaskRegr$new(
  id = "price_valid", 
  backend = dt[split == "val_early_stopping"], 
  target = "price"
)
gr <- 
  po("fixfactors") %>>% 
  po("encodeimpact", impute_zero = TRUE)
gr$train(task_train)
valid_early_stopping <- gr$predict(task_valid_early_stopping)
valid_early_stopping <- valid_early_stopping$encodeimpact.output$data()
cols <- setdiff(names(valid_early_stopping), "price")
valid_early_stopping <- xgboost::xgb.DMatrix(
  data = data.matrix(valid_early_stopping[, .SD, .SDcols = cols]),
  label = valid_early_stopping[, price]
)


# Обучение модели с early stopping
params <- fread("out_12.02.2022/tuning_results.csv")
setorder(params, regr.mae)
params <- params[1]

learner_xgboost <- lrn(
  "regr.xgboost", 
  id = "xgb",
  booster = "gbtree", 
  nrounds = 1000,
  early_stopping_rounds = 10,
  watchlist = list(valid = valid_early_stopping),
  eta = params$xgb.eta,
  max_depth = params$xgb.max_depth,
  colsample_bytree = params$xgb.colsample_bytree,
  nthread = 4
)
gr <- 
  po("fixfactors") %>>% 
  po("encodeimpact", impute_zero = TRUE) %>>% 
  po(learner_xgboost)
glearner <- as_learner(gr) # GraphLearner$new(gr)
glearner$train(task_train)
glearner$model
# evaluation_log:
#   iter valid_rmse
# 1  107.67871
# 2  100.15424
# ---                
# 82   30.13365
# 83   30.13437

# Определяем оптимальное значение nrounds
nrounds_final <- glearner$model$xgb$model$best_iteration
nrounds_final <- round(nrounds_final * 1.1)

# Обучение итоговой модели на всех данных с оптимальным nrounds
task_train <- TaskRegr$new(
  id = "price", 
  backend = dt, 
  target = "price"
)
learner_xgboost <- lrn(
  "regr.xgboost", 
  id = "xgb",
  booster = "gbtree", 
  nrounds = nrounds_final,
  eta = params$xgb.eta,
  max_depth = params$xgb.max_depth,
  colsample_bytree = params$xgb.colsample_bytree,
  nthread = 4
)
gr <- 
  po("fixfactors") %>>% 
  po("encodeimpact", impute_zero = TRUE) %>>% 
  po(learner_xgboost)
glearner <- as_learner(gr) 
glearner$train(task_train)

# Сохраняем модель
saveRDS(glearner, "model_12.02.2022/learner_xgboost.rda")
