library(data.table)
library(mlr3verse)
library(mlr3learners)
library(mlr3tuning)
library(xgboost)

# Загружаем модель xgboost
learner_xgboost <- readRDS("model_12.02.2022/learner_xgboost.rda")

test_data <- fread("data/test.csv")

# Усредняем по категориям предикторов
test_data <- test_data[, .(price = mean(price)), 
                       by = c("x1", "x2", "x3", "x4")]

# Конвертируем в факторы
cols <- learner_xgboost$model$xgb$model$feature_names
test_data[, (cols) := lapply(.SD, as.factor), .SDcols = cols]

task_test <- TaskRegr$new(
  id = "price", 
  backend = test_data, 
  target = "price"
)
preds_test_data <- learner_xgboost$predict(task_test)
preds_test_data$score(msrs(c("regr.mae", "regr.mape")))
preds_test_data <- as.data.table(preds_test_data)
