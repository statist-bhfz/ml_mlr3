library(data.table)
library(mlr3verse)
library(mlr3tuning)
library(xgboost)

if (! dir.exists("out_12.02.2022")) dir.create("out_12.02.2022")
if (! dir.exists("model_12.02.2022")) dir.create("model_12.02.2022")

dt <- fread("data/train.csv")
unique(dt[, 1:4])[, .N]

# Усредняем по категориям предикторов
dt <- dt[, .(price = mean(price)), by = c("x1", "x2", "x3", "x4")]

# Конвертируем в факторы
cols <- setdiff(names(dt), "price")
dt[, (cols) := lapply(.SD, as.factor), .SDcols = cols]

# 10% оставляем для валидации при использовании early stopping
# и еще 10% для проверки при тюнинге остальных гиперпараметров
set.seed(42)
split <- sample(
  c("train", "val", "val_early_stopping"), 
  size = dt[, .N], 
  prob = c(0.8, 0.1, 0.1), 
  replace = TRUE
)


###########################################################
# Предварительная трансформация данных для early stopping #
###########################################################

# Задачи
task_train <- TaskRegr$new(
  id = "price", 
  backend = dt[split == "train"], 
  target = "price"
)
task_valid_early_stopping <- TaskRegr$new(
  id = "price_valid", 
  backend = dt[split == "val_early_stopping"], 
  target = "price"
)

# Граф вычислений для кодировки факторов при помощи 
# conditional target value impact encoding
# https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops_encodeimpact.html
gr <- 
  po("fixfactors") %>>% 
  po("encodeimpact", impute_zero = TRUE)

# Обучение на обучающей выборке
gr$train(task_train)

# Преобразование набора данных для ранней остановки
valid_early_stopping <- gr$predict(task_valid_early_stopping)
valid_early_stopping <- valid_early_stopping$encodeimpact.output$data()
cols <- setdiff(names(valid_early_stopping), "price")
valid_early_stopping <- xgboost::xgb.DMatrix(
  data = data.matrix(valid_early_stopping[, .SD, .SDcols = cols]),
  label = valid_early_stopping[, price]
)

##########################
# Тюнинг гиперпараметров #
##########################

# Задача
task <- TaskRegr$new(
  id = "price", 
  backend = dt, 
  target = "price"
)

# Разбивка на обучающую и проверочную выборку
resampling_custom <- rsmp("custom")
resampling_custom$instantiate(
  task, 
  list(dt[, .(id = .I)][split == "train", id]), 
  list(dt[, .(id = .I)][split == "val", id])
)

# Модель xgboost
learner_xgboost <- lrn(
  "regr.xgboost", 
  id = "xgb",
  booster = "gbtree", 
  nrounds = 1000,
  early_stopping_rounds = 10,
  watchlist = list(valid = valid_early_stopping),
  eta = 0.1,
  max_depth = 9,
  colsample_bytree = 0.8,
  tree_method = "gpu_hist",
  nthread = 4
)

# Граф вычислений
gr <- 
  po("fixfactors") %>>% 
  po("encodeimpact", impute_zero = TRUE) %>>% 
  po(learner_xgboost)
glearner <- as_learner(gr)
# Альтернативный вариант
# glearner <- GraphLearner$new(gr)

# Диапазон значений настраиваемых гиперпараметров
parameters_xgboost <- ParamSet$new(list(
  ParamDbl$new("xgb.eta", lower = 0.01, upper = 0.1),
  ParamInt$new("xgb.max_depth", lower = 10, upper = 15),
  ParamInt$new("xgb.nrounds", lower = 1000, upper = 1000),
  ParamDbl$new("xgb.colsample_bytree", lower = 0.8, upper = 1.0)
  )
)
# Альтернативный вариант
# parameters_xgboost <- ps(xgb.eta = p_dbl(lower = 0.01, upper = 0.1))

# Объект, содержащий все необходимое для тюнинга гиперпараметров
instance <- TuningInstanceSingleCrit$new(
  task = task,
  learner = glearner,
  resampling = resampling_custom,
  measure = msr("regr.mae"),
  search_space = parameters_xgboost,
  terminator = trm("evals", n_evals = 3)
)

# Тюнер, реализующий поиск по сетке значений с заданным количеством значений
# каждого из гиперпараметров
tuner <- tnr(
  "grid_search", 
  param_resolutions = c("xgb.eta" = 10, 
                        "xgb.max_depth" = 6,
                        "xgb.nrounds" = 1, 
                        "xgb.colsample_bytree" = 3)
)

# Тюнинг
tuner$optimize(instance)

tuning_results <- as.data.table(instance$archive)
setorder(tuning_results, regr.mae)
fwrite(tuning_results[, 1:5], 
       "out_12.02.2022/tuning_results.csv", sep = ";", append = TRUE)
