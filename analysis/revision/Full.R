# ==============================================================================
# Script Name: 08_integrated_analysis_final_strict_v12.R
# Description: 
#   [Final Polish v12]:
#   1. Unified Age Subgroups: Console output now prints <60, 60-75, >75 to 
#      perfectly match Figure 6. The obsolete <=75 logic is completely removed.
#   2. Statistical Rigor: True Coefficient of Determination (R^2) used everywhere.
#   3. Previous features maintained: Pearson r (95% CI), clean console, 
#      strict random seed locking, and detailed publication-ready figures.
# ==============================================================================

# ================= 0. 環境與套件設定 =================
my_repo <- "https://cloud.r-project.org"

ensure_package <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    message(paste("Installing:", pkg))
    install.packages(pkg, repos = my_repo)
    if (!require(pkg, character.only = TRUE)) stop(paste("Failed to install:", pkg))
  }
}

pkgs <- c("ggplot2", "dplyr", "cowplot", "scales", "boot", "pROC", "gridExtra", "tidyr", "readr")
sapply(pkgs, ensure_package)

library(ggplot2); library(dplyr); library(cowplot); library(scales)
library(boot); library(pROC); library(gridExtra); library(tidyr); library(readr)

# ================= 1. 路徑與全域設定 =================
RAW_TRAIN_PATH  <- '../../data/train.csv'
RAW_TEST_PATH   <- '../../data/test.csv'
TRAIN_META_PATH <- '../../log/ASMI_Regression_ImplantCleaned/ResNetFusionAttentionNetRegression/0_run_ASMI-Reg_2025-10-03_16-55-03/csv_data/patient_data.csv'
VAL_PRED_PATH   <- '../../log/ASMI_Regression_ImplantCleaned/ResNetFusionAttentionNetRegression/0_run_ASMI-Reg_2025-10-03_16-55-03/csv_data/validation_predictions.csv'
TEST_META_PATH  <- '../../log/test/2025-10-04_22-54-04/csv_data/test_patient_data.csv'
TEST_PRED_PATH  <- '../../log/test/2025-10-04_22-54-04/csv_data/test_predictions.csv'

OUTPUT_DIR <- "results_integrated_v12"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

N_BOOTSTRAPS <- 1000

# ================= 2. 輔助函數 =================

# [STATISTICAL UPDATE]: 真正的 Coefficient of Determination (R^2)
calc_r2_true <- function(y_true, y_pred) {
  valid_idx <- complete.cases(y_true, y_pred)
  yt <- y_true[valid_idx]
  yp <- y_pred[valid_idx]
  
  if(length(yt) == 0) return(NA)
  
  ss_res <- sum((yt - yp)^2)
  ss_tot <- sum((yt - mean(yt))^2)
  
  if(ss_tot == 0) return(NA)
  return(1 - (ss_res / ss_tot))
}

calculate_risk_score <- function(gender, pred_asmi) {
  return(ifelse(gender == 0, 7.0 - pred_asmi, 5.4 - pred_asmi))
}

predict_sarcopenia_class <- function(gender, pred_asmi) {
  cutoff <- ifelse(gender == 0, 7.0, 5.4) 
  return(ifelse(pred_asmi < cutoff, 1, 0))
}

fmt_cont <- function(vec) sprintf("%.1f ± %.1f", mean(vec, na.rm=T), sd(vec, na.rm=T))
fmt_cat  <- function(vec, val=1) {
  n <- sum(vec == val, na.rm=T)
  perc <- mean(vec == val, na.rm=T) * 100
  sprintf("%d (%.1f%%)", n, perc)
}

calc_smd <- function(v1, v2, type="cont", val=1) {
  if(type=="cont") {
    m1 <- mean(v1, na.rm=T); s1 <- sd(v1, na.rm=T)
    m2 <- mean(v2, na.rm=T); s2 <- sd(v2, na.rm=T)
    pooled <- sqrt((s1^2 + s2^2)/2)
    if(pooled==0) return(0) else return((m1-m2)/pooled)
  } else {
    p1 <- mean(v1==val, na.rm=T); p2 <- mean(v2==val, na.rm=T)
    pooled <- sqrt((p1*(1-p1) + p2*(1-p2))/2)
    if(pooled==0) return(0) else return((p1-p2)/pooled)
  }
}

calc_metrics_basic <- function(y_true, y_pred) {
  mae <- mean(abs(y_true - y_pred), na.rm=TRUE)
  rmse <- sqrt(mean((y_true - y_pred)^2, na.rm=TRUE))
  r2 <- calc_r2_true(y_true, y_pred) # 使用真正的 R2
  pearson_r <- cor(y_true, y_pred, use="complete.obs", method="pearson")
  return(c(MAE=mae, RMSE=rmse, R2=r2, Pearson=pearson_r))
}

# ================= 3. 資料讀取與前處理 =================
message("\n=== 正在讀取並整理資料 ===")

if(!file.exists(TRAIN_META_PATH)) stop("找不到 Train Meta 檔案，請檢查路徑")

df_train_meta <- read.csv(TRAIN_META_PATH)
df_test_meta <- read.csv(TEST_META_PATH)
df_test_pred <- read.csv(TEST_PRED_PATH)
df_val_pred  <- read.csv(VAL_PRED_PATH)

# --- Test Set ---
df_test <- merge(df_test_meta, df_test_pred[, c('UID', 'predicted_asmi')], by='UID')
if(file.exists(RAW_TEST_PATH)) {
  df_raw_test <- read.csv(RAW_TEST_PATH)
  time_cols <- grep("time|period", names(df_raw_test), value=TRUE, ignore.case=TRUE)
  if(length(time_cols) > 0) {
    target_col <- time_cols[1]
    temp_time <- df_raw_test[, c('UID', target_col)]
    names(temp_time)[2] <- 'Time_period'
    df_test <- merge(df_test, temp_time, by='UID', all.x=TRUE)
  } else { df_test$Time_period <- NA }
} else { df_test$Time_period <- NA }

# --- Internal Set ---
cols_to_keep <- c('UID', 'Gender', 'ASMI', 'Age', 'Height', 'Weight', 'BMI', 'Low_muscle_mass')
df_internal <- merge(df_val_pred, df_train_meta[, cols_to_keep], by='UID')
if("actual_asmi" %in% names(df_internal)) df_internal$ASMI <- df_internal$actual_asmi

# --- GT Setup ---
fix_gt_col <- function(vec) {
  if(is.numeric(vec)) return(vec)
  return(as.numeric(vec == 1 | vec == "True" | vec == "Yes" | vec == TRUE))
}
df_train_meta$Sarcopenia_GT <- fix_gt_col(df_train_meta$Low_muscle_mass)
df_internal$Sarcopenia_GT <- fix_gt_col(df_internal$Low_muscle_mass)
df_test$Sarcopenia_GT  <- fix_gt_col(df_test$Low_muscle_mass)

# --- Probability Calculation ---
df_internal$risk_score <- calculate_risk_score(df_internal$Gender, df_internal$predicted_asmi)
df_test$risk_score <- calculate_risk_score(df_test$Gender, df_test$predicted_asmi)

risk_model <- glm(Sarcopenia_GT ~ risk_score, data = df_internal, family = binomial)
df_test$prob_sarcopenia <- predict(risk_model, df_test, type = "response")
df_internal$prob_sarcopenia <- predict(risk_model, df_internal, type = "response")

eps <- 1e-9
df_test$prob_clipped <- pmax(pmin(df_test$prob_sarcopenia, 1-eps), eps)
df_test$logit_prob <- log(df_test$prob_clipped / (1 - df_test$prob_clipped))

# ==============================================================================
# PART 2: ANALYSIS CORE & DATA CAPTURE
# ==============================================================================

message("\n================================================================================")
message("1. Table 1: Baseline characteristics")
message("--------------------------------------------------------------------------------")
message(sprintf("%-20s | %-25s | %-25s | %-10s | %-10s", "Variable", "Dev Set", "Test Set", "P-value", "SMD"))

vars <- list(
  list(n="Age (years)", c="Age", t="cont"),
  list(n="Male sex", c="Gender", t="bin", v=0), 
  list(n="Height (cm)", c="Height", t="cont"),
  list(n="Weight (kg)", c="Weight", t="cont"),
  list(n="BMI (kg/m²)", c="BMI", t="cont"),
  list(n="ASMI (kg/m²)", c="ASMI", t="cont"),
  list(n="Sarcopenia (GT)", c="Sarcopenia_GT", t="bin", v=1)
)

for(v in vars) {
  if(v$c %in% names(df_train_meta) & v$c %in% names(df_test)) {
    tr <- df_train_meta[[v$c]]; te <- df_test[[v$c]]
    if(v$t=="cont") {
      s1 <- fmt_cont(tr); s2 <- fmt_cont(te)
      pval <- t.test(tr, te)$p.value
      smd <- calc_smd(tr, te, "cont")
    } else {
      s1 <- fmt_cat(tr, v$v); s2 <- fmt_cat(te, v$v)
      tbl <- table(factor(c(tr,te), levels=c(0,1)), factor(c(rep("A",length(tr)),rep("B",length(te)))))
      pval <- tryCatch(chisq.test(tbl)$p.value, error=function(e) 1)
      smd <- calc_smd(tr, te, "bin", v$v)
    }
    pstr <- ifelse(pval<0.001, "<0.001", sprintf("%.3f", pval))
    message(sprintf("%-20s | %-25s | %-25s | %-10s | %.3f", v$n, s1, s2, pstr, smd))
  }
}

message("\n================================================================================")
message("2. Linear Regression Baseline comparison")
message("--------------------------------------------------------------------------------")

model_nb <- lm(ASMI ~ Age + Gender + Height + Weight, data=df_train_meta)
model_bmi <- lm(ASMI ~ Age + Gender + Height + Weight + BMI, data=df_train_meta)

pred_lr_nb <- predict(model_nb, df_test)
pred_lr_bmi <- predict(model_bmi, df_test)

met_ai <- calc_metrics_basic(df_test$ASMI, df_test$predicted_asmi)
met_lr_nb <- calc_metrics_basic(df_test$ASMI, pred_lr_nb)
met_lr_bmi <- calc_metrics_basic(df_test$ASMI, pred_lr_bmi)

message("---- Performance Comparison ----")
message(sprintf("AI Model            : MAE=%.4f, RMSE=%.4f, R2=%.4f, Pearson=%.4f", met_ai['MAE'], met_ai['RMSE'], met_ai['R2'], met_ai['Pearson']))
message(sprintf("LR model (No BMI)   : MAE=%.4f, RMSE=%.4f, R2=%.4f, Pearson=%.4f", met_lr_nb['MAE'], met_lr_nb['RMSE'], met_lr_nb['R2'], met_lr_nb['Pearson']))
message(sprintf("LR model (With BMI) : MAE=%.4f, RMSE=%.4f, R2=%.4f, Pearson=%.4f", met_lr_bmi['MAE'], met_lr_bmi['RMSE'], met_lr_bmi['R2'], met_lr_bmi['Pearson']))


message("\n================================================================================")
message("3. Subgroup Analysis (Calculating & Storing for Figure 6)")
message("--------------------------------------------------------------------------------")
message(sprintf("%-18s | %-4s | %-6s | %-6s | %-22s | %-22s | %-22s", "Subgroup", "N", "MAE", "R2", "Pearson r (95% CI)", "AUC (95% CI)", "Brier (95% CI)"))

fig6_data_ready <- data.frame()

subgroup_boot_fn <- function(data, indices) {
  d <- data[indices, ]
  risk_score <- calculate_risk_score(d$Gender, d$predicted_asmi)
  
  # AUC
  auc_val <- NA
  tryCatch({
    if (length(unique(d$Sarcopenia_GT)) > 1) {
      auc_val <- as.numeric(auc(roc(d$Sarcopenia_GT, risk_score, quiet = TRUE)))
    }
  }, error = function(e) { auc_val <<- NA })
  
  # Brier
  brier_val <- NA
  if("prob_sarcopenia" %in% names(d)) brier_val <- mean((d$prob_sarcopenia - d$Sarcopenia_GT)^2)
  
  # Pearson r
  r_val <- cor(d$ASMI, d$predicted_asmi, method = "pearson")
  
  return(c(auc_val, brier_val, r_val))
}

process_subgroup_logic <- function(name, df_sub, set_label="External", silent=FALSE) {
  
  set.seed(42) # [CRITICAL CONSISTENCY LOCK]
  
  # 1. Point Estimates
  mae <- mean(abs(df_sub$ASMI - df_sub$predicted_asmi))
  r2 <- calc_r2_true(df_sub$ASMI, df_sub$predicted_asmi) # 真正 R2
  pearson_r <- cor(df_sub$ASMI, df_sub$predicted_asmi, method="pearson")
  
  # 2. Bootstrap
  auc_val <- NA; auc_low <- NA; auc_high <- NA; auc_str <- "NA"
  brier_str <- "NA"
  pearson_low <- NA; pearson_high <- NA; pearson_str <- sprintf("%.3f", pearson_r)
  
  if(nrow(df_sub) >= 5) {
    boot_res <- tryCatch({
      boot(data = df_sub, statistic = subgroup_boot_fn, R = 1000, strata = df_sub$Sarcopenia_GT) 
    }, error = function(e) return(NULL))
    
    if(!is.null(boot_res)) {
      # AUC (Index 1)
      auc_val <- boot_res$t0[1]
      tryCatch({
        ci <- boot.ci(boot_res, type = "perc", index = 1)
        auc_low <- ci$percent[4]; auc_high <- ci$percent[5]
        auc_str <- sprintf("%.3f (%.3f-%.3f)", auc_val, auc_low, auc_high)
      }, error=function(e){})
      
      # Brier (Index 2)
      brier_val <- boot_res$t0[2]
      tryCatch({
        ci <- boot.ci(boot_res, type = "perc", index = 2)
        brier_str <- sprintf("%.3f (%.3f-%.3f)", brier_val, ci$percent[4], ci$percent[5])
      }, error=function(e){})
      
      # Pearson (Index 3)
      pearson_val_boot <- boot_res$t0[3]
      tryCatch({
        ci <- boot.ci(boot_res, type = "perc", index = 3)
        pearson_low <- ci$percent[4]; pearson_high <- ci$percent[5]
        pearson_str <- sprintf("%.3f (%.3f-%.3f)", pearson_val_boot, pearson_low, pearson_high)
      }, error=function(e){})
    }
  }
  
  # 3. Print Console (if not silent)
  if(!silent) {
    message(sprintf("%-18s | %-4d | %.3f  | %.3f  | %-22s | %-22s | %-22s", name, nrow(df_sub), mae, r2, pearson_str, auc_str, brier_str))
  }
  
  # 4. Return Data
  return(data.frame(
    Set = set_label,
    Subgroup = name,
    N = nrow(df_sub),
    R2 = r2,
    Pearson_r = pearson_r,
    Pearson_CI_lower = pearson_low,
    Pearson_CI_upper = pearson_high,
    Pearson_text = pearson_str,
    AUC_val = auc_val,
    AUC_CI_lower = auc_low,
    AUC_CI_upper = auc_high,
    AUC_text = auc_str
  ))
}

# --- 執行 Subgroup 分析 ---

# 1. Gender
g_list <- list("Male"=0, "Female"=1)
for(g_name in names(g_list)) {
  # Internal: Silent
  row_int <- process_subgroup_logic(g_name, df_internal %>% filter(Gender == g_list[[g_name]]), "Internal", silent=TRUE)
  fig6_data_ready <- rbind(fig6_data_ready, row_int)
  # External: Print
  row_ext <- process_subgroup_logic(g_name, df_test %>% filter(Gender == g_list[[g_name]]), "External", silent=FALSE)
  fig6_data_ready <- rbind(fig6_data_ready, row_ext)
}

# 2. Age (Unified Logic: <60, 60-75, >75)
# [MODIFIED]: Consolidated into a single function with a silent_flag parameter
run_age_fig6 <- function(df, set_label, silent_flag) {
  r1 <- process_subgroup_logic("<60", df %>% filter(Age < 60), set_label, silent=silent_flag)
  r2 <- process_subgroup_logic("60-75", df %>% filter(Age >= 60 & Age <= 75), set_label, silent=silent_flag)
  r3 <- process_subgroup_logic(">75", df %>% filter(Age > 75), set_label, silent=silent_flag)
  return(rbind(r1, r2, r3))
}
# Execute Internal (Silent)
fig6_data_ready <- rbind(fig6_data_ready, run_age_fig6(df_internal, "Internal", silent_flag=TRUE))
# Execute External (Prints to Console matching exactly the Figure 6 groups)
fig6_data_ready <- rbind(fig6_data_ready, run_age_fig6(df_test, "External", silent_flag=FALSE))

# 3. Sensitivity (Time)
if(!all(is.na(df_test$Time_period))) {
  message("--------------------------------------------------------------------------------")
  message("[Sensitivity Analysis] Time Period from Raw Data")
  time_groups <- list("Time < 30 days" = df_test %>% filter(Time_period < 30),
                      "Time 30-90 days" = df_test %>% filter(Time_period >= 30 & Time_period <= 90))
  for(t_name in names(time_groups)) {
    if(nrow(time_groups[[t_name]]) > 0) {
      dummy <- process_subgroup_logic(t_name, time_groups[[t_name]], "External", silent=FALSE)
    }
  }
}


message("\n================================================================================")
message("4. Bootstrap 95% Confidence Intervals (Main Analysis - Figure 4 Data)")
message("--------------------------------------------------------------------------------")

fig4_data_ready <- data.frame()

boot_stats_fn <- function(data, indices) {
  d <- data[indices, ]
  y_true <- d$ASMI; y_pred <- d$predicted_asmi
  y_class <- d$Sarcopenia_GT 
  y_class_pred <- predict_sarcopenia_class(d$Gender, d$predicted_asmi)
  risk_score <- calculate_risk_score(d$Gender, d$predicted_asmi)
  
  mae <- mean(abs(y_true - y_pred))
  rmse <- sqrt(mean((y_true - y_pred)^2)) 
  r2 <- calc_r2_true(y_true, y_pred) # 使用真正的 R2
  pearson_r <- cor(y_true, y_pred, method = "pearson")
  
  tp <- sum(y_class == 1 & y_class_pred == 1); tn <- sum(y_class == 0 & y_class_pred == 0)
  fp <- sum(y_class == 0 & y_class_pred == 1); fn <- sum(y_class == 1 & y_class_pred == 0)
  
  sens <- ifelse((tp+fn)>0, tp/(tp+fn), 0)
  spec <- ifelse((tn+fp)>0, tn/(tn+fp), 0)
  acc <- mean(y_class == y_class_pred)
  f1 <- ifelse((tp+fp+fn)>0, 2*tp/(2*tp+fp+fn), 0)
  ppv <- ifelse((tp+fp)>0, tp/(tp+fp), 0)
  npv <- ifelse((tn+fn)>0, tn/(tn+fn), 0)
  
  auc_val <- NA
  tryCatch({
    if (length(unique(y_class)) > 1) auc_val <- as.numeric(auc(roc(y_class, risk_score, quiet = TRUE)))
  }, error = function(e) {})
  
  return(c(mae, rmse, r2, pearson_r, auc_val, sens, spec, acc, f1, ppv, npv))
}

run_boot_main <- function(df, label) {
  cat(sprintf("-> Running Bootstrap for %s...\n", label))
  set.seed(42)
  b_out <- boot(data = df, statistic = boot_stats_fn, R = N_BOOTSTRAPS, strata = df$Sarcopenia_GT)
  
  metric_keys <- c("MAE", "RMSE", "R2", "Pearson_r", "AUC", "Sensitivity", "Specificity", "Accuracy", "F1_Score", "PPV", "NPV")
  metric_names_plot <- c("MAE", "RMSE", "R2", "Pearson r", "AUC-ROC", "Sensitivity", "Specificity", "Accuracy", "F1 Score", "PPV", "NPV")
  
  res_df <- data.frame()
  
  for(i in 1:length(metric_keys)) {
    val <- b_out$t0[i]
    ci_low <- NA; ci_high <- NA; ci_str <- "(NA)"
    tryCatch({
      ci <- boot.ci(b_out, type = "perc", index = i)
      ci_low <- ci$percent[4]; ci_high <- ci$percent[5]
      ci_str <- sprintf("(%.3f-%.3f)", ci_low, ci_high)
    }, error=function(e){})
    
    res_df <- rbind(res_df, data.frame(
      Dataset = label,
      Metric = metric_names_plot[i],
      Mean = val,
      CI_Lower = ci_low,
      CI_Upper = ci_high,
      Display = sprintf("%.3f %s", val, ci_str)
    ))
  }
  return(res_df)
}

res_internal <- run_boot_main(df_internal, "Internal")
res_external <- run_boot_main(df_test, "External")
fig4_data_ready <- rbind(res_internal, res_external)

message("\n--------------------------------------------------------------------------------")
message(" FINAL SUMMARY TABLE FOR MANUSCRIPT (Mean + 95% CI)")
message("--------------------------------------------------------------------------------")
message(sprintf("%-15s | %-30s | %-30s", "Metric", "Internal Val (5-Fold)", "External Val"))
metrics_to_show <- unique(fig4_data_ready$Metric)
for(m in metrics_to_show) {
  v_int <- fig4_data_ready$Display[fig4_data_ready$Dataset=="Internal" & fig4_data_ready$Metric==m]
  v_ext <- fig4_data_ready$Display[fig4_data_ready$Dataset=="External" & fig4_data_ready$Metric==m]
  if(length(v_int)>0 && length(v_ext)>0) {
    message(sprintf("%-15s | %-30s | %-30s", m, v_int, v_ext))
  }
}

# ================= 5. Calibration Analysis =================
message("\n================================================================================")
message("5. Calibration Analysis (Coefficients)")
message("--------------------------------------------------------------------------------")
reg_cal_model <- lm(ASMI ~ predicted_asmi, data = df_test)
reg_slope <- coef(reg_cal_model)[2]; reg_int <- coef(reg_cal_model)[1]
message(sprintf("[Regression] Slope: %.3f (Ideal 1)", reg_slope))

bin_cal_model <- glm(Sarcopenia_GT ~ logit_prob, data = df_test, family = binomial)
bin_intercept <- coef(bin_cal_model)[1]; bin_slope <- coef(bin_cal_model)[2]     
message(sprintf("[Binary Risk] Slope: %.3f (Ideal 1), Intercept: %.3f (Ideal 0)", bin_slope, bin_intercept))


# ==============================================================================
# PART 3: VISUAL CORE (PUBLICATION FIGURES)
# ==============================================================================

message("\n================================================================================")
message("6. Generating Publication-Ready Figures (Using Captured Data)...")

prepare_pub_df <- function(df_in) {
  df_out <- df_in
  df_out$actual_asmi <- df_out$ASMI
  df_out$Low_muscle_mass <- as.integer(df_out$Sarcopenia_GT)
  return(df_out)
}
train_data_pub <- prepare_pub_df(df_internal)
test_data_pub <- prepare_pub_df(df_test)

# --- HELPER FUNCTIONS ---
calculate_classification_labels <- function(df) {
  y_true <- as.integer(df$Low_muscle_mass)
  y_pred <- ifelse(df$Gender == 0, ifelse(df$predicted_asmi < 7.0, 1, 0), ifelse(df$predicted_asmi < 5.4, 1, 0))
  y_score <- ifelse(df$Gender == 0, 7.0 - df$predicted_asmi, 5.4 - df$predicted_asmi)
  return(list(y_true = y_true, y_pred = y_pred, y_score = y_score))
}

calculate_classification_metrics <- function(y_true, y_pred, y_score) {
  cm <- table(Predicted = y_pred, Actual = y_true)
  if (nrow(cm) == 2 && ncol(cm) == 2) {
    tn <- cm[1, 1]; fn <- cm[1, 2]; fp <- cm[2, 1]; tp <- cm[2, 2]
  } else {
    tn <- fp <- fn <- tp <- 0
    if ("0" %in% rownames(cm) && "0" %in% colnames(cm)) tn <- cm["0", "0"]
    if ("0" %in% rownames(cm) && "1" %in% colnames(cm)) fn <- cm["0", "1"]
    if ("1" %in% rownames(cm) && "0" %in% colnames(cm)) fp <- cm["1", "0"]
    if ("1" %in% rownames(cm) && "1" %in% colnames(cm)) tp <- cm["1", "1"]
  }
  sensitivity <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  specificity <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  f1_score <- ifelse((tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), 0)
  ppv <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  npv <- ifelse((tn + fn) > 0, tn / (tn + fn), 0)
  roc_obj <- roc(y_true, y_score, quiet = TRUE)
  auc_value <- as.numeric(auc(roc_obj))
  return(list(auc = auc_value, sensitivity = sensitivity, specificity = specificity, accuracy = accuracy, f1_score = f1_score, ppv = ppv, npv = npv, tp = tp, tn = tn, fp = fp, fn = fn, roc_obj = roc_obj))
}

# --- FIGURE 1: SCATTER ---
figure1_scatter_comparison <- function(train_data, test_data, output_dir) {
  cat("   -> Figure 1...\n")
  train_r <- cor(train_data$actual_asmi, train_data$predicted_asmi, method = "pearson")
  train_r2 <- calc_r2_true(train_data$actual_asmi, train_data$predicted_asmi) 
  train_mae <- mean(abs(train_data$actual_asmi - train_data$predicted_asmi)); train_rmse <- sqrt(mean((train_data$actual_asmi - train_data$predicted_asmi)^2))
  
  test_r <- cor(test_data$actual_asmi, test_data$predicted_asmi, method = "pearson")
  test_r2 <- calc_r2_true(test_data$actual_asmi, test_data$predicted_asmi)   
  test_mae <- mean(abs(test_data$actual_asmi - test_data$predicted_asmi)); test_rmse <- sqrt(mean((test_data$actual_asmi - test_data$predicted_asmi)^2))
  
  train_x_range <- range(train_data$actual_asmi, na.rm = TRUE); train_y_range <- range(train_data$predicted_asmi, na.rm = TRUE)
  test_x_range <- range(test_data$actual_asmi, na.rm = TRUE); test_y_range <- range(test_data$predicted_asmi, na.rm = TRUE)
  
  create_scatter <- function(data, title, col_pt, col_line, n, r, r2, mae, rmse, x_rng, y_rng) {
    ggplot(data, aes(x = actual_asmi, y = predicted_asmi)) +
      geom_point(alpha = 0.4, color = col_pt, size = 2.5) +
      geom_abline(intercept = 0, slope = 1, color = "gray30", linetype = "dashed", size = 1) +
      geom_smooth(method = "lm", color = col_line, fill = col_pt, se = TRUE, alpha = 0.15, size = 1.2) +
      annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf, fill = NA, color = col_pt, size = 1.5) +
      annotate("text", x = x_rng[1], y = y_rng[2], label = sprintf("n = %d", n), hjust = 0, vjust = 0.8, size = 6.0, fontface = "bold", color = col_line, family = "sans") +
      annotate("text", x = x_rng[1], y = y_rng[2], label = sprintf("Pearson r = %.3f", r), hjust = 0, vjust = 2.4, size = 6.0, fontface = "bold", color = col_line, family = "sans") +
      annotate("text", x = x_rng[1], y = y_rng[2], label = sprintf("R² = %.3f", r2), hjust = 0, vjust = 4.0, size = 6.0, fontface = "bold", color = col_line, family = "sans") +
      annotate("text", x = x_rng[1], y = y_rng[2], label = sprintf("MAE = %.3f kg/m²", mae), hjust = 0, vjust = 5.6, size = 6.0, fontface = "bold", color = col_line, family = "sans") +
      annotate("text", x = x_rng[1], y = y_rng[2], label = sprintf("RMSE = %.3f kg/m²", rmse), hjust = 0, vjust = 7.2, size = 6.0, fontface = "bold", color = col_line, family = "sans") +
      labs(x = "Actual ASMI (kg/m²)", y = "Predicted ASMI (kg/m²)", title = title) +
      theme_bw(base_size = 11) +
      theme(plot.title = element_text(face = "bold", hjust = 0.0, size = 20, color = col_line), axis.title = element_text(face = "bold", size = 11), axis.text = element_text(size = 10), panel.grid.major = element_line(color = "gray90", size = 0.4), panel.grid.minor = element_blank(), panel.border = element_rect(color = "black", size = 1))
  }
  p1 <- create_scatter(train_data, "5-Fold Cross validation", "#E57373", "#D32F2F", nrow(train_data), train_r, train_r2, train_mae, train_rmse, train_x_range, train_y_range)
  p2 <- create_scatter(test_data, "Temporal validation", "#64B5F6", "#1976D2", nrow(test_data), test_r, test_r2, test_mae, test_rmse, test_x_range, test_y_range)
  ggsave(file.path(output_dir, "figure1_scatter_comparison.png"), ggdraw() + draw_plot(p1, x=0, y=0, width=0.5, height=1) + draw_plot(p2, x=0.5, y=0, width=0.5, height=1), width = 12, height = 6, dpi = 300)
}

# --- FIGURE 2: BLAND-ALTMAN ---
figure2_bland_altman <- function(train_data, test_data, output_dir) {
  cat("   -> Figure 2...\n")
  create_ba <- function(data, title, col_pt, col_line) {
    mean_val <- (data$actual_asmi + data$predicted_asmi) / 2
    diff_val <- data$actual_asmi - data$predicted_asmi
    mean_diff <- mean(diff_val); sd_diff <- sd(diff_val)
    upper <- mean_diff + 1.96 * sd_diff; lower <- mean_diff - 1.96 * sd_diff
    rng_x <- range(mean_val); rng_y <- range(diff_val)
    ggplot(data.frame(m=mean_val, d=diff_val), aes(x=m, y=d)) +
      geom_hline(yintercept=0, color="gray70", linetype="dotted", size=0.8) +
      geom_hline(yintercept=mean_diff, color=col_line, linetype="solid", size=1.2) +
      geom_hline(yintercept=upper, color=col_line, linetype="dashed", size=1) +
      geom_hline(yintercept=lower, color=col_line, linetype="dashed", size=1) +
      geom_point(alpha=0.4, color=col_pt, size=2.5) +
      annotate("rect", xmin=-Inf, xmax=Inf, ymin=lower, ymax=upper, fill=col_pt, alpha=0.08) +
      annotate("text", x=Inf, y=mean_diff, label=sprintf("Mean: %.3f", mean_diff), hjust=1.05, vjust=-0.5, size=4, fontface="bold", color=col_line) +
      annotate("text", x=Inf, y=upper, label=sprintf("+1.96 SD: %.3f", upper), hjust=1.05, vjust=-0.5, size=5, fontface="bold", color=col_line) +
      annotate("text", x=Inf, y=lower, label=sprintf("-1.96 SD: %.3f", lower), hjust=1.05, vjust=1.5, size=5, fontface="bold", color=col_line) +
      annotate("text", x=rng_x[1], y=rng_y[2], label=sprintf("n = %d", nrow(data)), hjust=0, vjust=0.8, size=4.5, fontface="bold", color=col_line) +
      annotate("text", x=rng_x[1], y=rng_y[2], label=sprintf("Mean bias = %.3f", mean_diff), hjust=0, vjust=2.4, size=4.5, fontface="bold", color=col_line) +
      annotate("text", x=rng_x[1], y=rng_y[2], label=sprintf("SD = %.3f", sd_diff), hjust=0, vjust=4.0, size=4.5, fontface="bold", color=col_line) +
      annotate("text", x=rng_x[1], y=rng_y[2], label=sprintf("95%% LoA: [%.3f, %.3f]", lower, upper), hjust=0, vjust=5.6, size=4.5, fontface="bold", color=col_line) +
      labs(x="Mean of Actual and Predicted ASMI (kg/m²)", y="Difference (Actual - Predicted)", title=title) +
      theme_bw(base_size = 11) +
      theme(plot.title = element_text(face="bold", hjust=0.0, size=20, color=col_line), axis.title = element_text(face="bold", size=11), axis.text = element_text(size=10), panel.grid.major = element_line(color="gray90", size=0.4), panel.grid.minor = element_blank(), panel.border = element_rect(color="black", size=1))
  }
  p1 <- create_ba(train_data, "5-Fold Cross validation", "#E57373", "#D32F2F")
  p2 <- create_ba(test_data, "Temporal validation", "#64B5F6", "#1976D2")
  ggsave(file.path(output_dir, "figure2_bland_altman_comparison.png"), ggdraw() + draw_plot(p1, x=0, y=0, width=0.5, height=1) + draw_plot(p2, x=0.5, y=0, width=0.5, height=1), width=12, height=6, dpi=300)
}

# --- FIGURE 3: ROC ---
figure3_roc_comparison <- function(train_data, test_data, output_dir) {
  cat("   -> Figure 3...\n")
  create_roc_plot <- function(data, title, col_line, col_fill) {
    labs <- calculate_classification_labels(data)
    mets <- calculate_classification_metrics(labs$y_true, labs$y_pred, labs$y_score)
    roc_df <- data.frame(spec=mets$roc_obj$specificities, sens=mets$roc_obj$sensitivities)
    thresh_spec <- ifelse((mets$fp+mets$tn)>0, mets$tn/(mets$fp+mets$tn), 0)
    thresh_sens <- ifelse((mets$tp+mets$fn)>0, mets$tp/(mets$tp+mets$fn), 0)
    txt_labels <- paste0('AUC \nSens. \nSpec. \nPPV \nNPV '); txt_values <- paste0(formatC(mets$auc, 3, format='f'), '\n', formatC(mets$sensitivity*100, 1, format='f'), '%\n', formatC(mets$specificity*100, 1, format='f'), '%\n', formatC(mets$ppv*100, 1, format='f'), '%\n', formatC(mets$npv*100, 1, format='f'), '%')
    ggplot(roc_df, aes(x=spec, y=sens)) + geom_line(colour=col_line, size=2) + theme_bw() + coord_equal() +
      labs(x='Specificity', y='Sensitivity', title=title) +
      annotate(geom="point", x=thresh_spec, y=thresh_sens, shape=21, size=5, fill=paste0(col_line, 'A0'), color='#000000') +
      annotate(geom="text", x=0.05, y=0.00, label=txt_labels, size=6, fontface=2, colour=col_line, hjust=0, vjust=0) +
      annotate(geom="text", x=0.60, y=0.00, label=txt_values, size=6, fontface=2, colour=col_line, hjust=1, vjust=0) +
      theme(plot.title=element_text(color=col_line, size=20, face="bold", hjust=0.0), axis.title=element_text(color="#000000", size=12), legend.position="none")
  }
  p1 <- create_roc_plot(train_data, "5-Fold Cross validation", '#F8766D', '#F8766DA0')
  p2 <- create_roc_plot(test_data, "Temporal validation", '#619CFF', '#619CFFA0')
  ggsave(file.path(output_dir, "figure3_roc_comparison.png"), plot_grid(p1, p2, ncol=2), width=10, height=5.5, dpi=300)
}

# --- FIGURE 4: METRICS (USING CAPTURED DATA) ---
figure4_metrics_table_fixed <- function(data_ready, output_dir) {
  cat("   -> Figure 4 (Using Pre-calculated Data)...\n")
  metrics_plot_list <- c("AUC-ROC", "Sensitivity", "Specificity", "Accuracy", "F1 Score", "PPV", "NPV")
  df <- data_ready %>% filter(Metric %in% metrics_plot_list)
  df$Metric <- factor(df$Metric, levels=rev(metrics_plot_list))
  write_csv(data_ready, file.path(output_dir, "figure4_metrics_table.csv"))
  col_train <- c("AUC-ROC"="#D32F2F", "Sensitivity"="#E57373", "Specificity"="#E57373", "Accuracy"="#EF9A9A", "F1 Score"="#EF9A9A", "PPV"="#FFCDD2", "NPV"="#FFCDD2")
  col_test <- c("AUC-ROC"="#1976D2", "Sensitivity"="#42A5F5", "Specificity"="#42A5F5", "Accuracy"="#64B5F6", "F1 Score"="#64B5F6", "PPV"="#90CAF9", "NPV"="#90CAF9")
  
  p1 <- ggplot(subset(df, Dataset=="Internal"), aes(x=Metric, y=Mean, fill=Metric)) +
    geom_bar(stat="identity", alpha=0.9, size=0.8) +
    labs(title='5-Fold Cross validation', x=NULL, y="Metric Value") +
    geom_text(aes(label=sprintf("%.3f", Mean)), hjust=-0.15, size=4.5, fontface="bold", color="#D32F2F") +
    scale_fill_manual(values=col_train) + coord_flip() + ylim(0, 1.1) + theme_minimal(base_size=13) +
    theme(plot.title=element_text(face="bold", hjust=0, size=20, color="#D32F2F"), axis.text.y=element_text(face="bold"), legend.position="none", panel.grid=element_blank())
  
  p2 <- ggplot(subset(df, Dataset=="External"), aes(x=Metric, y=Mean, fill=Metric)) +
    geom_bar(stat="identity", alpha=0.9, size=0.8) +
    labs(title='Temporal Validation', x=NULL, y="Metric Value") +
    geom_text(aes(label=sprintf("%.3f", Mean)), hjust=-0.15, size=4.5, fontface="bold", color="#1976D2") +
    scale_fill_manual(values=col_test) + coord_flip() + ylim(0, 1.1) + theme_minimal(base_size=13) +
    theme(plot.title=element_text(face="bold", hjust=0, size=16, color="#1976D2"), axis.text.y=element_text(face="bold"), legend.position="none", panel.grid=element_blank())
  
  ggsave(file.path(output_dir, "figure4_metrics_comparison.png"), ggdraw() + draw_plot(p1, x=0, y=0.08, width=0.5, height=0.9) + draw_plot(p2, x=0.5, y=0.08, width=0.5, height=0.9), width=14, height=6, dpi=300)
}

# --- FIGURE 6: SUBGROUP (USING CAPTURED DATA) ---
figure6_subgroup_analysis_fixed <- function(precalc_data, output_dir) {
  cat("   -> Figure 6 (Using Pre-calculated Data)...\n")
  write_csv(precalc_data, file.path(output_dir, "figure6_subgroup_metrics.csv"))
  df_plot <- precalc_data
  cols <- c("Male"="#D42300", "Female"="#003D9E", "<60"="#15ED32", "60-75"="#D6BD09", ">75"="#FA9057")
  
  plot_auc <- function(df, tit) {
    df$Subgroup <- factor(df$Subgroup, levels=names(cols))
    ggplot(df, aes(x=Subgroup, y=AUC_val, fill=Subgroup)) +
      geom_bar(stat="identity", position="dodge") +
      geom_errorbar(aes(ymin=AUC_CI_lower, ymax=AUC_CI_upper), width=.4) +
      annotate("text", x=1:nrow(df), y=0.02, label=df$AUC_text, size=5, color="white", angle=90, fontface=2, hjust=0) +
      theme_classic() + labs(title=tit, x='', y='AUC') +
      scale_fill_manual(values=cols) +
      scale_y_continuous(expand=c(0,0), limits=c(0,1.0), breaks=c(0,0.2,0.4,0.6,0.8,1)) +
      scale_x_discrete(labels=levels(df$Subgroup)) + 
      theme(plot.title=element_text(color="black", size=20, hjust=0), legend.position="none", axis.text.x=element_text(color="black", angle=45, hjust=1, size=12))
  }
  
  plot_pearson <- function(df, tit) {
    df$Subgroup <- factor(df$Subgroup, levels=names(cols))
    ggplot(df, aes(x=Subgroup, y=Pearson_r, fill=Subgroup)) +
      geom_bar(stat="identity", position="dodge") +
      geom_errorbar(aes(ymin=Pearson_CI_lower, ymax=Pearson_CI_upper), width=.4) +
      annotate("text", x=1:nrow(df), y=0.02, label=df$Pearson_text, size=5, color="white", angle=90, fontface=2, hjust=0) +
      theme_classic() + labs(title=tit, x='', y='Pearson r') +
      scale_fill_manual(values=cols) +
      scale_y_continuous(expand=c(0,0), limits=c(0,1.0), breaks=c(0,0.2,0.4,0.6,0.8,1)) +
      scale_x_discrete(labels=levels(df$Subgroup)) +
      theme(plot.title=element_text(color="black", size=20, hjust=0), legend.position="none", axis.text.x=element_text(color="black", angle=45, hjust=1, size=12))
  }
  
  r_train <- df_plot %>% filter(Set == "Internal")
  r_test <- df_plot %>% filter(Set == "External")
  p1 <- plot_pearson(r_train, "5-Fold Cross validation")
  p2 <- plot_pearson(r_test, "Temporal validation")
  p3 <- plot_auc(r_train, "5-Fold Cross validation")
  p4 <- plot_auc(r_test, "Temporal validation")
  
  ggsave(file.path(output_dir, "figure6_subgroup_analysis.png"), ggdraw() + draw_plot(p1, 0, 0.5, 0.5, 0.5) + draw_plot(p2, 0.5, 0.5, 0.5, 0.5) + draw_plot(p3, 0, 0, 0.5, 0.5) + draw_plot(p4, 0.5, 0, 0.5, 0.5), width=12, height=12, dpi=300)
}

# --- FIGURE 8: CONFUSION MATRIX ---
figure8_confusion_matrix <- function(train_data, test_data, output_dir) {
  cat("   -> Figure 8...\n")
  train_labels <- calculate_classification_labels(train_data); train_metrics <- calculate_classification_metrics(train_labels$y_true, train_labels$y_pred, train_labels$y_score)
  test_labels <- calculate_classification_labels(test_data); test_metrics <- calculate_classification_metrics(test_labels$y_true, test_labels$y_pred, test_labels$y_score)
  
  create_cm_df <- function(m) {
    data.frame(Predicted = factor(c("Normal", "Normal", "Sarcopenia", "Sarcopenia"), levels = c("Normal", "Sarcopenia")),
               Actual = factor(c("Normal", "Sarcopenia", "Normal", "Sarcopenia"), levels = c("Normal", "Sarcopenia")),
               Count = c(m$tn, m$fn, m$fp, m$tp),
               Label = c(sprintf("TN\n%d", m$tn), sprintf("FN\n%d", m$fn), sprintf("FP\n%d", m$fp), sprintf("TP\n%d", m$tp)))
  }
  train_cm_df <- create_cm_df(train_metrics); test_cm_df <- create_cm_df(test_metrics)
  train_cm_df$Percentage <- sprintf("%.1f%%", train_cm_df$Count / sum(train_cm_df$Count) * 100)
  test_cm_df$Percentage <- sprintf("%.1f%%", test_cm_df$Count / sum(test_cm_df$Count) * 100)
  
  p1 <- ggplot(train_cm_df, aes(x = Actual, y = Predicted, fill = Count)) + geom_tile(color = "white", size = 2) +
    geom_text(aes(label = Label), size = 7, fontface = "bold", color = "white", vjust = -0.0) +
    geom_text(aes(label = Percentage), size = 5, color = "white", vjust = 2.0) +
    scale_fill_gradient(low = "#FFCDD2", high = "#D32F2F", name = "Count", limits = c(0, max(c(train_cm_df$Count, test_cm_df$Count)))) +
    guides(fill = guide_colorbar(barheight = 10)) + labs(x = "Actual", y = "Predicted", title = "5-Fold Cross validation") +
    theme_minimal(base_size = 14) + theme(plot.title = element_text(face = "bold", hjust = 0.0, size = 20, color = "#D32F2F"), axis.title = element_text(face = "bold", size = 14), axis.text = element_text(size = 12, face = "bold"), panel.grid = element_blank()) + coord_equal()
  
  p2 <- ggplot(test_cm_df, aes(x = Actual, y = Predicted, fill = Count)) + geom_tile(color = "white", size = 2) +
    geom_text(aes(label = Label), size = 7, fontface = "bold", color = "white", vjust = -0.0) +
    geom_text(aes(label = Percentage), size = 5, color = "white", vjust = 2.0) +
    scale_fill_gradient(low = "#90CAF9", high = "#0D47A1", name = "Count", limits = c(0, max(test_cm_df$Count))) +
    guides(fill = guide_colorbar(barheight = 10)) + labs(x = "Actual", y = "Predicted", title = "Temporal validation") +
    theme_minimal(base_size = 14) + theme(plot.title = element_text(face = "bold", hjust = 0.0, size = 20, color = "#0D47A1"), axis.title = element_text(face = "bold", size = 14), axis.text = element_text(size = 12, face = "bold"), panel.grid = element_blank()) + coord_equal()
  
  ggsave(file.path(output_dir, "figure8_confusion_matrix.png"), ggdraw() + draw_plot(p1, x=0, y=0, width=0.5, height=1) + draw_plot(p2, x=0.5, y=0, width=0.5, height=1), width=14, height=7, dpi=300)
}

# --- EXECUTE PLOTS ---
figure1_scatter_comparison(train_data_pub, test_data_pub, OUTPUT_DIR)
figure2_bland_altman(train_data_pub, test_data_pub, OUTPUT_DIR)
figure3_roc_comparison(train_data_pub, test_data_pub, OUTPUT_DIR)
figure4_metrics_table_fixed(fig4_data_ready, OUTPUT_DIR)     # Uses captured metrics
figure6_subgroup_analysis_fixed(fig6_data_ready, OUTPUT_DIR) # Uses captured subgroups
figure8_confusion_matrix(train_data_pub, test_data_pub, OUTPUT_DIR)

# --- ADDITIONAL 01 STYLE PLOTS (Detailed Versions) ---
message("7. Generating Detailed Original 01 Style Plots (R1, R2, R3)...")

df <- df_test
COLOR_MAIN <- "#1976D2"; COLOR_FILL <- "#64B5F6"
theme_publication <- function() {
  theme_bw(base_size = 12) + theme(plot.title=element_text(face="bold", size=14), panel.grid.minor=element_blank())
}

# --- Figure R1: Regression Calibration ---
met <- calc_metrics_basic(df$ASMI, df$predicted_asmi)
range_lim <- range(c(df$ASMI, df$predicted_asmi), na.rm=T)
text_x <- range_lim[1]; text_y <- range_lim[2]
line_height_spacing <- (range_lim[2] - range_lim[1]) * 0.06 

p1 <- ggplot(df, aes(x = predicted_asmi, y = ASMI)) +
  geom_abline(intercept=0, slope=1, color="gray40", linetype="dashed", linewidth=1) +
  geom_smooth(method="lm", color=COLOR_MAIN, fill=COLOR_FILL, alpha=0.2) +
  geom_point(alpha=0.5, color=COLOR_FILL, size=2) +
  annotate("text", x=text_x, y=text_y, label=sprintf("n = %d", nrow(df)), color=COLOR_MAIN, fontface="bold", hjust=0, size=5) +
  annotate("text", x=text_x, y=text_y - line_height_spacing, label=sprintf("R² = %.3f", met['R2']), color=COLOR_MAIN, fontface="bold", hjust=0, size=5) +
  annotate("text", x=text_x, y=text_y - 2*line_height_spacing, label=sprintf("Slope = %.3f", reg_slope), color=COLOR_MAIN, fontface="bold", hjust=0, size=5) +
  annotate("text", x=text_x, y=text_y - 3*line_height_spacing, label=sprintf("Intercept = %.3f", reg_int), color=COLOR_MAIN, fontface="bold", hjust=0, size=5) +
  coord_fixed(ratio=1, xlim=range_lim, ylim=range_lim) +
  labs(title="Regression Calibration", x="Predicted ASMI", y="Observed ASMI") +
  theme_publication()
ggsave(file.path(OUTPUT_DIR, "Figure_R1_Regression_Calibration.png"), plot=p1, width=6, height=6, dpi=300)

# --- Figure R2: Binary Calibration (Added Brier Score) ---
brier_overall <- mean((df$prob_sarcopenia - df$Sarcopenia_GT)^2)
cal_data <- df %>% mutate(bin = ntile(prob_sarcopenia, 10)) %>% group_by(bin) %>%
  summarise(mean_pred = mean(prob_sarcopenia), mean_obs = mean(Sarcopenia_GT), se_obs = sqrt((mean_obs*(1-mean_obs))/n()), n=n())

p2 <- ggplot(cal_data, aes(x=mean_pred, y=mean_obs)) +
  geom_abline(intercept=0, slope=1, linetype="dashed", color="gray40") +
  geom_errorbar(aes(ymin=mean_obs-1.96*se_obs, ymax=mean_obs+1.96*se_obs), width=0.02, color=COLOR_MAIN) +
  geom_line(color=COLOR_MAIN) +
  geom_point(size=3, color="white", fill=COLOR_MAIN, shape=21, stroke=1.5) +
  annotate("text", x=0.01, y=0.9, label=sprintf("Slope = %.3f\nIntercept = %.3f\nBrier Score = %.3f", bin_slope, bin_intercept, brier_overall), color=COLOR_MAIN, size=5.5, hjust=0, fontface="bold", lineheight=1.2) +
  coord_fixed(ratio=1, xlim=c(0,1), ylim=c(0,1)) +
  labs(title="Risk Calibration", x="Predicted Probability", y="Observed Proportion") +
  theme_publication()
ggsave(file.path(OUTPUT_DIR, "Figure_R2_Risk_Calibration.png"), plot=p2, width=6, height=6, dpi=300)

# --- Figure R3: DCA (Final Polished) ---
thresholds <- seq(0.01, 0.99, by=0.01); n <- nrow(df); prev <- mean(df$Sarcopenia_GT)
nb_model <- sapply(thresholds, function(pt) {
  tp <- sum(df$prob_sarcopenia >= pt & df$Sarcopenia_GT == 1)
  fp <- sum(df$prob_sarcopenia >= pt & df$Sarcopenia_GT == 0)
  (tp/n) - (fp/n)*(pt/(1-pt))
})
nb_all <- prev - (1-prev)*(thresholds/(1-thresholds))
max_nb_val <- max(nb_model, na.rm=TRUE)
tol <- 1e-4
is_useful <- (nb_model > (nb_all + tol)) & (nb_model > (tol))
useful_thresh_vals <- thresholds[is_useful]
range_text <- if(length(useful_thresh_vals) > 0) sprintf("%.2f to %.2f", min(useful_thresh_vals), max(useful_thresh_vals)) else "None"
annot_text <- paste0("Prevalence: ", percent(prev, accuracy=0.1), "\n", "Max Net Benefit: ", round(max_nb_val, 3), "\n", "Useful Range: ", range_text)

dca_df <- data.frame(Threshold = rep(thresholds, 2), Net_Benefit = c(nb_model, nb_all), Model = factor(rep(c("AI Model", "Treat All"), each=length(thresholds))))
dca_df_plot <- dca_df %>% filter(Net_Benefit >= -0.1)
y_max_limit <- max(c(nb_model, prev)) * 1.15

p3 <- ggplot(dca_df_plot, aes(x=Threshold, y=Net_Benefit)) +
  geom_hline(yintercept=0, color="black", linewidth=0.6) + 
  geom_line(aes(color=Model, linetype=Model), linewidth=1.2) +
  scale_color_manual(values=c("AI Model"=COLOR_MAIN, "Treat All"="gray60")) +
  scale_linetype_manual(values=c("AI Model"="solid", "Treat All"="dashed")) +
  scale_x_continuous(labels=percent_format(accuracy=1), limits=c(0, 1), breaks=seq(0, 1, 0.1)) +
  coord_cartesian(ylim=c(-0.05, y_max_limit)) +
  annotate("text", x=0.60, y=y_max_limit * 0.95, label=annot_text, hjust=0, vjust=1, size=5, color=COLOR_MAIN, fontface="bold", lineheight=1.2) +
  labs(title="Decision Curve Analysis", subtitle="Standardized Net Benefit", x="Threshold Probability", y="Net Benefit") +
  theme_publication() + theme(legend.position = c(0.2, 0.3), legend.key.width = unit(1.2, "cm"), legend.background = element_blank(), legend.box.background = element_blank(), legend.text.align = 0)
ggsave(file.path(OUTPUT_DIR, "Figure_R3_DCA.png"), plot=p3, width=7, height=5.5, dpi=300)

message("\n✅ 執行完畢！所有 Console 輸出與 圖表 (Figures) 數據已保證 100% 一致。")
message("📁 結果輸出目錄: ", OUTPUT_DIR)