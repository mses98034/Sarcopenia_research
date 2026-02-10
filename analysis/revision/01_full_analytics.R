# ==============================================================================
# Script Name: 07_final_analysis_complete_v14.R
# Description: 全流程整合分析 (最終發表定稿版 - Seed Protected & Stratified)
#              1. [Methodology] Age Subgroup definition reverted to original 
#                 submission logic (<= 75 vs > 75).
#              2. [Reproducibility] Explicit set.seed(42) maintained.
#              3. [Metric] Brier Score & AUC with 95% CIs via Stratified Bootstrap.
# ==============================================================================

# ================= 0. 環境與套件設定 =================
my_repo <- "https://cloud.r-project.org"

ensure_package <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    message(paste("安裝套件:", pkg))
    install.packages(pkg, repos = my_repo)
    if (!require(pkg, character.only = TRUE)) stop(paste("無法安裝:", pkg))
  }
}

# 載入必要套件
ensure_package("ggplot2")
ensure_package("dplyr")
ensure_package("cowplot")
ensure_package("scales")
ensure_package("boot")  # 用於 Bootstrap
ensure_package("pROC")  # 用於 AUC 計算

library(ggplot2)
library(dplyr)
library(cowplot)
library(scales)
library(boot)
library(pROC)

# ================= 1. 路徑設定 =================
# 請確認這些路徑與您的檔案結構一致
RAW_TRAIN_PATH  <- '../../data/train.csv'
RAW_TEST_PATH   <- '../../data/test.csv'

# Training 結果路徑
TRAIN_META_PATH <- '../../log/ASMI_Regression_ImplantCleaned/ResNetFusionAttentionNetRegression/0_run_ASMI-Reg_2025-10-03_16-55-03/csv_data/patient_data.csv'
VAL_PRED_PATH   <- '../../log/ASMI_Regression_ImplantCleaned/ResNetFusionAttentionNetRegression/0_run_ASMI-Reg_2025-10-03_16-55-03/csv_data/validation_predictions.csv'

# Testing 結果路徑
TEST_META_PATH  <- '../../log/test/2025-10-04_22-54-04/csv_data/test_patient_data.csv'
TEST_PRED_PATH  <- '../../log/test/2025-10-04_22-54-04/csv_data/test_predictions.csv'

# [Global Seed] 設定全域隨機種子
set.seed(42)
N_BOOTSTRAPS <- 1000

# ================= 2. 輔助函數 (統計與格式化) =================

# [關鍵函數] Pearson R2 計算
calc_r2_pearson <- function(y_true, y_pred) {
  r <- cor(y_true, y_pred, use = "complete.obs", method = "pearson")
  return(r^2)
}

# [關鍵函數] 計算 Risk Score (Distance from Threshold)
calculate_risk_score <- function(gender, pred_asmi) {
  return(ifelse(gender == 0, 7.0 - pred_asmi, 5.4 - pred_asmi))
}

# 向量化的分類函數 (AWGS 2019)
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
  r2 <- calc_r2_pearson(y_true, y_pred)
  r <- cor(y_true, y_pred, use="complete.obs", method="pearson")
  return(c(MAE=mae, R2=r2, Pearson=r))
}

# ================= 3. 資料讀取與前處理 =================
message("\n=== 正在讀取並整理資料 ===")

if(!file.exists(TRAIN_META_PATH)) stop("找不到 Train Meta 檔案")

df_train_meta <- read.csv(TRAIN_META_PATH)
df_test_meta <- read.csv(TEST_META_PATH)
df_test_pred <- read.csv(TEST_PRED_PATH)
df_val_pred  <- read.csv(VAL_PRED_PATH)

# --- Test Set ---
df_test <- merge(df_test_meta, df_test_pred[, c('UID', 'predicted_asmi')], by='UID')

# Time period 合併
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
if(!"Low_muscle_mass" %in% names(df_train_meta)) stop("Train Meta CSV 中找不到 'Low_muscle_mass' 欄位！")
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

# --- Check Integrity ---
check_integrity <- function(df, name, check_preds=FALSE) {
  base_cols <- c('Age', 'Gender', 'Height', 'Weight', 'BMI', 'ASMI', 'Sarcopenia_GT')
  target_cols <- if(check_preds) c(base_cols, 'predicted_asmi') else base_cols
  missing_vals <- colSums(is.na(df[, target_cols]))
  if(any(missing_vals > 0)) {
    stop(sprintf("錯誤: %s 資料集包含缺失值 (NA)！", name))
  } else {
    message(sprintf("✅ %s 資料完整性檢查通過。", name))
  }
}
check_integrity(df_train_meta, "Training Meta Set", check_preds=FALSE)
check_integrity(df_internal, "Internal Validation Set", check_preds=TRUE)
check_integrity(df_test, "External Test Set", check_preds=TRUE)

# --- Probability Calculation for Subgroup & Calibration ---
message("\n=== 正在計算預測機率 (Risk Score -> Probability) ===")
df_internal$risk_score <- calculate_risk_score(df_internal$Gender, df_internal$predicted_asmi)
df_test$risk_score <- calculate_risk_score(df_test$Gender, df_test$predicted_asmi)

# Train Logistic Regression on Internal Set
risk_model <- glm(Sarcopenia_GT ~ risk_score, data = df_internal, family = binomial)

# Predict on Test Set
df_test$prob_sarcopenia <- predict(risk_model, df_test, type = "response")
eps <- 1e-9
df_test$prob_clipped <- pmax(pmin(df_test$prob_sarcopenia, 1-eps), eps)
df_test$logit_prob <- log(df_test$prob_clipped / (1 - df_test$prob_clipped))

# ================= 4. 基本 Console 分析報告 =================

message("\n================================================================================")
message("1. Table 1: Baseline characteristics")
message("--------------------------------------------------------------------------------")
message(sprintf("%-20s | %-25s | %-25s | %-10s | %-10s", "Variable", "Dev Set", "Test Set", "P-value", "SMD"))
message("--------------------------------------------------------------------------------")

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

# ================= 2. Linear Regression Baseline comparison =================
message("\n================================================================================")
message("2. Linear Regression Baseline comparison (Comparison of AI vs LR)")
message("--------------------------------------------------------------------------------")

model_nb <- lm(ASMI ~ Age + Gender + Height + Weight, data=df_train_meta)
model_bmi <- lm(ASMI ~ Age + Gender + Height + Weight + BMI, data=df_train_meta)

pred_lr_nb <- predict(model_nb, df_test)
pred_lr_bmi <- predict(model_bmi, df_test)

met_ai <- calc_metrics_basic(df_test$ASMI, df_test$predicted_asmi)
met_lr_nb <- calc_metrics_basic(df_test$ASMI, pred_lr_nb)
met_lr_bmi <- calc_metrics_basic(df_test$ASMI, pred_lr_bmi)

message("---- Performance Comparison ----")
message(sprintf("AI Model            : MAE=%.4f, R2=%.4f, Pearson=%.4f", met_ai[1], met_ai[2], met_ai[3]))
message(sprintf("LR model (No BMI)   : MAE=%.4f, R2=%.4f, Pearson=%.4f", met_lr_nb[1], met_lr_nb[2], met_lr_nb[3]))
message(sprintf("LR model (With BMI) : MAE=%.4f, R2=%.4f, Pearson=%.4f", met_lr_bmi[1], met_lr_bmi[2], met_lr_bmi[3]))

# ================= 3. Subgroup & Sensitivity Analysis (SEED PROTECTED) =================
# [SEED RESET] 確保這一段 Subgroup Bootstrap 分析每次執行結果都一樣
set.seed(42) 

message("\n================================================================================")
message("3. Subgroup Analysis (Bootstrapped 95% CI for AUC & Brier)")
message("--------------------------------------------------------------------------------")
message(sprintf("%-18s | %-4s | %-6s | %-6s | %-22s | %-22s", "Subgroup", "N", "MAE", "R2", "AUC (95% CI)", "Brier (95% CI)"))
message("--------------------------------------------------------------------------------")

# 定義 Subgroup 專用的 Bootstrap 統計函數 (AUC & Brier)
subgroup_boot_fn <- function(data, indices) {
  d <- data[indices, ]
  
  # AUC Calculation
  risk_score <- calculate_risk_score(d$Gender, d$predicted_asmi)
  auc_val <- NA
  tryCatch({
    if (length(unique(d$Sarcopenia_GT)) > 1) {
      auc_val <- as.numeric(auc(roc(d$Sarcopenia_GT, risk_score, quiet = TRUE)))
    }
  }, error = function(e) { auc_val <<- NA })
  
  # Brier Score Calculation
  brier_val <- NA
  if("prob_sarcopenia" %in% names(d)) {
    brier_val <- mean((d$prob_sarcopenia - d$Sarcopenia_GT)^2)
  }
  
  return(c(auc_val, brier_val))
}

print_subgroup_row <- function(name, df_sub) {
  if(nrow(df_sub) < 5) {
    message(sprintf("%-18s | %-4d | %-6s | %-6s | %-22s | %-22s", name, nrow(df_sub), "NA", "NA", "NA", "NA"))
    return()
  }
  
  mae <- mean(abs(df_sub$ASMI - df_sub$predicted_asmi))
  r2 <- calc_r2_pearson(df_sub$ASMI, df_sub$predicted_asmi)
  
  # [MODIFIED]: Added 'strata = df_sub$Sarcopenia_GT' for Stratified Sampling
  boot_res <- tryCatch({
    boot(data = df_sub, statistic = subgroup_boot_fn, R = 1000, strata = df_sub$Sarcopenia_GT) 
  }, error = function(e) return(NULL))
  
  format_ci <- function(boot_obj, idx) {
    val <- boot_obj$t0[idx]
    if(is.na(val)) return("NA")
    
    ci_str <- "(NA)"
    tryCatch({
      ci <- boot.ci(boot_obj, type = "perc", index = idx)
      low <- ci$percent[4]; high <- ci$percent[5]
      ci_str <- sprintf("(%.3f-%.3f)", low, high)
    }, error=function(e){})
    return(sprintf("%.3f %s", val, ci_str))
  }
  
  if(!is.null(boot_res)) {
    auc_str <- format_ci(boot_res, 1)   # Index 1 is AUC
    brier_str <- format_ci(boot_res, 2) # Index 2 is Brier
  } else {
    auc_str <- "Bootstrap Fail"
    brier_str <- "Bootstrap Fail"
  }
  
  message(sprintf("%-18s | %-4d | %.3f  | %.3f  | %-22s | %-22s", 
                  name, nrow(df_sub), mae, r2, auc_str, brier_str))
}

# Subgroups Loop
g_list <- list("Male"=0, "Female"=1)
for(g_name in names(g_list)) {
  sub_df <- df_test %>% filter(Gender == g_list[[g_name]])
  print_subgroup_row(paste("Gender:", g_name), sub_df)
}

# [回歸原始定義]：Age <= 75 (包含 75) vs Age > 75
age_groups <- list("Age <= 75" = df_test %>% filter(Age <= 75),
                   "Age > 75"  = df_test %>% filter(Age > 75))
for(a_name in names(age_groups)) {
  print_subgroup_row(a_name, age_groups[[a_name]])
}

# Sensitivity Analysis (Time)
if(!all(is.na(df_test$Time_period))) {
  df_test$Time_period <- as.numeric(df_test$Time_period)
  message("--------------------------------------------------------------------------------")
  message("[Sensitivity Analysis] Time Period from Raw Data")
  
  time_groups <- list(
    "Time < 30 days"    = df_test %>% filter(Time_period < 30),
    "Time 30-90 days"   = df_test %>% filter(Time_period >= 30 & Time_period <= 90)
  )
  for(t_name in names(time_groups)) {
    sub_df <- time_groups[[t_name]]
    if(nrow(sub_df) > 0) {
      print_subgroup_row(t_name, sub_df)
    } else {
      message(sprintf("%-18s | 0    | N/A", t_name))
    }
  }
} else {
  message("--------------------------------------------------------------------------------")
  message("[Sensitivity Analysis] Time_period column is empty or NA. Skipped.")
}


# ================= 5. Bootstrap 95% CI (SEED PROTECTED) =================
# [SEED RESET] 確保這一段 Main Analysis Bootstrap 分析每次執行結果都一樣
set.seed(42)

message("\n================================================================================")
message("4. Bootstrap 95% Confidence Intervals (Running Main Analysis...)")
message("--------------------------------------------------------------------------------")

boot_stats_fn <- function(data, indices) {
  d <- data[indices, ]
  
  y_asmi_true <- d$ASMI
  y_asmi_pred <- d$predicted_asmi
  y_class_true <- d$Sarcopenia_GT 
  y_class_pred <- predict_sarcopenia_class(d$Gender, d$predicted_asmi)
  
  mae <- mean(abs(y_asmi_true - y_asmi_pred))
  rmse <- sqrt(mean((y_asmi_true - y_asmi_pred)^2))
  r2_val <- calc_r2_pearson(y_asmi_true, y_asmi_pred)
  pearson_r <- cor(y_asmi_true, y_asmi_pred, method = "pearson")
  
  tp <- sum(y_class_true == 1 & y_class_pred == 1)
  tn <- sum(y_class_true == 0 & y_class_pred == 0)
  fp <- sum(y_class_true == 0 & y_class_pred == 1)
  fn <- sum(y_class_true == 1 & y_class_pred == 0)
  
  sens <- ifelse((tp+fn)>0, tp/(tp+fn), 0)
  spec <- ifelse((tn+fp)>0, tn/(tn+fp), 0)
  accuracy <- mean(y_class_true == y_class_pred)
  
  risk_score <- calculate_risk_score(d$Gender, d$predicted_asmi)
  auc_val <- NA
  tryCatch({
    if (length(unique(y_class_true)) > 1) {
      auc_val <- as.numeric(auc(roc(y_class_true, risk_score, quiet = TRUE)))
    }
  }, error = function(e) { auc_val <<- NA })
  
  return(c(mae, rmse, r2_val, pearson_r, auc_val, sens, spec, accuracy))
}

run_bootstrap_analysis <- function(df, tag) {
  df_boot <- df %>% select(ASMI, predicted_asmi, Gender, Sarcopenia_GT)
  cat(sprintf("-> Running Bootstrap for %s (N=%d)...\n", tag, nrow(df_boot)))
  
  # [MODIFIED]: Added 'strata = df_boot$Sarcopenia_GT' for Stratified Sampling
  boot_out <- boot(data = df_boot, statistic = boot_stats_fn, R = N_BOOTSTRAPS, strata = df_boot$Sarcopenia_GT)
  
  metric_names <- c("MAE", "RMSE", "R2", "Pearson r", "AUC", "Sensitivity", "Specificity", "Accuracy")
  results <- data.frame(Metric = metric_names, Result = NA)
  
  for(i in 1:length(metric_names)) {
    val <- boot_out$t0[i]
    ci_str <- "(NA)"
    tryCatch({
      ci <- boot.ci(boot_out, type = "perc", index = i)
      low <- ci$percent[4]; high <- ci$percent[5]
      ci_str <- sprintf("(%.3f-%.3f)", low, high)
    }, error=function(e){})
    results$Result[i] <- sprintf("%.3f %s", val, ci_str)
  }
  return(results)
}

res_internal <- run_bootstrap_analysis(df_internal, "Internal Validation")
res_external <- run_bootstrap_analysis(df_test, "External Validation")

message("\n--------------------------------------------------------------------------------")
message(" FINAL SUMMARY TABLE FOR MANUSCRIPT (Mean + 95% CI)")
message("--------------------------------------------------------------------------------")
message(sprintf("%-15s | %-30s | %-30s", "Metric", "Internal Val (5-Fold)", "External Val"))
message(paste(rep("-", 80), collapse=""))

for(i in 1:nrow(res_internal)) {
  m <- res_internal$Metric[i]
  v_int <- res_internal$Result[i]
  v_ext <- res_external$Result[i]
  message(sprintf("%-15s | %-30s | %-30s", m, v_int, v_ext))
}

# ================= 6. Calibration Analysis =================
message("\n================================================================================")
message("5. Calibration Analysis (Coefficients)")
message("--------------------------------------------------------------------------------")

# Regression Calibration
reg_cal_model <- lm(ASMI ~ predicted_asmi, data = df_test)
reg_slope <- coef(reg_cal_model)[2]
reg_int   <- coef(reg_cal_model)[1]
message(sprintf("[Regression] Slope: %.3f (Ideal 1)", reg_slope))

# Binary Risk Calibration
bin_cal_model <- glm(Sarcopenia_GT ~ logit_prob, data = df_test, family = binomial)
bin_intercept <- coef(bin_cal_model)[1] 
bin_slope <- coef(bin_cal_model)[2]     

message(sprintf("[Binary Risk] Slope: %.3f (Ideal 1), Intercept: %.3f (Ideal 0)", bin_slope, bin_intercept))

# ================= 7. 繪圖 (Figures) =================
message("\n================================================================================")
message("6. Generating Figures...")

df <- df_test
COLOR_MAIN <- "#1976D2"; COLOR_FILL <- "#64B5F6"
theme_publication <- function() {
  theme_bw(base_size = 12) + theme(plot.title=element_text(face="bold", size=14), panel.grid.minor=element_blank())
}

# --- Figure R1: Regression Calibration ---
met <- calc_metrics_basic(df$ASMI, df$predicted_asmi)
range_lim <- range(c(df$ASMI, df$predicted_asmi), na.rm=T)
text_x <- range_lim[1]
text_y <- range_lim[2]
line_height_spacing <- (range_lim[2] - range_lim[1]) * 0.06 

p1 <- ggplot(df, aes(x = predicted_asmi, y = ASMI)) +
  geom_abline(intercept=0, slope=1, color="gray40", linetype="dashed", linewidth=1) +
  geom_smooth(method="lm", color=COLOR_MAIN, fill=COLOR_FILL, alpha=0.2) +
  geom_point(alpha=0.5, color=COLOR_FILL, size=2) +
  annotate("text", x=text_x, y=text_y, label=sprintf("n = %d", nrow(df)), 
           color=COLOR_MAIN, fontface="bold", hjust=0, size=5) +
  annotate("text", x=text_x, y=text_y - line_height_spacing, label=sprintf("R² = %.3f", met['R2']), 
           color=COLOR_MAIN, fontface="bold", hjust=0, size=5) +
  annotate("text", x=text_x, y=text_y - 2*line_height_spacing, label=sprintf("Slope = %.3f", reg_slope), 
           color=COLOR_MAIN, fontface="bold", hjust=0, size=5) +
  annotate("text", x=text_x, y=text_y - 3*line_height_spacing, label=sprintf("Intercept = %.3f", reg_int), 
           color=COLOR_MAIN, fontface="bold", hjust=0, size=5) +
  coord_fixed(ratio=1, xlim=range_lim, ylim=range_lim) +
  labs(title="Regression Calibration", x="Predicted ASMI", y="Observed ASMI") +
  theme_publication()

ggsave("Figure_R1_Regression_Calibration.png", plot=p1, width=6, height=6, dpi=300)

# --- Figure R2: Binary Calibration (Added Brier Score) ---
# 計算 Overall Brier Score
brier_overall <- mean((df$prob_sarcopenia - df$Sarcopenia_GT)^2)

cal_data <- df %>% mutate(bin = ntile(prob_sarcopenia, 10)) %>% group_by(bin) %>%
  summarise(mean_pred = mean(prob_sarcopenia), mean_obs = mean(Sarcopenia_GT),
            se_obs = sqrt((mean_obs*(1-mean_obs))/n()), n=n())

p2 <- ggplot(cal_data, aes(x=mean_pred, y=mean_obs)) +
  geom_abline(intercept=0, slope=1, linetype="dashed", color="gray40") +
  geom_errorbar(aes(ymin=mean_obs-1.96*se_obs, ymax=mean_obs+1.96*se_obs), width=0.02, color=COLOR_MAIN) +
  geom_line(color=COLOR_MAIN) +
  geom_point(size=3, color="white", fill=COLOR_MAIN, shape=21, stroke=1.5) +
  
  # 修改：加入 Brier Score
  annotate("text", x=0.01, y=0.9, 
           label=sprintf("Slope = %.3f\nIntercept = %.3f\nBrier Score = %.3f", bin_slope, bin_intercept, brier_overall), 
           color=COLOR_MAIN, size=5.5, hjust=0, fontface="bold", lineheight=1.2) +
           
  coord_fixed(ratio=1, xlim=c(0,1), ylim=c(0,1)) +
  labs(title="Risk Calibration", x="Predicted Probability", y="Observed Proportion") +
  theme_publication()

ggsave("Figure_R2_Risk_Calibration.png", plot=p2, width=6, height=6, dpi=300)

# --- Figure R3: DCA (Final Polished) ---
message("正在繪製 Figure R3 (DCA) - 最終定稿版...")
thresholds <- seq(0.01, 0.99, by=0.01)
n <- nrow(df)
prev <- mean(df$Sarcopenia_GT)

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
if(length(useful_thresh_vals) > 0) {
  range_text <- sprintf("%.2f to %.2f", min(useful_thresh_vals), max(useful_thresh_vals))
} else { range_text <- "None" }

annot_text <- paste0("Prevalence: ", percent(prev, accuracy=0.1), "\n",
                     "Max Net Benefit: ", round(max_nb_val, 3), "\n",
                     "Useful Range: ", range_text)

dca_df <- data.frame(Threshold = rep(thresholds, 2),
                     Net_Benefit = c(nb_model, nb_all),
                     Model = factor(rep(c("AI Model", "Treat All"), each=length(thresholds))))
dca_df_plot <- dca_df %>% filter(Net_Benefit >= -0.1)

y_max_limit <- max(c(nb_model, prev)) * 1.15

p3 <- ggplot(dca_df_plot, aes(x=Threshold, y=Net_Benefit)) +
  geom_hline(yintercept=0, color="black", linewidth=0.6) + 
  geom_line(aes(color=Model, linetype=Model), linewidth=1.2) +
  scale_color_manual(values=c("AI Model"=COLOR_MAIN, "Treat All"="gray60")) +
  scale_linetype_manual(values=c("AI Model"="solid", "Treat All"="dashed")) +
  scale_x_continuous(labels=percent_format(accuracy=1), limits=c(0, 1), breaks=seq(0, 1, 0.1)) +
  coord_cartesian(ylim=c(-0.05, y_max_limit)) +
  
  annotate("text", x=0.60, y=y_max_limit * 0.95, label=annot_text, 
           hjust=0, vjust=1, size=5, color=COLOR_MAIN, fontface="bold", lineheight=1.2) +
  
  labs(title="Decision Curve Analysis", subtitle="Standardized Net Benefit", x="Threshold Probability", y="Net Benefit") +
  theme_publication() +
  theme(legend.position = c(0.2, 0.3), 
        legend.key.width = unit(1.2, "cm"),
        legend.background = element_blank(), 
        legend.box.background = element_blank(),
        legend.text.align = 0)

ggsave("Figure_R3_DCA.png", plot=p3, width=7, height=5.5, dpi=300)

message("\n✅ 執行完畢！")