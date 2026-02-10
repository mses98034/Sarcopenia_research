#!/usr/bin/env Rscript
################################################################################
# Publication-Ready Figures Generator for ASMI Regression Analysis (R Version)
# Generates comparative figures between cross-validation and external test set
#
# Fixes applied:
# 1. Added set.seed(42) for reproducibility.
# 2. Replaced pROC::ci.auc with boot::boot(..., strata=...) for strict consistency.
# 3. Fixed dplyr (cur_data) and pROC (progress) warnings.
# 4. Kept original Age definitions (<60, 60-75, >75) and 5-color palette.
# 5. Fixed "Unknown column 'Subgroup'" error by adding rename step.
################################################################################

# [CRITICAL] Set seed for reproducibility
set.seed(42)

# Required packages
required_packages <- c(
  "ggplot2",      # Main plotting
  "cowplot",      # Combine plots
  "dplyr",        # Data manipulation
  "tidyr",        # Data reshaping
  "readr",        # Fast CSV reading
  "pROC",         # ROC curves (for AUC calculation only)
  "scales",       # Color scales
  "gridExtra",    # Additional grid functions
  "optparse",     # Command-line arguments
  "boot"          # Explicitly using boot for consistency with Script A
)

# Install missing packages
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  cat("Installing missing packages:", paste(new_packages, collapse=", "), "\n")
  install.packages(new_packages, repos = "https://cloud.r-project.org/")
}

# Load packages
suppressPackageStartupMessages({
  library(ggplot2)
  library(cowplot)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(pROC)
  library(scales)
  library(gridExtra)
  library(optparse)
  library(boot)
})

################################################################################
# Parse command-line arguments
################################################################################

option_list <- list(
  make_option(c("--train-log"), type="character", default=NULL,
              help="Training log directory path [required]", metavar="PATH"),
  make_option(c("--test-log"), type="character", default=NULL,
              help="Test log directory path [required]", metavar="PATH"),
  make_option(c("--output"), type="character", default="results/publication_figures",
              help="Output directory [default: %default]", metavar="PATH"),
  make_option(c("--type"), type="character", default="all",
              help="Figure type: all, scatter, bland_altman, roc, table, subgroup [default: %default]"),
  make_option(c("--dpi"), type="integer", default=300,
              help="DPI for output images [default: %default]"),
  make_option(c("--format"), type="character", default="png",
              help="Output format: png, pdf, svg [default: %default]")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Validate required arguments
if (is.null(opt$`train-log`) || is.null(opt$`test-log`)) {
  print_help(opt_parser)
  stop("Both --train-log and --test-log are required", call.=FALSE)
}

train_log_dir <- opt$`train-log`
test_log_dir <- opt$`test-log`
output_dir <- opt$output
figure_type <- opt$type
output_dpi <- opt$dpi
output_format <- opt$format

# Create output directory
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("\n", rep("=", 60), "\n", sep="")
cat("PUBLICATION FIGURE GENERATOR (R VERSION - FINAL FIX v3)\n")
cat(rep("=", 60), "\n", sep="")
cat("Training log:", train_log_dir, "\n")
cat("Test log:", test_log_dir, "\n")
cat("Output directory:", output_dir, "\n")
cat(rep("=", 60), "\n\n", sep="")

################################################################################
# Data Loading Functions
################################################################################

load_training_data <- function(log_dir) {
  cat("📂 Loading training data...\n")
  val_pred <- read_csv(file.path(log_dir, "csv_data/validation_predictions.csv"), show_col_types = FALSE)
  patient_data <- read_csv(file.path(log_dir, "csv_data/patient_data.csv"), show_col_types = FALSE)
  merged <- left_join(val_pred, patient_data, by = "UID")
  cat("   ✅ Training data loaded:", nrow(merged), "samples\n")
  return(merged)
}

load_test_data <- function(log_dir) {
  cat("📂 Loading test data...\n")
  test_pred <- read_csv(file.path(log_dir, "csv_data/test_predictions.csv"), show_col_types = FALSE)
  test_patient <- read_csv(file.path(log_dir, "csv_data/test_patient_data.csv"), show_col_types = FALSE)
  merged <- left_join(test_pred, test_patient, by = "UID")
  cat("   ✅ Test data loaded:", nrow(merged), "samples\n")
  return(merged)
}

################################################################################
# Classification Helper Functions
################################################################################

calculate_classification_labels <- function(df) {
  if (!"Low_muscle_mass" %in% colnames(df)) stop("Low_muscle_mass column not found")
  
  y_true <- as.integer(df$Low_muscle_mass)
  y_pred <- ifelse(df$Gender == 0, 
                   ifelse(df$predicted_asmi < 7.0, 1, 0),
                   ifelse(df$predicted_asmi < 5.4, 1, 0))
  
  y_score <- ifelse(df$Gender == 0,
                    7.0 - df$predicted_asmi,
                    5.4 - df$predicted_asmi)
  
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
  
  return(list(
    auc = auc_value, sensitivity = sensitivity, specificity = specificity,
    accuracy = accuracy, f1_score = f1_score, ppv = ppv, npv = npv,
    tp = tp, tn = tn, fp = fp, fn = fn, roc_obj = roc_obj
  ))
}

################################################################################
# Figure 1: Scatter Plot Comparison
################################################################################

figure1_scatter_comparison <- function(train_data, test_data, output_dir) {
  cat("\n📊 Generating Figure 1: Scatter Plot Comparison...\n")
  
  # Metrics
  train_r <- cor(train_data$actual_asmi, train_data$predicted_asmi, method = "pearson")
  train_r2 <- train_r^2
  train_mae <- mean(abs(train_data$actual_asmi - train_data$predicted_asmi))
  train_rmse <- sqrt(mean((train_data$actual_asmi - train_data$predicted_asmi)^2))
  
  test_r <- cor(test_data$actual_asmi, test_data$predicted_asmi, method = "pearson")
  test_r2 <- test_r^2
  test_mae <- mean(abs(test_data$actual_asmi - test_data$predicted_asmi))
  test_rmse <- sqrt(mean((test_data$actual_asmi - test_data$predicted_asmi)^2))
  
  # Ranges
  train_x_range <- range(train_data$actual_asmi, na.rm = TRUE)
  train_y_range <- range(train_data$predicted_asmi, na.rm = TRUE)
  test_x_range <- range(test_data$actual_asmi, na.rm = TRUE)
  test_y_range <- range(test_data$predicted_asmi, na.rm = TRUE)
  
  train_n <- nrow(train_data); test_n <- nrow(test_data)
  
  # Plotting Function
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
      theme(
        plot.title = element_text(face = "bold", hjust = 0.0, size = 20, color = col_line),
        axis.title = element_text(face = "bold", size = 11),
        axis.text = element_text(size = 10),
        panel.grid.major = element_line(color = "gray90", size = 0.4),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "black", size = 1),
        plot.margin = margin(10, 10, 10, 10)
      )
  }
  
  p1 <- create_scatter(train_data, "5-Fold Cross validation", "#E57373", "#D32F2F", train_n, train_r, train_r2, train_mae, train_rmse, train_x_range, train_y_range)
  p2 <- create_scatter(test_data, "External validation", "#64B5F6", "#1976D2", test_n, test_r, test_r2, test_mae, test_rmse, test_x_range, test_y_range)
  
  combined <- ggdraw() + draw_plot(p1, x = 0, y = 0, width = 0.5, height = 1) + draw_plot(p2, x = 0.5, y = 0, width = 0.5, height = 1)
  ggsave(file.path(output_dir, paste0("figure1_scatter_comparison.", output_format)), combined, width = 12, height = 6, dpi = output_dpi)
  cat("   ✅ Figure 1 saved\n")
}

################################################################################
# Figure 2: Bland-Altman Plot Comparison
################################################################################

figure2_bland_altman <- function(train_data, test_data, output_dir) {
  cat("\n📊 Generating Figure 2: Bland-Altman Plot Comparison...\n")
  
  create_ba <- function(data, title, col_pt, col_line) {
    mean_val <- (data$actual_asmi + data$predicted_asmi) / 2
    diff_val <- data$actual_asmi - data$predicted_asmi
    mean_diff <- mean(diff_val); sd_diff <- sd(diff_val)
    upper <- mean_diff + 1.96 * sd_diff; lower <- mean_diff - 1.96 * sd_diff
    n <- nrow(data)
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
      annotate("text", x=rng_x[1], y=rng_y[2], label=sprintf("n = %d", n), hjust=0, vjust=0.8, size=4.5, fontface="bold", color=col_line, family="sans") +
      annotate("text", x=rng_x[1], y=rng_y[2], label=sprintf("Mean bias = %.3f", mean_diff), hjust=0, vjust=2.4, size=4.5, fontface="bold", color=col_line, family="sans") +
      annotate("text", x=rng_x[1], y=rng_y[2], label=sprintf("SD = %.3f", sd_diff), hjust=0, vjust=4.0, size=4.5, fontface="bold", color=col_line, family="sans") +
      annotate("text", x=rng_x[1], y=rng_y[2], label=sprintf("95%% LoA: [%.3f, %.3f]", lower, upper), hjust=0, vjust=5.6, size=4.5, fontface="bold", color=col_line, family="sans") +
      labs(x="Mean of Actual and Predicted ASMI (kg/m²)", y="Difference (Actual - Predicted)", title=title) +
      theme_bw(base_size = 11) +
      theme(
        plot.title = element_text(face="bold", hjust=0.0, size=20, color=col_line),
        axis.title = element_text(face="bold", size=11),
        axis.text = element_text(size=10),
        panel.grid.major = element_line(color="gray90", size=0.4),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color="black", size=1)
      )
  }
  
  p1 <- create_ba(train_data, "5-Fold Cross validation", "#E57373", "#D32F2F")
  p2 <- create_ba(test_data, "External validation", "#64B5F6", "#1976D2")
  
  combined <- ggdraw() + draw_plot(p1, x=0, y=0, width=0.5, height=1) + draw_plot(p2, x=0.5, y=0, width=0.5, height=1)
  ggsave(file.path(output_dir, paste0("figure2_bland_altman_comparison.", output_format)), combined, width=12, height=6, dpi=output_dpi)
  cat("   ✅ Figure 2 saved\n")
}

################################################################################
# Figure 3: ROC Curve Comparison
################################################################################

figure3_roc_comparison <- function(train_data, test_data, output_dir) {
  cat("\n📊 Generating Figure 3: ROC Curve Comparison...\n")
  
  create_roc_plot <- function(data, title, col_line, col_fill) {
    labs <- calculate_classification_labels(data)
    mets <- calculate_classification_metrics(labs$y_true, labs$y_pred, labs$y_score)
    roc_df <- data.frame(spec=mets$roc_obj$specificities, sens=mets$roc_obj$sensitivities)
    
    thresh_spec <- ifelse((mets$fp+mets$tn)>0, mets$tn/(mets$fp+mets$tn), 0)
    thresh_sens <- ifelse((mets$tp+mets$fn)>0, mets$tp/(mets$tp+mets$fn), 0)
    
    txt_labels <- paste0('AUC \nSens. \nSpec. \nPPV \nNPV ')
    txt_values <- paste0(formatC(mets$auc, 3, format='f'), '\n',
                         formatC(mets$sensitivity*100, 1, format='f'), '%\n',
                         formatC(mets$specificity*100, 1, format='f'), '%\n',
                         formatC(mets$ppv*100, 1, format='f'), '%\n',
                         formatC(mets$npv*100, 1, format='f'), '%')
    
    ggplot(roc_df, aes(x=spec, y=sens)) +
      geom_line(colour=col_line, size=2) +
      theme_bw() + coord_equal() +
      labs(x='Specificity', y='Sensitivity', title=title) +
      annotate(geom="point", x=thresh_spec, y=thresh_sens, shape=21, size=5, fill=paste0(col_line, 'A0'), color='#000000') +
      annotate(geom="text", x=0.05, y=0.00, label=txt_labels, size=6, fontface=2, colour=col_line, hjust=0, vjust=0) +
      annotate(geom="text", x=0.60, y=0.00, label=txt_values, size=6, fontface=2, colour=col_line, hjust=1, vjust=0) +
      theme(plot.title=element_text(color=col_line, size=20, face="bold", hjust=0.0),
            axis.title=element_text(color="#000000", size=12),
            legend.position="none")
  }
  
  p1 <- create_roc_plot(train_data, "5-Fold Cross validation", '#F8766D', '#F8766DA0')
  p2 <- create_roc_plot(test_data, "External validation", '#619CFF', '#619CFFA0')
  
  final_plot <- plot_grid(p1, p2, ncol=2)
  ggsave(file.path(output_dir, paste0("figure3_roc_comparison.", output_format)), final_plot, width=10, height=5.5, dpi=output_dpi)
  cat("   ✅ Figure 3 saved\n")
}

################################################################################
# Figure 4: Metrics Table
################################################################################

figure4_metrics_table <- function(train_data, test_data, output_dir) {
  cat("\n📊 Generating Figure 4: Classification Metrics Comparison...\n")
  
  get_m <- function(d) {
    l <- calculate_classification_labels(d)
    m <- calculate_classification_metrics(l$y_true, l$y_pred, l$y_score)
    c(m$auc, m$sensitivity, m$specificity, m$accuracy, m$f1_score, m$ppv, m$npv)
  }
  
  df <- data.frame(
    Metric = rep(c("AUC-ROC", "Sensitivity", "Specificity", "Accuracy", "F1 Score", "PPV", "NPV"), 2),
    Value = c(get_m(train_data), get_m(test_data)),
    Dataset = rep(c("Cross_Validation", "External_Test"), each=7)
  )
  
  # Save CSV
  w_df <- data.frame(Metric=unique(df$Metric), Cross_Validation=get_m(train_data), External_Test=get_m(test_data))
  write_csv(w_df, file.path(output_dir, "figure4_metrics_table.csv"))
  cat("   ✅ Metrics CSV saved\n")
  
  # Plot
  df$Metric <- factor(df$Metric, levels=rev(unique(df$Metric)))
  
  col_train <- c("AUC-ROC"="#D32F2F", "Sensitivity"="#E57373", "Specificity"="#E57373", "Accuracy"="#EF9A9A", "F1 Score"="#EF9A9A", "PPV"="#FFCDD2", "NPV"="#FFCDD2")
  col_test <- c("AUC-ROC"="#1976D2", "Sensitivity"="#42A5F5", "Specificity"="#42A5F5", "Accuracy"="#64B5F6", "F1 Score"="#64B5F6", "PPV"="#90CAF9", "NPV"="#90CAF9")
  
  p1 <- ggplot(subset(df, Dataset=="Cross_Validation"), aes(x=Metric, y=Value, fill=Metric)) +
    geom_bar(stat="identity", alpha=0.9, size=0.8) +
    labs(title='5-Fold Cross validation', x=NULL, y="Metric Value") +
    geom_text(aes(label=sprintf("%.3f", Value)), hjust=-0.15, size=4.5, fontface="bold", color="#D32F2F") +
    scale_fill_manual(values=col_train) + coord_flip() + ylim(0, 1.1) + theme_minimal(base_size=13) +
    theme(plot.title=element_text(face="bold", hjust=0, size=20, color="#D32F2F"), axis.text.y=element_text(face="bold"), legend.position="none", panel.grid=element_blank())
  
  p2 <- ggplot(subset(df, Dataset=="External_Test"), aes(x=Metric, y=Value, fill=Metric)) +
    geom_bar(stat="identity", alpha=0.9, size=0.8) +
    labs(title='External Validation', x=NULL, y="Metric Value") +
    geom_text(aes(label=sprintf("%.3f", Value)), hjust=-0.15, size=4.5, fontface="bold", color="#1976D2") +
    scale_fill_manual(values=col_test) + coord_flip() + ylim(0, 1.1) + theme_minimal(base_size=13) +
    theme(plot.title=element_text(face="bold", hjust=0, size=16, color="#1976D2"), axis.text.y=element_text(face="bold"), legend.position="none", panel.grid=element_blank())
  
  combined <- ggdraw() + draw_plot(p1, x=0, y=0.08, width=0.5, height=0.9) + draw_plot(p2, x=0.5, y=0.08, width=0.5, height=0.9)
  ggsave(file.path(output_dir, paste0("figure4_metrics_comparison.", output_format)), combined, width=14, height=6, dpi=output_dpi)
  cat("   ✅ Figure 4 saved\n")
}

################################################################################
# Figure 6: Subgroup Analysis (NOW USING BOOT PACKAGE + STRATIFIED)
################################################################################

figure6_subgroup_analysis <- function(train_data, test_data, output_dir) {
  cat("\n📊 Generating Figure 6: Subgroup Analysis (Stratified Boot pkg)...\n")
  
  # 1. Groups (Original 3-group age definition)
  assign_age <- function(age) ifelse(age < 60, "<60", ifelse(age <= 75, "60-75", ">75"))
  train_data$Age_Group <- assign_age(train_data$Age)
  test_data$Age_Group <- assign_age(test_data$Age)
  train_data$Gender_Label <- ifelse(train_data$Gender == 0, "Male", "Female")
  test_data$Gender_Label <- ifelse(test_data$Gender == 0, "Male", "Female")
  
  # 2. Worker function for boot (calculates AUC)
  boot_auc_worker <- function(data, indices) {
    d <- data[indices, ]
    # Safety check: if sample has only 1 class, return NA (boot handles NA)
    if(length(unique(d$y_true)) < 2) return(NA)
    
    # Calculate AUC
    r <- roc(d$y_true, d$y_score, quiet=TRUE)
    as.numeric(auc(r))
  }
  
  # 3. Calculation Logic
  calc_sub <- function(data, grp) {
    data %>% group_by(!!sym(grp)) %>%
      summarise(
        N = n(),
        Pearson_r = cor(actual_asmi, predicted_asmi, method="pearson"),
        
        # A. Point Estimate (True AUC using ALL data in subgroup)
        AUC_val = tryCatch({
          lbs <- calculate_classification_labels(pick(everything()))
          as.numeric(auc(roc(lbs$y_true, lbs$y_score, quiet=TRUE)))
        }, error=function(e) NA),
        
        # B. Bootstrap CI using 'boot' package + strata
        boot_res = list(tryCatch({
          lbs <- calculate_classification_labels(pick(everything()))
          # Prepare data for boot
          b_data <- data.frame(y_true=lbs$y_true, y_score=lbs$y_score)
          # STRATIFIED SAMPLING HERE: strata = y_true
          boot(data=b_data, statistic=boot_auc_worker, R=1000, strata=b_data$y_true)
        }, error=function(e) NULL))
      ) %>%
      rowwise() %>%
      mutate(
        # Extract CI from boot object
        AUC_CI_lower = if(!is.null(boot_res)) {
          ci <- tryCatch(boot.ci(boot_res, type="perc"), error=function(e) NULL)
          if(!is.null(ci)) ci$percent[4] else NA
        } else NA,
        
        AUC_CI_upper = if(!is.null(boot_res)) {
          ci <- tryCatch(boot.ci(boot_res, type="perc"), error=function(e) NULL)
          if(!is.null(ci)) ci$percent[5] else NA
        } else NA,
        
        AUC_text = sprintf("%.3f (%.3f-%.3f)", AUC_val, AUC_CI_lower, AUC_CI_upper)
      ) %>%
      select(-boot_res) %>% # Remove the complex list column
      rename(Subgroup = !!sym(grp)) # [CRITICAL FIX] Rename column to 'Subgroup' for plotting
  }
  
  # 4. Compute
  r_train <- bind_rows(calc_sub(train_data, "Gender_Label"), calc_sub(train_data, "Age_Group"))
  r_test <- bind_rows(calc_sub(test_data, "Gender_Label"), calc_sub(test_data, "Age_Group"))
  
  write_csv(bind_rows(mutate(r_train, Set="Internal"), mutate(r_test, Set="External")), 
            file.path(output_dir, "figure6_subgroup_metrics.csv"))
  cat("   ✅ Subgroup CSV saved\n")
  
  # 5. Plotting (Original Colors)
  cols <- c("Male"="#D42300", "Female"="#003D9E", "<60"="#15ED32", "60-75"="#D6BD09", ">75"="#FA9057")
  
  plot_auc <- function(df, tit) {
    # Fix order to match colors
    df$Subgroup <- factor(df$Subgroup, levels=names(cols))
    ggplot(df, aes(x=x_pos, y=AUC_val, fill=Subgroup)) +
      geom_bar(stat="identity", position="dodge") +
      geom_errorbar(aes(ymin=AUC_CI_lower, ymax=AUC_CI_upper), width=.4) +
      annotate("text", x=1:nrow(df), y=0.02, label=df$AUC_text, size=5, color="white", angle=90, fontface=2, hjust=0) +
      theme_classic() + labs(title=tit, x='', y='AUC') +
      scale_fill_manual(values=cols) +
      scale_y_continuous(expand=c(0,0), limits=c(0,1.0), breaks=c(0,0.2,0.4,0.6,0.8,1)) +
      scale_x_continuous(limits=c(0.5, nrow(df)+0.5), breaks=1:nrow(df), labels=df$Subgroup) +
      theme(plot.title=element_text(color="black", size=20, hjust=0), legend.position="none", axis.text.x=element_text(color="black", angle=45, hjust=1, size=12))
  }
  
  plot_pearson <- function(df, tit) {
    df$Subgroup <- factor(df$Subgroup, levels=names(cols))
    ggplot(df, aes(x=x_pos, y=Pearson_r, fill=Subgroup)) +
      geom_bar(stat="identity", position="dodge") +
      annotate("text", x=1:nrow(df), y=0.02, label=sprintf("%.3f", df$Pearson_r), size=5, color="white", angle=90, fontface=2, hjust=0) +
      theme_classic() + labs(title=tit, x='', y='Pearson r') +
      scale_fill_manual(values=cols) +
      scale_y_continuous(expand=c(0,0), limits=c(0,1.0), breaks=c(0,0.2,0.4,0.6,0.8,1)) +
      scale_x_continuous(limits=c(0.5, nrow(df)+0.5), breaks=1:nrow(df), labels=df$Subgroup) +
      theme(plot.title=element_text(color="black", size=20, hjust=0), legend.position="none", axis.text.x=element_text(color="black", angle=45, hjust=1, size=12))
  }
  
  # Assign x_pos for plotting
  r_train$x_pos <- 1:nrow(r_train); r_test$x_pos <- 1:nrow(r_test)
  
  p1 <- plot_pearson(r_train, "5-Fold Cross validation")
  p2 <- plot_pearson(r_test, "External validation")
  p3 <- plot_auc(r_train, "5-Fold Cross validation")
  p4 <- plot_auc(r_test, "External validation")
  
  final_plot <- ggdraw() + 
    draw_plot(p1, 0, 0.5, 0.5, 0.5) + draw_plot(p2, 0.5, 0.5, 0.5, 0.5) +
    draw_plot(p3, 0, 0, 0.5, 0.5) + draw_plot(p4, 0.5, 0, 0.5, 0.5)
  
  ggsave(file.path(output_dir, paste0("figure6_subgroup_analysis.", output_format)), final_plot, width=12, height=12, dpi=output_dpi)
  cat("   ✅ Figure 6 saved\n")
}

################################################################################
# Figure 8: Confusion Matrix
################################################################################

figure8_confusion_matrix <- function(train_data, test_data, output_dir) {
  cat("\n📊 Generating Figure 8: Confusion Matrix Visualization...\n")
  
  # Calculate classification labels
  train_labels <- calculate_classification_labels(train_data)
  train_metrics <- calculate_classification_metrics(train_labels$y_true,
                                                    train_labels$y_pred,
                                                    train_labels$y_score)
  
  test_labels <- calculate_classification_labels(test_data)
  test_metrics <- calculate_classification_metrics(test_labels$y_true,
                                                   test_labels$y_pred,
                                                   test_labels$y_score)
  
  # Create confusion matrix data frames for train
  train_cm_df <- data.frame(
    Predicted = factor(c("Normal", "Normal", "Sarcopenia", "Sarcopenia"),
                       levels = c("Normal", "Sarcopenia")),
    Actual = factor(c("Normal", "Sarcopenia", "Normal", "Sarcopenia"),
                    levels = c("Normal", "Sarcopenia")),
    Count = c(train_metrics$tn, train_metrics$fn, train_metrics$fp, train_metrics$tp),
    Label = c(
      sprintf("TN\n%d", train_metrics$tn),
      sprintf("FN\n%d", train_metrics$fn),
      sprintf("FP\n%d", train_metrics$fp),
      sprintf("TP\n%d", train_metrics$tp)
    )
  )
  
  # Create confusion matrix data frames for test
  test_cm_df <- data.frame(
    Predicted = factor(c("Normal", "Normal", "Sarcopenia", "Sarcopenia"),
                       levels = c("Normal", "Sarcopenia")),
    Actual = factor(c("Normal", "Sarcopenia", "Normal", "Sarcopenia"),
                    levels = c("Normal", "Sarcopenia")),
    Count = c(test_metrics$tn, test_metrics$fn, test_metrics$fp, test_metrics$tp),
    Label = c(
      sprintf("TN\n%d", test_metrics$tn),
      sprintf("FN\n%d", test_metrics$fn),
      sprintf("FP\n%d", test_metrics$fp),
      sprintf("TP\n%d", test_metrics$tp)
    )
  )
  
  # Calculate percentages for annotations
  train_total <- sum(train_cm_df$Count)
  test_total <- sum(test_cm_df$Count)
  train_cm_df$Percentage <- sprintf("%.1f%%", train_cm_df$Count / train_total * 100)
  test_cm_df$Percentage <- sprintf("%.1f%%", test_cm_df$Count / test_total * 100)
  
  # Left panel: Training (Coral theme)
  p1 <- ggplot(train_cm_df, aes(x = Actual, y = Predicted, fill = Count)) +
    geom_tile(color = "white", size = 2) +
    geom_text(aes(label = Label), size = 7, fontface = "bold", color = "white", vjust = -0.0) +
    geom_text(aes(label = Percentage), size = 5, color = "white", vjust = 2.0) +
    scale_fill_gradient(low = "#FFCDD2", high = "#D32F2F",
                        name = "Count", limits = c(0, max(c(train_cm_df$Count, test_cm_df$Count)))) +
    guides(fill = guide_colorbar(barheight = 10)) +
    labs(x = "Actual",
         y = "Predicted",
         title = "5-Fold Cross validation") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.0, size = 20, color = "#D32F2F"),
      axis.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 12, face = "bold"),
      legend.position = "right",
      legend.title = element_text(face = "bold", size = 12),
      legend.text = element_text(size = 10),
      panel.grid = element_blank(),
      plot.margin = margin(10, 10, 10, 10)
    ) +
    coord_equal()
  
  # Right panel: External test (Blue theme)
  p2 <- ggplot(test_cm_df, aes(x = Actual, y = Predicted, fill = Count)) +
    geom_tile(color = "white", size = 2) +
    geom_text(aes(label = Label), size = 7, fontface = "bold", color = "white", vjust = -0.0) +
    geom_text(aes(label = Percentage), size = 5, color = "white", vjust = 2.0) +
    scale_fill_gradient(low = "#90CAF9", high = "#0D47A1",
                        name = "Count", limits = c(0, max(test_cm_df$Count))) +
    guides(fill = guide_colorbar(barheight = 10)) +
    labs(x = "Actual",
         y = "Predicted",
         title = "External validation") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.0, size = 20, color = "#0D47A1"),
      axis.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 12, face = "bold"),
      legend.position = "right",
      legend.title = element_text(face = "bold", size = 12),
      legend.text = element_text(size = 10),
      panel.grid = element_blank(),
      plot.margin = margin(10, 10, 10, 10)
    ) +
    coord_equal()
  
  # Combine plots using cowplot
  combined <- ggdraw() +
    draw_plot(p1, x = 0, y = 0, width = 0.5, height = 1) +
    draw_plot(p2, x = 0.5, y = 0, width = 0.5, height = 1)
  
  # Save
  output_path <- file.path(output_dir, paste0("figure8_confusion_matrix.", output_format))
  ggsave(output_path, combined, width = 14, height = 7, dpi = output_dpi)
  
  cat("   ✅ Figure 8 saved:", output_path, "\n")
  
  # Print summary
  cat("\n", rep("=", 60), "\n", sep="")
  cat("CONFUSION MATRIX SUMMARY\n")
  cat(rep("=", 60), "\n", sep="")
  cat(sprintf("Internal Validation - Total: %d\n", train_total))
  cat(sprintf("  TP: %d (%.1f%%), TN: %d (%.1f%%)\n",
              train_metrics$tp, train_metrics$tp/train_total*100,
              train_metrics$tn, train_metrics$tn/train_total*100))
  cat(sprintf("  FP: %d (%.1f%%), FN: %d (%.1f%%)\n",
              train_metrics$fp, train_metrics$fp/train_total*100,
              train_metrics$fn, train_metrics$fn/train_total*100))
  cat(sprintf("\nExternal Validation - Total: %d\n", test_total))
  cat(sprintf("  TP: %d (%.1f%%), TN: %d (%.1f%%)\n",
              test_metrics$tp, test_metrics$tp/test_total*100,
              test_metrics$tn, test_metrics$tn/test_total*100))
  cat(sprintf("  FP: %d (%.1f%%), FN: %d (%.1f%%)\n",
              test_metrics$fp, test_metrics$fp/test_total*100,
              test_metrics$fn, test_metrics$fn/test_total*100))
  cat(rep("=", 60), "\n", sep="")
}

################################################################################
# Main execution
################################################################################

# Load data
train_data <- load_training_data(train_log_dir)
test_data <- load_test_data(test_log_dir)

# Generate figures based on type
if (figure_type %in% c("all", "scatter")) {
  figure1_scatter_comparison(train_data, test_data, output_dir)
}

if (figure_type %in% c("all", "bland_altman")) {
  figure2_bland_altman(train_data, test_data, output_dir)
}

if (figure_type %in% c("all", "roc")) {
  figure3_roc_comparison(train_data, test_data, output_dir)
}

if (figure_type %in% c("all", "table")) {
  figure4_metrics_table(train_data, test_data, output_dir)
}

if (figure_type %in% c("all", "subgroup")) {
  figure6_subgroup_analysis(train_data, test_data, output_dir)
}

if (figure_type %in% c("all", "confusion")) {
  figure8_confusion_matrix(train_data, test_data, output_dir)
}

cat("\n", rep("=", 60), "\n", sep="")
cat("✅ All requested figures generated successfully!\n")
cat("📁 Output directory:", output_dir, "\n")
cat(rep("=", 60), "\n\n", sep="")