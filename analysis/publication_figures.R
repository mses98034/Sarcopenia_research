#!/usr/bin/env Rscript
################################################################################
# Publication-Ready Figures Generator for ASMI Regression Analysis (R Version)
# Generates comparative figures between cross-validation and external test set
#
# Usage:
#   Rscript publication_figures.R \
#     --train-log <training_log_dir> \
#     --test-log <test_log_dir> \
#     --output <output_dir>
################################################################################

# Required packages
required_packages <- c(
  "ggplot2",      # Main plotting
  "cowplot",      # Combine plots (replaces patchwork for reference style)
  "dplyr",        # Data manipulation
  "tidyr",        # Data reshaping
  "readr",        # Fast CSV reading
  "pROC",         # ROC curves with confidence intervals
  "scales",       # Color scales
  "gridExtra",    # Additional grid functions
  "optparse"      # Command-line arguments
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
cat("PUBLICATION FIGURE GENERATOR (R VERSION)\n")
cat(rep("=", 60), "\n", sep="")
cat("Training log:", train_log_dir, "\n")
cat("Test log:", test_log_dir, "\n")
cat("Output directory:", output_dir, "\n")
cat("Figure type:", figure_type, "\n")
cat("DPI:", output_dpi, "\n")
cat("Format:", output_format, "\n")
cat(rep("=", 60), "\n\n", sep="")

################################################################################
# Data Loading Functions
################################################################################

load_training_data <- function(log_dir) {
  cat("📂 Loading training data...\n")

  csv_dir <- file.path(log_dir, "csv_data")
  val_pred_path <- file.path(csv_dir, "validation_predictions.csv")
  patient_path <- file.path(csv_dir, "patient_data.csv")

  if (!file.exists(val_pred_path)) {
    stop("Validation predictions not found: ", val_pred_path)
  }
  if (!file.exists(patient_path)) {
    stop("Patient data not found: ", patient_path)
  }

  val_pred <- read_csv(val_pred_path, show_col_types = FALSE)
  patient_data <- read_csv(patient_path, show_col_types = FALSE)

  # Merge on UID
  merged <- left_join(val_pred, patient_data, by = "UID")

  cat("   ✅ Training data loaded:", nrow(merged), "samples from",
      length(unique(val_pred$fold)), "folds\n")

  return(merged)
}

load_test_data <- function(log_dir) {
  cat("📂 Loading test data...\n")

  csv_dir <- file.path(log_dir, "csv_data")
  test_pred_path <- file.path(csv_dir, "test_predictions.csv")
  test_patient_path <- file.path(csv_dir, "test_patient_data.csv")

  if (!file.exists(test_pred_path)) {
    stop("Test predictions not found: ", test_pred_path)
  }
  if (!file.exists(test_patient_path)) {
    stop("Test patient data not found: ", test_patient_path)
  }

  test_pred <- read_csv(test_pred_path, show_col_types = FALSE)
  test_patient <- read_csv(test_patient_path, show_col_types = FALSE)

  # Merge on UID
  merged <- left_join(test_pred, test_patient, by = "UID")

  cat("   ✅ Test data loaded:", nrow(merged), "samples\n")

  return(merged)
}

################################################################################
# Classification Helper Functions
################################################################################

calculate_classification_labels <- function(df) {
  # Ground truth from Low_muscle_mass column
  if (!"Low_muscle_mass" %in% colnames(df)) {
    stop("Low_muscle_mass column not found in dataframe")
  }

  y_true <- as.integer(df$Low_muscle_mass)
  predicted_asmi <- df$predicted_asmi
  genders <- df$Gender

  # Predicted binary labels using gender-specific thresholds
  y_pred <- ifelse(genders == 0,  # Male
                   ifelse(predicted_asmi < 7.0, 1, 0),
                   ifelse(predicted_asmi < 5.4, 1, 0))  # Female

  # ROC score: standardized distance from threshold
  y_score <- ifelse(genders == 0,
                    7.0 - predicted_asmi,  # Male threshold
                    5.4 - predicted_asmi)  # Female threshold

  return(list(y_true = y_true, y_pred = y_pred, y_score = y_score))
}

calculate_classification_metrics <- function(y_true, y_pred, y_score) {
  # Confusion matrix
  cm <- table(Predicted = y_pred, Actual = y_true)

  if (nrow(cm) == 2 && ncol(cm) == 2) {
    # Confusion matrix from table(Predicted, Actual):
    #              Actual
    #              0    1
    # Predicted 0  TN   FN
    #           1  FP   TP
    tn <- cm[1, 1]  # Predicted=0, Actual=0
    fn <- cm[1, 2]  # Predicted=0, Actual=1 (False Negative)
    fp <- cm[2, 1]  # Predicted=1, Actual=0 (False Positive)
    tp <- cm[2, 2]  # Predicted=1, Actual=1
  } else {
    # Handle edge cases
    tn <- fp <- fn <- tp <- 0
    if ("0" %in% rownames(cm) && "0" %in% colnames(cm)) tn <- cm["0", "0"]
    if ("0" %in% rownames(cm) && "1" %in% colnames(cm)) fn <- cm["0", "1"]  # Fixed: FN
    if ("1" %in% rownames(cm) && "0" %in% colnames(cm)) fp <- cm["1", "0"]  # Fixed: FP
    if ("1" %in% rownames(cm) && "1" %in% colnames(cm)) tp <- cm["1", "1"]
  }

  # Calculate metrics
  sensitivity <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  specificity <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  f1_score <- ifelse((tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), 0)
  ppv <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  npv <- ifelse((tn + fn) > 0, tn / (tn + fn), 0)

  # AUC using pROC
  roc_obj <- roc(y_true, y_score, quiet = TRUE)
  auc_value <- as.numeric(auc(roc_obj))

  return(list(
    auc = auc_value,
    sensitivity = sensitivity,
    specificity = specificity,
    accuracy = accuracy,
    f1_score = f1_score,
    ppv = ppv,
    npv = npv,
    tp = tp, tn = tn, fp = fp, fn = fn,
    roc_obj = roc_obj
  ))
}

################################################################################
# Figure 1: Scatter Plot Comparison (Enhanced)
################################################################################

figure1_scatter_comparison <- function(train_data, test_data, output_dir) {
  cat("\n📊 Generating Figure 1: Scatter Plot Comparison...\n")

  # Calculate regression metrics for training
  train_r <- cor(train_data$actual_asmi, train_data$predicted_asmi, method = "pearson")
  train_r2 <- cor(train_data$actual_asmi, train_data$predicted_asmi)^2
  train_mae <- mean(abs(train_data$actual_asmi - train_data$predicted_asmi))
  train_rmse <- sqrt(mean((train_data$actual_asmi - train_data$predicted_asmi)^2))

  # Calculate regression metrics for test
  test_r <- cor(test_data$actual_asmi, test_data$predicted_asmi, method = "pearson")
  test_r2 <- cor(test_data$actual_asmi, test_data$predicted_asmi)^2
  test_mae <- mean(abs(test_data$actual_asmi - test_data$predicted_asmi))
  test_rmse <- sqrt(mean((test_data$actual_asmi - test_data$predicted_asmi)^2))

  # Get axis ranges for precise positioning
  train_x_range <- range(train_data$actual_asmi, na.rm = TRUE)
  train_y_range <- range(train_data$predicted_asmi, na.rm = TRUE)
  test_x_range <- range(test_data$actual_asmi, na.rm = TRUE)
  test_y_range <- range(test_data$predicted_asmi, na.rm = TRUE)

  # Calculate sample sizes
  train_n <- nrow(train_data)
  test_n <- nrow(test_data)

  # Left panel: Training (Cross-validation) - Coral theme
  p1 <- ggplot(train_data, aes(x = actual_asmi, y = predicted_asmi)) +
    geom_point(alpha = 0.4, color = "#E57373", size = 2.5) +
    geom_abline(intercept = 0, slope = 1, color = "gray30", linetype = "dashed", size = 1) +
    geom_smooth(method = "lm", color = "#D32F2F", fill = "#E57373",
                se = TRUE, alpha = 0.15, size = 1.2) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf,
             fill = NA, color = "#E57373", size = 1.5) +
    # Metrics text in top-left corner (perfectly aligned)
    annotate("text", x = train_x_range[1], y = train_y_range[2],
             label = sprintf("n = %d", train_n),
             hjust = 0, vjust = 0.8, size = 6.0,
             fontface = "bold", color = "#D32F2F", family = "sans") +
    annotate("text", x = train_x_range[1], y = train_y_range[2],
             label = sprintf("Pearson r = %.3f", train_r),
             hjust = 0, vjust = 2.4, size = 6.0,
             fontface = "bold", color = "#D32F2F", family = "sans") +
    annotate("text", x = train_x_range[1], y = train_y_range[2],
             label = sprintf("R² = %.3f", train_r2),
             hjust = 0, vjust = 4.0, size = 6.0,
             fontface = "bold", color = "#D32F2F", family = "sans") +
    annotate("text", x = train_x_range[1], y = train_y_range[2],
             label = sprintf("MAE = %.3f kg/m²", train_mae),
             hjust = 0, vjust = 5.6, size = 6.0,
             fontface = "bold", color = "#D32F2F", family = "sans") +
    annotate("text", x = train_x_range[1], y = train_y_range[2],
             label = sprintf("RMSE = %.3f kg/m²", train_rmse),
             hjust = 0, vjust = 7.2, size = 6.0,
             fontface = "bold", color = "#D32F2F", family = "sans") +
    labs(x = "Actual ASMI (kg/m²)",
         y = "Predicted ASMI (kg/m²)",
         title = "Internal validation") +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.0, size = 20, color = "#D32F2F"),
      axis.title = element_text(face = "bold", size = 11),
      axis.text = element_text(size = 10),
      panel.grid.major = element_line(color = "gray90", size = 0.4),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", size = 1),
      plot.margin = margin(10, 10, 10, 10)
    )

  # Right panel: External test - Blue theme
  p2 <- ggplot(test_data, aes(x = actual_asmi, y = predicted_asmi)) +
    geom_point(alpha = 0.4, color = "#64B5F6", size = 2.5) +
    geom_abline(intercept = 0, slope = 1, color = "gray30", linetype = "dashed", size = 1) +
    geom_smooth(method = "lm", color = "#1976D2", fill = "#64B5F6",
                se = TRUE, alpha = 0.15, size = 1.2) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf,
             fill = NA, color = "#64B5F6", size = 1.5) +
    # Metrics text in top-left corner (perfectly aligned)
    annotate("text", x = test_x_range[1], y = test_y_range[2],
             label = sprintf("n = %d", test_n),
             hjust = 0, vjust = 0.8, size = 6.0,
             fontface = "bold", color = "#1976D2", family = "sans") +
    annotate("text", x = test_x_range[1], y = test_y_range[2],
             label = sprintf("Pearson r = %.3f", test_r),
             hjust = 0, vjust = 2.4, size = 6.0,
             fontface = "bold", color = "#1976D2", family = "sans") +
    annotate("text", x = test_x_range[1], y = test_y_range[2],
             label = sprintf("R² = %.3f", test_r2),
             hjust = 0, vjust = 4.0, size = 6.0,
             fontface = "bold", color = "#1976D2", family = "sans") +
    annotate("text", x = test_x_range[1], y = test_y_range[2],
             label = sprintf("MAE = %.3f kg/m²", test_mae),
             hjust = 0, vjust = 5.6, size = 6.0,
             fontface = "bold", color = "#1976D2", family = "sans") +
    annotate("text", x = test_x_range[1], y = test_y_range[2],
             label = sprintf("RMSE = %.3f kg/m²", test_rmse),
             hjust = 0, vjust = 7.2, size = 6.0,
             fontface = "bold", color = "#1976D2", family = "sans") +
    labs(x = "Actual ASMI (kg/m²)",
         y = "Predicted ASMI (kg/m²)",
         title = "External validation") +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.0, size = 20, color = "#1976D2"),
      axis.title = element_text(face = "bold", size = 11),
      axis.text = element_text(size = 10),
      panel.grid.major = element_line(color = "gray90", size = 0.4),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", size = 1),
      plot.margin = margin(10, 10, 10, 10)
    )

  # Combine plots using cowplot
  combined <- ggdraw() +
    draw_plot(p1, x = 0, y = 0, width = 0.5, height = 1) +
    draw_plot(p2, x = 0.5, y = 0, width = 0.5, height = 1) 
    # draw_plot_label("ASMI Prediction Performance",
    #                x = 0.5, y = 0.98, size = 14, hjust = 0.5, fontface = "bold")

  # Save
  output_path <- file.path(output_dir, paste0("figure1_scatter_comparison.", output_format))
  ggsave(output_path, combined, width = 12, height = 6, dpi = output_dpi)

  cat("   ✅ Figure 1 saved:", output_path, "\n")
  cat("   Training: r =", round(train_r, 3), ", MAE =", round(train_mae, 3), "\n")
  cat("   Test: r =", round(test_r, 3), ", MAE =", round(test_mae, 3), "\n")
}

################################################################################
# Figure 2: Bland-Altman Plot Comparison (Enhanced)
################################################################################

figure2_bland_altman <- function(train_data, test_data, output_dir) {
  cat("\n📊 Generating Figure 2: Bland-Altman Plot Comparison...\n")

  # Calculate Bland-Altman statistics for training
  train_mean <- (train_data$actual_asmi + train_data$predicted_asmi) / 2
  train_diff <- train_data$actual_asmi - train_data$predicted_asmi
  train_mean_diff <- mean(train_diff)
  train_sd_diff <- sd(train_diff)
  train_loa_upper <- train_mean_diff + 1.96 * train_sd_diff
  train_loa_lower <- train_mean_diff - 1.96 * train_sd_diff

  # Calculate Bland-Altman statistics for test
  test_mean <- (test_data$actual_asmi + test_data$predicted_asmi) / 2
  test_diff <- test_data$actual_asmi - test_data$predicted_asmi
  test_mean_diff <- mean(test_diff)
  test_sd_diff <- sd(test_diff)
  test_loa_upper <- test_mean_diff + 1.96 * test_sd_diff
  test_loa_lower <- test_mean_diff - 1.96 * test_sd_diff

  # Calculate ranges for text positioning
  train_x_range <- range(train_mean)
  train_y_range <- range(train_diff)
  train_n <- nrow(train_data)

  # Left panel: Training (Coral theme)
  p1 <- ggplot(data.frame(mean = train_mean, diff = train_diff),
               aes(x = mean, y = diff)) +
    geom_hline(yintercept = 0, color = "gray70", linetype = "dotted", size = 0.8) +
    geom_hline(yintercept = train_mean_diff, color = "#D32F2F", linetype = "solid", size = 1.2) +
    geom_hline(yintercept = train_loa_upper, color = "#D32F2F", linetype = "dashed", size = 1) +
    geom_hline(yintercept = train_loa_lower, color = "#D32F2F", linetype = "dashed", size = 1) +
    geom_point(alpha = 0.4, color = "#E57373", size = 2.5) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = train_loa_lower, ymax = train_loa_upper,
             fill = "#E57373", alpha = 0.08) +
    annotate("text", x = Inf, y = train_mean_diff,
             label = sprintf("Mean: %.3f", train_mean_diff),
             hjust = 1.05, vjust = -0.5, size = 4, fontface = "bold", color = "#D32F2F") +
    annotate("text", x = Inf, y = train_loa_upper,
             label = sprintf("+1.96 SD: %.3f", train_loa_upper),
             hjust = 1.05, vjust = -0.5, size = 5, fontface = "bold", color = "#D32F2F") +
    annotate("text", x = Inf, y = train_loa_lower,
             label = sprintf("-1.96 SD: %.3f", train_loa_lower),
             hjust = 1.05, vjust = 1.5, size = 5, fontface = "bold", color = "#D32F2F") +
    # Top-left metrics
    annotate("text", x = train_x_range[1], y = train_y_range[2],
             label = sprintf("n = %d", train_n),
             hjust = 0, vjust = 0.8, size = 4.5, fontface = "bold", color = "#D32F2F", family = "sans") +
    annotate("text", x = train_x_range[1], y = train_y_range[2],
             label = sprintf("Mean bias = %.3f", train_mean_diff),
             hjust = 0, vjust = 2.4, size = 4.5, fontface = "bold", color = "#D32F2F", family = "sans") +
    annotate("text", x = train_x_range[1], y = train_y_range[2],
             label = sprintf("SD = %.3f", train_sd_diff),
             hjust = 0, vjust = 4.0, size = 4.5, fontface = "bold", color = "#D32F2F", family = "sans") +
    annotate("text", x = train_x_range[1], y = train_y_range[2],
             label = sprintf("95%% LoA: [%.3f, %.3f]", train_loa_lower, train_loa_upper),
             hjust = 0, vjust = 5.6, size = 4.5, fontface = "bold", color = "#D32F2F", family = "sans") +
    labs(x = "Mean of Actual and Predicted ASMI (kg/m²)",
         y = "Difference (Actual - Predicted)",
         title = "Internal validation") +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.0, size = 20, color = "#D32F2F"),
      axis.title = element_text(face = "bold", size = 11),
      axis.text = element_text(size = 10),
      panel.grid.major = element_line(color = "gray90", size = 0.4),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", size = 1)
    )

  # Calculate ranges for text positioning
  test_x_range <- range(test_mean)
  test_y_range <- range(test_diff)
  test_n <- nrow(test_data)

  # Right panel: External test (Blue theme)
  p2 <- ggplot(data.frame(mean = test_mean, diff = test_diff),
               aes(x = mean, y = diff)) +
    geom_hline(yintercept = 0, color = "gray70", linetype = "dotted", size = 0.8) +
    geom_hline(yintercept = test_mean_diff, color = "#1976D2", linetype = "solid", size = 1.2) +
    geom_hline(yintercept = test_loa_upper, color = "#1976D2", linetype = "dashed", size = 1) +
    geom_hline(yintercept = test_loa_lower, color = "#1976D2", linetype = "dashed", size = 1) +
    geom_point(alpha = 0.4, color = "#64B5F6", size = 2.5) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = test_loa_lower, ymax = test_loa_upper,
             fill = "#64B5F6", alpha = 0.08) +
    annotate("text", x = Inf, y = test_mean_diff,
             label = sprintf("Mean: %.3f", test_mean_diff),
             hjust = 1.05, vjust = -0.5, size = 4, fontface = "bold", color = "#1976D2") +
    annotate("text", x = Inf, y = test_loa_upper,
             label = sprintf("+1.96 SD: %.3f", test_loa_upper),
             hjust = 1.05, vjust = -0.5, size = 5, fontface = "bold", color = "#1976D2") +
    annotate("text", x = Inf, y = test_loa_lower,
             label = sprintf("-1.96 SD: %.3f", test_loa_lower),
             hjust = 1.05, vjust = 1.5, size = 5, fontface = "bold", color = "#1976D2") +
    # Top-left metrics
    annotate("text", x = test_x_range[1], y = test_y_range[2],
             label = sprintf("n = %d", test_n),
             hjust = 0, vjust = 0.8, size = 4.5, fontface = "bold", color = "#1976D2", family = "sans") +
    annotate("text", x = test_x_range[1], y = test_y_range[2],
             label = sprintf("Mean bias = %.3f", test_mean_diff),
             hjust = 0, vjust = 2.4, size = 4.5, fontface = "bold", color = "#1976D2", family = "sans") +
    annotate("text", x = test_x_range[1], y = test_y_range[2],
             label = sprintf("SD = %.3f", test_sd_diff),
             hjust = 0, vjust = 4.0, size = 4.5, fontface = "bold", color = "#1976D2", family = "sans") +
    annotate("text", x = test_x_range[1], y = test_y_range[2],
             label = sprintf("95%% LoA: [%.3f, %.3f]", test_loa_lower, test_loa_upper),
             hjust = 0, vjust = 5.6, size = 4.5, fontface = "bold", color = "#1976D2", family = "sans") +
    labs(x = "Mean of Actual and Predicted ASMI (kg/m²)",
         y = "Difference (Actual - Predicted)",
         title = "External validation") +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.0, size = 20, color = "#1976D2"),
      axis.title = element_text(face = "bold", size = 11),
      axis.text = element_text(size = 10),
      panel.grid.major = element_line(color = "gray90", size = 0.4),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", size = 1)
    )

  # Combine plots using cowplot
  combined <- ggdraw() +
    draw_plot(p1, x = 0, y = 0, width = 0.5, height = 1) +
    draw_plot(p2, x = 0.5, y = 0, width = 0.5, height = 1) 
    # draw_plot_label("Bland-Altman Agreement Analysis",
    #                x = 0.5, y = 0.98, size = 14, hjust = 0.5, fontface = "bold")

  # Save
  output_path <- file.path(output_dir, paste0("figure2_bland_altman_comparison.", output_format))
  ggsave(output_path, combined, width = 12, height = 6, dpi = output_dpi)

  cat("   ✅ Figure 2 saved:", output_path, "\n")
}

################################################################################
# Figure 3: ROC Curve Comparison (Reference 002. ROC curve style)
################################################################################

figure3_roc_comparison <- function(train_data, test_data, output_dir) {
  cat("\n📊 Generating Figure 3: ROC Curve Comparison...\n")

  # Color scheme
  col_list <- c('internal' = '#F8766D', 'external' = '#619CFF')

  # Calculate classification labels
  train_labels <- calculate_classification_labels(train_data)
  train_metrics <- calculate_classification_metrics(train_labels$y_true,
                                                     train_labels$y_pred,
                                                     train_labels$y_score)

  test_labels <- calculate_classification_labels(test_data)
  test_metrics <- calculate_classification_metrics(test_labels$y_true,
                                                    test_labels$y_pred,
                                                    test_labels$y_score)

  # Calculate threshold operating points
  train_threshold_spec <- ifelse((train_metrics$fp + train_metrics$tn) > 0,
                                  train_metrics$tn / (train_metrics$fp + train_metrics$tn), 0)
  train_threshold_sens <- ifelse((train_metrics$tp + train_metrics$fn) > 0,
                                  train_metrics$tp / (train_metrics$tp + train_metrics$fn), 0)

  test_threshold_spec <- ifelse((test_metrics$fp + test_metrics$tn) > 0,
                                 test_metrics$tn / (test_metrics$fp + test_metrics$tn), 0)
  test_threshold_sens <- ifelse((test_metrics$tp + test_metrics$fn) > 0,
                                 test_metrics$tp / (test_metrics$tp + test_metrics$fn), 0)

  # Prepare ROC data for ggplot (spec vs sens, like reference)
  train_roc_df <- data.frame(
    spec = train_metrics$roc_obj$specificities,
    sens = train_metrics$roc_obj$sensitivities
  )

  test_roc_df <- data.frame(
    spec = test_metrics$roc_obj$specificities,
    sens = test_metrics$roc_obj$sensitivities
  )

  # Create metrics text (Reference style: labels and values separately)
  # Labels (left-aligned)
  roc_txt_labels <- paste0('AUC ',
                          '\nSens. ',
                          '\nSpec. ',
                          '\nPPV ',
                          '\nNPV ',
                          '')

  # Values for training (right-aligned)
  train_txt_values <- paste0(formatC(train_metrics$auc, 3, format = 'f'),
                            '\n', formatC(train_metrics$sensitivity * 100, 1, format = 'f'), '%',
                            '\n', formatC(train_metrics$specificity * 100, 1, format = 'f'), '%',
                            '\n', formatC(train_metrics$ppv * 100, 1, format = 'f'), '%',
                            '\n', formatC(train_metrics$npv * 100, 1, format = 'f'), '%')

  # Values for test (right-aligned)
  test_txt_values <- paste0(formatC(test_metrics$auc, 3, format = 'f'),
                            '\n', formatC(test_metrics$sensitivity * 100, 1, format = 'f'), '%',
                            '\n', formatC(test_metrics$specificity * 100, 1, format = 'f'), '%',
                            '\n', formatC(test_metrics$ppv * 100, 1, format = 'f'), '%',
                            '\n', formatC(test_metrics$npv * 100, 1, format = 'f'), '%')

  # Left panel: Training (Reference style)
  p1 <- ggplot(data = train_roc_df, aes(x = spec, y = sens)) +
    geom_line(colour = col_list['internal'], size = 1) +
    theme_bw() +
    coord_equal() +
    labs(x = 'Specificity', y = 'Sensitivity', title='Internal validation') +
    # Threshold point (Reference style: shape 21, semi-transparent fill)
    annotate(geom = "point", x = train_threshold_spec, y = train_threshold_sens,
             shape = 21, size = 5, fill = paste0(col_list['internal'], 'A0'), color = '#000000') +
    # Metrics text (Reference style: two text annotations)
    annotate(geom = "text", x = 0.05, y = 0.00, label = roc_txt_labels,
             size = 6, fontface = 2, colour = col_list['internal'], hjust = 0, vjust = 0) +
    annotate(geom = "text", x = 0.60, y = 0.00, label = train_txt_values,
             size = 6, fontface = 2, colour = col_list['internal'], hjust = 1, vjust = 0) +
    theme(plot.title = element_text(color = "#F8766D", size = 20, face = "bold", hjust = 0.0),
          axis.title = element_text(color = "#000000", size = 12),
          legend.position = "none")

  # Right panel: External test (Reference style)
  p2 <- ggplot(data = test_roc_df, aes(x = spec, y = sens)) +
    geom_line(colour = col_list['external'], size = 1) +
    theme_bw() +
    coord_equal() +
    labs(x = 'Specificity', y = 'Sensitivity', title='External validation') +
    # Threshold point
    annotate(geom = "point", x = test_threshold_spec, y = test_threshold_sens,
             shape = 21, size = 5, fill = paste0(col_list['external'], 'A0'), color = '#000000') +
    # Metrics text
    annotate(geom = "text", x = 0.05, y = 0.00, label = roc_txt_labels,
             size = 6, fontface = 2, colour = col_list['external'], hjust = 0, vjust = 0) +
    annotate(geom = "text", x = 0.60, y = 0.00, label = test_txt_values,
             size = 6, fontface = 2, colour = col_list['external'], hjust = 1, vjust = 0) +
    theme(plot.title = element_text(color = '#619CFF', size = 20, face = "bold", hjust = 0.0),
          axis.title = element_text(color = "#000000", size = 12),
          legend.position = "none")

  # Layout using cowplot (Reference style)
  final_plot <- plot_grid(p1, p2, ncol = 2)
  # ggdraw() +
  #   draw_plot(p1, x = 0, y = 0, width = 0.5, height = 0.95) +
  #   draw_plot(p2, x = 0.5, y = 0, width = 0.5, height = 0.95) +
  #   draw_plot_label(c("Internal validation", "External validation"),
  #                  x = c(0.005, 0.505), y = c(1.0, 1.0),
  #                  size = 18, colour = col_list, hjust = 0, fontface = 2)

  # Save
  output_path <- file.path(output_dir, paste0("figure3_roc_comparison.", output_format))
  ggsave(output_path, final_plot, width = 10, height = 5.5, dpi = output_dpi)

  cat("   ✅ Figure 3 saved:", output_path, "\n")
  cat("   Training AUC:", round(train_metrics$auc, 3), "\n")
  cat("   Test AUC:", round(test_metrics$auc, 3), "\n")

  return(list(train_auc = train_metrics$auc, test_auc = test_metrics$auc))
}

################################################################################
# Figure 4: Classification Metrics Horizontal Bar Chart
################################################################################

figure4_metrics_table <- function(train_data, test_data, output_dir) {
  cat("\n📊 Generating Figure 4: Classification Metrics Comparison...\n")

  # Calculate classification metrics
  train_labels <- calculate_classification_labels(train_data)
  train_metrics <- calculate_classification_metrics(train_labels$y_true,
                                                     train_labels$y_pred,
                                                     train_labels$y_score)

  test_labels <- calculate_classification_labels(test_data)
  test_metrics <- calculate_classification_metrics(test_labels$y_true,
                                                    test_labels$y_pred,
                                                    test_labels$y_score)

  # Create comparison dataframe
  metrics_df <- data.frame(
    Metric = c("AUC-ROC", "Sensitivity", "Specificity", "Accuracy",
               "F1 Score", "PPV", "NPV"),
    Cross_Validation = c(train_metrics$auc, train_metrics$sensitivity,
                         train_metrics$specificity, train_metrics$accuracy,
                         train_metrics$f1_score, train_metrics$ppv,
                         train_metrics$npv),
    External_Test = c(test_metrics$auc, test_metrics$sensitivity,
                      test_metrics$specificity, test_metrics$accuracy,
                      test_metrics$f1_score, test_metrics$ppv,
                      test_metrics$npv),
    stringsAsFactors = FALSE
  )

  # Save CSV
  csv_path <- file.path(output_dir, "figure4_metrics_table.csv")
  write_csv(metrics_df, csv_path)
  cat("   ✅ Metrics table CSV saved:", csv_path, "\n")

  # Define colors matching previous figures (coral gradient for train, blue gradient for test)
  # Use gradient from lighter to darker for visual appeal
  train_colors <- c(
    "AUC-ROC" = "#D32F2F",      # Darker coral-red
    "Sensitivity" = "#E57373",   # Medium coral
    "Specificity" = "#E57373",   # Medium coral
    "Accuracy" = "#EF9A9A",      # Light coral
    "F1 Score" = "#EF9A9A",      # Light coral
    "PPV" = "#FFCDD2",           # Very light coral
    "NPV" = "#FFCDD2"            # Very light coral
  )

  test_colors <- c(
    "AUC-ROC" = "#1976D2",       # Darker blue
    "Sensitivity" = "#42A5F5",   # Medium blue
    "Specificity" = "#42A5F5",   # Medium blue
    "Accuracy" = "#64B5F6",      # Light blue
    "F1 Score" = "#64B5F6",      # Light blue
    "PPV" = "#90CAF9",           # Very light blue
    "NPV" = "#90CAF9"            # Very light blue
  )

  # Reorder metrics for better visual hierarchy
  metrics_df$Metric <- factor(metrics_df$Metric,
                               levels = rev(c("AUC-ROC", "Sensitivity", "Specificity",
                                            "Accuracy", "F1 Score", "PPV", "NPV")))

  # Left panel: Cross-Validation (Coral theme)
  p1 <- ggplot(metrics_df, aes(x = Metric, y = Cross_Validation, fill = Metric)) +
    geom_bar(stat = "identity", alpha = 0.9, size = 0.8) + #, color = "#8B0000"
    labs(title='Cross Validation')+
    geom_text(aes(label = sprintf("%.3f", Cross_Validation)),
              hjust = -0.15, size = 4.5, fontface = "bold", color = "#D32F2F") +
    scale_fill_manual(values = train_colors) +
    coord_flip() +
    ylim(0, 1.1) +
    labs(x = NULL, y = "Metric Value") +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.0, size = 20, color = "#D32F2F"),
      axis.title.x = element_text(face = "bold", size = 12, margin = margin(t = 10)),
      axis.text = element_text(size = 11, color = "gray20"),
      axis.text.y = element_text(face = "bold"),
      legend.position = "none",
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_blank(), #element_line(color = "gray90", size = 0.5),
      panel.grid.minor = element_blank(),
      # panel.border = element_rect(color = "#E57373", fill = NA, size = 1.5),
      plot.margin = margin(10, 10, 10, 10)
    )

  # Right panel: External Test (Blue theme)
  p2 <- ggplot(metrics_df, aes(x = Metric, y = External_Test, fill = Metric)) +
    geom_bar(stat = "identity", alpha = 0.9, size = 0.8) + #, color = "#0D47A1"
    labs(title='External Validation')+
    geom_text(aes(label = sprintf("%.3f", External_Test)),
              hjust = -0.15, size = 4.5, fontface = "bold", color = "#1976D2") +
    scale_fill_manual(values = test_colors) +
    coord_flip() +
    ylim(0, 1.1) +
    labs(x = NULL, y = "Metric Value") +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.0, size = 16, color = "#1976D2"),
      axis.title.x = element_text(face = "bold", size = 12, margin = margin(t = 10)),
      axis.text = element_text(size = 11, color = "gray20"),
      axis.text.y = element_text(face = "bold"),
      legend.position = "none",
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_blank(), #element_line(color = "gray90", size = 0.5),
      panel.grid.minor = element_blank(),
      # panel.border = element_rect(color = "#42A5F5", fill = NA, size = 1.5),
      plot.margin = margin(10, 10, 10, 10)
    )

  # Combine plots using cowplot with color-coordinated labels
  combined <- ggdraw() +
    draw_plot(p1, x = 0, y = 0.08, width = 0.5, height = 0.9) +
    draw_plot(p2, x = 0.5, y = 0.08, width = 0.5, height = 0.9)
    # # Main title
    # draw_plot_label("Classification Performance Comparison",
    #                x = 0.5, y = 0.99, size = 16, hjust = 0.5, fontface = "bold") +
    # Dataset labels with matching colors
    # draw_plot_label(c("Internal Validation (5-Fold CV)", "External Test Set"),
    #                x = c(0.005, 0.505), y = c(1.0, 1.0),
    #                size = 14, colour = c("#D32F2F", "#1976D2"),
    #                hjust = 0, fontface = 2) 
    # # Confusion matrix info with color coding
    # draw_label(sprintf("CM: TP=%d TN=%d FP=%d FN=%d",
    #                   train_metrics$tp, train_metrics$tn,
    #                   train_metrics$fp, train_metrics$fn),
    #           x = 0.25, y = 0.03, size = 9, hjust = 0.5, color = "#D32F2F",
    #           fontface = "bold") +
    # draw_label(sprintf("CM: TP=%d TN=%d FP=%d FN=%d",
    #                   test_metrics$tp, test_metrics$tn,
    #                   test_metrics$fp, test_metrics$fn),
    #           x = 0.75, y = 0.03, size = 9, hjust = 0.5, color = "#1976D2",
    #           fontface = "bold")

  # Save
  output_path <- file.path(output_dir, paste0("figure4_metrics_comparison.", output_format))
  ggsave(output_path, combined, width = 14, height = 6, dpi = output_dpi)

  cat("   ✅ Figure 4 saved:", output_path, "\n")

  # Print to console
  cat("\n", rep("=", 60), "\n", sep="")
  cat("CLASSIFICATION METRICS COMPARISON\n")
  cat(rep("=", 60), "\n", sep="")
  print(metrics_df[, c("Metric", "Cross_Validation", "External_Test")])
  cat(rep("=", 60), "\n", sep="")
}

################################################################################
# Figure 6: Subgroup Analysis
################################################################################

figure6_subgroup_analysis <- function(train_data, test_data, output_dir) {
  cat("\n📊 Generating Figure 6: Subgroup Analysis...\n")

  # Define age groups (geriatric medicine standard)
  assign_age_group <- function(age) {
    ifelse(age < 60, "<60",
           ifelse(age <= 75, "60-75", ">75"))
  }

  train_data$Age_Group <- assign_age_group(train_data$Age)
  test_data$Age_Group <- assign_age_group(test_data$Age)

  # Gender labels
  train_data$Gender_Label <- ifelse(train_data$Gender == 0, "Male", "Female")
  test_data$Gender_Label <- ifelse(test_data$Gender == 0, "Male", "Female")

  # Calculate metrics with confidence intervals (Reference 003. Stratified analysis style)
  calculate_subgroup_metrics <- function(data, group_var) {
    data %>%
      group_by(!!sym(group_var)) %>%
      summarise(
        N = n(),
        # Regression metrics (no error bars for simplicity)
        Pearson_r = cor(actual_asmi, predicted_asmi, method = "pearson"),
        R2 = cor(actual_asmi, predicted_asmi)^2,
        MAE = mean(abs(actual_asmi - predicted_asmi)),
        .groups = 'drop'
      ) -> reg_metrics

    # Add AUC with pROC confidence intervals (Reference style)
    auc_results <- data %>%
      group_by(!!sym(group_var)) %>%
      summarise(
        AUC_val = tryCatch({
          labels <- calculate_classification_labels(cur_data())
          roc_obj <- roc(labels$y_true, labels$y_score, quiet = TRUE)
          ci_result <- ci.auc(roc_obj)
          ci_result[2]  # Point estimate
        }, error = function(e) NA_real_),
        AUC_CI_lower = tryCatch({
          labels <- calculate_classification_labels(cur_data())
          roc_obj <- roc(labels$y_true, labels$y_score, quiet = TRUE)
          ci_result <- ci.auc(roc_obj)
          ci_result[1]  # Lower bound
        }, error = function(e) NA_real_),
        AUC_CI_upper = tryCatch({
          labels <- calculate_classification_labels(cur_data())
          roc_obj <- roc(labels$y_true, labels$y_score, quiet = TRUE)
          ci_result <- ci.auc(roc_obj)
          ci_result[3]  # Upper bound
        }, error = function(e) NA_real_),
        .groups = 'drop'
      )

    # Format AUC text (Reference style: "0.XXX (0.XXX-0.XXX)")
    auc_results <- auc_results %>%
      mutate(
        AUC = AUC_val,
        AUC_text = sprintf("%.3f (%.3f-%.3f)", AUC_val, AUC_CI_lower, AUC_CI_upper)
      )

    left_join(reg_metrics, auc_results, by = group_var)
  }

  # Calculate for gender and age groups
  train_gender <- calculate_subgroup_metrics(train_data, "Gender_Label")
  train_age <- calculate_subgroup_metrics(train_data, "Age_Group")

  test_gender <- calculate_subgroup_metrics(test_data, "Gender_Label")
  test_age <- calculate_subgroup_metrics(test_data, "Age_Group")

  # Combine all subgroups
  train_gender$Subgroup <- train_gender$Gender_Label
  train_age$Subgroup <- train_age$Age_Group
  test_gender$Subgroup <- test_gender$Gender_Label
  test_age$Subgroup <- test_age$Age_Group

  train_all <- bind_rows(
    train_gender %>% select(Subgroup, Pearson_r, R2, MAE, AUC, AUC_CI_lower, AUC_CI_upper, AUC_text),
    train_age %>% select(Subgroup, Pearson_r, R2, MAE, AUC, AUC_CI_lower, AUC_CI_upper, AUC_text)
  )
  train_all$Dataset <- "Internal validation set"

  test_all <- bind_rows(
    test_gender %>% select(Subgroup, Pearson_r, R2, MAE, AUC, AUC_CI_lower, AUC_CI_upper, AUC_text),
    test_age %>% select(Subgroup, Pearson_r, R2, MAE, AUC, AUC_CI_lower, AUC_CI_upper, AUC_text)
  )
  test_all$Dataset <- "External validation set"

  # Save CSV with confidence intervals
  csv_path <- file.path(output_dir, "figure6_subgroup_metrics.csv")
  write_csv(bind_rows(train_all, test_all), csv_path)
  cat("   ✅ Subgroup metrics CSV saved:", csv_path, "\n")

  # Define colors (Reference style color palette)
  subgroup_colors <- c(
    "Male" = "#D42300",      # Red
    "Female" = "#003D9E",    # Blue
    "<60" = "#15ED32",       # Green
    "60-75" = "#D6BD09",     # Yellow
    ">75" = "#FA9057"        # Orange
  )

  # Assign x positions for bars (Reference style)
  train_all <- train_all %>%
    mutate(
      x_pos = 1:n(),
      col = subgroup_colors[Subgroup]
    )

  test_all <- test_all %>%
    mutate(
      x_pos = 1:n(),
      col = subgroup_colors[Subgroup]
    )

  # Create AUC bar plot (Reference 003. Stratified analysis style)
  create_auc_barplot <- function(data, title) {
    # Bar plot with error bars and text labels
    p <- ggplot(data, aes(x = x_pos, y = AUC, fill = col)) +
      geom_bar(position = "dodge", stat = "identity") +
      geom_errorbar(aes(ymin = AUC_CI_lower, ymax = AUC_CI_upper),
                   width = .4, position = position_dodge(.9)) +
      # Vertical text labels at bar bottom (Reference style)
      annotate(geom = "text", x = data$x_pos, y = 0.02, label = data$AUC_text,
              size = 5, color = "white", angle = 90, fontface = 2, hjust = 0) +
      theme_classic() +
      labs(title = title, x = '', y = 'AUC') +
      scale_fill_manual(values = factor(data$col) %>% levels()) +
      scale_y_continuous(expand = c(0, 0), limits = c(0, 1.0), breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1)) +
      scale_x_continuous(limits = c(0.5, max(data$x_pos) + 0.5),
                        breaks = data$x_pos,
                        labels = data$Subgroup) +
      theme(plot.title = element_text(color = "#000000", size = 20, hjust=0),
           legend.position = "none",
           axis.text.x = element_text(color = "#000000", angle = 45, hjust = 1, size = 8))

    return(p)
  }

  # Create plots for both datasets
  p_train <- create_auc_barplot(train_all, "Internal validation set")
  p_test <- create_auc_barplot(test_all, "External validation set")

  # Layout using cowplot (Reference style)
  final_plot <- ggdraw() +
    draw_plot(p_train, x = 0, y = 0, width = 0.5, height = 0.75) +
    draw_plot(p_test, x = 0.5, y = 0, width = 0.5, height = 0.75)


  # Save
  output_path <- file.path(output_dir, paste0("figure6_subgroup_analysis.", output_format))
  ggsave(output_path, final_plot, width = 12, height = 6, dpi = output_dpi)

  cat("   ✅ Figure 6 saved:", output_path, "\n")
}

################################################################################
# Figure 8: Confusion Matrix Visualization
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
         title = "Internal validation") +
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
