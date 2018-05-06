library(dplyr)
library(ggplot2)
library(gridExtra)


plot_legend <- function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}


custom_col <- c("#000000", "#009E73", "#0072B2", "#D55e00", "#CC79A7")

custom_col <- c("#000000", "red","#009E73", "purple","#0072B2", "#D55e00", "#CC79A7")

roc_curve <- ggplot(aes(x = fpr, y = tpr, group = model), data = ucsd_xgboost_results_df_roc) +
  geom_line(aes(color = model), size = 1.2) +
  scale_color_manual(name = "",values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1.2) +
  theme_bw(base_size = 18) + 
  ggtitle("ROC Curves") + 
  theme(plot.title = element_text(lineheight = 1,size = 26),
        legend.position="bottom",
        legend.direction="vertical",
        legend.text=element_text(size=30), 
        legend.key = element_rect(size = 10),
        legend.key.size = unit(2.5, 'lines')) +
  guides(colour = guide_legend(override.aes = list(size=3),ncol=3)) +
  labs(x = "False Positive Rate",
       y = "True Positive Rate")

pr_curve <- ggplot(aes(x = recall, y = precision, group = model), data = ucsd_xgboost_results_df_pr) +
  geom_line(aes(color = model), size = 1.2) +
  scale_color_manual(name = "",values = custom_col) +
  ggtitle("Precision-Recall") +
  theme_bw(base_size = 18) + 
  labs(x = "Precision",
       y = "Recall") +
  guides(colour = guide_legend(override.aes = list(size=3)),
         fill=guide_legend(ncol=3)) +
  theme(plot.title = element_text(lineheight = 1,size = 26),
        legend.direction="vertical")

my_legend <- plot_legend(roc_curve)

pr_roc_graph <- grid.arrange(arrangeGrob(roc_curve + theme(legend.position = "none"),
                                         pr_curve + theme(legend.position = "none"),
                                         nrow = 1),
                             my_legend, nrow = 2, heights = c(20,5))
