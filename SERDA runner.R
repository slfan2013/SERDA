# input file location.
data_file = "manuscript datasets - final\\T2D.xlsx" 

# connect to reticulate
library(reticulate)
repl_python()
from tfdeterminism import patch
patch()
exit

# hyperparameter options.
layer_units_options = expand.grid(
  x  = round(nrow(f)*c(seq(0.2, 2.2, by = 0.2))),
  activation_function = c("elu"), 
  drop_rates = c(3:6)/100,
  # drop_rates = 5/100, 
  patience = c(25), many_training = c(TRUE
                                      # ,FALSE
  ),many_training_n = c(500), noise_weight = 1, use_qc_sd_when_final_correct = c(T
                                                                                 # ,F
  )
)

layer_units_options = layer_units_options[1:3,]

# code is running.
# source("https://raw.githubusercontent.com/slfan2013/SERDA/main/SERDA%20script.R")
source("SERDA script.R")
