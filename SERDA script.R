# User can choose to disable GPU and use CPU instead.
# Disable GPU                                                                   
# Sys.setenv("CUDA_VISIBLE_DEVICES" = -1)                                     
# Sys.setenv("TF_DETERMINISTIC_OPS" = 1)
# keras::install_keras(tensorflow = "gpu")
# tfdeterminism
# reticulate::py_install("tensorflow-determinism",pip =TRUE)



pacman::p_load(keras,pROC,caret)
seed = 42



source("https://raw.githubusercontent.com/slfan2013/rcodes/master/SERDA%20utils.R")
# scatter plot
scatter_plot = TRUE
source("https://raw.githubusercontent.com/slfan2013/rcodes/master/read_data.R")




# read datasets
if(grepl("\\\\",data_file)){
  comp = strsplit(data_file,"\\\\")[[1]]
}else{
  comp = strsplit(data_file,"/")[[1]]
}



filename = gsub("\\.csv|\\.xlsx","",comp[length(comp)])
# root = paste0(paste0(comp[-length(comp)],collapse = "\\"),"\\")
root = paste0(paste0(comp[1:(length(comp)-1)],collapse = "\\"),"\\")
if(grepl("xlsx",root)){
  root = ""
}
dir = paste0(root,filename," - SERDA result")
dir.create(dir)



data = read_data(data_file)

f = data$f
p = data$p
e = data$e_matrix

mean_e = apply(e, 1, mean, na.rm = TRUE)

if("sample" %in% p$sampleType){
  no_sample = FALSE
}else{
  no_sample = TRUE
}

if(no_sample){
  e = cbind(e,e)

  p = rbind(p,p)
  
  p$sampleType[(length(p$sampleType)/2 + 1) :length(p$sampleType)] = 'sample'
  
}

e_raw = e
sampleType_NA_index = is.na(p$sampleType)

e_na = e[,is.na(p$sampleType)]
p_na = p[is.na(p$sampleType),]

e = e[,!is.na(p$sampleType)]
p = p[!is.na(p$sampleType),]


# Impute missing values first. Missing values bias the algorithm.
for(i in 1:nrow(e)){
  e[i,is.na(e[i,])] = rnorm(sum(is.na(e[i,])),mean = 0.5 * mean(e[i,!is.na(e[i,])]), sd = 0.05 * mean(e[i,!is.na(e[i,])]))
}




# log transform the data for normality.
e = transform(e)
lambda = e$lambda
e = e[[1]]



number_of_cross_validation = 5

qc_index = which(p$sampleType == 'qc')

with_validates = any(!p$sampleType %in% c("qc","sample"))
if(with_validates){
  validates = unique(p$sampleType)
  validates = validates[!validates %in% c("qc",'sample')]
  validates_indexes = list()
  for(i in 1:length(validates)){
    validates_indexes[[i]] = which(p$sampleType %in% validates[i])
  }
  names(validates_indexes) = validates
}else{
  validates = NULL
}

sample_index = which(p$sampleType == 'sample')


e_qc = e[,qc_index]
p_qc = p[qc_index,]
if(with_validates){
  e_validates = list()
  for(i in 1:length(validates)){
    e_validates[[i]] = e[,which(p$sampleType %in% validates[i])]
  }
}

# Even without validate, having this for cross-validation for selecting best hyperparameters.
validates_RSDs = validates_RSDs2 = list()
if(with_validates){
  e_validates_scale = list()
  for(i in 1:length(validates)){
    
    e_validates_scale[[i]] = scale_data(e_validates[[i]])
    
  }
  validates_RSDs[[i]] = 1
  validates_RSDs2[[i]] = 1
}




x_sample_scale = scale_data(e[,sample_index])
x_sample_scale_d = t(x_sample_scale$data_scale)
x_sample_scale_sd = x_sample_scale$sds
x_sample_scale_mean = x_sample_scale$means


x_qc_scale = scale_data(e[,qc_index])
x_qc_scale_d = t(x_qc_scale$data_scale)
x_qc_scale_sd = x_qc_scale$sds
x_qc_scale_mean = x_qc_scale$means


e_none = t(transform(e, forward = FALSE, lambda = lambda)[[1]])
# calculation time
calculation_time = c()

# Hyperparameter choices
patience = 50; layer_units = c(1400); batch_size = 128; epochs = 2000; verbose = 0;activations = c("elu"); drop_out_rate = 0.05; optimizer = "adam";s_index = NULL; t_index = NULL
# ae






e_norms = list()

warning_index = list()


QC_cv_RSDs = QC_cv_RSDs2 = list()


for(l in 1:nrow(layer_units_options)){
  start = Sys.time()
  warning_index[[l]] = c("")
  x_left_normalize = x_left_predict = x_qc_scale_d
  sample_index_temp = 1:nrow(x_left_normalize)
  x_lefts = split(sample_index_temp, sample_index_temp%%number_of_cross_validation)
  
  
  # Cross-validatd QC RSD.
  rsds = rsds2 = c()
  for(i in 1:length(x_lefts)){
    print(i)
    x_train = t(e_qc[,-x_lefts[[i]]])
    x_left = t(e_qc[,x_lefts[[i]]])
    # add noise to input data according to train and target.
    if(layer_units_options$many_training[l]){
      index = (1:layer_units_options$many_training_n[l])%%nrow(x_train)
      index[index==0] = nrow(x_train)
      x_train_output = x_train_input = x_train[index,] #sample(size = many_training_n, 1:nrow(t(e_qc)), replace = TRUE)
    }else{
      x_train_output = x_train
      x_train_input = x_train
    }
    
    x_train_var = apply(x_train_input,2,robust_sd)^2
    target_var = apply(x_left,2,robust_sd)^2
    for(j in 1:ncol(x_train_input)){
      if(target_var[j] > x_train_var[j]){
        set.seed(j)
        x_train_input[,j] = x_train_input[,j]+rnorm(length(x_train_input[,j]), sd = sqrt(target_var[j] - x_train_var[j])*layer_units_options$noise_weight[l])
      }
    }
    
    
    
    x_train_input = t(scale_data(t(x_train_input))[[1]])
    x_train_output = t(scale_data(t(x_train_output))[[1]])
    
    
    # 8/2 split QCs
    set.seed(seed)
    s_index = sample(1:nrow(x_train_input), size = nrow(x_train_input) * 0.8)
    t_index = (1:nrow(x_train_input))[!(1:nrow(x_train_input)) %in% s_index]
    x_test_input = x_train_input[t_index,]
    x_train_input = x_train_input[s_index,]
    x_test_output = x_train_output[t_index,]
    x_train_output = x_train_output[s_index,]

    # train model using training QC and apply to the test QC.
    ae = ae_model(x_train_input, x_test_input,x_train_output, x_test_output, layer_units = as.numeric(layer_units_options$x[l]), verbose = 0, patience = layer_units_options$patience[l], activations = as.character(layer_units_options$activation_function[l]), drop_out_rate = layer_units_options$drop_rates[l], optimizer = "adam")
    
    final_model = ae$final_model
    
    x_left_scale = t(scale_data(t(x_left))[[1]])
    
    x_left_predict = predict(final_model, x_left_scale)
    
    
    # normalize;
    x_left_normalize = x_left_scale
    x_left_predict_o = x_left_predict
    for(j in 1:ncol(x_left_normalize)){
      x_left_normalize[,j] = x_left_scale[,j] - (x_left_predict[,j] - mean(x_left_predict[,j]))
      x_left_normalize[,j] = x_left_normalize[,j] * x_qc_scale_sd[j] + x_qc_scale_mean[j]
    }
    # normalize batch effect
    x_left_normalize = t(rm_batch(t(x_left_normalize),batch = p_qc$batch[x_lefts[[i]]]))
    
    # Exponentiate values back to original values.
    x_left_normalize_exp_current = t(transform(t(x_left_normalize), forward = FALSE, lambda = lambda)[[1]])
    
    raw_means = apply(transform(e_qc[,x_lefts[[i]]], forward = FALSE, lambda = lambda)[[1]],1, mean)
    norm_means = apply(x_left_normalize_exp_current,2, mean)
    
    # put normalized dataset to the original scale.
    tran_temp = transform(e_qc, forward = FALSE, lambda = lambda)[[1]][,x_lefts[[i]]]
    for(k in 1:ncol(x_left_normalize_exp_current)){
      mean_adj = x_left_normalize_exp_current[,k] - (norm_means[k] - raw_means[k])
      if(any(mean_adj<0) | is.na(any(mean_adj<0))){
        # cat(k,": ",sum(mean_adj<0),"\n")
        if(sum(mean_adj<0)>length(mean_adj)/2 | is.na(sum(mean_adj<0)>length(mean_adj)/2)){
          warning_index[[l]] = c(warning_index[[l]], paste0("qc",k))
          mean_adj = tran_temp[k,]
        }else{
          mean_adj[mean_adj<0] = rnorm(sum(mean_adj<0), mean = min(mean_adj[mean_adj>0], na.rm = TRUE)/2, sd = min(mean_adj[mean_adj>0], na.rm = TRUE)/20)
        }  
      }
      x_left_normalize_exp_current[,k] = mean_adj
    }
    # Calculate RSDs.
    rsds[i] = median(RSD(t(x_left_normalize_exp_current),T), na.rm = TRUE)
    rsds2[i] = mean(RSD(t(x_left_normalize_exp_current),T), na.rm = TRUE)
    
  }
  
  QC_cv_RSDs[[l]] = mean(rsds)
  QC_cv_RSDs2[[l]] = mean(rsds2)
  
  if(l == 1){
    QC_cv_RSDs[[l]] = paste0(QC_cv_RSDs[[l]], " (raw: ", median(RSD(transform(e, forward = FALSE, lambda = lambda)[[1]][,qc_index]), na.rm = TRUE),")")
    QC_cv_RSDs2[[l]] = paste0(QC_cv_RSDs2[[l]], " (raw: ", mean(RSD(transform(e, forward = FALSE, lambda = lambda)[[1]][,qc_index]), na.rm = TRUE),")")
  }
  

  
  
  # train on validates
  if(with_validates){
    
    x_target_normalize_exps = list()
    
    for(i in 1:length(validates)){
      
      # add noise to input data according to train and target.
      if(layer_units_options$many_training[l]){
        
        index = (1:layer_units_options$many_training_n[l])%%nrow(t(e_qc))
        index[index==0] = nrow(t(e_qc))
        x_train_output = x_train_input = t(e_qc)[index,] #sample(size = many_training_n, 1:nrow(t(e_qc)), replace = TRUE)
      }else{
        x_train_input = t(e_qc)
        x_train_output = t(e_qc)
      }
      
      
      # add noise to input data according to train and target.
      x_train_var = apply(x_train_input,2,robust_sd)^2
      target_var = apply(t(e_validates[[i]]),2,robust_sd)^2
      for(j in 1:ncol(x_train_input)){
        if(target_var[j] > x_train_var[j]){
          set.seed(j)
          x_train_input[,j] = x_train_input[,j]+rnorm(length(x_train_input[,j]), sd = sqrt(target_var[j] - x_train_var[j])*layer_units_options$noise_weight[l])
          
        }
      }
      
      
      set.seed(seed)
      s_index = sample(1:nrow(x_train_input), size = nrow(x_train_input) * 0.8)
      t_index = (1:nrow(x_train_input))[!(1:nrow(x_train_input)) %in% s_index]
      
      x_train_input = t(scale_data(t(x_train_input))[[1]])
      x_train_output = t(scale_data(t(x_train_output))[[1]])
      
      x_current_train_input = x_train_input[s_index,]
      x_current_test_input = x_train_input[t_index,]
      x_current_train_output = x_train_output[s_index,]
      x_current_test_output = x_train_output[t_index,]
      
      
      ae = ae_model(x_current_train_input,x_current_test_input,x_current_train_output,x_current_test_output, layer_units = as.numeric(layer_units_options$x[l]), verbose = 0, patience = layer_units_options$patience[l], 
                    # activations = "selu", 0.1895314
                    # activations = "softmax", 0.3117842
                    activations = as.character(layer_units_options$activation_function[l]), #0.1873987
                    # activations = "softplus", 0.1896158
                    # activations = "softsign",0.2018778
                    # activations = "relu", 0.21986
                    # activations = "tanh", 0.2008602
                    # activations = "sigmoid", 0.2137153
                    # activations = "hard_sigmoid", #0.2183621
                    # activations = "exponential",#error
                    # activations = "linear",0.1832446
                    drop_out_rate = layer_units_options$drop_rates[l], 
                    optimizer = tolower("Adam"),
                    t_index = NULL,
                    s_index = NULL
      )
      
      
      x_target_d = t(e_validates_scale[[i]]$data_scale)
      x_target_sd = e_validates_scale[[i]]$sds
      x_target_mean = e_validates_scale[[i]]$means
      
      x_target_predict = predict(ae$final_model, x_target_d)
      x_target_normalize = x_target_d
      
      # normalize
      for(j in 1:ncol(x_target_normalize)){
        if(layer_units_options$use_qc_sd_when_final_correct[l]){
          x_target_predict[,j] = x_target_predict[,j] * min(x_target_sd[j], x_qc_scale_sd[j]) + x_qc_scale_mean[j]
        }else{
          x_target_predict[,j] = x_target_predict[,j] * x_target_sd[j] + x_target_mean[j]
        }
        x_target_normalize[,j] = (x_target_normalize[,j]* x_target_sd[j] + x_target_mean[j]) - (x_target_predict[,j] - mean(x_target_predict[,j]))
      }
      
      # normalize batch effect
      x_target_normalize = t(rm_batch(t(x_target_normalize), p[p$sampleType %in% validates[i],]$batch))
      
      # exponentiate it back to original scale.
      x_target_normalize_exp = t(transform(t(x_target_normalize), forward = FALSE, lambda = lambda)[[1]])
      raw_means = apply(transform(e_validates[[i]], forward = FALSE, lambda = lambda)[[1]],1, mean)
      norm_means = apply(x_target_normalize_exp,2, mean)
      for(k in 1:ncol(x_target_normalize_exp)){
        mean_adj = x_target_normalize_exp[,k] - (norm_means[k] - raw_means[k])
        if(any(mean_adj<0) | is.na(any(mean_adj<0))){
          #cat(k,": ",sum(mean_adj<0),"\n")
          if(sum(mean_adj<0)>length(mean_adj)/2 | is.na(sum(mean_adj<0)>length(mean_adj)/2)){
            warning_index[[l]] = c(warning_index[[l]], paste0(validates[i],k))
            # mean_adj = exp(e_validates[[i]])[k,]
            mean_adj = transform(e_validates[[i]], forward = FALSE, lambda = lambda)[[1]][k,]
          }else{
            mean_adj[mean_adj<0] = rnorm(sum(mean_adj<0), mean = min(mean_adj[mean_adj>0], na.rm = TRUE)/2, sd = min(mean_adj[mean_adj>0], na.rm = TRUE)/20)
          }
        }
        x_target_normalize_exp[,k] = mean_adj
      }
    
      
      validates_RSDs[[i]][l] = median(RSD(t(x_target_normalize_exp),T), na.rm = TRUE)
      validates_RSDs2[[i]][l] = mean(RSD(t(x_target_normalize_exp),T), na.rm = TRUE)
      if(l == 1){
        validates_RSDs[[i]][l] = paste0(validates_RSDs[[i]][l], "; (raw: ",median(RSD(transform(e[,validates_indexes[[i]]], forward = FALSE, lambda = lambda)[[1]]), na.rm = TRUE),")")
        validates_RSDs2[[i]][l] = paste0(validates_RSDs2[[i]][l], "; (raw: ",mean(RSD(transform(e[,validates_indexes[[i]]], forward = FALSE, lambda = lambda)[[1]]), na.rm = TRUE),")")
      }
      x_target_normalize_exps[[i]] = x_target_normalize_exp
    }
    names(validates_RSDs) = validates
    names(validates_RSDs2) = validates
    names(x_target_normalize_exps) = validates
    
  }
  
  
 
  # add noise to input data according to train and target.
  if(layer_units_options$many_training[l]){
    index = (1:layer_units_options$many_training_n[l])%%nrow(t(e_qc))
    index[index==0] = nrow(t(e_qc))
    x_train_output = x_train_input = t(e_qc)[index,] #sample(size = many_training_n, 1:nrow(t(e_qc)), replace = TRUE)
  }else{
    x_train_input = t(e_qc)
    x_train_output = t(e_qc)
  }
  
  
  
  # add noise to input data according to train and target.
  x_train_var = apply(x_train_input,2,robust_sd)^2
  target_var = apply(t(e[,sample_index]),2,robust_sd)^2
  # cat("here: ",sum(target_var>x_train_var)/length(target_var))
  for(j in 1:ncol(x_train_input)){
    if(target_var[j] > x_train_var[j]){
      set.seed(j)
      x_train_input[,j] = x_train_input[,j]+rnorm(length(x_train_input[,j]), sd = sqrt(target_var[j] - x_train_var[j])*layer_units_options$noise_weight[l])
    }
  }
  
  set.seed(seed)
  s_index = sample(1:nrow(x_train_input), size = nrow(x_train_input) * 0.8)
  t_index = (1:nrow(x_train_input))[!(1:nrow(x_train_input)) %in% s_index]
  
  x_train_input = t(scale_data(t(x_train_input))[[1]])
  x_train_output = t(scale_data(t(x_train_output))[[1]])

  x_current_train_input = x_train_input[s_index,]
  x_current_test_input = x_train_input[t_index,]
  x_current_train_output = x_train_output[s_index,]
  x_current_test_output = x_train_output[t_index,]
  
  ae = ae_model(x_current_train_input,x_current_test_input,x_current_train_output,x_current_test_output, layer_units = as.numeric(layer_units_options$x[l]), verbose = 0, patience = layer_units_options$patience[l], 
                # activations = "selu", 0.1895314
                # activations = "softmax", 0.3117842
                activations = as.character(layer_units_options$activation_function[l]), #0.1873987
                # activations = "softplus", 0.1896158
                # activations = "softsign",0.2018778
                # activations = "relu", 0.21986
                # activations = "tanh", 0.2008602
                # activations = "sigmoid", 0.2137153
                # activations = "hard_sigmoid", #0.2183621
                # activations = "exponential",#error
                # activations = "linear",0.1832446
                
                drop_out_rate = layer_units_options$drop_rates[l], 
                optimizer = tolower("Adam"),
                t_index = t_index,
                s_index = s_index
                # ,epochs = 300
  )
  
  # predict systematic error.
  x_sample_predict = predict(ae$final_model, x_sample_scale_d)
  x_sample_normalize = x_sample_scale_d
  
  
  # normalize
  for(j in 1:ncol(x_sample_normalize)){
    if(layer_units_options$use_qc_sd_when_final_correct[l]){
      x_sample_predict[,j] = x_sample_predict[,j] * min(x_sample_scale_sd[j], x_qc_scale_sd[j]) + 
        x_sample_scale_mean[j]
    }else{
      x_sample_predict[,j] = x_sample_predict[,j] * x_sample_scale_sd[j] + x_sample_scale_mean[j]
    }
  }
  
  
  for(j in 1:ncol(x_sample_normalize)){
    x_sample_normalize[,j] = (x_sample_normalize[,j]* x_sample_scale_sd[j] + x_sample_scale_mean[j]) - (x_sample_predict[,j] - mean(x_sample_predict[,j]))
  }
  # normalize batch effect
  x_sample_normalize = t(rm_batch(t(x_sample_normalize), p$batch[sample_index]))
  
  # Exponentiate values back to original values.
  x_sample_normalize_exp = t(transform(t(x_sample_normalize), forward = FALSE, lambda = lambda)[[1]])
  
  raw_means = apply(transform(e[,sample_index], forward = FALSE, lambda = lambda)[[1]],1, mean)
  norm_means = apply(x_sample_normalize_exp,2, mean)
  
  transform_temp = transform(e[,sample_index], forward = FALSE, lambda = lambda)[[1]]
  for(k in 1:ncol(x_sample_normalize_exp)){
    mean_adj = x_sample_normalize_exp[,k] - (norm_means[k] - raw_means[k])
    if(any(mean_adj<0) | is.na(any(mean_adj<0))){
      # cat(k,": ",sum(mean_adj<0),"\n")
      if(sum(mean_adj<0)>length(mean_adj)/2 | is.na(sum(mean_adj<0)>length(mean_adj)/2)){
        warning_index[[l]] = c(warning_index[[l]], paste0("sample",k))
        # mean_adj = exp(e[,sample_index])[k,]
        mean_adj = transform_temp[k,]
      }else{
        mean_adj[mean_adj<0] = rnorm(sum(mean_adj<0), mean = min(mean_adj[mean_adj>0], na.rm = TRUE)/2, sd = min(mean_adj[mean_adj>0], na.rm = TRUE)/20)
      }
    }
    x_sample_normalize_exp[,k] = mean_adj
  }
  
  
  
  # Putting all datasets back to original sample order.
  e_norm = transform(e, forward = FALSE, lambda = lambda)[[1]]
  e_norm[,sample_index] = t(x_sample_normalize_exp)
  if(with_validates){
    for(i in 1:length(validates)){
      e_norm[,p$sampleType %in% validates[i]] = t(x_target_normalize_exps[[validates[i]]])
    }
  }
  
  pred_train = ae$pred_train
  if (layer_units_options$many_training[l]) {
    pred_train = pred_train[1:length(qc_index), ]
  }
  
  for (j in 1:ncol(x_qc_scale_d)) {
    old = x_qc_scale_d[, j] * x_qc_scale_sd[j] + x_qc_scale_mean[j]
    pred_temp = pred_train[, j] * x_qc_scale_sd[j] + x_qc_scale_mean[j]
    new  = old - (pred_temp - mean(pred_temp))
    # e_norm[,qc_index][j,] = exp(new)
    e_norm[, qc_index][j, ] = transform(new, forward = FALSE, lambda = lambda[j])[[1]]
  }
  
  raw_means = apply(transform(e_qc, forward = FALSE, lambda = lambda)[[1]],1, mean)
  norm_means = apply(e_norm[,qc_index],1, mean)
  for(k in 1:nrow(e_norm[,qc_index])){
    mean_adj = e_norm[,qc_index][k,] - (norm_means[k] - raw_means[k])
    if(any(mean_adj<0) | is.na(any(mean_adj<0))){
      # cat(k,": ",sum(mean_adj<0),"\n")
      if(sum(mean_adj<0)>length(mean_adj)/2 | is.na(sum(mean_adj<0)>length(mean_adj)/2)){
        warning_index[[l]] = c(warning_index[[l]], paste0("aggregating",k))
        # mean_adj = exp(e_qc)[k,]
        mean_adj = transform(e_qc, forward = FALSE, lambda = lambda)[[1]][k,]
      }else{
        mean_adj[mean_adj<0] = rnorm(sum(mean_adj<0), mean = min(mean_adj[mean_adj>0], na.rm = TRUE)/2, sd = min(mean_adj[mean_adj>0], na.rm = TRUE)/20)
      }
      e_norm[,qc_index][k,] = mean_adj
    }
    e_norm[,qc_index][k,] = mean_adj
  }
  
  e_exp = transform(e, forward = FALSE, lambda = lambda)[[1]]
  sds = apply(e_exp,1,function(x){
    sd(x)
  })
  sds2 = apply(e_norm,1,function(x){
    sd(x)
  })
  
  
  
  if(with_validates){
    for(i in 1:length(validates)){
      validate_index = p$sampleType %in% validates[i]
      for(j in 1:nrow(e_norm)){
        e_norm[j,validate_index] = ((mean(e_exp[j,validate_index]) - mean(e_exp[j,sample_index]))/sds[j] * sd(e_norm[j,]) + mean(e_norm[j,sample_index]))/mean(e_norm[j,validate_index]) * e_norm[j,validate_index]
      }
    }
  }
  
  for(i in 1:nrow(e_norm)){
    e_norm[i,qc_index] = ((mean(e_exp[i,qc_index]) - mean(e_exp[i,sample_index]))/sds[i] * sd(e_norm[i,]) + mean(e_norm[i,sample_index]))/mean(e_norm[i,qc_index]) * e_norm[i,qc_index]
  }
  
  # generate scatter plots and PCA plots.
  if(l == 1){
    
    
    sds = apply(e_none, 2, sd, na.rm=TRUE)
    
    pca = prcomp(e_none[,sds>0], scale. = TRUE)
    
    plot(pca$x[,1], pca$x[,2],
         col = factor(p$sampleType, levels = c('sample','qc',validates))
         # col = colors
    )
  }
  
  
  e_none =  transform(e, forward = FALSE, lambda = lambda)[[1]]
  for(j in 1:nrow(e_norm)){
    if(!any(is.na(e_norm[j,]))){
      if(any(e_norm[j,]<0)){
        set.seed(j)
        e_norm[j,e_norm[j,]<0] = 1/2 * rnorm(sum(e_norm[j,]<0), mean = min(e_norm[j,e_norm[j,]>0], na.rm = TRUE), sd = min(e_norm[j,e_norm[j,]>0], na.rm = TRUE)*0.1)
      }
    }else{
      e_norm[j,] = e_none[j,]
    }
  }
  
  
  e_norm2 = cbind(e_norm, e_na)
  e_norm2[,sampleType_NA_index] = e_na
  e_norm2[,!sampleType_NA_index] = e_norm
  colnames(e_norm2)[sampleType_NA_index] = colnames(e_na)
  colnames(e_norm2)[!sampleType_NA_index] = colnames(e_norm)
  
  
  e_norm2[e_raw == 0] = 0
  e_norm2[is.na(e_raw)] = NA
  
  if(!no_sample){
    png(paste0(dir,"\\",l,"-th PCA.png"))
    
    sds = apply(e_norm2,1,sd,na.rm = TRUE)
    if(any(is.na(sds) | any(sds == 0))){
      
    }
    e_norm_pca_d = e_norm2[!is.na(sds) & !sds==0,]
    
    
    for(i in 1:nrow(e_norm_pca_d)){
      e_norm_pca_d[i,is.na(e_norm_pca_d[i,])] = 0.5 * min(e_norm_pca_d[i,!is.na(e_norm_pca_d[i,])])
    }
    
    pca = prcomp(t(e_norm_pca_d), scale. = TRUE)
    
    
    
    
    
    colorGradient <- colorRampPalette(c("green","red"), alpha=F)
    colors = colorGradient(30)[as.numeric(cut(as.numeric(p$time),breaks = 30))]
    
    plot(pca$x[,1], pca$x[,2], 
         col = factor(data$p$sampleType, levels = c('sample','qc',validates,unique(data$p$sampleType)[!unique(data$p$sampleType) %in% c('sample','qc',validates)]))
         # col = factor(p$biological_group)
         # col = colors
    )
    
    dev.off()
  }
  
  e_norm2 = e_norm2[,1:(ncol(e_norm2)/2)]
  
  
  
  
  fwrite(data.table(label = f$label,e_norm2),paste0(dir,"\\",l,"-th dataset.csv"))
  
  
  calculation_time[l] = Sys.time() - start
  fwrite(data.table(layer_units_options[1:l,],QC_cv_RSDs, QC_cv_RSDs2, do.call("cbind",validates_RSDs), do.call("cbind",validates_RSDs2),calculation_time),paste0(dir,"\\",'hyperparameter tuning-performance.csv'))
  
  
  if(scatter_plot & l==1 & !no_sample){ # time consuming.
    col = factor(p$sampleType, levels = c('sample','qc',validates))
    dots = c(1,16,rep(16, length(validates)))[as.numeric(col)]
    
    cat("Generating scatter plots for each compounds. Please be patient...\n")
    dir.create(paste0(dir,"\\","scatter plots"," - ",l))
    
    normalized = e_norm
    # e_none =  exp(e)
    
    e_none =  transform(e, forward = FALSE, lambda = lambda)[[1]]
    
    for(j in 1:nrow(e_none)){
      png(paste0(dir,"\\","scatter plots - ",l,"\\",j,"th.png"), width = 480*2, height = 480)
      par(mfrow=c(1,2))
      ylim = c(min(e_none[j,]), max(e_none[j,],normalized[j,]))
      if(Inf %in% ylim){
        ylim[2] = max(e_none[j,!is.infinite(e_none[j,])],normalized[j,!is.infinite(normalized[j,])])*1.1
      }
      if(sum(is.na(ylim))<1){
        
        
        plot(p$time,e_none[j,], col = factor(p$sampleType, levels = c('sample','qc',validates)), ylim = ylim, main = f$label[j], pch = dots)
        # abline(v = 190)
        
        
        plot(p$time,normalized[j,], col = factor(p$sampleType, levels = c('sample','qc',validates)), ylim = ylim, pch = dots)
        # abline(v = 190)
      };j = j+1
      
      
      dev.off()
    }
    
  }
}





















































































