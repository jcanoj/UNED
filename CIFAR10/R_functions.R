# Función que carga e instala en caso de no estarlo la librería de keras
carga_librerias <- function()
{
  if (!suppressWarnings(suppressMessages(require(keras))))
  {
    install.packages("keras")
    library(keras)
  }
  if (!suppressWarnings(suppressMessages(require(imager))))
  {
    install.packages("imager")
    library(imager)
  }
  if (!suppressWarnings(suppressMessages(require(tidyverse))))
  {
    install.packages("tidyverse")
    library(tidyverse)
  }
  if (!suppressWarnings(suppressMessages(require(caret))))
  {
    install.packages("caret")
    library(caret)
  }
  if (!suppressWarnings(suppressMessages(require(yardstick ))))
  {
    install.packages("yardstick ")
    library(yardstick )
  }
}

# Función que descarga y descomprimi las imagénes del dataset
carga_dataset <- function()
{

  # Descargamos el dataset
  cifar10 <- dataset_cifar10()
  
  # Hacemos estas variables globales, normalizamos en 0,1
  x_train <<- cifar10$train$x/255
  x_test <<- cifar10$test$x/255
  y_train <<- to_categorical(cifar10$train$y, num_classes = 10)
  y_test <<- to_categorical(cifar10$test$y, num_classes = 10)
  
}



visualiza_imagenes_random <- function( )
{
  par(mfrow=c(5,3))
  par(mar=c(1,1,1,1))

  for( i in 1:15)
  {
    plot(as.raster(x_train[runif(1,1,100),,,]))
  }
}



generar_imagenes_augmentation <- function()
{
  dir.create("./flores/train_augmentation")
  train_dir_augmentation_daisy <<- "./flores/train_augmentation/daisy"
  dir.create(train_dir_augmentation_daisy)
  train_dir_augmentation_dandelion <<- "./flores/train_augmentation/dandelion"
  dir.create(train_dir_augmentation_dandelion)
  train_dir_augmentation_roses <<- "./flores/train_augmentation/roses"
  dir.create(train_dir_augmentation_roses)
  train_dir_augmentation_sunflowers <<- "./flores/train_augmentation/sunflowers"
  dir.create(train_dir_augmentation_sunflowers)
  train_dir_augmentation_tulips <<- "./flores/train_augmentation/tulips"
  dir.create(train_dir_augmentation_tulips)
  
  
  train_generator <- image_data_generator(
    rescale = 1/255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = TRUE,
    fill_mode = "nearest"
  )
  
  imagenes <- list.files(paste0(train_dir, "/daisy"))
  imagenes.array <- array(NA,  dim = c(length(imagenes), 150, 150, 3) )
  
  for (i in 1:length(imagenes)) {
    
    temp <- image_load(paste0(train_dir,"/daisy/",imagenes[i]), 
                       target_size = c(150,150),  
                       grayscale = FALSE)
    
    temp.array <- image_to_array(temp, data_format = "channels_last")
    temp.array <- array_reshape(temp.array, c(1, dim(temp.array)))
    imagenes.array[i,,,] <- temp.array
    
  }
  
  generador <- flow_images_from_data(x = imagenes.array,
                                     generator = train_generator,
                                     batch_size = 1500,
                                     save_to_dir = train_dir_augmentation_daisy,
                                     save_format = "jpeg"
  )
  
  batch_1 <- generator_next(generador)
  batch_2 <- generator_next(generador)
  batch_3 <- generator_next(generador)
  
  imagenes <- list.files(paste0(train_dir, "/dandelion"))
  imagenes.array <- array(NA,  dim = c(length(imagenes), 150, 150, 3) )
  
  for (i in 1:length(imagenes)) {
    
    temp <- image_load(paste0(train_dir,"/dandelion/",imagenes[i]), 
                       target_size = c(150,150),  
                       grayscale = FALSE)
    
    temp.array <- image_to_array(temp, data_format = "channels_last")
    temp.array <- array_reshape(temp.array, c(1, dim(temp.array)))
    imagenes.array[i,,,] <- temp.array
    
  }
  
  generador <- flow_images_from_data(x = imagenes.array,
                                     generator = train_generator,
                                     batch_size = 1500,
                                     save_to_dir = train_dir_augmentation_dandelion,
                                     save_format = "jpeg"
  )
  
  batch_1 <- generator_next(generador)
  batch_2 <- generator_next(generador)
  batch_3 <- generator_next(generador)
 
  
  imagenes <- list.files(paste0(train_dir, "/roses"))
  imagenes.array <- array(NA,  dim = c(length(imagenes), 150, 150, 3) )
  
  for (i in 1:length(imagenes)) {
    
    temp <- image_load(paste0(train_dir,"/roses/",imagenes[i]), 
                       target_size = c(150,150),  
                       grayscale = FALSE)
    
    temp.array <- image_to_array(temp, data_format = "channels_last")
    temp.array <- array_reshape(temp.array, c(1, dim(temp.array)))
    imagenes.array[i,,,] <- temp.array
    
  }
  
  generador <- flow_images_from_data(x = imagenes.array,
                                     generator = train_generator,
                                     batch_size = 1500,
                                     save_to_dir = train_dir_augmentation_roses,
                                     save_format = "jpeg"
  )
  
  batch_1 <- generator_next(generador)
  batch_2 <- generator_next(generador)
  batch_3 <- generator_next(generador)
  
  imagenes <- list.files(paste0(train_dir, "/sunflowers"))
  imagenes.array <- array(NA,  dim = c(length(imagenes), 150, 150, 3) )
  
  for (i in 1:length(imagenes)) {
    
    temp <- image_load(paste0(train_dir,"/sunflowers/",imagenes[i]), 
                       target_size = c(150,150),  
                       grayscale = FALSE)
    
    temp.array <- image_to_array(temp, data_format = "channels_last")
    temp.array <- array_reshape(temp.array, c(1, dim(temp.array)))
    imagenes.array[i,,,] <- temp.array
    
  }
  
  generador <- flow_images_from_data(x = imagenes.array,
                                     generator = train_generator,
                                     batch_size = 1500,
                                     save_to_dir = train_dir_augmentation_sunflowers,
                                     save_format = "jpeg"
  )
  
  batch_1 <- generator_next(generador)
  batch_2 <- generator_next(generador)
  batch_3 <- generator_next(generador)
  
  imagenes <- list.files(paste0(train_dir, "/tulips"))
  imagenes.array <- array(NA,  dim = c(length(imagenes), 150, 150, 3) )
  
  for (i in 1:length(imagenes)) {
    
    temp <- image_load(paste0(train_dir,"/tulips/",imagenes[i]), 
                       target_size = c(150,150),  
                       grayscale = FALSE)
    
    temp.array <- image_to_array(temp, data_format = "channels_last")
    temp.array <- array_reshape(temp.array, c(1, dim(temp.array)))
    imagenes.array[i,,,] <- temp.array
    
  }
  
  generador <- flow_images_from_data(x = imagenes.array,
                                     generator = train_generator,
                                     batch_size = 1500,
                                     save_to_dir = train_dir_augmentation_tulips,
                                     save_format = "jpeg"
  )
  
  batch_1 <- generator_next(generador)
  batch_2 <- generator_next(generador)
  batch_3 <- generator_next(generador)
  
}



crea_modelo_1 <- function()
{
  # Construimos el modelo
  modelo_1 <- keras_model_sequential() %>%
    layer_conv_2d(
      filters = 16,
      kernel_size = c(3, 3),
      activation = "relu",
      input_shape = c(32, 32, 3)
    ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(
      filters = 32,
      kernel_size = c(3, 3),
      activation = "relu"
    ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_dense(units = 10, activation = "softmax")
  
  # Asignamos la función de pérdida
  # el optimizador
  # y la métrica
  modelo_1 %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy")
  )
  
  print(modelo_1) 
  modelo_1
}

crea_modelo_2 <- function()
{
  # Construimos el modelo
  modelo_2 <- keras_model_sequential() %>%
    layer_conv_2d(
      filters = 32,
      kernel_size = c(3, 3),
      activation = "relu",
      input_shape = c(32, 32, 3)
    ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(
      filters = 64,
      kernel_size = c(3, 3),
      activation = "relu"
    ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(
      filters = 128,
      kernel_size = c(3, 3),
      activation = "relu"
    ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_dense(units = 10, activation = "softmax")
  
  # Asignamos la función de pérdida
  # el optimizador
  # y la métrica
  modelo_2 %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy")
  )
  
  print(modelo_2)  
  modelo_2
}

crea_modelo_3 <- function()
{
  modelo_3 <- keras_model_sequential() %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu",
                  input_shape = c(32, 32, 3)) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dropout(0.25) %>%
    layer_dense(units = 10, activation = "softmax")
  
  # Asignamos la función de pérdida
  # el optimizador
  # y la métrica
  modelo_3 %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy")
  )
  
  print(modelo_3)
  modelo_3
}

crea_modelo_4 <- function()
{
  modelo_4 <- keras_model_sequential() %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu",
                  input_shape = c(32, 32, 3)) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dropout(0.25) %>%
    layer_dense(units = 10, activation = "softmax")
  
  # Asignamos la función de pérdida
  # el optimizador
  # y la métrica
  modelo_4 %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy")
  )
  
  print(modelo_4) 
  modelo_4
}

crea_modelo_5 <- function()
{
  modelo_5 <- keras_model_sequential() %>%
      conv_base %>% 
      layer_flatten() %>% 
      layer_dense(units = 256, activation = "relu") %>% 
      layer_dense(units = 10, activation = "softmax")
  
  # Evitamos que se entrenen los parámetros de la red preentrenada
  freeze_weights(conv_base)
  
  
  # Asignamos la función de pérdida
  # el optimizador
  # y la métrica
  modelo_5 %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy")
  )
  
  print(modelo_5) 
  modelo_5
}

crea_modelo_6 <- function()
{
  
  # Evitamos que se entrenen los parámetros de la red preentrenada
  # Vamos a habilitar el entrenamiento sólo del último bloque de la
  # red preentrenada conv5_block3
  
  # Nos aseguramos que todo está deshabilitado para entrenar
  freeze_weights(conv_base)
  # Habilitamos sólo a partir de conv5_block3_3_conv
  unfreeze_weights(conv_base, from = "block5_conv1")
  
  
  modelo_6 <- keras_model_sequential() %>%
    conv_base %>% 
    layer_flatten() %>% 
    layer_dense(units = 256, activation = "relu") %>% 
    layer_dense(units = 10, activation = "softmax")
  

  
  # Asignamos la función de pérdida
  # el optimizador
  # y la métrica
  modelo_6 %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy")
  )
  
  print(modelo_6) 
  modelo_6
}


entrena_modelo <- function (modelo)
{
  # Ejecutamos el entrenamiento
  # guardando el histórico de la
  # función de pérdida y la métrica
  historico <- modelo %>% fit(
    x_train,
    y_train,
    epochs = 10,
    validation_data = list(x_test,y_test),
    batch_size = 64
  )
  
  historico
  
}

entrena_modelo_augmentation <- function (modelo)
{
  # Ejecutamos el entrenamiento
  # guardando el histórico de la
  # función de pérdida y la métrica
  # Creamos el generador
  datagen <- image_data_generator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = TRUE
  )
  
  training_image_flow = flow_images_from_data(x_train, y_train, datagen, batch_size = 32)
  

  historico <- modelo %>% fit_generator(
    training_image_flow,
    steps_per_epoch = as.integer(50000/32), 
    epochs = 10, 
    validation_data = list(x_test, y_test)
  )
  
  historico
  
}

muestra_historico_entrenamiento <- function(historico)
{
  # Dibujamos la gráfica de evolución
  # del entrenamiento
  plot(historico)
}

evalua_modelo <- function( modelo )
{

  # Vamos a usar ahora los datos de test con nuestro modelo
  # Evaluamos los datos de test
  # Obtenemos su pérdida y y accuracy
  evalua <- evaluate(modelo, x_test, y_test, verbose=0)
  
  evalua
}

prediccion_clases <- function(mi_modelo)
{
  predicciones <- predict_classes(mi_modelo, x_test)
  # La columna indica la clase, el original es del 0 al 4, hay que restar 1 al final
  #prediccion_clases <- unlist(lapply(1:nrow(predicciones), function(i) which(predicciones[i,] == max(predicciones[i,]))))-1
  predicciones
  #predicciones
}

confusion_matrix <- function( miprediccion, mireal)
{
  mireal <- unlist(lapply(1:nrow(mireal), function(i) which(mireal[i,] == max(mireal[i,]))))-1
  confusion_matrix <- confusionMatrix(as.factor(miprediccion),as.factor(mireal))
  confusion_matrix
}

ggplot_confusion_matrix <- function(matriz_confusion){
  
  print(autoplot(conf_mat(matriz_confusion$table), type="heatmap") + scale_fill_gradient(low = "white", high = "steelblue"))
  
}


guarda_modelo <- function(modelo, nombre)
{
  # Guardamos el modelo para usarlo cuando lo necesitemos
  modelo %>% save_model_hdf5(nombre, overwrite = TRUE)
}

carga_modelo <- function(nombre)
{
  # Cargamos el modelo previamente usado
  modelo <- load_model_hdf5(nombre)
  
  modelo
}