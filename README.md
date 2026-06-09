# Multimodal-ML-Housing-Price-Prediction-Using-Images-and-Tabular-Data
Submited By :
Abdul Wasay
DHC-1768
Objective of the Task
The objective of this task is to build a multimodal machine learning model to predict housing prices. The model processes both structured tabular data and unstructured image data simultaneously to make a single unified prediction.

Methodology and Approach
Dataset Preparation: The California Housing dataset was used for tabular features. A synthetic image generation function was used to create corresponding house images to demonstrate multimodal capabilities.
Data Preprocessing: Tabular data was standardized using Scikit-learns StandardScaler. Image data was processed using TensorFlows MobileNetV2 preprocess_input. A tf.data pipeline was created to handle batching, shuffling, and prefetching.
Model Architecture: Two separate neural network branches were created. The image branch uses a pre-trained MobileNetV2 model as a feature extractor. The tabular branch uses dense layers with batch normalization and dropout. Features from both branches are concatenated and passed through fully connected layers to output a single continuous price prediction.
Training and Evaluation: The model was trained using Mean Squared Error loss with the Adam optimizer. EarlyStopping and ReduceLROnPlateau callbacks were implemented to prevent overfitting.

Key Results or Observations
The multimodal model successfully fused numerical features and image representations to perform regression tasks.
The model effectively learned from both modalities, as seen in the generated training curves.
Final evaluation metrics including Mean Absolute Error and Root Mean Squared Error were calculated on the original price scale to measure real world accuracy.
The trained model is successfully saved as multimodal_housing_model.keras, alongside generated prediction scatter plots and training curve graphs.
