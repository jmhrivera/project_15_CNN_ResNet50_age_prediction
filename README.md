# project_15_CNN_ResNet50_age_prediction

## Project Description

This project employs deep learning techniques to predict the real age from facial images. The model is built using the `TensorFlow` library and `PIL` (Python Imaging Library) for image manipulation. The base model used is `ResNet50`, a pre-trained convolutional neural network model.

## Project Structure

- **load_train(path)**: Function to load the training dataset from a specified path.
- **load_test(path)**: Function to load the validation dataset from a specified path.
- **create_model(input_shape)**: Function to create the age prediction model using `ResNet50` as the base.
- **train_model(model, train_data, test_data, batch_size=None, epochs=20, steps_per_epoch=None, validation_steps=None)**: Function to train the model using the training and validation data.

## Dependencies

Make sure you have the following dependencies installed:

```bash
pip install tensorflow
pip install pandas
pip install numpy
pip install matplotlib
pip install Pillow
```
