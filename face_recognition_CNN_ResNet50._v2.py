import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def load_train(path):
    labels = pd.read_csv(path + 'labels.csv')
    train_datagen = ImageDataGenerator(
        validation_split=0.25,
        horizontal_flip=True,
        rescale=1./255)
    train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=16,
        class_mode='raw',
        subset='training',
        seed=12345)

    return train_gen_flow

def load_test(path):
    labels = pd.read_csv(path + 'labels.csv')
    test_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255)
    test_gen_flow = test_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=16,
        class_mode='raw',
        subset='validation',
        seed=12345)

    return test_gen_flow

def create_model(input_shape):
    backbone = ResNet50(weights='imagenet',
                        input_shape=input_shape,
                        include_top=False)

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))

    optimizer = Adam(lr=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)

    return model


###

# 2024-05-27 20:20:23.684270: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer.so.6
# 2024-05-27 20:20:23.765938: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer_plugin.so.6
# Using TensorFlow backend.
# Found 5694 validated image filenames.
# Found 1897 validated image filenames.
# 2024-05-27 20:20:28.171111: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
# 2024-05-27 20:20:28.260613: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2024-05-27 20:20:28.260811: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
# pciBusID: 0000:00:1e.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
# coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.78GiB deviceMemoryBandwidth: 836.37GiB/s
# 2024-05-27 20:20:28.260847: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2024-05-27 20:20:28.260897: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2024-05-27 20:20:28.330060: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
# 2024-05-27 20:20:28.348247: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
# 2024-05-27 20:20:28.488728: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
# 2024-05-27 20:20:28.517682: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
# 2024-05-27 20:20:28.517804: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2024-05-27 20:20:28.518013: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2024-05-27 20:20:28.518250: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2024-05-27 20:20:28.518379: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
# 2024-05-27 20:20:28.518735: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# 2024-05-27 20:20:28.540683: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2300000000 Hz
# 2024-05-27 20:20:28.543248: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x423d8d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
# 2024-05-27 20:20:28.543311: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
# 2024-05-27 20:20:28.710050: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2024-05-27 20:20:28.710337: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e9ac0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
# 2024-05-27 20:20:28.710364: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
# 2024-05-27 20:20:28.710615: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2024-05-27 20:20:28.710801: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
# pciBusID: 0000:00:1e.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
# coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.78GiB deviceMemoryBandwidth: 836.37GiB/s
# 2024-05-27 20:20:28.710861: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2024-05-27 20:20:28.710882: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2024-05-27 20:20:28.710937: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
# 2024-05-27 20:20:28.710962: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
# 2024-05-27 20:20:28.710981: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
# 2024-05-27 20:20:28.711002: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
# 2024-05-27 20:20:28.711025: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2024-05-27 20:20:28.711086: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2024-05-27 20:20:28.711324: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2024-05-27 20:20:28.711493: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
# 2024-05-27 20:20:28.719045: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2024-05-27 20:20:30.765887: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2024-05-27 20:20:30.765934: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0 
# 2024-05-27 20:20:30.765948: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N 
# 2024-05-27 20:20:30.767194: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2024-05-27 20:20:30.767550: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2024-05-27 20:20:30.767788: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
# 2024-05-27 20:20:30.767835: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14988 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:1e.0, compute capability: 7.0)
# Downloading data from https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

#     8192/94765736 [..............................] - ETA: 2s
# 12304384/94765736 [==>...........................] - ETA: 0s
# 25157632/94765736 [======>.......................] - ETA: 0s
# 38166528/94765736 [===========>..................] - ETA: 0s
# 50929664/94765736 [===============>..............] - ETA: 0s
# 64028672/94765736 [===================>..........] - ETA: 0s
# 76783616/94765736 [=======================>......] - ETA: 0s
# 89219072/94765736 [===========================>..] - ETA: 0s
# 94773248/94765736 [==============================] - 0s 0us/step
# <class 'tensorflow.python.keras.engine.sequential.Sequential'>
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# Train for 356 steps, validate for 119 steps
# Epoch 1/20
# 2024-05-27 20:20:48.686546: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2024-05-27 20:20:50.108893: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 356/356 - 63s - loss: 198.4074 - mae: 10.6772 - val_loss: 328.4897 - val_mae: 13.4490
# Epoch 2/20
# 356/356 - 38s - loss: 132.0247 - mae: 8.7466 - val_loss: 213.4496 - val_mae: 11.4061
# Epoch 3/20
# 356/356 - 38s - loss: 108.7750 - mae: 7.9863 - val_loss: 173.9609 - val_mae: 9.8523
# Epoch 4/20
# 356/356 - 38s - loss: 96.2830 - mae: 7.5162 - val_loss: 112.5010 - val_mae: 8.0929
# Epoch 5/20
# 356/356 - 38s - loss: 76.6386 - mae: 6.6678 - val_loss: 151.5111 - val_mae: 9.5880
# Epoch 6/20
# 356/356 - 38s - loss: 66.3609 - mae: 6.2434 - val_loss: 114.1250 - val_mae: 8.3635
# Epoch 7/20
# 356/356 - 38s - loss: 56.6177 - mae: 5.7521 - val_loss: 130.6413 - val_mae: 8.7276
# Epoch 8/20
# 356/356 - 38s - loss: 48.8699 - mae: 5.3878 - val_loss: 91.5628 - val_mae: 7.0929
# Epoch 9/20
# 356/356 - 38s - loss: 38.5361 - mae: 4.7909 - val_loss: 174.3415 - val_mae: 10.9758
# Epoch 10/20
# 356/356 - 38s - loss: 32.3184 - mae: 4.4133 - val_loss: 87.6476 - val_mae: 7.0334
# Epoch 11/20
# 356/356 - 38s - loss: 28.9861 - mae: 4.1308 - val_loss: 153.0733 - val_mae: 9.2271
# Epoch 12/20
# 356/356 - 38s - loss: 27.6008 - mae: 4.0496 - val_loss: 88.6387 - val_mae: 7.0580
# Epoch 13/20
# 356/356 - 38s - loss: 25.7886 - mae: 3.9215 - val_loss: 121.5979 - val_mae: 8.5732
# Epoch 14/20
# 356/356 - 38s - loss: 29.7126 - mae: 4.1399 - val_loss: 118.2702 - val_mae: 8.3339
# Epoch 15/20
# 356/356 - 38s - loss: 25.8477 - mae: 3.9102 - val_loss: 86.9538 - val_mae: 7.1605
# Epoch 16/20
# 356/356 - 38s - loss: 20.0663 - mae: 3.4560 - val_loss: 82.1982 - val_mae: 6.7638
# Epoch 17/20
# 356/356 - 38s - loss: 16.6578 - mae: 3.1425 - val_loss: 78.4234 - val_mae: 6.6702
# Epoch 18/20
# 356/356 - 38s - loss: 15.9948 - mae: 3.0645 - val_loss: 90.2018 - val_mae: 7.0544
# Epoch 19/20
# 356/356 - 38s - loss: 15.2473 - mae: 3.0050 - val_loss: 88.3793 - val_mae: 7.2842
# Epoch 20/20
# 356/356 - 38s - loss: 14.8930 - mae: 2.9356 - val_loss: 84.4414 - val_mae: 6.8639
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# 119/119 - 9s - loss: 84.4414 - mae: 6.8639
# Test MAE: 6.8639