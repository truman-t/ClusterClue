from tensorflow import device
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from functions import clusterClue	


# Import validator model
with device('/cpu:0'):
    validator = Sequential([
    Conv2D(16, 3, activation='relu', input_shape=(64, 64,1), padding='same'),
    MaxPooling2D(pool_size=(4, 4)),
    Dropout(0.2),
    BatchNormalization(),
    Conv2D(32, 3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    BatchNormalization(),
    Conv2D(64, 3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    BatchNormalization(),
    Conv2D(128, 3, activation='relu', padding='same'),
    GlobalAveragePooling2D(),
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(20, activation='relu'),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
    ])
    validator.load_weights('../nn_weights/checkpoints')
    
    
# Run ClusterClue pipeline    
r = clusterClue(105, -4, .2, 2800, validator=validator) # location around 'Ivanov 9' cluster
# print result sample
print(r.head(5).to_string())


