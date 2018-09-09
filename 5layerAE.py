from keras.layers import Input, Dense
from keras import optimizers, initializers
from keras.models import Model
import sys
import csv


initWeight = float(0)
input_data = Input(shape=(16,))

encodedh = Dense(8, activation='sigmoid', kernel_initializer=initializers.Constant(value=initWeight),
                 bias_initializer='zero')(input_data)
encoded = Dense(4, activation='sigmoid', kernel_initializer=initializers.Constant(value=initWeight),
                bias_initializer='zero')(encodedh)
decodedh = Dense(8, activation='sigmoid', kernel_initializer=initializers.Constant(value=initWeight),
                 bias_initializer='zero')(encoded)
decoded = Dense(16, activation='sigmoid', kernel_initializer=initializers.Constant(value=initWeight),
                bias_initializer='zero')(decodedh)


autoencoder = Model(input_data, decoded)
encoderh = Model(input_data, encodedh)
encoder = Model(input_data, encoded)
decoderh = Model(input_data, decodedh)


# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

rms = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
autoencoder.compile(optimizer=rms, loss='binary_crossentropy')


x_train = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]


autoencoder.fit(x_train, x_train,
                nb_epoch=1500000,
                verbose=2,
                batch_size=256,
                shuffle=True,
                validation_data=(x_train, x_train))


half_encoded = encoderh.predict(x_train)
print(half_encoded)

encoded_data = encoder.predict(x_train)
print(encoded_data)

half_decoded = decoderh.predict(x_train)
print(half_decoded)

decoded_data = autoencoder.predict(x_train)
print(decoded_data)

with open('5l_i%dhalfencoded.csv' % int(initWeight*100), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerows(half_encoded)

with open('5l_i%d_encoded.csv' % int(initWeight*100), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerows(encoded_data)

with open('5l_i%d_halfdecoded.csv' % int(initWeight*100), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerows(half_decoded)

with open('5l_i%d_decoded.csv' % int(initWeight*100), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerows(decoded_data)

print('initial weight = %f' % initWeight)
