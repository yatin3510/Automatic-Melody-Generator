from preprocessing import generating_training_sequences, sequence_length
import tensorflow as tf

OUTPUT_UNITS = 111 # Number of items in our mappings dictionary
NUM_UNITS = [64] # we are passing it as a list because the LSTM layer can have more than 1 hidden layer
LOSS = "sparse_categorical_crossentropy" # Loss function
LEARNING_RATE = 0.01 # Learning rate of the model

EPOCHS = 10
BATCH_SIZE = 20 

SAVE_MODEL_PATH = "model.h5"



def build_model(output_units, num_units, loss, learning_rate):

    # create the model architecture
    input = tf.keras.layers.Input(
         shape = (None, # written "None" here because it allows us to have as many neurons for input data's 1st dimension(i.e., 64) I still don't know why didn't we write 64 explicitly
                 output_units # it corresponds to the 2nd dimension of the one-hot encoded input data
                 )
                )

    x = tf.keras.layers.LSTM(num_units[0])(input) # This specifies the number of neurons in LSTM layer
    x = tf.keras.layers.Dropout(0.2)(x) # just for regularisation, preventing overfitting

    output = tf.keras.layers.Dense(output_units, activation = "softmax")(x)

    model = tf.keras.Model(input, output)


    # compile the model
    model.compile(loss = loss, 
                optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE),
                metrics = ["accuracy"]
                )

    model.summary()

    return model


def train(output_units = OUTPUT_UNITS, num_units = NUM_UNITS, loss = LOSS, learning_rate = LEARNING_RATE):

    # generate the training sequences
    inputs, targets = generating_training_sequences(sequence_length)

    # build the LSTM Neural Network
    model = build_model(output_units, num_units, loss, learning_rate)

    # train the model
    model.fit(inputs, targets, epochs = EPOCHS, batch_size = BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
    train()