import os
import music21 as m21
import json
import numpy
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


DATASET_PATH = "C:\\Users\\Lenovo\\Desktop\\Project\\essen\\europa\\deutschl\\fink"
SAVED_SONGS = "C:\\Users\\Lenovo\\Desktop\\Project\\Saved_songs"
SINGLE_FILE_DATASET = "C:\\Users\\Lenovo\\Desktop\\Project\\Single_file_dataset"
SINGLE_SONG_SAVING_PATH = "C:\\Users\\Lenovo\\Desktop\\Project\\Single_file_dataset"
MAPPING_PATH = "C:\\Users\\Lenovo\\Desktop\\Project\\mapping.json"

sequence_length = 64 # This is going to be discussed in the next class
'''
I know if you will come back here after a month or so, then you won't be able to understand what the hell is this ACCEPTABLE_DURATIONS list for?
But I am telling you that you don't have to worry at all. I will explain it all.

All the songs have some notes, you understand this right.
And a single note can have different durations, remember those headphone like signs flying here
and there representing music notes, for example the logo of Wynk Music also represents an Eigth of Note
,the logo of Xiaomi Music represents a half note.

I am not talking about CDEFGAB. I am talking about the duration for whihc a note is played.
Those durations are representedin the ACCEPTABLE_DURATIONS list.

For more info, see this image : https://www.pinterest.com/pin/86623992819286093/
'''

# Dotted note means adding half of the duration in the current note
ACCEPTABLE_DURATIONS = [
    0.25, # Sixteenth of a note
    0.5, # Eighth of a note
    0.75, # Dotted Eighth of a note
    1, # Quarter of a note
    1.5, # Dotted Quarter of a note
    2, # Half note
    3, # Dotted Half note
    4 # One full note
]

def load_songs(DATASET_PATH):
    songs = []
    for path, subdirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    # print(len(songs))
    return songs

def has_acceptable_duration(song, acceptable_durations):
    # We 
    for note in song.flat.notes:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def transpose(song):
    
    # estimating the key from the somg
    key = song.analyze("key")
    # print(key)

    # getting the interval by which we should shift the score to get "C - major" or "A - minor"
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    
    # shifting the score by size of interval
    transposed_song = song.transpose(interval)
    key1 = transposed_song.analyze("key")

    # print(key1)

    return transposed_song

def encode_song(song, time_step = 0.25):

    encoded_song = []

    for event in song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # this statement assigns the midi value of the note to "symbol" variable

        elif isinstance(event, m21.note.Rest):
            symbol = "r"
            
        steps = int(event.duration.quarterLength/time_step)
        # The above statement is very difficult to understand but I will try my best
        ''' 
            Think of it this way. You are given quarter length which is length of a note in a bar, i.e
            0.25 -> 1/16th of a note
            0.5 -> 1/8th of a note
            1 -> 1/4th of a note, and so on

            And we have taken time step by default to 0.25 because that is what the minimum quarter length
            of a note can be on a bar.

            So, basically we are dividing the given quarter length by the minimum quarter length, so that
            we can get the number of steps for that particular note, meaning -> how much length it will have
            on the bar.
        '''

        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
     

    # converting encode_song into a string
    """We are using map() function here because not all the elements in the 
    encoded_song[] list are strings, so we first map them to str datatype and 
    then join all of them together to """
    encoded_song_str = " ".join(map(str, encoded_song))
    return encoded_song_str

def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

def single_file_converter(SAVED_SONGS, SINGLE_SONG_SAVING_PATH, sequence_length):

    new_song_delimiter = "/ " * sequence_length
    # I will  explain shortly why we keep delimiter same as that of the sequence length

    songs = ""

    for path, subdirs, files in os.walk(SAVED_SONGS): 
       for file in files:
           file_path = os.path.join(path, file)

           song = load(file_path)
           songs = songs + song + " " + new_song_delimiter

    songs = songs[:-1]

    # Save this file at another location

    with open(os.path.join(SINGLE_SONG_SAVING_PATH, "Single_collection_of_songs.txt"), "w") as fp:
        fp.write(songs)

    return songs

def create_mappings(composite_song, mapping_path):
    """We are creating a map because our NN can understand only integers not string datatype, so we are mapping 
    each character to an integer datatype"""
    
    # Remove all the duplicates from the composite song
    composite_song = composite_song.split()
    vocabulary = list(set(composite_song))

    # Create a dictionary "vocabulary" with each unique value in the composite song mapped to a unique number
    mappings = dict()
    for i in range(108):
        mappings[i + 21] = i

    mappings["/"] = 108
    mappings["r"] = 109
    mappings["_"] = 110

    # Save mappings to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent = 4)
    
    

def convert_songs_to_int(single_song):
    int_songs = []

    # load the hash values of notes form json file
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # convert the midi values of notes in single_collection_of_songs into respective hash values
    single_song = single_song.split()

    for symbol in single_song:
        int_songs.append(mappings[symbol])

    return int_songs

def generating_training_sequences(sequence_length):
    # load the songs and mapping them to their hash values
    path = os.path.join(SINGLE_FILE_DATASET, "Single_collection_of_songs.txt")
    with open(path, "r") as fp:
        songs = fp.read()

    int_songs = convert_songs_to_int(songs) # this variable consists of that single song ( which contained all the songs )
                                            # in their respective hash values
    
    # generating the training sequences
    num_of_sequences = len(int_songs) - sequence_length

    inputs = []
    targets = []

    for i in range(num_of_sequences):
        inputs.append(int_songs[i : i + sequence_length]) # Dimension -> [num_of_sequences * sequence_length]
        targets.append(int_songs[i + sequence_length]) # Dimension -> num_of_sequences

    # One hot encoding

    """
    This is similar to creating dummy columns in Data Wrangling, like Deleting the "Sex" column
    and creating "Male" Column and assigning it 0 and 1 values for each row
    """

    # Now the dimensions of input layer changes to [num_of_sequences * sequence_length * len(mappings)]
    mappings_size = len(set(int_songs))
    inputs = to_categorical(inputs, num_classes = 111)
    targets = numpy.array(targets)

    print(f"There are {len(targets)} inputs")

    return inputs, targets



def pre_processing():

    # Step 1
    print("Loading Songs...")
    songs = load_songs(DATASET_PATH)
    print(f"{len(songs)} songs are loaded\n")

    i = 0
    for song in songs:
        i+=1

        # Step 2
        if not has_acceptable_duration(song, ACCEPTABLE_DURATIONS):
            # print(f"Song Number {i} is not acceptable")
            continue

        # Step 3
        transposed_song = transpose(song)

        trans_songs = []
        trans_songs.append(transposed_song)

        # Step 4 -> Encoding the song
        encoded_song = encode_song(transposed_song)

        # Saving the encoded_song string as a text file
        song_name = f"Song {i}.txt"
        save_path = os.path.join(SAVED_SONGS, song_name)

        with open(save_path, "w") as fp:
            print(f"Saving Song {i}")
            fp.write(encoded_song)


    composite_song = single_file_converter(SAVED_SONGS, SINGLE_SONG_SAVING_PATH, sequence_length)
    create_mappings(composite_song, MAPPING_PATH)

    inputs, targets = generating_training_sequences(sequence_length)

    # song = songs[1]
    # song1 = transpose(song)

    # song.show()
    # song1.show()

            

# if __name__ == "__main__":
    # pre_processing()







# from preprocessing import generating_training_sequences, sequence_length
# import tensorflow as tf

OUTPUT_UNITS = 111 # Number of items in our mappings dictionary
NUM_UNITS = [256] # we are passing it as a list because the LSTM layer can have more than 1 hidden layer
LOSS = "sparse_categorical_crossentropy" # Loss function
LEARNING_RATE = 0.001 # Learning rate of the model

EPOCHS = 50
BATCH_SIZE = 100 

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
                optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
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