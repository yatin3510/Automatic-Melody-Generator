from music21 import converter
from tensorflow import keras
from preprocessing import encode_song, transpose, convert_songs_to_int, MAPPING_PATH
song = converter.parse('AnyConv.com__Bollywood Song In Piano.midi')

# Step 1
transposed_song = transpose(song)

trans_songs = []
trans_songs.append(transposed_song)

# Step 2 -> Encoding the song
# encoded_song = encode_song(transposed_song)

encoded_seed = encode_song(song = transposed_song)
print(encoded_seed, end = "\n\n\n")
encoded_seed_int = convert_songs_to_int(encoded_seed)
print(encoded_seed_int)


# one hot encodin of the song
one_hot_encoded_seed_int = keras.utils.to_categorical(encoded_seed_int, num_classes=111)
print(one_hot_encoded_seed_int.shape)