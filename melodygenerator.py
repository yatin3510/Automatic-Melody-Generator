import tensorflow.keras as keras
import json
import music21 as m21
import numpy as np
from preprocessing import sequence_length, MAPPING_PATH
from train import SAVE_MODEL_PATH

class MelodyGenerator:


    def __init__(self, model_path = SAVE_MODEL_PATH):
        
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        
        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * sequence_length


    def generate_melody(self, seed, num_steps, max_sequence_length = sequence_length, temperature = 0.7):

        seed = seed.split() # converting the seed to a list of notes and rests

        melody = seed

        # I don't know yet why we have added 64 '/' symbols in the start of the seed
        seed = self._start_symbols + seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # We are doing this because our model can take 64 symbols at a time to give the output
            seed = seed[-max_sequence_length : ]

            # one-hot encoding the seed because our model has been trained in the same way
            one_hot_seed = keras.utils.to_categorical(seed, num_classes = len(self._mappings))

            # adding another dimension to the seed because keras accepts an extra dimension
            # current dimension of seed -> max_sequence * len(mappings)
            # desired dimension of seed -> 1 * max_sequence * len(mappings)
            one_hot_seed = one_hot_seed[np.newaxis, ...]

            # Making a prediction

            # We are taking only the 0th index because our model will output a batch of outputs had we passed
            # in a batch of seeds but since we have passed only 1 seed, hence we will extract the first and
            # only item in the output list

            probabilities = self.model.predict(one_hot_seed)[0]

            """
            We will get a list of probabilities from this statement like :
            [0.2, 0.3, .... 0.1, 0.1, 0.2] ( total 18 elements[len(mappings)] )
            which will add up to give 1.

            One way to proceed from here is to get the most probable outcome an set it as output for the 
            input seed, but in that way we will be getting very rigid outputs.
            
            For example : if the most probable outcome for "I don't ..." is "know" then for everytime,
            the words "I don't" are encountered, the answer will always be "know" and then there will be
            no possibility of nuances in the output of our model.

            For reading more about it, go to https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277
            """

            output_int = self._sample_with_temperature(probabilities, temperature)

            # update the seed
            seed.append(output_int)

            # map the integers back to the midi values
            key_list = list(self._mappings.keys())
            val_list = list(self._mappings.values())

            position = val_list.index(output_int)

            output_symbol = key_list[position]

            if(output_symbol == '/'):
                break

            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilities, temperature):

        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p = probabilities)

        return index

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.midi"):
        """Converts a melody into a MIDI file
        :param melody (list of str):
        :param min_duration (float): Duration of each time step in quarter length
        :param file_name (str): Name of midi file
        :return:
        """

        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)






from music21 import converter
from tensorflow import keras
from preprocessing import encode_song, transpose, convert_songs_to_int, MAPPING_PATH


if __name__ == "__main__":

    song = converter.parse('AnyConv.com__Bollywood Song In Piano.midi')

    # Step 1
    transposed_song = transpose(song)

    trans_songs = []
    trans_songs.append(transposed_song)

    # Step 2 -> Encoding the song
    # encoded_song = encode_song(transposed_song)

    encoded_seed = encode_song(song = transposed_song)
    # print(encoded_seed, end = "\n\n\n")
    encoded_seed_int = convert_songs_to_int(encoded_seed)
    # print(encoded_seed_int)

    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    keys_list = list(mappings.keys())
    values_list = list(mappings.values())

    buff = []
    for val in encoded_seed_int:
        position = values_list.index(val)
        buff.append(keys_list[position])
    

    encoded_seed_int_str = " ".join([str(ele) for ele in buff])

    mg = MelodyGenerator()
    seed = encoded_seed_int_str

    melody = mg.generate_melody(seed, num_steps = 500, max_sequence_length = sequence_length, temperature = 0.7)
    print(melody)


    mg.save_melody(melody)