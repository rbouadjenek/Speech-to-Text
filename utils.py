import pandas as pd
import csv
import tensorflow as tf
import config
def character_mapping():
    character_map = {" ": 1}
    for i in range(97, 123):
        character_map[chr(i)] = len(character_map) + 1

    return character_map

def get_max_len(train_direction="./train_data/train.csv"):
    train_df = pd.read_csv(train_direction)
    max_spec_length = train_df["spec_length"].max()
    max_label_length = train_df["labels_length"].max()
    return max_spec_length, max_label_length

def create_data_generator(directory, batch_size= config.training["batch_size"], ds = "train"):
    if ds == "train":
        file = "train.csv"
    elif ds == "valid" or ds == "dev":
        file = "dev.csv"
    else:
        file ="test.csv"
    max_spec_length, max_label_length = utils.get_max_len()
    x, y, input_lengths, label_lengths = [], [], [], []
    with open(os.path.join(directory, file), 'r') as metadata:
        metadata_reader = csv.DictReader(metadata, fieldnames=['filename', 'spec_length', 'labels_length', 'labels'])
        next(metadata_reader)
        for row in metadata_reader:
            audio = np.load(os.path.join(directory, row['filename'] + '.npy'))
            x.append(audio)
            y.append([int(i) for i in row['labels'].split(' ')])
            input_lengths.append(int(row['spec_length']))
            label_lengths.append(int(row['labels_length']))
            if len(x) == batch_size:
                yield {
                    'inputs': tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_input_length, padding='post'),
                    'labels': tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=max_label_length, padding='post'),
                    'input_lengths': np.asarray(input_lengths),
                    'label_lengths': np.asarray(label_lengths)
                }, {
                    'ctc': np.zeros([batch_size])
                }
                x, y, input_lengths, label_lengths = [], [], [], []