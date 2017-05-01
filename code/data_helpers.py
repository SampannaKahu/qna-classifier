import numpy as np
import re
from tqdm import tqdm


def clean_string(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(questions_file_path, answers_file_path):
    """
    Loads data from files, filters them, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    print 'Loading questions data...'
    questions = read_lines(questions_file_path)
    print 'Loading answers data...'
    answers = read_lines(answers_file_path)

    x_text = questions + answers

    print 'Cleaning data...'
    x_text = [clean_string(sent) for sent in tqdm(x_text)]

    print 'Generating labels...'
    question_labels = [[1, 0] for _ in questions]
    answer_labels = [[0, 1] for _ in answers]
    y = np.concatenate(
        [question_labels, answer_labels], 0)
    print 'Data and labels have been successfully loaded.'
    return [x_text, y]


def read_lines(file_path):
    """
    Reads a file and returns a list of lines
    :param file_path: The path of the file that needs to be read.
    :return: The list of the lines that was read.
    """
    with open(file_path) as my_file:
        lines = my_file.readlines()
        lines = [line.strip() for line in lines]
        return lines


def generate_batch_iterator(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for the passed dataset.    
    :param data: The data for which the iterator needs to be generated. 
    :param batch_size: The desired size of the batch.
    :param num_epochs: 
    :param shuffle: Whether to shuffle the given data in each epoch.
    :return: 
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
