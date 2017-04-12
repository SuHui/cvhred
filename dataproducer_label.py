import tensorflow as tf
import random
import pickle
import numpy as np

"""
    labels: vocab index labels, max_length*batch_size, padding labels are 0s
    length: length of every sequence, batch_size
"""


class labeled_data_producer:
    def __init__(self, frs, num_epochs):
        self.__dict__.update(locals())
        self.file_queue = tf.train.string_input_producer(frs)  # , num_epochs=self.num_epochs)
        self.reader = tf.TFRecordReader()

    # label: vocabulary index list, len
    # convert one-line dialogue into a tf.SequenceExample
    def __make_example(self, label, a_label,e_label, length):
        ex = tf.train.SequenceExample()
        # one sequence
        ex.context.feature["len"].int64_list.value.append(length)
        for w in label:
            ex.feature_lists.feature_list['seq'].feature.add().int64_list.value.append(w + 1)  # prevent 0 for padding
        for a in a_label:
            ex.feature_lists.feature_list['alabel'].feature.add().int64_list.value.append(int(a))
        for e in e_label:
            ex.feature_lists.feature_list['elabel'].feature.add().int64_list.value.append(int(e))
        
        return ex

    """
    slice a dialogue with size limit and make example
    slide over the whole dialogue, the next sequence share the end of the last sequence
    all results are padded with 0s if the length is less than limit
    """

    def slice_dialogue(self, dialogue, a,e, limit):
        exs = []
        start = 0
        while ((len(dialogue) - 1) > start):
            length = limit
            if start + limit > len(dialogue):  # padding 0
                length = len(dialogue) - start
                dialogue.extend([-1] * (start + limit - len(dialogue)))
                a.extend([0] * (start + limit - len(a)))
                e.extend([0] * (start + limit - len(e)))
            ex = self.__make_example(dialogue[start:start + limit], a[start:start + limit],e[start:start + limit], length)
            start += (limit - 1)
            exs.append(ex)
        return exs

    # labels: list of label(length), save as tfrecord form
    def save_record(self, labels, a_labels,e_labels, fout, limit=40):
        writer = tf.python_io.TFRecordWriter(fout)
        num_record = 0
        for dialogue, a, e in zip(labels, a_labels, e_labels):
            if num_record % 100 == 0:
                print(num_record)
            for ex in self.slice_dialogue(dialogue, a, e, limit):
                num_record += 1
                writer.write(ex.SerializeToString())
        writer.close()
        print('num_record:', num_record)

    # read from a list of TF_Record files frs, return a parsed Sequence_example
    # Every Sequence_example contains one dialogue
    # emotion=0 indicates no emotion information stored
    def __read_record(self, emotion=0):
        # first construct a queue containing a list of filenames.
        # All data can be split up in multiple files to keep size down
        # serialized_example is a Tensor of type string.
        _, serialized_example = self.reader.read(self.file_queue)
        # create mapping to parse a sequence_example
        context_features = {'len': tf.FixedLenFeature([], dtype=tf.int64)}
        sequence_features = {'seq': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                             'alabel':tf.FixedLenSequenceFeature([], dtype=tf.int64),
                             'elabel':tf.FixedLenSequenceFeature([], dtype=tf.int64)}
        # sequences is a sequence_example for one dialogue
        length, sequences = tf.parse_single_sequence_example(
            serialized_example,
            context_features=context_features,
            sequence_features=sequence_features)
        return length, sequences

    """
    get the next batch from a list of files frs
    emotion=0 indicates no emotion information stored
    return length, labels
    """
    def batch_data(self, batch_size, emotion=0):
        length, sequences = self.__read_record(emotion)  # one-line dialogue
        print sequences
        # shuffled batch
        batched_seq, batched_alabel, batched_elabel, batched_len = tf.train.batch(
            tensors=[sequences['seq'], sequences['alabel'], sequences['elabel'], length['len']],
            batch_size=batch_size, capacity=500, shapes=[[40],[40],[40],[]]
        )

        return tf.transpose(batched_seq, perm=[1, 0]), tf.transpose(batched_alabel, perm=[1, 0]), tf.transpose(batched_elabel, perm=[1, 0]), batched_len

if __name__ == '__main__':
    producer = data_producer(['./tfrecord/input.tfrecord'], 1)
    with open('Training_part.dialogues.pkl', 'rb') as f,open('actlabel.pkl', 'rb') as f1, open('emotionlabel.pkl', 'rb') as f2:
            
            labels = pickle.load(f)
            a_labels = pickle.load(f1)
            e_labels = pickle.load(f2)
            
            for i in range(30):
                print(len(labels[i]))
                print(len(a_labels[i]))
                print(len(e_labels[i]))
            random.seed(1)
            random.shuffle(labels)
            random.seed(1)
            random.shuffle(a_labels)
            random.seed(1)
            random.shuffle(e_labels)
            producer.save_record(labels,a_labels,e_labels, 'tfrecord_part_mixture_labels')
