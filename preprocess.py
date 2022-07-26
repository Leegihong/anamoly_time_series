import pandas as pd
import numpy as np
import torch

class TimeseriesPreprocess:
    def __init__(self, data):
        self.data = pd.read_csv(data)


    def make_window_sequence(self, size):
        seq = []
        for i in range(0, len(self.data) - size):
            seq.append(self.data[i :i+size])
        return seq

    def make_same_length_seq(self, id, value, length = 320):
        seq = []
        id_list = self.data[id].unique()
        for i in id_list:
            tmp = self.data[self.data[id] == i][value].tolist()
            seq.append(tmp)
        sequences = []
        for i, sequence in enumerate(seq):
            mask = True
            while mask:
                if len(sequence)< 51:
                    print(f"{i}th sequnce Length Under")
                    mask = False
                elif len(sequence) < length:
                    sequence.append(0)
                elif len(sequence) == length:
                    # print(f"{i} sequence over")
                    sequences.append(sequence)
                    mask = False
                else:
                    print(f"{i}th sequnce Length Over")
                    mask = False
        print("ALL SEQUNCE MAKING OVER")
        print(np.shape(seq))
        return sequences
    
    def create_dataset(seq):
        sequences = seq.astype(np.float32).to_numpy().tolist()
        dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
        n_seq, seq_len, n_features = torch.stack(dataset).shape
        return dataset, seq_len, n_features
    
    
