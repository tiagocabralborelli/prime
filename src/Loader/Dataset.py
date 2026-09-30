
from Bio import SeqIO
import numpy as np
import pandas as pd



ALLOWED_AA = set('ACDEFGHIKLMNPQRSTVWY')
def filter_sequence(seq):
    seq_upper = seq.upper()
    for ch in seq_upper:
        if ch not in ALLOWED_AA:
            return False
    return True

class LoadFasta:
    def __init__(self, train,test,name):
        if not isinstance(train, str) or not isinstance(test, str):
            raise TypeError("Input must be a string path to a FASTA file.")
        self.train = train
        self.test = test
        self.name = name

    def load(self, sample = False):
        extension_train = self.train.split('.')[-1].lower() 
        if extension_train not in ['fasta', 'fa', 'fna']:
            raise ValueError(f"Unsupported file extension: {extension_train}. Please provide a FASTA file with .fasta, .fa, or .fna extension.")

        extension_test = self.test.split('.')[-1].lower() 
        if extension_train not in ['fasta', 'fa', 'fna']:
            raise ValueError(f"Unsupported file extension: {extension_test}. Please provide a FASTA file with .fasta, .fa, or .fna extension.")

        else:
            train_id, train_seq = [],[]
            for record in SeqIO.parse(self.train, "fasta"):
                if filter_sequence(str(record.seq)):
                    train_seq.append(str(record.seq).upper())
                    train_id.append(float(str(record.id).upper().split('|')[-1]))
                else:
                    print(f"  Warning: Skipping sequence with invalid AA in the training: {record.id}")

            test_id, test_seq = [],[]
            for record in SeqIO.parse(self.test, "fasta"):
                if filter_sequence(str(record.seq)):
                    test_seq.append(str(record.seq).upper())
                    test_id.append(float(str(record.id).upper().split('|')[-1]))
                    print(f"  Warning: Skipping sequence with invalid AA in the test set: {record.id}")

            self.train_id  = np.array(train_id)
            self.train_seq = train_seq
            self.test_id  =  np.array(test_id)
            self.test_seq  = test_seq


    def clean(self):
        """
        Check for duplicated sequences or cantamination
        """
        seen = set()
        self._clean_train_seq, self._clean_train_labels = [], []
        for s, l in zip(self.train_seq, self.train_id):
            if s not in seen:
                seen.add(s)
                self._clean_train_seq.append(s)
                self._clean_train_labels.append(l)
        train_set = set(self._clean_train_seq)
        self._clean_test_seqs, self.clean_test_labels = [], []
        for s, l in zip(self.test_seq, self.test_id):
            if s not in train_set:
                self._clean_test_seqs.append(s)
                self.clean_test_labels.append(l)

        self.train_seq = self._clean_train_seq
        self.train_id = np.asarray(self._clean_train_labels)
        self.test_seq = self._clean_test_seqs
        self.test_id = np.asarray(self.clean_test_labels)
        

class DataSet(LoadFasta):
    def load(self,sample):
        super().load(sample)
    def clean(self):
        super().clean()