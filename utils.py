'''
Author: LOTEAT
Date: 2023-05-31 16:10:57
'''  
class Seq2Text:
    def __init__(self, vocab_dict, end_idx):
        self.reverse_word_map = {value: key for key, value in vocab_dict.items()}
        self.end_idx = end_idx

    def sequence_to_text(self, list_of_indices):
        words = [self.reverse_word_map.get(idx) for idx in list_of_indices if idx != self.end_idx]
        return ' '.join(words)

