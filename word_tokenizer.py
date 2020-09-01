# coding: utf-8 

class WordTokenizer(object):

    def __int__(self, vocab=None, unk_token='[UNK]'):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_token = unk_token

    def convert_tokens_to_ids(self, tokens, max_length, pad_id = 0, unk_id = 1, uncased=True):
        output = []
        for token in tokens:
            if uncased:
                token = token.lower()
            if token in self.vocab:
                output.append(self.vocab[token])
            else:
                output.append(unk_id)
        if len(output) > max_length:
            output = output[:max_length]
        else:
            while len(output) < max_length:
                output.append(pad_id)
        return output
