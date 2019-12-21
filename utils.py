import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import json
import editdistance as ed

class cha_encdec():
    def __init__(self, dict_file, case_sensitive = True):
        self.dict = []
        self.case_sensitive = case_sensitive
        lines = open(dict_file , 'r').readlines()
        for line in lines:
            self.dict.append(line.replace('\n', ''))
    def encode(self, label_batch):
        max_len = max([len(s) for s in label_batch])
        out = torch.zeros(len(label_batch), max_len+1).long()
        for i in range(0, len(label_batch)):
            if not self.case_sensitive:
                cur_encoded = torch.tensor([self.dict.index(char.lower()) if char.lower() in self.dict else len(self.dict)
                                     for char in label_batch[i]]) + 1
            else:
                cur_encoded = torch.tensor([self.dict.index(char) if char in self.dict else len(self.dict)
                                     for char in label_batch[i]]) + 1
            out[i][0:len(cur_encoded)] = cur_encoded
        return out
    def decode(self, net_out, length):
    # decoding prediction into text with geometric-mean probability
    # the probability is used to select the more realiable prediction when using bi-directional decoders
        out = []
        out_prob = [] 
        net_out = F.softmax(net_out, dim = 1)
        for i in range(0, length.shape[0]):
            current_idx_list = net_out[int(length[:i].sum()) : int(length[:i].sum() + length[i])].topk(1)[1][:,0].tolist()
            current_text = ''.join([self.dict[_-1] if _ > 0 and _ <= len(self.dict) else '' for _ in current_idx_list])
            current_probability = net_out[int(length[:i].sum()) : int(length[:i].sum() + length[i])].topk(1)[0][:,0]
            current_probability = torch.exp(torch.log(current_probability).sum() / current_probability.size()[0])
            out.append(current_text)
            out_prob.append(current_probability)
        return (out, out_prob)

class Attention_AR_counter():
    def __init__(self, display_string, dict_file, case_sensitive):
        self.correct = 0
        self.total_samples = 0.
        self.distance_C = 0
        self.total_C = 0.
        self.distance_W = 0
        self.total_W = 0.
        self.display_string = display_string
        self.case_sensitive = case_sensitive
        self.de = cha_encdec(dict_file, case_sensitive)

    def clear(self):
        self.correct = 0
        self.total_samples = 0.
        self.distance_C = 0
        self.total_C = 0.
        self.distance_W = 0
        self.total_W = 0.
        
    def add_iter(self, output, out_length, label_length, labels):
        start = 0
        start_o = 0
        self.total_samples += label_length.size()[0]
        raw_prdts = output.topk(1)[1]
        prdt_texts, prdt_prob = self.de.decode(output, out_length)
        for i in range(0, len(prdt_texts)):
            if not self.case_sensitive:
                prdt_texts[i] = prdt_texts[i].lower()
                labels[i] = labels[i].lower()
            all_words = []
            for w in labels[i].split('|') + prdt_texts[i].split('|'):
                if w not in all_words:
                    all_words.append(w)
            l_words = [all_words.index(_) for _ in labels[i].split('|')]
            p_words = [all_words.index(_) for _ in prdt_texts[i].split('|')]
            self.distance_C += ed.eval(labels[i], prdt_texts[i])
            self.distance_W += ed.eval(l_words, p_words)
            self.total_C += len(labels[i])
            self.total_W += len(l_words)
            self.correct = self.correct + 1 if labels[i] == prdt_texts[i] else self.correct

    def show(self):
    # Accuracy for scene text. 
    # CER and WER for handwritten text.
        print(self.display_string)
        if self.total_samples == 0:
            pass
        print('Accuracy: {:.6f}, AR: {:.6f}, CER: {:.6f}, WER: {:.6f}'.format(
            self.correct / self.total_samples,
            1 - self.distance_C / self.total_C,
            self.distance_C / self.total_C,
            self.distance_W / self.total_W))
        self.clear()

class Loss_counter():
    def __init__(self, display_interval):
        self.display_interval = display_interval
        self.total_iters = 0.
        self.loss_sum = 0
    
    def add_iter(self, loss):
        self.total_iters += 1
        self.loss_sum += float(loss)

    def clear(self):
        self.total_iters = 0
        self.loss_sum = 0
    
    def get_loss(self):
        loss = self.loss_sum / self.total_iters if self.total_iters > 0 else 0
        self.total_iters = 0
        self.loss_sum = 0
        return loss