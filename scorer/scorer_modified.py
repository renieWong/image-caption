import os
import sys
import numpy as np
import pickle
import pdb
from lib.config import cfg

from scorer.cider import Cider
from scorer.bleu import Bleu

bleu = Bleu(4)

factory = {
    'CIDEr': Cider,
    'Bleu': Bleu, 
}

def get_sents(sent):
    words = []
    for word in sent:
        words.append(word)
        if word == 0:
            break
    return words

def array_to_str(sent):
    out = ''
    for word in sent:
        out += str(word) + ' '
        if word == 0:
            break
    return out.strip()

class Scorer(object):
    def __init__(self):
        super(Scorer, self).__init__()
        self.scorers = []
        self.weights = cfg.SCORER.WEIGHTS
        self.gts = pickle.load(open(cfg.SCORER.GT_PATH, 'rb'), encoding='bytes')
        # pdb.set_trace()
        for name in cfg.SCORER.TYPES:
            self.scorers.append(factory[name]())

    def __call__(self, ids, res):
        hypo = [get_sents(r) for r in res]
        hypo_ = {ids[i]: [array_to_str(res[i])] for i in range(len(ids))}
        gts = [self.gts[i] for i in ids]
        gts_ = {i: [array_to_str(self.gts[i][0])] for i in ids}
        pdb.set_trace()

        rewards_info = {}
        rewards = np.zeros(len(ids))
        for i, scorer in enumerate(self.scorers):
            if i==0:
                score, scores = scorer.compute_score(gts, hypo)
            else:
                score, scores = bleu.compute_score(gts_, hypo_)
                scores = np.array(scores[3])
                score = score[3]
            rewards += self.weights[i] * scores
            rewards_info[cfg.SCORER.TYPES[i]] = score
        return rewards, rewards_info
