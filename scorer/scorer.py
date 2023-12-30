import os
import sys
import numpy as np 
import pickle 
from lib.config import cfg
from collections import OrderedDict 

from scorer.cider import Cider
from scorer.bleu import Bleu
from scorer.meteor import Meteor
from scorer.rouge import Rouge
import ipdb

factory = {
    'CIDEr': Cider,
    'Bleu': Bleu,
    'METEOR': Meteor,
    'ROUGE_L': Rouge,
}

def get_sents(sent):
    words = []
    for word in sent:
        words.append(word)
        if word == 0: 
            break
    return words

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


class Scorer(object):
    def __init__(self):
        super(Scorer, self).__init__()
        self.scorers = []
        self.weights = cfg.SCORER.WEIGHTS
        self.gts = pickle.load(open(cfg.SCORER.GT_PATH, 'rb'), encoding='bytes')
        for name in cfg.SCORER.TYPES:
            self.scorers.append(factory[name]())

    def __call__(self, ids, res):
        # hypo = OrderedDict()
        # gts_ = OrderedDict()
        pred = [get_sents(r) for r in res]
        hypo = [array_to_str(res[i]) for i in range(len(res))]
        

        gts = [self.gts[i] for i in ids]
        gts_ = [array_to_str(gts[i][0]) for i in range(len(gts))]

        res_ = [{'image_id': i, 'caption': hypo[i]} for i in range(len(hypo))] 
        res__ = {i: [hypo[i]] for i in range(len(hypo))}
        gts__ = {i: [gts_[i]] for i in range(len(gts_))}
        rewards_info = {}
        rewards = np.zeros(len(ids))
        for i, scorer in enumerate(self.scorers):
            if i==1:
                # score, scores = Bleu(4).compute_score(gts_, res__)
                score, scores = Bleu(4).compute_score(gts, pred)
                scores_1 = np.array(scores[0])
                score_1 = score[0]
                scores_2 = np.array(scores[1])
                score_2 = score[1]
                scores_3 = np.array(scores[2])
                score_3 = score[2]
                scores_4 = np.array(scores[3])
                score_4 = score[3]
                scores = 2 * scores_1 + 2 * scores_2 + 1 * scores_3 + 1 * scores_4
                score = 2 * score_1 + 2 * score_2 + 1*score_3 + 1*score_4
                score = score_4               # TODO 
                scores = scores_4 
                # print('scores: ', scores)
            elif i==2:
                # print('gts__', gts__)
                # print('res__', res__)
                score, scores = scorer.compute_score(gts__, res__)
                scores = np.array(scores)
                # ipdb.set_trace()

            else:
                # score, scores = scorer.compute_score(gts_, res_)
                score, scores = scorer.compute_score(gts, pred)

            rewards += self.weights[i] * scores
            rewards_info[cfg.SCORER.TYPES[i]] = score
        return rewards, rewards_info
