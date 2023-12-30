#!/usr/bin/env python
# 
# File Name : bleu.py
#
# Description : Wrapper for BLEU scorer.
#
# Creation Date : 06-01-2015
# Last Modified : Thu 19 Mar 2015 09:13:28 PM PDT
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .bleu_scorer import BleuScorer


class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        # assert(list(gts.keys()) == list(res.keys()))
        # imgIds = list(gts.keys())
        # imgIds = gts.keys()
        
        bleu_scorer = BleuScorer(n=self._n)
        '''
        for id in imgIds:
            hypo = res[id]
            ref = [gts[id]]
        '''

        for i, hypo in enumerate(res):
            ref = gts[i]

            # print('hypo:', hypo)
            # print('ref:', ref)
        #for res_id in res:
        #    print('res_id:', res_id)
        #    hypo = res_id['caption']
        #    ref = [gts[res_id['image_id']]]
            # Sanity check.
            # assert(type(hypo) is list)
            # assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            bleu_scorer += (hypo, ref)

        #score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=1)
        #score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        return score, scores

    def method(self):
        return "Bleu"