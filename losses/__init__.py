from losses.cross_entropy import CrossEntropy
from losses.label_smoothing import LabelSmoothing
from losses.reward_criterion import RewardCriterion
from losses.infonce import InfoNCE

__factory = {
    'CrossEntropy': CrossEntropy,
    'LabelSmoothing': LabelSmoothing,
    'RewardCriterion': RewardCriterion,
    'InfoNCE':InfoNCE,
}

def names():
    return sorted(__factory.keys())

def create(name):
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name]()