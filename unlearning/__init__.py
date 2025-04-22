from unlearning.GA import GA
from unlearning.IF import IF
from unlearning.certified import certified
from unlearning.fisher import fisher
from unlearning.negGrad import NegGrad
from unlearning.retrain import retrain
from unlearning.retrain_dp import retrain_dp
from unlearning.scrub import scrub
from unlearning.sisa import sisa
from unlearning.sparsity import sparsity


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "retrain":
        return retrain
    elif name == "GA":
        return GA
    elif name == "sparsity":
        return sparsity
    elif name =='IF':
        return IF
    elif name =='fisher':
        return fisher
    elif name =='scrub':
        return scrub
    elif name =='sisa':
        return sisa
    elif name == "retrain_dp":
        return retrain_dp
    elif name == "certified":
        return certified
    elif name== 'NegGrad':
        return NegGrad
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
