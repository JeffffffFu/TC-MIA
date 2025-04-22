from attack.Double_Attack import Double_Attack
from attack.TC_MIA import TC_MIA
from attack.U_Leak import U_Leak


def get_attack_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "U_Leak":
        return U_Leak
    elif name == "TC_MIA":
        return TC_MIA
    elif name == "Double_Attack":
        return Double_Attack
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
