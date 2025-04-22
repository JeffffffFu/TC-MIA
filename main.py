from attack import get_attack_method

from parameter_parser import parameter_parser

from unlearning import get_unlearn_method
from unlearning.retrain import retrain_save_target_for_population_attack_batch, retrain_save_shadow_for_population_attack_batch

def main(args):
    if args['U_method']!='None' and args['attack_method']=='None' and args['pre_train'] =='both':
        unlearn_method=get_unlearn_method(args['U_method'])
        unlearn_method(args)

    if args['attack_method']!='None' :
        attack_method=get_attack_method(args['attack_method'])
        attack_method(args)

    else:
        raise ValueError("this algorithm is not exist")
if __name__ == '__main__':
    args = parameter_parser()
    main(args)
