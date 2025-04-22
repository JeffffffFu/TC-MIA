import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        choices=[ 'mnist', 'fmnist','cifar10', 'stl10', 'cifar100','svhn','celebA','tinyimagenet','cinic10'])

    parser.add_argument('--device', type=str, default='cuda:0',
                        help="Choose the  device")

    parser.add_argument('--random', type=int, default=0)
    ######################### target model related parameters ################################
    parser.add_argument('--net_name', type=str, default='resnet18',
                        choices=[ 'simple_cnn', 'resnet18', 'resnet20','vgg','resnet18_dp', 'densenet','simple_cnn_dropout'])
    # parser.add_argument('--attack_model', type=str, default='DT',
    #                     choices=['DT', 'MLP', 'LR', 'RF'])
    parser.add_argument('--U_method', type=str, default='None',
                        choices=['retrain', 'sisa','GA','sparsity','IF','fisher','scrub','sisa','retrain_dp','certified','NegGrad','None'])
    parser.add_argument('--retrain', type=int, default=0)
    parser.add_argument('--pre_train', type=str, default='both',choices=['both','target', 'shadow'])
    parser.add_argument('--attack_method', type=str, default='None',
                        choices=['U_LIRA', 'TC_MIA','U_Leak','Double_Attack','None'])
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--optim', type=str, default="Adam",
                        choices=['Adam', 'SGD'])
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: .1)", )
    ######################### attack related parameters ################################
    parser.add_argument('--trials', type=int, default=3,
                        help="number of trials")
    parser.add_argument('--observations', type=int, default=5,
                        help="number of observations")
    parser.add_argument('--base_num_class', type=int, default=3,
                        help="number of class for baseline: 2 or 3")
    parser.add_argument('--proportion_of_group_unlearn', type=float, default=0.02,
                        help=">=1 mean the exact number of unlearn")

    # For DPSGD
    parser.add_argument('--sigma', type=float, default=1.0,
                        help="noise of DPSGD")
    parser.add_argument('--eps', type=float, default=1.0,
                        help="privacy budget of DPSGD")
    parser.add_argument('--C', type=float, default=1.0,
                        help="C of model parameters (for DPSGD)")



    args = vars(parser.parse_args())

    return args