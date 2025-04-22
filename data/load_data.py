import torch
from torchvision import datasets, transforms
import os

from data.cinic10 import fetch_cinic10, get_cinic10
from data.tinyimagenet import fetch_tinyimagenet, TinyImageNet

SHAPES = {
    "cifar10": (32, 32, 3),
    "fmnist": (28, 28, 1),
    "mnist": (28, 28, 1),
    "cifar100": (32, 32, 3),
    "svhn": (32, 32, 3),
    "celebA": (128, 128, 3),
    "cinic10": (32, 32, 3)
}

def get_data(name, augment=True, **kwargs):
    load_path=os.getcwd()+'/data'
    if name == "cifar10":
        if augment==True:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            train_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]

            train_set = datasets.CIFAR10(root=load_path, train=True,
                                         transform=transforms.Compose(train_transforms),
                                         download=True)

            test_set = datasets.CIFAR10(root=load_path, train=False,
                                        transform=transforms.Compose(
                                            train_transforms
                                        ), download=True)

        else:
            train_set = datasets.CIFAR10(root=load_path, train=True,transform=transforms.Compose([transforms.ToTensor()]),download=True)

            test_set = datasets.CIFAR10(root=load_path, train=False,transform=transforms.Compose([transforms.ToTensor()]), download=True)

    elif name == "fmnist":

        transform=transforms.ToTensor()

        train_set = datasets.FashionMNIST(root=load_path, train=True,
                                          transform=transform,
                                          download=True)

        test_set = datasets.FashionMNIST(root=load_path, train=False,
                                         transform=transform,
                                         download=True)

    elif name == "mnist":

        transform=transforms.ToTensor()
        train_set = datasets.MNIST(root=load_path, train=True,
                                   transform=transform,
                                   download=True)

        test_set = datasets.MNIST(root=load_path, train=False,
                                  transform=transform,
                                  download=True)

    elif name == "cifar100":

        if augment==True:
            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                             std=[0.2675, 0.2565, 0.2761])

            train_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]


            train_set = datasets.CIFAR100(root=load_path, train=True,
                                          transform=transforms.Compose(train_transforms),
                                          download=True)

            test_set = datasets.CIFAR100(root=load_path, train=False,
                                         transform=transforms.Compose(
                                             [transforms.ToTensor(), normalize]
                                             ), download=True )
        else:

            train_set = datasets.CIFAR100(root=load_path, train=True,transform=transforms.Compose([transforms.ToTensor()]),
                                          download=True)

            test_set = datasets.CIFAR100(root=load_path, train=False,transform=transforms.Compose([transforms.ToTensor()]),
                                          download=True)
    elif name == "cinic10":
        if augment==True:
            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                             std=[0.2675, 0.2565, 0.2761])
            train_transforms =  transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ])


           # raw_train_set, raw_test_set = fetch_cinic10(load_path)  # need to download first

            train_set = datasets.ImageFolder(root=f'{load_path}/cinic-10-batches-py/train', transform=train_transforms)
            test_set = datasets.ImageFolder(root=f'{load_path}/cinic-10-batches-py/test', transform=train_transforms)
        else:
            train_set = datasets.ImageFolder(root=f'{load_path}/cinic-10-batches-py/train',transform=transforms.Compose([transforms.ToTensor()]))
            test_set = datasets.ImageFolder(root=f'{load_path}/cinic-10-batches-py/test',transform=transforms.Compose([transforms.ToTensor()]))

    elif name == "tinyimagenet":
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_transforms =  transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])


      #  raw_train_set,raw_test_set = fetch_tinyimagenet(load_path)  # need to download first
        train_set = datasets.ImageFolder(root=f'{load_path}/tiny-imagenet-200/train', transform=train_transforms)
        test_set = datasets.ImageFolder(root=f'{load_path}/tiny-imagenet-200/val', transform=train_transforms)



    elif name == "svhn":
        normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                         std=[0.198, 0.201, 0.197])

        if augment:
            train_transforms = [
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,
            ]

        train_set = datasets.SVHN(root=load_path, split='train',
                                  transform=transforms.Compose(train_transforms),
                                  download=True)

        test_set = datasets.SVHN(root=load_path, split='test',
                                 transform=transforms.Compose(
                                     [transforms.ToTensor(), normalize]
                                 ), download=True)


    elif name == "celebA":  #need torchvision >0.17.0
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        augment = True

        if augment:
            train_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(178),
                transforms.Resize(128),
                transforms.ToTensor(),
                normalize,
            ]
        else:
            train_transforms = [
                transforms.CenterCrop(178),
                transforms.Resize(128),
                transforms.ToTensor(),
                normalize,
            ]

        train_set = datasets.CelebA(root=load_path, split='train', transform=transforms.Compose(train_transforms),
                                    download=True)
        test_set = datasets.CelebA(root=load_path, split='test', transform=transforms.Compose(
            [transforms.CenterCrop(178), transforms.Resize(128), transforms.ToTensor(), normalize]), download=True)

    elif name == "stl10":
        normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.198, 0.201, 0.197])


        augment = True
        if augment:
            train_transforms = transforms.Compose([
                transforms.Resize((32, 32)),  # Resize to 32x32
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,
            ]

        train_set = datasets.SVHN(root=load_path, split='train',
                                  transform=transforms.Compose(train_transforms),
                                  download=True)

        test_set = datasets.SVHN(root=load_path, split='test',
                                 transform=transforms.Compose(
                                     [transforms.ToTensor(), normalize]
                                 ), download=True)

    else:
        raise ValueError(f"unknown dataset {name}")

    return train_set, test_set


