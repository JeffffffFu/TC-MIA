import os
import torch
import shutil
import logging
import torchvision

logger = logging.getLogger(__name__)


# dataset wrapper module
class TinyImageNet(torchvision.datasets.ImageFolder):
    base_folder = 'tiny-imagenet-200'
    zip_md5 = '90528d7ca1a48142e341f4ef8d21d0de'
    splits = ('train', 'val', 'test')
    filename = 'tiny-imagenet-200.zip'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    def __init__(self, root, split, download=True, transform=None, **kwargs):
        self.data_root = os.path.expanduser(root)
        self.split = torchvision.datasets.utils.verify_str_arg(split, 'split', self.splits)
        if download:
            self.download()
        if not self._check_exists():
            err = 'Dataset not found or corrupted. You can use download=True to download it'
            logger.exception(err)
            raise RuntimeError(err)

        #self.normalize_tin_val_folder_structure(os.path.join(self.dataset_folder, 'val'))

        super().__init__(root=self.split_folder, transform=transform, **kwargs)

    def normalize_tin_val_folder_structure(self, path, images_folder='images', annotations_file='val_annotations.txt'):
        images_folder = os.path.join(path, images_folder)
        annotations_file = os.path.join(path, annotations_file)

        if not os.path.exists(images_folder) or not os.path.exists(annotations_file):
            err = 'Validation folder is missing required files!'
            logger.exception(err)
            raise RuntimeError(err)

        if not os.listdir(path):
            err = 'Validation folder is empty!'
            logger.exception(err)
            raise RuntimeError(err)

        with open(annotations_file) as f:
            for line in f:
                values = line.split()
                if len(values) < 2:
                    continue

                img, label = values[:2]
                img_file = os.path.join(images_folder, img)
                label_folder = os.path.join(path, label)

                os.makedirs(label_folder, exist_ok=True)

                try:
                    shutil.move(img_file, os.path.join(label_folder, img))
                except FileNotFoundError:
                    logger.warning(f"File not found: {img_file}")
                    continue
                except Exception as e:
                    logger.error(f"Error moving file {img_file}: {str(e)}")
                    continue

        try:
            if os.path.exists(images_folder):
                shutil.rmtree(images_folder)
            if os.path.exists(annotations_file):
                os.remove(annotations_file)
        except Exception as e:
            logger.error(f"Error cleaning up files: {str(e)}")

    @property
    def dataset_folder(self):
        return os.path.join(self.data_root, self.base_folder)

    @property
    def split_folder(self):
        return os.path.join(self.dataset_folder, self.split)

    def _check_exists(self):
        return os.path.exists(self.split_folder)

    def download(self):
        if self._check_exists():
            return
        torchvision.datasets.utils.download_and_extract_archive(
            self.url, self.data_root, filename=self.filename,
            remove_finished=True, md5=self.zip_md5
        )
       # assert 'val' in self.splits
        self.normalize_tin_val_folder_structure(os.path.join(self.dataset_folder, 'val'))

    def __repr__(self):
        rep_str = {'train': 'CLIENT', 'test': 'SERVER'}
        return f'[TinyImageNet] {rep_str[self.split]}'


# helper method to fetch Tiny ImageNet dataset
def fetch_tinyimagenet( root):
    logger.info('[LOAD] [TINYIMAGENET] Fetching dataset!')

    # create training dataset instance
    raw_train = TinyImageNet(root,split='train')


    # create test dataset instance
    raw_test = TinyImageNet(root,split='val')

    logger.info('[LOAD] [CINIC10] ...fetched dataset!')
    return raw_train, raw_test

