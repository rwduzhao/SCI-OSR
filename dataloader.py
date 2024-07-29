import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import cv2
from PIL import Image

# Transformations
RC   = transforms.RandomCrop(32, padding=4)
RHF  = transforms.RandomHorizontalFlip()
RVF  = transforms.RandomVerticalFlip()
NRM  = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
TT   = transforms.ToTensor()
TPIL = transforms.ToPILImage()


class BloodMNIST_Dataset(Dataset):
    def __init__(self):
        self.trainset = BloodMNIST_Dataset(root='./data', train=True)
        self.testset = BloodMNIST_Dataset(root='./data', train=False)
        self.classDict = {'neutrophils': 0,
                          'eosinophils': 1,
                          'basophils': 2,
                          'lymphocytes': 3,
                          'monocytes': 4,
                          'mmature granulocytes': 5,
                          'erythroblasts': 6,
                          'platelets': 7}
        self.transform_train = transforms.Compose([
            transforms.Resize(28),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))])

    def sampler(self, seed, args):
        if seed is not None:
            random.seed(seed)
        num_class = 8
        num_seen = 4
        num_unseen = num_class - num_seen
        seen_classes = random.sample(range(0, num_class), num_seen)
        unseen_classes = [idx for idx in range(num_class) if idx not in seen_classes]

        trainset, valset, testset = construct_ocr_dataset_aug(self.trainset, self.testset, seen_classes, unseen_classes,
                                                              self.transform_train, self.transform_test, args)
        return trainset, valset, testset


class OCTMNIST_Dataset(Dataset):
    def __init__(self):
        self.trainset = OCTMNIST_MNIST(root='./data', train=True)
        self.testset = OCTMNIST_MNIST(root='./data', train=False)
        self.classDict = {'normal': 0,
                          'horoidal neovascularization': 1,
                          'diabetic macular edema': 2,
                          'drusen': 3}
        self.transform_train = transforms.Compose([
            transforms.Resize(28),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        self.transform_test = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    def sampler(self, seed, args):
        if seed is not None:
            random.seed(seed)
        num_class = 4
        num_seen = 3
        num_unseen = num_class - num_seen
        seen_classes = random.sample(range(0, num_class), num_seen)
        unseen_classes = [idx for idx in range(num_class) if idx not in seen_classes]

        trainset, valset, testset = construct_ocr_dataset_aug(self.trainset, self.testset, seen_classes, unseen_classes,
                                                              self.transform_train, self.transform_test, args)
        return trainset, valset, testset


class ISIC2019andDerm7pt_Dataset(Dataset):
    def __init__(self):
        self.trainset = ISIC2019andDerm7pt(root='./data', train=True)
        self.testset = ISIC2019andDerm7pt(root='./data', train=False)
        self.classDict = {'I0': 0, 'I1': 1, 'I2': 2, 'I4': 3, 'I5': 4, 'I6': 5, 'I7': 6, 'I8': 7, 'D1': 8}
        self.transform_train = transforms.Compose([
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))])

    def sampler(self, seed, args):
        if seed is not None:
            random.seed(seed)
        num_class = 9
        num_seen = 8
        num_unseen = num_class - num_seen
        seen_classes = range(0, num_seen)
        unseen_classes = [idx for idx in range(num_class) if idx not in seen_classes]

        trainset, valset, testset = construct_ocr_dataset_aug(self.trainset, self.testset, seen_classes, unseen_classes,
                                                              self.transform_train, self.transform_test, args)
        return trainset, valset, testset


def construct_ocr_dataset_aug(trainset, testset, seen_classes, unseen_classes, transform_train, transform_test, args):
    osr_trainset = DatasetBuilder(
        [get_class_i(trainset.data, trainset.targets, idx) for idx in seen_classes],
        transform_train)

    osr_valset = DatasetBuilder(
        [get_class_i(testset.data, testset.targets, idx) for idx in seen_classes],
        transform_test)

    osr_testset = DatasetBuilder(
        [get_class_i(testset.data, testset.targets, idx) for idx in unseen_classes],
        transform_test)

    return osr_trainset, osr_valset, osr_testset


def construct_ocr_dataset_add(trainset, testset, unknownset, seen_classes, unseen_classes, transform_train, transform_test, args):

    osr_trainset = DatasetBuilder(
        [get_class_i(trainset.data, trainset.targets, idx) for idx in seen_classes],
        transform_train)

    osr_valset = DatasetBuilder(
        [get_class_i(testset.data, testset.targets, idx) for idx in seen_classes],
        transform_test)

    osr_testset = DatasetBuilder(
        [get_class_i(unknownset.data, unknownset.targets, idx) for idx in unseen_classes],
        transform_test)

    return osr_trainset, osr_valset, osr_testset


def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]
    return x_i

# borrow from https://gist.github.com/Miladiouss/6ba0876f0e2b65d0178be7274f61ad2f
class DatasetBuilder(Dataset):
    def __init__(self, datasets, transformFunc):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)

        img = self.datasets[class_label][index_wrt_class]

        if isinstance(img, torch.Tensor):
            img = Image.fromarray(img.numpy(), mode='L')
        elif type(img).__module__ == np.__name__:
            if np.argmin(img.shape) == 0:
                img = img.transpose(1, 2, 0)
            img = Image.fromarray(img)
        elif isinstance(img, tuple): #ImageNet
            # img = Image.open(img[0])
            img = cv2.imread(img[0])
            img = Image.fromarray(img)

        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class