import os

import torch
import torchvision
import torchvision.transforms.v2
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image

__all__: list[str] = [
    "VOCDataset",
    "get_voc_labels",
    "load_images_and_anns"
]

def load_images_and_anns(im_sets, label2idx, ann_fname, split):
    r"""
    Method to get the xml files and for each file
    get all the objects and their ground truth detection
    information for the dataset
    :param im_sets: Sets of images to consider
    :param label2idx: Class Name to index mapping for dataset
    :param ann_fname: txt file containing image names{trainval.txt/test.txt}
    :param split: train/test
    :return:
    """
    im_infos = []
    for im_set in im_sets:
        im_names = []
        # Fetch all image names in txt file for this imageset
        for line in open(os.path.join(
                im_set, 'ImageSets', 'Main', '{}.txt'.format(ann_fname))):
            im_names.append(line.strip())

        # Set annotation and image path
        ann_dir = os.path.join(im_set, 'Annotations')
        im_dir = os.path.join(im_set, 'JPEGImages')

        for im_name in im_names:
            ann_file = os.path.join(ann_dir, '{}.xml'.format(im_name))
            im_info = {}
            ann_info = ET.parse(ann_file)
            root = ann_info.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            im_info['img_id'] = os.path.basename(ann_file).split('.xml')[0]
            im_info['filename'] = os.path.join(
                im_dir, '{}.jpg'.format(im_info['img_id'])
            )
            im_info['width'] = width
            im_info['height'] = height
            detections = []
            for obj in ann_info.findall('object'):
                det = {}
                label = label2idx[obj.find('name').text]
                difficult = int(obj.find('difficult').text)
                bbox_info = obj.find('bndbox')
                bbox = [
                    int(bbox_info.find('xmin').text) - 1,
                    int(bbox_info.find('ymin').text) - 1,
                    int(bbox_info.find('xmax').text) - 1,
                    int(bbox_info.find('ymax').text) - 1
                ]
                det['label'] = label
                det['bbox'] = bbox
                det['difficult'] = difficult
                detections.append(det)
            im_info['detections'] = detections
            # Because we are using 25 as num_queries,
            # so we ignore all images in VOC with greater
            # than 25 target objects.
            # This is okay, since this just means we are
            # ignoring a small number of images(15 to be precise)
            if len(detections) <= 25:
                im_infos.append(im_info)
    print('Total {} images found'.format(len(im_infos)))
    return im_infos

def get_voc_labels(transform_input):
    """Global helper function to extract labels for pickling compatibility on Windows."""
    return transform_input[1]["labels"]

class VOCDataset(Dataset):
    r"""
    Pascal VOC detection dataset.

    :param split: which imageset to read - 'train' reads trainval.txt, anything
        else reads test.txt
    :param im_sets: VOC root dirs (VOC2007 / VOC2007+VOC2012 / VOC2007-test)
    :param im_size: square size every image is resized to
    :param transform_split: which transform pipeline to apply, independent of
        which imageset was read. Defaults to ``split``.

        This exists so the held-out validation slice of the *trainval* imageset
        can be read with the train file list but evaluated with the *test*
        (augmentation-free) pipeline. Without it, ``split`` would decide both,
        and validation images would be randomly flipped, zoomed, cropped and
        colour-jittered - which makes the validation loss a moving target and
        measures the model on inputs that never occur at inference.
    """

    def __init__(self, split, im_sets, im_size=640, transform_split=None):
        self.split = split
        # Decoupled from self.split - see the class docstring.
        self.transform_split = transform_split or split

        # Imagesets for this dataset instance (VOC2007/VOC2007+VOC2012/VOC2007-test)
        self.im_sets = im_sets
        self.fname = 'trainval' if self.split == 'train' else 'test'
        self.im_size = im_size
        self.im_mean = [123.0, 117.0, 104.0]
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

        # Train and test transformations
        self.transforms = {
            'train': torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.v2.RandomZoomOut(fill=self.im_mean),
                torchvision.transforms.v2.RandomIoUCrop(),
                torchvision.transforms.v2.RandomPhotometricDistort(),
                torchvision.transforms.v2.Resize(size=(self.im_size, self.im_size)),
                torchvision.transforms.v2.SanitizeBoundingBoxes(
                    labels_getter=get_voc_labels),
                # torchvision.transforms.v2.SanitizeBoundingBoxes(
                #     labels_getter=lambda transform_input:
                #     (transform_input[1]["labels"], transform_input[1]["difficult"])),
                torchvision.transforms.v2.ToPureTensor(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=self.imagenet_mean,
                                                    std=self.imagenet_std)

            ]),
            'test': torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.Resize(size=(self.im_size, self.im_size)),
                torchvision.transforms.v2.ToPureTensor(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=self.imagenet_mean,
                                                    std=self.imagenet_std)
            ]),
        }

        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]
        classes = sorted(classes)
        # VFL: no background class. VariFocal is a per-class *sigmoid* loss, so
        # "background" is not a label - it is simply every class sigmoid being
        # low. Labels are therefore 0..19 (foreground only). The softmax
        # formulation this replaced prepended 'background' at index 0, making
        # them 1..20; dataset_params.num_classes drops 21 -> 20 to match.

        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images_info = load_images_and_anns(self.im_sets,
                                                self.label2idx,
                                                self.fname,
                                                self.split)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = read_image(im_info['filename'])

        # Get annotations for this image
        targets = {}
        targets['boxes'] = tv_tensors.BoundingBoxes(
            [detection['bbox'] for detection in im_info['detections']],
            format='XYXY', canvas_size=im.shape[-2:])
        targets['labels'] = torch.as_tensor(
            [detection['label'] for detection in im_info['detections']])
        targets['difficult'] = torch.as_tensor(
            [detection['difficult']for detection in im_info['detections']])

        # Transform the image and targets
        # transform_split, not split: the two are deliberately independent so a
        # slice of trainval can be evaluated without augmentation.
        transformed_info = self.transforms[self.transform_split](im, targets)
        im_tensor, targets = transformed_info

        h, w = im_tensor.shape[-2:]

        # Boxes returned are in x1y1x2y2 format normalized from 0-1
        wh_tensor = torch.as_tensor([[w, h, w, h]]).expand_as(targets['boxes'])
        targets['boxes'] = targets['boxes'] / wh_tensor
        return im_tensor, targets, im_info['filename']