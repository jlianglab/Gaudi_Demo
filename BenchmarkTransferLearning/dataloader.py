import os
import torch
import random
import copy
import csv
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import pydicom as dicom
import cv2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
     RandomBrightnessContrast, RandomGamma,OneOf,
    ToFloat,GridDistortion, ElasticTransform,OpticalDistortion,RandomSizedCrop
)

def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0
    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height).T

def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []
    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence

def build_transform_segmentation():
  AUGMENTATIONS_TRAIN = Compose([
    OneOf([
        RandomBrightnessContrast(),
        RandomGamma(),
         ], p=0.3),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    RandomSizedCrop(min_max_height=(156, 224), height=224, width=224,p=0.25),
    ToFloat(max_value=1)
    ],p=1)

  return AUGMENTATIONS_TRAIN

class ChestXray14Dataset(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=14, annotaion_percent=100):
    self.img_list = []
    self.img_label = []
    self.augment = augment
    with open(file_path, "r") as fileDescriptor:
      line = True
      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split()
          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]
          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)
    indexes = np.arange(len(self.img_list))
    if annotaion_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotaion_percent / 100.0)
      indexes = indexes[:num_data]
      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []
      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])
    print("Number of images:", len(self.img_list))
  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: imageData = self.augment(imageData)
    return imageData, imageLabel

  def __len__(self):
    return len(self.img_list)


# ---------------------------------------------Downstream CheXpert------------------------------------------
class CheXpertDataset(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100):
    self.img_list = []
    self.img_label = []
    self.augment = augment
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label
    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              label[i] = -1
          else:
            label[i] = unknown_label # unknown label
        self.img_list.append(imagePath)
        imageLabel = [int(i) for i in label]
        self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]
      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []
      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    label = []
    for l in self.img_label[index]:
      if l == -1:
        if self.uncertain_label == "Ones":
          label.append(1)
        elif self.uncertain_label == "Zeros":
          label.append(0)
        elif self.uncertain_label == "LSR-Ones":
          label.append(random.uniform(0.55, 0.85))
        elif self.uncertain_label == "LSR-Zeros":
          label.append(random.uniform(0, 0.3))
      else:
        label.append(l)
    imageLabel = torch.FloatTensor(label)
    if self.augment != None: imageData = self.augment(imageData)
    return imageData, imageLabel

  def __len__(self):
    return len(self.img_list)

# ---------------------------------------------Downstream Shenzhen------------------------------------------
class ShenzhenCXR(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=1, annotation_percent=100):
    self.img_list = []
    self.img_label = []
    self.augment = augment
    with open(file_path, "r") as fileDescriptor:
      line = True
      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split(',')
          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]
          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: imageData = self.augment(imageData)
    return imageData, imageLabel

  def __len__(self):
    return len(self.img_list)

# ---------------------------------------------Downstream VinDrCXR------------------------------------------
class VinDrCXR(Dataset):
    def __init__(self, images_path, file_path, augment, num_class=6, annotation_percent=100):
        self.img_list = []
        self.img_label = []
        self.augment = augment
        with open(file_path, "r") as fr:
            line = fr.readline().strip()
            while line:
                lineItems = line.split()
                imagePath = os.path.join(images_path, lineItems[0]+".jpg")
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                self.img_list.append(imagePath)
                self.img_label.append(imageLabel)
                line = fr.readline()

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor(self.img_label[index])
        imageData = Image.open(imagePath).convert('RGB')
        if self.augment != None: imageData = self.augment(imageData)
        return imageData, imageLabel
    def __len__(self):
        return len(self.img_list)
#__________________________________________Lung Segmentation, Montgomery dataset --------------------------------------------------
class MontgomeryDataset(Dataset):
    def __init__(self, pathImageDirectory, pathMaskDirectory,transforms,dim=(224, 224, 3), anno_percent=100,num_class=1,normalization=None):
        self.transforms = transforms
        self.dim = dim
        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =pathMaskDirectory
        self.normalization = normalization
        self.img_list= os.listdir(pathImageDirectory)
        indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * anno_percent / 100.0)
            indexes = indexes[:num_data]
            _img_list= copy.deepcopy(self.img_list)
            self.img_list = []
            for i in indexes:
                self.img_list.append(_img_list[i])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image_name= self.img_list[idx]
        image = Image.open(os.path.join(self.pathImageDirectory,image_name))
        image = image.convert('RGB')
        image = (np.array(image)).astype('uint8')
        mask = Image.open(os.path.join(self.pathMaskDirectory,image_name))
        mask = mask.convert('L')
        mask = (np.array(mask)).astype('uint8')
        image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 255
        if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                im=augmented['image']
                mask=augmented['mask']
                im=np.array(im) / 255.
                mask=np.array(mask) / 255.
        else:
            im = np.array(image) / 255.
            mask = np.array(mask) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            im = (im-mean)/std
        mask = np.array(mask)
        im=im.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0).astype('uint8')
        return (im, mask)


#__________________________________________DRIVE dataset --------------------------------------------------

class DriveDataset(Dataset):
    def __init__(self, pathImageDirectory, pathMaskDirectory,size=512):
        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =pathMaskDirectory
        files = os.listdir(pathImageDirectory)
        data = []
        labels = []
        for i in files:
            im = Image.open(os.path.join(pathImageDirectory,i))
            im = im.convert('RGB')
            im = (np.array(im)).astype('uint8')
            label = Image.open(os.path.join(pathMaskDirectory, i.split('_')[0] + '_manual1.png'))
            label = label.convert('L')
            label = (np.array(label)).astype('uint8')
            data.append(cv2.resize(im, (size, size)))
            temp = cv2.resize(label, (size, size))
            _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
            labels.append(temp)
        self.data = np.array(data)
        self.label = np.array(labels)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.data = self.data.astype('float32') / 255.
        self.label = self.label.astype('float32') / 255.
        for i in range(3):
            self.data[:, :, :, i] = (self.data[:, :, :, i] - mean[i]) / std[i]
        self.data = np.reshape(self.data, (
            len(self.data), size, size, 3))
        self.label = np.reshape(self.label,
                             (len(self.label), size, size, 1))
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        mask = self.label[idx]
        image = image.transpose(2, 0, 1).astype('float32')
        mask = mask.transpose(2, 0, 1).astype('float32')
        return (image, mask)

#__________________________________________SIIM Pneumothorax segmentation dataset --------------------------------------------------
#changed to version from Adam-v2
class PNEDataset(Dataset):
    def __init__(self, root, directory, transforms, dim=(224, 224, 3),normalization=None,annotation=100):
        self.directory = directory
        self.images_masks = np.load(self.directory, allow_pickle=True)
        self.transforms = transforms
        self.dim = dim
        self.root=root
        self.normalization = normalization
        if annotation < 100:
            _images_masks = copy.deepcopy(self.images_masks)
            self.images_masks = []
            healthy=[]
            diseased=[]
            for i in range (_images_masks.shape[0]):
                m=_images_masks[i]
                if m[1].replace(" ", "") == '-1':
                    healthy.append(m)
                else:
                    diseased.append(m)
            num_healthy = int(len(healthy) * annotation / 100)
            num_diseased = int(len(diseased) * annotation / 100)
            indexes_healthy = np.arange(len(healthy))
            indexes_diseased = np.arange(len(diseased))
            random.Random().shuffle(indexes_healthy)
            random.Random().shuffle(indexes_diseased)
            indexes_healthy = indexes_healthy[:num_healthy]
            indexes_diseased = indexes_diseased[:num_diseased]
            for i in indexes_healthy:
                self.images_masks.append(healthy[i])
            for j in indexes_diseased:
                self.images_masks.append(diseased[j])

            self.images_masks = np.array(self.images_masks)
        print("number of data:",len(self.images_masks))
    def __len__(self):
        return self.images_masks.shape[0]
    def __getitem__(self, idx):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image_mask = self.images_masks[idx]
        ds = dicom.dcmread(os.path.join(self.root, str(image_mask[0])))
        img = np.array(ds.pixel_array)
        im = cv2.resize(img, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        im = (np.array(im)).astype('uint8')
        if len(im.shape) == 2:
            im = np.repeat(im[..., None], 3, 2)
        if image_mask[1].replace(" ", "") == '-1':
            msk = np.zeros((input_rows, input_cols), dtype="int")
        else:
            mask = rle2mask(image_mask[1], 1024, 1024)
            msk = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
            msk[msk > 0] = 255
            msk = (np.array(msk)).astype('uint8')
        if self.transforms:
                augmented = self.transforms(image=im, mask=msk)
                im=augmented['image']
                mask=augmented['mask']
                im=np.array(im) / 255.
                mask=np.array(mask) / 255.
        else:
            im = np.array(im) / 255.
            mask = np.array(msk) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            im = (im-mean)/std
        im=im.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0)
        return (im, mask)

#__________________________________________SCR Heart and Clavicle segmentation dataset --------------------------------------------------
class SCRDataset(Dataset):
    def __init__(self, pathImageDirectory, pathMaskDirectory, pathDatasetFile,transforms,dim=(224, 224, 3), anno_percent=100,num_class=1,normalization=None):
        self.transforms = transforms
        self.dim = dim
        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =pathMaskDirectory
        self.normalization = normalization
        self.img_list = []
        with open(pathDatasetFile, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline().strip('\n')
                if line:
                    self.img_list.append(line)
        indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * anno_percent / 100.0)
            if num_data ==0:
                num_data =1
            indexes = indexes[:num_data]
            _img_list = copy.deepcopy(self.img_list)
            self.img_list = []
            for i in indexes:
                self.img_list.append(_img_list[i])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image_name= self.img_list[idx]
        image = Image.open(os.path.join(self.pathImageDirectory,image_name))
        image = image.convert('RGB')
        image = (np.array(image)).astype('uint8')
        mask = Image.open(os.path.join(self.pathMaskDirectory,image_name))
        mask = mask.convert('L')
        mask = (np.array(mask)).astype('uint8')
        image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 255
        if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                im=augmented['image']
                mask=augmented['mask']
                im=np.array(im) / 255.
                mask=np.array(mask) / 255.
        else:
            im = np.array(image) / 255.
            mask = np.array(mask) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            im = (im-mean)/std
        mask = np.array(mask)
        im=im.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0).astype('uint8')
        return (im, mask)

#__________________________________________US Breast Cancer dataset --------------------------------------------------
class USBreastDataset(Dataset):
    def __init__(self, pathImageDirectory,pathMaskDirectory, pathDatasetFile,transforms,dim=(224, 224, 3), anno_percent=100,num_class=1,normalization=None):
        self.transforms = transforms
        self.dim = dim
        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory = pathMaskDirectory
        self.normalization = normalization
        self.img_list = []
        with open(pathDatasetFile, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline().strip('\n')
                if line:
                    self.img_list.append(line)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image_name= self.img_list[idx]
        image = Image.open(os.path.join(self.pathImageDirectory,image_name))
        image = image.convert('RGB')
        image = (np.array(image)).astype('uint8')
        mask = Image.open(os.path.join(self.pathMaskDirectory,image_name[:-4]+"_mask.png"))
        mask = mask.convert('L')
        mask = (np.array(mask)).astype('uint8')
        image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 255
        if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                im=augmented['image']
                mask=augmented['mask']
                im=np.array(im) / 255.
                mask=np.array(mask) / 255.
        else:
            im = np.array(image) / 255.
            mask = np.array(mask) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            im = (im-mean)/std
        mask = np.array(mask)
        im=im.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0).astype('uint8')
        return (im, mask)