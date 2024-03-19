import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom dataset that mimics the behavior of torchvision.datasets.ImageFolder.
        :param root_dir: Directory with all the images organized in subdirectories
                         (one subdirectory per class).
        :param transform: Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(root_dir)
        self.samples = self._make_dataset(root_dir, self.class_to_idx)
        
    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, dir, class_to_idx):
        """
        Creates a list of samples with their class indices.
        """
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self._is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
        return images

    @staticmethod
    def _is_image_file(filename):
        """
        Checks if a file is an image.
        """
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Fetches a sample and its label at a given index.
        """
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
            
        if self.transform:
            image = self.transform(image)

        return image, target