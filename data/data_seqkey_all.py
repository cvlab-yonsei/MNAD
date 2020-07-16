import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
def make_dataset(dir, class_to_idx):
    frames = []
    print(sorted(class_to_idx.keys()))
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        print(target)
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
#         new_fnames = []
              
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
#                 fname = fname.split('.')[0]
#                 seq = fname.split('_')[0][1:]
#                 fname = fname.split('_')[1]
#                 fname = fname.zfill(4)
#                 new_fnames.append('V'+seq+'_'+fname+'.png')
                
                path = os.path.join(root, fname)
                frames.append(path)
       
    return frames


class DatasetFolder(data.Dataset):
   

    def __init__(self, root, loader=default_loader,transform=None, target_transform=None, length=5):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root))
        
        self.root = root
        self.loader = loader
        self.length = length
#         self.stride = np.random.choice(3,1) + 1
        self.classes = classes
        self.class_to_idx = class_to_idx
#         self.samples_gt = samples[self.length:]
        self.samples = samples[:-(self.length-1)]
        
        self.samples_all = samples
        self.samples_pool = samples[1:] 
#         self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (samples, gt(+length)) 
        
        """
        
        
        sample = []          
        
        path_start = self.samples[index]
        sample_start = self.loader(path_start)
        if self.transform is not None:
            sample_start = self.transform(sample_start)
       
       
        sample.append(sample_start) 
        
        for i in range(self.length - 1):
            path = self.samples_all[index + (i+1)]
            sample_immediate = self.loader(path)
            if self.transform is not None:
                sample_immediate = self.transform(sample_immediate)
             
            sample.append(sample_immediate)
        
        
#         path_gt = self.samples_gt[index]
#         sample_gt = self.loader(path_gt)
     
#         if self.transform is not None:
#             sample_gt = self.transform(sample_gt)
        
        sample_input = sample[0]
        for i in range(self.length-1):
            sample_input = torch.cat((sample_input,sample[i+1]), dim=0)

        return sample_input
    
    def _stride(self):
        
        stride = int(np.random.choice(3,1) + 1)
        #if stride != 1:
#             self.samples_gt = self.samples_all[self.length*stride:]
         #   self.samples = self.samples_all[:-(self.length*stride)]
        
        return stride

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']





class ImageFolder(DatasetFolder):
    
    
    
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, length=5):
        super(ImageFolder, self).__init__(root, loader,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
        


