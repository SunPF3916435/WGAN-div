import torch
import os, glob
import random, csv
import torch.utils.data
import torchvision
import PIL.Image


class ImageData(torch.utils.data.Dataset):
    def __init__(self, directory, resize=224):
        super(ImageData, self).__init__()
        self.directory = directory
        self.resize = resize
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(directory))):
            if not os.path.isdir(os.path.join(directory, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        self.images, self.labels = self.load_csv('images.csv')

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.directory, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.directory, name, '*.png'))
                images += glob.glob(os.path.join(self.directory, name, '*.jpg'))
                images += glob.glob(os.path.join(self.directory, name, '*.jpeg'))
            random.shuffle(images)
            with open(os.path.join(self.directory, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
        images, labels = [], []
        with open(os.path.join(self.directory, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        transform = torchvision.transforms.Compose([
            lambda x: PIL.Image.open(x).convert('RGB'),
            torchvision.transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            torchvision.transforms.ToTensor()
        ])

        image = transform(image)
        label = torch.tensor(float(label))

        return image, label
