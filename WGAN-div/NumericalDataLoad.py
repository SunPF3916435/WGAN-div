import torch
import numpy as np
import torch.utils.data


class NumericalData(torch.utils.data.Dataset):
    def __init__(self, FileName, feature_num):
        super(NumericalData, self).__init__()
        self.FileName = FileName
        self.feature_num = feature_num
        self.data, self.labels = self.load_csv(self.FileName)

    def load_csv(self, filename):
        temp = np.loadtxt(self.FileName, delimiter=',', skiprows=1)
        data = temp[:, 0: self.feature_num]
        labels = temp[:, self.feature_num]
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].astype(np.float32)
        # x = torch.from_numpy(x)
        y = self.labels[idx].astype(np.float32)
        # y = torch.from_numpy(y)

        return x, y


