import torch


class BaseGenerator(torch.nn.Module):
    def __init__(self, input, output, h_dim=40):
        super(BaseGenerator, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input, h_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(h_dim, output)
        )

    def forward(self, z):
        output = self.net(z)
        return output
