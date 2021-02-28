import torch


class BaseDiscriminator(torch.nn.Module):
    def __init__(self, input, output=1, h_dim=40):
        super(BaseDiscriminator, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input, h_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(h_dim, output)
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)
