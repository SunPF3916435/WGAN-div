import torch
import torch.utils.data
import Hyper_parameters as hp
import Generator
import Discriminator
import NumericalDataLoad
import MathFunction
import visdom

viz = visdom.Visdom()

epochs = hp.epochs
batch_size = hp.batch_size
learning_rate = hp.learning_rate
random_seed = hp.random_seed

gradient_penalty_k = hp.gradient_penalty_k
gradient_penalty_p = hp.gradient_penalty_p

db = NumericalDataLoad.NumericalData('Tesla.csv', 6)
real_data = torch.utils.data.DataLoader(db, batch_size=32, shuffle=True)


Generator = Generator.BaseGenerator(3, 6)
Discriminator = Discriminator.BaseDiscriminator(6)

optimizer_G = torch.optim.Adam(Generator.parameters(), lr=5e-4, betas=(0.5, 0.9))
optimizer_D = torch.optim.Adam(Discriminator.parameters(), lr=5e-4, betas=(0.5, 0.9))

for epoch in range(epochs):
    for _ in range(5):
        x_real = next(iter(real_data))[0]
        pred_real = Discriminator(x_real)
        loss_real = -pred_real.mean()

        z = torch.randn(batch_size, 3)
        x_fake = Generator(z)
        pred_fake = Discriminator(x_fake)
        loss_fake = pred_fake.mean()

        gradient_penalty = MathFunction.gradient_penalty(Discriminator, x_real, x_fake.detach(), batch_size,
                                                         gradient_penalty_k, gradient_penalty_p)

        loss_D = loss_fake + loss_real + gradient_penalty

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

    z = torch.randn(batch_size, 3)
    x_fake = Generator(z)
    pred_fake = Discriminator(x_fake)
    loss_G = -pred_fake.mean()
    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()
    if epoch % 2 == 0:
        viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')
        print(loss_D.item(), loss_G.item())