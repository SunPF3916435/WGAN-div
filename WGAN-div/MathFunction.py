import torch


def gradient_penalty(discriminator, x_real, x_fake, batch_size, k, p):
    t = torch.rand(batch_size, 1)
    t = t.expand_as(x_real)
    interpolate = t * x_real + (1 - t) * x_fake
    interpolate.requires_grad_()
    pred = discriminator(interpolate)

    gradient_x = torch.autograd.grad(outputs=pred, inputs=interpolate, grad_outputs=torch.ones(batch_size),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = torch.pow(gradient_x.norm(2, dim=1), p).mean()

    gp = k * gp

    return gp
