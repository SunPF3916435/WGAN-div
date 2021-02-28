import torch.utils.data
import NumericalDataLoad
import Discriminator

db = NumericalDataLoad.NumericalData('Tesla.csv', 6)
train = torch.utils.data.DataLoader(db, batch_size=32, shuffle=True)

a = next(iter(train))

t = torch.randn(2, 6)
model = Discriminator.BaseDiscriminator(6)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

out1 = model(a[0])

loss = criterion(out1, a[1])
loss.backward()
optimizer.step()
print(loss)

