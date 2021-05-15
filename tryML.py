# PyTorch 1.8.1-CPU virtual env.
# Python 3.9.4 Windows 10
import torch
import numpy as np
print(torch.__version__)

device= torch.device("cpu")
class HouseDataset(torch.utils.data.Dataset):
    # AC  sq ft   style  price   school
    # -1  0.2500  0 1 0  0.5650  0 1 0
    #  1  0.1275  1 0 0  0.3710  0 0 1
    # air condition: -1 = no, +1 = yes
    # style: art_deco, bungalow, colonial
    # school: johnson, kennedy, lincoln
    def __init__(self, src_file, m_rows=None):
        all_xy = np.loadtxt(src_file, max_rows=m_rows,
        usecols=[0,1,2,3,4,5,6,7,8], delimiter="\t",
        comments="#", skiprows=0, dtype=np.float32)

        tmp_x = all_xy[:,[0,1,2,3,4,6,7,8]]
        tmp_y = all_xy[:,5].reshape(-1,1)

        self.x_data = torch.tensor(tmp_x, \
        dtype=torch.float32).to(device)
        self.y_data = torch .tensor(tmp_y, \
        dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        preds = self.x_data[idx,:]  # or just [idx]
        price = self.y_data[idx,:] 
        return (preds, price)       # tuple of matrices

# test_dataset.py
# PyTorch 1.7.0 CPU
# Python 3.7.6

torch.manual_seed(0)

src = ".\\Data\\houses_train.txt"
train_ds = HouseDataset(src, m_rows=5)
train_ldr = torch.utils.data.DataLoader(train_ds,
  batch_size=2, shuffle=True)
for epoch in range(2):
  print("\n\n Epoch = " + str(epoch))
  for (bat_idx, batch) in enumerate(train_ldr):
    print("------------------------------")
    X = batch[0]  # batch is tuple of two matrices
    Y = batch[1]
    print("bat_idx = " + str(bat_idx))
    print(X)
    print(Y)

print("\nEnd test ")

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = torch.nn.Linear(8, 10)  # 8-(10-10)-1
    #self.drop1 = torch.nn.Dropout(0.50)
    self.hid2 = torch.nn.Linear(10, 10)
    #self.drop2 = torch.nn.Dropout(0.25)
    self.oupt = torch.nn.Linear(10, 1)

    torch.nn.init.xavier_uniform_(self.hid1.weight)
    torch.nn.init.zeros_(self.hid1.bias)
    torch.nn.init.xavier_uniform_(self.hid2.weight)
    torch.nn.init.zeros_(self.hid2.bias)
    torch.nn.init.xavier_uniform_(self.oupt.weight)
    torch.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    z = torch.relu(self.hid1(x))
    #z = self.drop1(z)
    z = torch.relu(self.hid2(z))
    #z = self.drop2(z)
    z = self.oupt(z)  # no activation
    return z

net = Net().to(device)

#loop max_epochs times
#     loop thru all batches of train data
#       read a batch of data (inputs, targets)
#       compute outputs using the inputs
#       compute error between outputs and targets
#       use error to update weights and biases
#     end-loop (all batches)
#   end-loop (all epochs)

print("\nBegin test of training code\n")
  
torch.manual_seed(1)
np.random.seed(1)
train_file = ".\\Data\\houses_train.txt"
train_ds = HouseDataset(train_file, m_rows=200) 

bat_size = 10
train_ldr = torch.utils.data.DataLoader(train_ds,
  batch_size=bat_size, shuffle=True)

net = Net().to(device)
net.train()  # set mode

lrn_rate = 0.005
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),
  lr=lrn_rate)

for epoch in range(0, 500):
  # T.manual_seed(1 + epoch)  # recovery reproduce
  epoch_loss = 0.0  # sum avg loss per item

  for (batch_idx, batch) in enumerate(train_ldr):
    X = batch[0]  # predictors shape [10,8]
    Y = batch[1]  # targets shape [10,1] 

    optimizer.zero_grad()
    oupt = net(X)            # shape [10,1]

    loss_val = loss_func(oupt, Y)  # avg loss in batch
    epoch_loss += loss_val.item()  # a sum of averages
    loss_val.backward()
    optimizer.step()

  if epoch % 100 == 0:
    print(" epoch = %4d   loss = %0.4f" % \
     (epoch, epoch_loss))
    # TODO: save checkpoint

print("\nDone ")
 
import time

if epoch % 50 == 0:
  dt = time.strftime("%Y_%m_%d-%H_%M_%S")
  fn = ".\\Log\\" + str(dt) + str("-") + \
   str(epoch) + "_checkpoint.pt"
  info_dict = { 
    'epoch' : epoch,
    'net_state' : net.state_dict(),
    'optimizer_state' : optimizer.state_dict() 
  }
  torch.save(info_dict, fn)

fn = ".\\Log\\2021_03_25-10_32_57-700_checkpoint.pt"
chkpt = T.load(fn)
net = Net().to(device)
net.load_state_dict(chkpt['net_state'])
optimizer.load_state_dict(chkpt['optimizer_state'])
  . . .
epoch_saved = chkpt['epoch'] + 1
for epoch in range(epoch_saved, max_epochs):
  T.manual_seed(1 + epoch)
  # resume training as usual


#loop thru each data item
#  get item predictor input values
#  get item target value
#  use inputs to compute output value 
#  
#  compute actual difference predicted, target
#  compute max allowed difference
#  if actual diff < max allowed
#    correct prediction
#  else
#    wrong prediction
#end-loop
#return num correct / (num correct + num wrong)
#

# ----------------------------------------------------

def accuracy(model, ds, pct):
  n_correct = 0; n_wrong = 0

  for i in range(len(ds)):    # one item at a time
    (X, Y) = ds[i]            # (predictors, target)
    with torch.no_grad():
      oupt = model(X)         # computed price

    print("-----------------------------")
    abs_delta = np.abs(oupt.item() - Y.item())
    max_allow = np.abs(pct * Y.item())
    print("predicted = %0.4f  actual = %0.4f \
 delta = %0.4f  max_allow = %0.4f : " % (oupt.item(),\
 Y.item(), abs_delta, max_allow),  end="")
    if abs_delta < max_allow:
      print("correct")
      n_correct +=1
    else:
      print("wrong")
      n_wrong += 1

  acc = (n_correct * 1.0) / (n_correct + n_wrong)
  return acc

# ----------------------------------------------------

print("Begin accuracy() test for House prices")

torch.manual_seed(1)
np.random.seed(1)
train_file = ".\\Data\\houses_train.txt"
train_ds = HouseDataset(train_file, m_rows=6)

net = Net().to(device)
# net = net.train()
# train network

net = net.eval()
acc = accuracy(net, train_ds, 0.10)  # within 10%
print("\nAccuracy = %0.4f" % acc)
print("\nEnd test ")

#save model
print("Saving trained model state dict ")
path = ".\\Models\\houses_model.pth"
torch.save(net.state_dict(), path)

#load model
model = Net().to(device)
path = ".\\Models\\houses_model.pth"
model.load_state_dict(torch.load(path))
x = torch.tensor([[-1, 0.2300,  0,0,1,  0,1,0]],
  dtype=torch.float32)
with torch.no_grad():
  y = model(x)
print("Prediction is " + str(y))