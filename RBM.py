import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, SGD, RMSprop, Adagrad, Adadelta,AdamW


class RBM(nn.Module):
    def init_optimizer(self, optimizer_name=None, learning_rate=None, momentum=None, weight_decay=None):
        if optimizer_name == "adam":
            self.opt =  Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            self.opt = SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == "RMSprop":
            self.opt = RMSprop(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "AdamW":
            self.opt = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "Adadelta":
            self.opt = Adadelta(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def __init__(self,
                 n_vis=256,
                 n_hin=500,
                 k=5,
                 device=None,optimizer_name = "adam",learning_rate=0.0001,weight_decay=1e-5,momentum=1,weight_initializer='xavier'):
        super(RBM, self).__init__()
        self.device = device
        self.W = nn.Parameter(torch.randn(n_hin, n_vis) * 1e-2)

        if weight_initializer == "xavier_uniform":
            self.W = torch.nn.init.xavier_uniform_(self.W, gain=1.0)
        elif weight_initializer == "xavier_normal":
            self.W = torch.nn.init.xavier_normal(self.W, gain=1.0)
        elif weight_initializer == "kaiming_normal":
            self.W = torch.nn.init.kaiming_normal(self.W)
        elif weight_initializer == "kaiming_uniform":
            self.W = torch.nn.init.kaiming_uniform(self.W)
        elif weight_initializer == "orthogonal":
            self.W = torch.nn.init.orthogonal(self.W)

        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k
        self.verbose_levels = [0, 1, 10, 100, 1000]
        self.History = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "epoch_time": []}

        self.num_features = n_vis
        self.init_optimizer(optimizer_name,learning_rate,weight_decay,momentum)


    def sample_from_p(self, p):
        return F.relu(torch.sign(p - torch.rand(p.size()).to(self.device)))

    def v_to_h(self, v):
        p_h = F.relu(F.linear(v.to(self.device), self.W.to(self.device), self.h_bias.to(self.device)).to(self.device))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        p_v = F.relu(F.linear(h.to(self.device), self.W.t().to(self.device), self.v_bias.to(self.device)))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def forward(self, v):
        pre_h1, h1 = self.v_to_h(v)

        h_ = h1
        for _ in range(self.k):
            pre_v_, v_ = self.h_to_v(h_)
            pre_h_, h_ = self.v_to_h(v_)

        return v, v_

    def free_energy(self, v):
        v = v.to(self.device)
        vbias_term = v.mv(self.v_bias.to(self.device))
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return torch.mean((-hidden_term - vbias_term))

    def print_progress(self, phase, loss, current_epoch, verbose):
        if current_epoch % self.verbose_levels[verbose] == 0:
            print("Phase " + str(phase) + " - " + "loss: " + str(loss))

    def fit(self,num_epochs=10,train_loader=None,val_loader=None,prob=0.5,verbose=1):
        for epoch in range(num_epochs):
            start = time.time()
            if (verbose != 0):
                if (epoch + 1) % self.verbose_levels[verbose] == 0:
                    print("\n")
                    print("Epoch: " + str(epoch + 1) + "/" + str(num_epochs) + " - ║{0:20s}║ {1:.1f}%".format(
                        '🟩' * int((epoch + 1) / num_epochs * 20), (epoch + 1) / num_epochs * 100))
            loss_ = []
            total_loss = 0
            val_loss_ = []
            total_val_loss = 0
            for _, (data, target) in enumerate(train_loader):
                data = Variable(data.view(-1, self.num_features))
                sample_data = torch.bernoulli(data,p=prob)
                v, v1 = self.forward(sample_data)
                loss = self.free_energy(v) - self.free_energy(v1)
                total_loss += loss
                loss_.append(loss)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            self.print_progress("Train",(sum(loss_) / len(loss_)).item(), epoch + 1, verbose)
            with torch.no_grad():
                self.eval()
                for _, (data, target) in enumerate(val_loader):
                    data = Variable(data.view(-1, self.num_features)).to(self.device)
                    sample_data = torch.bernoulli(data,p=prob)

                    v, v1 = self.forward(sample_data)
                    val_loss = self.free_energy(v) - self.free_energy(v1)
                    total_val_loss += val_loss
                    val_loss_.append(val_loss)
            self.print_progress("Validation", (sum(val_loss_) / len(val_loss_)).item(), epoch + 1, verbose)
            end = time.time()
            print("Epoch time elapsed: " + str((end - start)))
            self.History["loss"].append((total_loss.cpu().detach().numpy() / len(train_loader)))
            self.History["val_loss"].append((total_val_loss.cpu().detach().numpy() / len(val_loader)))
            self.History["epoch_time"].append(end - start)

        return self.History