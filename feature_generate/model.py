import torch.nn as nn
import torch

'''class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.layer2 = nn.Linear(opt.ngh, opt.feaSize)
        self.elu = nn.ELU()

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h1 = self.elu(self.layer1(h))
        h2 = self.elu(self.layer2(h1))
        return h2

class Critic(nn.Module):
    def __init__(self, opt):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(opt.attSize + opt.feaSize, opt.ndh)
        self.layer2 = nn.Linear(opt.ndh, 1)
        self.elu = nn.ELU()

    def forward(self, fea, att):
        h = torch.cat((fea, att), 1)
        h1 = self.elu(self.layer1(h))
        h2 = self.elu(self.layer2(h1))
        return h2'''

''''class Generator_trans(nn.Module):
    def __init__(self, opt):
        super(Generator_trans, self).__init__()
        self.layer1 = nn.Linear(opt.attSize, int(opt.ngh / 2))
        self.layer2 = nn.Linear(int(opt.ngh / 2) + opt.nz, opt.ngh)
        self.layer3 = nn.Linear(opt.ngh, opt.feaSize)
        self.elu = nn.ELU()

    def forward(self, noise, att):
        hs = self.elu(self.layer1(att))
        h = torch.cat((noise, hs), 1)
        h1 = self.elu(self.layer2(h))
        h2 = self.elu(self.layer3(h1))
        return h2

class Critic_trans(nn.Module):
    def __init__(self, opt):
        super(Critic_trans, self).__init__()
        self.layer1 = nn.Linear(opt.attSize, int(opt.ngh / 2))
        self.layer2 = nn.Linear(int(opt.ngh / 2) + opt.feaSize, opt.ndh)
        self.layer3 = nn.Linear(opt.ndh, 1)
        self.elu = nn.ELU()

    def forward(self, fea, att):
        hs = self.elu(self.layer1(att))
        h = torch.cat((fea, hs), 1)
        h1 = self.elu(self.layer2(h))
        h2 = self.elu(self.layer3(h1))
        return h2'''

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        if self.opt.G_mode == 'ori':
            self.layer2 = nn.Linear(self.opt.attSize + self.opt.nz, self.opt.ngh)
        elif self.opt.G_mode == 'trans':
            self.layer1 = nn.Linear(self.opt.attSize, int(self.opt.ngh / 2))
            self.layer2 = nn.Linear(int(self.opt.ngh / 2) + self.opt.nz, self.opt.ngh)
        elif self.opt.G_mode == 'split':
            if self.opt.attSize == 7:
                self.layer1_1 = nn.Linear(5, int(self.opt.ngh / 6))
                self.layer1_2 = nn.Linear(1, int(self.opt.ngh / 6))
                self.layer1_3 = nn.Linear(1, int(self.opt.ngh / 6))
                self.layer2 = nn.Linear(3 * int(self.opt.ngh / 6) + self.opt.nz, self.opt.ngh)
            elif self.opt.attSize == 21:
                self.layer1_1 = nn.Linear(5, int(self.opt.ngh / 6))
                self.layer1_2 = nn.Linear(3, int(self.opt.ngh / 6))
                self.layer1_3 = nn.Linear(13, int(self.opt.ngh / 6))
                self.layer2 = nn.Linear(3 * int(self.opt.ngh / 6) + self.opt.nz, self.opt.ngh)
            else:
                print('not in att list')
                self.layer1 = nn.Linear(self.opt.attSize, int(self.opt.ngh / 2))
                self.layer2 = nn.Linear(int(self.opt.ngh / 2) + self.opt.nz, self.opt.ngh)
        else:
            print('not in G_mode list')
        self.layer3 = nn.Linear(self.opt.ngh, self.opt.feaSize)
        self.elu = nn.ELU()
        self.leaky = nn.LeakyReLU()
        self.floor_thickness = nn.Parameter(torch.FloatTensor([0.5]))
        self.floor_thickness.requires_grad_(True)
        self.floor_height = nn.Parameter(torch.FloatTensor([4.5/6]))
        self.floor_height.requires_grad_(True)
        self.receiver_height = nn.Parameter(torch.FloatTensor([1.5/6]))
        self.receiver_height.requires_grad_(True)
        self.x_param = nn.Parameter(torch.FloatTensor([10]))
        self.x_param.requires_grad_(True)

    def forward(self, noise, att):
        if self.opt.att_mode == 'lsp':
            if self.opt.attSize == 7:
                #print(att[0, :])
                hs_1 = att[:, 0:5]
                hs_2 = 1 / (torch.abs(self.floor_thickness * att[:, 5:6]) + torch.abs(self.floor_height * att[:, 5:6] - self.receiver_height))
                hs_3 = self.x_param / torch.sqrt(torch.pow(att[:, 6:7], 2 + 16))
                att = torch.cat((hs_1, hs_2, hs_3), 1)
                print(self.floor_thickness, self.floor_height, self.receiver_height, self.x_param)

            else:
                raise NameError('wrong att')
        else:
            pass

        if self.opt.G_mode == 'ori':
            h = torch.cat((noise, att), 1)
        elif self.opt.G_mode == 'trans':
            hs = self.elu(self.layer1(att))
            h = torch.cat((noise, hs), 1)
        elif self.opt.G_mode == 'split':
            if self.opt.attSize == 7:
                hs_1 = self.elu(self.layer1_1(att[:, 0:5]))
                hs_2 = self.elu(self.layer1_2(att[:, 5:6]))
                hs_3 = self.elu(self.layer1_3(att[:, 6:7]))
                hs = torch.cat((hs_1, hs_2, hs_3), 1)
            elif self.opt.attSize == 21:
                hs_1 = self.elu(self.layer1_1(att[:, 0:5]))
                hs_2 = self.elu(self.layer1_2(att[:, 5:8]))
                hs_3 = self.elu(self.layer1_3(att[:, 8:21]))
                hs = torch.cat((hs_1, hs_2, hs_3), 1)
            else:
                hs = self.elu(self.layer1(att))
            h = torch.cat((noise, hs), 1)
        else:
            raise NameError('not in G_mode list')
        h1 = self.elu(self.layer2(h))
        h2 = self.elu(self.layer3(h1))
        return h2

class Discriminator(nn.Module):
    def __init__(self, opt, floor_thickness=None, floor_height=None, receiver_height=None, x_param=None):
        super(Discriminator, self).__init__()
        self.opt = opt
        if self.opt.D_mode == 'ori':
            self.layer2 = nn.Linear(self.opt.attSize + self.opt.feaSize, self.opt.ndh)
        elif self.opt.D_mode == 'trans':
            self.layer1 = nn.Linear(self.opt.attSize, int(self.opt.ngh / 2))
            self.layer2 = nn.Linear(int(self.opt.ngh / 2) + self.opt.feaSize, self.opt.ndh)
        elif self.opt.G_mode == 'split':
            if self.opt.attSize == 7:
                self.layer1_1 = nn.Linear(5, int(self.opt.ndh / 6))
                self.layer1_2 = nn.Linear(1, int(self.opt.ndh / 6))
                self.layer1_3 = nn.Linear(1, int(self.opt.ndh / 6))
                self.layer2 = nn.Linear(3 * int(self.opt.ndh / 6) + self.opt.feaSize, self.opt.ndh)
            elif self.opt.attSize == 21:
                self.layer1_1 = nn.Linear(5, int(self.opt.ndh / 6))
                self.layer1_2 = nn.Linear(3, int(self.opt.ndh / 6))
                self.layer1_3 = nn.Linear(13, int(self.opt.ndh / 6))
                self.layer2 = nn.Linear(3 * int(self.opt.ndh / 6) + self.opt.feaSize, self.opt.ndh)
            else:
                print('not in att list')
                self.layer1 = nn.Linear(self.opt.attSize, int(self.opt.ndh / 2))
                self.layer2 = nn.Linear(int(self.opt.ndh / 2) + self.opt.feaSize, self.opt.ndh)
        else:
            print('not in D_mode list')
        self.layer3 = nn.Linear(self.opt.ndh, 1)
        self.elu = nn.ELU()
        self.leaky = nn.LeakyReLU()
        self.floor_thickness = floor_thickness
        self.floor_height = floor_height
        self.receiver_height = receiver_height
        self.x_param = x_param

        '''self.floor_thickness = nn.Parameter(torch.FloatTensor([1]))
        self.floor_thickness.requires_grad_(True)
        self.floor_height = nn.Parameter(torch.FloatTensor([1]))
        self.floor_height.requires_grad_(True)
        self.receiver_height = nn.Parameter(torch.FloatTensor([1]))
        self.receiver_height.requires_grad_(True)
        self.x_param = nn.Parameter(torch.FloatTensor([1]))
        self.x_param.requires_grad_(True)'''

    def forward(self, fea, att):
        if self.opt.att_mode == 'lsp':
            if self.opt.attSize == 7:
                hs_1 = att[:, 0:5]
                hs_2 = 1 / (torch.abs(self.floor_thickness * att[:, 5:6]) + torch.abs(self.floor_height * att[:, 5:6] - self.receiver_height))
                hs_3 = self.x_param / torch.sqrt(torch.pow(att[:, 6:7], 2 + 16))
                att = torch.cat((hs_1, hs_2, hs_3), 1)
            else:
                raise NameError('wrong att')
        else:
            pass
        if self.opt.G_mode == 'ori':
            h = torch.cat((fea, att), 1)
        elif self.opt.G_mode == 'trans':
            hs = self.elu(self.layer1(att))
            h = torch.cat((fea, hs), 1)
        elif self.opt.G_mode == 'split':
            if self.opt.attSize == 7:
                hs_1 = self.elu(self.layer1_1(att[:, 0:5]))
                hs_2 = self.elu(self.layer1_2(att[:, 5:6]))
                hs_3 = self.elu(self.layer1_3(att[:, 6:7]))
                hs = torch.cat((hs_1, hs_2, hs_3), 1)
            elif self.opt.attSize == 21:
                hs_1 = self.elu(self.layer1_1(att[:, 0:5]))
                hs_2 = self.elu(self.layer1_2(att[:, 5:8]))
                hs_3 = self.elu(self.layer1_3(att[:, 8:21]))
                hs = torch.cat((hs_1, hs_2, hs_3), 1)
            else:
                hs = self.elu(self.layer1(att))
            h = torch.cat((fea, hs), 1)
        else:
            raise NameError('not in D_mode list')
        h1 = self.elu(self.layer2(h))
        h2 = self.elu(self.layer3(h1))
        return h2
