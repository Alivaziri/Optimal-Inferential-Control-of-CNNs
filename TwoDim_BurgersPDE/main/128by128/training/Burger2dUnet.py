# the original code in https://github.com/isds-neu/PhyCRNet/blob/main/Codes/PhyCRNet_burgers.py
# The current code the modified version of the code written by Ali Vaziri @ alivaziri@ku.edu
# If you use this code, please consider to cite our paper.

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
import torch.nn.functional as F
from torch.backends import cudnn
import torch.utils.data as data
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)

# define the high-order finite difference kernels
lapl_op = [[[[    0,   0, -1/12,   0,     0],
             [    0,   0,   4/3,   0,     0],
             [-1/12, 4/3,    -5, 4/3, -1/12],
             [    0,   0,   4/3,   0,     0],
             [    0,   0, -1/12,   0,     0]]]]

partial_y = [[[[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [1/12, -8/12, 0, 8/12, -1/12],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]]]

partial_x = [[[[0, 0, 1/12, 0, 0],
               [0, 0, -8/12, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 8/12, 0, 0],
               [0, 0, -1/12, 0, 0]]]]

def initialize_weights(module):

    if isinstance(module, nn.Conv2d):
        #nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        c = 1 #0.5
        module.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), 
            c*np.sqrt(1 / (3 * 3 * 320)))
     
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class _EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, polling=True, bn=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU(),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)
        self.pool = None
        if polling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
            nn.GELU(),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.decode(x)

class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, upscale_factor = 8, step=1, effective_step=[1], dt = 0.002, bn=False, factors=1):
        super(UNet, self).__init__()
        self.step = step
        self.dt = dt
        self.effective_step = effective_step
        self.upscale_factor = upscale_factor
        self.enc1 = _EncoderBlock(in_channels, 32 * factors, polling=False, bn=bn)
        self.enc2 = _EncoderBlock(32 * factors, 64 * factors, bn=bn)
        self.enc3 = _EncoderBlock(64 * factors, 128 * factors, bn=bn)
        self.enc4 = _EncoderBlock(128 * factors, 256 * factors, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(256 * factors, 512 * factors, 256 * factors, bn=bn)
        self.dec4 = _DecoderBlock(512 * factors, 256 * factors, 128 * factors, bn=bn)
        self.dec3 = _DecoderBlock(256 * factors, 128 * factors, 64 * factors, bn=bn)
        self.dec2 = _DecoderBlock(128 * factors, 64 * factors, 32 * factors, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 * factors, 32 * factors, kernel_size=3, padding=0, padding_mode='circular'),
            nn.BatchNorm2d(32 * factors) if bn else nn.GroupNorm(32, 32 * factors),
            nn.GELU(),
            nn.Conv2d(32 * factors, 32 * factors, kernel_size=1,  padding=0, padding_mode='circular'),
            nn.BatchNorm2d(32 * factors) if bn else nn.GroupNorm(32, 32 * factors),
            nn.GELU(),
        )
        self.final = nn.Conv2d(32 * factors, num_classes, kernel_size=1,  padding=1, padding_mode='circular')

        self.apply(initialize_weights)
        nn.init.zeros_(self.final.bias)

    def forward(self, x):
        outputs = []
        for step in range(self.step):
            xt = x
            enc1 = self.enc1(x)
            enc2 = self.enc2(enc1)
            enc3 = self.enc3(enc2)
            enc4 = self.enc4(enc3)
            center = self.center(self.polling(enc4))
            dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                    mode='bilinear'), enc4], 1))
            dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                    mode='bilinear'), enc3], 1))
            dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                    mode='bilinear'), enc2], 1))
            dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                    mode='bilinear'), enc1], 1))
            final = self.final(dec1)
            # residual connection
            x = xt + self.dt * final
                
            if step in self.effective_step:
                outputs.append(x)        
        
        return outputs

class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol 
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)
        
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class loss_generator(nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, dt = (10.0/200), dx = (20.0/128)):
        ''' Construct the derivatives, X = Width, Y = Height '''
       
        super(loss_generator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv2dDerivative(
            DerFilter = lapl_op,
            resol = (dx**2),
            kernel_size = 5,
            name = 'laplace_operator').cuda()

        self.dx = Conv2dDerivative(
            DerFilter = partial_x,
            resol = (dx*1),
            kernel_size = 5,
            name = 'dx_operator').cuda()

        self.dy = Conv2dDerivative(
            DerFilter = partial_y,
            resol = (dx*1),
            kernel_size = 5,
            name = 'dy_operator').cuda()

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1, 0, 1]]],
            resol = (dt*2),
            kernel_size = 3,
            name = 'partial_t').cuda()

    def get_phy_Loss(self, output):

        # spatial derivatives
        laplace_u = self.laplace(output[1:-1, 0:1, :, :])  # [t,c,h,w]
        laplace_v = self.laplace(output[1:-1, 1:2, :, :])

        u_x = self.dx(output[1:-1, 0:1, :, :])
        u_y = self.dy(output[1:-1, 0:1, :, :])
        v_x = self.dx(output[1:-1, 1:2, :, :])
        v_y = self.dy(output[1:-1, 1:2, :, :])

        # temporal derivative - u
        u = output[:, 0:1, 2:-2, 2:-2]
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx*leny,1,lent)
        u_t = self.dt(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent-2)
        u_t = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        # temporal derivative - v
        v = output[:, 1:2, 2:-2, 2:-2]
        v_conv1d = v.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        v_conv1d = v_conv1d.reshape(lenx*leny,1,lent)
        v_t = self.dt(v_conv1d)  # lent-2 due to no-padding
        v_t = v_t.reshape(leny, lenx, 1, lent-2)
        v_t = v_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u = output[1:-1, 0:1, 2:-2, 2:-2]  # [t, c, height(Y), width(X)]
        v = output[1:-1, 1:2, 2:-2, 2:-2]  # [t, c, height(Y), width(X)]

        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        R = 200.0

        # 2D burgers eqn
        f_u = u_t + u * u_x + v * u_y - (1/R) * laplace_u
        f_v = v_t + u * v_x + v * v_y - (1/R) * laplace_v

        return f_u, f_v


def compute_loss(output, loss_func):
    ''' calculate the phycis loss '''
    
    # Padding x axis due to periodic boundary condition
    # shape: [t, c, h, w]
    output = torch.cat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), dim=3)

    # Padding y axis due to periodic boundary condition
    # shape: [t, c, h, w]
    output = torch.cat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), dim=2)

    # get physics loss
    mse_loss = nn.MSELoss()
    f_u, f_v = loss_func.get_phy_Loss(output)
    loss =  mse_loss(f_u, torch.zeros_like(f_u).cuda()) + mse_loss(f_v, torch.zeros_like(f_v).cuda()) 

    return loss


def train(model, input, n_iters, time_batch_size, learning_rate, 
          dt, dx, save_path, pre_model_save_path, num_time_batch):

    train_loss_list = []
    
    prev_output = []

    batch_loss = 0.0
    best_loss = 1e4

    # load previous model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
    scheduler = StepLR(optimizer, step_size=100, gamma=0.97)  
    model, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler, 
        pre_model_save_path)


    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    loss_func = loss_generator(dt, dx)
        
    for epoch in range(n_iters):
        # input: [t,b,c,h,w]
        random_index = random.randint(0, 1000)
        uv0 = input[random_index:random_index+1,...]
        inputuv0 = torch.tensor(uv0, dtype=torch.float32).cuda()
        
        optimizer.zero_grad()
        batch_loss = 0

        for time_batch_id in range(num_time_batch):
            # update the first input for each time batch
            if time_batch_id == 0:

                u0 = inputuv0
            else:

                u0 = prev_output[-2:-1].detach() # second last output

            # output is a list
            output = model(u0)

            # [t, c, height (Y), width (X)]
            output = torch.cat(tuple(output), dim=0)  

            # concatenate the initial state to the output for central diff
            output = torch.cat((u0.cuda(), output), dim=0)

            # get loss
            loss = compute_loss(output, loss_func)
            loss.backward(retain_graph=True)
            batch_loss += loss.item()

            # update the state and output for next batch
            prev_output = output

        optimizer.step()
        scheduler.step()

        # print loss in each epoch
        print('[%d/%d %d%%] loss: %.10f' % ((epoch+1), n_iters, ((epoch+1)/n_iters*100.0), 
            batch_loss))
        train_loss_list.append(batch_loss)

        # save model
        if batch_loss < best_loss:
            save_checkpoint(model, optimizer, scheduler, save_path)
            best_loss = batch_loss
    
    return train_loss_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def post_process(output, true, axis_lim, uv_lim, num, fig_save_path):
    # get the limit 
    xmin, xmax, ymin, ymax = axis_lim
    u_min, u_max, v_min, v_max = uv_lim

    # grid
    x = np.linspace(xmin, xmax, 128+1)
    # x = x[:-1]
    x_star, y_star = np.meshgrid(x, x)
    
    u_star = true[num, 0, 1:-1, 1:-1]
    u_pred = output[num, 0, 1:-1, 1:-1].detach().cpu().numpy()

    v_star = true[num, 1, 1:-1, 1:-1]
    v_pred = output[num, 1, 1:-1, 1:-1].detach().cpu().numpy()

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    cf = ax[0, 0].scatter(x_star, y_star, c=u_pred, alpha=0.9, edgecolors='none', 
        cmap='RdYlBu', marker='s', s=4, vmin=u_min, vmax=u_max)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_title('u')
    fig.colorbar(cf, ax=ax[0, 0])

    cf = ax[0, 1].scatter(x_star, y_star, c=u_star, alpha=0.9, edgecolors='none', 
        cmap='RdYlBu', marker='s', s=4, vmin=u_min, vmax=u_max)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_title('u-Ref.')
    fig.colorbar(cf, ax=ax[0, 1])

    cf = ax[1, 0].scatter(x_star, y_star, c=v_pred, alpha=0.9, edgecolors='none', 
        cmap='RdYlBu', marker='s', s=4, vmin=v_min, vmax=v_max)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    cf.cmap.set_under('whitesmoke')
    cf.cmap.set_over('black')
    ax[1, 0].set_title('v')
    fig.colorbar(cf, ax=ax[1, 0])

    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=0.9, edgecolors='none', 
        cmap='RdYlBu', marker='s', s=4, vmin=v_min, vmax=v_max)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    cf.cmap.set_under('whitesmoke')
    cf.cmap.set_over('black')
    ax[1, 1].set_title('v-Ref.')
    fig.colorbar(cf, ax=ax[1, 1])

    # plt.draw()
    plt.savefig(fig_save_path + 'uv_comparison_'+str(num).zfill(3)+'.png')
    plt.close('all')

    return u_star, u_pred, v_star, v_pred

def save_checkpoint(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, save_dir)


def load_checkpoint(model, optimizer, scheduler, save_dir):
    '''load model and optimizer'''
    
    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    if (not optimizer is None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('Pretrained model loaded!')

    return model, optimizer, scheduler


def summary_parameters(model):
    for i in model.parameters():
        print(i.shape)

def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))


if __name__ == '__main__':
    ######### download the ground truth data ############
    data_dir = '.../data/2dBurgers/burgers_50x2x128x128.pt'    
    data = scio.loadmat(data_dir)
    uv = data['uv'] # [t,c,h,w]  
    M = 128
    input = uv
    # grid parameters
    time_steps = 10
    dt = 0.002
    dx = 1.0 / M

    ################# build the model ###################
    time_batch_size = time_steps
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    num_time_batch = int(time_steps / time_batch_size)
    n_iters_adam = 10000
    lr_adam = 6e-4 #1e-3
    pre_model_save_path = 'path/to/models/model_name.pt'
    model_save_path = 'path/to/file/main/128by128/models/checkpoint.pt'
    fig_save_path = 'path/to/file/main/128by128/training/'
    cudnn.benchmark = True
    model  = UNet(in_channels=2, num_classes=2, upscale_factor = 8,
        step = steps, 
        effective_step = effective_step, dt = dt).cuda()

    start = time.time()
    train_loss = train(model, input, n_iters_adam, time_batch_size, 
        lr_adam, dt, dx, model_save_path, pre_model_save_path, num_time_batch)
    end = time.time()
    
    print('The training time is: ', (end-start))
    # plot train loss
    plt.figure()
    plt.plot(train_loss, label = 'train loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(fig_save_path + 'train loss.png', dpi = 300)

    ########### model inference ##################
    time_batch_size_load = time_steps
    steps_load = time_batch_size_load + 1
    num_time_batch = int(time_steps / time_batch_size_load)
    effective_step = list(range(0, steps_load))  
    
    model  = UNet(in_channels=2, num_classes=2, upscale_factor = 8,
        step = steps, 
        effective_step = effective_step, dt= dt).cuda()

    model, _, _ = load_checkpoint(model, optimizer=None, scheduler=None, save_dir=model_save_path) 
    output= model(input)

    # shape: [t, c, h, w] 
    output = torch.cat(tuple(output), dim=0)  
    output = torch.cat((input.cuda(), output), dim=0)
  
    # Padding x and y axis due to periodic boundary condition
    output = torch.cat((output[:, :, :, -1:], output, output[:, :, :, 0:2]), dim=3)
    output = torch.cat((output[:, :, -1:, :], output, output[:, :, 0:2, :]), dim=2)

    # [t, c, h, w]
    truth = uv[0:steps_load,:,:,:]

    # [101, 2, 131, 131]
    truth = np.concatenate((truth[:, :, :, -1:], truth, truth[:, :, :, 0:2]), axis=3)
    truth = np.concatenate((truth[:, :, -1:, :], truth, truth[:, :, 0:2, :]), axis=2)

    # post-process
    ten_true = []
    ten_pred = []
    for i in range(0, time_steps): 
        u_star, u_pred, v_star, v_pred = post_process(output, truth, [0,1,0,1], 
            [-0.7,0.7,-1.0,1.0], num=i, fig_save_path=fig_save_path)

        ten_true.append([u_star, v_star])
        ten_pred.append([u_pred, v_pred])

    # compute the error
    error = frobenius_norm(np.array(ten_pred)-np.array(ten_true)) / frobenius_norm(
        np.array(ten_true))

    print('The predicted error is: ', error)

    u_pred = output[:-1, 0, :, :].detach().cpu().numpy()
    u_pred = np.swapaxes(u_pred, 1, 2) # [h,w] = [y,x]
    u_true = truth[:, 0, :, :]

    t_true = np.linspace(0, 2, time_steps)
    t_pred = np.linspace(0, 2, time_steps)

    plt.plot(t_pred, u_pred[:, 32, 32], label='x=32, y=32')
    plt.plot(t_true, u_true[:, 32, 32], '--', label='x=32, y=32, Ref.')
    plt.xlabel('t')
    plt.ylabel('u')
    plt.xlim(0, 2)
    plt.legend()
    plt.savefig(fig_save_path + "x=32,y=32.png")
    plt.close("all")























