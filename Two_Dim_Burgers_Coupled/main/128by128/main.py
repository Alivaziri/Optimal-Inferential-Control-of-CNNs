# __author__ = "Ali Vaziri"
# __copyright__ = "Copyright (C) 2024 Ali Vaziri"
# __license__ = "Public Domain"
# __version__ = "1.0"
# TIf questions, email me at: alivaziri@ku.edu
# If you use this code, please consider to cite our paper.

import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import scipy.io as scio
from torch.backends import cudnn
import random
import warnings
from scipy.stats import matrix_normal
warnings.filterwarnings('ignore')
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

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
        xt = x[:,0:2,:,:]
        enc1 = self.enc1(xt)
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
        output = xt + self.dt * (final + x[:,2:4,:,:])   
        
        return output


def load_checkpoint(model, optimizer, scheduler, save_dir):
    '''load model and optimizer'''

    checkpoint = torch.load(save_dir, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    if (not optimizer is None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('Pretrained model loaded!')

    return model, optimizer, scheduler

def measurement(self):
    return self.samples[:, -self.nxbar:][:, :self.nx_Domain]

class KalmanMPC(nn.Module):
    '''
    Kalman-MPC model 
    '''
    def __init__(self, model, Ns, x0, nx_Domain, ny_Domain, nt, Xref):
        super(KalmanMPC, self).__init__()
        self.model = model
        self.Ns = Ns
        self.nx_Domain = 2*nx_Domain
        self.ny_Domain = ny_Domain
        self.k = 1
        self.mean = torch.zeros((self.nx_Domain, self.ny_Domain))
        self.nxbar = self.nx_Domain * 3

        self.R = torch.eye((self.nx_Domain), device = device) * 1e-10
        self.sigma_ProcessCov = torch.eye(self.nx_Domain) * 1e-7
        self.psi_ProcessCov = torch.eye(self.ny_Domain) * 1e-7

        self.u = torch.zeros((self.nx_Domain, self.ny_Domain, nt), device=device)
        self.v = torch.zeros((self.nx_Domain, self.ny_Domain, nt), device=device)
        self.v[:, :, 0] = x0.reshape(self.nx_Domain, self.ny_Domain)

        self.samples = self.initialize_samples(self.v[:, :, 0])
        self.sample_mean = torch.mean(self.samples, dim=0).to(device)
        self.yk = Xref
        self.MPC_iter = 0

    def initialize_samples(self, v0):
        delta_uk = torch.tensor(matrix_normal.rvs(mean=self.mean, rowcov = self.sigma_ProcessCov, colcov = self.psi_ProcessCov, size=self.Ns)).to(device)
        uk = torch.tensor(matrix_normal.rvs(mean=self.mean, rowcov = self.sigma_ProcessCov, colcov = self.psi_ProcessCov, size=self.Ns)).to(device)
        uk = uk + delta_uk
        v0_repeated = v0.unsqueeze(0).repeat(self.Ns, 1, 1)
        samples = torch.cat([v0_repeated, uk, delta_uk], dim=1)
        return samples

    def predict(self): 
        particles_current = self.samples[:, -self.nxbar:,]
        State_particles_current = particles_current[:, :self.nx_Domain, ]
        Control_particles_current = particles_current[:, self.nx_Domain: 2 * self.nx_Domain, ]
        x_pred = self.model(torch.cat([State_particles_current.reshape(self.Ns,2,self.ny_Domain,self.ny_Domain).to(dtype=torch.float32), Control_particles_current.reshape(self.Ns,2,self.ny_Domain,self.ny_Domain).to(dtype=torch.float32)], dim = 1))
        w_bar = torch.tensor(matrix_normal.rvs(mean=self.mean, rowcov = self.sigma_ProcessCov, colcov = self.psi_ProcessCov, size=self.Ns)).to(device)
        Control_particles_current += w_bar
        x_pred = torch.cat([x_pred.reshape(self.Ns, self.nx_Domain, self.ny_Domain), Control_particles_current, w_bar], dim=1)
        self.samples = torch.cat([self.samples, x_pred], dim=1)
        self.sample_mean = torch.mean(self.samples, dim=0)

    def update(self):
        y_particles = measurement(self) 
        y_hat = torch.mean(y_particles, dim=0)
        Py = (1 / (self.Ns)) * torch.sum(torch.bmm((y_particles - y_hat.unsqueeze(0).repeat(self.Ns, 1, 1)) , (y_particles - y_hat.unsqueeze(0).repeat(self.Ns, 1, 1)) .permute(0, 2, 1)), dim=0)  + self.R 
        Pxy = (1 / (self.Ns)) * torch.sum(torch.bmm((self.samples - self.sample_mean.unsqueeze(0).repeat(self.Ns, 1, 1)), (y_particles - y_hat.unsqueeze(0).repeat(self.Ns, 1, 1)).permute(0, 2, 1)), dim=0)
        invPy = torch.inverse(Py)
        K = Pxy @ invPy
        self.samples = self.samples + \
            torch.bmm(K.repeat(self.Ns, 1, 1), (self.yk[..., self.MPC_iter] - y_particles))
        
    def warmstart(self, k):
        delta_uk = torch.tensor(matrix_normal.rvs(mean=self.mean, rowcov = self.sigma_ProcessCov, colcov = self.psi_ProcessCov, size=self.Ns)).to(device)
        self.samples = torch.cat([self.v[..., k - 1].unsqueeze(0).repeat(self.Ns, 1, 1),
                                       delta_uk + self.samples[:, self.nxbar + self.nx_Domain:self.nxbar + 2* self.nx_Domain, :],
                                       delta_uk], dim=1)
        self.sample_mean = torch.mean(self.samples, dim=0)

# main code
torch.manual_seed(5)
data_dir = 'path/to/data/2dBurgers/burgers_50x2x128x128.pt'    
data = scio.loadmat(data_dir)
uv = data['uv']  # [t,c,h,w]
# initial conidtion
counter = random.randint(0, 49)
x0 = uv[counter, ...]
x0 = torch.tensor(x0, dtype=torch.float32)
input = uv[counter,...]
nx_Domain = 128
ny_Domain = 128
L = 1
T = 2
dx = L / nx_Domain
dy = L / ny_Domain
dt = 0.002
nt = int(T / dt)
t = torch.arange(0, T + dt, dt)
sim_time = torch.zeros(nt)
RMSE1 = torch.zeros(nt+1).to(device)
RMSE2 = torch.zeros(nt+1).to(device)
Xref = 0 * torch.ones((2*nx_Domain, nx_Domain, nt + 50)).to(device)

# Kalman-MPC
Ns = 100
Horizon = 5
time_steps = Horizon
effective_step = list(range(0, time_steps))


model_save_path = '...path/to/models/checkpoint.pt'
fig_save_path = '...path/to/figure_control/'
model  = UNet(in_channels=2, num_classes=2, upscale_factor = 8,
        effective_step = effective_step, dt = dt).to(device)
model, _, _ = load_checkpoint(model, optimizer=None, scheduler=None, save_dir=model_save_path)
EnKS = KalmanMPC(model, Ns, x0, nx_Domain, ny_Domain,nt,Xref)
EnKS.to(device)

# main loop for Kalman-MPC
for k in range(1, nt):
    # warmstart
    if k > 1:
        EnKS.warmstart(k)

    time_iter = 0
    for j in range(1, Horizon):
        start_time = time.time()
        EnKS.MPC_iter = k + j - 2
        # prediction step for Kalman-MPC
        EnKS.predict()
        # update step for Kalman-MPC
        EnKS.update()
        time_iter += time.time() - start_time

    sim_time[k - 1] = time_iter
    print(f"Iteration {k}, Elapsed Time: {time_iter:.6f} seconds")
    EnKS.u[..., k - 1] = torch.mean(EnKS.samples[:, EnKS.nx_Domain:2* EnKS.nx_Domain], dim=0)

    # Run system
    EnKS.v[..., k] = model(torch.cat([EnKS.v[..., k-1].reshape(1,2,nx_Domain,nx_Domain).to(dtype=torch.float32), EnKS.u[..., k - 1].reshape(1,2,nx_Domain,nx_Domain).to(dtype=torch.float32)], dim = 1)).reshape(2*nx_Domain,nx_Domain)
    EnKS.k = k + 1

    if k % 500 == 0:
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 12))
        ax1.plot(t[1:k-1], RMSE1[1:k-1].cpu().numpy())
        ax1.set_title('RMSE of first velocity dimension with Reference', fontsize=16)
        ax1.set_xlabel('time', fontsize=16)
        ax1.set_ylabel('RMSE', fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.ticklabel_format(style='plain', useOffset=False)
        ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax1.grid(True)

        ax2.plot(t[1:k-1], RMSE2[1:k-1].cpu().numpy())
        ax2.set_title('RMSE of second velocity dimension with Reference', fontsize=16)
        ax2.set_xlabel('time', fontsize=16)
        ax2.set_ylabel('RMSE', fontsize=16)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.ticklabel_format(style='plain', useOffset=False)
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax2.grid(True)
        plt.tight_layout()
        fig.savefig(fig_save_path + f'error_plot_iteration_{k}.png')
        plt.close(fig)

    RMSE1[k] = torch.sqrt(torch.mean(torch.abs(EnKS.v[:nx_Domain,:, k - 1] - Xref[:nx_Domain,:, k-1]).pow(2)))
    RMSE2[k] = torch.sqrt(torch.mean(torch.abs(EnKS.v[nx_Domain:2*nx_Domain,:, k - 1] - Xref[nx_Domain:2*nx_Domain,:, k-1]).pow(2)))