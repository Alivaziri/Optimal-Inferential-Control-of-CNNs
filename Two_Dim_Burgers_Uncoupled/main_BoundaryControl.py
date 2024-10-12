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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
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
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
            nn.GELU(),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes=2, in_channels = 3, upscale_factor = 8, step=1, effective_step=[1], bn=False, factors=1, dt = None):
        super(UNet, self).__init__()
        self.step = step
        self.dt = dt
        self.effective_step = effective_step
        self.upscale_factor = upscale_factor

        self.enc1 = _EncoderBlock(in_channels, 32 * factors, polling=False, bn=bn)
        self.enc2 = _EncoderBlock(32 * factors, 64 * factors, bn=bn)
        self.enc3 = _EncoderBlock(64 * factors, 128 * factors, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(128 * factors, 256 * factors, 128 * factors, bn=bn)
        self.dec3 = _DecoderBlock(256 * factors, 128 * factors, 64 * factors, bn=bn)
        self.dec2 = _DecoderBlock(128 * factors, 64 * factors, 32 * factors, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 * factors, 32 * factors, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(32 * factors) if bn else nn.GroupNorm(32, 32 * factors),
            nn.GELU(),
            nn.Conv2d(32 * factors, 32 * factors, kernel_size=1, padding=0),
            nn.BatchNorm2d(32 * factors) if bn else nn.GroupNorm(32, 32 * factors),
            nn.GELU(),
        )
        self.final = nn.Conv2d(32 * factors, num_classes, kernel_size=1)
        self.apply(initialize_weights)
        nn.init.zeros_(self.final.bias)

    def forward(self, x):
        xt = x[:,0:1,:,:]
        enc1 = self.enc1(xt)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        center = self.center(self.polling(enc3))
        dec3 = self.dec3(torch.cat([F.interpolate(center, enc3.size()[-2:], align_corners=False,
                                                mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        output = xt + self.dt * (final + x[:,1:2,:,:])
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
    particles_current = self.samples[:, -self.nxbar:]
    return particles_current[:, :self.nx_Domain]

def RK4_Burgers_solver_real(self, k):
    Vn = self.v[..., k-1]
    F = EnKS.u[..., k - 1]
    F = torch.tensor(F)
    viscosity = 1/200

    ind_x = torch.zeros((self.nx_Domain, 3))
    ind_x[:, 1] = torch.arange(0, self.nx_Domain)  # i
    ind_x[:, 0] = torch.roll(ind_x[:, 1], 1, dims=0)  # i-1
    ind_x[:, -1] = torch.roll(ind_x[:, 1], -1, dims=0)  # i+1

    ind_y = torch.zeros((self.ny_Domain, 3))
    ind_y[:, 1] = torch.arange(0, self.ny_Domain)  # i
    ind_y[:, 0] = torch.roll(ind_y[:, 1], 1, dims=0)  # i-1
    ind_y[:, -1] = torch.roll(ind_y[:, 1], -1, dims=0)  # i+1

    C = get_nonlinear_term_hd2_real(Vn, ind_x, self.dx)
    du2dx2 = viscosity * \
        get_second_derivative_real(Vn, ind_x, ind_y, self.dx, self.dy)
    k1 = du2dx2 + C + F
    return Vn + self.dt * k1

def get_nonlinear_term_hd2_real(Un, ind, h):
    du2dx = get_first_derivative_real(Un * Un, ind, h)
    vecC = -du2dx / 2
    return vecC

def get_first_derivative_real(Un, ind, h):
    ind = ind.long()
    dudx = (Un[ind[:, -1], ...] - Un[ind[:, 0], ...]) * 0.5 / h
    return dudx

def get_second_derivative_real(Un, ind_x, ind_y, h_x, h_y):
    ind_x = ind_x.long()
    ind_y = ind_y.long()
    du2dx2_x = (Un[ind_x[:, 0], ...] - 2 * Un[ind_x[:, 1], ...] +
                Un[ind_x[:, 2], ...]) / (h_x * h_x)
    du2dx2_y = (Un[:, ind_y[:, 0], ...] - 2 * Un[:, ind_y[:, 1], ...] +
                Un[:, ind_y[:, 2], ...]) / (h_y * h_y)
    du2dx2 = du2dx2_x + du2dx2_y
    return du2dx2

class KalmanMPC(nn.Module):
    '''
    KalmanMPC model 
    '''
    def __init__(self, model, Ns, x0, nx_Domain, ny_Domain, xr_t, nt):
        super(KalmanMPC, self).__init__()
        self.model = model
        self.nu = 2
        self.Ns = Ns
        self.nx_Domain = nx_Domain
        self.ny_Domain = ny_Domain
        self.k = 1
        self.mean = torch.zeros((self.nu, self.ny_Domain))
        self.nxbar = self.nx_Domain + self.nu * 2
        self.R = torch.eye(self.nx_Domain, device = device) * 1e-10
        epsilon = 5e-4
        self.sigma_ProcessCov = torch.ones((self.nu, self.nu)) * epsilon
        # Make the matrix symmetric
        self.sigma_ProcessCov = torch.mm(self.sigma_ProcessCov, self.sigma_ProcessCov.T) + epsilon * torch.eye(self.nu)
        self.psi_ProcessCov =  torch.ones((self.ny_Domain, self.ny_Domain)) * epsilon
        # Make the matrix symmetric
        self.psi_ProcessCov = torch.mm(self.psi_ProcessCov, self.psi_ProcessCov.T) + epsilon * torch.eye(self.ny_Domain)

        self.u = torch.zeros((self.nx_Domain, self.ny_Domain, nt), device=device)
        self.v = torch.zeros((self.nx_Domain, self.ny_Domain, nt), device=device)
        self.v[:, :, 0] = x0[0,0, ...]

        self.samples = self.initialize_samples(self.v[:, :, 0])
        self.sample_mean = torch.mean(self.samples, dim=0).to(device)
        self.yk = xr_t
        self.MPC_iter = 0
        self.force = torch.zeros((self.Ns, self.nx_Domain, self.ny_Domain), device=device)

    def initialize_samples(self, v0):

        # Use Matrix Normal for sampling du noise
        delta_uk = torch.tensor(matrix_normal.rvs(mean=self.mean, rowcov = self.sigma_ProcessCov, colcov = self.psi_ProcessCov, size=self.Ns)).to(device)
        # u noise
        uk = torch.tensor(matrix_normal.rvs(mean=self.mean, rowcov = self.sigma_ProcessCov, colcov = self.psi_ProcessCov, size=self.Ns)).to(device)
        uk = uk + delta_uk

        v0_repeated = v0.unsqueeze(0).repeat(self.Ns, 1, 1)
        samples = torch.cat([v0_repeated, uk, delta_uk], dim=1)
        return samples

    def predict(self):
        particles_current = self.samples[:, -self.nxbar:,]
        State_particles_current = particles_current[:, :self.nx_Domain, ]
        Control_particles_current = particles_current[:, self.nx_Domain: self.nx_Domain + self.nu, ]
        self.force[:, 0, :] = Control_particles_current[:, 0, ]
        self.force[:, -1, :] = Control_particles_current[:, 1, ]
        x_pred = self.model(torch.cat([State_particles_current[:, None, ...].to(dtype=torch.float32), self.force[:, None, ...].to(dtype=torch.float32)], dim = 1))[:,0,...]
        w_bar = torch.tensor(matrix_normal.rvs(mean=self.mean, rowcov = self.sigma_ProcessCov, colcov = self.psi_ProcessCov, size=self.Ns)).to(device)
        Control_particles_current += w_bar
        x_pred = torch.cat([x_pred, Control_particles_current, w_bar], dim=1)
        self.samples = torch.cat([self.samples, x_pred], dim=1)
        self.sample_mean = torch.mean(self.samples, dim=0)

    def update(self):
        y_particles = measurement(self)
        y_hat = torch.mean(y_particles, dim=0)
        Py = (1 / (self.Ns - 1)) * torch.sum(torch.bmm((y_particles - y_hat.unsqueeze(0).repeat(self.Ns, 1, 1)), (y_particles - y_hat.unsqueeze(0).repeat(self.Ns, 1, 1)).permute(0, 2, 1)), dim=0) + self.R
        Pxy = (1 / (self.Ns - 1)) * torch.sum(torch.bmm((self.samples - self.sample_mean.unsqueeze(0).repeat(self.Ns, 1, 1)), (y_particles - y_hat.unsqueeze(0).repeat(self.Ns, 1, 1)).permute(0, 2, 1)), dim=0)
        invPy = torch.inverse(Py)
        K = Pxy @ invPy
        self.samples = self.samples + torch.bmm(K.repeat(self.Ns, 1, 1), (self.yk[..., self.MPC_iter] - y_particles))
            
    def warmstart(self, k):
        delta_uk = torch.tensor(matrix_normal.rvs(mean=self.mean, rowcov = self.sigma_ProcessCov, colcov = self.psi_ProcessCov, size=self.Ns)).to(device)
        self.samples = torch.cat([self.v[..., k - 1].unsqueeze(0).repeat(self.Ns, 1, 1),
                                       delta_uk + self.samples[:, self.nxbar + self.nu:self.nxbar + 2 * self.nu, :],
                                       delta_uk], dim=1)
        self.sample_mean = torch.mean(self.samples, dim=0)

# main code
torch.manual_seed(5)
nx_Domain = 32
ny_Domain = 32
uv = torch.load('./data/burgers_100x1x1x32x32.pt')
# initial conidtion
counter = random.randint(0, 49)
x0 = uv[counter, 0:1, 0:1, ...]
x0 = torch.tensor(x0, dtype=torch.float32)
L = 1
T = 5
dx = L / nx_Domain
dy = L / ny_Domain
dt = 0.001
nt = int(T / dt)
t = torch.arange(0, T + dt, dt)

Xref = 1 * torch.ones((nx_Domain, nx_Domain, nt + 50)).to(device)
zero_force = torch.zeros((nx_Domain, ny_Domain), device=device)
# KalmanMPC
Ns = 100
Horizon = 10
time_steps = Horizon
effective_step = list(range(0, time_steps))

fig_save_path = '.../figure_control/'
model_save_path =   './trained_models/checkpoint'

model  = UNet(num_classes=1, in_channels = 1, upscale_factor = 8,
    step = time_steps,
    effective_step = effective_step, dt = dt).to(device)

model, _, _ = load_checkpoint(model, optimizer=None, scheduler=None, save_dir=model_save_path)

EnKS = KalmanMPC(model, Ns, x0, nx_Domain, ny_Domain, Xref, nt)
EnKS.to(device)
sim_time = torch.zeros(nt)
RMSE1 = torch.zeros(nt).to(device)
for k in range(1, nt):
    # warmstart
    if k > 1:
        EnKS.warmstart(k)

    time_iter = 0
    for j in range(1, Horizon):
        start_time = time.time()
        EnKS.MPC_iter = j
        # prediction step of KalmanMPC
        EnKS.predict()
        # update step of KalmanMPC
        EnKS.update()
        time_iter += time.time() - start_time

    sim_time[k - 1] = time_iter
    EnKS.MPC_iter = 0
    print(f"Iteration {k}, Elapsed Time: {time_iter:.6f} seconds")
    a = torch.mean(EnKS.samples[:, EnKS.nx_Domain:EnKS.nu + EnKS.nx_Domain], dim=0)
    # bringing the force space to the space of PDE, so it can be applied to the PDE
    zero_force[0, :] = a[0, ]
    zero_force[ -1, :] = a[1, ]
    EnKS.u[..., k - 1] = zero_force
    # Run system
    EnKS.v[..., k] = RK4_Burgers_solver_real(EnKS, k)
    EnKS.k = k + 1
    if k != 0:
        RMSE1[k-1] = torch.sqrt(torch.mean(torch.abs(EnKS.v[..., k - 1] - Xref[..., k-1]).pow(2)))

plt.figure()
plt.plot(t[:-2],RMSE1[:-1].cpu().numpy())
plt.xlabel('time')
plt.ylabel('RMSE')
plt.show()
