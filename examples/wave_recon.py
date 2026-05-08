import torch

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_sim.phantoms import shepp_logan
from mr_sim.coil_maps import surface_coil_maps
from mr_recon.utils import gen_grd, normalize
from mr_recon.linops import sense_linop, batching_params
from mr_recon.fourier import sigpy_nufft
from mr_recon.recons import CG_SENSE_recon
from mr_recon.imperfections.field import alpha_segementation, phi_alpha_svd

# Params
R = 8
C = 20
gamma = 42.58e6 # MHz/T
sigma = 1e-2
max_iter = 40
im_size = (220, 220)
torch_dev = torch.device(3)

# Image phantom and coil maps
img_gt = shepp_logan(torch_dev).img(im_size).mT
mps = surface_coil_maps(C, im_size, r_coil=1/8, 
                        espirit_crp=0.95,
                        img=img_gt, torch_dev=torch_dev)

# Load gradient field patterns
paths = [
    # './designs/bfield_linpcnt=None_dsv=17cm.pt',
    # './designs/bfield_linpcnt=0.1_dsv=17cm.pt',
    # './designs/bfield_linpcnt=None_dsv=8cm.pt',
    # './designs/bfield_linpcnt=0.1_dsv=8cm.pt',
    './designs/bfield_linpcnt=0.1_dsv=20cm.pt',
    './designs/bfield_linpcnt=None_dsv=20cm.pt',
]
bfields = [torch.load(path, map_location=torch_dev, weights_only=True) for path in paths]
bfields = torch.stack(bfields, dim=0).mT
assert bfields.shape == (len(paths), *im_size)
bfields *= (img_gt.abs() > 0).float()

# Current waveform
nread = 5_000
I = 50 # A
dt = 2e-6
f0 = 20e3
ts = torch.arange(nread, device=torch_dev) * dt
current_int = I * torch.sin(2 * torch.pi * f0 * ts) / (2 * torch.pi * f0)

# Build trajectory
trj = gen_grd((nread, im_size[1]), im_size).to(torch_dev)
trj = trj[:, ::R, :]

# Simulate regular data
nufft = sigpy_nufft(im_size, width=3)
nufft.beta = nufft.optimal_beta(torch_dev=torch_dev)
bparams = batching_params(C)
A = sense_linop(trj, mps,
                nufft=nufft, 
                bparams=bparams)
ksp = A(img_gt)
ksp += torch.randn_like(ksp) * sigma
img = CG_SENSE_recon(A, ksp,
                        max_iter=max_iter,
                        max_eigen=1.0).cpu()
img = normalize(img, img_gt.cpu(), mag=True, ofs=False)


# Simulate each bfield pattern
for k in range(len(bfields)):
    # Get bfield
    bfield = bfields[k]
    
    # Build forward models
    alphas = current_int[None, :, None]
    phis = bfield[None, :, :] * gamma
    bs, hs, _ = alpha_segementation(phis, alphas,
                                   L=30, interp_type='lstsq', use_type3=True)
    A_wave = sense_linop(trj, mps,
                         nufft=nufft, 
                         spatial_funcs=bs,
                         temporal_funcs=hs,
                         bparams=bparams)
    
    # Simulate data with noise 
    ksp_wave = A_wave(img_gt)
    ksp_wave += torch.randn_like(ksp_wave) * sigma
    
    # Recon 
    img_wave = CG_SENSE_recon(A_wave, ksp_wave,
                              max_iter=max_iter,
                              max_eigen=1.0).cpu()
    img_wave = normalize(img_wave, img_gt.cpu(), mag=True, ofs=False)
    
    # Show
    vmax = img_gt.abs().median() + 3 * img_gt.abs().std()
    imgs_all = [img_gt.cpu(), img.cpu(), img_wave.cpu()]
    titles = ['Ground Truth', 'Regular Recon', 'Wave Recon']
    plt.figure(figsize=(14, 7))
    for i in range(len(imgs_all)):
        plt.subplot(2, 2, i+1)
        plt.title(titles[i])
        plt.imshow(imgs_all[i].cpu().abs(), cmap='gray', vmin=0, vmax=vmax)
        plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.title('Gradient Field (mT/m/A)')
    gfield = bfield.diff(dim=1) / (1e-3)
    gfield = torch.cat([gfield[:, -1:], gfield], dim=1)
    plt.imshow(gfield.cpu()*1e3, cmap='RdBu_r', vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
plt.show()
quit()
