# Magopt

Simulation and optimization of magnetic fields for MRI gradient coil design.

## Features

- **Gradient Coil Models**: Multiple coil geometries including circular, elliptical frustum, and matrix-based designs
- **Field Simulation**: Compute magnetic (B), electric (E), and gradient (dBz/dx, dBz/dy, dBz/dz) fields from coil geometries
- **PNS Modeling**: Peripheral nerve stimulation prediction using charge-based body models
- **ADMM Optimization**: Constrained optimization with support for energy minimization, field constraints, and PNS limits
- **Surface Optimized Designs**: Unrolled ADMM combined with a torch ADAM optimizer to jointly solve for optimal gradient coil surfaces and winding patterns.

## Installation

### Using Conda (recommended)

```bash
git clone https://github.com/danielabrahamgit/magopt
cd magopt
conda env create -f environment.yml
conda activate magopt
pip install -e .
```

### Using pip

```bash
git clone https://github.com/danielabrahamgit/magopt
cd magopt
pip install -e .
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- NumPy, SciPy, Matplotlib, tqdm, einops

