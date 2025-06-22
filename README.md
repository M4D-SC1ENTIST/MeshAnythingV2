<p align="center">
  <h3 align="center"><strong>MeshAnything V2:<br> Artist-Created Mesh Generation<br>With Adjacent Mesh Tokenization</strong></h3>

<p align="center">
    <a href="https://buaacyw.github.io/">Yiwen Chen</a><sup>1,2</sup>,
    <a href="https://yikaiw.github.io/">Yikai Wang</a><sup>3</sup><span class="note">*</span>,
    <a href="https://github.com/Luo-Yihao">Yihao Luo</a><sup>4</sup>,
    <a href="https://thuwzy.github.io/">Zhengyi Wang</a><sup>2,3</sup>,
    <br>
    <a href="https://scholar.google.com/citations?user=2pbka1gAAAAJ&hl=en">Zilong Chen</a><sup>2,3</sup>,
    <a href="https://ml.cs.tsinghua.edu.cn/~jun/index.shtml">Jun Zhu</a><sup>2,3</sup>,
    <a href="https://icoz69.github.io/">Chi Zhang</a><sup>5</sup><span class="note">*</span>,
    <a href="https://guosheng.github.io/">Guosheng Lin</a><sup>1</sup><span class="note">*</span>
    <br>
    <sup>*</sup>Corresponding authors.
    <br>
    <sup>1</sup>S-Lab, Nanyang Technological University,
    <sup>2</sup>Shengshu,
    <br>
    <sup>3</sup>Tsinghua University,
    <sup>4</sup>Imperial College London,
    <sup>5</sup>Westlake University
</p>



<div align="center">

<a href='https://arxiv.org/abs/2408.02555'><img src='https://img.shields.io/badge/arXiv-2408.02555-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://buaacyw.github.io/meshanything-v2/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/Yiwen-ntu/MeshAnythingV2/tree/main"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/spaces/Yiwen-ntu/MeshAnythingV2"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-HF-orange"></a>

</div>


<p align="center">
    <img src="demo/demo_video.gif" alt="Demo GIF" width="512px" />
</p>


## Contents
- [Contents](#contents)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Important Notes](#important-notes)
- [Acknowledgement](#acknowledgement)
- [BibTeX](#bibtex)

## Unified CUDA/MPS Support

This unified version of MeshAnythingV2 works on both **NVIDIA GPUs (CUDA)** and **Apple Silicon (MPS)** devices automatically. The system will detect your hardware and configure itself appropriately.

### Device Detection Features

- **Automatic Device Detection**: Automatically detects and uses the best available device (CUDA → MPS → CPU)
- **Cross-Platform Compatibility**: Works seamlessly on both NVIDIA and Apple Silicon devices
- **Optimized Performance**: Uses device-specific optimizations (flash attention for CUDA, eager attention for MPS)
- **Smart Precision Handling**: Automatically uses fp16 for CUDA and fp32 for MPS for optimal compatibility

### Supported Devices

- **NVIDIA GPUs**: Full CUDA support with flash attention and fp16 precision
- **Apple Silicon (M1/M2/M3)**: MPS backend with optimized settings for Apple GPUs
- **CPU Fallback**: Works on any system as a fallback

### Installation

```bash
# Install PyTorch with appropriate backend support
# For CUDA (NVIDIA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon (MPS):
pip install torch torchvision

# Install other requirements
pip install -r requirements.txt
```

### Usage

The unified codebase automatically detects your device:

```python
from MeshAnything.models.meshanything_v2 import MeshAnythingV2

# Device detection happens automatically
model = MeshAnythingV2.from_pretrained("Yiwen-ntu/meshanythingv2")
```

### Device-Specific Optimizations

#### NVIDIA GPU (CUDA)
- Flash Attention 2 for improved performance
- Mixed precision training (fp16)
- Better Transformer optimizations

#### Apple Silicon (MPS)
- Eager attention implementation (MPS compatible)
- Full precision (fp32) for stability
- Optimized tensor operations for Apple GPUs

#### CPU
- Standard PyTorch operations
- Full precision for compatibility

### Running the Demo

```bash
# Run the Gradio interface
python app.py
```

The app will automatically:
1. Detect your device (CUDA/MPS/CPU)
2. Load the model with appropriate settings
3. Configure the interface for optimal performance

### Key Improvements in Unified Version

1. **No Manual Device Configuration**: Automatic device detection removes the need for manual setup
2. **Cross-Platform Compatibility**: Single codebase works on all supported hardware
3. **Optimized for Each Platform**: Device-specific optimizations for best performance
4. **Seamless Migration**: Existing workflows continue to work without changes

### Technical Details

The unified version includes:

- `utils.py`: Device detection and compatibility utilities
- Updated model classes with conditional device handling
- Automatic precision and attention mechanism selection
- Cross-platform tensor type management

### Performance Notes

- **CUDA**: Fastest performance with flash attention and fp16
- **MPS**: Good performance on Apple Silicon with optimized fp32 operations
- **CPU**: Functional but slower, suitable for development and testing

### Original README Content

**MeshAnything** converts any 3D representation into meshes created by human artists, i.e., Artist-Created Meshes (AMs).

**TL;DR:** MeshAnything is **a trained model** that can transform any 3D representation (e.g., point clouds, multi-view images, NeRF, 3D Gaussian, textual description) into high-quality Artist-Created Meshes.

[![Demo](https://img.shields.io/badge/Demo-Gradio-blue)](https://huggingface.co/spaces/Yiwen-ntu/MeshAnythingV2)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://buaacyw.github.io/mesh-anything/)
[![Paper](https://img.shields.io/badge/arXiv-2406.10163-red)](https://arxiv.org/abs/2406.10163)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HuggingFace-orange)](https://huggingface.co/Yiwen-ntu/MeshAnything)

## Environment

Our environment has been tested on Ubuntu 22, with both CUDA 11.8 and Apple Silicon MPS support.

```bash
conda create -n MeshAnything python=3.10
conda activate MeshAnything
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

For training, we recommend using [xformers](https://github.com/facebookresearch/xformers) for its memory efficiency and speed.

```bash
pip install -r training_requirement.txt
```

We adopt Flash Attention for training acceleration on CUDA devices, and automatically fall back to eager attention on MPS devices.

## Quick Start

Download model weights and run the demo:

```bash
# Direct download
python app.py

# Or download via Hugging Face CLI
huggingface-cli download Yiwen-ntu/MeshAnything --local-dir ./MeshAnything/

# Or use git lfs
git lfs install
git clone https://huggingface.co/Yiwen-ntu/MeshAnything ./MeshAnything/
```

## Model Inference

```python
import torch
from MeshAnything.models.meshanything_v2 import MeshAnythingV2

# Automatic device detection and optimization
model = MeshAnythingV2.from_pretrained("Yiwen-ntu/meshanythingv2")

# Your point cloud
# pc_normal: A tensor with shape [batch_size, n_points, 6]
# Here, 6 corresponds to [x, y, z, nx, ny, nz]
# where (x,y,z) is the coordinate and (nx,ny,nz) is the normal vector.

# Generate mesh
vertices = model(pc_normal)  # [batch_size, n_faces, 3, 3]
```

## Training

See [MeshAnything/training_data_prepare.md](MeshAnything/training_data_prepare.md) for data preparation details.

```bash
cd meshanything_train
accelerate launch --config_file accelerate_config.yaml \
  --num_processes 8 \
  train.py \
  --config_path ./train_config.yaml \
  --project_name MeshAnything
```

The training supports both CUDA and MPS devices with automatic optimization.

## Dataset

See [MeshAnything/dataset.md](MeshAnything/dataset.md) for dataset details.

## Citation

```bibtex
@article{chen2024meshanything,
  title={MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers},
  author={Chen, Yiwen and Wang, Tong and Yang, Yikun and Zhang, Guo and Liu, Yawei and Tang, Hao and Zhao, Hengshuang and Di, Xihui and Wang, Ceyuan},
  journal={arXiv preprint arXiv:2406.10163},
  year={2024}
}
```

## License

S-Lab-1.0 LICENSE. Please refer to the [LICENSE file](LICENSE.txt) for details.

## Contact

If you have any questions, feel free to open a discussion or contact us at yiwen002@e.ntu.edu.sg.
