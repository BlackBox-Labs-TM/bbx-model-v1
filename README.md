# bbx-model-v1
Building up on codeGPTSensor from "Distinguishing LLM-generated from Human-written Code by Contrastive Learning"



## Dependencies
We support Ubuntu 22.04 (native or via WSL on Windows) and macOS (CPU/MPS; CUDA not available on macOS).
Required:

Python 3.9 (via Miniconda recommended)

PyTorch 2.5.1 (GPU optional)

CUDA Toolkit 11.8 + recent NVIDIA driver (Linux/WSL only)

Git
### Useful links
Miniconda (recommended): https://www.anaconda.com/docs/getting-started/miniconda/install

PyTorch “Get Started” selector (choose your OS + CUDA 11.8 or CPU): https://pytorch.org/get-started/previous-versions/

CUDA Toolkit downloads (select Version 11.8 for your OS): https://developer.nvidia.com/cuda-11-8-0-download-archive

Ubuntu 22.04 LTS: https://releases.ubuntu.com/22.04/

Windows users: install WSL (Ubuntu 22.04) → https://learn.microsoft.com/windows/wsl/install


## File Structure

The engine directory contains the backbone model made by the team that designed CodeGPTSensor
the pipeline directory contains our retrofitting implementation to work with Github apps + tree sitter ensemble

```
bbx-model-v1/
├── LICENSE
├── README.md
├── data/
│   └── dataset.zip
└── src/
    ├── engine/
    │   ├── README.md
    │   ├── assets/
    │   │   └── framework.png
    │   └── CodeGPTSensor/
    │       ├── model.py
    │       ├── run.py
    │       └── utils/
    │           ├── __init__.py
    │           └── early_stopping.py
    └── pipeline/
```
