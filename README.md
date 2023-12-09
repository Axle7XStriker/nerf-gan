# Nerfacto + GAN Research

NeRF research into incorporating GANs in Nerfacto using Nerfstudio.



### Installation

1. Install [Nerfstudio](https://docs.nerf.studio/quickstart/installation.html)
2. `cd nerf-gan`
3. `pip install -e .`
4. `export NERFSTUDIO_METHOD_CONFIGS="ganerf=ganerf.ganerf_config:ganerf_method"`



### Training our method

` ns-train ganerf --data [DATA] `
