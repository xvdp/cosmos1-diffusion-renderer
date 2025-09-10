
# [DIFFUSION RENDERER: Neural Inverse and Forward Rendering with Video Diffusion Models](https://arxiv.org/abs/2501.18590)
/home/z/Documents/Generative/Diffusion

Paper Notes
* fine tune a video diffusion model for `inverse render(images) -> G-buffers` from real vid: image editing interface
* `fw render (G-buffers) -> photoreal images` w/o explicit light transport sim. 
1. **train** `inverse render video diffusionl model (synthetic data)` -> generalizes well and allows auto-label real-world videos
2. **co-train** `fw render_model(synthetic, auto-labeled real-world data)`. 

-> relighting, material editing, object insertion.

conditioning: input geometry, material buffers, hdr env maps
* Q. can I extract hdrs from images?

training: including data with noisy conditions to ensure robustness.

## Related
Zeng Deschaintre Georgiev Geoffroy 2024 Adobe [RGB↔X: Image decomposition and synthesis using material and lighting-aware diffusion models](https://arxiv.org/html/2405.00666v1)
* github  https://github.com/zheng95z/rgbx also uses diffusion models

## Notes on Deferred rendering and PBR
PBR Params:  Albedo, Normal, Roughness, Metalicity, Depth, AO, Emission, ClearCoat
1. G-Buffer Pass: Render to surface attributes per pixel to PBR parameters
2. Lighting Pass: For each light, per pixel, read G-Buffers and output color

PBR 
* $ L_i(\mathbf{p},\mathbf{\omega}_i)$ incoming radiance
* $L_o(\mathbf{p}, \mathbf{\omega}_o)$ outgoing radiance at point $\mathbf{p}$ with direction $\mathbf{\omega}_o$
* $\int_\Omega$ integral over hemisphere
* $ f_r(\mathbf{p},\mathbf{\omega}_o, \mathbf{\omega}_i)$ BRDF
* $| \mathbf{n}\cdot \mathbf{\omega}_i|$ cos angle incoming normal
$$L_o(\mathbf{p}, \mathbf{\omega}_o) =\int_\Omega f_r(\mathbf{p},\mathbf{\omega}_o, \mathbf{\omega}_i) L_i(\mathbf{p},\mathbf{\omega}_i) | \mathbf{n}\cdot \mathbf{\omega}_i| d \mathbf{\omega}_i $$

## Video diffusion models (VDMs)
* Sohl-Dickstein, Weiss, Maheswaranathan, and Ganguli, 2015, [Deep unsupervised learning using nonequilibrium thermodynamics](https://arxiv.org/abs/1503.03585)
* Ho, Jain, and Pieter Abbeel, 2020 [Denoising diffusion probabilistic models](https://arxiv.org/abs/2006.11239)
* Dhariwal and Nichol, 2021 [Diffusion models beat GANs on image synthesis](https://arxiv.org/abs/2105.05233)
Diffusion description in paper: 
* Input RGB video: $\mathbf{I} \in \mathbb{R}^{F\times H\times W\times3}$, compressed by VAE  $\rightarrow \mathbf{z} = \mathcal{E}(\mathbf{I}) \in \mathbb{R}^{F'\times h\times w\times C}$
* output RGB video  $\mathbf{\hat{I}}$ from decoder $\mathcal{D}$

uses: **Andreas Blattmann et al. 2023 [Stable video diffusion: Scaling latent video diffusion models to large datasets](https://arxiv.org/abs/2311.15127)**
* shape($\mathbf{z}$): $F`, \frac {H}{8}, \frac {W}{8}, 4$ 
* Noise frames  $\mathbf{z}_\tau = \alpha_\tau \mathbf{z}_0 +\sigma_\tau $ follows $ \alpha_\tau$ and $\sigma_\tau$ from 
* Karras et al, 2022 [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) https://github.com/NVlabs/edm
* denoising funciton $\mathbf{f}_\theta$, ***trained with  score matching objective*** 

is it eq.3 and 68 of Elucidating? 

$$\nabla_x \log p(x;\sigma) = \frac{D(x;\sigma) - x}{\sigma^2}$$

 or EDMLoss from https://github.com/NVlabs/edm/training/loss.py
$$
x \sim N(I,0) ,\quad s_p = 1.2,\quad\mu_p = -1.2, \sigma_{data} = 0.5\\
\sigma = e^{x \cdot s_p +\mu_p } \\
w = \frac{\sigma ^2 + \sigma_{data}^2}{(\sigma \cdot \sigma_{data})^2}\\
n \sim N(I,0) \cdot \sigma \\
\mathcal{L} = w (f(y + n, \sigma ) - y)^2
$$
Conditioning 
* concatenating condition channels with $\mathbf{z}_\tau$
    * Blattman, Stable Video Diffusion .. 
    * Zeng, RGBX
    * Ke et al, 2024. [Repurposing diffusion-based image generators for monocular depth estimation](https://openaccess.thecvf.com/content/CVPR2024/papers/Ke_Repurposing_Diffusion-Based_Image_Generators_for_Monocular_Depth_Estimation_CVPR_2024_paper.pdf) code https://github.com/prs-eth/marigold
    * Kocsis et al 2023, [Intrinsic Image Diffusion for Indoor Single-view Material Estimation](https://arxiv.org/abs/2312.12274) code https://github.com/Peter-Kocsis/IntrinsicImageDiffusion

* injecting condition thru cross attention: 
    * Blattman, Stable Video Diffusion ..
    * Rombach et al [High-resolution image synthesis with latent diffusion model](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) code https://github.com/CompVis/latent-diffusion