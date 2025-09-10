
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
* denoising funciton $\mathbf{f}_\theta$, trained with  score matching objective (eq.3 and 68 of Elucidating) $\nabla_x \log p(x;\sigma) = (D(x;\sigma) - x)/\sigma^2$

 or 
 ``` python
@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss
```




