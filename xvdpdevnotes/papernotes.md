
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

## Related Work
Zeng Deschaintre Georgiev Geoffroy 2024 Adobe [RGB↔X: Image decomposition and synthesis using material and lighting-aware diffusion models](https://arxiv.org/html/2405.00666v1)
* github  https://github.com/zheng95z/rgbx finetuned Rombach et al 2022, on interior render datasets. 
* Diffusion Renderer is much more robust.
### RGBX Qs
* could failure points (images that come out of film log that have been applied a LUT ) occur because of lack of training data? or augmentation?
* conditioning on prompt does not do much, why? https://github.com/xvdp/rgbx/blob/dev/NOTES.md

RGBX uses v-prediction stated as $$\mathbf{v}_t^{RGB\to X} = \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1 - \bar{\alpha}_t}\mathbf{z}_0^X$$ instead of prediciton of $\epsilon$

* $\mathbf{v}$ is velocity? or tangent? of the rotation $\epsilon \to \mathbf{x}$

From Salimans and Ho 2022 [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) https://github.com/google-research/google-research/tree/master/diffusion_distillation, 
```python
def predict_x_from_v(*, z, v, logsnr):
    logsnr = utils.broadcast_from_left(logsnr, z.shape)
    alpha_t = jnp.sqrt(jax.nn.sigmoid(logsnr))
    sigma_t = jnp.sqrt(jax.nn.sigmoid(-logsnr))
    return alpha_t * z - sigma_t * v
``` 
x from v https://github.com/google-research/google-research/blob/master/diffusion_distillation/diffusion_distillation/dpm.py  line 110.

### derivation of v: Salimans et al, Section D. page 14. 
$$\text{where sampling data, } \mathbf{x},  \text{ from  noise, } \epsilon, \quad \mathbf{x} = f(\mathbf{z}; \theta)\\ \mathbf{v}_\phi \equiv \frac{d\mathbf{z}_\phi}{d\phi} = \frac{d \cos(\phi)}{d \phi}\mathbf{x} + \frac{d \sin(\phi)}{d\phi}\epsilon \\
...\\
\mathbf{x} = \cos(\phi)\mathbf{z} - \sin(\phi)\mathbf{v}_\phi, \qquad \epsilon = \sin(\phi)\mathbf{z}_\phi + \cos(\phi)\mathbf{v}_\phi\\
\text{defining predicted velocity }\qquad\mathbf{\hat{v}}_\phi(\mathbf{z}_\phi) \equiv \cos(\phi)\hat{\epsilon}_\phi (\mathbf{z}_\phi) - \sin(\phi)\mathbf{\hat{x}}_\phi(\mathbf{z}_\phi)$$
DDIM rotates z, thats neat. 
* The cool thing about that Progressive Distillation is the minimization of denoising steps. 
https://github.com/xvdp/google-research/blob/fixflaxversion/diffusion_distillation/diffusion_distillation.ipynb
```python
# ommitting jax shenaningans
while state.num_sample_steps >= 4: # goes from maybe 8192 to 4 diffusion steps 
    # train the student against the teacher model recursively 8192->4096, 4096->2048...8->4 
    for step in range(10): # steps_per_distill_iter
        state, metrics = train_step(state, next_batch)
    # student becomes new teacher
    model.teacher_state = jax.device_get(state.replace(tx=None)) # tx is optax. optimizer in flax.training.train_state.TrainState
    # new student with 1/2 sample steps, copying ema parameters, 
```


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
* denoising function $\mathbf{f}_\theta$, ***trained with  score matching objective*** 

### Q. Score matching objective ?
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
## Conditioning 
* concatenating condition channels with $\mathbf{z}_\tau$ : **TODO: develop, sketch to Dif.**
    * Blattman, Stable Video Diffusion .. 
    * Zeng, RGBX <- evaluated, less robust. >
    * Ke et al, 2024. [Repurposing diffusion-based image generators for monocular depth estimation](https://openaccess.thecvf.com/content/CVPR2024/papers/Ke_Repurposing_Diffusion-Based_Image_Generators_for_Monocular_Depth_Estimation_CVPR_2024_paper.pdf) code https://github.com/prs-eth/marigold
    * Kocsis et al 2023, [Intrinsic Image Diffusion for Indoor Single-view Material Estimation](https://arxiv.org/abs/2312.12274) code https://github.com/Peter-Kocsis/IntrinsicImageDiffusion

* injecting condition thru cross attention:  **TODO: develop, add latent cmd to sketch/ layer effects like StyleGan**
    * Blattman, Stable Video Diffusion ..
    * Rombach et al [High-resolution image synthesis with latent diffusion model](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) code https://github.com/CompVis/latent-diffusion


## Relighting
* Kocsis Intrinsic.. 
* Liang ECCV 24 Photorealistic object insertion with diffusion-guided inverse rendering.
* Phongthawee 2023 DiffusionLight: light probes for free by painting a chrome ball. https://github.com/DiffusionLight/DiffusionLight

### 3d from multiview thru inverse 
* Boss et al 2021 NeRD: neural reflectance decomposition from image collection ICCV
* W Chen Nvidia,  2021,  DIB-R++: Learning to predict lighting and material with a hybrid differentiable renderer. In NeurIPS, (Fidler)
* Hasselgren Shape, light, and material decomposition from images using Monte Carlo rendering and denoising
* Jiang 2024 Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces.
* R. Liang 2021 Envidr: Implicit differentiable renderer with neural environment lighting.
* Zh. Liang 2023 Gs-ir: 3d gaussian splatting for inverse rendering
* Munkberng 2021 Extracting triangular 3D models, materials, and lighting from images (Fidler)
* Rudnev MPI NeRF for outdoor scene relighting (Theobalt)
* Shi 2023, Gir: 3d gaussian inverse rendering for relightable scene factorization.
* Wang 2023 Nvidia Neural fields meet explicit geometric representations for inverse rendering of urban scenes. (Fidler)
* Xi 2024 Intrinsic Anything: learning diffusion priors for inverse rendering under unknown illumination. 
* K,Zhang 2021 PhySG: Inverse rendering with spherical Gaussians for physics-based material editing and relighting (Snavely)
* K. Zhang 2022 IRON: inverse rendering by optimizing neural SDFs and ma-terials from photometric images (Snavely)
* X Zhang 21 NerFactor: neural factorization of shape and reflectance underan unknown illumination (Debcefec)'
## Relighting with diffusion
* 2024 Jin Neural gaffer: Relighting any object via diffusion. (Snavely)
* 2024 Kocsis, LightIt: illuminationmodeling and control for diffusion models
* 2024 Poirer-Ginter A Diffusion Approach to Radiance Field Relighting using Multi-Illumination Synthesis.
* 2024 DiLightNet: fine-grained lighting control for diffusion-based image generation

## Finetuning diffusion
* Ke 2024  Repurposing diffusion-based image generators for monocular depth estimation.
* Kocsis Intrinsic...
* Zheng RGBX

# 4 Method.
Two models
1. neural inverse renderign
2. neural forward rendering

## Lighting :
* apply Reighhard tonemapping to to betweeh HDR and LDR -1,1  / like neural gaffer 
```
Video -> Vae Encoder -> Diffusion net (+ domain embeddings color, normal ,depth metallic) -> Decoder Gbuffers

    Env light -> Encoder -> Env Encoder net -> condition Diffusion Unet
            LORA-> condition Diffusion Unet
Gbuffers -> encoder ->  Diffusion Unet -> decoder
```

* Domain embedding. -> pixel aligned buffers by concat
* Env maps -> , and repurpose text/image CLIP features for lighting condition  -> generalize coditional signals multi res feature maps? ok
