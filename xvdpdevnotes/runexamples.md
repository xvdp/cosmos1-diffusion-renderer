From app/ cosmos_predict1 root
# Cosmos
```bash
cd /home/z/work24/gits/Diffusion/cosmos1-diffusion-renderer
# build
docker build . -t nvcr.io/xvdp/cosmos-predict1:latest
# run
docker run --gpus device=1 --cpuset-cpus=0-10 --network=host  -it --rm --shm-size 20g -v `pwd`:/app --workdir /app -e TORCH_EXTENSIONS_DIR=/app/tmp nvcr.io/xvdp/cosmos-predict1:latest
```



## Image inputs
* Stable diffusion uses H/8 W/8 setup.
* cosmos by default does 1280 704
* `imij,resize_fit(ls()[4], size=(704,1280), expand=True, out_name="1280_704/dogs.png", color=0)`


# README EXAMPLES
## DELIGHT
PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
--checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
--dataset_path=asset/examples/image_examples/ --num_video_frames 1 \
--group_mode webdataset --video_save_folder=asset/example_results/image_delighting/ \
--save_video=False

# RELIGHT
    --dataset_path=asset/example_results/image_delighting/
    --video_save_folder=asset/example_results/image_relighting/
    --envlight_ind 0 1 2 3 --use_custom_envmap=True
    ENV_LIGHT_PATH_LIST = [ "asset/examples/hdri_examples/sunny_vondelpark_2k.hdr",... # hardcoded
PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=asset/example_results/image_delighting/gbuffer_frames --num_video_frames 1 \
    --envlight_ind 0 1 2 3 --use_custom_envmap=True --video_save_folder=asset/example_results/image_relighting/

# RANDOM RELIGHT from SEED
    --use_custom_envmap=False # --envlight_ind 0 1 2 3 is ignored
```
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=asset/example_results/image_delighting/gbuffer_frames --num_video_frames 1 \
    --envlight_ind 0 1 2 3 --use_custom_envmap=False \
    --video_save_folder=asset/example_results/image_relighting_random/
```
# MY EXAMPLES
scale or proportion image to 1280x704

    --dataset_path=asset/mytests/images/
    --video_save_folder=asset/mytests/image_delighting

# DELIGHT
```
PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B --dataset_path=asset/mytests/images/ --num_video_frames 1 --group_mode webdataset --video_save_folder=asset/mytests/image_delighting/ --save_video=False
```


    --dataset_path=asset/mytests/image_delighting/gbuffer_frames
    --video_save_folder=asset/mytests/

# RELIGHT
```
PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B --dataset_path=asset/mytests/image_delighting/gbuffer_frames --num_video_frames 1 --envlight_ind 0 1 2 3 --use_custom_envmap=True --video_save_folder=asset/mytests/image_relighting/
```

```
# OTHER OPTIONS cosmos_predict1/diffusion/inference/inference_forward_renderer.py
# from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.utils_env_proj import process_environment_map
#
# --rotate_light bool, rotate fixed amount per frame
# --use_fixed_frame_ind     bool [false]
# --fixed_frame_ind         int [0] `
# rots = np.linspace(0, 2 * np.pi, num_frames) if rotate_envlight else [0] * num_frames
```

```
docker run --gpus device=1 --cpuset-cpus=0-10 --network=host  -it --rm --shm-size 30g -v `pwd`:/app -v /home/z/work24/gits/Diffusion/mysamples/originals/1280_704:/app/image_1280_704 --workdir /app -e TORCH_EXTENSIONS_DIR=/app/tmp nvidia/xvdp/cosmos-predict1:latest
```

## Delight Again
```
IMGS="image_1280_704"
OUT="asset/mytests/delight2"
PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B --dataset_path=${IMGS} --num_video_frames 1 --group_mode webdataset --video_save_folder=${OUT} --save_video=False
```
## Relight again
```
PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B --dataset_path=${OUT}/gbuffer_frames --num_video_frames 1 --envlight_ind 0 1 2 3 --use_custom_envmap=True --video_save_folder=${OUT}/relit/
```
## pick one env and render a video rotation

PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=asset/example_results/video_delighting/gbuffer_frames --num_video_frames 57 \
    --envlight_ind 0 1 2 3 --use_custom_envmap=True \
    --video_save_folder=asset/example_results/video_relighting_rotation/ --rotate_light=True --use_fixed_frame_ind=True

