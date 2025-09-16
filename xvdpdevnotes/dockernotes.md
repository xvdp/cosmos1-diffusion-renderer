# running notes

* fixed Dockerfile, original depot failed on some conda shenanigan, either activation or permissions or both, as included in here builds and passes tests
* did pull request... 
https://github.com/nv-tlabs/cosmos1-diffusion-renderer/pull/23/commits/2ed132efc6be9efe8cf05e777e07c2f0578fa0a1

## Build
* Ensure TORCH_EXTENSIONS exists, maybe store per cuda version?
* tested with A6000 48G. 

`checkpoints` are stored in `/mnt/Data/data/weights/NvidiaCosmos/checkpoints` link as a -v with dockerstart.sh

```bash
# build
docker build . -t nvcr.io/xvdp/cosmos-predict1:latest
# run
bash dockerstart.sh # loads volumes for checkpoints and for images
```


## TODO . TEST - Dockerfile using FROM nvcr image with 12.6 and uv, no conda.