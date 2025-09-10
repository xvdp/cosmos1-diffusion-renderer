# running notes

* fixed Dockerfile, original depot failed on some conda shenanigan, either activation or permissions or both, as included in here builds and passes tests
* did pull request... 
https://github.com/nv-tlabs/cosmos1-diffusion-renderer/pull/23/commits/2ed132efc6be9efe8cf05e777e07c2f0578fa0a1


```bash
# build
docker build . -t nvcr.io/xvdp/cosmos-predict1:latest
# run
docker run --gpus device=1 --cpuset-cpus=0-10 --network=host  -it --rm --shm-size 10g -v `pwd`:/app --workdir /app -e TORCH_EXTENSIONS_DIR=/app/tmp nvcr.io/xvdp/cosmos-predict1:latest
```

## TEST - Dockerfile using FROM nvcr image with 12.6 and uv, no conda.
