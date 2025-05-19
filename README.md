# ReTabAD

## Enviroment setting
### Docker
```sh
docker build -t retabad:1.0.0 .
docker run -itd --rm --name retabad --gpus '"device=4,5,6,7"' -m 375g -v /:/workspace retabad:1.0.0
```