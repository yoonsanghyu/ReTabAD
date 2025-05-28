# ReTabAD

## Environment setting

### Data Directory Configuration
Create a `.env` file in the project root and add the following line to specify your data directory:

```
# .env
DATA_DIR=/PATH/TO/YOUR/DATA/DIR
```

Replace the path with the location of your data if different.

### Docker
```sh
docker build -t retabad:1.0.0 .
docker run -itd --rm --name retabad --gpus '"device=4,5,6,7"' -m 375g -v /:/workspace retabad:1.0.0
```

