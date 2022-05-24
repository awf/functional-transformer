
Running
```sh
$ export JAX_PLATFORM_NAME=gpu # or cpu
$ export JAX_LOG_COMPILES=1 # or 0
$ export XLA_FLAGS=--xla_dump_to=./xla-dumps/  # Also dumps jaxprs to this folder
$ python main.py -help
$ python main.py -layers 3 -dmodel 512 -heads 8 -dk 64 -dff 2048 
```

Results at https://wandb.ai/awfidius/pure-transformer
