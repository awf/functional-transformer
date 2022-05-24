

Setup

$ git submodule add https://github.com/awf/awf-jaxutils jaxutils
$ git submodule add https://github.com/LucienShui/timer

Running

$ export JAX_PLATFORM_NAME=gpu # or cpu
$ export JAX_LOG_COMPILES=1 # or 0
$ export XLA_FLAGS=--xla_dump_to=./xla-dumps/  # Also dumps jaxprs to this folder
