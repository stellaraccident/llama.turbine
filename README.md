# Llama.turbine

This is an experimental pytorch model that can load from llama.cpp gguf files, run eagerly and compile to Turbine/IREE. We are using it to evaluate different approaches for interfacing with the llama.cpp ecosystem.

## Setup instructions

### Prerequisite
Setup SHARK-Turbine and it's python required environment.

### Prepping model
```sh
python python/turbine_llamacpp/model_downloader.py --hf_model_name="openlm-research/open_llama_3b"
```
by this point you should see a directory named `downloaded_open_llama_3b` in you working directory.
This is typically the `downloaded_<name_of_your_model>`.

### Setup Llama.cpp

```sh
git clone https://github.com/ggerganov/llama.cpp
pip install gguf
```

### Converting HF model to GGUF.

```sh
# Convert HF to GGUF and quantize to q8.
python llama.cpp/convert.py downloaded_open_llama_3b --outfile ggml-model-q8_0.gguf --outtype q8_0
```

### Running Llama.turbine

Running on pytorch
```sh
python python/turbine_llamacpp/model.py  --gguf_path=/path/to/ggml-model-q8_0.gguf
```

Generating MLIR
```sh
python python/turbine_llamacpp/compile.py  --gguf_path=python/turbine_llamacpp/ggml-model-q8_0.gguf
```
