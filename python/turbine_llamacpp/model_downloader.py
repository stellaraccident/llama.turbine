from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_auth_token",
    type=str,
    help="The Hugging Face auth token, required",
    default="",
)
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="openlm-research/open_llama_3b",
)


def donwload_hf_model(hf_auth_token, hf_model_name):
    auth_token = hf_auth_token if len(hf_auth_token) != 0 else None
    model_name = hf_model_name.split("/")[-1]
    snapshot_download(
        repo_id=hf_model_name,
        local_dir="downloaded_" + model_name,
        local_dir_use_symlinks=False,
        revision="main",
        token=auth_token,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    donwload_hf_model(args.hf_auth_token, args.hf_model_name)
