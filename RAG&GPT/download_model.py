from huggingface_hub import snapshot_download
import os

def download_model(repo_id, local_dir):
    cache_dir = os.path.join(local_dir, "cache")
    try:
        snapshot_download(cache_dir=cache_dir,
                          local_dir=local_dir,
                          repo_id=repo_id,
                          local_dir_use_symlinks=False,
                          resume_download=True,
                          allow_patterns=["*.model", "*.json", "*.bin", "*.py", "*.md", "*.txt"],
                          ignore_patterns=["*.safetensors", "*.msgpack", "*.h5", "*.ot"])
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    # 设置本地目录
    local_dir = './models'
    # 设置模型的repo_id
    repo_id = "sentence-transformers/all-mpnet-base-v2"
    # 下载模型
    download_model(repo_id, local_dir)