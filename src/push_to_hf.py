import argparse

from huggingface_hub import HfApi


def push(local_path, repo_id, private=False, commit_message="Upload model"):
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        commit_message=commit_message,
    )
    print(f"Pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", required=True, help="local model or adapter dir")
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. user/qwen-sft-lora")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    push(args.local, args.repo, private=args.private)
