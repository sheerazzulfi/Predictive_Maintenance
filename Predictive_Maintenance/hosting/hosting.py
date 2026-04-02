from huggingface_hub import HfApi
import os

repo_ID = "sheerazzulfi/Predictive_Maintenance"
repo_Type = "space"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="Predictive_Maintenance/deployment",     # the local folder containing your files
    repo_id="sheerazzulfi/Predictive_Maintenance",          # the target repo
    repo_type="space",
    path_in_repo="",
)
