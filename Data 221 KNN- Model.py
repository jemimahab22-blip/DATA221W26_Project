# import statements
import kagglehub

# Download the latest version of the dateset
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)