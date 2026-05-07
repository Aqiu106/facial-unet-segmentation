"""
download_helen_dataset.py
下载 HELEN 数据集。
"""
import kagglehub


def download_helen_dataset():
    """下载 HELEN 数据集并返回本地路径。"""
    path = kagglehub.dataset_download("abtahimajeed/helen-dataset")
    print("Path to dataset files:", path)
    return path


if __name__ == "__main__":
    download_helen_dataset()
