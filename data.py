from monai.apps import download_and_extract
import os



    

def retrieve_data_from_link():
    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    md5 = "410d4a301da4e5b2f6f86ec3ddba524e"


    root_dir = "/mnt/Enterprise2/shirshak/"
    compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
    data_dir = os.path.join(root_dir, "Task09_Spleen")

    if not os.path.exists(data_dir):
        download_and_extract(resource,compressed_file, root_dir, md5)


