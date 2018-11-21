import os
import tensorflow
from functools import partial
import PIL.Image
import urllib.request
import zipfile
import numpy as numpy

def main():
    #download the pre-trained neural network
    url = "https://storage.googleapis.com/downloads.tensorflow.org/models/inception5h.zip"
    data_dir = "../"
    model_name = os.path.split(url)[-1]  #to retrive the model name which
                                         #return inception5h.zip

    local_zip_file = os.path.join(data_dir, model_name)


    #download the file
    if not os.path.exists(local_zip_file):
        model_url = urllib.request.urlopen(url) #download it
        with open(local_zip_file, "wb"): #write in binary mode so we can write the download data to it

        with zipfile.ZipFile(local_zip_file, "r") as zip_ref:
            zip_ref.extractall(data_dir)


main()