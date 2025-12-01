import os
import urllib

import gdown
import zipfile

class BostonFiles:
    def __init__(self, source_id_img, source_id_msk):

        self.source_url_img = f"https://drive.google.com/uc?id={source_id_img}"
        self.source_url_msk = f"https://drive.google.com/uc?id={source_id_msk}"

        self.img_path = 'images'
        self.msk_path = 'masks'


    def download (self):
        if not os.path.isfile(self.img_path+'.zip'):
            print('Downloading images to ', self.img_path+'.zip')
            gdown.download(self.source_url_img, self.img_path+'.zip', quiet=False)
        if not os.path.isfile(self.msk_path+'.zip'):
            print('Downloading masks to ', self.msk_path+'.zip')
            gdown.download(self.source_url_msk, self.msk_path+'.zip', quiet=False)


    def unzip (self):
        if not os.path.isdir(self.img_path):
            print('Unzipping images to ', self.img_path)
            with zipfile.ZipFile(self.img_path+'.zip', 'r') as archive:
                archive.extractall(self.img_path)
        if not os.path.isdir(self.msk_path):
            print('Unzipping masks to ', self.msk_path)
            with zipfile.ZipFile(self.msk_path+'.zip', 'r') as archive:
                archive.extractall(self.msk_path)


