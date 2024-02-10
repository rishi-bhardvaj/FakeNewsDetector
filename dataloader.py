import os
import config
import sys
import shutil
import zipfile
TOKENIZER_DOWNLOADER = False

def data_load():
    if not os.path.isfile(os.path.join(config.DATA_PATH, config.DATASET_TITLE)):
        os.system('kaggle datasets download -d' + config.DATASET)
        if not os.path.isdir(config.DATA_PATH):
            os.makedirs(config.DATA_PATH)
        shutil.move(config.DATASET_TITLE, os.path.join(config.DATA_PATH, config.DATASET_TITLE))
        if config.DATASET_TITLE.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(config.DATA_PATH, config.DATASET_TITLE)) as zip_data:
                zip_data.extractall(config.DATA_PATH)
    else:
        sys.stdout.write('Skipping downloading as data already exists')

