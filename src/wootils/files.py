import os
# import sys
import glob
import tarfile
import zipfile


# Extract a tarfile
def extract_tar(tar_url, extract_path=None, verbose=False):
    #### Example usase 
    # extract_tar(file, extract_path=file, verbose=True)
    if verbose:
        print(f'Extracting {os.path.split(tar_url)[1]}....')
    if extract_path is None:
        extract_path = tar_url[:extract_path.rfind('.')]
    elif tar_url==extract_path:
        extract_path = tar_url[:extract_path.rfind('.')]
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    tar = tarfile.open(tar_url, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract_tar(item.name, "./" + item.name[:item.name.rfind('/')])
    tar.close()
    if verbose:
        print(f'Extracted {os.path.split(tar_url)[1]} to {extract_path}')


# Extract a zipfile
def extract_zip(zip_url, extract_path=None, verbose=False):
    #### Example usase 
    # extract_zip(file, extract_path=file, verbose=True)
    with zipfile.ZipFile(zip_url, 'r') as zip_ref:
        if verbose:
            print(f'Extracting {os.path.split(zip_url)[1]}....')
        if extract_path is None:
            extract_path = zip_url[:extract_path.rfind('.')]
        elif zip_url==extract_path:
            extract_path = zip_url[:extract_path.rfind('.')]
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)

        zip_ref.extractall(extract_path)
    if verbose:
        print(f'Extracted {os.path.split(zip_url)[1]} to {extract_path}')
    


        
