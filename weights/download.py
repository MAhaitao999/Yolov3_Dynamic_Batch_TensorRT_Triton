from builtins import *

import hashlib
import os.path

import wget


def generate_md5_checksum(local_path):
    """Returns the MD5 checksum of a local file.

    Keyword argument:
    local_path -- path of the file whose checksum shall be generated
    """
    with open(local_path, 'rb') as local_file:
        data = local_file.read()
        return hashlib.md5(data).hexdigest()


def download_file(local_path, link, checksum_reference=None):
    """Checks if a local file is present and downloads it from the specified path otherwise.
    If checksum_reference is specified, the file's md5 checksum is compared against the
    expected value.

    Keyword arguments:
    local_path -- path of the file whose checksum shall be generated
    link -- link where the file shall be downloaded from if it is not found locally
    checksum_reference -- expected MD5 checksum of the file
    """
    if not os.path.exists(local_path):
        print('Downloading from %s, this may take a while...' % link)
        wget.download(link, local_path)
        print()
    if checksum_reference is not None:
        checksum = generate_md5_checksum(local_path)
        if checksum != checksum_reference:
            raise ValueError(
                'The MD5 checksum of local file %s differs from %s, please manually remove \
                 the file and try again.' %
                (local_path, checksum_reference))
    return local_path


if __name__ == "__main__":
    weights_file_path = download_file(
        'yolov3.weights',
        'https://pjreddie.com/media/files/yolov3.weights',
        'c84e5b99d0e52cd466ae710cadf6d84c')

