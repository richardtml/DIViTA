import functools
import hashlib
import os
import shutil
import tarfile
from os.path import isfile, join

import requests
from tqdm import tqdm

from files import FILES


def calculate_md5(filepath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_integrity(filepath, md5=None):
    if not isfile(filepath):
        return False
    if md5 is None:
        return True
    return md5 == calculate_md5(filepath)


def download_url(url, root, name):

    req = requests.get(url, stream=True, allow_redirects=True)
    if req.status_code != 200:
        req.raise_for_status()
        raise RuntimeError(
            f"Request to {url} returned status code {req.status_code}")

    size = int(req.headers.get('Content-Length', 0))

    os.makedirs(root, exist_ok=True)
    filepath = join(root, name)

    print(f'Downloading {url} to {filepath}')

    size = size if size != 0 else None
    req.raw.read = functools.partial(req.raw.read, decode_content=True)
    with tqdm.wrapattr(req.raw, 'read', total=size, ncols=75) as req_raw:
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(req_raw, f)


def extract_file(path, dst, compression='gz'):
    with tarfile.open(path, f'r:{compression}') as tar:
        tar.extractall(dst)


def download_extract(url, data_dir, name, md5=None, extract=False):
    path = join(data_dir, name)
    download_url(url, data_dir, name)
    if md5 is not None:
        print(f'Verifiying integrity of {path} with {md5}')
        check_integrity(path, md5)
    if extract and name.endswith('gz'):
        print(f'Extracting {path}')
        extract_file(path, data_dir)


def download(data_dir='trailers12k', remove_tgz=True):
    """Download Trailers12k MTGC and representations.

    Parameters
    ----------
    data_dir : str, optional
        Directory to save the data, by default 'trailers12k'.
    remove_tgz : [type], bool
        Remove tar.gz files.
    """
    for name, md5, url in FILES:
        is_gz = name.endswith('gz')
        download_extract(url, data_dir, name, md5, is_gz)
        if remove_tgz and is_gz:
            os.remove(join(data_dir, name))


if __name__ == '__main__':
    import fire
    fire.Fire(download)
