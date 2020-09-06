#import math
#import numpy as np
#from typing import Any, Union
#import six
#import validators
from skimage.transform import AffineTransform
from skimage import transform
import tensorflow as tf

import pprint
import os
import requests
import zipfile
import shutil
import hashlib
import time
from flask import Flask, request
from flask import render_template
import json
import copy
import re


if not os.path.isfile(os.path.join("model","model-resnet_custom_v3.h5")) or not os.path.isfile(os.path.join("model","tags.txt")):
    print("downloading model and tags")  # https://api.github.com/repos/KichangKim/DeepDanbooru/releases
    with requests.get("https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20200101-sgd-e30/deepdanbooru-v3-20200101-sgd-e30.zip", stream=True, allow_redirects=True) as r:
        with open("model.zip", 'wb') as zip:
            shutil.copyfileobj(r.raw, zip)
    with zipfile.ZipFile("model.zip", 'r') as zip_ref:
        zip_ref.extractall("model")
    print("download complete")
    os.mkdir("images")

tf.config.threading.set_inter_op_parallelism_threads(12)
tf.config.threading.set_intra_op_parallelism_threads(12)
print(tf.config.threading.get_inter_op_parallelism_threads())
print(tf.config.threading.get_intra_op_parallelism_threads())

model = tf.keras.models.load_model(os.path.join("model","model-resnet_custom_v3.h5"))
imagenet_labels = open(os.path.join("model","tags.txt")).read().splitlines()

model.run_eagerly=False

def load_image_for_evaluate(input, width: int, height: int):
    image_raw = tf.io.read_file(input)
    image = tf.io.decode_png(image_raw, channels=3)
    image = tf.image.resize(image, size=(height, width), method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=True)
    image = image.numpy()  # EagerTensor to np.array
    image = transform.warp(image, (AffineTransform(translation=(-image.shape[1] * 0.5, -image.shape[0] * 0.5)) + AffineTransform(translation=(width * 0.5, height * 0.5))).inverse,
                           output_shape=(width, height), order=1, mode='edge')
    image = image / 255.0
    return image


def download_image(url, filename):
    url_hash = hashlib.sha256(str(url).encode()).hexdigest()
    if os.path.isdir(os.path.join("images", url_hash)) and os.path.isfile(os.path.join("images", url_hash, filename)):
        return os.path.join("images", url_hash, filename)
    os.mkdir(os.path.join("images", url_hash))
    picture_request = requests.get(url, timeout=5,headers={'referer': "https://www.pixiv.net/"} if "pximg" in url else {})
    if picture_request.status_code != 200:
        raise Exception("error " + str(picture_request.status_code) + " on " + url + "  " + picture_request.text)
    with open(os.path.join("images", url_hash, filename), 'wb') as f:
        f.write(picture_request.content)
    return os.path.join("images", url_hash, filename)


def process_image(imagepath):
    r = load_image_for_evaluate(imagepath, 512, 512)
    image = r.reshape((1, r.shape[0], r.shape[1], r.shape[2]))

    #image = [image, image, image, image, image, image, image, image, image, image]
    a0 = time.time()
    a1 = time.process_time()
    y = model.predict(image,verbose=1)
    print("lenght"+str(len(y)))
    y=y[0]

    b0 = time.time()
    b1 = time.process_time()
    print(b0 - a0)
    print(b1 - a1)

    rez = zip(list(y.tolist()), list(range(0, len(y) - 1)))
    t = sorted(rez, key=lambda x: x[0], reverse=True)
    cor = [(imagenet_labels[b], a) for a, b in t]
    pprint.pprint(dict(cor)['realistic'])
    pprint.pprint(cor[:50])
    return cor


url = "https://danbooru.donmai.us/data/sample/sample-3caa27ff7617fa2691e0727ef9104240.jpg"
process_image(download_image(url,url.split('/')[-1]))

from urllib.parse import parse_qs, urlparse

from requests.auth import HTTPBasicAuth


def proxy_get(
        user, api_key, url,
        get=requests.get,
        proxy_url="https://danbooru.donmai.us/uploads/image_proxy",
):
    return get(
        proxy_url,
        auth=HTTPBasicAuth(user, api_key),
        params={"url": url},
    )


def parse_url(url, proxy_hostname="donmai.us"):
    parsed_url = urlparse(url)
    proxy_used = (
            parsed_url.path == "/uploads/image_proxy" and (
            parsed_url.netloc == proxy_hostname or
            parsed_url.netloc.endswith("." + proxy_hostname)
    )
    )

    if proxy_used:
        proxy_hostname = parsed_url.netloc
        parsed_query = parse_qs(parsed_url.query)

        try:
            url = parsed_query["url"][0]
        except (KeyError, IndexError):
            proxy_hostname = ""
        else:
            parsed_url = urlparse(url)
    else:
        proxy_hostname = ""

    return url, re.sub('[^-a-zA-Z0-9_.() ]+', '', os.path.basename(parsed_url.path)), proxy_hostname




app = Flask(__name__)

# <li class="tag-type-0" data-tag-name="1girl"><a class="search-tag selected" href="/posts?tags=1girl">1gil</a></li>
htmlinsert = ["""
<div class="tag-column deep-tags-column is-empty-false">
    <h6>deep</h6>
    <ul>""",
              """
                  </ul>
                  </div>
                  """
              ]
import concurrent.futures

#executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

@app.route("/deep", methods=['GET', 'POST'])
def eval():
    print(request.args)
    if request.args.get('url') is None:
        return "error none"

    url, filename, proxy_hostname = parse_url(request.args.get('url'))
    imagefilepath = download_image(url, filename)

    if os.path.isfile(os.path.join("images", hashlib.sha256(str(url).encode()).hexdigest(), "evaluation.json")):
        with open(os.path.join("images", hashlib.sha256(str(url).encode()).hexdigest(), "evaluation.json"), 'r') as f:
            output = json.load(f)
    else:
        #fut = executor.submit(process_image, imagefilepath)
        output = dict(process_image(imagefilepath))
        #output = dict(fut.result(timeout=60))
        with open(os.path.join("images", hashlib.sha256(str(url).encode()).hexdigest(), "evaluation.json"), 'w') as f:
            json.dump(output, f)

    threshold = 0.1 if request.args.get('threshold') is None else float(request.args.get('threshold'))

    output = dict(filter(lambda x: x[1] > threshold, output.items()))

    if request.args.get('html') is not None:
        htmlLines = []  # copy.deepcopy(htmlinsert)
        for key, value in output.items():
            htmlLines.append(f'<li class="tag-type-0" data-tag-name="{str(key)}"><a class="search-tag" href="/posts?tags={str(key)}">{str(key)}</a></li>')  # htmlLines.append(f'<li class="tag-type-0" data-tag-name="{str(key)}"><a class="search-tag" href="/posts?tags={str(key)}">{str(key)} - {str(value)}</a></li>')
        result = copy.deepcopy(htmlinsert)
        result.insert(1, '\n'.join(htmlLines))
        return '\n'.join(result)
    elif request.args.get('readable') is not None:
        htmlLines = []
        for key, value in output.items():
            htmlLines.append(str(key) + " " + str(value) + "<br/>")
        return '\n'.join(htmlLines)
    else:
        return json.dumps(dict(output), indent=4)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
