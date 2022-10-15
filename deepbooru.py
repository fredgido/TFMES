#!/usr/bin/env python3
import os
from base64 import b64encode
from pathlib import Path
from typing import Union, cast

import keras.backend
import requests
import tensorflow as tf
from flask import Flask, request, jsonify
from jinja2 import Template
from skimage import transform
from skimage.transform import AffineTransform

if n_threads := os.getenv("TF_INTER_THREADS"):
    tf.config.threading.set_inter_op_parallelism_threads(int(n_threads))
if n_threads := os.getenv("TF_INTRA_THREADS"):
    tf.config.threading.set_intra_op_parallelism_threads(int(n_threads))

model = tf.keras.models.load_model(list(Path("model").glob("*.h5"))[0])
with (Path("model") / "tags.txt").open() as model_file:
    model_labels = model_file.read().splitlines()
    model_labels = [label[0:8] if label.startswith("rating:") else label for label in model_labels]
    deprecated_tag_converted = {
        "(9)": "circled_9",
        "(o)_(o)": "solid_circle_pupils",
        "/\\/\\/\\": "^^^",
        "alice_(wonderland)_(cosplay)": "alice_(alice_in_wonderland)_(cosplay)",
        "shimakaze_(kantai_collection)_(cosplay)": "shimakaze_(kancolle)_(cosplay)",
    }
    model_labels = [deprecated_tag_converted.get(label, label) for label in model_labels]


def image_to_tensor(image_raw: bytes, width: int, height: int):
    image = tf.io.decode_png(image_raw, channels=3)
    image = tf.image.resize(image, size=(height, width), method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=True)
    image = image.numpy()  # EagerTensor to np.array
    transformation = (
        AffineTransform(translation=(-image.shape[1] * 0.5, -image.shape[0] * 0.5))
        + AffineTransform(translation=(width * 0.5, height * 0.5))
    ).inverse
    image = transform.warp(image, transformation, output_shape=(width, height), order=1, mode="edge")
    image = image / 255.0
    return image


def download_image(url: str) -> bytes:
    r = requests.get(url, timeout=5, headers={"referer": "https://www.pixiv.net/"} if "pximg" in url else {})
    if r.status_code != 200:
        raise Exception(f"Error {r.status_code} on url {url}: {r.text}")
    return r.content


def process_images(
    input_images: Union[list[str], list[bytes]], min_score: float = 0.1
) -> list[list[tuple[str, float]]]:
    if isinstance(input_images[0], str):
        input_images = [tf.io.read_file(image_path) for image_path in input_images]
    images = [image_to_tensor(cast(bytes, file), model.input_shape[1], model.input_shape[2]) for file in input_images]
    images = [image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) for image in images]

    if len(images) > 10:
        predictions = model.predict(images.__iter__(), verbose=1)
        # time per image: 1->2.182 7->5.257 10->6.128 500->258
    else:
        predictions = [model(keras.backend.constant(x)).numpy().tolist()[0] for x in images]
        # time per image: 1->0.762 7->4.960 10->7.421 500->378

    outputs = []
    for prediction in predictions:
        tag_score = sorted(zip(prediction, range(0, len(prediction))), key=lambda x: x[0], reverse=True)
        output = [(model_labels[tag], float(score)) for score, tag in tag_score if score > min_score]
        # pprint.pprint(dict(output)['realistic'])
        # pprint.pprint(dict(output)['photorealistic'])
        # pprint.pprint(output[:50])
        outputs.append(output)
    return outputs


app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


@app.route("/", methods=["GET", "POST"])
@app.route("/evaluate", methods=["GET", "POST"])
def evaluate():
    if not (urls := request.values.get("url")) and not (files := request.files.getlist("file")):
        return """
            <form action="/evaluate" method="post" enctype="multipart/form-data">
                <input type="text" name="url" placeholder="image url">
                <input type="file" name="file">
                <input type="number" name="min_score" min="0" max="1" step="0.1" value="0.1">
                <input type="hidden" name="is_html" value="True">
                <input type="submit" value="Submit">
            </form>
            """
    images = [download_image(url) for url in urls.split(",")] if urls else [file.stream.read() for file in files]
    is_danbooru_compatible = request.args.get("threshold") or request.form.get("threshold")
    min_score = is_danbooru_compatible or request.args.get("min_score") or request.form.get("min_score")
    evaluation = process_images(images, float(min_score or 0.1))
    if not (request.form.get("is_html") or request.args.get("is_html")):
        if not is_danbooru_compatible:
            return jsonify(evaluation if len(evaluation) > 1 else evaluation[0])
        else:
            return jsonify(
                [{"filename": file.filename, "tags": dict(result)} for file, result in zip(files, evaluation)]
            )
    else:
        return Template(
            """
<a href="/">&lt; Back</a>
{% for img , prediction in predictions %}
<div style="flex-direction: row; max-height: 100vh; display: flex;">
 <div style="justify-content: center; align-items: center; flex: 1 1 0%; display: flex;">
  <img style="max-width: 100%; max-height: 100%; height: auto;" src="data:image/jpg;base64,{{ img | safe }}">
</div>
<div style="overflow: scroll;">
 <table style=" border-collapse: collapse; line-height: 1rem;">
 {% for tag, score in prediction %}
  <tr>
   <td>
    <a href="https://danbooru.donmai.us/wiki_pages/{{ tag | urlencode }}">?</a>
    <a href="https://danbooru.donmai.us/posts?tags={{ tag | urlencode }}">{{tag | replace("_", " ")}}</a>
   </td>
   <td >{{ "{:.0f}%".format(100 * score) }}</td>
  </tr>
  {% endfor %}
 </table>
 <textarea rows="4">{{ " ".join(prediction|map(attribute=0)) }}</textarea>
</div>
{% endfor %}
"""
        ).render(predictions=zip([b64encode(img).decode() for img in images], evaluation))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT") or 8000), debug=os.getenv("DEBUG") == "True")
