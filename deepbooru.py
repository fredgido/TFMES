import hashlib
import os

import keras.backend
import requests
import tensorflow as tf
from flask import Flask, request, jsonify
from skimage import transform
from skimage.transform import AffineTransform

(n_threads := os.getenv("TF_INTER_THREADS")) and tf.config.threading.set_inter_op_parallelism_threads(int(n_threads))
(n_threads := os.getenv("TF_INTRA_THREADS")) and tf.config.threading.set_intra_op_parallelism_threads(int(n_threads))

model = tf.keras.models.load_model(os.path.join("model", "model-resnet_custom_v4.h5"))
imagenet_labels = open(os.path.join("model", "tags.txt")).read().splitlines()


def image_to_tensor(image_path: str, width: int, height: int):
    image_raw = tf.io.read_file(image_path)
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


def download_image(url: str, filename: str) -> str:
    url_hash = hashlib.sha256(str(url).encode()).hexdigest()
    not os.path.exists("images") and not os.path.isdir("images") and os.mkdir("images")
    if os.path.isfile(os.path.join("images", url_hash, filename)):
        return os.path.join("images", url_hash, filename)
    os.mkdir(os.path.join("images", url_hash))
    r = requests.get(url, timeout=5, headers={"referer": "https://www.pixiv.net/"} if "pximg" in url else {})
    if r.status_code != 200:
        raise Exception("error " + str(r.status_code) + " on " + url + "  " + r.text)
    with open(os.path.join("images", url_hash, filename), "wb") as f:
        f.write(r.content)
    return os.path.join("images", url_hash, filename)


def process_images(image_paths: list[str], min_score: float = 0) -> list[list[tuple[str, float]]]:
    images = [image_to_tensor(image_path, model.input_shape[1], model.input_shape[2]) for image_path in image_paths]
    images = [image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) for image in images]

    if len(images) > 10:
        predictions = model.predict(images.__iter__(), verbose=1)
        # time per image: 1->2.182 7->5.257 10->6.128 500->258
    else:
        predictions = [model(keras.backend.constant(x)).numpy().tolist()[0] for x in images]
        # time per image: 1->0.762 7->4.960 10->7.421 500->378

    outputs = []
    for prediction in predictions:
        tag_score = sorted(zip(list(prediction), list(range(0, len(prediction) - 1))), key=lambda x: x[0], reverse=True)
        output = [(imagenet_labels[tag], score) for score, tag in tag_score if score > min_score]
        # pprint.pprint(dict(output)['realistic'])
        # pprint.pprint(dict(output)['photorealistic'])
        # pprint.pprint(output[:50])
        outputs.append(output)
    return outputs


app = Flask(__name__)


@app.route("/", methods=["GET"])
@app.route("/evaluate", methods=["GET"])
def evaluate():
    if not (urls := request.args.get("url")):
        return """
            <form action="/evaluate" method="get">
                <input type="text" name="url" placeholder="image url">
                <input type="number" name="min_score" min="0" max="1" step="0.01" value="0.1">
                <input type="submit" value="Submit">
            </form>
            """
    evaluation = process_images(
        [download_image(url, url.split("/")[-1]) for url in urls.split(",")],
        float(request.args.get("min_score", 0)),
    )
    return jsonify(evaluation if len(evaluation) > 1 else evaluation[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT") or 8000, debug=os.getenv("DEBUG") == "True")
