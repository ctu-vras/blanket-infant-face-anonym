# TODO - move this file somewhere else

from datetime import datetime
import time
import os
import base64
import json
import urllib.request
import cv2
import numpy as np


# webui_server_url = 'http://127.0.0.1:7862'

# out_dir = 'api_out'
# out_dir_t2i = os.path.join(out_dir, 'txt2img')
# out_dir_i2i = os.path.join(out_dir, 'img2img')
# out_dir_inpaint = os.path.join(out_dir, "inpaint")
# os.makedirs(out_dir_t2i, exist_ok=True)
# os.makedirs(out_dir_i2i, exist_ok=True)
# os.makedirs(out_dir_inpaint, exist_ok=True)


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def encode_bgr_to_base64(img_bgr, extension=".png"):
    action_result, img_buffer = cv2.imencode(extension, img_bgr)

    if action_result:
        return base64.b64encode(img_buffer).decode('utf-8')
    else:
        raise ValueError("Unable to encode image to format {}".format(extension))


def decode_base64_to_bgr(base64_str):
    return cv2.imdecode(np.fromstring(base64.b64decode(base64_str), np.uint8), cv2.IMREAD_COLOR)


def call_api(webui_server_url, api_endpoint, **payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))


# def call_txt2img_api(**payload):
#     response = call_api('sdapi/v1/txt2img', **payload)
#     for index, image in enumerate(response.get('images')):
#         save_path = os.path.join(out_dir_t2i, f'txt2img-{timestamp()}-{index}.png')
#         decode_and_save_base64(image, save_path)
#
#
# def call_img2img_api(**payload):
#     response = call_api('sdapi/v1/img2img', **payload)
#     for index, image in enumerate(response.get('images')):
#         save_path = os.path.join(out_dir_i2i, f'img2img-{timestamp()}-{index}.png')
#         decode_and_save_base64(image, save_path)
