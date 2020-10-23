#!/usr/bin/env python3

"""
Made by @nizhib
"""
import os
import base64
import io
import logging
import sys
import time
from http import HTTPStatus

import numpy as np
import requests
from flask import Flask, request, jsonify
from imageio import imsave
from PIL import Image
from waitress import serve

import torch.backends.cudnn

from api import Segmentator

torch.backends.cudnn.benchmark = True
torch.manual_seed(123)
torch.cuda.random.manual_seed(123)

# Set proper device for computations
dtype, device, cuda_device_id = torch.float32, None, 0
os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(str(cuda_device_id) if cuda_device_id is not None else '')
if cuda_device_id is not None and torch.cuda.is_available():
    device = 'cuda:{0:d}'.format(0)
else:
    device = torch.device('cpu')

print(dtype, device)

LOGGING_LEVEL = 'INFO'
LOGGING_FORMAT = '[%(asctime)s] %(name)s:%(lineno)d: %(message)s'

logging.basicConfig(format=LOGGING_FORMAT, level=LOGGING_LEVEL)

segmentator = Segmentator(path='./checkpoints/last_full.pth', dtype=dtype, device=device)

app = Flask(__name__)
logger = logging.getLogger(__file__)


@app.route('/segment', methods=['POST'])
def handle():
    start = time.time()
    status = HTTPStatus.OK
    result = {'success': False}

    try:
        data = request.json
        if 'image' in data:
            blob = io.BytesIO(base64.b64decode(data['image']))
            img = Image.open(blob).convert('RGB')
        elif 'url' in data:
            blob = io.BytesIO(requests.get(data['url']).content)
            img = Image.open(blob).convert('RGB')
        else:
            raise ValueError(
                f'No image source found in request fields: {data.keys()}')

        st_time = time.time()
        mask = segmentator.predict(img)
        logger.info('2. Segmentation took {} s.'.format(time.time() - st_time))

        mask = (mask * 255).astype(np.uint8)

        fmem = io.BytesIO()
        imsave(fmem, mask, 'png')
        fmem.seek(0)
        mask64 = base64.b64encode(fmem.read()).decode('utf-8')

        result['data'] = {'mask': mask64}
        result['success'] = True
    except Exception as e:
        logger.exception(e)
        result['message'] = str(e)
        status = HTTPStatus.INTERNAL_SERVER_ERROR

    result['total'] = time.time() - start

    return jsonify(result), status


if __name__ == '__main__':
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 5000

    serve(app, host='0.0.0.0', port=port)
