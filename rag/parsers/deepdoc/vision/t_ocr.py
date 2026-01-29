#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import argparse
import asyncio
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")))

from deepdoc.vision.seeit import draw_box
from rag.parsers.deepdoc.vision import OCR, init_in_out


def main(args):
    import torch.cuda

    cuda_devices = torch.cuda.device_count()
    limiter = [asyncio.Semaphore(1) for _ in range(cuda_devices)] if cuda_devices > 1 else None
    ocr = OCR()
    images, outputs = init_in_out(args)

    def __ocr(i, device_id, img):
        logging.info("Task {} start".format(i))
        bxs = ocr(np.array(img), device_id)
        bxs = [(line[0], line[1][0]) for line in bxs]
        bxs = [{"text": t, "bbox": [b[0][0], b[0][1], b[1][0], b[-1][1]], "type": "ocr", "score": 1} for b, t in bxs if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]]
        img = draw_box(img, bxs, ["ocr"], 1.0)
        img.save(outputs[i], quality=95)
        with open(outputs[i] + ".txt", "w", encoding="utf-8") as f:
            f.write("\n".join([o["text"] for o in bxs]))

        logging.info("Task {} done".format(i))

    async def __ocr_thread(i, device_id, img, limiter=None):
        if limiter:
            async with limiter:
                logging.info(f"Task {i} use device {device_id}")
                await asyncio.to_thread(__ocr, i, device_id, img)
        else:
            await asyncio.to_thread(__ocr, i, device_id, img)

    async def __ocr_launcher():
        tasks = []
        for i, img in enumerate(images):
            dev_id = i % cuda_devices if cuda_devices > 1 else 0
            semaphore = limiter[dev_id] if limiter else None
            tasks.append(asyncio.create_task(__ocr_thread(i, dev_id, img, semaphore)))

        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error("OCR tasks failed: {}".format(e))
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    asyncio.run(__ocr_launcher())

    print("OCR tasks are all done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", help="Directory where to store images or PDFs, or a file path to a single image or PDF", required=True)
    parser.add_argument("--output_dir", help="Directory where to store the output images. Default: './ocr_outputs'", default="./ocr_outputs")
    parser.add_argument("--devices", default=None, help="CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    main(args)
