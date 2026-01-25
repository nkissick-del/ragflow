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

import logging
import math
import os
import re

from collections import Counter
from copy import deepcopy

import cv2
import numpy as np
from huggingface_hub import snapshot_download

from common.file_utils import get_project_base_directory
from rag.parsers.deepdoc.vision import Recognizer
from .operators import nms


LAYOUT_LABELS = [
    "_background_",
    "Text",
    "Title",
    "Figure",
    "Figure caption",
    "Table",
    "Table caption",
    "Header",
    "Footer",
    "Reference",
    "Equation",
]


class LayoutRecognizer(Recognizer):
    labels = LAYOUT_LABELS

    def __init__(self, domain):
        try:
            model_dir = os.path.join(get_project_base_directory(), "rag/res/deepdoc")
            super().__init__(self.labels, domain, model_dir)
        except (FileNotFoundError, OSError, IOError, RuntimeError) as e:
            logging.warning(f"Failed to load local model: {e}, downloading...")
            model_dir = snapshot_download(repo_id="InfiniFlow/deepdoc", local_dir=os.path.join(get_project_base_directory(), "rag/res/deepdoc"), local_dir_use_symlinks=False)
            super().__init__(self.labels, domain, model_dir)

        self.garbage_layouts = ["footer", "header", "reference"]
        self.client = None
        if os.environ.get("TENSORRT_DLA_SVR"):
            from deepdoc.vision.dla_cli import DLAClient

            self.client = DLAClient(os.environ["TENSORRT_DLA_SVR"])

    def process_layouts(self, ocr_result, layout_result, image_shape, scale_factor, thr=0.4, drop=True):
        bxs = list(ocr_result)
        lts = [
            {
                "type": b["type"],
                "score": float(b["score"]),
                "x0": b["bbox"][0] / scale_factor,
                "x1": b["bbox"][2] / scale_factor,
                "top": b["bbox"][1] / scale_factor,
                "bottom": b["bbox"][-1] / scale_factor,
                "page_number": b.get("page_number", 0),
            }
            for b in layout_result
            if float(b["score"]) >= thr or b["type"] not in self.garbage_layouts
        ]
        if lts:
            lts = self.sort_Y_firstly(lts, np.mean([lt["bottom"] - lt["top"] for lt in lts]) / 2)
        lts = self.layouts_cleanup(bxs, lts)

        garbages = {}

        def findLayout(ty):
            nonlocal bxs, lts
            lts_ = [lt for lt in lts if lt["type"] == ty]
            i = 0
            while i < len(bxs):
                if bxs[i].get("layout_type"):
                    i += 1
                    continue

                # Check garbage text pattern inline or via helper if needed
                # For now assuming simple text check or keeping logic here
                patt = [r"^•+$", r"^[0-9]{1,2} / ?[0-9]{1,2}$", r"^[0-9]{1,2} of [0-9]{1,2}$", r"^http://[^ ]{12,}", r"\(cid *: *[0-9]+ *\)"]
                if any(re.search(p, bxs[i].get("text", "")) for p in patt):
                    bxs.pop(i)
                    continue

                ii = self.find_overlapped_with_threshold(bxs[i], lts_, thr=thr)
                if ii is None:
                    bxs[i]["layout_type"] = ""
                    i += 1
                    continue
                lts_[ii]["visited"] = True
                keep_feats = [
                    lts_[ii]["type"] == "footer" and bxs[i]["bottom"] < image_shape[1] * 0.9 / scale_factor,
                    lts_[ii]["type"] == "header" and bxs[i]["top"] > image_shape[1] * 0.1 / scale_factor,
                ]
                if drop and lts_[ii]["type"] in self.garbage_layouts and not any(keep_feats):
                    if lts_[ii]["type"] not in garbages:
                        garbages[lts_[ii]["type"]] = []
                    garbages[lts_[ii]["type"]].append(bxs[i]["text"])
                    bxs.pop(i)
                    continue

                bxs[i]["layoutno"] = f"{ty}-{ii}"
                bxs[i]["layout_type"] = lts_[ii]["type"] if lts_[ii]["type"] != "equation" else "figure"
                i += 1

        for lt in ["footer", "header", "reference", "figure caption", "table caption", "title", "table", "text", "figure", "equation"]:
            findLayout(lt)

        # add box to figure layouts which has not text box
        for i, lt in enumerate([lt for lt in lts if lt["type"] in ["figure", "equation"]]):
            if lt.get("visited"):
                continue
            lt = deepcopy(lt)
            if "type" in lt:
                del lt["type"]
            lt["text"] = ""
            lt["layout_type"] = "figure"
            lt["layoutno"] = f"figure-{i}"
            bxs.append(lt)

        garbage_set = set()
        for k in garbages.keys():
            garbages[k] = Counter(garbages[k])
            for g, c in garbages[k].items():
                if c > 1:
                    garbage_set.add(g)

        ocr_res = [b for b in bxs if b["text"].strip() not in garbage_set]
        return ocr_res, lts

    def __call__(self, image_list, ocr_res, scale_factor=3, thr=0.2, batch_size=16, drop=True):
        if self.client:
            layouts = self.client.predict(image_list)
        else:
            layouts = super().__call__(image_list, thr, batch_size)

        assert len(image_list) == len(ocr_res)
        assert len(image_list) == len(layouts)

        final_ocr_res = []
        final_page_layout = []

        for pn, lts in enumerate(layouts):
            # Pass image_shape as (W, H) or (H, W) depending on usage, here we use image.size (W,H) or shape (H,W,C)
            # PIL image.size is (W, H). cv2/numpy shape is (H, W, C)
            # In process_layouts we used image_shape[1] for height check, so we expect (W, H) if passing PIL size-like tuple
            # or if passing shape (H, W), we need to be careful.
            # Assuming image_list contains PIL images or numpy arrays.
            if hasattr(image_list[pn], "size"):
                h = image_list[pn].size[1]
                w = image_list[pn].size[0]
            else:
                h = image_list[pn].shape[0]
                w = image_list[pn].shape[1]

            res, page_lt = self.process_layouts(ocr_res[pn], lts, (w, h), scale_factor, thr, drop)
            final_ocr_res.extend(res)
            final_page_layout.append(page_lt)

        return final_ocr_res, final_page_layout

    def forward(self, image_list, thr=0.7, batch_size=16):
        return super().__call__(image_list, thr, batch_size)


class LayoutRecognizer4YOLOv10(LayoutRecognizer):
    labels = LAYOUT_LABELS

    def __init__(self, domain="layout"):
        super().__init__(domain)
        self.auto = False
        self.scaleFill = False
        self.scaleup = True
        self.stride = 32
        self.center = True

    def preprocess(self, image_list):
        inputs = []
        new_shape = self.input_shape  # height, width
        for img in image_list:
            shape = img.shape[:2]  # current shape [height, width]
            # Scale ratio (new / old)
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
            # Compute padding
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
            dw /= 2  # divide padding into 2 sides
            dh /= 2
            ww, hh = new_unpad
            img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).astype(np.float32)
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
            img /= 255.0
            img = img.transpose(2, 0, 1)
            img = img[np.newaxis, :, :, :].astype(np.float32)
            inputs.append({self.input_names[0]: img, "scale_factor": [shape[1] / ww, shape[0] / hh, dw, dh]})

        return inputs

    def postprocess(self, boxes, inputs, thr):
        boxes = np.squeeze(boxes)
        scores = boxes[:, 4]
        boxes = boxes[scores > thr, :]
        scores = scores[scores > thr]
        if len(boxes) == 0:
            return []
        class_ids = boxes[:, -1].astype(int)
        boxes = boxes[:, :4]
        boxes[:, 0] -= inputs["scale_factor"][2]
        boxes[:, 2] -= inputs["scale_factor"][2]
        boxes[:, 1] -= inputs["scale_factor"][3]
        boxes[:, 3] -= inputs["scale_factor"][3]
        input_shape = np.array([inputs["scale_factor"][0], inputs["scale_factor"][1], inputs["scale_factor"][0], inputs["scale_factor"][1]])
        boxes = np.multiply(boxes, input_shape, dtype=np.float32)

        unique_class_ids = np.unique(class_ids)
        indices = []
        for class_id in unique_class_ids:
            class_indices = np.where(class_ids == class_id)[0]
            class_boxes = boxes[class_indices, :]
            class_scores = scores[class_indices]
            class_keep_boxes = nms(class_boxes, class_scores, 0.45)
            indices.extend(class_indices[class_keep_boxes])

        return [{"type": self.label_list[class_ids[i]].lower(), "bbox": [float(t) for t in boxes[i].tolist()], "score": float(scores[i])} for i in indices]


class AscendLayoutRecognizer(LayoutRecognizer):
    labels = LAYOUT_LABELS

    def __init__(self, domain):
        from ais_bench.infer.interface import InferSession

        model_dir = os.path.join(get_project_base_directory(), "rag/res/deepdoc")
        model_file_path = os.path.join(model_dir, domain + ".om")

        if not os.path.exists(model_file_path):
            raise ValueError(f"Model file not found: {model_file_path}")

        device_id = int(os.getenv("ASCEND_LAYOUT_RECOGNIZER_DEVICE_ID", 0))
        self.session = InferSession(device_id=device_id, model_path=model_file_path)
        self.input_shape = self.session.get_inputs()[0].shape[2:4]  # H,W
        self.garbage_layouts = ["footer", "header", "reference"]

    def preprocess(self, image_list):
        inputs = []
        H, W = self.input_shape
        for img in image_list:
            h, w = img.shape[:2]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

            r = min(H / h, W / w)
            new_unpad = (int(round(w * r)), int(round(h * r)))
            dw, dh = (W - new_unpad[0]) / 2.0, (H - new_unpad[1]) / 2.0

            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

            img /= 255.0
            img = img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

            inputs.append(
                {
                    "image": img,
                    "scale_factor": [w / new_unpad[0], h / new_unpad[1]],
                    "pad": [dw, dh],
                    "orig_shape": [h, w],
                }
            )
        return inputs

    def postprocess(self, boxes, inputs, thr=0.25):
        arr = np.squeeze(boxes)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        results = []
        if arr.shape[1] == 6:
            # [x1,y1,x2,y2,score,cls]
            m = arr[:, 4] >= thr
            arr = arr[m]
            if arr.size == 0:
                return []
            xyxy = arr[:, :4].astype(np.float32)
            scores = arr[:, 4].astype(np.float32)
            cls_ids = arr[:, 5].astype(np.int32)

            if "pad" in inputs:
                dw, dh = inputs["pad"]
                sx, sy = inputs["scale_factor"]
                xyxy[:, [0, 2]] -= dw
                xyxy[:, [1, 3]] -= dh
                xyxy *= np.array([sx, sy, sx, sy], dtype=np.float32)
            else:
                # backup
                sx, sy = inputs["scale_factor"]
                xyxy *= np.array([sx, sy, sx, sy], dtype=np.float32)

            keep_indices = []
            for c in np.unique(cls_ids):
                idx = np.where(cls_ids == c)[0]
                k = nms(xyxy[idx], scores[idx], 0.45)
                keep_indices.extend(idx[k])

            for i in keep_indices:
                cid = int(cls_ids[i])
                if 0 <= cid < len(self.labels):
                    results.append({"type": self.labels[cid].lower(), "bbox": [float(t) for t in xyxy[i].tolist()], "score": float(scores[i])})
            return results

        raise ValueError(f"Unexpected output shape: {arr.shape}")

    def __call__(self, image_list, ocr_res, scale_factor=3, thr=0.2, batch_size=16, drop=True):
        assert len(image_list) == len(ocr_res)

        images = [np.array(im) if not isinstance(im, np.ndarray) else im for im in image_list]

        conf_thr = max(thr, 0.08)
        layouts_raw = []

        batch_loop_cnt = math.ceil(float(len(images)) / batch_size)
        for bi in range(batch_loop_cnt):
            s = bi * batch_size
            e = min((bi + 1) * batch_size, len(images))
            batch_images = images[s:e]

            inputs_list = self.preprocess(batch_images)
            logging.debug("preprocess done")

            for ins in inputs_list:
                feeds = [ins["image"]]
                out_list = self.session.infer(feeds=feeds, mode="static")

                for out in out_list:
                    # Ascend postprocess returns list of dicts with bbox in original image coordinates
                    lts = self.postprocess(out, ins, conf_thr)
                    layouts_raw.append(lts)

        final_ocr_res = []
        final_page_layout = []

        for pn, lts in enumerate(layouts_raw):
            if hasattr(image_list[pn], "size"):
                h = image_list[pn].size[1]
                w = image_list[pn].size[0]
            else:
                h = image_list[pn].shape[0]
                w = image_list[pn].shape[1]

            res, page_lt = self.process_layouts(ocr_res[pn], lts, (w, h), scale_factor, thr, drop)
            final_ocr_res.extend(res)
            final_page_layout.append(page_lt)

        return final_ocr_res, final_page_layout
