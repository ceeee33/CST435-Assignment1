import base64
import cv2
import numpy as np
from PIL import Image
import io
import requests
import time

import torch
import torchvision.transforms as transforms
import torchvision.models as models

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import uvicorn


app = FastAPI(title="Image Classifier Service")


def _b64decode(data: str) -> bytes:
    return base64.b64decode(data.encode("utf-8"))


class _Model:
    def __init__(self) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.labels = self._load_imagenet_labels()

    def _load_imagenet_labels(self):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return [line.strip() for line in response.text.split('\n') if line.strip()]
        except Exception:
            return ["unknown"]


_M = _Model()


@app.post("/classify")
def classify(payload: dict = Body(...)):
    try:
        start = time.time()
        img_b64 = payload.get("image_data")
        if not img_b64:
            return JSONResponse(status_code=400, content={"detail": "image_data is required"})
        pil_image = Image.open(io.BytesIO(_b64decode(img_b64)))
        np_img = np.array(pil_image)
        if len(np_img.shape) == 3 and np_img.shape[2] == 4:
            rgb = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        elif len(np_img.shape) == 3 and np_img.shape[2] == 3:
            rgb = np_img
        elif len(np_img.shape) == 2:
            rgb = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = np_img

        input_tensor = _M.preprocess(rgb).unsqueeze(0)
        with torch.no_grad():
            output = _M.model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)
        label = _M.labels[int(top_catid[0])]
        end = time.time()
        print(f"[classifier] Classify took {(end - start) * 1000:.2f} ms -> {label}", flush=True)
        return {"label": label}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


def serve(port: int = 50063) -> None:
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    serve()
