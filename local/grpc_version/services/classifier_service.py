import grpc
from concurrent import futures
import cv2
import numpy as np
from PIL import Image
import io
import requests
import time
import os
import sys

import torch
import torchvision.transforms as transforms
import torchvision.models as models

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import pipeline_pb2
import pipeline_pb2_grpc


class ImageClassifierServicer(pipeline_pb2_grpc.ImageClassifierServicer):
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

    def Classify(self, request: pipeline_pb2.ClassifyRequest, context: grpc.ServicerContext) -> pipeline_pb2.ClassifyResponse:
        try:
            start = time.time()
            pil_image = Image.open(io.BytesIO(request.image_data))
            # Handle PNG RGBA or other modes
            np_img = np.array(pil_image)
            if len(np_img.shape) == 3 and np_img.shape[2] == 4:
                rgb = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
            elif len(np_img.shape) == 3 and np_img.shape[2] == 3:
                rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB) if pil_image.mode == 'BGR' else np_img
            elif len(np_img.shape) == 2:
                rgb = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
            else:
                rgb = np_img

            input_tensor = self.preprocess(rgb).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
            label = self.labels[int(top_catid[0])]
            resp = pipeline_pb2.ClassifyResponse(label=label)
            end = time.time()
            print(f"[classifier] Classify took {(end - start) * 1000:.2f} ms -> {label}", flush=True)
            return resp
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pipeline_pb2.ClassifyResponse()


def serve(port: int = 50043) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pipeline_pb2_grpc.add_ImageClassifierServicer_to_server(ImageClassifierServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Classifier service started on port {port}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()


