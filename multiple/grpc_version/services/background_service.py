import grpc
from concurrent import futures
import cv2
import numpy as np
from PIL import Image
import io
import time
import os
import sys

from rembg import remove, new_session

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import pipeline_pb2
import pipeline_pb2_grpc


class BackgroundRemoverServicer(pipeline_pb2_grpc.BackgroundRemoverServicer):
    def __init__(self) -> None:
        self.session = new_session("u2net")

    def Remove(self, request: pipeline_pb2.BackgroundRemoveRequest, context: grpc.ServicerContext) -> pipeline_pb2.BackgroundRemoveResponse:
        try:
            start = time.time()
            pil_image = Image.open(io.BytesIO(request.image_data))
            output_rgba = remove(pil_image, session=self.session)
            buf = io.BytesIO()
            output_rgba.save(buf, format='PNG')
            resp = pipeline_pb2.BackgroundRemoveResponse(image_data=buf.getvalue())
            end = time.time()
            print(f"[background] Remove took {(end - start) * 1000:.2f} ms", flush=True)
            return resp
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pipeline_pb2.BackgroundRemoveResponse()


def serve(port: int = 50042) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pipeline_pb2_grpc.add_BackgroundRemoverServicer_to_server(BackgroundRemoverServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Background removal service started on port {port}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()


