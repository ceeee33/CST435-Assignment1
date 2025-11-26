import grpc
from concurrent import futures
import cv2
import numpy as np
from PIL import Image
import io
import time
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import pipeline_pb2
import pipeline_pb2_grpc


class NoiseReducerServicer(pipeline_pb2_grpc.NoiseReducerServicer):
    def Reduce(self, request: pipeline_pb2.NoiseReductionRequest, context: grpc.ServicerContext) -> pipeline_pb2.NoiseReductionResponse:
        try:
            start = time.time()
            image = Image.open(io.BytesIO(request.image_data))
            cv_image = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
            denoised_image = cv2.medianBlur(cv_image, 5)
            output = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
            pil_out = Image.fromarray(output)
            buf = io.BytesIO()
            pil_out.save(buf, format='PNG')
            resp = pipeline_pb2.NoiseReductionResponse(image_data=buf.getvalue())
            end = time.time()
            print(f"[noise] Reduce took {(end - start) * 1000:.2f} ms", flush=True)
            return resp
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pipeline_pb2.NoiseReductionResponse()


def serve(port: int = 50041) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pipeline_pb2_grpc.add_NoiseReducerServicer_to_server(NoiseReducerServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Noise reduction service started on port {port}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()


