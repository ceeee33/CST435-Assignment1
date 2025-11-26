import base64
import cv2
import numpy as np
from PIL import Image
import io
import time

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import uvicorn


app = FastAPI(title="Noise Reduction Service")


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _b64decode(data: str) -> bytes:
    return base64.b64decode(data.encode("utf-8"))


@app.post("/reduce")
def reduce_noise(payload: dict = Body(...)):
    try:
        start = time.time()
        img_b64 = payload.get("image_data")
        if not img_b64:
            return JSONResponse(status_code=400, content={"detail": "image_data is required"})
        image = Image.open(io.BytesIO(_b64decode(img_b64)))
        cv_image = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        denoised_image = cv2.medianBlur(cv_image, 5)
        output = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
        pil_out = Image.fromarray(output)
        buf = io.BytesIO()
        pil_out.save(buf, format='PNG')
        end = time.time()
        print(f"[noise] Reduce took {(end - start) * 1000:.2f} ms", flush=True)
        return {"image_data": _b64encode(buf.getvalue())}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


def serve(port: int = 50061) -> None:
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    serve()
