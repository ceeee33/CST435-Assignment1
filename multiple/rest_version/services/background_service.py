import base64
import io
import time
from PIL import Image

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import uvicorn

from rembg import remove, new_session


app = FastAPI(title="Background Removal Service")


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _b64decode(data: str) -> bytes:
    return base64.b64decode(data.encode("utf-8"))


class _Session:
    def __init__(self) -> None:
        self.session = new_session("u2net")


_S = _Session()


@app.post("/remove")
def remove_background(payload: dict = Body(...)):
    try:
        start = time.time()
        img_b64 = payload.get("image_data")
        if not img_b64:
            return JSONResponse(status_code=400, content={"detail": "image_data is required"})
        pil_image = Image.open(io.BytesIO(_b64decode(img_b64)))
        output_rgba = remove(pil_image, session=_S.session)
        buf = io.BytesIO()
        output_rgba.save(buf, format='PNG')
        end = time.time()
        print(f"[background] Remove took {(end - start) * 1000:.2f} ms", flush=True)
        return {"image_data": _b64encode(buf.getvalue())}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


def serve(port: int = 50062) -> None:
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    serve()
