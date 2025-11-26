const express = require('express');

const app = express();
app.use(express.json({ limit: '50mb' }));

function b64encode(buf) {
  return Buffer.from(buf).toString('base64');
}

app.post('/upload', async (req, res) => {
  const start = Date.now();
  try {
    const payload = req.body || {};
    const imgB64 = payload.image_data;
    const name = payload.file_name || 'unknown';

    if (!imgB64) {
      return res.status(400).json({ detail: 'No image_data provided' });
    }

    // Decode and immediately re-encode to mirror original behavior
    const raw = Buffer.from(imgB64, 'base64');
    const resp = { image_data: b64encode(raw), file_name: name };

    const end = Date.now();
    console.log(`[uploader] Upload took ${(end - start).toFixed(2)} ms for file '${name}'`);
    return res.json(resp);
  } catch (e) {
    return res.status(500).json({ detail: String(e && e.message ? e.message : e) });
  }
});

function serve(port = 50060) {
  const host = '0.0.0.0';
  app.listen(port, host, () => {
    console.log(`Image Uploader Service listening on http://${host}:${port}`);
  });
}

if (require.main === module) {
  const port = process.env.PORT ? Number(process.env.PORT) : 50060;
  serve(port);
}

module.exports = { app, serve };
