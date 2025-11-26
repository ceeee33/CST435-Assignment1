const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');
const fs = require('fs');
const path = require('path');

// Resolve pipeline.proto across both unified (copied into parent directory) and standalone (same directory) layouts
const protoCandidates = [
  path.join(__dirname, 'pipeline.proto'),
  path.join(__dirname, '..', 'pipeline.proto'),
  '/usr/src/app/pipeline.proto',
  '/app/pipeline.proto'
];
const PROTO_PATH = protoCandidates.find(p => fs.existsSync(p));
if (!PROTO_PATH) {
  console.error('pipeline.proto not found. Checked: ' + protoCandidates.join(', '));
  process.exit(1);
}
const UPLOAD_DIR = path.join(__dirname, '..', 'image_input');

const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
});
// The .proto uses package image_pipeline, so the loaded object will have image_pipeline.* services
const loaded = grpc.loadPackageDefinition(packageDefinition);
const pipelineProto = loaded.image_pipeline || loaded; // fallback if package name changes

function uploadImage(call, callback) {
  try {
    const { image_data, file_name } = call.request;
    const imageBuffer = Buffer.from(image_data);
    const name = file_name && file_name.length > 0 ? file_name : 'uploaded_image.jpg';

    if (!fs.existsSync(UPLOAD_DIR)) {
      fs.mkdirSync(UPLOAD_DIR, { recursive: true });
    }

    const imagePath = path.join(UPLOAD_DIR, name);
    fs.writeFile(imagePath, imageBuffer, (err) => {
      if (err) {
        return callback({
          code: grpc.status.INTERNAL,
          details: 'Failed to save image',
        });
      }
      console.log(`Image saved to ${imagePath}`);
      callback(null, { image_data: imageBuffer, file_name: name });
    });
  } catch (e) {
    return callback({
      code: grpc.status.INVALID_ARGUMENT,
      details: 'Invalid upload request',
    });
  }
}

function main() {
  const server = new grpc.Server({
    'grpc.max_receive_message_length': 64 * 1024 * 1024,
    'grpc.max_send_message_length': 64 * 1024 * 1024,
  });
  // Register only the ImageUploader service implemented here
  if (!pipelineProto.ImageUploader || !pipelineProto.ImageUploader.service) {
    console.error('ImageUploader service definition not found in loaded proto.');
    return;
  }
  server.addService(pipelineProto.ImageUploader.service, {
    Upload: uploadImage,
  });

  const port = 50040;
  server.bindAsync(`0.0.0.0:${port}`, grpc.ServerCredentials.createInsecure(), (err, port) => {
    if (err) {
      console.error('Failed to bind server:', err);
      return;
    }
    console.log(`Uploader service running on port ${port}`);
  });
}

main();
