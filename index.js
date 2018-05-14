const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')
tf.setBackend('tensorflow')
global.fetch = require('node-fetch')
// const yolo = require('tfjs-yolo-tiny')
const sharp = require('sharp')
const imageUrl = './person-and-dog.jpg'
const MOBILENET_MODEL_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
const IMAGE_SIZE = 224
const imagenetClasses = require('./imagenetClasses')


const run = async () => {
  console.log('Downloading model...')
  const model = await tf.loadModel(MOBILENET_MODEL_PATH);
  console.log('Got model.')

  model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  const layer = model.getLayer('conv_pw_13_relu');
  const freezed = tf.model({inputs: model.inputs, outputs: layer.output});
  // // const model = await yolo.downloadModel()
  const img = await sharp(imageUrl)
  await img.resize(IMAGE_SIZE, IMAGE_SIZE)
  const imgBuffer = await img.toBuffer()

  tf.tidy(() => {
    const data = Float32Array.from(imgBuffer)
    const input = new Float32Array(IMAGE_SIZE * IMAGE_SIZE * 3)

  //   for (let i = 0, len = data.length; i < len; i += 4) {
  //     input[i / 4] = data[i + 3] / 255
  //   }
  // //
    const img = tf.tensor3d(input, [IMAGE_SIZE, IMAGE_SIZE, 3], 'float32')
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (IMAGE_SIZE / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (IMAGE_SIZE / 2);
    const finalInput = img.slice([beginHeight, beginWidth, 0], [IMAGE_SIZE, IMAGE_SIZE, 3]);
    const batchedImage = finalInput.expandDims(0)
    const finalImage = batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1))
    // console.log(batchedImage)
    let pred = model.predict(finalImage);
    let cls = pred.argMax().buffer().values[0];
    console.log(imagenetClasses[cls]);
    console.log(cls)
    // console.log(pred)
  });
  //   // let cls = pred.argMax().buffer().values[0];
  //   // if(last_update>time_between_predictions && last_prediction!=IMAGENET_CLASSES[cls]){
  //   //   status(IMAGENET_CLASSES[cls]);
  //   //   last_update = 0;
  //   //   last_prediction = IMAGENET_CLASSES[cls];
  //   // }
  //   // var t1 = performance.now();
  //   // last_update = last_update + (t1-t0);
  // });
  // console.log(input)
  // console.log(finalInput)
  // console.log(typeof finalInput)
  // console.log(finalInput.toPixels())

  // console.log(model)
  // model.predict(tf.fill([1, 416, 416, 3], 0));
  // const result = await yolo.yolo(finalInput, model)
  // console.log(result)
  // let image;
  // let index = 0;
  //
  // const recordBytes = 416 * 416;
  // const downsize = 1.0 / 255.0;
  //
  // while (index < buffer.byteLength) {
  //   const array = new Float32Array(recordBytes);
  //   for (let i = 0; i < recordBytes; i++) {
  //     array[i] = buffer.readUInt8(index++) * downsize;
  //   }
  //   image = array;
  // }
  // console.log(image)
  // const data = Float32Array.from(buffer)
  // console.log(buffer)
  // const pixels = tf.fromPixels(data)
  // console.log(pixels)
};

run()
