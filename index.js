global.tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')
tf.setBackend('tensorflow')
const yolo = require('tfjs-yolo-tiny')
const sharp = require('sharp')
const imageUrl = './jacqueline-day-619814-unsplash.jpg'
const cbGetPixels = require("get-pixels")
const util = require('util')
const getPixels = util.promisify(cbGetPixels)

const run = async () => {
  console.log('Downloading model...')
  const model = await yolo.downloadModel()
  console.log('Got model.')

  // const img = await getPixels('dude.png');
  // console.log(img)
  // const data = img.data;
  const img = await sharp(imageUrl)
  await img.resize(416, 416)
  // // await img.toFile('dude.png')
  const imgBuffer = await img.raw().toBuffer()
  const data = Float32Array.from(imgBuffer)
  console.log(data);

  // for (let i = 0, len = data.length; i < len; i += 3) {
  //   input[i / 3] = data[i + 2] / 255
  // }
  // console.log(input)
  const finalInput = tf.tensor4d(data, [1, 416, 416, 3], 'float32')
  // console.log(finalInput)
  console.log(finalInput)
  // console.log(finalInput.toPixels())

  const result = await yolo.yolo(finalInput, model)
  console.log(result)
  const first = result[0];

  const img2 = await sharp(imageUrl)
  await img2.resize(416, 416)
  await img2.extract({
    left: Math.round(first.left),
    top: Math.round(first.top),
    width: Math.round(first.right - first.left),
    height: Math.round(first.bottom - first.top),
  }).toFile('extracted');


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
