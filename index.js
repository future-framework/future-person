global.tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')
tf.setBackend('tensorflow')

const yolo = require('tfjs-yolo-tiny')
const sharp = require('sharp')
const f = require('future-framework')
const _ = require('lodash')

const NETWORK_IMAGE_SIZE = 416;

module.exports = f(async ({ imageUrl }) => {
  const model = await yolo.downloadModel()

  const img = await sharp(imageUrl)
  const metadata = await img.metadata()

  await img.resize(NETWORK_IMAGE_SIZE, NETWORK_IMAGE_SIZE)
  const imgBuffer = await img.raw().toBuffer()
  const data = Float32Array.from(imgBuffer)

  const finalInput = tf.tensor4d(data, [1, NETWORK_IMAGE_SIZE, NETWORK_IMAGE_SIZE, 3], 'float32')

  const result = await yolo.yolo(finalInput, model)
  const people = _.filter(result, { className: 'person' });
  const person = _.first(people);

  if (!person) return {};

  const horizontalScalingRatio = metadata.width / NETWORK_IMAGE_SIZE;
  const verticalScalingRatio = metadata.height / NETWORK_IMAGE_SIZE;

  return {
    probability: person.classProb,
    bbox: {
      left: person.left * horizontalScalingRatio,
      right: person.right * horizontalScalingRatio,
      top: person.top * verticalScalingRatio,
      bottom: person.bottom * verticalScalingRatio,
    },
  }
}, {
  name: 'person',
  input: {
    imageUrl: 'String',
  },
  output: {
    probability: 'Float',
    bbox: 'Bbox',
  },
})
