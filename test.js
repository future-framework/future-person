const sharp = require('sharp')
const person = require('./index')
const imageUrl = './ethan-hoover-325427-unsplash.jpg'

const run = async () => {
  const result = await person({ imageUrl });

  await sharp(imageUrl)
    .extract({
      left: Math.round(result.bbox.left),
      top: Math.round(result.bbox.top),
      width: Math.round(result.bbox.right - result.bbox.left),
      height: Math.round(result.bbox.bottom - result.bbox.top),
    })
    .toFile('extracted.png')
};

run()
