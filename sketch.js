import * as posenet from '@tensorflow-models/posenet';

async function estimateMultiplePosesOnImage(imageElement) {
  const net = await posenet.load({
      architecture: 'ResNet50',
      outputStride: 32,
      inputResolution: {width: 423, height: 423},
      quantBytes: 2
  });

  // estimate poses
  const poses = await net.estimateMultiplePoses(imageElement, {
        flipHorizontal: false,
        maxDetections: 2,
        scoreThreshold: 0.6,
        nmsRadius: 20});

  return poses;
}

const imageElement = document.getElementById('people');

const poses = estimateMultiplePosesOnImage(imageElement);

console.log(poses);