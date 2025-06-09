/**
 * Extracts landmarks from MediaPipe results into a single flat array.
 * The output array will have a fixed size of 1662.
 */
export function extractKeypoints(results) {
  const pose = results.poseLandmarks
    ? results.poseLandmarks.map(res => [res.x, res.y, res.z, res.visibility]).flat()
    : new Array(33 * 4).fill(0);

  const face = results.faceLandmarks
    ? results.faceLandmarks.map(res => [res.x, res.y, res.z]).flat()
    : new Array(468 * 3).fill(0);

  const lh = results.leftHandLandmarks
    ? results.leftHandLandmarks.map(res => [res.x, res.y, res.z]).flat()
    : new Array(21 * 3).fill(0);

  const rh = results.rightHandLandmarks
    ? results.rightHandLandmarks.map(res => [res.x, res.y, res.z]).flat()
    : new Array(21 * 3).fill(0);

  return [...pose, ...face, ...lh, ...rh];
}