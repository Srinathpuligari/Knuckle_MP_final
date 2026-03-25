export const CAMERA_VIDEO_CONSTRAINTS = {
  width: { ideal: 1280 },
  height: { ideal: 720 },
  facingMode: 'user',
};

export const KNUCKLE_FOCUS_BOX = {
  width: 172,
  height: 128,
  offsetY: 36,
};

const PREVIEW_FRAME = {
  width: 640,
  height: 480,
};

const CAPTURE_OUTPUT = {
  width: 256,
  height: 192,
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export function captureCenteredKnuckleImage(video, focusBox = KNUCKLE_FOCUS_BOX) {
  if (!video) {
    return null;
  }

  const sourceWidth = video.videoWidth || PREVIEW_FRAME.width;
  const sourceHeight = video.videoHeight || PREVIEW_FRAME.height;
  const displayWidth = video.clientWidth || PREVIEW_FRAME.width;
  const displayHeight = video.clientHeight || PREVIEW_FRAME.height;
  const scaleX = sourceWidth / displayWidth;
  const scaleY = sourceHeight / displayHeight;

  const cropWidth = Math.min(sourceWidth, Math.round(focusBox.width * scaleX));
  const cropHeight = Math.min(sourceHeight, Math.round(focusBox.height * scaleY));

  const centerX = sourceWidth / 2;
  const centerY = sourceHeight / 2 + Math.round((focusBox.offsetY || 0) * scaleY);
  const cropX = clamp(Math.round(centerX - cropWidth / 2), 0, Math.max(sourceWidth - cropWidth, 0));
  const cropY = clamp(Math.round(centerY - cropHeight / 2), 0, Math.max(sourceHeight - cropHeight, 0));

  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) {
    return null;
  }

  canvas.width = CAPTURE_OUTPUT.width;
  canvas.height = CAPTURE_OUTPUT.height;
  context.drawImage(
    video,
    cropX,
    cropY,
    cropWidth,
    cropHeight,
    0,
    0,
    CAPTURE_OUTPUT.width,
    CAPTURE_OUTPUT.height,
  );

  return canvas.toDataURL('image/jpeg', 0.95);
}
