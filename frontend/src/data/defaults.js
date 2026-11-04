// Frontend-only defaults. No patient, room, event or risk data lives here.
export const DEFAULT_SETTINGS = {
  ai: { yolo: '0.50', medium: '0.40', high: '0.70', fall: '0.90', seqLen: '20', stride: '5', minFrames: '3' },
  camera: { source: 'RTSP stream', cameraId: '', resolution: '1920 × 1080', fps: '' },
};
