import { useRef, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
import { Camera, X, Check } from 'lucide-react';
import './CameraCapture.css';
import { CAMERA_VIDEO_CONSTRAINTS, KNUCKLE_FOCUS_BOX, captureCenteredKnuckleImage } from '../utils/knuckleCapture';

function CameraCapture({ onCapture, maxImages = 20, minImages = 5 }) {
  const webcamRef = useRef(null);
  const [images, setImages] = useState([]);
  const [isCameraReady, setIsCameraReady] = useState(false);

  const capture = useCallback(() => {
    if (images.length >= maxImages) return;
    
    const video = webcamRef.current?.video;
    if (!video) return;
    const capturedImage = captureCenteredKnuckleImage(video, KNUCKLE_FOCUS_BOX);
    if (!capturedImage) return;
    setImages(prev => [...prev, capturedImage]);
  }, [images.length, maxImages]);

  const removeImage = (index) => {
    setImages(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = () => {
    if (images.length >= minImages) {
      onCapture(images);
    }
  };

  return (
    <div className="camera-capture">
      <div className="camera-container">
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          videoConstraints={CAMERA_VIDEO_CONSTRAINTS}
          onUserMedia={() => setIsCameraReady(true)}
          className="webcam"
        />
        
        {/* Focus Box Overlay */}
        <div className="focus-overlay">
          <div
            className="focus-box"
            style={{
              width: `${KNUCKLE_FOCUS_BOX.width}px`,
              height: `${KNUCKLE_FOCUS_BOX.height}px`,
              transform: `translateY(${KNUCKLE_FOCUS_BOX.offsetY || 0}px)`,
            }}
          >
            <span className="focus-label">Place Knuckle Here</span>
            <div className="corner tl"></div>
            <div className="corner tr"></div>
            <div className="corner bl"></div>
            <div className="corner br"></div>
          </div>
        </div>
      </div>

      <div className="camera-actions">
        <button 
          className="capture-btn"
          onClick={capture}
          disabled={!isCameraReady || images.length >= maxImages}
        >
          <Camera size={24} />
          <span>Capture ({images.length}/{maxImages})</span>
        </button>
      </div>

      {/* Image Counter & Progress */}
      <div className="capture-progress">
        <div className="progress-bar">
          <div 
            className="progress-fill"
            style={{ width: `${(images.length / minImages) * 100}%` }}
          ></div>
        </div>
        <span className="progress-text">
          {images.length < minImages 
            ? `Capture at least ${minImages - images.length} more images`
            : `Ready to submit! (${images.length} images captured)`
          }
        </span>
      </div>

      {/* Captured Images Preview */}
      {images.length > 0 && (
        <div className="captured-images">
          <h3>Captured Images</h3>
          <div className="images-grid">
            {images.map((img, index) => (
              <div key={index} className="image-preview">
                <img src={img} alt={`Capture ${index + 1}`} />
                <button 
                  className="remove-btn"
                  onClick={() => removeImage(index)}
                >
                  <X size={14} />
                </button>
                <span className="image-number">{index + 1}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Submit Button */}
      <button 
        className={`submit-btn ${images.length >= minImages ? 'ready' : 'disabled'}`}
        onClick={handleSubmit}
        disabled={images.length < minImages}
      >
        <Check size={20} />
        <span>Submit {images.length} Images</span>
      </button>
    </div>
  );
}

export default CameraCapture;
