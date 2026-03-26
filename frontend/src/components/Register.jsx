import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { User, Fingerprint, CheckCircle, AlertCircle, Loader, Phone, Mail, MapPin, Calendar } from 'lucide-react';
import CameraCapture from './CameraCapture';
import { API_URL } from '../config';
import './Register.css';

const MIN_PROCESSING_MS = 3000;
const PROCESSING_STAGES = [
  {
    title: 'Preprocessing captures',
    detail: 'Cropping, centering, and enhancing the clearest knuckle images.'
  },
  {
    title: 'Building the biometric template',
    detail: 'Extracting feature embeddings and combining them into one registration template.'
  },
  {
    title: 'Preparing registration assets',
    detail: 'Finalizing the processed knuckle captures and supporting registration files.'
  },
  {
    title: 'Saving the registration',
    detail: 'Writing the UID and biometric template into storage.'
  }
];

function wait(ms) {
  return new Promise(resolve => {
    window.setTimeout(resolve, ms);
  });
}

function Register() {
  const navigate = useNavigate();
  const [step, setStep] = useState(1); // 1: Details, 2: Camera, 3: Processing, 4: Success
  const [formData, setFormData] = useState({
    name: '',
    phone: '',
    email: '',
    dob: '',
    address: '',
    gender: ''
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [processingStage, setProcessingStage] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const progressIntervalRef = useRef(null);
  const progressTimeoutsRef = useRef([]);

  const clearProcessingTimers = () => {
    if (progressIntervalRef.current) {
      window.clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
    progressTimeoutsRef.current.forEach(timeoutId => window.clearTimeout(timeoutId));
    progressTimeoutsRef.current = [];
  };

  useEffect(() => () => clearProcessingTimers(), []);

  const startProcessingAnimation = () => {
    clearProcessingTimers();
    setProcessingStage(0);
    setProcessingProgress(8);

    progressIntervalRef.current = window.setInterval(() => {
      setProcessingProgress(prev => {
        if (prev >= 94) {
          return prev;
        }
        return Math.min(94, prev + 1.4);
      });
    }, 120);

    const stageTimes = [0, 800, 1650, 2400];
    progressTimeoutsRef.current = stageTimes.map((timeout, index) => (
      window.setTimeout(() => {
        setProcessingStage(index);
      }, timeout)
    ));
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleFormSubmit = (e) => {
    e.preventDefault();
    if (formData.name.trim().length >= 2 && formData.phone.length >= 10) {
      setStep(2);
    }
  };

  const handleImagesCapture = async (images) => {
    setStep(3);
    setError(null);
    setResult(null);
    startProcessingAnimation();

    try {
      const submitData = new FormData();
      submitData.append('name', formData.name.trim());
      submitData.append('phone', formData.phone.trim());
      submitData.append('email', formData.email.trim());
      submitData.append('dob', formData.dob);
      submitData.append('address', formData.address.trim());
      submitData.append('gender', formData.gender);

      // Convert base64 images to blobs and append
      for (let i = 0; i < images.length; i++) {
        const response = await fetch(images[i]);
        const blob = await response.blob();
        submitData.append('images', blob, `image_${i}.jpg`);
      }

      const requestPromise = axios.post(`${API_URL}/register`, submitData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      const [res] = await Promise.all([requestPromise, wait(MIN_PROCESSING_MS)]);

      clearProcessingTimers();
      setProcessingStage(PROCESSING_STAGES.length - 1);
      setProcessingProgress(100);
      await wait(250);
      setResult(res.data);
      setStep(4);
    } catch (err) {
      clearProcessingTimers();
      setProcessingProgress(0);
      setProcessingStage(0);
      setError(err.response?.data?.message || 'Registration failed. Please try again.');
      setStep(2);
    }
  };

  const resetRegistration = () => {
    clearProcessingTimers();
    setStep(1);
    setFormData({ name: '', phone: '', email: '', dob: '', address: '', gender: '' });
    setResult(null);
    setError(null);
    setProcessingStage(0);
    setProcessingProgress(0);
  };

  return (
    <div className="register-page">
      <div className="register-container">
        {/* Progress Steps */}
        <div className="progress-steps">
          <div className={`step ${step >= 1 ? 'active' : ''} ${step > 1 ? 'completed' : ''}`}>
            <div className="step-number">1</div>
            <span>Personal Details</span>
          </div>
          <div className="step-line"></div>
          <div className={`step ${step >= 2 ? 'active' : ''} ${step > 2 ? 'completed' : ''}`}>
            <div className="step-number">2</div>
            <span>Capture Images</span>
          </div>
          <div className="step-line"></div>
          <div className={`step ${step >= 3 ? 'active' : ''} ${step > 3 ? 'completed' : ''}`}>
            <div className="step-number">3</div>
            <span>Processing</span>
          </div>
          <div className="step-line"></div>
          <div className={`step ${step >= 4 ? 'active' : ''}`}>
            <div className="step-number">4</div>
            <span>Complete</span>
          </div>
        </div>

        {/* Step 1: Personal Details */}
        {step === 1 && (
          <div className="step-content">
            <div className="step-header">
              <User size={48} className="step-icon" />
              <h2>Personal Information</h2>
              <p>Enter your details to create a unique biometric ID</p>
            </div>
            <form onSubmit={handleFormSubmit} className="registration-form">
              <div className="form-row">
                <div className="input-group">
                  <User size={20} className="input-icon" />
                  <input
                    type="text"
                    name="name"
                    placeholder="Full Name *"
                    value={formData.name}
                    onChange={handleInputChange}
                    minLength={2}
                    required
                  />
                </div>
                <div className="input-group">
                  <Phone size={20} className="input-icon" />
                  <input
                    type="tel"
                    name="phone"
                    placeholder="Phone Number *"
                    value={formData.phone}
                    onChange={handleInputChange}
                    pattern="[0-9]{10}"
                    maxLength={10}
                    required
                  />
                </div>
              </div>

              <div className="form-row">
                <div className="input-group">
                  <Mail size={20} className="input-icon" />
                  <input
                    type="email"
                    name="email"
                    placeholder="Email Address"
                    value={formData.email}
                    onChange={handleInputChange}
                  />
                </div>
                <div className="input-group">
                  <Calendar size={20} className="input-icon" />
                  <input
                    type="date"
                    name="dob"
                    placeholder="Date of Birth"
                    value={formData.dob}
                    onChange={handleInputChange}
                  />
                </div>
              </div>

              <div className="form-row">
                <div className="input-group gender-group">
                  <label>Gender:</label>
                  <div className="radio-options">
                    <label className="radio-label">
                      <input
                        type="radio"
                        name="gender"
                        value="male"
                        checked={formData.gender === 'male'}
                        onChange={handleInputChange}
                      />
                      <span>Male</span>
                    </label>
                    <label className="radio-label">
                      <input
                        type="radio"
                        name="gender"
                        value="female"
                        checked={formData.gender === 'female'}
                        onChange={handleInputChange}
                      />
                      <span>Female</span>
                    </label>
                    <label className="radio-label">
                      <input
                        type="radio"
                        name="gender"
                        value="other"
                        checked={formData.gender === 'other'}
                        onChange={handleInputChange}
                      />
                      <span>Other</span>
                    </label>
                  </div>
                </div>
              </div>

              <div className="input-group full-width">
                <MapPin size={20} className="input-icon" />
                <input
                  type="text"
                  name="address"
                  placeholder="Address"
                  value={formData.address}
                  onChange={handleInputChange}
                />
              </div>

              <button type="submit" className="next-btn">
                Continue to Camera Capture
              </button>
            </form>
          </div>
        )}

        {/* Step 2: Camera Capture */}
        {step === 2 && (
          <div className="step-content">
            <div className="step-header">
              <Fingerprint size={48} className="step-icon" />
              <h2>Capture Knuckle Images</h2>
              <p>Place your finger knuckle inside the focus box and capture 5-20 images</p>
            </div>
            {error && (
              <div className="error-message">
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}
            <CameraCapture onCapture={handleImagesCapture} minImages={5} maxImages={20} />
            <button className="back-btn" onClick={() => setStep(1)}>
              ← Back to Name
            </button>
          </div>
        )}

        {/* Step 3: Processing */}
        {step === 3 && (
          <div className="step-content processing">
            <Loader size={64} className="spinner" />
            <h2>Processing Your Registration</h2>
            <p>{PROCESSING_STAGES[processingStage].detail}</p>
            <div className="processing-progress-card">
              <div className="processing-progress-meta">
                <span>{PROCESSING_STAGES[processingStage].title}</span>
                <span>{Math.round(processingProgress)}%</span>
              </div>
              <div className="processing-progress-bar">
                <div
                  className="processing-progress-fill"
                  style={{ width: `${processingProgress}%` }}
                ></div>
              </div>
            </div>
            <div className="processing-steps">
              {PROCESSING_STAGES.map((stage, index) => (
                <div
                  key={stage.title}
                  className={`proc-step ${index < processingStage ? 'completed' : ''} ${index === processingStage ? 'active' : ''}`}
                >
                  <span className="dot"></span>
                  <span>{stage.title}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Step 4: Success */}
        {step === 4 && result && (
          <div className="step-content success">
            <CheckCircle size={80} className="success-icon" />
            <h2>Registration Successful!</h2>
            <div className="result-card">
              <div className="result-item">
                <span className="label">Name</span>
                <span className="value">{formData.name}</span>
              </div>
              <div className="result-item">
                <span className="label">Phone</span>
                <span className="value">{formData.phone}</span>
              </div>
              <div className="result-item uid">
                <span className="label">Your Unique ID (UID)</span>
                <span className="value uid-value">{result.uid}</span>
              </div>
              {typeof result.quality === 'number' && (
                <div className="result-item">
                  <span className="label">Registration Quality</span>
                  <span className="value">{(result.quality * 100).toFixed(1)}%</span>
                </div>
              )}
              {typeof result.images_used === 'number' && (
                <div className="result-item">
                  <span className="label">Images Used</span>
                  <span className="value">{result.images_used}</span>
                </div>
              )}
              {result.message && (
                <p className="uid-note">{result.message}</p>
              )}
              <p className="uid-note">
                ⚠️ Save this 12-digit UID! You'll need it for verification.
              </p>
            </div>
            <div className="action-buttons">
              <button className="primary-btn" onClick={() => navigate('/verify')}>
                Go to Verification
              </button>
              <button className="secondary-btn" onClick={resetRegistration}>
                Register Another
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Register;
