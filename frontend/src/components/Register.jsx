import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { User, Fingerprint, CheckCircle, AlertCircle, Loader, Phone, Mail, MapPin, Calendar } from 'lucide-react';
import CameraCapture from './CameraCapture';
import { API_URL } from '../config';
import './Register.css';

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

      const res = await axios.post(`${API_URL}/register`, submitData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setResult(res.data);
      setStep(4);
    } catch (err) {
      setError(err.response?.data?.message || 'Registration failed. Please try again.');
      setStep(2);
    }
  };

  const resetRegistration = () => {
    setStep(1);
    setFormData({ name: '', phone: '', email: '', dob: '', address: '', gender: '' });
    setResult(null);
    setError(null);
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
            <p>Preprocessing captures, creating your knuckle template, and saving the enrollment...</p>
            <div className="processing-steps">
              <div className="proc-step active">
                <span className="dot"></span>
                <span>Preprocessing images</span>
              </div>
              <div className="proc-step">
                <span className="dot"></span>
                <span>Building knuckle template</span>
              </div>
              <div className="proc-step">
                <span className="dot"></span>
                <span>Calibrating match thresholds</span>
              </div>
              <div className="proc-step">
                <span className="dot"></span>
                <span>Saving to database</span>
              </div>
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
