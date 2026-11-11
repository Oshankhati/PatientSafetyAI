
import { useState } from 'react';
import '../styles/signup.css';

export default function Signup({ onSignup, onBackToLogin }) {
  const [form, setForm] = useState({
    fullName: '',
    staffId: '',
    email: '',
    department: '',
    password: '',
    confirmPassword: '',
  });

  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (Object.values(form).some(value => !value.trim())) {
      setError('Please fill in all fields.');
      return;
    }

    if (form.password.length < 8) {
      setError('Password must contain at least 8 characters.');
      return;
    }

    if (form.password !== form.confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    setError('');
    setLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          full_name: form.fullName.trim(),
          staff_id: form.staffId.trim(),
          email: form.email.trim(),
          department: form.department.trim(),
          password: form.password,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Unable to create account.');
      }

      alert('Staff account created successfully! Please sign in.');
      onSignup?.();

    } catch (err) {
      setError(
        err.message === 'Failed to fetch'
          ? 'Cannot connect to the server. Please ensure the backend is running.'
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="signup-page">
      <div className="signup-card">
        <div className="signup-brand">
          <div className="signup-logo">+</div>
          <h1>PatientSafetyAI</h1>
          <p>Intelligent Patient Monitoring System</p>
        </div>

        <div className="signup-heading">
          <h2>Create Staff Account</h2>
          <p>Register to access the patient monitoring dashboard</p>
        </div>

        <form onSubmit={handleSubmit}>
          <label>Full Name</label>
          <input
            name="fullName"
            value={form.fullName}
            onChange={handleChange}
            placeholder="Enter your full name"
            required
          />

          <label>Staff ID</label>
          <input
            name="staffId"
            value={form.staffId}
            onChange={handleChange}
            placeholder="Enter your staff ID"
            required
          />

          <label>Email Address</label>
          <input
            type="email"
            name="email"
            value={form.email}
            onChange={handleChange}
            placeholder="Enter your email address"
            required
          />

          <label>Role</label>
          <input value="Nurse" readOnly />

          <label>Department</label>
          <input
            name="department"
            value={form.department}
            onChange={handleChange}
            placeholder="e.g. Emergency, ICU, General Ward"
            required
          />

          <label>Password</label>
          <input
            type="password"
            name="password"
            value={form.password}
            onChange={handleChange}
            placeholder="Create a password"
            autoComplete="new-password"
            required
          />

          <label>Confirm Password</label>
          <input
            type="password"
            name="confirmPassword"
            value={form.confirmPassword}
            onChange={handleChange}
            placeholder="Confirm your password"
            autoComplete="new-password"
            required
          />

          {error && (
            <div className="signup-error" role="alert">
              {error}
            </div>
          )}

          <button
            type="submit"
            className="signup-button"
            disabled={loading}
          >
            {loading ? 'Creating Account...' : 'Create Account'}
          </button>
        </form>

        <div className="signup-footer">
          Already have an account?{' '}
          <button type="button" onClick={onBackToLogin}>
            Sign In
          </button>
        </div>

        <p className="signup-note">
          Authorized hospital staff access only
        </p>
      </div>
    </div>
  );
}
