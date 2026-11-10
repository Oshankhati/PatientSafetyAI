
import { useState } from 'react';
import '../styles/login.css';

export default function Login({ onLogin, onSignup }) {
  const [staffId, setStaffId] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!staffId.trim() || !password) {
      setError('Please enter your Staff ID and password.');
      return;
    }

    setError('');
    setLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          staff_id: staffId.trim(),
          password: password,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Login failed. Please try again.');
      }

      onLogin(data.staff, data.access_token);

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
    <main className="login-page">
      <div className="login-wrapper">

        <header className="login-brand">
          <div className="brand-icon">+</div>
          <div>
            <h1>PatientSafetyAI</h1>
            <p>AI-Powered Patient Safety &amp; Fall Monitoring</p>
          </div>
        </header>

        <section className="login-card">
          <span className="login-eyebrow">SECURE STAFF PORTAL</span>

          <h2>Welcome back</h2>
          <p className="login-subtitle">
            Sign in to access the patient monitoring system.
          </p>

          <form onSubmit={handleSubmit}>
            <div className="login-field">
              <label htmlFor="staffId">Staff ID</label>
              <input
                id="staffId"
                type="text"
                placeholder="e.g. NS-1042"
                value={staffId}
                onChange={(e) => setStaffId(e.target.value)}
                autoComplete="username"
                required
              />
            </div>

            <div className="login-field">
              <label htmlFor="password">Password</label>

              <div className="password-wrapper">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  required
                />

                <button
                  type="button"
                  className="password-toggle"
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? 'Hide' : 'Show'}
                </button>
              </div>
            </div>

            {error && (
              <div className="login-error" role="alert">
                <span>!</span>
                {error}
              </div>
            )}

            <button
              type="submit"
              className="login-submit"
              disabled={loading}
            >
              {loading ? 'Signing in...' : 'Sign in'}
            </button>
          </form>

          <div className="login-signup-link">
            Don't have an account?{' '}
            <button type="button" onClick={onSignup}>
              Create Account
            </button>
          </div>

          <div className="login-info">
            <div className="staff-icon">♧</div>
            <p>
              <strong>Nurse</strong> monitors and manages alerts.
              <strong> Admin</strong> can also configure system settings.
            </p>
          </div>
        </section>

        <footer className="login-footer">
          <span>Authorized hospital staff only</span>
          <span>Privacy-first · Skeleton output only</span>
        </footer>

      </div>
    </main>
  );
}
