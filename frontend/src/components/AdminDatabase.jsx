import { useState } from 'react';
import axios from 'axios';
import { 
  Database, 
  Lock, 
  Unlock,
  User,
  Phone,
  Mail,
  Trash2,
  RefreshCw,
  Eye,
  EyeOff,
  Users,
  Shield
} from 'lucide-react';
import { API_URL } from '../config';
import './AdminDatabase.css';

function AdminDatabase() {
  const [isUnlocked, setIsUnlocked] = useState(false);
  const [code, setCode] = useState('');
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showCode, setShowCode] = useState(false);

  const handleUnlock = async (e) => {
    e.preventDefault();
    const normalizedCode = code.trim();
    if (!normalizedCode) {
      setError('Enter access code');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await axios.post(`${API_URL}/admin/users`, { code: normalizedCode });

      if (res.data.status === 'success') {
        setIsUnlocked(true);
        setUsers(res.data.users);
        setCode(normalizedCode);
      } else {
        setError(res.data.message || 'Failed to authenticate');
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Invalid access code');
      setIsUnlocked(false);
    } finally {
      setLoading(false);
    }
  };

  const refreshData = async () => {
    const normalizedCode = code.trim();
    setLoading(true);
    try {
      const res = await axios.post(`${API_URL}/admin/users`, { code: normalizedCode });
      if (res.data.status === 'success') {
        setUsers(res.data.users);
      }
    } catch {
      setError('Failed to refresh data');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (uid, name) => {
    if (!window.confirm(`Are you sure you want to delete ${name} (${uid})?`)) {
      return;
    }

    try {
      const res = await axios.post(`${API_URL}/admin/delete/${uid}`, { code: code.trim() });
      if (res.data.status === 'success') {
        setUsers(users.filter(u => u.uid !== uid));
      }
    } catch {
      alert('Failed to delete user');
    }
  };

  const lockDatabase = () => {
    setIsUnlocked(false);
    setCode('');
    setUsers([]);
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A';
    try {
      return new Date(dateStr).toLocaleDateString('en-IN', {
        day: '2-digit',
        month: 'short',
        year: 'numeric'
      });
    } catch {
      return dateStr;
    }
  };

  // Locked State - Show Login
  if (!isUnlocked) {
    return (
      <div className="admin-section">
        <div className="admin-lock-container">
          <div className="lock-icon">
            <Shield size={48} />
          </div>
          <h3>Admin Database Access</h3>
          <p>Enter access code to view registered users</p>
          
          <form onSubmit={handleUnlock} className="unlock-form">
            <div className="code-input-wrapper">
              <Lock size={18} className="input-icon" />
              <input
                type={showCode ? 'text' : 'password'}
                placeholder="Enter access code"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                autoComplete="off"
              />
              <button 
                type="button" 
                className="toggle-visibility"
                onClick={() => setShowCode(!showCode)}
              >
                {showCode ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
            
            {error && <p className="error-msg">{error}</p>}
            
            <button type="submit" className="unlock-btn" disabled={loading || !code.trim()}>
              {loading ? <RefreshCw size={18} className="spin" /> : <Unlock size={18} />}
              <span>{loading ? 'Verifying...' : 'Unlock Database'}</span>
            </button>
          </form>
        </div>
      </div>
    );
  }

  // Unlocked State - Show Database
  return (
    <div className="admin-section unlocked">
      <div className="admin-header">
        <div className="header-left">
          <Database size={24} />
          <h3>Registered Users Database</h3>
          <span className="user-count">{users.length} users</span>
        </div>
        <div className="header-actions">
          <button className="refresh-btn" onClick={refreshData} disabled={loading}>
            <RefreshCw size={18} className={loading ? 'spin' : ''} />
          </button>
          <button className="lock-btn" onClick={lockDatabase}>
            <Lock size={18} />
            <span>Lock</span>
          </button>
        </div>
      </div>

      {users.length === 0 ? (
        <div className="no-users">
          <Users size={48} />
          <p>No registered users in database</p>
        </div>
      ) : (
        <div className="users-table-container">
          <table className="users-table">
            <thead>
              <tr>
                <th>UID</th>
                <th>Name</th>
                <th>Phone</th>
                <th>Email</th>
                <th>DOB</th>
                <th>Gender</th>
                <th>Registered</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map((user) => (
                <tr key={user.uid}>
                  <td className="uid-cell">
                    <code>{user.uid}</code>
                  </td>
                  <td>
                    <div className="user-name">
                      <User size={14} />
                      <span>{user.name}</span>
                    </div>
                  </td>
                  <td>
                    <div className="cell-with-icon">
                      <Phone size={14} />
                      <span>{user.phone || 'N/A'}</span>
                    </div>
                  </td>
                  <td>
                    <div className="cell-with-icon">
                      <Mail size={14} />
                      <span>{user.email || 'N/A'}</span>
                    </div>
                  </td>
                  <td>{user.dob || 'N/A'}</td>
                  <td>{user.gender || 'N/A'}</td>
                  <td className="date-cell">{formatDate(user.registered_at)}</td>
                  <td>
                    <button 
                      className="delete-btn"
                      onClick={() => handleDelete(user.uid, user.name)}
                      title="Delete user"
                    >
                      <Trash2 size={16} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default AdminDatabase;
