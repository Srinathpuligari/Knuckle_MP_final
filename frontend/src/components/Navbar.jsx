import { Link, useLocation } from 'react-router-dom';
import { UserPlus, Search, Home, Fingerprint } from 'lucide-react';
import './Navbar.css';

function Navbar() {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <Fingerprint size={32} />
        <span>Knuckle Biometric System</span>
      </div>
      <div className="navbar-links">
        <Link 
          to="/" 
          className={location.pathname === '/' ? 'active' : ''}
        >
          <Home size={20} />
          <span>Home</span>
        </Link>
        <Link 
          to="/register" 
          className={location.pathname === '/register' ? 'active' : ''}
        >
          <UserPlus size={20} />
          <span>Register</span>
        </Link>
        <Link 
          to="/verify" 
          className={location.pathname === '/verify' ? 'active' : ''}
        >
          <Search size={20} />
          <span>Verify</span>
        </Link>
      </div>
    </nav>
  );
}

export default Navbar;
