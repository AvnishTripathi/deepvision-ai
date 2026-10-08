import React from 'react';
import { Link } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { Brain, User, Menu, X, LogOut, LayoutDashboard } from 'lucide-react';

const Navbar = () => {
  const { user, logout } = useAuthStore();
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <nav className="sticky top-0 z-50 glass border-b border-slate-700/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center gap-2">
              <Brain className="h-8 w-8 text-primary-500" />
              <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary-400 to-primary-600">
                DeepFaceGuard
              </span>
            </Link>
          </div>
          
          {/* Desktop Menu */}
          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-4">
              <Link to="/" className="hover:bg-slate-700/50 px-3 py-2 rounded-md text-sm font-medium transition-colors">Home</Link>
              <Link to="/detect" className="hover:bg-slate-700/50 px-3 py-2 rounded-md text-sm font-medium transition-colors">Detect</Link>
              <Link to="/dashboard" className="hover:bg-slate-700/50 px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-1.5">
                <LayoutDashboard className="w-4 h-4 text-primary-400" /> Dashboard
              </Link>
              <a 
                href="http://localhost:8000/docs" 
                target="_blank" 
                rel="noreferrer" 
                className="hover:bg-slate-700/50 px-3 py-2 rounded-md text-sm font-medium transition-colors text-slate-300 hover:text-white"
              >
                API Docs
              </a>
              
              {user ? (
                <>
                  <span className="text-slate-400 text-sm px-2">Hi, {user.name || user.email.split('@')[0]}</span>
                  <button onClick={logout} className="hover:bg-danger-500/20 text-danger-500 px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-1">
                    <LogOut className="w-4 h-4" /> Logout
                  </button>
                </>
              ) : (
                <>
                  <Link to="/login" className="hover:bg-slate-700/50 px-3 py-2 rounded-md text-sm font-medium transition-colors">Login</Link>
                  <Link to="/register" className="bg-primary-600 hover:bg-primary-700 px-4 py-2 rounded-md text-sm font-medium transition-colors">Register</Link>
                </>
              )}
            </div>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button onClick={() => setIsOpen(!isOpen)} className="inline-flex items-center justify-center p-2 rounded-md text-slate-400 hover:text-white hover:bg-slate-700 focus:outline-none">
              {isOpen ? <X className="block h-6 w-6" /> : <Menu className="block h-6 w-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
        <div className="md:hidden glass border-t border-slate-700/50 absolute w-full">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            <Link to="/" onClick={() => setIsOpen(false)} className="block px-3 py-2 rounded-md text-base font-medium hover:bg-slate-700/50">Home</Link>
            <Link to="/detect" onClick={() => setIsOpen(false)} className="block px-3 py-2 rounded-md text-base font-medium hover:bg-slate-700/50">Detect</Link>
            <Link to="/dashboard" onClick={() => setIsOpen(false)} className="block px-3 py-2 rounded-md text-base font-medium hover:bg-slate-700/50 flex items-center gap-2">
              <LayoutDashboard className="w-4 h-4 text-primary-400" /> Dashboard
            </Link>
            <a 
              href="http://localhost:8000/docs" 
              target="_blank" 
              rel="noreferrer" 
              onClick={() => setIsOpen(false)}
              className="block px-3 py-2 rounded-md text-base font-medium hover:bg-slate-700/50 text-slate-300"
            >
              API Docs
            </a>
            {user ? (
              <>
                <button onClick={() => { logout(); setIsOpen(false); }} className="block w-full text-left px-3 py-2 rounded-md text-base font-medium text-danger-500 hover:bg-danger-500/20">Logout</button>
              </>
            ) : (
              <>
                <Link to="/login" onClick={() => setIsOpen(false)} className="block px-3 py-2 rounded-md text-base font-medium hover:bg-slate-700/50">Login</Link>
                <Link to="/register" onClick={() => setIsOpen(false)} className="block px-3 py-2 rounded-md text-base font-medium bg-primary-600 hover:bg-primary-700">Register</Link>
              </>
            )}
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
