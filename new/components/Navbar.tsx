import React, { useState, useEffect } from 'react';
import MagneticButton from './MagneticButton';
import { motion } from 'framer-motion';

interface NavbarProps {
  theme?: 'modern' | 'academic';
  onToggleTheme?: () => void;
}

const Navbar: React.FC<NavbarProps> = ({ theme = 'modern', onToggleTheme }) => {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`fixed top-0 left-0 right-0 z-40 transition-all duration-500 ease-out-expo ${isScrolled ? 'py-4' : 'py-8'}`}>
      <div className={`mx-auto px-6 md:px-10 lg:px-[72px] max-w-[1800px] flex justify-between items-center transition-all duration-500 ${isScrolled ? 'bg-surface/80 backdrop-blur-md rounded-full border border-grey-100 py-3 px-6 shadow-sm mx-4 md:mx-[72px]' : ''}`}>
        
        <div className="flex items-center gap-2 cursor-pointer group">
           <MagneticButton className="w-10 h-10 rounded-full bg-on-surface text-surface flex items-center justify-center font-bold text-xs relative overflow-hidden">
             <span className="relative z-10 font-sans">S</span>
             <div className="absolute inset-0 bg-primary opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
           </MagneticButton>
           <span className="font-medium text-lg tracking-tight group-hover:text-primary transition-colors font-display">SpecSure | 澜瞳</span>
        </div>

        <div className="hidden md:flex items-center gap-2">
          {['Technology', 'Applications', 'Research', 'Team'].map((item) => (
            <MagneticButton key={item} className="px-4 py-2 text-sm font-medium text-on-surface-variant hover:text-on-surface transition-colors rounded-lg hover:bg-surface-container">
              {item}
            </MagneticButton>
          ))}
        </div>

        <div className="flex items-center gap-4">
          {/* Theme Toggle */}
          <button 
            onClick={onToggleTheme}
            className="relative h-9 rounded-full bg-surface-container border border-grey-200 p-1 flex items-center cursor-pointer overflow-hidden"
            title="Switch Mode"
          >
            <motion.div 
              className="absolute top-1 bottom-1 w-[45%] bg-white shadow-sm rounded-full z-0"
              animate={{ left: theme === 'modern' ? '4px' : 'calc(100% - 4px - 45%)' }}
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
            />
            <span className={`relative z-10 px-3 text-xs font-medium transition-colors duration-300 ${theme === 'modern' ? 'text-primary' : 'text-on-surface-variant'}`}>
              Modern
            </span>
            <span className={`relative z-10 px-3 text-xs font-medium transition-colors duration-300 ${theme === 'academic' ? 'text-primary font-serif italic' : 'text-on-surface-variant'}`}>
              Academic
            </span>
          </button>

          <MagneticButton className="bg-on-surface text-surface px-6 py-2.5 rounded-full text-sm font-medium hover:bg-grey-900 transition-colors hidden sm:block">
            Contact
          </MagneticButton>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;