import React, { useState, useEffect } from 'react';
import MagneticButton from './MagneticButton';
import { motion } from 'framer-motion';

interface NavbarProps {
  theme?: 'modern' | 'academic' | 'word';
  onToggleTheme?: (theme: 'modern' | 'academic' | 'word') => void;
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

  const themes = ['modern', 'academic', 'word'] as const;

  return (
    <nav className={`fixed top-0 left-0 right-0 z-40 transition-all duration-500 ease-out-expo ${isScrolled ? 'py-4' : 'py-8'}`}>
      <div className={`mx-auto px-6 md:px-10 lg:px-[72px] max-w-[1800px] flex justify-between items-center transition-all duration-500 ${isScrolled ? 'bg-surface/80 backdrop-blur-md rounded-full border border-grey-100 py-3 px-6 shadow-sm mx-4 md:mx-[72px]' : ''}`}>
        
        <div 
          className="flex items-center gap-2 cursor-pointer group interactive"
          onClick={() => {
            // 点击Logo切换首页和功能页
            const isAppPage = window.location.pathname.includes('/app');
            window.location.href = isAppPage ? '/' : '/app/';
          }}
          title="点击切换首页/功能页"
        >
           <MagneticButton className="w-10 h-10 rounded-full bg-on-surface text-surface flex items-center justify-center font-bold text-xs relative overflow-hidden">
             <span className="relative z-10 font-sans">S</span>
             <div className="absolute inset-0 bg-primary opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
           </MagneticButton>
           <span className="font-medium text-lg tracking-tight group-hover:text-primary transition-colors font-display">SpecSure | 澜瞳</span>
        </div>

        <div className="hidden md:flex items-center gap-2">
          {['Technology', 'Introduction', 'Research', 'Team'].map((item) => (
            <MagneticButton 
              key={item} 
              className="interactive px-4 py-2 text-sm font-medium text-on-surface-variant hover:text-on-surface transition-colors rounded-lg hover:bg-surface-container"
              onClick={() => {
                if (item === 'Research') {
                  window.location.href = '/app/';
                } else if (item === 'Team') {
                  // 滚动到团队轮播展示区域 (Our Squad / Meet the Minds)
                  const teamCarousel = document.getElementById('team-carousel');
                  if (teamCarousel) {
                    teamCarousel.scrollIntoView({ behavior: 'smooth' });
                  }
                } else if (item === 'Technology' ) {
                  // 滚动到对应区域
                  const sectionId = item.toLowerCase() + '-section';
                  const section = document.getElementById(sectionId);
                  if (section) {
                    section.scrollIntoView({ behavior: 'smooth' });
                  }
                }
                else if (item === 'Introduction') {
  const section = document.getElementById('applications-section');
  if (section) {
    section.scrollIntoView({ behavior: 'smooth' });
  }
}
              }}
            >
              {item}
            </MagneticButton>
          ))}
        </div>

        <div className="flex items-center gap-4">
          {/* Theme Toggle */}
          <div className="flex items-center bg-surface-container border border-grey-200 rounded-full p-1 relative interactive">
            {themes.map((t) => (
              <button
                key={t}
                onClick={() => onToggleTheme?.(t)}
                className={`relative px-4 py-1 text-xs font-medium z-10 capitalize transition-colors duration-300 ${theme === t ? 'text-primary' : 'text-on-surface-variant hover:text-on-surface'}`}
              >
                {theme === t && (
                  <motion.div
                    layoutId="theme-bubble"
                    className="absolute inset-0 bg-white rounded-full shadow-sm -z-10"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
                {t}
              </button>
            ))}
          </div>

          <MagneticButton className="interactive bg-on-surface text-surface px-6 py-2.5 rounded-full text-sm font-medium hover:bg-grey-900 transition-colors hidden sm:block">
            Contact
          </MagneticButton>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;