import React, { useEffect, useState } from 'react';
import { motion, useSpring, useMotionValue } from 'framer-motion';

const CustomCursor: React.FC = () => {
  const [isHovering, setIsHovering] = useState(false);
  const [isVisible, setIsVisible] = useState(true);
  
  const cursorX = useMotionValue(-100);
  const cursorY = useMotionValue(-100);

  // Stiffer spring for the main dot
  const springConfig = { damping: 25, stiffness: 400, mass: 0.5 };
  const smoothX = useSpring(cursorX, springConfig);
  const smoothY = useSpring(cursorY, springConfig);

  // Trailing ring with more lag
  const trailConfig = { damping: 30, stiffness: 200, mass: 0.8 };
  const trailX = useSpring(cursorX, trailConfig);
  const trailY = useSpring(cursorY, trailConfig);

  useEffect(() => {
    // Check if we are in 'word' theme
    const checkTheme = () => {
       const app = document.querySelector('[data-theme="word"]');
       if (app) setIsVisible(false);
       else setIsVisible(true);
    };
    
    // Initial check
    checkTheme();
    
    // Observer for theme changes
    const observer = new MutationObserver(checkTheme);
    const appElement = document.getElementById('root')?.parentElement; // Body or html usually has data-theme in App.tsx div
    if(appElement) observer.observe(document.body, { attributes: true, subtree: true });

    const moveCursor = (e: MouseEvent) => {
      cursorX.set(e.clientX); 
      cursorY.set(e.clientY);
    };

    const handleMouseOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (
        target.tagName === 'BUTTON' || 
        target.tagName === 'A' || 
        target.closest('.interactive') ||
        target.closest('[role="button"]') ||
        target.tagName === 'H1'
      ) {
        setIsHovering(true);
      } else {
        setIsHovering(false);
      }
    };

    window.addEventListener('mousemove', moveCursor);
    window.addEventListener('mouseover', handleMouseOver);

    return () => {
      window.removeEventListener('mousemove', moveCursor);
      window.removeEventListener('mouseover', handleMouseOver);
      observer.disconnect();
    };
  }, [cursorX, cursorY]);

  if (!isVisible) return null;

  return (
    <>
        {/* Main Dot */}
        <motion.div
        className="fixed top-0 left-0 pointer-events-none z-[9999] rounded-full bg-primary mix-blend-difference"
        style={{
            x: smoothX,
            y: smoothY,
            translateX: '-50%',
            translateY: '-50%',
        }}
        animate={{
            width: isHovering ? 20 : 12,
            height: isHovering ? 20 : 12,
            backgroundColor: isHovering ? '#3077f3' : '#adf6bd'
        }}
        transition={{ duration: 0.2 }}
        />
        
        {/* Trailing Ring */}
        <motion.div
        className="fixed top-0 left-0 pointer-events-none z-[9998] rounded-full border border-primary/50 mix-blend-difference"
        style={{
            x: trailX,
            y: trailY,
            translateX: '-50%',
            translateY: '-50%',
        }}
        animate={{
            width: isHovering ? 50 : 30,
            height: isHovering ? 50 : 30,
            borderColor: isHovering ? '#3077f3' : '#adf6bd',
            opacity: isHovering ? 0.8 : 0.4
        }}
        transition={{ duration: 0.3 }}
        />
    </>
  );
};

export default CustomCursor;