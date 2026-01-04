import React, { useRef } from 'react';
import { motion, useScroll, useTransform, useSpring } from 'framer-motion';

interface FloatingElementProps {
  children: React.ReactNode;
  speed?: number; // 1 is normal scroll, >1 is faster, <1 is slower (parallax)
  className?: string;
  delay?: number;
}

const FloatingElement: React.FC<FloatingElementProps> = ({ children, speed = 0.5, className = "", delay = 0 }) => {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", "end start"]
  });

  const y = useTransform(scrollYProgress, [0, 1], [100 * speed, -100 * speed]);
  const smoothY = useSpring(y, { stiffness: 100, damping: 30, mass: 1 });

  return (
    <motion.div 
      ref={ref}
      style={{ y: smoothY }}
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-10%" }}
      transition={{ duration: 0.8, delay, ease: [0.16, 1, 0.3, 1] }} // easeOutExpo approximation
      className={className}
    >
      {children}
    </motion.div>
  );
};

export default FloatingElement;