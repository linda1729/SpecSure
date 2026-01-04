import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

const MeshGradient: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = window.innerWidth;
    let height = window.innerHeight;
    
    const stars: {x: number, y: number, size: number, alpha: number, speed: number}[] = [];
    const numStars = 10000;

    const initStars = () => {
        stars.length = 0;
        for(let i=0; i<numStars; i++) {
            stars.push({
                x: Math.random() * width,
                y: Math.random() * height,
                size: 25,
                alpha: Math.random(),
                speed: Math.random() * 1 + 0.2
            });
        }
    };

    const resize = () => {
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width;
        canvas.height = height;
        initStars();
    };

    const animate = () => {
        ctx.clearRect(0, 0, width, height);
        
        ctx.fillStyle = "#ffffff";
        stars.forEach(star => {
            star.y -= star.speed;
            if (star.y < 0) star.y = height;
            
            ctx.globalAlpha = star.alpha * 0.5; // Subtle stars
            ctx.beginPath();
            ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
            ctx.fill();
        });
        
        requestAnimationFrame(animate);
    };

    window.addEventListener('resize', resize);
    resize();
    animate();

    return () => window.removeEventListener('resize', resize);
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none -z-10 overflow-hidden bg-[#191919]">
       {/* Canvas for Stars */}
       <canvas ref={canvasRef} className="absolute inset-0 z-0" />

       {/* Blurred Orbs - Using colors from reference: #adf6bd (Green) #3077f3 (Blue) */}
       <div className="absolute inset-0 z-10 mix-blend-screen opacity-40">
           <motion.div 
             animate={{ 
                x: [0, 100, -100, 0], 
                y: [0, -100, 100, 0],
                scale: [1, 1.2, 0.8, 1] 
             }}
             transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
             className="absolute top-[10%] left-[20%] w-[600px] h-[600px] rounded-full bg-[#adf6bd] blur-[120px]"
           />
           
           <motion.div 
             animate={{ 
                x: [0, -150, 150, 0], 
                y: [0, 150, -50, 0],
                scale: [1, 1.3, 0.9, 1] 
             }}
             transition={{ duration: 25, repeat: Infinity, ease: "easeInOut" }}
             className="absolute bottom-[20%] right-[10%] w-[500px] h-[500px] rounded-full bg-[#3077f3] blur-[120px]"
           />

           <motion.div 
             animate={{ 
                x: [0, 50, -50, 0], 
                scale: [1, 1.1, 1] 
             }}
             transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
             className="absolute top-[40%] left-[50%] -translate-x-1/2 w-[400px] h-[400px] rounded-full bg-[#61eaf4] blur-[100px]"
           />
       </div>
    </div>
  );
};

export default MeshGradient;