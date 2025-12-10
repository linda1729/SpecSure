import React, { useEffect, useRef } from 'react';

interface ParticleSystemProps {
  theme?: 'modern' | 'academic';
}

// Google Brand Colors for the dots (Modern Mode)
const COLORS_MODERN = ['#ea4335', '#fbbc04', '#34a853', '#4285f4', '#E1E6EC', '#9AA0A6'];
const TIDE_COLOR_MODERN = { r: 26, g: 115, b: 232 }; // Blue

// Academic Luxury Colors (Gold, Stone, Grey)
const COLORS_ACADEMIC = ['#C5A059', '#1C1917', '#57534E', '#E7E5E4', '#C5A059'];
const TIDE_COLOR_ACADEMIC = { r: 197, g: 160, b: 89 }; // Gold

// Helper to convert hex to rgb for interpolation
const hexToRgb = (hex: string) => {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : { r: 0, g: 0, b: 0 };
};

class Dot {
  x: number;
  y: number;
  originX: number;
  originY: number;
  vx: number;
  vy: number;
  color: { r: number, g: number, b: number };
  baseColor: { r: number, g: number, b: number };
  targetColor: { r: number, g: number, b: number };
  size: number;
  spring: number;
  friction: number;
  
  constructor(w: number, h: number, palette: string[]) {
    this.x = Math.random() * w;
    this.y = Math.random() * h;
    this.originX = this.x;
    this.originY = this.y;
    this.vx = 0;
    this.vy = 0;
    
    const rgbPalette = palette.map(hexToRgb);
    this.baseColor = rgbPalette[Math.floor(Math.random() * rgbPalette.length)];
    this.color = { ...this.baseColor };
    this.targetColor = { ...this.baseColor };
    
    // Size variation: slightly smaller for cleaner look
    this.size = Math.random() * 1.2 + 0.5;

    // Physics
    this.spring = 0.01 + Math.random() * 0.02; 
    this.friction = 0.90 + Math.random() * 0.05; // Higher friction = smoother glide
  }

  updatePalette(palette: string[]) {
    const rgbPalette = palette.map(hexToRgb);
    this.baseColor = rgbPalette[Math.floor(Math.random() * rgbPalette.length)];
  }

  update(ctx: CanvasRenderingContext2D, mouse: { x: number, y: number }, width: number, height: number, time: number, index: number, total: number, scrollY: number, theme: 'modern' | 'academic') {
    const centerX = width / 2;
    // The target text center moves up as we scroll.
    const textCenterY = (height / 2) - scrollY;
    
    // Determine if mouse is hovering over the specific text block area
    // Reduced area slightly to prevent accidental triggers
    const dxMouse = Math.abs(mouse.x - centerX);
    const dyMouse = Math.abs(mouse.y - textCenterY);
    
    const isHovering = (dxMouse < 400) && (dyMouse < 150);

    // --- MOUSE INTERACTION (WAVE EFFECT) ---
    // Repel particles from mouse cursor
    const dxCursor = this.x - mouse.x;
    const dyCursor = this.y - mouse.y;
    const distCursor = Math.sqrt(dxCursor * dxCursor + dyCursor * dyCursor);
    const interactionRadius = 150;

    if (distCursor < interactionRadius) {
        const force = (interactionRadius - distCursor) / interactionRadius;
        const angle = Math.atan2(dyCursor, dxCursor);
        const pushStrength = theme === 'academic' ? 0.8 : 2.0; 
        
        this.vx += Math.cos(angle) * force * pushStrength;
        this.vy += Math.sin(angle) * force * pushStrength;
    }

    let desiredColor = this.baseColor;

    if (isHovering) {
      // --- FORM TIDE ICON (Active State) ---
      const waveCount = 3;
      const scrambledIndex = (index * 13) % total; 
      
      const pointsPerWave = Math.floor(total / waveCount);
      const waveId = Math.floor(scrambledIndex / pointsPerWave); 
      const progress = (scrambledIndex % pointsPerWave) / pointsPerWave; 

      // Wave Configuration
      const waveWidth = 500; 
      const xOffset = (progress - 0.5) * waveWidth;
      
      const frequency = 2 * Math.PI * 1.5; 
      const amplitude = 40; 
      const flowSpeed = theme === 'academic' ? 0.02 : 0.05;
      
      const phase = time * flowSpeed + (waveId * 1.0); 
      const ySpacing = 90;
      const yBase = (waveId - 1) * ySpacing; 

      const targetX = centerX + xOffset;
      const targetY = textCenterY + yBase + Math.sin(progress * frequency + phase) * amplitude;

      // Physics: Spring to target
      const springStrength = theme === 'academic' ? this.spring * 0.5 : this.spring;
      this.vx += (targetX - this.x) * springStrength;
      this.vy += (targetY - this.y) * springStrength;

      desiredColor = theme === 'academic' ? TIDE_COLOR_ACADEMIC : TIDE_COLOR_MODERN;

    } else {
      // --- IDLE STATE ---
      // Drift back to original random position
      const returnForce = 0.005;
      this.vx += (this.originX - this.x) * returnForce;
      this.vy += (this.originY - this.y) * returnForce;

      // Add gentle noise (Brownian motion)
      // FIX: Reduced noise significantly from 0.2 to 0.02 to stop "vibrating" chaos
      const noise = theme === 'academic' ? 0.01 : 0.02;
      this.vx += (Math.random() - 0.5) * noise;
      this.vy += (Math.random() - 0.5) * noise;

      desiredColor = this.baseColor;
    }

    // Color Transition Interpolation
    this.color.r += (desiredColor.r - this.color.r) * 0.05;
    this.color.g += (desiredColor.g - this.color.g) * 0.05;
    this.color.b += (desiredColor.b - this.color.b) * 0.05;

    // Apply Friction
    this.vx *= this.friction;
    this.vy *= this.friction;

    // Move
    this.x += this.vx;
    this.y += this.vy;

    // Draw
    ctx.fillStyle = `rgb(${Math.round(this.color.r)}, ${Math.round(this.color.g)}, ${Math.round(this.color.b)})`;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.fill();
  }
}

const ParticleSystem: React.FC<ParticleSystemProps> = ({ theme = 'modern' }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dotsRef = useRef<Dot[]>([]);
  const themeRef = useRef(theme);
  const requestRef = useRef<number | null>(null);

  useEffect(() => {
    themeRef.current = theme;
    const palette = theme === 'academic' ? COLORS_ACADEMIC : COLORS_MODERN;
    dotsRef.current.forEach(dot => dot.updatePalette(palette));
  }, [theme]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let time = 0;
    let scrollY = window.scrollY;
    
    // FIX: Reduced particle count for a cleaner aesthetic
    // High res screens get 1200, smaller screens get 600
    let width = window.innerWidth;
    let height = window.innerHeight;
    const PARTICLE_COUNT = width > 1000 ? 1000 : 600; 

    // State
    const mouse = { x: -1000, y: -1000 };

    const init = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      
      // FIX: High DPI (Retina) support
      const dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      
      // Normalize coordinate system
      ctx.scale(dpr, dpr);
      
      const palette = themeRef.current === 'academic' ? COLORS_ACADEMIC : COLORS_MODERN;
      dotsRef.current = [];
      for(let i=0; i<PARTICLE_COUNT; i++) {
        dotsRef.current.push(new Dot(width, height, palette));
      }
    };

    const handleResize = () => {
      init();
    };

    const handleMouseMove = (e: MouseEvent) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    };

    const handleScroll = () => {
      scrollY = window.scrollY;
      if (canvasRef.current) {
        const fadeStart = 0;
        const fadeEnd = height * 0.8;
        const opacity = Math.max(0, 1 - (scrollY - fadeStart) / (fadeEnd - fadeStart));
        canvasRef.current.style.opacity = (opacity * 0.8).toString();
      }
    };

    const render = () => {
      ctx.clearRect(0, 0, width, height);
      time++;
      
      dotsRef.current.forEach((dot, index) => {
        dot.update(ctx, mouse, width, height, time, index, dotsRef.current.length, scrollY, themeRef.current);
      });

      requestRef.current = requestAnimationFrame(render);
    };

    init();
    render();
    
    window.addEventListener('resize', handleResize);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('scroll', handleScroll);

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('scroll', handleScroll);
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, []);

  return (
    <canvas 
      ref={canvasRef} 
      className="fixed inset-0 pointer-events-none z-0 transition-opacity duration-300 ease-out"
      style={{ opacity: 0.8 }}
    />
  );
};

export default ParticleSystem;