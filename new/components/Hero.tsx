import React from 'react';
import { motion } from 'framer-motion';

// Determine if we are in academic mode via CSS variable inspection or context could be better,
// but for simplicity we will style elements to be reactive to the CSS variables set by data-theme.

const Hero: React.FC = () => {
  return (
    <section className="relative min-h-[90vh] flex flex-col items-center justify-center px-6 md:px-10 lg:px-[72px] pt-32 pb-20 overflow-hidden text-center">
      
      {/* Decorative floating blurred orbs - Only visible in Modern Theme via CSS blending or opacity control if needed.
          Since we have 3D background in Academic, these CSS blurs might interfere visually. 
          We can hide them using CSS selector in index.html if strictly needed, but let's make them subtle.
      */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden -z-10">
          <motion.div 
            animate={{ y: [0, -40, 0], opacity: [0.3, 0.5, 0.3] }}
            transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
            className="absolute top-20 right-[10%] w-64 h-64 bg-blue-50/50 rounded-full blur-3xl opacity-0 [[data-theme='modern']_&]:opacity-100 transition-opacity duration-1000"
          />
          <motion.div 
            animate={{ y: [0, 60, 0], x: [0, 30, 0] }}
            transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
            className="absolute bottom-40 left-[5%] w-96 h-96 bg-gray-50/50 rounded-full blur-3xl opacity-0 [[data-theme='modern']_&]:opacity-100 transition-opacity duration-1000"
          />
      </div>

      <div className="max-w-[1400px] mx-auto w-full relative z-10 flex flex-col items-center">
        
        {/* Interactive Zone Wrapper */}
        <div className="pointer-events-auto mb-8 relative group flex flex-col items-center">
            
            {/* Eyebrow Text / Badge */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1.2, ease: [0.19, 1, 0.22, 1] }}
              className="mb-8"
            >
                {/* Modern Badge */}
                <span className="hidden [[data-theme='modern']_&]:inline-flex items-center gap-2 text-on-surface-variant font-medium text-lg bg-white/50 backdrop-blur-sm px-4 py-1 rounded-full border border-white/20">
                    <span className="font-symbol text-2xl text-primary">change_history</span>
                    <span>Hyperspectral Intelligence</span>
                </span>

                {/* Academic Eyebrow - "Journal Header" style */}
                <span className="hidden [[data-theme='academic']_&]:block font-code text-primary tracking-[0.2em] text-sm uppercase font-bold border-b border-primary/30 pb-2">
                    BlueArray Intelligence • Vol. 1 • 2024
                </span>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.2, delay: 0.1, ease: [0.19, 1, 0.22, 1] }}
              className="text-6xl md:text-8xl lg:text-[130px] lg:leading-[1.0] text-on-surface tracking-tighter mb-6 font-display"
            >
              SpecSure | 澜瞳
            </motion.h1>
            
            <motion.h2
               initial={{ opacity: 0, y: 50 }}
               animate={{ opacity: 1, y: 0 }}
               transition={{ duration: 1.2, delay: 0.2, ease: [0.19, 1, 0.22, 1] }}
               className="text-3xl md:text-5xl lg:text-6xl text-on-surface-variant tracking-tight max-w-4xl mx-auto"
            >
               <span className="[[data-theme='academic']_&]:font-serif [[data-theme='academic']_&]:italic">
                 Revealing the invisible through spectral precision.
               </span>
            </motion.h2>
        </div>

        <motion.div
           initial={{ opacity: 0, y: 30 }}
           animate={{ opacity: 1, y: 0 }}
           transition={{ duration: 1.2, delay: 0.4, ease: [0.19, 1, 0.22, 1] }}
           className="pointer-events-auto max-w-3xl mx-auto mb-16"
        >
             <p className="text-xl md:text-2xl text-on-surface leading-relaxed font-light">
               The advanced hyperspectral intelligence platform by <strong className="font-medium text-primary">BlueArray</strong>.
               <br/>
               <span className="text-on-surface-variant/70 text-lg mt-2 block">
                  Integrating proprietary SpecNet architecture with high-fidelity optics.
               </span>
             </p>
        </motion.div>

        {/* Buttons */}
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.6 }}
            className="pointer-events-auto flex flex-col sm:flex-row gap-4 items-center"
        >
            <button className="h-14 px-10 rounded-full bg-on-surface text-surface font-medium text-lg hover:opacity-90 transition-all active:scale-95 shadow-lg">
                Start Research
            </button>
            <button className="h-14 px-10 rounded-full bg-surface/50 backdrop-blur-md text-on-surface border border-on-surface/10 font-medium text-lg hover:bg-surface transition-all active:scale-95">
                View Documentation
            </button>
        </motion.div>

      </div>
    </section>
  );
};

export default Hero;