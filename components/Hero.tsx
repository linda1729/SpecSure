import React from 'react';
import { motion } from 'framer-motion';

// Determine if we are in academic mode via CSS variable inspection or context could be better,
// but for simplicity we will style elements to be reactive to the CSS variables set by data-theme.

const Hero: React.FC = () => {
  const handleStartResearch = () => {
    // Navigate to the functional pages (mockfrontend)
    window.location.href = '/mockfrontend/index.html';
  };
  return (
    <section className="relative min-h-[90vh] flex flex-col items-center justify-center px-6 md:px-10 lg:px-[72px] pt-32 pb-20 overflow-hidden text-center">
      
      {/* Decorative floating blurred orbs */}
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
            
            <motion.h1
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.2, delay: 0.1, ease: [0.19, 1, 0.22, 1] }}
              className="flex flex-col items-center mb-12"
            >
              <span className="text-6xl md:text-8xl lg:text-[130px] lg:leading-[1.0] text-on-surface tracking-tighter font-display">
                SpecSure
              </span>
              <span className="text-3xl md:text-5xl lg:text-[54px] font-cn font-light text-on-surface/80 mt-4 tracking-[0.5em] ml-[0.5em]">
                澜瞳
              </span>
            </motion.h1>
            
        </div>

        <motion.div
           initial={{ opacity: 0, y: 30 }}
           animate={{ opacity: 1, y: 0 }}
           transition={{ duration: 1.2, delay: 0.4, ease: [0.19, 1, 0.22, 1] }}
           className="pointer-events-auto max-w-3xl mx-auto mb-16"
        >
             <p className="text-xl md:text-2xl text-on-surface leading-relaxed font-light">
               The advanced hyperspectral intelligence platform by <strong className="font-medium text-primary">BlueArray</strong>.
             </p>
        </motion.div>

        {/* Buttons */}
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.6 }}
            className="pointer-events-auto flex flex-col sm:flex-row gap-4 items-center"
        >
            <button 
              onClick={handleStartResearch}
              className="h-14 px-10 rounded-full bg-on-surface text-surface font-medium text-lg hover:opacity-90 transition-all active:scale-95 shadow-lg">
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