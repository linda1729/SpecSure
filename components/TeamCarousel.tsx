import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence, useInView } from 'framer-motion';

// Mock data with professional portraits (Unsplash IDs)
const teamMembers = [
  { 
    id: 'linda', 
    name: "Linda1729", 
    role: "Backend Architect", 
    desc: "You never know how many APIs she has built. The silent engine of the platform.",
    image: "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?q=80&w=1200&auto=format&fit=crop", 
    link: "https://blog.linda1729.com/"
  },
  { 
    id: 'chen', 
    name: "Chenmomo", 
    role: "Deep Learning Engineer (CNN)", 
    desc: "Chief Wizard of CNN Magic. Turning pixels into understanding.",
    image: "https://images.unsplash.com/photo-1580489944761-15a19d654956?q=80&w=1200&auto=format&fit=crop", 
    link: "https://blog.linda1729.com/"
  },
  { 
    id: 'xixiy', 
    name: "XiXiYHaHa", 
    role: "Machine Learning Engineer (SVM)", 
    desc: "The handsome master of Support Vector Machines. Optimizing hyperplanes daily.",
    image: "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?q=80&w=1200&auto=format&fit=crop", 
    link: "https://blog.linda1729.com/"
  },
  { 
    id: 'keeping', 
    name: "Keeping", 
    role: "Frontend Developer", 
    desc: "Guardian of Frontend Aesthetics. Fighting three clients at once.",
    image: "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?q=80&w=1200&auto=format&fit=crop", 
    link: "https://blog.linda1729.com/"
  },
  { 
    id: 'gong', 
    name: "Gong", 
    role: "Chief Technology Officer", 
    desc: "The Great Leader. Orchestrating the symphony of algorithms and architecture.",
    image: "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?q=80&w=1200&auto=format&fit=crop", 
    link: "https://blog.linda1729.com/"
  }
];

const Typewriter: React.FC<{ text: string; keyTrigger: number }> = ({ text, keyTrigger }) => {
  const [displayedText, setDisplayedText] = useState("");
  const ref = useRef(null);
  // Increased bottom margin to ensure it's well inside viewport before triggering
  const isInView = useInView(ref, { margin: "0px 0px -50px 0px", once: true });

  useEffect(() => {
    if (!isInView) return;

    setDisplayedText("");
    let i = 0;
    let timer: any = null;
    
    // Add a small delay to ensure visibility before typing starts
    const startDelay = setTimeout(() => {
        timer = setInterval(() => {
          if (i < text.length) {
            setDisplayedText((prev) => text.slice(0, i + 1));
            i++;
          } else {
            clearInterval(timer);
          }
        }, 40);
    }, 200);

    return () => {
      clearTimeout(startDelay);
      if (timer) clearInterval(timer);
    };
  }, [text, keyTrigger, isInView]);

  return (
    <span ref={ref} className="inline-block min-h-[1.2em]">
      {displayedText}
      <span className="animate-pulse ml-1 text-primary">|</span>
    </span>
  );
};

const TeamCarousel: React.FC = () => {
  const [activeIndex, setActiveIndex] = useState(0);
  const [direction, setDirection] = useState(0);

  const paginate = (newDirection: number) => {
    setDirection(newDirection);
    setActiveIndex((prev) => {
      let nextIndex = prev + newDirection;
      if (nextIndex < 0) nextIndex = teamMembers.length - 1;
      if (nextIndex >= teamMembers.length) nextIndex = 0;
      return nextIndex;
    });
  };

  const currentMember = teamMembers[activeIndex];

  const variants = {
    enter: (direction: number) => ({
      x: direction > 0 ? 300 : -300,
      opacity: 0,
      scale: 0.95
    }),
    center: {
      zIndex: 1,
      x: 0,
      opacity: 1,
      scale: 1
    },
    exit: (direction: number) => ({
      zIndex: 0,
      x: direction < 0 ? 300 : -300,
      opacity: 0,
      scale: 0.95
    })
  };

  return (
    <section className="relative py-24 px-6 md:px-10 lg:px-[72px] bg-surface overflow-hidden min-h-[850px] flex items-center">
      
      {/* Background Abstract Decoration */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none opacity-20">
        <div className="absolute top-[-20%] right-[-10%] w-[600px] h-[600px] rounded-full bg-gradient-to-br from-primary/10 to-transparent blur-3xl" />
        <div className="absolute bottom-[-20%] left-[-10%] w-[600px] h-[600px] rounded-full bg-gradient-to-tr from-blue-300/10 to-transparent blur-3xl" />
      </div>

      <div className="max-w-[1600px] mx-auto w-full relative z-10">
        
        {/* Header */}
        <div className="mb-12 flex items-end justify-between relative z-30">
          <div>
            <span className="font-code text-sm uppercase tracking-widest text-primary mb-2 block">Our Squad</span>
            <h2 className="text-4xl md:text-5xl font-light text-on-surface">Meet the Minds</h2>
          </div>
        </div>

        {/* Carousel Content */}
        <div className="relative h-[650px] w-full">
          
          <AnimatePresence initial={false} custom={direction} mode="wait">
            <motion.div
              key={activeIndex}
              custom={direction}
              variants={variants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{
                x: { type: "spring", stiffness: 300, damping: 30 },
                opacity: { duration: 0.3 }
              }}
              className="absolute inset-0 flex flex-col lg:flex-row items-center gap-8 lg:gap-12"
            >
              
              {/* Image Side */}
              <div className="w-full lg:w-1/2 h-full relative group cursor-pointer z-20" onClick={() => window.open(currentMember.link, '_blank')}>
                <div className="relative w-full h-full rounded-[40px] overflow-hidden shadow-2xl bg-surface-container">
                  <img 
                    src={currentMember.image} 
                    alt={currentMember.name}
                    className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
                  />
                  
                  {/* Subtle Gradient Overlay */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                  
                  {/* Controls Overlay - Bottom Left */}
                  <div className="absolute bottom-8 left-8 md:bottom-12 md:left-12 flex gap-3 pointer-events-auto z-30">
                     <button 
                        onClick={(e) => { e.stopPropagation(); paginate(-1); }}
                        className="w-14 h-14 rounded-full bg-white/90 backdrop-blur-md border border-white/20 flex items-center justify-center hover:bg-white hover:scale-105 transition-all shadow-lg group/btn cursor-pointer"
                        aria-label="Previous member"
                     >
                        <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" className="fill-grey-900 group-hover/btn:fill-primary transition-colors">
                            <path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z"/>
                        </svg>
                     </button>
                     <button 
                        onClick={(e) => { e.stopPropagation(); paginate(1); }}
                        className="w-14 h-14 rounded-full bg-white/90 backdrop-blur-md border border-white/20 flex items-center justify-center hover:bg-white hover:scale-105 transition-all shadow-lg group/btn cursor-pointer"
                        aria-label="Next member"
                     >
                         <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" className="fill-grey-900 group-hover/btn:fill-primary transition-colors">
                            <path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/>
                         </svg>
                     </button>
                  </div>
                  
                  {/* Blog Link Top Right */}
                  <div className="absolute top-8 right-8 px-5 py-2.5 bg-white/30 backdrop-blur-md rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all duration-300 shadow-sm border border-white/20 hover:bg-white/50">
                    <span className="font-code text-sm font-bold text-white tracking-wide">Blog</span>
                  </div>
                </div>
              </div>

              {/* Text Side */}
              <div className="w-full lg:w-1/2 flex flex-col justify-center pointer-events-none pl-4 lg:pl-10">
                <div className="pointer-events-auto">
                    <h3 className="text-6xl md:text-8xl font-medium tracking-tighter mb-4 text-on-surface">
                        {currentMember.name}
                    </h3>
                    
                    <div className="text-2xl md:text-3xl font-code text-primary mb-8 h-12 flex items-center">
                        <Typewriter text={currentMember.role} keyTrigger={activeIndex} />
                    </div>

                    <p className="text-xl md:text-2xl text-on-surface-variant font-light leading-relaxed max-w-lg mb-12">
                        {currentMember.desc}
                    </p>

                    <div className="flex flex-wrap gap-4">
                        <span className="px-4 py-2 rounded-full bg-surface-container border border-grey-200 text-sm font-code uppercase tracking-wider text-grey-800">
                        BlueArray
                        </span>
                        <span className="px-4 py-2 rounded-full bg-surface-container border border-grey-200 text-sm font-code uppercase tracking-wider text-grey-800">
                        Est. 2025
                        </span>
                    </div>
                </div>
              </div>

            </motion.div>
          </AnimatePresence>
        </div>

      </div>
    </section>
  );
};

export default TeamCarousel;