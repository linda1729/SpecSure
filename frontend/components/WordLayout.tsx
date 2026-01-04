import React, { useState, useEffect, useRef } from 'react';
import { motion, useInView, useScroll, useTransform, useSpring } from 'framer-motion';

// --- Data ---
const features = [
  { id: '01', title: 'Spectral Analysis', subtitle: '400-1000nm Range', desc: 'Capturing the invisible. Our sensors decompose light into hundreds of bands, revealing chemical compositions instantly.', image: 'https://raw.githubusercontent.com/linda1729/SpecSure/feature-frontend/iamges/img_5.png'},
  { id: '02', title: 'Model A: Traditional ML', subtitle: 'SVM / Random Forest', desc: 'Support Vector Machines and Random Forest algorithms. Classic, reliable, efficient and stable approaches for hyperspectral classification.', image: 'https://raw.githubusercontent.com/linda1729/SpecSure/feature-frontend/iamges/PU_pseudocolor_pca%3D15_window%3D25_lr%3D0.001_epochs%3D100.png' },
  { id: '03', title: 'Model B: Deep Learning', subtitle: '3D CNN / HybridSN', desc: '3D Convolutional Neural Networks and Hybrid Spectral-Spatial Networks. Fusing spatial-spectral information, specifically optimized for hyperspectral data analysis.', image: 'https://raw.githubusercontent.com/linda1729/SpecSure/feature-frontend/iamges/IndianPines_pseudocolor_pca%3D30_window%3D25_lr%3D0.001_epochs%3D1.png' },
];

const teamMembers = [
  { 
    id: 'linda', 
    name: "Linda1729", 
    role: "Backend Architect", 
    desc: "You never know how many APIs she has built. The silent engine of the platform.",
    image: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?q=80&w=800&auto=format&fit=crop", // Minimalist Portrait
    blog: "https://example.com/linda"
  },
  { 
    id: 'chen', 
    name: "Chenmomo", 
    role: "Deep Learning Engineer (CNN)", 
    desc: "Chief Wizard of CNN Magic. Turning pixels into understanding.",
    image: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?q=80&w=800&auto=format&fit=crop", // Clean Portrait
    blog: "https://example.com/chen"
  },
  { 
    id: 'xixiy', 
    name: "XiXiYHaHa", 
    role: "Machine Learning Engineer (SVM)", 
    desc: "The handsome master of Support Vector Machines. Optimizing hyperplanes daily.",
    image: "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?q=80&w=800&auto=format&fit=crop", 
    blog: "https://example.com/xixi"
  },
  { 
    id: 'keeping', 
    name: "Merciless Killer", 
    role: "Frontend Developer", 
    desc: "Guardian of Frontend Aesthetics. Fighting three clients at once.",
    image: "https://raw.githubusercontent.com/linda1729/SpecSure/feature-frontend/iamges/1.jpeg", 
    blog: "https://example.com/keeping"
  },
  { 
    id: 'gong', 
    name: "Gong", 
    role: "Chief Technology Officer", 
    desc: "The Great Leader. Orchestrating the symphony of algorithms and architecture.",
    image: "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?q=80&w=800&auto=format&fit=crop", 
    blog: "https://example.com/gong"
  }
];

// --- Floating Hero Items Data ---
// Refined assets: Minimalist shapes, Memoji-style avatars, Clean UI elements
const floatingItems = [
    { type: 'image', src: 'https://images.unsplash.com/photo-1633332755192-727a05c4013d?q=80&w=400&auto=format&fit=crop', x: '10%', y: '20%', rotate: -10, scale: 1.4 }, // Avatar 1
    { type: 'emoji', content: '🔮', x: '85%', y: '25%', rotate: 15, scale: 2.8 },
    { type: 'image', src: 'https://images.unsplash.com/photo-1599566150163-29194dcaad36?q=80&w=400&auto=format&fit=crop', x: '80%', y: '65%', rotate: 5, scale: 1.5 }, // Avatar 2
    { type: 'emoji', content: '⚡', x: '15%', y: '70%', rotate: -10, scale: 3.5 },
    { type: 'symbol', content: 'view_in_ar', x: '80%', y: '15%', rotate: -8, scale: 1.8 }, // 3D Icon
    { type: 'symbol', content: 'code_blocks', x: '5%', y: '45%', rotate: 20, scale: 2.1 }  // Tech Icon
];

// --- Components ---

const FadeIn: React.FC<{ children: React.ReactNode; delay?: number }> = ({ children, delay = 0 }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-10%" }}
        transition={{ duration: 0.8, delay, ease: [0.22, 1, 0.36, 1] }}
    >
        {children}
    </motion.div>
);

const ParallaxImage: React.FC<{ src: string; alt: string }> = ({ src, alt }) => {
    const ref = useRef(null);
    const { scrollYProgress } = useScroll({
        target: ref,
        offset: ["start end", "end start"]
    });
    
    // Subtle parallax shift
    const y = useTransform(scrollYProgress, [0, 1], ["-3%", "3%"]);

    return (
        <div ref={ref} className="w-full h-full relative flex items-center justify-center">
            <motion.div style={{ y }} className="w-full h-full flex items-center justify-center">
                <img src={src} alt={alt} className="w-full h-full object-contain" />
            </motion.div>
        </div>
    )
}

const TypewriterText: React.FC<{ text: string; delay?: number }> = ({ text, delay = 0 }) => {
  const [displayedText, setDisplayedText] = useState("");
  const ref = useRef(null);
  const isInView = useInView(ref, { margin: "0px 0px -50px 0px", once: true });

  useEffect(() => {
    if (!isInView) return;

    const startTimeout = setTimeout(() => {
        let i = 0;
        const timer = setInterval(() => {
          if (i < text.length) {
            setDisplayedText((prev) => text.slice(0, i + 1));
            i++;
          } else {
            clearInterval(timer);
          }
        }, 30);
        return () => clearInterval(timer);
    }, delay * 1000 + 300);

    return () => clearTimeout(startTimeout);
  }, [text, isInView, delay]);

  return (
    <span ref={ref} className="inline-block min-h-[1.5em]">
      {displayedText}
      {displayedText.length < text.length && isInView && (
        <span className="animate-pulse ml-1 text-secondary">|</span>
      )}
    </span>
  );
};

// --- Floating Hero Item ---
const FloatingHeroItem: React.FC<{ item: any; index: number }> = ({ item, index }) => {
    // 1. Entrance Animation: Fly in from Center (50% 50%)
    // 2. Continuous Animation: Bob gently
    
    return (
        <motion.div
            custom={index}
            initial={{ opacity: 0, scale: 0, left: "50%", top: "50%", x: "-50%", y: "-50%" }}
            animate={{ 
                opacity: 1, 
                scale: item.scale,
                left: item.x,
                top: item.y,
                x: "-50%",
                y: "-50%"
            }}
            transition={{
                duration: 1.5,
                delay: 0.2 + index * 0.1,
                type: "spring",
                stiffness: 80,
                damping: 15
            }}
            style={{ 
                position: 'absolute', 
                zIndex: 0
            }}
            className="pointer-events-none hidden md:block"
        >
            {/* Inner Floating Animation */}
            <motion.div
                animate={{ 
                    y: [0, -15, 0],
                    rotate: [item.rotate - 5, item.rotate + 5, item.rotate - 5]
                }}
                transition={{ 
                    duration: 4 + Math.random() * 2, 
                    repeat: Infinity, 
                    ease: "easeInOut",
                    delay: 1.5 + index * 0.2 // Start floating after entrance
                }}
            >
                {item.type === 'image' && (
                    <div className="w-24 h-24 md:w-32 md:h-32 rounded-full overflow-hidden shadow-2xl border-4 border-white transform hover:scale-105 transition-transform duration-500 bg-gray-100">
                        <img src={item.src} alt="" className="w-full h-full object-cover" />
                    </div>
                )}
                {item.type === 'emoji' && (
                    <div className="text-6xl md:text-8xl drop-shadow-xl filter grayscale-[0.2] hover:grayscale-0 transition-all">
                        {item.content}
                    </div>
                )}
                {item.type === 'symbol' && (
                    <div className="w-20 h-20 bg-black text-white rounded-3xl flex items-center justify-center shadow-xl rotate-12 hover:rotate-0 transition-transform">
                        <span className="material-symbols-outlined text-4xl">{item.content}</span>
                    </div>
                )}
            </motion.div>
        </motion.div>
    )
}

// --- Winding Path Component ---
const WindingTeamSection: React.FC = () => {
    const containerRef = useRef<HTMLDivElement>(null);
    const { scrollYProgress } = useScroll({
        target: containerRef,
        offset: ["start center", "end end"]
    });

    const pathLength = useSpring(scrollYProgress, { stiffness: 400, damping: 90 });

    // Item Height approx 600px
    const ITEM_HEIGHT = 600; 
    const TOTAL_HEIGHT = teamMembers.length * ITEM_HEIGHT;
    
    // Construct Path
    const AMPLITUDE = 250; 
    
    const points = teamMembers.map((_, i) => {
       const isLeft = i % 2 === 0;
       const y = (i * ITEM_HEIGHT) + (ITEM_HEIGHT / 2);
       const x = isLeft ? 500 - AMPLITUDE : 500 + AMPLITUDE; 
       return { x, y };
    });

    // Generate 'S' curves
    let d = `M 500 0 `;
    
    // Initial curve
    d += `C 500 ${points[0].y * 0.5}, ${points[0].x} ${points[0].y * 0.5}, ${points[0].x} ${points[0].y} `;
    
    // Connecting curves
    for (let i = 0; i < points.length - 1; i++) {
        const curr = points[i];
        const next = points[i+1];
        const midY = (curr.y + next.y) / 2;
        d += `C ${curr.x} ${midY}, ${next.x} ${midY}, ${next.x} ${next.y} `;
    }

    // Final curve
    const last = points[points.length - 1];
    d += `C ${last.x} ${last.y + 200}, 500 ${last.y + 200}, 500 ${TOTAL_HEIGHT + 200}`;


    return (
        <div ref={containerRef} className="relative w-full max-w-[1200px] mx-auto py-32">
            
            {/* Background Path Layer - Explicitly Z-0 to stay behind images */}
            <div className="absolute inset-0 top-32 pointer-events-none hidden md:block z-0">
                 <svg 
                    className="w-full h-full"
                    viewBox={`0 0 1000 ${TOTAL_HEIGHT + 200}`}
                    preserveAspectRatio="none"
                 >
                     {/* Base faint line */}
                     <path 
                        d={d}
                        fill="none"
                        stroke="rgba(0,0,0,0.05)"
                        strokeWidth="2"
                        strokeDasharray="12 12"
                     />
                     {/* Animated Scroll Progress Line */}
                     <motion.path 
                        d={d}
                        fill="none"
                        stroke="rgba(0,0,0,0.8)"
                        strokeWidth="3"
                        strokeDasharray="12 12"
                        style={{ pathLength }}
                     />
                 </svg>
            </div>

            {/* Team Members - Relative Z-10 to stay above the line */}
            <div className="relative z-10 flex flex-col gap-24 md:gap-0">
                {teamMembers.map((member, i) => {
                    const isLeft = i % 2 === 0;
                    
                    // SWAPPED LAYOUT LOGIC:
                    // If Even (Curve Left): Flex Row (Image Left, Text Right) -> Line passes behind Image Left
                    // If Odd (Curve Right): Flex Row Reverse (Text Left, Image Right) -> Line passes behind Image Right
                    const rowClass = isLeft ? 'md:flex-row' : 'md:flex-row-reverse';
                    
                    // ALIGNMENT LOGIC:
                    // If Image Left (isLeft): Text should align Left (towards center).
                    // If Image Right (!isLeft): Text should align Right (towards center).
                    const textAlignClass = isLeft ? 'md:items-start md:text-left md:pl-24' : 'md:items-end md:text-right md:pr-24';
                    
                    return (
                        <div 
                            key={i} 
                            className={`flex flex-col md:flex-row items-center w-full min-h-[600px] px-6 md:px-12 ${rowClass}`}
                        >
                            {/* --- IMAGE SIDE (Now First in DOM structure, but flex handles position) --- */}
                            <div className="w-full md:w-1/2 px-6 flex justify-center relative">
                                {/* Optional: Marker on the path behind the image */}
                                <div className="hidden md:block absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-4 h-4 bg-black rounded-full z-0"></div>

                                <motion.div 
                                    initial={{ opacity: 0, scale: 0.8, rotate: isLeft ? -10 : 10 }}
                                    whileInView={{ opacity: 1, scale: 1, rotate: isLeft ? -2 : 2 }}
                                    transition={{ duration: 0.8, type: "spring", bounce: 0.4 }}
                                    whileHover={{ scale: 1.05, rotate: 0 }}
                                    onClick={() => window.open(member.blog, '_blank')}
                                    className="aspect-[3/4] w-full max-w-[280px] relative shadow-2xl bg-white p-3 cursor-pointer group z-10"
                                >
                                    <div className="w-full h-full overflow-hidden bg-gray-100 relative">
                                        <img 
                                            src={member.image} 
                                            alt={member.name} 
                                            className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
                                        />
                                        <div className="absolute inset-0 border border-black/5 pointer-events-none"></div>
                                        
                                        {/* Blog Overlay */}
                                        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center backdrop-blur-[2px]">
                                            <div className="bg-white text-black px-6 py-2 rounded-full font-code text-sm font-bold tracking-widest transform translate-y-4 group-hover:translate-y-0 transition-transform duration-300">
                                                READ BLOG
                                            </div>
                                        </div>
                                    </div>
                                    
                                    {/* Decoration Tape */}
                                    <div className="absolute -top-3 left-1/2 -translate-x-1/2 w-20 h-5 bg-black/10 rotate-1 shadow-sm backdrop-blur-sm"></div>
                                </motion.div>
                            </div>

                            {/* --- SPACER (Center) --- */}
                            <div className="hidden md:block w-32 relative flex justify-center">
                                {/* Empty spacer, the line goes through here on the background if using previous logic, 
                                    but now the line curves to the Image side. */}
                            </div>

                            {/* --- TEXT SIDE --- */}
                            <div className={`w-full md:w-1/2 flex flex-col justify-center text-center mb-12 md:mb-0 ${textAlignClass}`}>
                                <FadeIn>
                                    <h3 className="text-5xl md:text-7xl font-serif mb-4 whitespace-nowrap">{member.name}</h3>
                                    <span className="inline-block bg-black text-white px-4 py-1 font-code uppercase tracking-widest text-xs mb-6 rounded-sm">
                                        {member.role}
                                    </span>
                                    <div className="text-xl md:text-2xl font-light text-gray-600 leading-relaxed max-w-md">
                                        <TypewriterText text={member.desc} delay={0.3} />
                                    </div>
                                </FadeIn>
                            </div>

                        </div>
                    );
                })}
            </div>

            {/* End of Path Decoration */}
            <div className="text-center mt-32 relative z-10">
                <span className="font-serif italic text-2xl text-gray-400">...and growing</span>
            </div>
        </div>
    );
};


const WordLayout: React.FC = () => {
  const [showVideo, setShowVideo] = useState(false);

  return (
    <div className="bg-white text-black min-h-screen selection:bg-black selection:text-white overflow-x-hidden">
      
      {/* --- HERO: Minimalist & Bold + Fly-In Elements --- */}
      <section className="min-h-[95vh] relative flex flex-col justify-center items-center px-6 md:px-12 pt-32 pb-12 border-b border-black/10 overflow-hidden">
         
         {/* Floating Background Elements (Fly In from Center) */}
         <div className="absolute inset-0 w-full h-full max-w-[1600px] mx-auto pointer-events-none">
             {floatingItems.map((item, i) => (
                 <FloatingHeroItem key={i} item={item} index={i} />
             ))}
         </div>

         <div className="relative z-10 text-center max-w-4xl mx-auto">
             {/* Eyebrow */}
             <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 1, delay: 0.8 }}
                className="flex items-center justify-center gap-4 mb-12"
             >
                 <span className="h-[1px] w-12 bg-black"></span>
                 <span className="font-code text-sm md:text-lg uppercase tracking-widest text-gray-500">Vol. 1 — 2025</span>
                 <span className="h-[1px] w-12 bg-black"></span>
             </motion.div>

             <motion.h1 
                initial={{ opacity: 0, scale: 0.8, filter: "blur(10px)" }} 
                animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }} 
                transition={{ duration: 1.5, ease: "easeOut" }}
                className="mb-8 flex flex-col items-center"
             >
                 <span style={{ fontSize: '200px', fontWeight: 900, fontFamily: 'Google Sans, Inter, Arial, sans-serif', lineHeight: 1 }}>SpecSure</span>
                 <span style={{ fontSize: '130px', fontWeight: 900, fontFamily: 'Noto Sans SC, sans-serif', marginTop: 24, letterSpacing: '0.2em', opacity: 0.8, lineHeight: 1 }}>澜瞳</span>
             </motion.h1>

             {/* Description removed as requested */}
            
            <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 1, delay: 1.5 }}
                className="mt-16 flex justify-center gap-4"
            >
                <span className="font-code text-sm border border-black/20 rounded-full px-6 py-3 hover:bg-black hover:text-white transition-colors cursor-pointer bg-white/50 backdrop-blur-sm">
                    SCROLL TO EXPLORE
                </span>
                <button 
                    onClick={() => setShowVideo(true)}
                    className="font-code text-sm border border-black/20 rounded-full px-6 py-3 hover:bg-black hover:text-white transition-colors cursor-pointer bg-white/50 backdrop-blur-sm"
                >
                    VIEW DOCUMENT
                </button>
            </motion.div>
         </div>
      </section>

      {/* Video Modal */}
      {showVideo && (
        <div 
            className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4"
            onClick={() => setShowVideo(false)}
        >
            <div 
                className="relative w-full max-w-5xl bg-black rounded-lg overflow-hidden"
                onClick={(e) => e.stopPropagation()}
            >
                <button
                    onClick={() => setShowVideo(false)}
                    className="absolute top-4 right-4 z-10 w-10 h-10 bg-white/10 hover:bg-white/20 rounded-full flex items-center justify-center backdrop-blur-sm transition-colors"
                >
                    <span className="text-white text-2xl">×</span>
                </button>
                <video 
                    className="w-full"
                    controls
                    autoPlay
                >
                    <source src="/1.mp4" type="video/mp4" />
                    您的浏览器不支持视频播放。
                </video>
            </div>
        </div>
      )}

      {/* --- CAPABILITIES: Sticky Side Layout --- */}
      <section className="border-b border-black/10">
          <div className="flex flex-col md:flex-row">
              {/* Sticky Left Column */}
              <div className="md:w-[40%] md:h-screen sticky top-0 flex flex-col justify-center p-6 md:p-20 bg-white z-10 border-r border-black/10">
                  <FadeIn>
                      <span className="font-code text-sm uppercase tracking-widest text-gray-400 mb-8 block">System Overview</span>
                      <h2 className="text-4xl md:text-6xl font-serif leading-tight mb-10">
                          SpecSure:<br/>Hyperspectral<br/>Analysis System
                      </h2>
                      <p className="text-lg md:text-xl text-gray-600 font-sans leading-relaxed mb-10">
                          A comprehensive hyperspectral data analysis system supporting <span className="font-semibold text-gray-800">preprocessing → classification → visualization → metrics evaluation</span>.
                      </p>
                      <p className="text-base md:text-lg text-gray-500 font-sans italic mb-12 leading-relaxed">
                          Think of it as a mini ENVI + AI—lighter, faster, and specifically optimized for coastal remote sensing.
                      </p>
                      
                      {/* Feature List with Stagger Animation */}
                      <div className="space-y-5">
                          {[
                            'Supports hyperspectral cubes (HSI)',
                            'Supports spectral curve visualization',
                            'Supports model comparison: Traditional ML vs Deep Learning',
                            'Outputs classification maps, confusion matrices, OA/Kappa metrics and more'
                          ].map((feature, i) => (
                            <motion.div
                              key={i}
                              initial={{ opacity: 0, x: -20 }}
                              whileInView={{ opacity: 1, x: 0 }}
                              viewport={{ once: true, margin: "-10%" }}
                              transition={{ duration: 0.5, delay: 0.8 + i * 0.15 }}
                              className="flex items-start gap-4 group"
                            >
                              <motion.span 
                                className="text-green-600 font-bold text-2xl mt-1 flex-shrink-0"
                                initial={{ scale: 0 }}
                                whileInView={{ scale: 1 }}
                                viewport={{ once: true }}
                                transition={{ duration: 0.3, delay: 0.8 + i * 0.15 + 0.2, type: "spring" }}
                              >
                                ✔
                              </motion.span>
                              <span className="text-base md:text-lg text-gray-700 leading-relaxed group-hover:text-black transition-colors">
                                {feature}
                              </span>
                            </motion.div>
                          ))}
                      </div>
                  </FadeIn>
              </div>

              {/* Scrolling Right Column */}
              <div className="md:w-[60%] bg-white z-20 relative">
                  {features.map((item, i) => (
                      <div key={item.id} className="min-h-screen flex flex-col justify-center p-8 md:p-24 border-b border-black/5 last:border-b-0 bg-white">
                          <FadeIn>
                            <div className="flex items-center gap-4 mb-8">
                                <span className="font-code text-sm font-bold border-b border-black pb-1">
                                    {item.id}
                                </span>
                                <span className="font-code text-sm text-gray-400 uppercase tracking-wider">
                                    {item.subtitle}
                                </span>
                            </div>

                            {/* Image - Full Color, Subtle Parallax */}
                            <div className="aspect-square w-full overflow-hidden mb-12 bg-gray-100 rounded-lg">
                                <ParallaxImage src={item.image} alt={item.title} />
                            </div>

                            <h3 className="text-4xl md:text-5xl font-serif mb-6">{item.title}</h3>
                            <p className="text-xl md:text-2xl font-sans font-light leading-relaxed text-gray-600 max-w-2xl">
                                {item.desc}
                            </p>
                          </FadeIn>
                      </div>
                  ))}
              </div>
          </div>
      </section>

      {/* --- TEAM: Winding Journey Layout --- */}
      <section className="py-24 bg-[#FAFAFA] overflow-hidden relative">
          <div className="max-w-7xl mx-auto px-6 mb-12 text-center relative z-10">
              <FadeIn>
                  <h2 className="text-6xl md:text-9xl font-serif mb-6">The Journey</h2>
                  <p className="text-xl text-gray-500 font-sans">A team navigating the path to the future.</p>
              </FadeIn>
          </div>

          <WindingTeamSection />
          
      </section>

      {/* --- FOOTER: Clean & Clear --- */}
      <footer className="py-24 px-6 md:px-12 bg-white text-black border-t border-black/10 relative z-10">
          <div className="flex flex-col md:flex-row justify-between items-start gap-16 pt-12">
              <div className="md:w-1/2">
                  <h2 className="text-5xl md:text-7xl font-serif mb-8 leading-tight">
                      Ready to see<br/>the invisible?
                  </h2>
                  <a href="mailto:hello@specsure.ai" className="group flex items-center gap-4 text-2xl md:text-3xl font-sans hover:text-gray-600 transition-colors">
                      <span className="border-b border-black pb-1 group-hover:border-gray-600">hello@specsure.ai</span>
                      <span className="material-symbols-outlined text-3xl transition-transform group-hover:translate-x-2">arrow_forward</span>
                  </a>
              </div>
              
              <div className="md:w-1/2 flex flex-col md:flex-row gap-12 md:gap-24">
                  <div>
                      <h4 className="font-code text-xs uppercase tracking-widest text-gray-400 mb-6">Socials</h4>
                      <ul className="space-y-4 font-sans text-lg">
                          <li><a href="#" className="hover:underline">LinkedIn</a></li>
                          <li><a href="#" className="hover:underline">Twitter / X</a></li>
                          <li><a href="#" className="hover:underline">Instagram</a></li>
                      </ul>
                  </div>
                  <div>
                      <h4 className="font-code text-xs uppercase tracking-widest text-gray-400 mb-6">Office</h4>
                      <address className="not-italic font-sans text-lg text-gray-600 space-y-2">
                          <p>1200 Spectral Way</p>
                          <p>Shanghai, CN 200000</p>
                      </address>
                  </div>
              </div>
          </div>
          
          <div className="mt-24 flex justify-between items-end font-code text-xs text-gray-400 uppercase tracking-widest">
              <span>© 2025 BlueArray Intelligence</span>
              <span>SpecSure™ Platform</span>
          </div>
      </footer>

    </div>
  );
};

export default WordLayout;