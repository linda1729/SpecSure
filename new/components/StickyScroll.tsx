import React from 'react';
import { motion } from 'framer-motion';

const features = [
  {
    title: "Spectral Analysis",
    desc: "Real-time decomposition of light spectrums beyond visible range.",
    metric: "400-1000nm"
  },
  {
    title: "AI Segmentation",
    desc: "Pixel-perfect material identification using proprietary ML models.",
    metric: "99.8% Acc"
  },
  {
    title: "Cloud Processing",
    desc: "Instantaneous data uplink and distributed processing for heavy workloads.",
    metric: "<50ms"
  }
];

const StickyScroll: React.FC = () => {
  return (
    <section className="py-24 px-6 md:px-10 lg:px-[72px] bg-surface-container">
       <div className="max-w-[1800px] mx-auto w-full">
         <div className="grid grid-cols-1 lg:grid-cols-12 gap-[64px]">
           
           <div className="lg:col-span-4 relative">
             <div className="sticky top-32">
               <span className="text-sm font-code uppercase tracking-wider text-grey-800 mb-4 block">Core Capabilities</span>
               <h2 className="text-5xl md:text-7xl tracking-tighter mb-8 text-on-surface">
                 Beyond <br/> Vision.
               </h2>
               <p className="text-xl text-on-surface-variant mb-8">
                 The SpecSure platform integrates hardware and software to see what the human eye cannot.
               </p>
               <button className="bg-on-surface text-surface rounded-full px-8 py-4 font-medium hover:scale-105 transition-transform">
                 View Specs
               </button>
             </div>
           </div>

           <div className="lg:col-span-7 lg:col-start-6 flex flex-col gap-24 py-12">
              {features.map((feature, idx) => (
                <motion.div 
                  key={idx}
                  initial={{ opacity: 0, y: 40 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ margin: "-20%" }}
                  transition={{ duration: 0.8 }}
                  className="bg-surface p-8 md:p-12 rounded-4xl shadow-sm border border-grey-100 hover:border-grey-200 transition-colors"
                >
                  <div className="flex justify-between items-start mb-8">
                    <div className="w-12 h-12 bg-surface-container rounded-full flex items-center justify-center">
                      <span className="font-code font-bold">{idx + 1}</span>
                    </div>
                    <span className="font-code text-3xl md:text-5xl text-grey-900 tracking-tighter opacity-20">{feature.metric}</span>
                  </div>
                  <h3 className="text-3xl font-medium mb-4">{feature.title}</h3>
                  <p className="text-lg text-on-surface-variant">{feature.desc}</p>
                </motion.div>
              ))}
           </div>

         </div>
       </div>
    </section>
  );
};

export default StickyScroll;