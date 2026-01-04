import React from 'react';
import FloatingElement from './FloatingElement';

interface ContentSectionProps {
  title: string;
  subtitle?: string;
  description: string;
  image: string;
  reversed?: boolean;
  tag: string;
}

const ContentSection: React.FC<ContentSectionProps> = ({ 
  title, 
  subtitle, 
  description, 
  image, 
  reversed = false,
  tag
}) => {
  return (
    <section className="py-24 md:py-36 px-6 md:px-10 lg:px-[72px]">
      <div className="max-w-[1800px] mx-auto w-full">
        <div className={`grid grid-cols-1 lg:grid-cols-12 gap-x-[64px] gap-y-16 items-center ${reversed ? 'direction-rtl' : ''}`}>
          
          {/* Text Column */}
          <div className={`lg:col-span-5 ${reversed ? 'lg:order-2' : 'lg:order-1'}`}>
            <FloatingElement speed={0.2} delay={0.1}>
              <span className="inline-block py-1 px-3 rounded-full border border-grey-200 bg-surface-container text-xs font-code uppercase tracking-wider mb-6">
                {tag}
              </span>
              <h3 className="text-4xl md:text-5xl lg:text-6xl font-medium tracking-tight mb-6 leading-[1.1] text-on-surface">
                {title}
                {subtitle && <span className="block text-on-surface-variant opacity-60 mt-2">{subtitle}</span>}
              </h3>
              <p className="text-lg md:text-xl text-on-surface-variant leading-relaxed max-w-md">
                {description}
              </p>
              
              <div className="mt-10">
                <button className="arrow-link group inline-flex items-center text-lg font-medium hover:opacity-70 transition-opacity text-primary">
                  Learn more
                  <span className="ml-2 font-symbol material-symbols-outlined group-hover:translate-x-1 transition-transform">→</span>
                </button>
              </div>
            </FloatingElement>
          </div>

          {/* Spacer Column */}
          <div className={`hidden lg:block lg:col-span-1 ${reversed ? 'lg:order-3' : 'lg:order-2'}`}></div>

          {/* Image Column */}
          <div className={`lg:col-span-6 ${reversed ? 'lg:order-1' : 'lg:order-3'}`}>
             <FloatingElement speed={0.4} className="relative group cursor-pointer">
                {/* 
                   Interactive Image Container:
                   - Scale 1.03 on hover
                   - Soft shadow on hover
                   - Nobel Gold border fade in
                */}
                <div className="rounded-4xl md:rounded-5xl overflow-hidden bg-surface-container-high transition-all duration-500 ease-out transform group-hover:scale-[1.03] group-hover:shadow-[0_20px_40px_rgba(197,160,89,0.15)] border-2 border-transparent group-hover:border-primary/30">
                   <div className="aspect-[4/3] relative">
                      <img 
                        src={image} 
                        alt={title} 
                        className="w-full h-full object-cover transition-transform duration-700"
                      />
                      {/* Overlay gradient for depth */}
                      <div className="absolute inset-0 bg-gradient-to-tr from-on-surface/5 to-transparent pointer-events-none" />
                   </div>
                </div>
                
                {/* Decorative Icon Badge - Floats independently */}
                <div className={`absolute -bottom-8 ${reversed ? '-left-8' : '-right-8'} w-24 h-24 bg-surface rounded-3xl shadow-lg flex items-center justify-center z-10 transition-transform duration-500 group-hover:translate-y-[-10px]`}>
                   <span className="font-symbol text-3xl text-primary">deployed_code</span>
                </div>
             </FloatingElement>
          </div>

        </div>
      </div>
    </section>
  );
};

export default ContentSection;