import React from 'react';
import FloatingElement from './FloatingElement';

interface ContentSectionProps {
  title: string;
  subtitle?: string;
  description: string;
  image: string;
  reversed?: boolean;
  tag: string;
  id?: string;
  items?: string[];
}

const ContentSection: React.FC<ContentSectionProps> = ({ 
  title, 
  subtitle, 
  description, 
  image, 
  reversed = false,
  tag,
  id,
  items
}) => {
  return (
    <section id={id} className="py-24 md:py-36 px-6 md:px-10 lg:px-[72px]">
      <div className="max-w-[1800px] mx-auto w-full">
        <div className={`grid grid-cols-1 lg:grid-cols-12 gap-x-[64px] gap-y-16 items-center ${reversed ? 'direction-rtl' : ''}`}>
          
          {/* Text Column */}
          <div className={`lg:col-span-5 ${reversed ? 'lg:order-2' : 'lg:order-1'}`}>
            <FloatingElement speed={0.2} delay={0.1}>
              <span className="inline-block py-1 px-3 rounded-full border border-grey-200 bg-surface-container text-xs font-code uppercase tracking-wider mb-6">
                {tag}
              </span>
              <h3 className="text-4xl md:text-5xl lg:text-6xl font-medium tracking-tight mb-6 leading-[1.1] text-on-surface whitespace-nowrap">
                {title}
                {subtitle && <span className="block text-on-surface-variant opacity-60 mt-2">{subtitle}</span>}
              </h3>
              <p className="text-xl md:text-2xl lg:text-2xl text-on-surface-variant leading-relaxed max-w-2xl">
                {description}
              </p>
              
              {items && items.length > 0 && (
                <ul className="mt-10 space-y-5">
                  {items.map((item, idx) => (
                    <li key={idx} className="flex items-start gap-4 text-xl md:text-2xl text-on-surface-variant">
                      <span className="text-primary font-bold mt-0.5 flex-shrink-0 text-2xl">✔</span>
                      <span className="leading-relaxed">{item}</span>
                    </li>
                  ))}
                </ul>
              )}
              
              <div className="mt-10">
                <button className="arrow-link group inline-flex items-center text-lg font-medium hover:opacity-70 transition-opacity text-primary">
                  Learn more
                  <span className="ml-2 font-symbol material-symbols-outlined group-hover:translate-x-1 transition-transform">→</span>
                </button>
              </div>
            </FloatingElement>
          </div>

          {/* Spacer Column */}
          <div className={`hidden lg:block lg:col-span-2 ${reversed ? 'lg:order-3' : 'lg:order-2'}`}></div>

          {/* Image Column */}
          <div className={`lg:col-span-5 ${reversed ? 'lg:order-1' : 'lg:order-3'}`}>
             <FloatingElement speed={0.4} className="relative group cursor-pointer max-w-[500px] ml-auto mt-16">
                {/* 
                   Interactive Image Container:
                   - Scale 1.03 on hover
                   - Soft shadow on hover
                   - Nobel Gold border fade in
                */}
                <div className="rounded-4xl md:rounded-5xl overflow-hidden bg-surface-container-high transition-all duration-500 ease-out transform group-hover:scale-[1.03] group-hover:shadow-[0_20px_40px_rgba(197,160,89,0.15)] border-2 border-primary/20 group-hover:border-primary/30">
                   <div className="w-full aspect-square relative">
                      <img 
                        src={image} 
                        alt={title} 
                        className="absolute inset-0 w-full h-full object-cover transition-transform duration-700"
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