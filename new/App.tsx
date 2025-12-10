import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import ContentSection from './components/ContentSection';
import StickyScroll from './components/StickyScroll';
import TeamCarousel from './components/TeamCarousel';
import Footer from './components/Footer';
import ParticleSystem from './components/ParticleSystem';
import AcademicBackground from './components/AcademicBackground';

function App() {
  const [theme, setTheme] = useState<'modern' | 'academic'>('modern');

  const toggleTheme = () => {
    setTheme(prev => prev === 'modern' ? 'academic' : 'modern');
  };

  return (
    <div data-theme={theme} className="min-h-screen text-on-surface selection:bg-primary selection:text-white transition-colors duration-500">
      
      {theme === 'modern' ? (
        <ParticleSystem theme={theme} />
      ) : (
        <AcademicBackground />
      )}
      
      <Navbar theme={theme} onToggleTheme={toggleTheme} />
      
      <main className="relative z-10">
        <Hero />
        
        <div className="space-y-24 md:space-y-48 pb-24">
          <ContentSection 
            tag="Precision"
            title="Hyperspectral Imaging"
            subtitle="Seeing the unseen"
            description="Our sensors capture hundreds of spectral bands per pixel, revealing chemical composition, moisture levels, and material properties invisible to standard cameras."
            image="https://picsum.photos/seed/hyperspectral1/1600/1200"
          />

          <ContentSection 
            tag="Intelligence"
            title="Neural Processing"
            description="SpecSure utilizes a proprietary lightweight neural network, SpecNet, designed to process high-dimensional spectral cubes in real-time on edge devices."
            image="https://picsum.photos/seed/neural/1600/1200"
            reversed={true}
          />

          <StickyScroll />

          {/* New Carousel Section */}
          <TeamCarousel />

          <ContentSection 
            tag="Application"
            title="Agricultural Monitor"
            subtitle="Crop health at scale"
            description="From drone-mounted surveys to stationary greenhouse monitors, SpecSure identifies pest infestation and nutrient deficiencies days before visual symptoms appear."
            image="https://picsum.photos/seed/agro/1600/1200"
          />
        </div>
      </main>

      <Footer />
    </div>
  );
}

export default App;