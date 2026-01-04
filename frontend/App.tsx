import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import ContentSection from './components/ContentSection';
import StickyScroll from './components/StickyScroll';
import TeamCarousel from './components/TeamCarousel';
import Footer from './components/Footer';
import ParticleSystem from './components/ParticleSystem';
import AcademicBackground from './components/AcademicBackground';
import WordLayout from './components/WordLayout';
import CustomCursor from './components/CustomCursor';

function App() {
  const [theme, setTheme] = useState<'modern' | 'academic' | 'word'>('modern');

  const toggleTheme = (newTheme: 'modern' | 'academic' | 'word') => {
    setTheme(newTheme);
  };

  return (
    <div data-theme={theme} className="min-h-screen text-on-surface selection:bg-primary selection:text-white transition-colors duration-500">
      
      {/* Backgrounds based on theme */}
      {theme === 'modern' && <ParticleSystem theme={theme} />}
      {theme === 'academic' && <AcademicBackground />}
      
      {/* Custom Cursor only for Word mode */}
      {theme === 'word' && <CustomCursor />}
      
      <Navbar theme={theme} onToggleTheme={toggleTheme} />
      
      {theme === 'word' ? (
        // Specialized Layout for Word Mode
        <WordLayout />
      ) : (
        // Standard Layout for Modern and Academic
        <main className="relative z-10">
          <Hero />
          
          <div className="space-y-24 md:space-y-48 pb-24">
            <ContentSection 
              id="technology-section"
              tag="Precision"
              title="Model A: Traditional ML"
              description="Support Vector Machines and Random Forest algorithms. Classic, reliable, efficient and stable approaches for hyperspectral classification."
              image="https://raw.githubusercontent.com/linda1729/SpecSure/refs/heads/feature-frontend/iamges/PU_pseudocolor_pca%3D15_window%3D25_lr%3D0.001_epochs%3D100.png"
            />

            <ContentSection 
              tag="Intelligence"
              title="Model B: Deep Learning"
              description="3D Convolutional Neural Networks and Hybrid Spectral-Spatial Networks. Fusing spatial-spectral information, specifically optimized for hyperspectral data analysis."
              image="https://raw.githubusercontent.com/linda1729/SpecSure/refs/heads/feature-frontend/iamges/IndianPines_pseudocolor_pca%3D30_window%3D25_lr%3D0.001_epochs%3D1.png"
              reversed={true}
            />

            <StickyScroll />

            {/* Team Carousel Section (Our Squad / Meet the Minds) */}
            <TeamCarousel />

            <ContentSection 
              id="applications-section"
              tag="introduction"
              title="SpecSure: Hyperspectral Analysis System"
              description="SpecSure is a comprehensive hyperspectral data analysis system supporting preprocessing → classification → visualization → metrics evaluation. Think of it as a mini ENVI + AI—lighter, faster, and specifically optimized for coastal remote sensing."
              items={[
                "Supports hyperspectral cubes (HSI)",
                "Supports spectral curve visualization",
                "Supports model comparison: Traditional ML vs Deep Learning",
                "Outputs classification maps, confusion matrices, OA/Kappa metrics and more"
              ]}
              image="https://raw.githubusercontent.com/linda1729/SpecSure/refs/heads/feature-frontend/iamges/img_5.png"
            />
          </div>
          <Footer />
        </main>
      )}

    </div>
  );
}

export default App;