import React, { Suspense, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, MeshDistortMaterial, Environment, Sphere, Torus } from '@react-three/drei';
import * as THREE from 'three';

// --- Types ---
interface QuantumParticleProps {
  position: [number, number, number];
  color: string;
  scale?: number;
  speed?: number;
  wireframe?: boolean;
}

interface MacroscopicWaveProps {
  color?: string;
  wireframe?: boolean;
}

/**
 * COMPONENT 1: The Floating "Liquid" Sphere
 * Implements the specific distorion and manual animation logic.
 */
const QuantumParticle: React.FC<QuantumParticleProps> = ({ 
  position, 
  color, 
  scale = 1, 
  speed = 1, 
  wireframe = false 
}) => {
  const ref = useRef<THREE.Mesh>(null);
  
  // useFrame runs 60 times per second. 
  useFrame((state) => {
    if (ref.current) {
      const t = state.clock.getElapsedTime();
      // Add a gentle vertical bobbing motion separate from the <Float> wrapper
      // This matches the snippet's logic: pos.y = base + sin(...)
      ref.current.position.y = position[1] + Math.sin(t * 2 + position[0]) * 0.1; // Reduced amplitude slightly for background
      // Slow rotation to show off the lighting reflections
      ref.current.rotation.x = t * 0.5 * speed;
      ref.current.rotation.z = t * 0.3 * speed;
    }
  });

  return (
    <Sphere ref={ref} args={[1, 64, 64]} position={position} scale={scale}>
      {/* 
         MeshDistortMaterial settings from snippet:
         distort={0.4} for liquid look, speed scaled by prop.
      */}
      <MeshDistortMaterial
        color={color}
        envMapIntensity={wireframe ? 0.5 : 1}
        clearcoat={wireframe ? 0 : 1}        
        clearcoatRoughness={0.1}
        metalness={wireframe ? 0.1 : 0.5} 
        distort={0.4}                        
        speed={2 * speed}                    
        wireframe={wireframe}
      />
    </Sphere>
  );
};

/**
 * COMPONENT 2: The Giant Rotating Ring
 * Represents the "Macroscopic Wave Function".
 */
const MacroscopicWave: React.FC<MacroscopicWaveProps> = ({ 
  color = "#C5A059", 
  wireframe = true 
}) => {
  const ref = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (ref.current) {
       const t = state.clock.getElapsedTime();
       // Complex rotation: Main rotation on Y, slight wobble on X
       ref.current.rotation.x = Math.sin(t * 0.2) * 0.2; // Gentle wobble
       ref.current.rotation.y = t * 0.1; // Continuous slow spin
    }
  });

  return (
    // Torus args: [radius, tube_diameter, radial_segments, tubular_segments]
    // Matches snippet: [3, 0.1, 16, 100]
    <Torus ref={ref} args={[3, 0.1, 16, 100]} rotation={[Math.PI / 2, 0, 0]}>
      <meshStandardMaterial 
        color={color} 
        emissive={color} 
        emissiveIntensity={0.5} 
        transparent 
        opacity={0.6}
        wireframe={wireframe} // WIREFRAME MODE: Key to the technical look
        roughness={0.5}
        metalness={0.8}
      />
    </Torus>
  );
};

const AcademicBackground: React.FC = () => {
  return (
    <div className="fixed inset-0 pointer-events-none -z-10 h-screen w-full transition-opacity duration-1000 ease-in-out opacity-100">
      <Canvas 
        camera={{ position: [0, 0, 6], fov: 45 }} 
        gl={{ alpha: true, antialias: true }}
        dpr={[1, 2]}
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} color="#ffffff" />
        <directionalLight position={[-10, -5, -5]} intensity={1} color="#C5A059" />
        
        <Suspense fallback={null}>
            {/* Environment: "City" for realistic reflections */}
            <Environment preset="city" />

            {/* --- COMPONENT 1: The Macroscopic Wave --- */}
            <Float speed={1.5} rotationIntensity={0.2} floatIntensity={0.5}>
               <MacroscopicWave color="#C5A059" wireframe={true} />
            </Float>

            {/* --- COMPONENT 2: Quantum Particles --- */}
            
            {/* Center: Ether (Cream/Glassy) - Represents the core */}
            <Float speed={1.5} rotationIntensity={0.2} floatIntensity={0.5}>
              <QuantumParticle 
                  position={[0, 0, -1]} 
                  scale={1.5} 
                  color="#F9F8F4" 
                  speed={0.8}
              />
            </Float>

            {/* Left: Stone (Grey/Ceramic) - Secondary element */}
            <Float speed={2} rotationIntensity={0.5} floatIntensity={1}>
                <QuantumParticle 
                    position={[-2.8, 1.2, 0.5]} 
                    scale={0.6} 
                    color="#D6D3D1" 
                    speed={1.2}
                />
            </Float>

            {/* Right: Gold (Metallic) - Accent element */}
            <Float speed={1.8} rotationIntensity={0.4} floatIntensity={0.8}>
                <QuantumParticle 
                    position={[2.5, -1.5, 0.5]} 
                    scale={0.7} 
                    color="#C5A059" 
                    speed={1}
                />
            </Float>
            
        </Suspense>
      </Canvas>
      
      {/* Paper Grain Texture Overlay */}
      <div className="absolute inset-0 opacity-[0.03] pointer-events-none mix-blend-multiply" 
           style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='1'/%3E%3C/svg%3E")` }}>
      </div>
      
      {/* 
         Layer 1 (Mask Layer): 
         Radial gradient semi-transparent mask to reduce background contrast 
         and ensure text readability.
      */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(249,248,244,0.4)_0%,rgba(249,248,244,1)_85%)] pointer-events-none"></div>
    </div>
  );
};

export default AcademicBackground;