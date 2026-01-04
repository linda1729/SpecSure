import React from 'react';
import { motion } from 'framer-motion';

const teamMembers = [
  { id: 'linda', name: "linda1729", role: "后端", desc: "你永远不知道她做了多少接口" },
  { id: 'chen', name: "Chenmomo", role: "CNN", desc: "会CNN魔法的首席大法师（自称）" },
  { id: 'xixi', name: "xixiyhaha", role: "SVM", desc: "每天都在嘻嘻哈哈做SVM的木木大帅哥" },
  { id: 'keep', name: "KeepingMoving", role: "前端", desc: "前端审美的守门人（被三个甲方围攻，并随时准备造反 的乙方）" },
  { id: 'gong', name: "Gong", role: "负责人", desc: "伟大的负责人宫学长（真-首席大法师）" }
];

const TypewriterText: React.FC<{ text: string; delay?: number }> = ({ text, delay = 0 }) => {
  const characters = Array.from(text);
  
  return (
    <motion.span
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: "-10%" }}
      transition={{ staggerChildren: 0.05, delayChildren: delay }}
    >
      {characters.map((char, index) => (
        <motion.span
          key={index}
          variants={{
            hidden: { opacity: 0, y: 5 },
            visible: { opacity: 1, y: 0 }
          }}
          transition={{ duration: 0.2 }}
        >
          {char}
        </motion.span>
      ))}
    </motion.span>
  );
};

const TeamSection: React.FC = () => {
  return (
    <section className="py-24 px-6 md:px-10 lg:px-[72px] bg-surface">
      <div className="max-w-[1800px] mx-auto w-full">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-16 border-b border-grey-200 pb-8"
        >
          <span className="font-code text-sm uppercase tracking-widest text-primary mb-2 block">Our Squad</span>
          <h2 className="text-4xl md:text-5xl font-light text-on-surface">BlueArray Team | 潮霸小组</h2>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {teamMembers.map((member, idx) => (
            <motion.div
              key={member.id}
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true, margin: "-10%" }}
              transition={{ duration: 0.5, delay: idx * 0.1 }}
              className="group relative p-8 rounded-4xl bg-surface-container border border-transparent hover:border-grey-200 transition-all hover:shadow-lg overflow-hidden"
            >
              <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-20 transition-opacity">
                 <span className="text-9xl font-bold font-code text-grey-900 leading-none select-none">
                   {idx + 1}
                 </span>
              </div>

              <div className="relative z-10 flex flex-col h-full">
                <div className="mb-6">
                  <span className="inline-block px-3 py-1 rounded-full bg-grey-900 text-surface text-xs font-code mb-4">
                    {member.role}
                  </span>
                  <h3 className="text-3xl font-medium tracking-tight mb-2">{member.name}</h3>
                </div>
                
                <div className="mt-auto pt-6 border-t border-grey-200/50">
                  <p className="text-lg text-on-surface-variant font-sans leading-relaxed min-h-[3.5em]">
                    <TypewriterText text={member.desc} delay={0.2 + idx * 0.1} />
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TeamSection;