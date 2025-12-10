import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-grey-900 text-surface pt-24 pb-12 px-6 md:px-10 lg:px-[72px] mt-24 rounded-t-5xl mx-2 md:mx-4">
      <div className="max-w-[1800px] mx-auto w-full">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 mb-24">
          <div className="col-span-1 md:col-span-2 lg:col-span-2">
            <h2 className="text-4xl md:text-5xl font-light mb-6">SpecSure | 澜瞳</h2>
            <p className="text-grey-400 text-xl max-w-md">
              A BlueArray (潮霸) Innovation.
              <br/>
              Redefining perception through advanced hyperspectral intelligence.
            </p>
          </div>
          
          <div>
            <h4 className="text-lg font-medium mb-6">Platform</h4>
            <ul className="space-y-4 text-grey-400">
              <li><a href="#" className="hover:text-surface transition-colors">Hardware</a></li>
              <li><a href="#" className="hover:text-surface transition-colors">Software SDK</a></li>
              <li><a href="#" className="hover:text-surface transition-colors">Cloud Analysis</a></li>
              <li><a href="#" className="hover:text-surface transition-colors">Integration</a></li>
            </ul>
          </div>

          <div>
            <h4 className="text-lg font-medium mb-6">Group</h4>
            <ul className="space-y-4 text-grey-400">
              <li><a href="#" className="hover:text-surface transition-colors">About BlueArray</a></li>
              <li><a href="#" className="hover:text-surface transition-colors">Careers</a></li>
              <li><a href="#" className="hover:text-surface transition-colors">Contact</a></li>
              <li><a href="#" className="hover:text-surface transition-colors">Privacy</a></li>
            </ul>
          </div>
        </div>

        <div className="border-t border-grey-800 pt-8 flex flex-col md:flex-row justify-between items-center text-sm text-grey-400">
          <p>© {new Date().getFullYear()} BlueArray Group. All rights reserved.</p>
          <div className="flex gap-8 mt-4 md:mt-0 font-code">
            <span>CN / EN</span>
            <span>Est. 2024</span>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;