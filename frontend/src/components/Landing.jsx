import React, { useState, useEffect, useRef } from 'react';

// Icon components
const ChevronRight = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>);
const ChevronLeft = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>);
const Globe = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>);
const Heart = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" /></svg>);
const Users = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" /></svg>);
const Zap = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>);
const MessageSquare = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>);
const Shield = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>);
const Award = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" /></svg>);
const Menu = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>);
const X = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>);
const ArrowRight = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" /></svg>);
const Play = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>);
const Star = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" /></svg>);
const MapPin = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg>);
const FileText = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>);

/**
 * FIXED: CountUp component now correctly handles both integer and float `end` values.
 */
const CountUp = ({ end, suffix = '' }) => {
  const [count, setCount] = useState(0);
  const countRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setIsVisible(true);
        observer.disconnect();
      }
    }, { threshold: 0.5 });
    if (countRef.current) observer.observe(countRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!isVisible) return;
    const timer = setInterval(() => {
      setCount(prevCount => {
        const step = (end - prevCount) / 10;
        const nextCount = prevCount + step;
        if (Math.abs(end - nextCount) < 0.01) {
          clearInterval(timer);
          return end;
        }
        return nextCount;
      });
    }, 50);
    return () => clearInterval(timer);
  }, [isVisible, end]);

  const displayedValue = () => {
    if (end % 1 !== 0) {
      const decimalPlaces = end.toString().split('.')[1]?.length || 2;
      return count.toFixed(decimalPlaces);
    }
    return Math.floor(count);
  };

  return <span ref={countRef}>{displayedValue()}{suffix}</span>;
};

/**
 * NEW: A helper component to parse stat values and render the CountUp component.
 */
const StatCounter = ({ value }) => {
  const numberMatch = value.match(/[\d.]+/);
  const suffixMatch = value.match(/[^\d.]+/);
  if (!numberMatch) return <span>{value}</span>;
  const end = parseFloat(numberMatch[0]);
  const suffix = suffixMatch ? suffixMatch[0] : '';
  return <CountUp end={end} suffix={suffix} />;
};

const Landing = ({ onTryNow, onViewDocumentation }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [activeTestimonial, setActiveTestimonial] = useState(0);
  const [hoveredFeature, setHoveredFeature] = useState(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => setActiveTestimonial(prev => (prev + 1) % 3), 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const handleMouseMove = (e) => setMousePosition({ x: e.clientX, y: e.clientY });
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const smoothScroll = (e, targetId) => {
    e.preventDefault();
    document.querySelector(targetId)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    setIsMenuOpen(false);
  };

  const features = [
    { icon: <Zap />, title: "MediaPipe Holistic Integration", description: "Extracts 1662-dimensional features from pose, face, and hand landmarks using standard webcams" },
    { icon: <Heart />, title: "Healthcare-Focused Vocabulary", description: "24 carefully curated medical and relational sign words for effective patient-doctor communication" },
    { icon: <Globe />, title: "Indian Sign Language Support", description: "Specifically trained on ISL gestures with custom-recorded dataset to address data scarcity" },
    { icon: <Shield />, title: "Attention-Based Bi-LSTM", description: "Advanced neural network architecture for accurate temporal sequence processing" },
    { icon: <Users />, title: "Visual Language Extension", description: "Supplementary emoji/image selection for expressing complex medical conditions" },
    { icon: <Award />, title: "LLM-Powered Translation", description: "Converts recognized signs into grammatically correct, contextually appropriate sentences" }
  ];

  const testimonials = [
    { name: "Dr. Priya Sharma", role: "Senior Physician, City Hospital", content: "INSIGNIA has transformed how I communicate with my deaf patients. The medical vocabulary and emoji integration make complex symptom discussions possible.", rating: 5 },
    { name: "Rajesh Kumar", role: "ISL Interpreter", content: "The 92.14% accuracy rate is impressive! The attention-based Bi-LSTM captures subtle gestures that other systems miss. A breakthrough for healthcare accessibility.", rating: 5 },
    { name: "Anjali Verma", role: "Patient Advocate", content: "Finally, a system designed specifically for Indian Sign Language in medical contexts. The visual supplement feature helps patients express symptoms clearly.", rating: 5 }
  ];
  
  const stats = [
    { value: "24", label: "Medical Signs" },
    { value: "92.14%", label: "Test Accuracy" },
    { value: "1662", label: "Feature Dimensions" },
    { value: "60fps", label: "Real-time Processing" }
  ];

  const handlePrevTestimonial = () => {
    setActiveTestimonial((prev) => (prev - 1 + testimonials.length) % testimonials.length);
  };

  const handleNextTestimonial = () => {
    setActiveTestimonial((prev) => (prev + 1) % testimonials.length);
  };

  const handleViewDocumentation = () => {
    if (onViewDocumentation) {
      onViewDocumentation();
    } else {
      window.open('/docs/documentation.pdf', '_blank');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 text-white font-sans overflow-x-hidden">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-20 -right-20 sm:-top-40 sm:-right-40 w-40 h-40 sm:w-80 sm:h-80 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" style={{ transform: `translate(${mousePosition.x * 0.05}px, ${mousePosition.y * 0.05}px)` }}></div>
        <div className="absolute -bottom-20 -left-20 sm:-bottom-40 sm:-left-40 w-40 h-40 sm:w-80 sm:h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse animation-delay-2000" style={{ transform: `translate(${-mousePosition.x * 0.05}px, ${-mousePosition.y * 0.05}px)` }}></div>
        <div className="absolute top-1/2 left-1/2 w-48 h-48 sm:w-96 sm:h-96 bg-indigo-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse animation-delay-4000" style={{ transform: `translate(calc(-50% + ${mousePosition.x * 0.03}px), calc(-50% + ${mousePosition.y * 0.03}px))` }}></div>
      </div>

      <nav className={`fixed top-0 w-full z-50 transition-all duration-300 ${scrolled ? 'bg-gray-900/95 backdrop-blur-md shadow-lg' : 'bg-transparent'}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-2">
              <img src="/images/logo.png" alt="INSIGNIA Logo" className="w-10 h-10 sm:w-12 sm:h-12 object-contain" />
              <span className="text-lg sm:text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">INSIGNIA</span>
            </div>
            <div className="hidden md:flex items-center space-x-8">
              <a href="#features" onClick={(e) => smoothScroll(e, '#features')} className="hover:text-cyan-400 transition-colors">Features</a>
              <a href="#how-it-works" onClick={(e) => smoothScroll(e, '#how-it-works')} className="hover:text-cyan-400 transition-colors">How it Works</a>
              <a href="#testimonials" onClick={(e) => smoothScroll(e, '#testimonials')} className="hover:text-cyan-400 transition-colors">Testimonials</a>
              <a href="#contact" onClick={(e) => smoothScroll(e, '#contact')} className="hover:text-cyan-400 transition-colors">Contact</a>
              <button onClick={onTryNow} className="px-6 py-2 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full hover:from-cyan-600 hover:to-blue-700 transition-all transform hover:scale-105 shadow-lg relative overflow-hidden group">
                <span className="absolute inset-0 bg-white opacity-0 group-hover:opacity-20 transition-opacity duration-300"></span><span className="relative">Try Now</span>
              </button>
            </div>
            <button onClick={() => setIsMenuOpen(!isMenuOpen)} className="md:hidden p-2 hover:bg-gray-800 rounded-lg transition-colors" aria-label="Open menu">{isMenuOpen ? <X className="w-5 h-5 sm:w-6 sm:h-6" /> : <Menu className="w-5 h-5 sm:w-6 sm:h-6" />}</button>
          </div>
        </div>
        <div className={`md:hidden fixed inset-0 z-40 transition-all duration-300 ${isMenuOpen ? 'visible' : 'invisible'}`}>
          <div className={`absolute inset-0 bg-gray-900/80 backdrop-blur-sm transition-opacity duration-300 ${isMenuOpen ? 'opacity-100' : 'opacity-0'}`} onClick={() => setIsMenuOpen(false)}></div>
          <div className={`absolute right-0 top-0 h-full w-64 max-w-[80vw] bg-gray-900/95 backdrop-blur-md transform transition-transform duration-300 ${isMenuOpen ? 'translate-x-0' : 'translate-x-full'}`}>
            <div className="p-6">
              <button onClick={() => setIsMenuOpen(false)} className="mb-8 ml-auto block" aria-label="Close menu"><X className="w-6 h-6" /></button>
              <div className="space-y-4">
                <a href="#features" onClick={(e) => smoothScroll(e, '#features')} className="block px-4 py-3 hover:bg-gray-800 rounded-lg transition-all hover:translate-x-2 transform">Features</a>
                <a href="#how-it-works" onClick={(e) => smoothScroll(e, '#how-it-works')} className="block px-4 py-3 hover:bg-gray-800 rounded-lg transition-all hover:translate-x-2 transform">How it Works</a>
                <a href="#testimonials" onClick={(e) => smoothScroll(e, '#testimonials')} className="block px-4 py-3 hover:bg-gray-800 rounded-lg transition-all hover:translate-x-2 transform">Testimonials</a>
                <a href="#contact" onClick={(e) => smoothScroll(e, '#contact')} className="block px-4 py-3 hover:bg-gray-800 rounded-lg transition-all hover:translate-x-2 transform">Contact</a>
                <button onClick={() => { onTryNow(); setIsMenuOpen(false); }} className="w-full mt-4 px-4 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-lg hover:from-cyan-600 hover:to-blue-700 transition-all">Try Now</button>
              </div>
            </div>
          </div>
        </div>
      </nav>

      <section id="hero" className="relative pt-24 sm:pt-32 pb-16 sm:pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-8 lg:gap-12 items-center">
            <div className="space-y-6 sm:space-y-8 animate-fade-in text-center lg:text-left">
              <div className="inline-flex items-center px-3 sm:px-4 py-2 bg-gray-800/50 rounded-full border border-gray-700"><span className="w-2 h-2 bg-green-400 rounded-full animate-pulse mr-2"></span><span className="text-xs sm:text-sm">Bi-LSTM • MediaPipe • Healthcare-Focused</span></div>
              <h1 className="text-4xl sm:text-5xl lg:text-7xl font-bold leading-tight">Bridging Communication with <span className="bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent animate-gradient">Indian Sign Language AI</span></h1>
              <p className="text-lg sm:text-xl text-gray-300 leading-relaxed max-w-2xl mx-auto lg:mx-0">INSIGNIA: An intelligent sign language recognition system enabling deaf and mute patients to communicate symptoms and medical needs through real-time gesture recognition with 92.14% accuracy.</p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                <button onClick={onTryNow} className="group relative px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full hover:from-cyan-600 hover:to-blue-700 transition-all transform hover:scale-105 shadow-xl flex items-center justify-center overflow-hidden"><span className="absolute inset-0 bg-white opacity-0 group-hover:opacity-20 transition-opacity duration-300"></span><Play className="w-5 h-5 mr-2 group-hover:animate-pulse" /><span className="font-semibold">Try Live Demo</span><ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" /></button>
                <button onClick={handleViewDocumentation} className="group px-8 py-4 border border-gray-600 rounded-full hover:bg-gray-800 hover:border-cyan-500 transition-all flex items-center justify-center relative overflow-hidden"><span className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 to-blue-600/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span><FileText className="w-5 h-5 mr-2" /><span className="relative">View Documentation</span><ChevronRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" /></button>
              </div>
              <div className="flex items-center justify-center lg:justify-start space-x-4 sm:space-x-8 pt-4">{stats.map((stat) => (<div key={stat.label} className="text-center min-w-[80px] transform hover:scale-110 transition-transform duration-200"><div className="text-xl sm:text-2xl font-bold text-cyan-400"><StatCounter value={stat.value} /></div><div className="text-xs sm:text-sm text-gray-400">{stat.label}</div></div>))}</div>
            </div>
            <div className="relative"><div className="relative rounded-2xl overflow-hidden shadow-2xl"><img src="https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=800&h=600&fit=crop" alt="Doctor typing on a laptop with a stethoscope nearby, symbolizing digital healthcare." className="w-full h-auto" /><div className="absolute inset-0 bg-gradient-to-t from-gray-900/80 via-transparent to-transparent"></div><div className="absolute bottom-6 left-6 right-6"><div className="bg-gray-900/80 backdrop-blur-md rounded-lg p-4 border border-gray-700"><div className="flex items-center justify-between"><div className="flex items-center space-x-3"><div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div><span className="text-sm">Live Translation Active</span></div><span className="text-xs text-gray-400">92.14% Accuracy</span></div></div></div></div><div className="absolute -top-4 -right-4 bg-gradient-to-br from-purple-600 to-pink-600 rounded-lg p-3 shadow-xl animate-float"><Heart className="w-6 h-6" /></div><div className="absolute -bottom-4 -left-4 bg-gradient-to-br from-cyan-600 to-blue-600 rounded-lg p-3 shadow-xl animate-float animation-delay-2000"><Globe className="w-6 h-6" /></div></div>
          </div>
        </div>
      </section>

      <section id="features" className="py-16 sm:py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto"><div className="text-center mb-12 sm:mb-16"><h2 className="text-3xl sm:text-4xl font-bold mb-4">Powerful Features for Healthcare Communication</h2><p className="text-lg sm:text-xl text-gray-400 max-w-3xl mx-auto">Combining cutting-edge AI with practical medical applications</p></div><div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">{features.map((feature, index) => (<div key={index} className="group relative bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 hover:border-cyan-500 transition-all duration-300 hover:transform hover:-translate-y-2 hover:shadow-xl cursor-pointer overflow-hidden" onMouseEnter={() => setHoveredFeature(index)} onMouseLeave={() => setHoveredFeature(null)}><div className={`absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-blue-600/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300`}></div><div className="relative z-10"><div className={`w-10 h-10 sm:w-12 sm:h-12 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center mb-4 transition-all duration-300 ${hoveredFeature === index ? 'scale-110 rotate-3' : ''}`}>{React.cloneElement(feature.icon, { className: "w-5 h-5 sm:w-6 sm:h-6 text-white" })}</div><h3 className="text-lg sm:text-xl font-semibold mb-2 group-hover:text-cyan-400 transition-colors">{feature.title}</h3><p className="text-sm sm:text-base text-gray-400 group-hover:text-gray-300 transition-colors">{feature.description}</p><div className="absolute bottom-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity"><ArrowRight className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-400" /></div></div></div>))}</div></div>
      </section>

      <section id="how-it-works" className="py-16 sm:py-20 px-4 sm:px-6 lg:px-8 bg-gray-800/30">
        <div className="max-w-7xl mx-auto"><div className="text-center mb-12 sm:mb-16"><h2 className="text-3xl sm:text-4xl font-bold mb-4">How It Works</h2><p className="text-lg sm:text-xl text-gray-400">Three simple steps to start communicating</p></div><div className="grid sm:grid-cols-2 md:grid-cols-3 gap-6 sm:gap-8">{[{step:"1",title:"Video Capture",description:"Record sign language gestures using a standard webcam. The system processes video streams in real-time.",icon:"🎥"},{step:"2",title:"Feature Extraction",description:"MediaPipe Holistic extracts 1662-dimensional features including pose, face, and hand landmarks from each frame.",icon:"🔍"},{step:"3",title:"AI Recognition & Translation",description:"Bi-LSTM with attention processes sequences, then LLM generates grammatically correct sentences.",icon:"🧠"}].map((item, index, arr) => (<div key={index} className="relative group"><div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 sm:p-8 border border-gray-700 hover:border-cyan-500 transition-all transform hover:-translate-y-2 hover:shadow-xl"><div className="text-4xl sm:text-5xl mb-4 transform group-hover:scale-110 transition-transform">{item.icon}</div><div className="text-cyan-400 font-bold mb-2 text-sm">Step {item.step}</div><h3 className="text-xl sm:text-2xl font-semibold mb-3">{item.title}</h3><p className="text-gray-400 text-sm sm:text-base">{item.description}</p></div>{index < arr.length - 1 && (<div className="hidden md:block absolute top-1/2 -right-4 transform -translate-y-1/2 z-10"><ChevronRight className="w-8 h-8 text-gray-600 animate-pulse" /></div>)}</div>))}</div></div>
      </section>

      <section id="testimonials" className="py-16 sm:py-20 px-4 sm:px-6 lg:px-8 pb-24">
        <div className="max-w-7xl mx-auto"><div className="text-center mb-12 sm:mb-16"><h2 className="text-3xl sm:text-4xl font-bold mb-4">Advancing ISL Recognition Research</h2><p className="text-lg sm:text-xl text-gray-400 max-w-3xl mx-auto">Built on the INCLUDE dataset with custom medical vocabulary extensions</p></div><div className="relative max-w-4xl mx-auto"><div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-6 sm:p-8 md:p-12 border border-gray-700 overflow-hidden relative min-h-[250px]">{testimonials.map((testimonial, index) => (<div key={index} className={`transition-opacity duration-500 absolute inset-0 p-6 sm:p-8 md:p-12 ${activeTestimonial === index ? 'opacity-100' : 'opacity-0'}`}><div className="flex mb-4">{[...Array(testimonial.rating)].map((_, i) => (<Star key={i} className="w-4 h-4 sm:w-5 sm:h-5 text-yellow-400 fill-current" />))}</div><p className="text-lg sm:text-xl md:text-2xl mb-6 leading-relaxed">"{testimonial.content}"</p><div><p className="font-semibold text-sm sm:text-base">{testimonial.name}</p><p className="text-gray-400 text-sm">{testimonial.role}</p></div></div>))}</div>
        <button onClick={handlePrevTestimonial} className="absolute left-2 sm:left-0 top-1/2 -translate-y-1/2 sm:-translate-x-4 lg:-translate-x-6 bg-gray-800/80 backdrop-blur-sm p-2 sm:p-3 rounded-full border border-gray-700 hover:border-cyan-500 hover:bg-gray-700 transition-all" aria-label="Previous testimonial"><ChevronLeft className="w-5 h-5 sm:w-6 sm:h-6" /></button>
        <button onClick={handleNextTestimonial} className="absolute right-2 sm:right-0 top-1/2 -translate-y-1/2 sm:translate-x-4 lg:translate-x-6 bg-gray-800/80 backdrop-blur-sm p-2 sm:p-3 rounded-full border border-gray-700 hover:border-cyan-500 hover:bg-gray-700 transition-all" aria-label="Next testimonial"><ChevronRight className="w-5 h-5 sm:w-6 sm:h-6" /></button>
        <div className="flex justify-center mt-6 space-x-2">{testimonials.map((_, index) => (<button key={index} onClick={() => setActiveTestimonial(index)} className={`w-2 h-2 rounded-full transition-all ${activeTestimonial === index ? 'bg-cyan-400 w-8' : 'bg-gray-600'}`} aria-label={`View testimonial ${index + 1}`}/>))}</div></div></div>
      </section>

      <section id="cta" className="py-16 sm:py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center"><div className="bg-gradient-to-r from-cyan-600 to-blue-600 rounded-2xl p-8 sm:p-12 shadow-2xl relative overflow-hidden"><div className="absolute inset-0 bg-black/10"></div><div className="relative z-10"><h2 className="text-3xl sm:text-4xl font-bold mb-4">Experience INSIGNIA in Action</h2><p className="text-lg sm:text-xl mb-8 text-cyan-100">Bridging the communication gap in healthcare with AI-powered ISL recognition</p><button onClick={onTryNow} className="group px-6 sm:px-8 py-3 sm:py-4 bg-white text-gray-900 rounded-full hover:bg-gray-100 transition-all transform hover:scale-105 shadow-xl font-semibold inline-flex items-center justify-center">Launch Demo Application<ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" /></button><p className="mt-4 text-xs sm:text-sm text-cyan-100">Academic Project • B.Tech CSE 2024-25</p></div></div></div>
      </section>

      <footer id="contact" className="bg-gray-900 border-t border-gray-800 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto"><div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8"><div className="col-span-1 sm:col-span-2 lg:col-span-1"><div className="flex items-center space-x-2 mb-4"><img src="/images/logo.png" alt="INSIGNIA Logo" className="w-10 h-10 sm:w-12 sm:h-12 object-contain" /><span className="text-lg sm:text-xl font-bold">INSIGNIA</span></div><p className="text-gray-400">A Sign Language Recognition System with Attention-Based Bi-LSTM and LLM.</p><p className="text-sm text-gray-500 mt-2">Developed at Netaji Subhash Engineering College</p></div><div><h3 className="font-semibold mb-4">Research</h3><ul className="space-y-2 text-gray-400"><li><a href="#" className="hover:text-white transition-colors">Technical Paper</a></li><li><a href="#" className="hover:text-white transition-colors">Dataset Info</a></li><li><a href="#" className="hover:text-white transition-colors">Model Architecture</a></li><li><a href="#" className="hover:text-white transition-colors">Results & Analysis</a></li></ul></div><div><h3 className="font-semibold mb-4">Team</h3><ul className="space-y-2 text-gray-400"><li><a href="#" className="hover:text-white transition-colors text-sm">Sourik Roy</a></li><li><a href="#" className="hover:text-white transition-colors text-sm">Mayukh Ganguly</a></li><li><a href="#" className="hover:text-white transition-colors text-sm">Riddhi Mondal</a></li><li><a href="#" className="hover:text-white transition-colors text-sm">Rudranil Bhattacharjee</a></li><li><a href="#" className="hover:text-white transition-colors text-sm">Harshit Narayan Trivedi</a></li></ul></div><div><h3 className="font-semibold mb-4">Project Details</h3><ul className="space-y-3 text-gray-400"><li className="flex items-start space-x-3"><Award className="w-4 h-4 sm:w-5 sm:h-5 mt-0.5 flex-shrink-0" /><span className="text-xs sm:text-sm">Guided by Dr. Chandra Das</span></li><li className="flex items-start space-x-3"><MapPin className="w-4 h-4 sm:w-5 sm:h-5 mt-0.5 flex-shrink-0" /><span className="text-xs sm:text-sm">NSEC, Garia, Kolkata - 700152</span></li><li className="flex items-start space-x-3"><Star className="w-4 h-4 sm:w-5 sm:h-5 mt-0.5 flex-shrink-0" /><span className="text-xs sm:text-sm">Academic Year 2024-25</span></li></ul></div></div><div className="mt-8 sm:mt-12 pt-8 border-t border-gray-800 flex flex-col sm:flex-row justify-between items-center space-y-4 sm:space-y-0"><p className="text-gray-400 text-xs sm:text-sm text-center sm:text-left">© 2024 INSIGNIA Project. Netaji Subhash Engineering College.</p><div className="flex space-x-4 sm:space-x-6"><a href="#" className="text-gray-400 hover:text-white transition-colors text-xs sm:text-sm">Project Report</a><a href="#" className="hover:text-white transition-colors text-xs sm:text-sm">GitHub Repository</a><a href="#" className="hover:text-white transition-colors text-xs sm:text-sm">Documentation</a></div></div></div>
      </footer>

      <style jsx>{`@keyframes float{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-10px) rotate(3deg)}}.animate-float{animation:float 3s ease-in-out infinite}.animation-delay-2000{animation-delay:2s}.animation-delay-4000{animation-delay:4s}@media (max-width:640px){.animate-float{animation-duration:4s}}@keyframes gradient{0%{background-position:0 50%}50%{background-position:100% 50%}100%{background-position:0 50%}}.animate-gradient{background-size:200% 200%;animation:gradient 5s ease infinite}@keyframes fade-in{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}.animate-fade-in{animation:fade-in 1s ease-out}`}</style>
    </div>
  );
};

export default Landing;