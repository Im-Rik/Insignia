import React, { useState, useEffect, useRef } from 'react';

// A simple utility for conditional class names
const clsx = (...classes) => classes.filter(Boolean).join(' ');

// Icon components
const ChevronRight = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>);
const ChevronLeft = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>);
const Globe = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>);
const Heart = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" /></svg>);
const Users = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" /></svg>);
const Zap = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>);
const Shield = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>);
const Award = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" /></svg>);
const Menu = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>);
const X = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>);
const ArrowRight = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" /></svg>);
const Play = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>);
const Star = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" /></svg>);
const MapPin = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg>);
const FileText = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>);
const Cpu = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 15V9a2 2 0 012-2h10a2 2 0 012 2v6a2 2 0 01-2 2H7a2 2 0 01-2-2z"/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 15v3H5v-3m14 0v3h-4v-3M9 9V6h6v3M4 12h1M20 12h-1m-3-4V4h-4v4m0 12v-4h4v4"/></svg>);
const VideoCamera = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>);
const BrainCircuit = ({ className }) => (<svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 8h6M9 12h6m-6 4h6m2-12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>);

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
        let animationFrameId;
        const start = 0;
        const duration = 2000; // 2 seconds
        let startTimestamp = null;

        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const current = progress * (end - start);
            setCount(current);
            if (progress < 1) {
                animationFrameId = window.requestAnimationFrame(step);
            } else {
                setCount(end); // Ensure it ends exactly on the end value
            }
        };

        animationFrameId = window.requestAnimationFrame(step);
        return () => window.cancelAnimationFrame(animationFrameId);
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
    const [isNavbarTop, setIsNavbarTop] = useState(true);
    const [hoveredFeature, setHoveredFeature] = useState(null);
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
    const [visibleSteps, setVisibleSteps] = useState({});
    const timelineRefs = useRef([]);

    useEffect(() => {
        const handleScroll = () => {
            const isTop = window.scrollY < 20;
            setScrolled(!isTop);
            setIsNavbarTop(isTop);
        };
        window.addEventListener('scroll', handleScroll, { passive: true });
        handleScroll();

        const observers = [];
        timelineRefs.current.forEach((ref, index) => {
            if (!ref) return;
            const observer = new IntersectionObserver(([entry]) => {
                if (entry.isIntersecting) {
                    setVisibleSteps(prev => ({ ...prev, [index]: true }));
                }
            }, { threshold: 0.3 });
            observer.observe(ref);
            observers.push(observer);
        });

        return () => {
            window.removeEventListener('scroll', handleScroll);
            observers.forEach(observer => observer.disconnect());
        };
    }, []);

    useEffect(() => {
        const handleMouseMove = (e) => setMousePosition({ x: e.clientX, y: e.clientY });
        window.addEventListener('mousemove', handleMouseMove);
        return () => window.removeEventListener('mousemove', handleMouseMove);
    }, []);

    useEffect(() => {
        const ctaCard = document.querySelector('#cta .group');
        if (!ctaCard) return;
        const handleMouseMove = (e) => {
            const rect = ctaCard.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            ctaCard.style.setProperty('--mouse-x', `${x}px`);
            ctaCard.style.setProperty('--mouse-y', `${y}px`);
        };
        ctaCard.addEventListener('mousemove', handleMouseMove);
        return () => ctaCard.removeEventListener('mousemove', handleMouseMove);
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
    
    const stats = [
        { value: "92.14%", label: "Test Accuracy", icon: <Award className="w-8 h-8 text-cyan-300" /> },
        { value: "24", label: "Medical Signs", icon: <FileText className="w-8 h-8 text-cyan-300" /> },
        { value: "1662", label: "Feature Dimensions", icon: <Cpu className="w-8 h-8 text-cyan-300" /> },
        { value: "60fps", label: "Real-time Processing", icon: <Zap className="w-8 h-8 text-cyan-300" /> }
    ];

    const howItWorksSteps = [
        { index: 0, title: "Live Video Capture", description: "Gestures are captured in real-time using a standard webcam, requiring no special hardware.", icon: <VideoCamera className="w-8 h-8 text-cyan-300" /> },
        { index: 1, title: "Holistic Feature Extraction", description: "MediaPipe extracts 1662 key points from the hands, face, and body pose for detailed analysis.", icon: <Cpu className="w-8 h-8 text-cyan-300" /> },
        { index: 2, title: "AI Recognition & Translation", description: "Our Bi-LSTM model recognizes the sign, and an LLM translates it into a complete sentence.", icon: <BrainCircuit className="w-8 h-8 text-cyan-300" /> }
    ];

    const handleViewDocumentation = () => {
        if (onViewDocumentation) {
            onViewDocumentation();
        } else {
            window.open('https://drive.google.com/file/d/1J6BwG-xoXMhuL2hhmUldrzmIkuHFdsyI/view?usp=drivesdk', '_blank');
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 text-white font-sans overflow-x-hidden">
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute -top-20 -right-20 sm:-top-40 sm:-right-40 w-40 h-40 sm:w-80 sm:h-80 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" style={{ transform: `translate(${mousePosition.x * 0.05}px, ${mousePosition.y * 0.05}px)` }}></div>
                <div className="absolute -bottom-20 -left-20 sm:-bottom-40 sm:-left-40 w-40 h-40 sm:w-80 sm:h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse animation-delay-2000" style={{ transform: `translate(${-mousePosition.x * 0.05}px, ${-mousePosition.y * 0.05}px)` }}></div>
                <div className="absolute top-1/2 left-1/2 w-48 h-48 sm:w-96 sm:h-96 bg-indigo-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse animation-delay-4000" style={{ transform: `translate(calc(-50% + ${mousePosition.x * 0.03}px), calc(-50% + ${mousePosition.y * 0.03}px))` }}></div>
            </div>

            <header className={clsx(
                'fixed left-0 right-0 z-50 transition-transform duration-500 ease-in-out',
                // Base position and padding for mobile and unscrolled desktop
                'top-0 pt-3 md:pt-5',
                // Transform logic for DESKTOP ONLY
                scrolled ? 'md:translate-y-[calc(100vh-100%-0.75rem)]' : 'md:translate-y-0'
            )}>
                <nav className={clsx(
                    'transition-all duration-500 bg-white/10 backdrop-blur-xl shadow-2xl ring-1 ring-white/10',
                    // Static mobile styles
                    'mx-4 rounded-lg',
                    // Original desktop styles, which override mobile
                    scrolled
                        ? 'md:max-w-xl md:rounded-full md:mx-auto'
                        : 'md:max-w-7xl md:rounded-lg md:mx-auto lg:mx-auto'
                )}>
                    <div className={clsx("flex items-center justify-between",
                        // Static mobile styles
                        'h-14 px-4',
                        // Original desktop styles, which override mobile
                        scrolled
                            ? 'md:h-14 md:px-4'
                            : 'md:h-14 md:px-6'
                    )}>
                        <div className={clsx("flex items-center justify-start transition-all duration-300", 
                            // Original desktop scroll effect
                            scrolled && "md:opacity-0 md:w-0 md:overflow-hidden"
                        )}>
                            <span className="font-bold text-lg bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                                INSIGNIA
                            </span>
                        </div>

                        <div className={clsx("hidden md:flex items-center space-x-8 transition-all duration-300", 
                            // Original desktop scroll effect
                            scrolled ? "mx-auto" : "justify-center flex-1"
                        )}>
                            <a href="#features" onClick={(e) => smoothScroll(e, '#features')} className="hover:text-cyan-400 transition-colors text-sm font-medium">Features</a>
                            <a href="#how-it-works" onClick={(e) => smoothScroll(e, '#how-it-works')} className="hover:text-cyan-400 transition-colors text-sm font-medium">How it Works</a>
                            <a href="#contact" onClick={(e) => smoothScroll(e, '#contact')} className="hover:text-cyan-400 transition-colors text-sm font-medium">Contact</a>
                        </div>
                        
                        <div className="flex items-center justify-end">
                             <div className={clsx("hidden md:block transition-all duration-300", 
                                // Original desktop scroll effect
                                scrolled && "md:opacity-0 md:w-0 md:overflow-hidden"
                            )}>
                                 <button onClick={onTryNow} className="px-5 py-2 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full hover:from-cyan-600 hover:to-blue-700 transition-all transform hover:scale-105 shadow-lg relative overflow-hidden group text-sm">
                                     <span className="absolute inset-0 bg-white opacity-0 group-hover:opacity-20 transition-opacity duration-300"></span>
                                     <span className="relative">Try Now</span>
                                 </button>
                             </div>
                            <button onClick={() => setIsMenuOpen(!isMenuOpen)} className="md:hidden p-2 hover:bg-gray-800 rounded-full transition-colors" aria-label="Open menu">
                                {isMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                            </button>
                        </div>
                    </div>
                </nav>
            </header>
            
            <div className={clsx('md:hidden fixed inset-0 z-40 transition-all duration-300', isMenuOpen ? 'visible' : 'invisible')}>
                <div className={clsx("absolute inset-0 bg-gray-900/80 backdrop-blur-sm transition-opacity duration-300", isMenuOpen ? 'opacity-100' : 'opacity-0')} onClick={() => setIsMenuOpen(false)}></div>
                <div className={clsx('absolute right-0 top-0 h-full w-64 max-w-[80vw] bg-gray-900/95 backdrop-blur-md transform transition-transform duration-300', isMenuOpen ? 'translate-x-0' : 'translate-x-full')}>
                    <div className="p-6">
                        <button onClick={() => setIsMenuOpen(false)} className="mb-8 ml-auto block p-2 hover:bg-gray-800 rounded-full transition-colors" aria-label="Close menu"><X className="w-6 h-6" /></button>
                        <nav className="space-y-4">
                            <a href="#features" onClick={(e) => smoothScroll(e, '#features')} className="block px-4 py-3 hover:bg-gray-800 rounded-lg transition-all hover:translate-x-2 transform">Features</a>
                            <a href="#how-it-works" onClick={(e) => smoothScroll(e, '#how-it-works')} className="block px-4 py-3 hover:bg-gray-800 rounded-lg transition-all hover:translate-x-2 transform">How it Works</a>
                            <a href="#contact" onClick={(e) => smoothScroll(e, '#contact')} className="block px-4 py-3 hover:bg-gray-800 rounded-lg transition-all hover:translate-x-2 transform">Contact</a>
                            <button onClick={() => { onTryNow(); setIsMenuOpen(false); }} className="w-full mt-6 px-4 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-lg hover:from-cyan-600 hover:to-blue-700 transition-all">Try Now</button>
                        </nav>
                    </div>
                </div>
            </div>

            <main>
                <section id="hero" className="relative flex items-center min-h-screen pt-20 pb-16 sm:pb-20 px-4 sm:px-6 lg:px-8">
                    <div className="max-w-7xl mx-auto w-full">
                        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
                            <div className="space-y-6 sm:space-y-8 animate-fade-in text-center lg:text-left">
                                <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold leading-tight">Bridging Communication with <span className="bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent animate-gradient">ISL and ML</span></h1>
                                <p className="text-lg sm:text-xl text-gray-300 leading-relaxed max-w-2xl mx-auto lg:mx-0">An intelligent sign language recognition system enabling deaf and mute patients to communicate medical needs through real-time gesture recognition with 92.14% accuracy.</p>
                                <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                                    <button onClick={onTryNow} className="group relative px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full hover:from-cyan-600 hover:to-blue-700 transition-all transform hover:scale-105 shadow-xl flex items-center justify-center overflow-hidden">
                                        <span className="absolute inset-0 bg-white opacity-0 group-hover:opacity-20 transition-opacity duration-300"></span>
                                        <Play className="w-5 h-5 mr-2 group-hover:animate-pulse" />
                                        <span className="font-semibold">Try Live Demo</span>
                                        <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
                                    </button>
                                    <button onClick={handleViewDocumentation} className="group px-8 py-4 border border-gray-600 rounded-full hover:bg-gray-800 hover:border-cyan-500 transition-all flex items-center justify-center relative overflow-hidden">
                                        <span className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 to-blue-600/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
                                        <FileText className="w-5 h-5 mr-2" />
                                        <span className="relative">View Documentation</span>
                                        <ChevronRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
                                    </button>
                                </div>
                            </div>
                            <div className="relative animate-fade-in">
                                <div className="relative rounded-2xl overflow-hidden shadow-2xl transform hover:scale-105 transition-transform duration-500">
                                    <img src="https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=800&h=600&fit=crop" alt="Doctor using a laptop, symbolizing digital healthcare." className="w-full h-auto" />
                                    <div className="absolute inset-0 bg-gradient-to-t from-gray-900/80 via-transparent to-transparent"></div>
                                    <div className="absolute bottom-6 left-6 right-6">
                                        <div className="bg-gray-900/80 backdrop-blur-md rounded-lg p-4 border border-gray-700">
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center space-x-3">
                                                    <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                                                    <span className="text-sm">Live Translation Active</span>
                                                </div>
                                                <span className="text-xs text-gray-400">92.14% Accuracy</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div className="absolute -top-4 -right-4 bg-gradient-to-br from-purple-600 to-pink-600 rounded-lg p-3 shadow-xl animate-float"><Heart className="w-6 h-6" /></div>
                                <div className="absolute -bottom-4 -left-4 bg-gradient-to-br from-cyan-600 to-blue-600 rounded-lg p-3 shadow-xl animate-float animation-delay-2000"><Globe className="w-6 h-6" /></div>
                            </div>
                        </div>
                    </div>
                </section>

                <section id="stats" className="pb-16 sm:pb-20 px-4 sm:px-6 lg:px-8 -mt-12">
                    <div className="max-w-7xl mx-auto">
                        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6 sm:gap-8">
                            {stats.map((stat) => (
                                <div key={stat.label} className="group relative rounded-xl p-0.5 bg-gradient-to-r from-white/10 via-white/30 to-white/10 transition-all duration-300 transform hover:scale-105 hover:-translate-y-1 shadow-2xl shadow-black/40 hover:shadow-cyan-500/30">
                                    <div className="rounded-[11px] bg-gray-900/80 backdrop-blur-lg h-full w-full p-6">
                                        <div className="flex items-center space-x-5">
                                            <div className="flex-shrink-0">{stat.icon}</div>
                                            <div className="flex-grow">
                                                <div className="text-3xl sm:text-4xl font-bold text-white">
                                                    <StatCounter value={stat.value} />
                                                </div>
                                                <div className="text-sm text-gray-400 font-light">{stat.label}</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                <section id="features" className="py-16 sm:py-20 px-4 sm:px-6 lg:px-8">
                    <div className="max-w-7xl mx-auto">
                        <div className="text-center mb-12 sm:mb-16">
                            <h2 className="text-3xl sm:text-4xl font-bold mb-4">Powerful Features for Healthcare Communication</h2>
                            <p className="text-lg sm:text-xl text-gray-400 max-w-3xl mx-auto">Combining cutting-edge AI with practical medical applications.</p>
                        </div>
                        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
                            {features.map((feature, index) => (
                                <div key={index} className="group relative rounded-xl p-0.5 bg-gradient-to-r from-white/10 via-white/30 to-white/10 transition-all duration-300 transform hover:scale-105 hover:-translate-y-1 shadow-2xl shadow-black/40 hover:shadow-cyan-500/30 cursor-pointer" onMouseEnter={() => setHoveredFeature(index)} onMouseLeave={() => setHoveredFeature(null)}>
                                    <div className="rounded-[11px] bg-gray-900/80 backdrop-blur-lg h-full w-full p-6 text-left">
                                        <div className={clsx('w-12 h-12 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center mb-5 transition-all duration-300', hoveredFeature === index && 'scale-110 rotate-6')}>
                                            {React.cloneElement(feature.icon, { className: "w-6 h-6 text-white" })}
                                        </div>
                                        <h3 className="text-lg sm:text-xl font-semibold mb-2 text-white">{feature.title}</h3>
                                        <p className="text-sm sm:text-base text-gray-400 font-light">{feature.description}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                <section id="how-it-works" className="py-16 sm:py-24 px-4 sm:px-6 lg:px-8 bg-gray-800/20">
                    <div className="max-w-4xl mx-auto">
                        <div className="text-center mb-12 sm:mb-16">
                            <h2 className="text-3xl sm:text-4xl font-bold mb-4">How It Works</h2>
                            <p className="text-lg sm:text-xl text-gray-400">From gesture to sentence in three seamless steps.</p>
                        </div>
                        
                        {/* Mobile version */}
                        <div className="md:hidden space-y-6">
                            {howItWorksSteps.map((step, index) => (
                                <div key={step.title} ref={el => timelineRefs.current[index] = el} 
                                     className={clsx('transform transition-all duration-700', 
                                                     visibleSteps[index] ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10')}>
                                    <div className="bg-gray-800/50 backdrop-blur-md border border-white/10 rounded-lg p-6 shadow-lg">
                                        <div className="flex items-start gap-4">
                                            <div className="flex-shrink-0">
                                                <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center">
                                                    {React.cloneElement(step.icon, { className: "w-6 h-6 text-white" })}
                                                </div>
                                            </div>
                                            <div className="flex-grow">
                                                <div className="flex items-center gap-3 mb-2">
                                                    <h3 className="text-lg font-semibold text-white">{step.title}</h3>
                                                    <span className="text-xs bg-cyan-500/20 text-cyan-400 px-2 py-1 rounded-full">Step {index + 1}</span>
                                                </div>
                                                <p className="text-gray-400 font-light text-sm">{step.description}</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Desktop version */}
                        <div className="hidden md:block relative">
                            <div className="absolute left-1/2 -translate-x-1/2 top-0 w-0.5 h-full bg-gray-700/50" aria-hidden="true"></div>
                            <div className="space-y-16">
                                {howItWorksSteps.map((step, index) => {
                                    const isLeft = index % 2 === 0;
                                    return (
                                        <div key={step.title} ref={el => timelineRefs.current[index] = el} className="relative flex items-center">
                                            <div className={clsx('absolute left-1/2 -translate-x-1/2 w-4 h-4 rounded-full border-2 border-cyan-500 transition-all duration-500', 
                                                                 visibleSteps[index] ? 'bg-cyan-500 scale-125' : 'bg-gray-700')}></div>
                                            <div className={clsx('w-1/2', isLeft ? 'pr-10' : 'pl-10 ml-auto')}>
                                                <div className={clsx('p-6 rounded-lg bg-gray-800/50 backdrop-blur-md border border-white/10 shadow-lg transform transition-all duration-700', 
                                                                      visibleSteps[index] ? 'opacity-100 translate-x-0' : `opacity-0 ${isLeft ? '-translate-x-10' : 'translate-x-10'}`)}>
                                                    <div className={clsx('flex items-center gap-4', isLeft ? 'justify-end' : 'justify-start')}>
                                                        <h3 className={clsx('text-lg font-semibold text-white', isLeft ? 'order-1' : 'order-2')}>{step.title}</h3>
                                                        <div className={clsx('text-cyan-400', isLeft ? 'order-2' : 'order-1')}>{step.icon}</div>
                                                    </div>
                                                    <p className={clsx('mt-2 text-gray-400 font-light text-sm', isLeft ? 'text-right' : 'text-left')}>{step.description}</p>
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    </div>
                </section>

                <section id="cta" className="py-24 sm:py-32 px-4 sm:px-6 lg:px-8">
                    <div className="group relative max-w-4xl mx-auto text-center">
                        <div className="absolute -inset-1 bg-gradient-to-r from-purple-600 via-cyan-500 to-blue-600 rounded-2xl opacity-50 group-hover:opacity-75 transition duration-500"></div>
                        <div className="relative rounded-2xl bg-gray-900/80 backdrop-blur-xl border border-white/10 p-8 sm:p-12 overflow-hidden">
                            <div className="absolute inset-0 bg-grid-pattern opacity-20"></div>
                            <div className="radial-gradient absolute inset-0 opacity-0 group-hover:opacity-40 transition-opacity duration-500"></div>
                            <div className="relative z-10 space-y-6">
                                <h2 className="text-3xl sm:text-4xl font-bold text-white">Experience INSIGNIA in Action</h2>
                                <p className="text-lg sm:text-xl text-gray-300 max-w-2xl mx-auto">
                                    Bridge the communication gap in healthcare. Launch the live demo now and witness the future of accessible patient care.
                                </p>
                                <div className="pt-4">
                                    <button onClick={onTryNow} className="group/button relative px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full hover:from-cyan-600 hover:to-blue-700 transition-all transform hover:scale-105 shadow-xl flex items-center justify-center overflow-hidden mx-auto">
                                        <span className="absolute -inset-full top-0 block group-hover/button:translate-x-full transition-transform duration-500 ease-in-out bg-gradient-to-r from-transparent via-white/30 to-transparent"></span>
                                        <Play className="w-6 h-6 mr-3" />
                                        <span className="font-semibold text-lg">Launch Live Demo</span>
                                    </button>
                                </div>
                                <p className="pt-6 text-xs text-gray-500 uppercase tracking-widest">Academic Project • B.Tech CSE 2024-25</p>
                            </div>
                        </div>
                    </div>
                </section>
            </main>

            <footer id="contact" className="bg-gray-900 border-t border-gray-800 py-12 px-4 sm:px-6 lg:px-8">
                <div className="max-w-7xl mx-auto">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
                        <div className="md:col-span-2 lg:col-span-1">
                            <div className="flex items-center space-x-2 mb-4">
                                <img src="/images/logo.png" alt="INSIGNIA Logo" className="w-10 h-10 sm:w-12 sm:h-12 object-contain" />
                                <span className="text-lg sm:text-xl font-bold">INSIGNIA</span>
                            </div>
                            <p className="text-gray-400 text-sm">A Sign Language Recognition System with Attention-Based Bi-LSTM and LLM.</p>
                            <p className="text-xs text-gray-500 mt-2">Developed at Netaji Subhash Engineering College</p>
                        </div>
                        <div>
                            <h3 className="font-semibold mb-4">Research</h3>
                            <ul className="space-y-2 text-sm text-gray-400">
                                <li><a href="#" className="hover:text-white transition-colors">Technical Paper</a></li>
                                <li><a href="#" className="hover:text-white transition-colors">Dataset Info</a></li>
                                <li><a href="#" className="hover:text-white transition-colors">Model Architecture</a></li>
                                <li><a href="#" className="hover:text-white transition-colors">Results & Analysis</a></li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="font-semibold mb-4">Team</h3>
                            <ul className="space-y-2 text-sm text-gray-400">
                                <li><a href="https://www.linkedin.com/in/sourik-ray-0755561b1/" target="_blank" className="hover:text-white transition-colors">Sourik Roy</a></li>
                                <li><a href="https://www.linkedin.com/in/mayukh-ganguly-319904315/" target="_blank" className="hover:text-white transition-colors">Mayukh Ganguly</a></li>
                                <li><a href="https://www.linkedin.com/in/riddhi-mondal-659b91222/" target="_blank" className="hover:text-white transition-colors">Riddhi Mondal</a></li>
                                <li><a href="https://www.linkedin.com/in/rudranilbhattacharjee/" target="_blank" className="hover:text-white transition-colors">Rudranil Bhattacharjee</a></li>
                                <li><a href="https://www.linkedin.com/in/harshit-narayan-trivedi/" target="_blank" className="hover:text-white transition-colors">Harshit Narayan Trivedi</a></li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="font-semibold mb-4">Project Details</h3>
                            <ul className="space-y-3 text-sm text-gray-400">
                                <li className="flex items-start space-x-3"><Award className="w-5 h-5 mt-0.5 flex-shrink-0" /><span >Guided by Dr. Chandra Das</span></li>
                                <li className="flex items-start space-x-3"><MapPin className="w-5 h-5 mt-0.5 flex-shrink-0" /><span >NSEC, Garia, Kolkata - 700152</span></li>
                                <li className="flex items-start space-x-3"><Star className="w-5 h-5 mt-0.5 flex-shrink-0" /><span >Academic Year 2024-25</span></li>
                            </ul>
                        </div>
                    </div>
                    <div className="mt-8 sm:mt-12 pt-8 border-t border-gray-800 flex flex-col sm:flex-row justify-between items-center space-y-4 sm:space-y-0">
                        <p className="text-gray-400 text-sm text-center sm:text-left">© 2025 INSIGNIA Project. Netaji Subhash Engineering College.</p>
                        <div className="flex space-x-4 sm:space-x-6 text-sm">
                            <a href="https://drive.google.com/file/d/1J6BwG-xoXMhuL2hhmUldrzmIkuHFdsyI/view" target="_blank" className="text-gray-400 hover:text-white transition-colors">Project Report</a>
                            <a href="https://github.com/Im-Rik/Insignia" target="_blank" className="hover:text-white transition-colors">GitHub Repository</a>
                            <a href="https://drive.google.com/file/d/1J6BwG-xoXMhuL2hhmUldrzmIkuHFdsyI/view" target="_blank" className="hover:text-white transition-colors">Documentation</a>
                        </div>
                    </div>
                    <div className="mt-8 sm:mt-12 pt-8 border-t border-gray-800 flex flex-col sm:flex-row justify-between items-center space-y-4 sm:space-y-0"></div>
                </div>
            </footer>

            <style jsx>{`
                @keyframes float {
                    0%, 100% { transform: translateY(0) rotate(0); }
                    50% { transform: translateY(-10px) rotate(3deg); }
                }
                .animate-float { animation: float 3s ease-in-out infinite; }
                .animation-delay-2000 { animation-delay: 2s; }
                .animation-delay-4000 { animation-delay: 4s; }
                
                @keyframes gradient {
                    0% { background-position: 0% 50%; }
                    50% { background-position: 100% 50%; }
                    100% { background-position: 0% 50%; }
                }
                .animate-gradient {
                    background-size: 200% 200%;
                    animation: gradient 5s ease infinite;
                }
                
                @keyframes fade-in {
                    from { opacity: 0; transform: translateY(20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .animate-fade-in { animation: fade-in 1s ease-out; }

                .bg-grid-pattern {
                    background-image: 
                        linear-gradient(rgba(255, 255, 255, 0.07) 1px, transparent 1px),
                        linear-gradient(to right, rgba(255, 255, 255, 0.07) 1px, transparent 1px);
                    background-size: 2rem 2rem;
                }
                .radial-gradient {
                    background: radial-gradient(400px circle at var(--mouse-x) var(--mouse-y), rgba(107, 114, 128, 0.2), transparent 80%);
                }
            `}</style>
        </div>
    );
};

export default Landing;