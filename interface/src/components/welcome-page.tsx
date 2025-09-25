import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { GalleryVerticalEnd, UploadCloud, Loader2, User, Bot, Plus, Send, Paperclip } from "lucide-react";

export default function WelcomePage() {
  const [step, setStep] = useState<"home" | "uploader" | "scanning" | "chat">("home");
  const [pdfFiles, setPdfFiles] = useState<File[]>([]); // Changed from single file to array
  const [uploadProgress, setUploadProgress] = useState<{[key: string]: number}>({});
  const [chat, setChat] = useState<{ role: "user" | "ai"; text: string }[]>([]);
  const [input, setInput] = useState("");
  const [aiTyping, setAiTyping] = useState(false);
  const [isOfflineMode, setIsOfflineMode] = useState(true); // Changed to offline by default as per backend
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Step 1: Homepage with two options
  const handleStartUpload = () => setStep("uploader");
  const handleStartDirectChat = () => setStep("chat");
  const handleGoHome = () => {
    setStep("home");
    setPdfFiles([]); // Changed from setPdfFile(null)
    setUploadProgress({});
    setChat([]);
    setInput("");
  };

  // Step 2: Real PDF upload to backend - modified for multiple files
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const files = Array.from(e.target.files);
      setPdfFiles(files);
      setStep("scanning");
      
      // Upload files one by one to existing endpoint
      await uploadMultipleFiles(files);
    }
  };

  const uploadMultipleFiles = async (files: File[]) => {
    try {
      // Upload each file individually to the existing /api/upload endpoint
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();
        formData.append('pdf', file);
        
        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
          throw new Error(data.error);
        }
        
        console.log(`File ${i + 1}/${files.length} uploaded:`, data.message);
      }
      
      console.log('All files uploaded successfully');
      setStep("chat");
    } catch (error) {
      console.error('Upload failed:', error);
      alert(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setStep("uploader");
    }
  };

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const files = Array.from(e.dataTransfer.files).filter(file => file.type === 'application/pdf');
      
      if (files.length === 0) {
        alert('Please upload only PDF files');
        return;
      }
      
      setPdfFiles(files);
      setStep("scanning");
      
      await uploadMultipleFiles(files);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    const userMessage = input;
    setChat(prev => [...prev, { role: "user", text: userMessage }]);
    setInput("");
    setAiTyping(true);
    
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      setChat(prev => [...prev, { role: "ai", text: data.response }]);
    } catch (error) {
      console.error('Chat failed:', error);
      setChat(prev => [...prev, { role: "ai", text: `Sorry, there was an error: ${error instanceof Error ? error.message : 'Unknown error'}` }]);
    } finally {
      setAiTyping(false);
    }
  };

  const handleToggleMode = async () => {
    const newOfflineMode = !isOfflineMode;
    setIsOfflineMode(newOfflineMode);
    
    try {
      const response = await fetch('/api/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ offline: newOfflineMode })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.error) {
        console.warn('Toggle warning:', data.error);
      } else {
        console.log('Mode toggled:', data);
      }
    } catch (error) {
      console.error('Toggle failed:', error);
      // Revert the toggle if backend call failed
      setIsOfflineMode(!newOfflineMode);
    }
  };

  // Navigation function to go back to uploader
  const handleGoToUploader = () => setStep("uploader");

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 relative overflow-hidden">
      {/* Enhanced dark theme background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950"></div>
        <div className="absolute top-32 left-1/2 -translate-x-1/2 w-[40rem] h-[20rem] bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-32 right-16 w-80 h-80 bg-violet-500/5 rounded-full blur-2xl"></div>
        <div className="absolute top-1/2 left-16 w-64 h-64 bg-emerald-500/5 rounded-full blur-xl"></div>
      </div>
      
      {/* Enhanced dark header with model toggle */}
      <header className="relative z-10 w-full max-w-7xl mx-auto px-6 py-6 border-b border-slate-800/50 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div 
            className="flex items-center space-x-3 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={handleGoHome}
          >
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg flex items-center justify-center shadow-lg">
              <GalleryVerticalEnd className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-semibold tracking-tight text-slate-100">Promptly</span>
          </div>
          
          <div className="flex items-center gap-6">
            {/* Model Toggle Switch */}
            <div className="flex items-center gap-3">
              <span className={`text-sm font-medium transition-colors ${isOfflineMode ? 'text-slate-300' : 'text-slate-500'}`}>
                Offline
              </span>
              <button
                onClick={handleToggleMode}
                className={`relative w-11 h-6 rounded-full transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-blue-500/20 ${
                  !isOfflineMode 
                    ? 'bg-blue-600' 
                    : 'bg-slate-700'
                }`}
              >
                <div className={`absolute top-1 w-4 h-4 bg-white rounded-full shadow-sm transition-all duration-300 ${
                  !isOfflineMode ? 'translate-x-5' : 'translate-x-1'
                }`}></div>
              </button>
              <span className={`text-sm font-medium transition-colors ${!isOfflineMode ? 'text-slate-300' : 'text-slate-500'}`}>
                Online
              </span>
            </div>
            
            <div className="text-sm text-slate-400 font-medium">AI Document Assistant</div>
          </div>
        </div>
      </header>

      <main className="relative z-10 w-full max-w-5xl mx-auto px-6 py-8">
        {/* Enhanced dark hero section */}
        {step === "home" && (
          <div className="min-h-[75vh] flex flex-col items-center justify-center text-center space-y-16 animate-fadein">
            {/* Hero Content */}
            <div className="space-y-8 max-w-4xl">
              <div className="w-20 h-20 bg-gradient-to-br from-blue-600 to-blue-700 rounded-2xl flex items-center justify-center mx-auto shadow-2xl shadow-blue-500/20">
                <GalleryVerticalEnd className="w-10 h-10 text-white" />
              </div>
              
              <div className="space-y-6">
                <h1 className="text-5xl md:text-6xl font-bold text-slate-100 leading-tight tracking-tight">
                  Intelligent Document
                  <span className="block bg-gradient-to-r from-blue-400 to-violet-400 bg-clip-text text-transparent">Assistant</span>
                </h1>
                <p className="text-xl text-slate-300 max-w-3xl mx-auto leading-relaxed">
                  Enhance your productivity with AI-powered document analysis and intelligent conversations. 
                  Choose your preferred interaction method below.
                </p>
              </div>
              
              {/* Compact CTA buttons */}
              <div className="flex flex-col sm:flex-row gap-4 max-w-lg mx-auto mt-12">
                <div 
                  onClick={handleStartUpload}
                  className="group flex-1 relative overflow-hidden bg-gradient-to-br from-blue-600/15 to-blue-700/25 border border-blue-500/25 rounded-xl p-6 cursor-pointer transition-all duration-300 hover:from-blue-600/25 hover:to-blue-700/35 hover:border-blue-400/40 hover:shadow-lg hover:shadow-blue-500/15 backdrop-blur-sm"
                >
                  <div className="relative z-10 text-center space-y-3">
                    <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center mx-auto group-hover:bg-blue-500/30 transition-colors">
                      <UploadCloud className="w-5 h-5 text-blue-400" />
                    </div>
                    <h3 className="text-base font-semibold text-slate-100">Document Analysis</h3>
                    <p className="text-xs text-slate-500 leading-relaxed">Upload and analyze PDFs</p>
                  </div>
                </div>
                
                <div 
                  onClick={handleStartDirectChat}
                  className="group flex-1 relative overflow-hidden bg-gradient-to-br from-violet-600/15 to-violet-700/25 border border-violet-500/25 rounded-xl p-6 cursor-pointer transition-all duration-300 hover:from-violet-600/25 hover:to-violet-700/35 hover:border-violet-400/40 hover:shadow-lg hover:shadow-violet-500/15 backdrop-blur-sm"
                >
                  <div className="relative z-10 text-center space-y-3">
                    <div className="w-10 h-10 bg-violet-500/20 rounded-lg flex items-center justify-center mx-auto group-hover:bg-violet-500/30 transition-colors">
                      <Bot className="w-5 h-5 text-violet-400" />
                    </div>
                    <h3 className="text-base font-semibold text-slate-100">Direct Chat</h3>
                    <p className="text-xs text-slate-500 leading-relaxed">Instant AI conversation</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Redesigned feature showcase */}
            <div className="w-full max-w-5xl grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="group text-center space-y-3 p-6 rounded-xl bg-slate-900/20 border border-slate-800/30 backdrop-blur-sm hover:bg-slate-900/30 transition-all duration-300">
                <div className="w-10 h-10 bg-blue-500/15 rounded-lg flex items-center justify-center mx-auto group-hover:bg-blue-500/25 transition-colors">
                  <UploadCloud className="w-5 h-5 text-blue-400" />
                </div>
                <h3 className="text-base font-semibold text-slate-100">Secure Processing</h3>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Enterprise-grade encryption and privacy protection
                </p>
              </div>

              <div className="group text-center space-y-3 p-6 rounded-xl bg-slate-900/20 border border-slate-800/30 backdrop-blur-sm hover:bg-slate-900/30 transition-all duration-300">
                <div className="w-10 h-10 bg-violet-500/15 rounded-lg flex items-center justify-center mx-auto group-hover:bg-violet-500/25 transition-colors">
                  <Bot className="w-5 h-5 text-violet-400" />
                </div>
                <h3 className="text-base font-semibold text-slate-100">Advanced AI</h3>
                <p className="text-xs text-slate-500 leading-relaxed">
                  State-of-the-art language models for intelligent responses
                </p>
              </div>

              <div className="group text-center space-y-3 p-6 rounded-xl bg-slate-900/20 border border-slate-800/30 backdrop-blur-sm hover:bg-slate-900/30 transition-all duration-300">
                <div className="w-10 h-10 bg-emerald-500/15 rounded-lg flex items-center justify-center mx-auto group-hover:bg-emerald-500/25 transition-colors">
                  <Loader2 className="w-5 h-5 text-emerald-400" />
                </div>
                <h3 className="text-base font-semibold text-slate-100">Real-time Results</h3>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Instant responses and insights for immediate productivity
                </p>
              </div>
            </div>
          </div>
        )}
        
        {/* Enhanced dark uploader */}
        {step === "uploader" && (
          <div className="min-h-[60vh] flex flex-col items-center justify-center animate-fadein">
            <div
              className="w-full max-w-3xl mx-auto flex flex-col items-center justify-center gap-8 border-2 border-dashed border-slate-700 hover:border-blue-500/50 rounded-2xl py-20 bg-slate-900/30 cursor-pointer transition-all duration-300 hover:bg-slate-800/30 backdrop-blur-sm group"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="w-20 h-20 bg-blue-500/10 rounded-2xl flex items-center justify-center group_hover:bg-blue-500/20 transition-colors group-hover:scale-110 duration-300">
                <UploadCloud className="w-10 h-10 text-blue-400 group-hover:animate-bounce" />
              </div>
              <div className="text-center space-y-4">
                <h2 className="text-3xl font-semibold text-slate-100">Upload your document</h2>
                <p className="text-slate-300 text-lg">Drag and drop your PDF file here, or click to browse</p>
                <div className="flex items-center justify-center gap-6 text-sm text-slate-500 mt-6">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
                    <span>PDF Support</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                    <span>Max 50MB</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-violet-400 rounded-full"></div>
                    <span>Secure</span>
                  </div>
                </div>
              </div>
              <Input
                ref={fileInputRef}
                id="pdf"
                type="file"
                accept="application/pdf"
                multiple // Added multiple attribute
                onChange={handleFileChange}
                className="hidden"
              />
            </div>
          </div>
        )}

        {/* Enhanced dark scanning animation - updated for multiple files */}
        {step === "scanning" && (
          <div className="min-h-[60vh] flex flex-col items-center justify-center space-y-8 animate-fadein">
            <div className="relative">
              <div className="w-24 h-24 bg-blue-500/10 rounded-2xl flex items-center justify-center">
                <Loader2 className="w-12 h-12 text-blue-400 animate-spin" />
              </div>
              <div className="absolute inset-0 bg-blue-500/5 rounded-2xl animate-ping"></div>
            </div>
            <div className="text-center space-y-4">
              <h2 className="text-3xl font-semibold text-slate-100">Processing documents...</h2>
              <p className="text-slate-300 text-lg">
                We&apos;re analyzing your {pdfFiles.length} PDF{pdfFiles.length > 1 ? 's' : ''} to enable intelligent conversations
              </p>
              
              {/* Show file list during upload */}
              {pdfFiles.length > 1 && (
                <div className="max-w-md mx-auto space-y-2 mt-6">
                  {pdfFiles.map((file, index) => (
                    <div key={index} className="flex items-center justify-between bg-slate-800/30 rounded-lg px-4 py-2 text-sm">
                      <span className="text-slate-300 truncate flex-1">{file.name}</span>
                      <span className="text-blue-400 ml-2">{(file.size / 1024 / 1024).toFixed(1)}MB</span>
                    </div>
                  ))}
                </div>
              )}
              
              <div className="flex items-center justify-center gap-3 mt-6">
                <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce"></div>
                <div className="w-3 h-3 bg-violet-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-3 h-3 bg-emerald-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
            </div>
          </div>
        )}

        {/* Futuristic chat interface */}
        {step === "chat" && (
          <div className="w-full h-[75vh] flex flex-col animate-fadein border border-cyan-500/20 rounded-2xl bg-slate-950/80 overflow-hidden backdrop-blur-xl shadow-2xl shadow-cyan-500/10 relative">
            {/* Futuristic glow effect */}
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 via-transparent to-blue-500/5 pointer-events-none"></div>
            
            {/* Futuristic chat header - Remove plus button */}
            <div className="relative p-5 border-b border-cyan-500/20 bg-slate-900/60 backdrop-blur-lg">
              <div className="flex items-center gap-4">
                <div className="relative w-10 h-10 bg-gradient-to-br from-cyan-500/30 to-blue-600/30 rounded-xl flex items-center justify-center shadow-lg border border-cyan-400/30">
                  <div className="absolute inset-0 bg-cyan-400/10 rounded-xl animate-pulse"></div>
                  {pdfFiles.length > 0 ? <UploadCloud className="w-5 h-5 text-cyan-300 relative z-10" /> : <Bot className="w-5 h-5 text-cyan-300 relative z-10" />}
                </div>
                <div className="flex-1">
                  <h2 className="text-lg font-semibold text-cyan-100">
                    {pdfFiles.length > 0 ? "Document Analysis" : "AI Assistant"}
                  </h2>
                  <div className="flex items-center gap-3 mt-1">
                    <p className="text-slate-400 text-sm">
                      {pdfFiles.length > 0 
                        ? `${pdfFiles.length} document${pdfFiles.length > 1 ? 's' : ''} processed - Neural analysis active`
                        : "Advanced AI conversation mode"
                      }
                    </p>
                    <div className="flex items-center gap-1.5 text-xs">
                      <div className={`w-2 h-2 rounded-full animate-pulse ${!isOfflineMode ? 'bg-cyan-400' : 'bg-slate-400'}`}></div>
                      <span className="text-slate-500 font-medium">
                        {!isOfflineMode ? 'ONLINE' : 'OFFLINE'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-6 relative">
              {chat.length === 0 && (
                <div className="text-center py-16">
                  <div className="relative w-16 h-16 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-2xl flex items-center justify-center mx-auto mb-6 border border-cyan-400/30">
                    <div className="absolute inset-0 bg-cyan-400/10 rounded-2xl animate-pulse"></div>
                    <Bot className="w-8 h-8 text-cyan-400 relative z-10" />
                  </div>
                  <h3 className="text-2xl font-semibold text-cyan-100 mb-3">
                    {pdfFiles.length > 0 ? "Neural Analysis Ready" : "AI System Online"}
                  </h3>
                  <p className="text-slate-300 mb-8">
                    {pdfFiles.length > 0 
                      ? `${pdfFiles.length} document${pdfFiles.length > 1 ? 's' : ''} processed and ready for queries.`
                      : "Start a conversation with the AI assistant."
                    }
                  </p>
                </div>
              )}
              
              {chat.map((msg, i) => (
                msg.role === "user" ? (
                  <div key={i} className="flex justify-end fade-in">
                    <div className="flex items-end gap-3 max-w-[70%]">
                      <div className="relative bg-gradient-to-br from-cyan-600/80 to-blue-700/80 text-white px-3 py-2 rounded-2xl rounded-br-md shadow-lg border border-cyan-400/30 inline-block">
                        <div className="absolute inset-0 bg-cyan-400/10 rounded-2xl rounded-br-md"></div>
                        <p className="leading-relaxed relative z-10 text-center">{msg.text}</p>
                      </div>
                      <div className="relative w-8 h-8 bg-gradient-to-br from-cyan-600/70 to-blue-700/70 rounded-xl flex items-center justify-center flex-shrink-0 shadow-sm border border-cyan-400/30">
                        <div className="absolute inset-0 bg-cyan-400/10 rounded-xl animate-pulse"></div>
                        <User className="w-4 h-4 text-white relative z-10" />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div key={i} className="flex justify-start fade-in">
                    <div className="flex items-end gap-3 max-w-[70%]">
                      <div className="relative w-8 h-8 bg-slate-900/70 border border-slate-600/40 rounded-xl flex items-center justify-center flex-shrink-0 shadow-sm">
                        <div className="absolute inset-0 bg-slate-700/20 rounded-xl animate-pulse"></div>
                        <Bot className="w-4 h-4 text-slate-300 relative z-10" />
                      </div>
                      <div className="relative bg-slate-900/60 border border-slate-600/40 px-3 py-2 rounded-2xl rounded-bl-md shadow-lg backdrop-blur-sm inline-block">
                        <div className="absolute inset-0 bg-slate-700/10 rounded-2xl rounded-bl-md"></div>
                        <p className="leading-relaxed text-slate-100 relative z-10">{msg.text}</p>
                      </div>
                    </div>
                  </div>
                )
              ))}
              
              {aiTyping && (
                <div className="flex justify-start fade-in">
                  <div className="flex items-end gap-3 max-w-[70%]">
                    <div className="relative w-8 h-8 bg-slate-900/70 border border-slate-600/40 rounded-xl flex items-center justify-center flex-shrink-0 shadow-sm">
                      <div className="absolute inset-0 bg-slate-700/20 rounded-xl animate-pulse"></div>
                      <Bot className="w-4 h-4 text-slate-300 relative z-10" />
                    </div>
                    <div className="relative bg-slate-900/60 border border-slate-600/40 px-4 py-3 rounded-2xl rounded-bl-md shadow-lg backdrop-blur-sm">
                      <div className="absolute inset-0 bg-slate-700/10 rounded-2xl rounded-bl-md"></div>
                      <div className="flex items-center space-x-2 relative z-10">
                        <div className="flex space-x-1">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce"></div>
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                        </div>
                        <span className="text-cyan-300 text-sm">processing...</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            <div className="relative p-5 border-t border-cyan-500/20 bg-slate-900/60 backdrop-blur-lg">
              <form onSubmit={handleSend} className="flex items-center gap-3">
                <div className="relative flex-1">
                  <Input
                    type="text"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    placeholder={pdfFiles.length > 0 ? "Query documents..." : "Enter command..."}
                    className="w-full h-12 bg-slate-900/70 border-cyan-500/30 text-cyan-100 placeholder:text-slate-400 focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-400/20 rounded-xl px-4 py-3 pr-4 text-base backdrop-blur-sm transition-all duration-200 shadow-inner"
                  />
                  <div className="absolute inset-0 bg-gradient-to-r from-cyan-400/5 via-transparent to-blue-400/5 rounded-xl pointer-events-none"></div>
                </div>
                
                {/* Improved Upload Button */}
                <Button 
                  type="button"
                  onClick={handleGoToUploader}
                  className="relative w-12 h-12 bg-slate-800/60 hover:bg-slate-700/70 border border-slate-600/40 hover:border-slate-500/50 rounded-xl p-0 transition-all duration-200 group flex-shrink-0 shadow-lg"
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-slate-600/10 to-slate-700/20 rounded-xl"></div>
                  <div className="absolute inset-0 bg-slate-400/5 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-200"></div>
                  <Paperclip className="w-5 h-5 text-slate-300 group-hover:text-slate-200 relative z-10 group-hover:scale-110 group-hover:rotate-12 transition-all duration-200" />
                </Button>
                
                {/* Improved Send Button */}
                <Button 
                  type="submit" 
                  disabled={!input.trim()}
                  className="relative h-12 bg-gradient-to-r from-cyan-600 to-blue-700 hover:from-cyan-500 hover:to-blue-600 disabled:from-slate-700 disabled:to-slate-800 text-white px-8 py-3 rounded-xl font-semibold shadow-lg shadow-cyan-500/25 disabled:shadow-none disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 border border-cyan-400/40 disabled:border-slate-600/30 flex-shrink-0"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-cyan-400/15 to-blue-400/15 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-200"></div>
                  <span className="relative z-10 flex items-center gap-2">
                    <Send className="w-4 h-4" />
                    SEND
                  </span>
                </Button>
              </form>
            </div>
          </div>
        )}
      </main>
      
      {/* Minimal footer */}
      <footer className="relative z-10 w-full max-w-7xl mx-auto px-6 py-6 mt-auto border-t border-slate-800/30">
        <div className="text-center">
          <p className="text-xs text-slate-500">© 2025 Promptly • AI Document Assistant</p>
        </div>
      </footer>

      {/* Professional animations */}
      <style jsx>{`
        .animate-fadein {
          animation: fadeInUp 0.6s ease-out;
        }
        .fade-in {
          animation: fadeInUp 0.4s ease-out;
        }
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
