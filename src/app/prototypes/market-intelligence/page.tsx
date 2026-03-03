'use client';

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  AreaChart, Area, PieChart as RechartsPieChart, Pie, Cell, Legend
} from 'recharts';
import { 
  Search, TrendingUp, TrendingDown, Star, Bell, Moon, Sun,
  RefreshCw, X, AlertTriangle, MessageCircle, Send, Minimize2, Maximize2, 
  Briefcase, BarChart3, Calculator, Target, Wallet, Eye, EyeOff, Home, PieChart
} from 'lucide-react';
import { 
  Stock, 
  MarketType, 
  fetchRealTimeData, 
  getStocksByMarket, 
  getCurrencySymbol,
  searchStocks,
  getStockQuote
} from '@/utils/stockApi';
import { BASE_URL } from '@/constants';
import Link from 'next/link';

// Additional types
interface WatchlistItem {
  symbol: string;
  addedAt: Date;
}

interface PortfolioItem {
  symbol: string;
  shares: number;
  avgCost: number;
}

interface Alert {
  id: string;
  symbol: string;
  type: 'price_above' | 'price_below';
  value: number;
  active: boolean;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: string[];
}

interface NewsItem {
  title: string;
  source: string;
  date: string;
  sentiment: 'positive' | 'negative' | 'neutral';
}

// Generate historical data
const generateHistoricalData = (basePrice: number, days: number) => {
  const data = [];
  let price = basePrice * 0.85;
  const now = new Date();
  
  for (let i = days; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    const volatility = 0.02;
    const change = (Math.random() - 0.48) * volatility * price;
    price = Math.max(price + change, basePrice * 0.5);
    
    data.push({
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      close: price,
      volume: Math.floor(Math.random() * 50000000) + 10000000,
    });
  }
  return data;
};

// AI Stock Analysis
const generateAIAnalysis = (stock: Stock): {
  summary: string;
  metrics: { label: string; value: string }[];
  technicals: { indicator: string; value: string; signal: string }[];
  recentNews: NewsItem[];
} => {
  const news: NewsItem[] = [
    { title: `${stock.symbol} Reports Strong Q4 Earnings, Beats Estimates`, source: 'Reuters', date: '2 days ago', sentiment: 'positive' },
    { title: `Analysts Raise Price Target for ${stock.symbol}`, source: 'Bloomberg', date: '3 days ago', sentiment: 'positive' },
    { title: `${stock.symbol} Announces New Product Launch`, source: 'CNBC', date: '1 week ago', sentiment: 'neutral' },
  ];
  
  const currency = stock.marketCap > 1e12 ? '$' : '₹';
  const marketCapValue = stock.marketCap > 1e12 ? (stock.marketCap / 1e12).toFixed(2) : (stock.marketCap / 1e9).toFixed(2);
  const marketCapUnit = stock.marketCap > 1e12 ? 'T' : 'B';
  
  return {
    summary: `${stock.name} (${stock.symbol}) is currently trading at ${currency}${stock.price.toFixed(2)}, with a ${stock.changePercent >= 0 ? '+' : ''}${stock.changePercent.toFixed(2)}% change today. With a P/E ratio of ${stock.pe}x and market cap of ${currency}${marketCapValue}${marketCapUnit}, ${stock.symbol} remains a significant player in the ${stock.sector} sector.`,
    metrics: [
      { label: 'Market Cap', value: `${currency}${marketCapValue}${marketCapUnit}` },
      { label: 'P/E Ratio', value: stock.pe.toFixed(2) },
      { label: 'EPS', value: `${currency}${stock.eps.toFixed(2)}` },
      { label: 'Dividend', value: `${((stock.dividend / stock.price) * 100).toFixed(2)}%` },
      { label: '52W High', value: `${currency}${stock.high52w.toFixed(2)}` },
      { label: '52W Low', value: `${currency}${stock.low52w.toFixed(2)}` },
    ],
    technicals: [
      { indicator: 'RSI (14)', value: (45 + Math.random() * 30).toFixed(1), signal: 'Neutral' },
      { indicator: 'MACD', value: stock.changePercent > 0 ? 'Bullish' : 'Bearish', signal: stock.changePercent > 0 ? 'Buy' : 'Sell' },
      { indicator: 'SMA 50', value: `${currency}${(stock.price * (0.95 + Math.random() * 0.1)).toFixed(2)}`, signal: 'Above' },
      { indicator: 'SMA 200', value: `${currency}${(stock.price * (0.90 + Math.random() * 0.15)).toFixed(2)}`, signal: 'Above' },
    ],
    recentNews: news,
  };
};

// Calculate projections
const calculateProjections = (initialInvestment: number, stock: Stock) => {
  const projections = [
    { period: '6 Months', months: 6, annualReturn: 0.12 },
    { period: '1 Year', months: 12, annualReturn: 0.15 },
    { period: '3 Years', months: 36, annualReturn: 0.18 },
    { period: '5 Years', months: 60, annualReturn: 0.20 },
  ];
  
  return projections.map(p => {
    const monthlyReturn = p.annualReturn / 12;
    const projectedValue = initialInvestment * Math.pow(1 + monthlyReturn, p.months);
    const totalReturn = ((projectedValue - initialInvestment) / initialInvestment) * 100;
    const cagr = (Math.pow(projectedValue / initialInvestment, 12 / p.months) - 1) * 100;
    
    return {
      period: p.period,
      projectedValue: Math.round(projectedValue * 100) / 100,
      totalReturn: Math.round(totalReturn * 100) / 100,
      cagr: Math.round(cagr * 100) / 100,
    };
  });
};

// AI Chat Response
const generateAIResponse = (query: string, stocks: Stock[], market: MarketType) => {
  const lowerQuery = query.toLowerCase();
  const currency = getCurrencySymbol(market);
  let response = '';
  const sources: string[] = [];
  
  if (lowerQuery.includes('best') || lowerQuery.includes('recommend')) {
    const topStock = stocks.filter(s => s.changePercent > 0).sort((a, b) => b.changePercent - a.changePercent)[0];
    if (topStock) {
      response = `Based on current market data, ${topStock.name} (${topStock.symbol}) shows strong performance with a ${topStock.changePercent.toFixed(2)}% gain today. It has a P/E ratio of ${topStock.pe}x and offers a dividend yield of ${((topStock.dividend / topStock.price) * 100).toFixed(2)}%.`;
      sources.push(topStock.symbol);
    }
  } else if (lowerQuery.includes('dividend')) {
    const dividendStocks = stocks.filter(s => s.dividend > 0).sort((a, b) => (b.dividend / b.price) - (a.dividend / a.price));
    if (dividendStocks.length > 0) {
      const top = dividendStocks[0];
      response = `${top.name} (${top.symbol}) offers the highest dividend yield at ${((top.dividend / top.price) * 100).toFixed(2)}%.`;
      sources.push(top.symbol);
    }
  } else if (lowerQuery.includes('technology') || lowerQuery.includes('tech')) {
    const techStocks = stocks.filter(s => s.sector === 'Technology');
    response = `Technology sector stocks: ${techStocks.map(s => `${s.symbol}: ${currency}${s.price.toFixed(2)}`).join(', ')}`;
    sources.push(...techStocks.map(s => s.symbol));
  } else {
    response = `I can help you analyze stocks. Try asking about:\n- "What are the best stocks to buy?"\n- "Which stocks pay dividends?"\n- "Technology sector stocks"`;
  }
  
  return { response, sources };
};

export default function MarketIntelligenceDashboard() {
  const [mounted, setMounted] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [market, setMarket] = useState<MarketType>('US');
  const [dataSource, setDataSource] = useState<'real' | 'sample'>('sample');
  const [dataSourceName, setDataSourceName] = useState<string>('Sample');
  const [isLoadingData, setIsLoadingData] = useState(false);
  const [dataMode, setDataMode] = useState<'sample' | 'real'>('sample');
  const [realStocks, setRealStocks] = useState<Stock[]>([]);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'charts' | 'calculator' | 'search' | 'portfolio'>('dashboard');
  const [selectedStock, setSelectedStock] = useState<Stock | null>(null);
  const [stockSearch, setStockSearch] = useState('');
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([
    { symbol: 'AAPL', addedAt: new Date() },
    { symbol: 'GOOGL', addedAt: new Date() },
    { symbol: 'MSFT', addedAt: new Date() },
  ]);
  const [portfolio, setPortfolio] = useState<PortfolioItem[]>([
    { symbol: 'AAPL', shares: 50, avgCost: 165.00 },
    { symbol: 'NVDA', shares: 20, avgCost: 750.00 },
  ]);
  const [alerts, setAlerts] = useState<Alert[]>([
    { id: '1', symbol: 'AAPL', type: 'price_above', value: 185, active: true },
    { id: '2', symbol: 'TSLA', type: 'price_below', value: 240, active: true },
  ]);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMinimized, setChatMinimized] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(
    []
  );
  const [chatInput, setChatInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [timeframe, setTimeframe] = useState<'1D' | '1W' | '1M' | '3M' | '1Y' | '5Y'>('1M');
  const [investmentAmount, setInvestmentAmount] = useState(1000);
  const [lastRefresh] = useState(new Date());
  const [searchResults, setSearchResults] = useState<Stock[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchedStock, setSearchedStock] = useState<Stock | null>(null);
  
  const chatEndRef = useRef<HTMLDivElement>(null);
  const currency = getCurrencySymbol(market);
  
  useEffect(() => {
    setMounted(true);
  }, []);
  
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);
  
  // Clear search when switching tabs or when stock is selected
  useEffect(() => {
    setStockSearch('');
    setSearchResults([]);
    setSearchedStock(null);
  }, [activeTab, selectedStock?.symbol]);
  
  useEffect(() => {
    const loadData = async () => {
      setIsLoadingData(true);
      console.log('Loading data for market:', market, 'mode:', dataMode);
      
      if (dataMode === 'sample') {
        setDataSource('sample');
        setDataSourceName('Sample');
        setRealStocks([]);
        const marketStocks = getStocksByMarket(market);
        console.log('Using sample stocks for market:', market, 'count:', marketStocks.length);
        if (marketStocks.length > 0) {
          setSelectedStock(marketStocks[0]);
        }
        setIsLoadingData(false);
        return;
      }
      
      // When in real mode, fetch from API with fallback
      console.log('Fetching real-time data for:', market);
      const { stocks, isReal, source } = await fetchRealTimeData(market);
      console.log('Fetch result:', { isReal, source, stocksCount: stocks?.length });
      
      if (isReal && stocks && stocks.length > 0) {
        console.log('Setting real stocks with prices:', stocks.map(s => ({ symbol: s.symbol, price: s.price })));
        setRealStocks(stocks);
        setDataSource('real');
        setDataSourceName(source || 'Live');
        setSelectedStock(stocks[0]);
      } else {
        console.log('Falling back to sample data');
        setRealStocks([]);
        setDataSource('sample');
        setDataSourceName('Sample');
        const marketStocks = getStocksByMarket(market);
        if (marketStocks.length > 0) {
          setSelectedStock(marketStocks[0]);
        }
      }
      setIsLoadingData(false);
    };
    
    loadData();
  }, [market, dataMode]);
  
  const filteredStocks = useMemo(() => {
    if (isLoadingData) {
      return getStocksByMarket(market);
    }
    const stocks = (dataMode === 'real' && realStocks.length > 0) ? realStocks : getStocksByMarket(market);
    if (!stockSearch) return stocks;
    return stocks.filter(s => 
      s.symbol.toLowerCase().includes(stockSearch.toLowerCase()) ||
      s.name.toLowerCase().includes(stockSearch.toLowerCase())
    );
  }, [stockSearch, market, dataMode, realStocks, isLoadingData]);

  const currentStocks = useMemo(() => {
    return filteredStocks.length > 0 ? filteredStocks : getStocksByMarket(market);
  }, [filteredStocks, market]);
  
  const safeSelectedStock = useMemo(() => {
    if (selectedStock) return selectedStock;
    if (currentStocks.length > 0) return currentStocks[0];
    return getStocksByMarket(market)[0];
  }, [selectedStock, currentStocks, market]);
  
  // Initialize chat with welcome message when stock changes
  useEffect(() => {
    if (safeSelectedStock) {
      setChatMessages([
        { id: '1', role: 'assistant', content: `Hello! I'm your AI stock assistant for ${safeSelectedStock.symbol}. Ask me anything about this stock!`, timestamp: new Date() },
      ]);
    }
  }, [safeSelectedStock?.symbol]);
  
  const stockData = useMemo(() => {
    const stock = safeSelectedStock;
    const days = timeframe === '1D' ? 1 : timeframe === '1W' ? 7 : timeframe === '1M' ? 30 : timeframe === '3M' ? 90 : timeframe === '1Y' ? 365 : 1825;
    return generateHistoricalData(stock.price, days);
  }, [safeSelectedStock, timeframe]);
  
  const projections = useMemo(() => {
    const stock = safeSelectedStock;
    return calculateProjections(investmentAmount, stock);
  }, [safeSelectedStock, investmentAmount]);
  
  const aiAnalysis = useMemo(() => {
    if (!safeSelectedStock) return null;
    return generateAIAnalysis(safeSelectedStock);
  }, [safeSelectedStock]);
  
  const portfolioStats = useMemo(() => {
    let totalValue = 0;
    let totalCost = 0;
    portfolio.forEach(p => {
      const stock = currentStocks.find(s => s.symbol === p.symbol);
      if (stock) {
        totalValue += stock.price * p.shares;
        totalCost += p.avgCost * p.shares;
      }
    });
    const totalReturn = totalCost > 0 ? ((totalValue - totalCost) / totalCost) * 100 : 0;
    return { totalValue, totalCost, totalReturn };
  }, [portfolio, currentStocks]);
  
  const handleStockSelect = (stock: Stock) => {
    setSelectedStock(stock);
    setStockSearch('');
    setSearchResults([]);
    setSearchedStock(null);
    setActiveTab('dashboard');
  };
  
  const handleStockSearch = async (query: string) => {
    setStockSearch(query);
    
    if (query.length < 2) {
      setSearchResults([]);
      setSearchedStock(null);
      return;
    }
    
    // First check if it's in our current stocks (by symbol or name)
    const existingStock = currentStocks.find(
      s => s.symbol.toLowerCase() === query.toLowerCase() || 
           s.name.toLowerCase().includes(query.toLowerCase())
    );
    
    if (existingStock) {
      setSearchResults([]);
      setSearchedStock(null);
      // Auto-select if found in current stocks
      setSelectedStock(existingStock);
      return;
    }
    
    // Search for stocks not in the list (by symbol or name)
    setIsSearching(true);
    const results = await searchStocks(query, market);
    setSearchResults(results);
    setIsSearching(false);
    // Clear previously searched stock when doing new search
    setSearchedStock(null);
  };
  
  const handleSelectSearchResult = async (stock: Stock) => {
    // Fetch real-time quote for this stock
    const quote = await getStockQuote(stock.symbol);
    let updatedStock: Stock;
    if (quote) {
      updatedStock = {
        ...stock,
        price: quote.price,
        change: quote.change,
        changePercent: quote.changePercent,
      };
    } else {
      updatedStock = stock;
    }
    setSelectedStock(updatedStock);
    // Clear search results and close dropdown - clear ALL search state
    setSearchResults([]);
    setSearchedStock(null);
    setStockSearch('');
  };
  
  const removeFromWatchlist = (symbol: string) => {
    setWatchlist(watchlist.filter(w => w.symbol !== symbol));
  };
  
  const toggleAlert = (id: string) => {
    setAlerts(alerts.map(a => a.id === id ? { ...a, active: !a.active } : a));
  };
  
  const deleteAlert = (id: string) => {
    setAlerts(alerts.filter(a => a.id !== id));
  };
  
  const sendChatMessage = async () => {
    if (!chatInput.trim() || !safeSelectedStock) return;
    
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: chatInput,
      timestamp: new Date(),
    };
    setChatMessages(prev => [...prev, userMessage]);
    const currentInput = chatInput;
    const currentHistory = chatMessages.filter(m => m.role !== 'assistant' || m.id !== '1');
    setChatInput('');
    setLoading(true);
    
    try {
      console.log('Sending chat request for stock:', safeSelectedStock.symbol);
      const response = await fetch('/api/stock-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          stock: safeSelectedStock,
          question: currentInput,
          history: currentHistory.map(m => ({ role: m.role, content: m.content }))
        })
      });
      
      console.log('Chat response status:', response.status);
      const data = await response.json();
      console.log('Chat response data:', data);
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer,
        timestamp: new Date(),
        sources: [safeSelectedStock.symbol],
      };
      setChatMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      // Fallback to local response
      const { response, sources } = generateAIResponse(currentInput, currentStocks, market);
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `I apologize, but I couldn't connect to the AI service. Here's a basic response:\n\n${response}\n\nNote: For better answers, please ensure the Gemini API key is properly configured.`,
        timestamp: new Date(),
        sources,
      };
      setChatMessages(prev => [...prev, assistantMessage]);
    } finally {
      setLoading(false);
    }
  };
  
  const formatCurrency = (value: number): string => {
    return `${currency}${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  if (!mounted) {
    return (
      <div className={`min-h-screen flex items-center justify-center ${darkMode ? 'bg-gray-900' : 'bg-gray-50'}`}>
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
          <p className={darkMode ? 'text-gray-400' : 'text-gray-500'}>Loading Market Intelligence...</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className={`min-h-screen ${darkMode ? 'bg-gray-900' : 'bg-gray-50'} transition-colors duration-300`}>
      {/* Header */}
      <header className={`${darkMode ? 'bg-gray-800' : 'bg-white'} border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'} px-6 py-4 sticky top-0 z-40`}>
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-4">
            <Link href={BASE_URL} className="flex items-center gap-2">
              <div className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                <Home className={`w-5 h-5 ${darkMode ? 'text-blue-400' : 'text-blue-600'}`} />
              </div>
            </Link>
            <nav className="hidden md:flex items-center gap-1 ml-4">
              {[
                { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
                { id: 'charts', label: 'Charts', icon: PieChart },
                { id: 'calculator', label: 'Calculator', icon: Calculator },
                { id: 'search', label: 'Search', icon: Search },
                { id: 'portfolio', label: 'Portfolio', icon: Briefcase },
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as typeof activeTab)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    activeTab === tab.id
                      ? 'bg-blue-600 text-white'
                      : `${darkMode ? 'text-gray-300 hover:bg-gray-700' : 'text-gray-600 hover:bg-gray-100'}`
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
          
          <div className="flex items-center gap-3">
            <button className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-100 text-gray-600'}`}>
              <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button onClick={() => setDarkMode(!darkMode)} className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-100 text-gray-600'}`}>
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
            <div className={`relative p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-100 text-gray-600'}`}>
              <Bell className="w-5 h-5" />
              {alerts.filter(a => a.active).length > 0 && (
                <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full text-xs text-white flex items-center justify-center">
                  {alerts.filter(a => a.active).length}
                </span>
              )}
            </div>
            <div className={`w-10 h-10 rounded-full bg-blue-600 flex items-center justify-center text-white font-bold`}>
              U
            </div>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="max-w-7xl mx-auto p-6">
        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {/* Search */}
            <div className="relative">
              <Search className={`absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
              <input
                type="text"
                value={stockSearch}
                onChange={(e) => handleStockSearch(e.target.value)}
                placeholder="Search any stock (e.g., ABNB, Airbnb)..."
                className={`w-full pl-12 pr-4 py-3 rounded-xl ${darkMode ? 'bg-gray-800 border-gray-700 text-white' : 'bg-white border-gray-200 text-gray-900'} border focus:ring-2 focus:ring-blue-500`}
              />
              {isSearching && (
                <div className={`absolute right-4 top-1/2 -translate-y-1/2`}>
                  <RefreshCw className="w-4 h-4 animate-spin text-blue-500" />
                </div>
              )}
              {(searchResults.length > 0 || searchedStock) && (
                <div className={`absolute top-full left-0 right-0 mt-2 rounded-xl overflow-hidden ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border shadow-xl z-50 max-h-80 overflow-y-auto`}>
                  {searchResults.length > 0 && (
                    <>
                      <div className={`px-4 py-2 text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'} border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                        Search Results (click to add)
                      </div>
                      {searchResults.map(stock => (
                        <button
                          key={stock.symbol}
                          onClick={() => handleSelectSearchResult(stock)}
                          className={`w-full px-4 py-3 flex items-center justify-between ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-50'}`}
                        >
                          <div className="flex items-center gap-3">
                            <span className="font-bold text-blue-500">{stock.symbol}</span>
                            <span className={darkMode ? 'text-gray-300' : 'text-gray-600'}>{stock.name}</span>
                          </div>
                          <span className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                            {stock.sector}
                          </span>
                        </button>
                      ))}
                    </>
                  )}
                  {searchedStock && (
                    <div className={`px-4 py-3 ${darkMode ? 'bg-blue-900/30' : 'bg-blue-50'}`}>
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="font-bold text-blue-500">{searchedStock.symbol}</span>
                          <span className={`ml-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>{searchedStock.name}</span>
                        </div>
                        <span className={searchedStock.changePercent >= 0 ? 'text-green-500' : 'text-red-500'}>
                          {searchedStock.price > 0 ? `${formatCurrency(searchedStock.price)} (${searchedStock.changePercent >= 0 ? '+' : ''}${searchedStock.changePercent.toFixed(2)}%)` : 'Loading...'}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}
              {stockSearch && !isSearching && searchResults.length === 0 && !searchedStock && (
                <div className={`absolute top-full left-0 right-0 mt-2 rounded-xl overflow-hidden ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border shadow-xl z-50 p-4`}>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    No stocks found for "{stockSearch}". Try a different search term.
                  </p>
                </div>
              )}
            </div>
            
            {/* Stats */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className={`p-5 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <div className="flex items-center justify-between mb-3">
                  <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>Portfolio Value</span>
                  <Wallet className="w-5 h-5 text-blue-500" />
                </div>
                <p className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  {formatCurrency(portfolioStats.totalValue)}
                </p>
                <p className={`text-sm ${portfolioStats.totalReturn >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {portfolioStats.totalReturn >= 0 ? '+' : ''}{portfolioStats.totalReturn.toFixed(2)}%
                </p>
              </div>
              <div className={`p-5 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <div className="flex items-center justify-between mb-3">
                  <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>Total Invested</span>
                  <Target className="w-5 h-5 text-purple-500" />
                </div>
                <p className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  {formatCurrency(portfolioStats.totalCost)}
                </p>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{portfolio.length} positions</p>
              </div>
              <div className={`p-5 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <div className="flex items-center justify-between mb-3">
                  <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>Active Alerts</span>
                  <Bell className="w-5 h-5 text-orange-500" />
                </div>
                <p className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{alerts.filter(a => a.active).length}</p>
              </div>
              <div className={`p-5 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <div className="flex items-center justify-between mb-3">
                  <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>Watchlist</span>
                  <Eye className="w-5 h-5 text-green-500" />
                </div>
                <p className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{watchlist.length}</p>
              </div>
            </div>
            
            {/* Main Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Chart */}
              <div className={`lg:col-span-2 p-6 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <div className="flex items-center justify-between mb-4">
                  <h2 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    {safeSelectedStock?.symbol || 'AAPL'} - {safeSelectedStock?.name || 'Apple Inc.'}
                  </h2>
                  <div className="flex items-center gap-2">
                    {['1D', '1W', '1M', '3M', '1Y', '5Y'].map(tf => (
                      <button
                        key={tf}
                        onClick={() => setTimeframe(tf as typeof timeframe)}
                        className={`px-3 py-1 rounded-lg text-xs font-medium ${
                          timeframe === tf
                            ? 'bg-blue-600 text-white'
                            : darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-600'
                        }`}
                      >
                        {tf}
                      </button>
                    ))}
                  </div>
                </div>
                
                <div className="flex items-center gap-4 mb-4">
                  <span className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    {formatCurrency(safeSelectedStock?.price || 0)}
                  </span>
                  <span className={`flex items-center gap-1 ${(safeSelectedStock?.changePercent || 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {(safeSelectedStock?.changePercent || 0) >= 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
                    {((safeSelectedStock?.changePercent || 0) >= 0 ? '+' : '')}{(safeSelectedStock?.changePercent || 0).toFixed(2)}%
                  </span>
                </div>
                
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={stockData}>
                      <defs>
                        <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#374151' : '#e5e7eb'} />
                      <XAxis dataKey="date" stroke={darkMode ? '#9ca3af' : '#6b7280'} fontSize={12} />
                      <YAxis stroke={darkMode ? '#9ca3af' : '#6b7280'} fontSize={12} domain={['auto', 'auto']} />
                      <Tooltip contentStyle={{ backgroundColor: darkMode ? '#1f2937' : '#fff', border: `1px solid ${darkMode ? '#374151' : '#e5e7eb'}`, borderRadius: '8px' }} />
                      <Area type="monotone" dataKey="close" stroke="#3b82f6" strokeWidth={2} fillOpacity={1} fill="url(#colorPrice)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              {/* Watchlist */}
              <div className={`p-6 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <h2 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-900'} mb-4`}>Watchlist</h2>
                <div className="space-y-3">
                  {watchlist.map(w => {
                    const stock = currentStocks.find(s => s.symbol === w.symbol);
                    if (!stock) return null;
                    return (
                      <div
                        key={w.symbol}
                        onClick={() => handleStockSelect(stock)}
                        className={`p-3 rounded-xl cursor-pointer ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-50'} ${selectedStock?.symbol === w.symbol ? 'bg-blue-600/20' : ''}`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{stock.symbol}</span>
                          <button onClick={(e) => { e.stopPropagation(); removeFromWatchlist(w.symbol); }} className={`p-1 rounded ${darkMode ? 'hover:bg-gray-600 text-gray-400' : 'hover:bg-gray-200 text-gray-500'}`}>
                            <X className="w-3 h-3" />
                          </button>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{stock.name}</span>
                          <span className={stock.changePercent >= 0 ? 'text-green-500' : 'text-red-500'}>
                            {stock.changePercent >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
                
                <div className={`mt-6 pt-6 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                  <h3 className={`font-semibold mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Market Overview</h3>
                  {currentStocks.slice(0, 5).map(stock => (
                    <div key={stock.symbol} className="flex items-center justify-between text-sm py-1">
                      <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>{stock.symbol}</span>
                      <span className={stock.changePercent >= 0 ? 'text-green-500' : 'text-red-500'}>
                        {stock.changePercent >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            {/* Alerts */}
            <div className={`p-6 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <h2 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-900'} mb-4`}>Price Alerts</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {alerts.map(alert => (
                  <div key={alert.id} className={`p-4 rounded-xl ${darkMode ? 'bg-gray-700' : 'bg-gray-50'} border ${darkMode ? 'border-gray-600' : 'border-gray-200'}`}>
                    <div className="flex items-center justify-between mb-2">
                      <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{alert.symbol}</span>
                      <div className="flex items-center gap-2">
                        <button onClick={() => toggleAlert(alert.id)} className={`p-1 rounded ${alert.active ? 'text-green-500' : 'text-gray-400'}`}>
                          {alert.active ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                        </button>
                        <button onClick={() => deleteAlert(alert.id)} className={`p-1 rounded ${darkMode ? 'hover:bg-gray-600 text-gray-400' : 'hover:bg-gray-200 text-gray-500'}`}>
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                    <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                      {alert.type === 'price_above' ? 'Price above' : 'Price below'} {formatCurrency(alert.value)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
        
        {/* Charts Tab */}
        {activeTab === 'charts' && (
          <div className="space-y-6">
            {/* Stock Chart Section */}
            <div className={`p-6 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Interactive Charts</h2>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'} mt-1`}>
                    {safeSelectedStock?.symbol} - {safeSelectedStock?.name}
                  </p>
                </div>
                <select
                  value={safeSelectedStock?.symbol || 'AAPL'}
                  onChange={(e) => {
                    const stock = currentStocks.find(s => s.symbol === e.target.value);
                    if (stock) setSelectedStock(stock);
                  }}
                  className={`px-4 py-2 rounded-lg ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200 text-gray-900'} border outline-none`}
                >
                  {(dataMode === 'real' && realStocks.length > 0 ? realStocks : currentStocks).map(s => (
                    <option key={s.symbol} value={s.symbol}>{s.name} ({s.symbol})</option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-2 mb-6">
                {(['1D', '1W', '1M', '3M', '1Y', '5Y'] as const).map(tf => (
                  <button
                    key={tf}
                    onClick={() => setTimeframe(tf)}
                    className={`px-4 py-2 rounded-lg font-medium ${timeframe === tf ? 'bg-blue-600 text-white' : darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-600'}`}
                  >
                    {tf}
                  </button>
                ))}
              </div>
              
              <div className="flex items-center gap-4 mb-4">
                <span className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  {formatCurrency(safeSelectedStock?.price || 0)}
                </span>
                <span className={`flex items-center gap-1 ${(safeSelectedStock?.changePercent || 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {(safeSelectedStock?.changePercent || 0) >= 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
                  {((safeSelectedStock?.changePercent || 0) >= 0 ? '+' : '')}{(safeSelectedStock?.changePercent || 0).toFixed(2)}%
                </span>
              </div>
              
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={stockData}>
                    <defs>
                      <linearGradient id="colorPrice2" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#374151' : '#e5e7eb'} />
                    <XAxis dataKey="date" stroke={darkMode ? '#9ca3af' : '#6b7280'} fontSize={12} />
                    <YAxis stroke={darkMode ? '#9ca3af' : '#6b7280'} fontSize={12} domain={['auto', 'auto']} />
                    <Tooltip contentStyle={{ backgroundColor: darkMode ? '#1f2937' : '#fff', border: `1px solid ${darkMode ? '#374151' : '#e5e7eb'}`, borderRadius: '8px' }} />
                    <Area type="monotone" dataKey="close" stroke="#3b82f6" strokeWidth={2} fillOpacity={1} fill="url(#colorPrice2)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            {/* Shareholding Distribution */}
            <div className={`p-6 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <h2 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'} mb-6`}>Shareholding Distribution</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Pie Chart */}
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsPieChart>
                      <Pie
                        data={[
                          { name: 'Promoters', value: 52, color: '#3b82f6' },
                          { name: 'FII', value: 18, color: '#8b5cf6' },
                          { name: 'DII', value: 12, color: '#10b981' },
                          { name: 'Public', value: 18, color: '#f59e0b' },
                        ]}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={80}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {[
                          { name: 'Promoters', value: 52, color: '#3b82f6' },
                          { name: 'FII', value: 18, color: '#8b5cf6' },
                          { name: 'DII', value: 12, color: '#10b981' },
                          { name: 'Public', value: 18, color: '#f59e0b' },
                        ].map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: darkMode ? '#1f2937' : '#fff', 
                          border: `1px solid ${darkMode ? '#374151' : '#e5e7eb'}`, 
                          borderRadius: '8px' 
                        }}
                      />
                    </RechartsPieChart>
                  </ResponsiveContainer>
                </div>
                
                {/* Legend */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 rounded-lg bg-blue-500/10">
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 rounded-full bg-blue-500"></div>
                      <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>Promoters</span>
                    </div>
                    <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>52%</span>
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg bg-purple-500/10">
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 rounded-full bg-purple-500"></div>
                      <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>FII (Foreign Institutional)</span>
                    </div>
                    <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>18%</span>
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg bg-green-500/10">
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 rounded-full bg-green-500"></div>
                      <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>DII (Domestic Institutional)</span>
                    </div>
                    <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>12%</span>
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg bg-yellow-500/10">
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 rounded-full bg-yellow-500"></div>
                      <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>Public / Retail</span>
                    </div>
                    <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>18%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Calculator Tab */}
        {activeTab === 'calculator' && (
          <div className={`p-6 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
            <h2 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'} mb-6`}>Investment Projection Calculator</h2>
            
            {/* Calculator Search */}
            <div className="relative mb-6">
              <Search className={`absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
              <input
                type="text"
                value={stockSearch}
                onChange={(e) => handleStockSearch(e.target.value)}
                placeholder="Search stocks to calculate projections..."
                className={`w-full pl-12 pr-4 py-3 rounded-xl ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200 text-gray-900'} border outline-none focus:ring-2 focus:ring-blue-500`}
              />
              {isSearching && (
                <div className={`absolute right-4 top-1/2 -translate-y-1/2`}>
                  <RefreshCw className="w-4 h-4 animate-spin text-blue-500" />
                </div>
              )}
              {searchResults.length > 0 && (
                <div className={`absolute top-full left-0 right-0 mt-2 rounded-xl overflow-hidden ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'} border shadow-xl z-10 max-h-48 overflow-y-auto`}>
                  {searchResults.slice(0, 5).map(stock => (
                    <button
                      key={stock.symbol}
                      onClick={async () => {
                        const quote = await getStockQuote(stock.symbol);
                        let updatedStock = stock;
                        if (quote) {
                          updatedStock = { ...stock, price: quote.price, change: quote.change, changePercent: quote.changePercent };
                        }
                        setSelectedStock(updatedStock);
                        setStockSearch('');
                        setSearchResults([]);
                      }}
                      className={`w-full px-4 py-2 flex items-center justify-between ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-50'}`}
                    >
                      <div className="flex items-center gap-3">
                        <span className="font-bold text-blue-500">{stock.symbol}</span>
                        <span className={darkMode ? 'text-gray-300' : 'text-gray-600'}>{stock.name}</span>
                      </div>
                      {stock.price > 0 && (
                        <span className={stock.changePercent >= 0 ? 'text-green-500' : 'text-red-500'}>
                          {formatCurrency(stock.price)}
                        </span>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
            
            {/* Current Stock Info */}
            {safeSelectedStock && (
              <div className={`p-4 rounded-xl mb-6 ${darkMode ? 'bg-gray-700' : 'bg-blue-50'} border ${darkMode ? 'border-gray-600' : 'border-blue-200'}`}>
                <div className="flex items-center justify-between">
                  <div>
                    <span className={`text-lg font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{safeSelectedStock.symbol}</span>
                    <span className={`ml-2 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{safeSelectedStock.name}</span>
                  </div>
                  <div className="text-right">
                    <span className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{formatCurrency(safeSelectedStock.price)}</span>
                    <span className={`ml-2 ${safeSelectedStock.changePercent >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      {safeSelectedStock.changePercent >= 0 ? '+' : ''}{safeSelectedStock.changePercent.toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              <div>
                <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>Initial Investment</label>
                <div className="relative">
                  <span className={`absolute left-4 top-1/2 -translate-y-1/2 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{currency}</span>
                  <input
                    type="number"
                    value={investmentAmount}
                    onChange={(e) => setInvestmentAmount(Number(e.target.value))}
                    className={`w-full pl-8 pr-4 py-3 rounded-xl ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200 text-gray-900'} border outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
              </div>
              <div>
                <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>Select Stock</label>
                <select
                  value={safeSelectedStock?.symbol || 'AAPL'}
                  onChange={(e) => {
                    const stock = currentStocks.find(s => s.symbol === e.target.value);
                    if (stock) setSelectedStock(stock);
                  }}
                  className={`w-full px-4 py-3 rounded-xl ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200 text-gray-900'} border outline-none focus:ring-2 focus:ring-blue-500`}
                >
                  {(dataMode === 'real' && realStocks.length > 0 ? realStocks : currentStocks).map(s => (
                    <option key={s.symbol} value={s.symbol}>{s.name} ({s.symbol}) - {formatCurrency(s.price)}</option>
                  ))}
                </select>
              </div>
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className={darkMode ? 'border-b border-gray-700' : 'border-b border-gray-200'}>
                    <th className={`text-left py-3 px-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'} font-medium`}>Period</th>
                    <th className={`text-right py-3 px-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'} font-medium`}>Projected Value</th>
                    <th className={`text-right py-3 px-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'} font-medium`}>Total return</th>
                    <th className={`text-right py-3 px-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'} font-medium`}>CAGR</th>
                  </tr>
                </thead>
                <tbody>
                  {projections.map((proj) => (
                    <tr key={proj.period} className={darkMode ? 'border-b border-gray-700' : 'border-b border-gray-100'}>
                      <td className={`py-4 px-4 font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>{proj.period}</td>
                      <td className={`py-4 px-4 text-right ${darkMode ? 'text-white' : 'text-gray-900'}`}>{formatCurrency(proj.projectedValue)}</td>
                      <td className={`py-4 px-4 text-right ${proj.totalReturn >= 0 ? 'text-green-500' : 'text-red-500'}`}>{proj.totalReturn >= 0 ? '+' : ''}{proj.totalReturn.toFixed(2)}%</td>
                      <td className={`py-4 px-4 text-right ${proj.cagr >= 0 ? 'text-green-500' : 'text-red-500'}`}>{proj.cagr >= 0 ? '+' : ''}{proj.cagr.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className={`mt-6 p-4 rounded-xl ${darkMode ? 'bg-gray-700/50' : 'bg-yellow-50'} border ${darkMode ? 'border-gray-600' : 'border-yellow-200'}`}>
              <div className="flex items-start gap-3">
                <AlertTriangle className={`w-5 h-5 mt-0.5 ${darkMode ? 'text-yellow-400' : 'text-yellow-600'}`} />
                <p className={`text-sm ${darkMode ? 'text-gray-300' : 'text-yellow-800'}`}>
                  These projections are based on historical performance and current market trends. They are for informational purposes only and should not be considered as investment advice.
                </p>
              </div>
            </div>
          </div>
        )}
        
        {/* Search Tab */}
        {activeTab === 'search' && (
          <div className="space-y-6">
            <div className="relative">
              <Search className={`absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
              <input
                type="text"
                value={stockSearch}
                onChange={(e) => handleStockSearch(e.target.value)}
                placeholder="Search any stock (e.g., ABNB, Airbnb, Tesla)..."
                className={`w-full pl-12 pr-4 py-4 rounded-xl text-lg ${darkMode ? 'bg-gray-800 border-gray-700 text-white' : 'bg-white border-gray-200 text-gray-900'} border focus:ring-2 focus:ring-blue-500`}
              />
              {isSearching && (
                <div className={`absolute right-4 top-1/2 -translate-y-1/2`}>
                  <RefreshCw className="w-5 h-5 animate-spin text-blue-500" />
                </div>
              )}
            </div>
            
            {/* Search Results or Stock Grid */}
            {(searchResults.length > 0 || searchedStock) ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {searchResults.map(stock => (
                  <div
                    key={stock.symbol}
                    onClick={() => {
                      handleSelectSearchResult(stock);
                      setActiveTab('dashboard');
                    }}
                    className={`p-5 rounded-2xl cursor-pointer transition-all ${darkMode ? 'bg-gray-800 border-gray-700 hover:border-blue-500' : 'bg-white border-gray-200 hover:border-blue-500'} border`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <h3 className={`font-bold text-lg ${darkMode ? 'text-white' : 'text-gray-900'}`}>{stock.symbol}</h3>
                        <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{stock.name}</p>
                      </div>
                      <button className={`p-2 rounded-lg ${darkMode ? 'text-gray-500 hover:text-yellow-500' : 'text-gray-400 hover:text-yellow-500'}`}>
                        <Star className="w-5 h-5" />
                      </button>
                    </div>
                    {stock.price > 0 ? (
                      <>
                        <div className="flex items-center justify-between">
                          <span className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{formatCurrency(stock.price)}</span>
                          <span className={`flex items-center gap-1 ${stock.changePercent >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                            {stock.changePercent >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                            {stock.changePercent >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                          </span>
                        </div>
                        <div className={`mt-3 pt-3 border-t ${darkMode ? 'border-gray-700' : 'border-gray-100'} flex items-center justify-between text-sm`}>
                          <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>{stock.sector}</span>
                        </div>
                      </>
                    ) : (
                      <div className="flex items-center gap-2">
                        <RefreshCw className="w-4 h-4 animate-spin text-blue-500" />
                        <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Fetching price...</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  {dataMode === 'real' && realStocks.length > 0 
                    ? `Showing ${filteredStocks.length} stocks with real-time prices`
                    : `Showing ${filteredStocks.length} stocks. Click "Real" below for live data.`}
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {(dataMode === 'real' && realStocks.length > 0 ? realStocks : filteredStocks).slice(0, 12).map(stock => (
                    <div
                      key={stock.symbol}
                      onClick={() => handleStockSelect(stock)}
                      className={`p-5 rounded-2xl cursor-pointer transition-all ${darkMode ? 'bg-gray-800 border-gray-700 hover:border-blue-500' : 'bg-white border-gray-200 hover:border-blue-500'} border`}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h3 className={`font-bold text-lg ${darkMode ? 'text-white' : 'text-gray-900'}`}>{stock.symbol}</h3>
                          <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{stock.name}</p>
                        </div>
                        <button className={`p-2 rounded-lg ${darkMode ? 'text-gray-500 hover:text-yellow-500' : 'text-gray-400 hover:text-yellow-500'}`}>
                          <Star className="w-5 h-5" />
                        </button>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{formatCurrency(stock.price)}</span>
                        <span className={`flex items-center gap-1 ${stock.changePercent >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                          {stock.changePercent >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                          {stock.changePercent >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                        </span>
                      </div>
                      <div className={`mt-3 pt-3 border-t ${darkMode ? 'border-gray-700' : 'border-gray-100'} flex items-center justify-between text-sm`}>
                        <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>{stock.sector}</span>
                        <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>P/E: {stock.pe.toFixed(1)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
        
        {/* Portfolio Tab */}
        {activeTab === 'portfolio' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className={`p-6 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'} mb-2`}>Total Value</p>
                <p className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{formatCurrency(portfolioStats.totalValue)}</p>
              </div>
              <div className={`p-6 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'} mb-2`}>Total Cost</p>
                <p className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{formatCurrency(portfolioStats.totalCost)}</p>
              </div>
              <div className={`p-6 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'} mb-2`}>Total Return</p>
                <p className={`text-3xl font-bold ${portfolioStats.totalReturn >= 0 ? 'text-green-500' : 'text-red-500'}`}>{portfolioStats.totalReturn >= 0 ? '+' : ''}{portfolioStats.totalReturn.toFixed(2)}%</p>
              </div>
            </div>
            
            <div className={`p-6 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <h2 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'} mb-6`}>Holdings</h2>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className={darkMode ? 'border-b border-gray-700' : 'border-b border-gray-200'}>
                      <th className={`text-left py-3 px-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'} font-medium`}>Symbol</th>
                      <th className={`text-right py-3 px-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'} font-medium`}>Shares</th>
                      <th className={`text-right py-3 px-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'} font-medium`}>Avg Cost</th>
                      <th className={`text-right py-3 px-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'} font-medium`}>Price</th>
                      <th className={`text-right py-3 px-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'} font-medium`}>Value</th>
                      <th className={`text-right py-3 px-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'} font-medium`}>Return</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolio.map(p => {
                      const stock = currentStocks.find(s => s.symbol === p.symbol);
                      if (!stock) return null;
                      const value = stock.price * p.shares;
                      const cost = p.avgCost * p.shares;
                      const returnPct = ((value - cost) / cost) * 100;
                      return (
                        <tr key={p.symbol} className={darkMode ? 'border-b border-gray-700' : 'border-b border-gray-100'}>
                          <td className={`py-4 px-4 font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>{stock.symbol}</td>
                          <td className={`py-4 px-4 text-right ${darkMode ? 'text-white' : 'text-gray-900'}`}>{p.shares}</td>
                          <td className={`py-4 px-4 text-right ${darkMode ? 'text-white' : 'text-gray-900'}`}>{formatCurrency(p.avgCost)}</td>
                          <td className={`py-4 px-4 text-right ${darkMode ? 'text-white' : 'text-gray-900'}`}>{formatCurrency(stock.price)}</td>
                          <td className={`py-4 px-4 text-right font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>{formatCurrency(value)}</td>
                          <td className={`py-4 px-4 text-right ${returnPct >= 0 ? 'text-green-500' : 'text-red-500'}`}>{returnPct >= 0 ? '+' : ''}{returnPct.toFixed(2)}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
            
            {selectedStock && aiAnalysis && (
              <div className={`p-6 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <div className="flex items-center gap-2 mb-4">
                  <MessageCircle className="w-5 h-5 text-blue-500" />
                  <h2 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>AI Stock Analysis</h2>
                </div>
                <p className={`mb-6 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>{aiAnalysis.summary}</p>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
                  {aiAnalysis.metrics.map((metric, idx) => (
                    <div key={idx} className={`p-3 rounded-xl ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                      <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'} mb-1`}>{metric.label}</p>
                      <p className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{metric.value}</p>
                    </div>
                  ))}
                </div>
                <div className="mb-6">
                  <h3 className={`font-semibold mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Technical Indicators</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {aiAnalysis.technicals.map((tech, idx) => (
                      <div key={idx} className={`p-3 rounded-xl ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                        <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{tech.indicator}</p>
                        <p className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{tech.value}</p>
                        <span className={`text-xs ${tech.signal === 'Buy' ? 'text-green-500' : tech.signal === 'Sell' ? 'text-red-500' : 'text-yellow-500'}`}>{tech.signal}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h3 className={`font-semibold mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Recent News</h3>
                  <div className="space-y-3">
                    {aiAnalysis.recentNews.map((news, idx) => (
                      <div key={idx} className={`p-3 rounded-xl ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                        <p className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>{news.title}</p>
                        <p className={`text-xs mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{news.source} - {news.date}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
      
      {/* Floating Chat */}
      <button onClick={() => setChatOpen(true)} className="fixed bottom-6 right-6 w-14 h-14 bg-blue-600 text-white rounded-full shadow-lg flex items-center justify-center hover:bg-blue-700 z-50">
        <MessageCircle className="w-6 h-6" />
      </button>
      
      {chatOpen && (
        <div className={`fixed bottom-24 right-6 w-96 h-[500px] rounded-2xl shadow-2xl border z-50 flex flex-col ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} ${chatMinimized ? 'h-14' : ''}`}>
          <div className={`flex items-center justify-between p-4 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                <MessageCircle className="w-4 h-4 text-white" />
              </div>
              <div>
                <p className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>AI Stock Assistant</p>
                <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Chatting about {safeSelectedStock?.symbol || 'stocks'}</p>
              </div>
            </div>
            <button 
              onClick={() => setChatOpen(false)} 
              className={darkMode ? 'text-gray-400 hover:text-white' : 'text-gray-500 hover:text-gray-900'}
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          
          {!chatMinimized && (
            <>
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {chatMessages.map(msg => (
                  <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[80%] p-3 rounded-xl ${msg.role === 'user' ? 'bg-blue-600 text-white' : darkMode ? 'bg-gray-700 text-gray-100' : 'bg-gray-100 text-gray-900'}`}>
                      <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                      {msg.sources && msg.sources.length > 0 && (
                        <p className={`text-xs mt-2 pt-2 border-t ${msg.role === 'user' ? 'text-blue-200 border-blue-500' : darkMode ? 'text-gray-400 border-gray-600' : 'text-gray-500 border-gray-200'}`}>
                          Sources: {msg.sources.join(', ')}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex justify-start">
                    <div className={`p-3 rounded-xl ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
              
              <div className={`p-4 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
                    placeholder={`Ask about ${safeSelectedStock?.symbol || 'stocks'}...`}
                    className={`flex-1 px-4 py-2 rounded-xl ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-100 border-gray-200 text-gray-900'} border outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                  <button onClick={sendChatMessage} disabled={loading || !chatInput.trim()} className="p-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50">
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      )}
      
      {/* Bottom Controls */}
      <div className={`fixed bottom-6 left-6 flex flex-col gap-3 ${darkMode ? 'text-gray-500' : 'text-gray-400'} text-xs`}>
        <div className="flex items-center gap-4">
          <span>Last updated: {lastRefresh.toLocaleTimeString()}</span>
        </div>
        <div className="flex items-center gap-3">
          {/* Market Toggle */}
          <div className={`flex items-center rounded-lg overflow-hidden border ${darkMode ? 'border-gray-600' : 'border-gray-300'}`}>
            <button
              onClick={() => setMarket('US')}
              className={`px-3 py-1.5 text-sm font-medium transition-colors ${market === 'US' ? 'bg-blue-600 text-white' : darkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-white text-gray-700 hover:bg-gray-100'}`}
            >
              US
            </button>
            <button
              onClick={() => setMarket('India')}
              className={`px-3 py-1.5 text-sm font-medium transition-colors ${market === 'India' ? 'bg-blue-600 text-white' : darkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-white text-gray-700 hover:bg-gray-100'}`}
            >
              India
            </button>
          </div>
          {/* Data Source Toggle */}
          <div className={`flex items-center rounded-lg overflow-hidden border ${darkMode ? 'border-gray-600' : 'border-gray-300'}`}>
            <button
              onClick={() => setDataMode('sample')}
              className={`px-3 py-1.5 text-sm font-medium transition-colors ${dataMode === 'sample' ? 'bg-yellow-600 text-white' : darkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-white text-gray-700 hover:bg-gray-100'}`}
            >
              Sample
            </button>
            <button
              onClick={() => setDataMode('real')}
              className={`px-3 py-1.5 text-sm font-medium transition-colors ${dataMode === 'real' ? 'bg-green-600 text-white' : darkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-white text-gray-700 hover:bg-gray-100'}`}
            >
              Real
            </button>
          </div>
          {/* Data Status */}
          {isLoadingData ? (
            <span className="flex items-center gap-1 px-2 py-1 rounded-full bg-blue-500/20 text-blue-500">
              <RefreshCw className="w-3 h-3 animate-spin" />
              Loading...
            </span>
          ) : (
            <span className={`flex items-center gap-1 px-2 py-1 rounded-full ${dataSource === 'real' ? 'bg-green-500/20 text-green-500' : 'bg-yellow-500/20 text-yellow-500'}`}>
              <span className={`w-2 h-2 rounded-full ${dataSource === 'real' ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'}`}></span>
              {dataSourceName}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
