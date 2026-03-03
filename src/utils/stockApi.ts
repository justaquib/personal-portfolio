'use client';

// Stock type definition
export interface Stock {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  high52w: number;
  low52w: number;
  pe: number;
  eps: number;
  dividend: number;
  sector: string;
}

export type MarketType = 'US' | 'India';

// Sample stock data
const US_STOCKS: Stock[] = [
  { symbol: 'AAPL', name: 'Apple Inc.', price: 178.72, change: 2.34, changePercent: 1.33, volume: 52340000, marketCap: 2780000000000, high52w: 198.23, low52w: 124.17, pe: 28.5, eps: 6.27, dividend: 0.96, sector: 'Technology' },
  { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 141.80, change: -0.85, changePercent: -0.60, volume: 23450000, marketCap: 1780000000000, high52w: 153.78, low52w: 102.21, pe: 24.2, eps: 5.86, dividend: 0, sector: 'Technology' },
  { symbol: 'MSFT', name: 'Microsoft Corp.', price: 378.91, change: 4.12, changePercent: 1.10, volume: 21340000, marketCap: 2810000000000, high52w: 420.82, low52w: 245.61, pe: 35.8, eps: 10.58, dividend: 3.00, sector: 'Technology' },
  { symbol: 'AMZN', name: 'Amazon.com Inc.', price: 178.25, change: 1.56, changePercent: 0.88, volume: 45670000, marketCap: 1850000000000, high52w: 189.77, low52w: 101.26, pe: 62.4, eps: 2.85, dividend: 0, sector: 'Consumer Cyclical' },
  { symbol: 'NVDA', name: 'NVIDIA Corp.', price: 875.38, change: 15.67, changePercent: 1.82, volume: 38920000, marketCap: 2160000000000, high52w: 974.00, low52w: 222.97, pe: 65.3, eps: 13.40, dividend: 0.16, sector: 'Technology' },
  { symbol: 'TSLA', name: 'Tesla Inc.', price: 248.50, change: -3.21, changePercent: -1.28, volume: 98760000, marketCap: 790000000000, high52w: 299.29, low52w: 138.80, pe: 72.1, eps: 3.45, dividend: 0, sector: 'Consumer Cyclical' },
  { symbol: 'META', name: 'Meta Platforms', price: 505.95, change: 8.23, changePercent: 1.65, volume: 15670000, marketCap: 1290000000000, high52w: 542.81, low52w: 274.38, pe: 32.7, eps: 15.47, dividend: 2.00, sector: 'Technology' },
  { symbol: 'JPM', name: 'JPMorgan Chase', price: 195.42, change: 1.12, changePercent: 0.58, volume: 8920000, marketCap: 565000000000, high52w: 205.88, low52w: 135.19, pe: 11.8, eps: 16.56, dividend: 4.60, sector: 'Financial' },
  { symbol: 'V', name: 'Visa Inc.', price: 279.80, change: 2.45, changePercent: 0.88, volume: 6340000, marketCap: 571000000000, high52w: 290.96, low52w: 227.14, pe: 30.2, eps: 9.26, dividend: 2.08, sector: 'Financial' },
  { symbol: 'WMT', name: 'Walmart Inc.', price: 165.23, change: 0.87, changePercent: 0.53, volume: 7230000, marketCap: 445000000000, high52w: 169.94, low52w: 137.14, pe: 28.9, eps: 5.72, dividend: 2.28, sector: 'Consumer Defensive' },
];

const INDIAN_STOCKS: Stock[] = [
  { symbol: 'RELIANCE', name: 'Reliance Industries Ltd.', price: 2945.50, change: 25.30, changePercent: 0.87, volume: 5234000, marketCap: 1980000000000, high52w: 3200.00, low52w: 2100.00, pe: 28.5, eps: 103.35, dividend: 9.00, sector: 'Conglomerate' },
  { symbol: 'TCS', name: 'Tata Consultancy Services', price: 4125.80, change: -15.20, changePercent: -0.37, volume: 2345000, marketCap: 1520000000000, high52w: 4500.00, low52w: 3200.00, pe: 32.1, eps: 128.53, dividend: 120.00, sector: 'Technology' },
  { symbol: 'HDFCBANK', name: 'HDFC Bank Ltd.', price: 1680.25, change: 12.45, changePercent: 0.75, volume: 8920000, marketCap: 1280000000000, high52w: 1850.00, low52w: 1300.00, pe: 24.8, eps: 67.73, dividend: 19.50, sector: 'Financial' },
  { symbol: 'INFY', name: 'Infosys Ltd.', price: 1890.40, change: 22.60, changePercent: 1.21, volume: 4567000, marketCap: 785000000000, high52w: 2050.00, low52w: 1400.00, pe: 29.5, eps: 64.08, dividend: 34.00, sector: 'Technology' },
  { symbol: 'ICICIBANK', name: 'ICICI Bank Ltd.', price: 1185.60, change: -8.40, changePercent: -0.70, volume: 7890000, marketCap: 695000000000, high52w: 1300.00, low52w: 850.00, pe: 22.3, eps: 53.14, dividend: 9.00, sector: 'Financial' },
  { symbol: 'BHARTIARTL', name: 'Bharti Airtel Ltd.', price: 1580.75, change: 35.25, changePercent: 2.28, volume: 3892000, marketCap: 935000000000, high52w: 1700.00, low52w: 1050.00, pe: 35.8, eps: 44.16, dividend: 8.00, sector: 'Telecom' },
  { symbol: 'WIPRO', name: 'Wipro Ltd.', price: 585.20, change: 5.80, changePercent: 1.00, volume: 2134000, marketCap: 305000000000, high52w: 650.00, low52w: 420.00, pe: 26.2, eps: 22.33, dividend: 5.00, sector: 'Technology' },
  { symbol: 'HINDUNILVR', name: 'Hindustan Unilever Ltd.', price: 2850.30, change: -12.50, changePercent: -0.44, volume: 1234000, marketCap: 665000000000, high52w: 3100.00, low52w: 2200.00, pe: 55.2, eps: 51.64, dividend: 78.00, sector: ' FMCG' },
  { symbol: 'LT', name: 'Larsen & Toubro Ltd.', price: 3520.45, change: 45.80, changePercent: 1.32, volume: 2134000, marketCap: 498000000000, high52w: 3800.00, low52w: 2400.00, pe: 33.5, eps: 105.09, dividend: 36.00, sector: 'Infrastructure' },
  { symbol: 'SBIN', name: 'State Bank of India', price: 825.30, change: 18.70, changePercent: 2.32, volume: 9876000, marketCap: 735000000000, high52w: 895.00, low52w: 580.00, pe: 16.8, eps: 49.12, dividend: 13.70, sector: 'Financial' },
];

export const getStocksByMarket = (market: MarketType): Stock[] => market === 'US' ? US_STOCKS : INDIAN_STOCKS;
export const getCurrencySymbol = (market: MarketType): string => market === 'US' ? '$' : '₹';

// Twelve Data API fetch
const fetchFromTwelveData = async (market: MarketType): Promise<Stock[] | null> => {
  const apiKey = process.env.NEXT_PUBLIC_TWELVEDATA_API_KEY || '';
  
  if (!apiKey) {
    console.log('Twelve Data: No API key');
    return null;
  }
  
  const symbols = market === 'US' 
    ? ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'WMT']
    : ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS', 'WIPRO.NS', 'HINDUNILVR.NS', 'LT.NS', 'SBIN.NS'];
  
  const baseStocks = getStocksByMarket(market);
  
  try {
    const fetchPromises = symbols.map(async (symbol) => {
      try {
        const response = await fetch(
          `https://api.twelvedata.com/quote?symbol=${symbol}&apikey=${apiKey}`
        );
        
        if (!response.ok) return null;
        
        const data = await response.json();
        
        if (data.code === 400 || data.code === 401 || data.message) {
          return null;
        }
        
        if (data.price && data.price !== '0.000000') {
          return {
            symbol: data.symbol,
            price: parseFloat(data.price),
            change: parseFloat(data.change) || 0,
            changePercent: parseFloat(data.percent_change) || 0,
            volume: parseInt(data.volume) || 0,
          };
        }
        return null;
      } catch {
        return null;
      }
    });
    
    const results = await Promise.all(fetchPromises);
    const successfulQuotes = results.filter(r => r !== null);
    
    if (successfulQuotes.length === 0) {
      console.log('Twelve Data: No successful responses');
      return null;
    }
    
    const quoteMap = new Map<string, any>();
    successfulQuotes.forEach(q => {
      const key = q!.symbol.replace('.NS', '').replace('.BSE', '');
      quoteMap.set(key, q);
    });
    
    const stocks: Stock[] = baseStocks.map((s) => {
      const quote = quoteMap.get(s.symbol);
      if (quote) {
        return { ...s, price: quote.price, change: quote.change, changePercent: quote.changePercent, volume: quote.volume };
      }
      return s;
    });
    
    console.log(`Twelve Data: Fetched ${successfulQuotes.length} stocks`);
    return stocks;
  } catch (error) {
    console.log('Twelve Data error:', error);
    return null;
  }
};

// Alpha Vantage API fetch
const fetchFromAlphaVantage = async (market: MarketType): Promise<Stock[] | null> => {
  const apiKey = process.env.NEXT_PUBLIC_ALPHAVANTAGE_API_KEY || '';
  
  if (!apiKey) {
    console.log('Alpha Vantage: No API key');
    return null;
  }
  
  // Only fetch first 5 due to rate limits (5/min)
  const symbols = market === 'US' 
    ? ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA']
    : ['RELIANCE.BSE', 'TCS.BSE', 'HDFCBANK.BSE', 'INFY.BSE', 'ICICIBANK.BSE'];
  
  const baseStocks = getStocksByMarket(market);
  
  try {
    const quoteMap = new Map<string, any>();
    
    for (let i = 0; i < symbols.length; i++) {
      try {
        if (i > 0) await new Promise(resolve => setTimeout(resolve, 13000)); // Rate limit delay
        
        const response = await fetch(
          `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbols[i]}&apikey=${apiKey}`
        );
        
        if (!response.ok) continue;
        
        const data = await response.json();
        
        if (data['Error Message'] || data['Note'] || data['Information']) {
          continue;
        }
        
        const quote = data['Global Quote'];
        if (quote && quote['05. price']) {
          quoteMap.set(symbols[i].replace('.BSE', ''), {
            symbol: quote['01. symbol'],
            price: parseFloat(quote['05. price']),
            change: parseFloat(quote['09. change']) || 0,
            changePercent: parseFloat(quote['10. change percent']?.replace('%', '')) || 0,
            volume: parseInt(quote['06. volume']) || 0,
          });
        }
      } catch {
        continue;
      }
    }
    
    if (quoteMap.size === 0) {
      console.log('Alpha Vantage: No successful responses');
      return null;
    }
    
    const stocks: Stock[] = baseStocks.map((s) => {
      const quote = quoteMap.get(s.symbol);
      if (quote) {
        return { ...s, price: quote.price, change: quote.change, changePercent: quote.changePercent, volume: quote.volume };
      }
      return s;
    });
    
    console.log(`Alpha Vantage: Fetched ${quoteMap.size} stocks`);
    return stocks;
  } catch (error) {
    console.log('Alpha Vantage error:', error);
    return null;
  }
};

// Finnhub API fetch (using npm package)
const fetchFromFinnhub = async (market: MarketType): Promise<Stock[] | null> => {
  const apiKey = process.env.NEXT_PUBLIC_FINNHUB_API_KEY || '';
  
  if (!apiKey) {
    console.log('Finnhub: No API key');
    return null;
  }
  
  // For Indian stocks, use NSE suffix (.NS) for Finnhub
  // For US stocks, use plain symbols
  const symbols = market === 'US' 
    ? ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'WMT']
    : ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS', 'WIPRO.NS', 'HINDUNILVR.NS', 'LT.NS', 'SBIN.NS'];
  
  const baseSymbols = market === 'US' 
    ? ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'WMT']
    : ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'BHARTIARTL', 'WIPRO', 'HINDUNILVR', 'LT', 'SBIN'];
  
  const baseStocks = getStocksByMarket(market);
  
  try {
    console.log('Finnhub: Fetching for symbols:', symbols);
    
    const fetchPromises = symbols.map(async (symbol, index) => {
      try {
        console.log('Finnhub: Fetching quote for:', symbol);
        const response = await fetch(
          `https://finnhub.io/api/v1/quote?symbol=${symbol}&token=${apiKey}`
        );
        
        if (!response.ok) {
          console.log('Finnhub: Response not OK for', symbol, response.status);
          return null;
        }
        
        const data = await response.json();
        console.log('Finnhub: Response for', symbol, ':', data);
        
        // Finnhub returns: { c: current, d: change, dp: percent change, h: high, l: low, o: open, pc: previous close, t: timestamp }
        if (data.c && data.c > 0) {
          // Use base symbol (without .NS) as the key for mapping
          console.log('Finnhub: Got price for', symbol, ':', data.c);
          return {
            symbol: baseSymbols[index],
            price: data.c,
            change: data.d || 0,
            changePercent: data.dp || 0,
            volume: 0, // Finnhub quote doesn't include volume
          };
        }
        console.log('Finnhub: No valid price for', symbol);
        return null;
      } catch (err) {
        console.log('Finnhub: Error for', symbol, err);
        return null;
      }
    });
    
    const results = await Promise.all(fetchPromises);
    const successfulQuotes = results.filter(r => r !== null);
    
    console.log('Finnhub: Total successful quotes:', successfulQuotes.length);
    
    if (successfulQuotes.length === 0) {
      console.log('Finnhub: No successful responses');
      return null;
    }
    
    const quoteMap = new Map<string, any>();
    successfulQuotes.forEach(q => {
      quoteMap.set(q!.symbol, q);
    });
    
    const stocks: Stock[] = baseStocks.map((s) => {
      const quote = quoteMap.get(s.symbol);
      if (quote) {
        console.log('Finnhub: Mapping', s.symbol, 'to price', quote.price);
        return { ...s, price: quote.price, change: quote.change, changePercent: quote.changePercent, volume: quote.volume };
      }
      console.log('Finnhub: No quote for', s.symbol, '- using base price', s.price);
      return s;
    });
    
    console.log(`Finnhub: Fetched ${successfulQuotes.length} stocks`);
    return stocks;
  } catch (error) {
    console.log('Finnhub error:', error);
    return null;
  }
};

// Main fetch function with fallback
export const fetchRealTimeData = async (market: MarketType): Promise<{ stocks: Stock[] | null; isReal: boolean; source: string }> => {
  console.log(`Fetching real-time data for ${market} market...`);
  
  // Try Twelve Data first (best rate limits)
  console.log('Trying Twelve Data API...');
  let stocks = await fetchFromTwelveData(market);
  if (stocks) {
    console.log('Twelve Data succeeded!');
    return { stocks, isReal: true, source: 'Twelve Data' };
  }
  console.log('Twelve Data failed, trying Alpha Vantage...');
  
  // Try Alpha Vantage second
  stocks = await fetchFromAlphaVantage(market);
  if (stocks) {
    console.log('Alpha Vantage succeeded!');
    return { stocks, isReal: true, source: 'Alpha Vantage' };
  }
  console.log('Alpha Vantage failed, trying Finnhub...');
  
  // Try Finnhub last
  stocks = await fetchFromFinnhub(market);
  if (stocks) {
    console.log('Finnhub succeeded!');
    return { stocks, isReal: true, source: 'Finnhub' };
  }
  console.log('Finnhub failed - all APIs failed, using sample data');
  
  // All APIs failed
  return { stocks: null, isReal: false, source: 'None' };
};

// Finnhub API: Search for stocks by symbol or name and fetch quotes
export const searchStocks = async (query: string, market: MarketType): Promise<Stock[]> => {
  const apiKey = process.env.NEXT_PUBLIC_FINNHUB_API_KEY || '';
  
  if (!apiKey || !query.trim()) {
    return [];
  }
  
  try {
    // Search across all exchanges (don't filter by exchange to get more results)
    const response = await fetch(
      `https://finnhub.io/api/v1/search?q=${encodeURIComponent(query)}&token=${apiKey}`
    );
    
    if (!response.ok) {
      console.log('Stock search: Response not OK', response.status);
      return [];
    }
    
    const data = await response.json();
    console.log('Stock search results for:', query, data);
    
    if (!data.result || data.result.length === 0) {
      return [];
    }
    
    // Filter for common stock types and get top 10 results
    // Include all types: Common Stock, ETF, etc.
    const searchResults = data.result
      .filter((item: any) => 
        item.type === 'Common Stock' || 
        item.type === 'ETP' || 
        item.type === 'ETF' || 
        item.type === 'REIT'
      )
      .slice(0, 10);
    
    // Fetch quotes for each stock
    const stocks: Stock[] = [];
    
    for (const item of searchResults) {
      try {
        // Fetch quote for this stock
        const quoteResponse = await fetch(
          `https://finnhub.io/api/v1/quote?symbol=${encodeURIComponent(item.symbol)}&token=${apiKey}`
        );
        
        let price = 0;
        let change = 0;
        let changePercent = 0;
        
        if (quoteResponse.ok) {
          const quoteData = await quoteResponse.json();
          if (quoteData.c && quoteData.c > 0) {
            price = quoteData.c;
            change = quoteData.d || 0;
            changePercent = quoteData.dp || 0;
          }
        }
        
        stocks.push({
          symbol: item.symbol,
          name: item.description,
          price: price,
          change: change,
          changePercent: changePercent,
          volume: 0,
          marketCap: 0,
          high52w: 0,
          low52w: 0,
          pe: 0,
          eps: 0,
          dividend: 0,
          sector: item.friendlyType || 'Stock',
        });
      } catch (err) {
        // If quote fetch fails, still add the stock with 0 price
        stocks.push({
          symbol: item.symbol,
          name: item.description,
          price: 0,
          change: 0,
          changePercent: 0,
          volume: 0,
          marketCap: 0,
          high52w: 0,
          low52w: 0,
          pe: 0,
          eps: 0,
          dividend: 0,
          sector: item.friendlyType || 'Stock',
        });
      }
    }
    
    console.log('Filtered stocks with quotes:', stocks);
    return stocks;
  } catch (error) {
    console.log('Stock search error:', error);
    return [];
  }
};

// Finnhub API: Get quote for a specific stock
export const getStockQuote = async (symbol: string): Promise<{ price: number; change: number; changePercent: number } | null> => {
  const apiKey = process.env.NEXT_PUBLIC_FINNHUB_API_KEY || '';
  
  if (!apiKey || !symbol.trim()) {
    return null;
  }
  
  try {
    const response = await fetch(
      `https://finnhub.io/api/v1/quote?symbol=${encodeURIComponent(symbol)}&token=${apiKey}`
    );
    
    if (!response.ok) {
      return null;
    }
    
    const data = await response.json();
    
    if (data.c && data.c > 0) {
      return {
        price: data.c,
        change: data.d || 0,
        changePercent: data.dp || 0,
      };
    }
    
    return null;
  } catch (error) {
    console.log('Stock quote error:', error);
    return null;
  }
};
