class TechnicalIndicators {
    // 简单移动平均线 (MA)
    static calculateMA(data, period = 20) {
        if (data.length < period) return [];
        
        const ma = [];
        for (let i = period - 1; i < data.length; i++) {
            const sum = data.slice(i - period + 1, i + 1).reduce((acc, item) => acc + item.price, 0);
            ma.push({
                time: data[i].time,
                value: sum / period
            });
        }
        return ma;
    }
    
    // 指数移动平均线 (EMA)
    static calculateEMA(data, period = 20) {
        if (data.length < period) return [];
        
        const ema = [];
        const multiplier = 2 / (period + 1);
        
        // 第一个EMA值使用简单移动平均
        const firstMA = data.slice(0, period).reduce((acc, item) => acc + item.price, 0) / period;
        ema.push({
            time: data[period - 1].time,
            value: firstMA
        });
        
        // 计算后续EMA值
        for (let i = period; i < data.length; i++) {
            const currentEMA = (data[i].price - ema[ema.length - 1].value) * multiplier + ema[ema.length - 1].value;
            ema.push({
                time: data[i].time,
                value: currentEMA
            });
        }
        
        return ema;
    }
    
    // 布林带 (BOLL)
    static calculateBOLL(data, period = 20, stdDev = 2) {
        if (data.length < period) return { upper: [], middle: [], lower: [] };
        
        const ma = this.calculateMA(data, period);
        const boll = { upper: [], middle: [], lower: [] };
        
        for (let i = period - 1; i < data.length; i++) {
            const prices = data.slice(i - period + 1, i + 1).map(item => item.price);
            const mean = prices.reduce((acc, price) => acc + price, 0) / period;
            const variance = prices.reduce((acc, price) => acc + Math.pow(price - mean, 2), 0) / period;
            const standardDeviation = Math.sqrt(variance);
            
            boll.upper.push({
                time: data[i].time,
                value: mean + (stdDev * standardDeviation)
            });
            boll.middle.push({
                time: data[i].time,
                value: mean
            });
            boll.lower.push({
                time: data[i].time,
                value: mean - (stdDev * standardDeviation)
            });
        }
        
        return boll;
    }
    
    // 相对强弱指标 (RSI)
    static calculateRSI(data, period = 14) {
        if (data.length < period + 1) return [];
        
        const rsi = [];
        const gains = [];
        const losses = [];
        
        // 计算价格变化
        for (let i = 1; i < data.length; i++) {
            const change = data[i].price - data[i - 1].price;
            gains.push(change > 0 ? change : 0);
            losses.push(change < 0 ? Math.abs(change) : 0);
        }
        
        // 计算初始平均收益和损失
        let avgGain = gains.slice(0, period).reduce((acc, gain) => acc + gain, 0) / period;
        let avgLoss = losses.slice(0, period).reduce((acc, loss) => acc + loss, 0) / period;
        
        // 计算第一个RSI值
        if (avgLoss === 0) {
            rsi.push({
                time: data[period].time,
                value: 100
            });
        } else {
            const rs = avgGain / avgLoss;
            rsi.push({
                time: data[period].time,
                value: 100 - (100 / (1 + rs))
            });
        }
        
        // 计算后续RSI值
        for (let i = period + 1; i < data.length; i++) {
            const currentGain = gains[i - 1];
            const currentLoss = losses[i - 1];
            
            avgGain = (avgGain * (period - 1) + currentGain) / period;
            avgLoss = (avgLoss * (period - 1) + currentLoss) / period;
            
            if (avgLoss === 0) {
                rsi.push({
                    time: data[i].time,
                    value: 100
                });
            } else {
                const rs = avgGain / avgLoss;
                rsi.push({
                    time: data[i].time,
                    value: 100 - (100 / (1 + rs))
                });
            }
        }
        
        return rsi;
    }
    
    // MACD (异同移动平均线)
    static calculateMACD(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
        if (data.length < slowPeriod + signalPeriod) return { macd: [], signal: [], histogram: [] };
        
        // 计算EMA
        const fastEMA = this.calculateEMA(data, fastPeriod);
        const slowEMA = this.calculateEMA(data, slowPeriod);
        
        // 计算MACD线
        const macdLine = [];
        const startIndex = slowPeriod - 1;
        
        for (let i = startIndex; i < fastEMA.length; i++) {
            const fastValue = fastEMA[i].value;
            const slowValue = slowEMA[i - (slowPeriod - fastPeriod)].value;
            macdLine.push({
                time: fastEMA[i].time,
                value: fastValue - slowValue
            });
        }
        
        // 计算信号线 (MACD线的EMA)
        const signalLine = this.calculateEMA(macdLine, signalPeriod);
        
        // 计算柱状图
        const histogram = [];
        const signalStartIndex = signalPeriod - 1;
        
        for (let i = signalStartIndex; i < macdLine.length; i++) {
            const macdValue = macdLine[i].value;
            const signalValue = signalLine[i - signalStartIndex].value;
            histogram.push({
                time: macdLine[i].time,
                value: macdValue - signalValue
            });
        }
        
        return {
            macd: macdLine,
            signal: signalLine,
            histogram: histogram
        };
    }
    
    // KDJ (随机指标)
    static calculateKDJ(data, kPeriod = 9, dPeriod = 3, jPeriod = 3) {
        if (data.length < kPeriod + 2) return { k: [], d: [], j: [] };
        
        const kValues = [];
        const dValues = [];
        const jValues = [];
        
        // 计算RSV (Raw Stochastic Value)
        for (let i = kPeriod - 1; i < data.length; i++) {
            const periodData = data.slice(i - kPeriod + 1, i + 1);
            const low = Math.min(...periodData.map(item => item.price));
            const high = Math.max(...periodData.map(item => item.price));
            const close = data[i].price;
            
            if (high === low) {
                kValues.push({ time: data[i].time, value: 50 }); // 如果最高价等于最低价，RSV为50
            } else {
                const rsv = ((close - low) / (high - low)) * 100;
                kValues.push({ time: data[i].time, value: rsv });
            }
        }
        
        // 计算K值 (RSV的3日移动平均)
        const kLine = this.calculateMA(kValues, dPeriod);
        
        // 计算D值 (K值的3日移动平均)
        const dLine = this.calculateMA(kLine, dPeriod);
        
        // 计算J值 (3K - 2D)
        const jStartIndex = dPeriod - 1;
        for (let i = jStartIndex; i < kLine.length; i++) {
            const kValue = kLine[i].value;
            const dValue = dLine[i - jStartIndex].value;
            jValues.push({
                time: kLine[i].time,
                value: 3 * kValue - 2 * dValue
            });
        }
        
        return {
            k: kLine,
            d: dLine,
            j: jValues
        };
    }
    
    // 计算所有可用的技术指标
    static calculateAllIndicators(priceData, options = {}) {
        const {
            maPeriod = 20,
            emaPeriod = 20,
            bollPeriod = 20,
            rsiPeriod = 14,
            macdFast = 12,
            macdSlow = 26,
            macdSignal = 9,
            kdjK = 9,
            kdjD = 3,
            kdjJ = 3
        } = options;
        
        return {
            ma: this.calculateMA(priceData, maPeriod),
            ema: this.calculateEMA(priceData, emaPeriod),
            boll: this.calculateBOLL(priceData, bollPeriod),
            rsi: this.calculateRSI(priceData, rsiPeriod),
            macd: this.calculateMACD(priceData, macdFast, macdSlow, macdSignal),
            kdj: this.calculateKDJ(priceData, kdjK, kdjD, kdjJ)
        };
    }
}

module.exports = TechnicalIndicators;