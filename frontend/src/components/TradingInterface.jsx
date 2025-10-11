import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  DollarSign,
  TrendingUp,
  TrendingDown,
  Calculator,
  Info,
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap,
  Shield,
} from 'lucide-react';

const TradingInterface = ({ market, onTrade, userBalance = 1000 }) => {
  const [tradeType, setTradeType] = useState('BUY');
  const [outcome, setOutcome] = useState('OUTCOME_A');
  const [amount, setAmount] = useState('');
  const [orderType, setOrderType] = useState('MARKET');
  const [limitPrice, setLimitPrice] = useState('');
  const [stopPrice, setStopPrice] = useState('');
  const [timeInForce, setTimeInForce] = useState('GTC');
  const [expiresAt, setExpiresAt] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // Real-time price updates
  const [priceA, setPriceA] = useState(market.price_a || 0.5);
  const [priceB, setPriceB] = useState(market.price_b || 0.5);
  const [volume24h, setVolume24h] = useState(market.volume_24h || 0);

  useEffect(() => {
    // Simulate real-time price updates
    const interval = setInterval(() => {
      const fluctuation = (Math.random() - 0.5) * 0.01;
      setPriceA(prev => Math.max(0.01, Math.min(0.99, prev + fluctuation)));
      setPriceB(prev => Math.max(0.01, Math.min(0.99, prev - fluctuation)));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const currentPrice = outcome === 'OUTCOME_A' ? priceA : priceB;
  const currentPriceDisplay = `$${(currentPrice * 100).toFixed(0)}`;

  const calculateTradeValue = () => {
    if (!amount || isNaN(amount)) return 0;
    const numAmount = parseFloat(amount);
    const price = orderType === 'MARKET' ? currentPrice : parseFloat(limitPrice || 0);
    return numAmount * price;
  };

  const calculateFees = () => {
    const tradeValue = calculateTradeValue();
    return tradeValue * 0.02; // 2% trading fee
  };

  const calculateTotal = () => {
    const tradeValue = calculateTradeValue();
    const fees = calculateFees();
    return tradeValue + fees;
  };

  const calculateShares = () => {
    if (!amount || isNaN(amount)) return 0;
    const numAmount = parseFloat(amount);
    const price = orderType === 'MARKET' ? currentPrice : parseFloat(limitPrice || 0);
    return numAmount / price;
  };

  const validateTrade = () => {
    if (!amount || isNaN(amount) || parseFloat(amount) <= 0) {
      return 'Please enter a valid amount';
    }

    if (orderType === 'LIMIT' && (!limitPrice || isNaN(limitPrice) || parseFloat(limitPrice) <= 0)) {
      return 'Please enter a valid limit price';
    }

    if (orderType === 'STOP' && (!stopPrice || isNaN(stopPrice) || parseFloat(stopPrice) <= 0)) {
      return 'Please enter a valid stop price';
    }

    if (calculateTotal() > userBalance) {
      return 'Insufficient balance';
    }

    return null;
  };

  const handleTrade = async () => {
    const validationError = validateTrade();
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsSubmitting(true);
    setError('');
    setSuccess('');

    try {
      const tradeData = {
        trade_type: tradeType,
        outcome: outcome,
        amount: parseFloat(amount),
        market_id: market.id,
        order_type: orderType,
        price: orderType === 'LIMIT' ? parseFloat(limitPrice) : undefined,
        stop_price: orderType === 'STOP' ? parseFloat(stopPrice) : undefined,
        time_in_force: timeInForce,
        expires_at: expiresAt || undefined,
      };

      await onTrade(tradeData);
      setSuccess('Trade executed successfully!');
      
      // Reset form
      setAmount('');
      setLimitPrice('');
      setStopPrice('');
      setExpiresAt('');
    } catch (err) {
      setError(err.message || 'Trade failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  const formatCurrency = (value) => {
    return `$${value.toFixed(2)}`;
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <div className="space-y-6">
      {/* Market Info */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <DollarSign className="h-5 w-5" />
            <span>Trade {market.title}</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{market.outcome_a}</span>
                <div className="flex items-center space-x-2">
                  <span className="text-lg font-bold">
                    ${(priceA * 100).toFixed(0)}
                  </span>
                  <Badge variant="secondary">
                    {formatPercentage(priceA)}
                  </Badge>
                </div>
              </div>
              <Progress value={priceA * 100} className="h-2" />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{market.outcome_b}</span>
                <div className="flex items-center space-x-2">
                  <span className="text-lg font-bold">
                    ${(priceB * 100).toFixed(0)}
                  </span>
                  <Badge variant="secondary">
                    {formatPercentage(priceB)}
                  </Badge>
                </div>
              </div>
              <Progress value={priceB * 100} className="h-2" />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4 mt-4 text-center">
            <div>
              <div className="text-sm text-gray-600">Volume 24h</div>
              <div className="font-bold">{formatCurrency(volume24h)}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Total Volume</div>
              <div className="font-bold">{formatCurrency(market.volume_total || 0)}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Liquidity</div>
              <div className="font-bold">High</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Trading Form */}
      <Card>
        <CardHeader>
          <CardTitle>Place Order</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Trade Type */}
          <div className="grid grid-cols-2 gap-4">
            <Button
              variant={tradeType === 'BUY' ? 'default' : 'outline'}
              onClick={() => setTradeType('BUY')}
              className="flex items-center space-x-2"
            >
              <TrendingUp className="h-4 w-4" />
              <span>Buy</span>
            </Button>
            <Button
              variant={tradeType === 'SELL' ? 'default' : 'outline'}
              onClick={() => setTradeType('SELL')}
              className="flex items-center space-x-2"
            >
              <TrendingDown className="h-4 w-4" />
              <span>Sell</span>
            </Button>
          </div>

          {/* Outcome Selection */}
          <div className="space-y-2">
            <Label>Outcome</Label>
            <Select value={outcome} onValueChange={setOutcome}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="OUTCOME_A">{market.outcome_a}</SelectItem>
                <SelectItem value="OUTCOME_B">{market.outcome_b}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Order Type */}
          <div className="space-y-2">
            <Label>Order Type</Label>
            <Select value={orderType} onValueChange={setOrderType}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="MARKET">Market Order</SelectItem>
                <SelectItem value="LIMIT">Limit Order</SelectItem>
                <SelectItem value="STOP">Stop Order</SelectItem>
                <SelectItem value="STOP_LIMIT">Stop Limit Order</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Amount */}
          <div className="space-y-2">
            <Label>Amount ($)</Label>
            <Input
              type="number"
              placeholder="Enter amount"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              min="0"
              step="0.01"
            />
          </div>

          {/* Limit Price */}
          {orderType === 'LIMIT' && (
            <div className="space-y-2">
              <Label>Limit Price ($)</Label>
              <Input
                type="number"
                placeholder="Enter limit price"
                value={limitPrice}
                onChange={(e) => setLimitPrice(e.target.value)}
                min="0"
                step="0.01"
              />
            </div>
          )}

          {/* Stop Price */}
          {(orderType === 'STOP' || orderType === 'STOP_LIMIT') && (
            <div className="space-y-2">
              <Label>Stop Price ($)</Label>
              <Input
                type="number"
                placeholder="Enter stop price"
                value={stopPrice}
                onChange={(e) => setStopPrice(e.target.value)}
                min="0"
                step="0.01"
              />
            </div>
          )}

          {/* Time in Force */}
          <div className="space-y-2">
            <Label>Time in Force</Label>
            <Select value={timeInForce} onValueChange={setTimeInForce}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="GTC">Good Till Cancelled</SelectItem>
                <SelectItem value="IOC">Immediate or Cancel</SelectItem>
                <SelectItem value="FOK">Fill or Kill</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Expiration */}
          {timeInForce === 'GTC' && (
            <div className="space-y-2">
              <Label>Expires At (Optional)</Label>
              <Input
                type="datetime-local"
                value={expiresAt}
                onChange={(e) => setExpiresAt(e.target.value)}
              />
            </div>
          )}

          {/* Trade Summary */}
          <Card className="bg-gray-50">
            <CardContent className="p-4">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Current Price:</span>
                  <span className="font-medium">{currentPriceDisplay}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Trade Value:</span>
                  <span className="font-medium">{formatCurrency(calculateTradeValue())}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Trading Fee (2%):</span>
                  <span className="font-medium">{formatCurrency(calculateFees())}</span>
                </div>
                <div className="flex justify-between border-t pt-2">
                  <span className="text-sm font-medium">Total Cost:</span>
                  <span className="font-bold">{formatCurrency(calculateTotal())}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Shares:</span>
                  <span className="font-medium">{calculateShares().toFixed(2)}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Error/Success Messages */}
          {error && (
            <div className="flex items-center space-x-2 text-red-600 bg-red-50 p-3 rounded-md">
              <AlertTriangle className="h-4 w-4" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          {success && (
            <div className="flex items-center space-x-2 text-green-600 bg-green-50 p-3 rounded-md">
              <CheckCircle className="h-4 w-4" />
              <span className="text-sm">{success}</span>
            </div>
          )}

          {/* Submit Button */}
          <Button
            onClick={handleTrade}
            disabled={isSubmitting || !amount}
            className="w-full"
            size="lg"
          >
            {isSubmitting ? (
              <>
                <Clock className="h-4 w-4 mr-2 animate-spin" />
                Executing Trade...
              </>
            ) : (
              <>
                <Zap className="h-4 w-4 mr-2" />
                {tradeType} {outcome === 'OUTCOME_A' ? market.outcome_a : market.outcome_b}
              </>
            )}
          </Button>

          {/* Balance Info */}
          <div className="text-center text-sm text-gray-600">
            Available Balance: {formatCurrency(userBalance)}
          </div>
        </CardContent>
      </Card>

      {/* Risk Warning */}
      <Card className="border-yellow-200 bg-yellow-50">
        <CardContent className="p-4">
          <div className="flex items-start space-x-2">
            <Shield className="h-5 w-5 text-yellow-600 mt-0.5" />
            <div className="text-sm text-yellow-800">
              <p className="font-medium mb-1">Risk Warning</p>
              <p>
                Trading prediction markets involves risk. You may lose your entire investment.
                Only trade with money you can afford to lose. Past performance does not guarantee future results.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default TradingInterface;
