import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import {
  TrendingUp,
  TrendingDown,
  Users,
  DollarSign,
  Clock,
  Eye,
  MessageSquare,
  Share2,
  Bookmark,
  BookmarkCheck,
} from 'lucide-react';

const MarketCard = ({ market, onTrade, onView, onBookmark, isBookmarked = false }) => {
  const [priceA, setPriceA] = useState(market.price_a || 0.5);
  const [priceB, setPriceB] = useState(market.price_b || 0.5);
  const [volume24h, setVolume24h] = useState(market.volume_24h || 0);
  const [trendingScore, setTrendingScore] = useState(market.trending_score || 0);

  useEffect(() => {
    // Simulate real-time price updates
    const interval = setInterval(() => {
      // Small random price fluctuations
      const fluctuation = (Math.random() - 0.5) * 0.02;
      setPriceA(prev => Math.max(0.01, Math.min(0.99, prev + fluctuation)));
      setPriceB(prev => Math.max(0.01, Math.min(0.99, prev - fluctuation)));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getCategoryColor = (category) => {
    const colors = {
      POLITICS: 'bg-red-100 text-red-800',
      SPORTS: 'bg-green-100 text-green-800',
      ECONOMICS: 'bg-blue-100 text-blue-800',
      TECHNOLOGY: 'bg-purple-100 text-purple-800',
      ENTERTAINMENT: 'bg-pink-100 text-pink-800',
      SCIENCE: 'bg-indigo-100 text-indigo-800',
      OTHER: 'bg-gray-100 text-gray-800',
    };
    return colors[category] || colors.OTHER;
  };

  const getStatusColor = (status) => {
    const colors = {
      OPEN: 'bg-green-100 text-green-800',
      CLOSED: 'bg-yellow-100 text-yellow-800',
      RESOLVED: 'bg-blue-100 text-blue-800',
      CANCELLED: 'bg-red-100 text-red-800',
    };
    return colors[status] || colors.OPEN;
  };

  const formatTimeRemaining = (closesAt) => {
    const now = new Date();
    const closeTime = new Date(closesAt);
    const diff = closeTime - now;

    if (diff <= 0) return 'Closed';

    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));

    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  const formatVolume = (volume) => {
    if (volume >= 1000000) return `$${(volume / 1000000).toFixed(1)}M`;
    if (volume >= 1000) return `$${(volume / 1000).toFixed(1)}K`;
    return `$${volume.toFixed(0)}`;
  };

  const formatPrice = (price) => {
    return `$${(price * 100).toFixed(0)}`;
  };

  const getPriceChange = (current, previous) => {
    if (!previous) return 0;
    return ((current - previous) / previous) * 100;
  };

  const priceChangeA = getPriceChange(priceA, market.price_a);
  const priceChangeB = getPriceChange(priceB, market.price_b);

  return (
    <Card className="hover:shadow-lg transition-shadow duration-200">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center space-x-2 mb-2">
              <Badge className={getCategoryColor(market.category)}>
                {market.category}
              </Badge>
              <Badge className={getStatusColor(market.status)}>
                {market.status}
              </Badge>
              {trendingScore > 0.7 && (
                <Badge variant="secondary" className="bg-orange-100 text-orange-800">
                  <TrendingUp className="h-3 w-3 mr-1" />
                  Trending
                </Badge>
              )}
            </div>
            <CardTitle className="text-lg leading-tight">
              {market.title}
            </CardTitle>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onBookmark(market.id)}
            className="ml-2"
          >
            {isBookmarked ? (
              <BookmarkCheck className="h-4 w-4 text-blue-600" />
            ) : (
              <Bookmark className="h-4 w-4" />
            )}
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Question */}
        <p className="text-sm text-gray-600 line-clamp-2">
          {market.question}
        </p>

        {/* Price Display */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-600">
                {market.outcome_a}
              </span>
              <div className="flex items-center space-x-1">
                <span className="text-lg font-bold">
                  {formatPrice(priceA)}
                </span>
                {priceChangeA !== 0 && (
                  <span className={`text-xs ${
                    priceChangeA > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {priceChangeA > 0 ? '+' : ''}{priceChangeA.toFixed(1)}%
                  </span>
                )}
              </div>
            </div>
            <Progress value={priceA * 100} className="h-2" />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-600">
                {market.outcome_b}
              </span>
              <div className="flex items-center space-x-1">
                <span className="text-lg font-bold">
                  {formatPrice(priceB)}
                </span>
                {priceChangeB !== 0 && (
                  <span className={`text-xs ${
                    priceChangeB > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {priceChangeB > 0 ? '+' : ''}{priceChangeB.toFixed(1)}%
                  </span>
                )}
              </div>
            </div>
            <Progress value={priceB * 100} className="h-2" />
          </div>
        </div>

        {/* Market Stats */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-sm font-medium text-gray-600">Volume 24h</div>
            <div className="text-lg font-bold">
              {formatVolume(volume24h)}
            </div>
          </div>
          <div>
            <div className="text-sm font-medium text-gray-600">Total Volume</div>
            <div className="text-lg font-bold">
              {formatVolume(market.volume_total || 0)}
            </div>
          </div>
          <div>
            <div className="text-sm font-medium text-gray-600">Time Left</div>
            <div className="text-lg font-bold">
              {formatTimeRemaining(market.closes_at)}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between pt-2">
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => onView(market.id)}
            >
              <Eye className="h-4 w-4 mr-1" />
              View
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => onTrade(market.id)}
            >
              <DollarSign className="h-4 w-4 mr-1" />
              Trade
            </Button>
          </div>
          <div className="flex items-center space-x-2">
            <Button variant="ghost" size="sm">
              <MessageSquare className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm">
              <Share2 className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Additional Info */}
        <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t">
          <div className="flex items-center space-x-4">
            <span>Created {new Date(market.created_at).toLocaleDateString()}</span>
            <span>•</span>
            <span>ID: {market.id}</span>
          </div>
          <div className="flex items-center space-x-1">
            <Users className="h-3 w-3" />
            <span>1.2K traders</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default MarketCard;
