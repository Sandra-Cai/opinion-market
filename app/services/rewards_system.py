from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
import json
import logging

from app.core.database import SessionLocal
from app.models.user import User
from app.models.market import Market
from app.models.trade import Trade
from app.models.position import Position
from app.models.governance import GovernanceToken

logger = logging.getLogger(__name__)

class RewardType:
    """Reward types for the gamification system"""
    DAILY_LOGIN = "daily_login"
    FIRST_TRADE = "first_trade"
    TRADE_STREAK = "trade_streak"
    WINNING_STREAK = "winning_streak"
    MARKET_CREATION = "market_creation"
    VOLUME_MILESTONE = "volume_milestone"
    PROFIT_MILESTONE = "profit_milestone"
    GOVERNANCE_PARTICIPATION = "governance_participation"
    COMMUNITY_CONTRIBUTION = "community_contribution"
    REFERRAL_BONUS = "referral_bonus"
    SEASONAL_EVENT = "seasonal_event"
    ACHIEVEMENT_UNLOCK = "achievement_unlock"

class Achievement:
    """Achievement definitions"""
    ACHIEVEMENTS = {
        "first_blood": {
            "name": "First Blood",
            "description": "Complete your first trade",
            "reward": {"tokens": 50, "xp": 100},
            "condition": "first_trade"
        },
        "market_maker": {
            "name": "Market Maker",
            "description": "Create your first market",
            "reward": {"tokens": 100, "xp": 200},
            "condition": "first_market"
        },
        "volume_king": {
            "name": "Volume King",
            "description": "Reach $10,000 in trading volume",
            "reward": {"tokens": 200, "xp": 500},
            "condition": "volume_milestone",
            "threshold": 10000
        },
        "profit_master": {
            "name": "Profit Master",
            "description": "Earn $1,000 in profits",
            "reward": {"tokens": 300, "xp": 750},
            "condition": "profit_milestone",
            "threshold": 1000
        },
        "streak_champion": {
            "name": "Streak Champion",
            "description": "Maintain a 7-day trading streak",
            "reward": {"tokens": 150, "xp": 300},
            "condition": "trade_streak",
            "threshold": 7
        },
        "governance_participant": {
            "name": "Governance Participant",
            "description": "Vote on your first proposal",
            "reward": {"tokens": 75, "xp": 150},
            "condition": "first_governance_vote"
        },
        "community_leader": {
            "name": "Community Leader",
            "description": "Create 10 successful markets",
            "reward": {"tokens": 500, "xp": 1000},
            "condition": "markets_created",
            "threshold": 10
        },
        "prediction_guru": {
            "name": "Prediction Guru",
            "description": "Achieve 80% win rate with 50+ trades",
            "reward": {"tokens": 1000, "xp": 2000},
            "condition": "win_rate",
            "threshold": 0.8,
            "min_trades": 50
        }
    }

class RewardsSystem:
    """Comprehensive rewards and gamification system"""
    
    def __init__(self):
        self.daily_rewards = {}
        self.streak_trackers = {}
        self.achievement_cache = {}
    
    def check_daily_login_reward(self, user_id: int) -> Dict:
        """Check and award daily login reward"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {"error": "User not found"}
            
            today = datetime.utcnow().date()
            last_login = user.last_login.date() if user.last_login else None
            
            # Check if already claimed today
            if last_login == today:
                return {"message": "Daily reward already claimed", "claimed": True}
            
            # Calculate streak
            streak = self._calculate_login_streak(user_id, db)
            
            # Award tokens based on streak
            base_tokens = 10
            streak_bonus = min(streak * 2, 50)  # Max 50 bonus tokens
            total_tokens = base_tokens + streak_bonus
            
            # Award XP
            xp_gained = 50 + (streak * 10)
            
            # Update user
            user.available_balance += total_tokens
            user.last_login = datetime.utcnow()
            
            # Update governance tokens
            governance_tokens = db.query(GovernanceToken).filter(
                GovernanceToken.user_id == user_id
            ).first()
            
            if governance_tokens:
                governance_tokens.token_amount += total_tokens
            else:
                governance_tokens = GovernanceToken(
                    user_id=user_id,
                    token_amount=total_tokens
                )
                db.add(governance_tokens)
            
            db.commit()
            
            return {
                "reward_type": RewardType.DAILY_LOGIN,
                "tokens_awarded": total_tokens,
                "xp_gained": xp_gained,
                "streak": streak,
                "next_reward": "tomorrow"
            }
            
        finally:
            db.close()
    
    def _calculate_login_streak(self, user_id: int, db: Session) -> int:
        """Calculate user's login streak"""
        # This would typically be stored in a separate table
        # For now, we'll simulate based on last login
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.last_login:
            return 0
        
        # Simple streak calculation (in real implementation, would track daily logins)
        return 1  # Placeholder
    
    def check_trade_rewards(self, trade: Trade, user: User) -> Dict:
        """Check and award trade-related rewards"""
        db = SessionLocal()
        try:
            rewards = []
            
            # Check first trade reward
            if self._is_first_trade(user.id, db):
                first_trade_reward = self._award_first_trade_reward(user, db)
                rewards.append(first_trade_reward)
            
            # Check trade streak
            streak_reward = self._check_trade_streak(user.id, db)
            if streak_reward:
                rewards.append(streak_reward)
            
            # Check winning streak
            if trade.profit_loss > 0:
                winning_reward = self._check_winning_streak(user.id, db)
                if winning_reward:
                    rewards.append(winning_reward)
            
            # Check volume milestones
            volume_reward = self._check_volume_milestones(user, db)
            if volume_reward:
                rewards.append(volume_reward)
            
            # Check profit milestones
            profit_reward = self._check_profit_milestones(user, db)
            if profit_reward:
                rewards.append(profit_reward)
            
            return {
                "rewards_awarded": rewards,
                "total_tokens": sum(r.get("tokens", 0) for r in rewards),
                "total_xp": sum(r.get("xp", 0) for r in rewards)
            }
            
        finally:
            db.close()
    
    def _is_first_trade(self, user_id: int, db: Session) -> bool:
        """Check if this is the user's first trade"""
        trade_count = db.query(Trade).filter(Trade.user_id == user_id).count()
        return trade_count == 1
    
    def _award_first_trade_reward(self, user: User, db: Session) -> Dict:
        """Award first trade reward"""
        tokens = 50
        xp = 100
        
        user.available_balance += tokens
        
        # Update governance tokens
        governance_tokens = db.query(GovernanceToken).filter(
            GovernanceToken.user_id == user.id
        ).first()
        
        if governance_tokens:
            governance_tokens.token_amount += tokens
        else:
            governance_tokens = GovernanceToken(
                user_id=user.id,
                token_amount=tokens
            )
            db.add(governance_tokens)
        
        db.commit()
        
        return {
            "type": RewardType.FIRST_TRADE,
            "tokens": tokens,
            "xp": xp,
            "message": "First trade completed!"
        }
    
    def _check_trade_streak(self, user_id: int, db: Session) -> Optional[Dict]:
        """Check and award trade streak rewards"""
        # Get recent trades
        recent_trades = db.query(Trade).filter(
            Trade.user_id == user_id,
            Trade.created_at >= datetime.utcnow() - timedelta(days=30)
        ).order_by(Trade.created_at).all()
        
        if len(recent_trades) < 2:
            return None
        
        # Calculate streak
        streak = self._calculate_trade_streak(recent_trades)
        
        # Award streak rewards
        if streak in [3, 7, 14, 30]:
            tokens = streak * 5
            xp = streak * 10
            
            return {
                "type": RewardType.TRADE_STREAK,
                "tokens": tokens,
                "xp": xp,
                "streak": streak,
                "message": f"{streak}-day trading streak!"
            }
        
        return None
    
    def _calculate_trade_streak(self, trades: List[Trade]) -> int:
        """Calculate consecutive days of trading"""
        if not trades:
            return 0
        
        # Group trades by date
        trade_dates = set(trade.created_at.date() for trade in trades)
        sorted_dates = sorted(trade_dates)
        
        # Find longest streak
        max_streak = 1
        current_streak = 1
        
        for i in range(1, len(sorted_dates)):
            if (sorted_dates[i] - sorted_dates[i-1]).days == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        
        return max_streak
    
    def _check_winning_streak(self, user_id: int, db: Session) -> Optional[Dict]:
        """Check and award winning streak rewards"""
        # Get recent winning trades
        recent_trades = db.query(Trade).filter(
            Trade.user_id == user_id,
            Trade.profit_loss > 0,
            Trade.created_at >= datetime.utcnow() - timedelta(days=30)
        ).order_by(Trade.created_at).all()
        
        if len(recent_trades) < 3:
            return None
        
        # Calculate winning streak
        streak = self._calculate_winning_streak(recent_trades)
        
        # Award streak rewards
        if streak in [3, 5, 10]:
            tokens = streak * 10
            xp = streak * 20
            
            return {
                "type": RewardType.WINNING_STREAK,
                "tokens": tokens,
                "xp": xp,
                "streak": streak,
                "message": f"{streak}-trade winning streak!"
            }
        
        return None
    
    def _calculate_winning_streak(self, trades: List[Trade]) -> int:
        """Calculate consecutive winning trades"""
        if not trades:
            return 0
        
        max_streak = 1
        current_streak = 1
        
        for i in range(1, len(trades)):
            if trades[i].profit_loss > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        
        return max_streak
    
    def _check_volume_milestones(self, user: User, db: Session) -> Optional[Dict]:
        """Check and award volume milestone rewards"""
        milestones = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        
        for milestone in milestones:
            if user.total_volume >= milestone and not self._milestone_awarded(user.id, "volume", milestone, db):
                tokens = milestone // 100
                xp = milestone // 50
                
                self._mark_milestone_awarded(user.id, "volume", milestone, db)
                
                return {
                    "type": RewardType.VOLUME_MILESTONE,
                    "tokens": tokens,
                    "xp": xp,
                    "milestone": milestone,
                    "message": f"${milestone:,} trading volume milestone!"
                }
        
        return None
    
    def _check_profit_milestones(self, user: User, db: Session) -> Optional[Dict]:
        """Check and award profit milestone rewards"""
        milestones = [100, 500, 1000, 5000, 10000, 50000, 100000]
        
        for milestone in milestones:
            if user.total_profit >= milestone and not self._milestone_awarded(user.id, "profit", milestone, db):
                tokens = milestone // 10
                xp = milestone // 5
                
                self._mark_milestone_awarded(user.id, "profit", milestone, db)
                
                return {
                    "type": RewardType.PROFIT_MILESTONE,
                    "tokens": tokens,
                    "xp": xp,
                    "milestone": milestone,
                    "message": f"${milestone:,} profit milestone!"
                }
        
        return None
    
    def _milestone_awarded(self, user_id: int, milestone_type: str, value: int, db: Session) -> bool:
        """Check if milestone was already awarded"""
        # This would typically be stored in a rewards table
        # For now, return False (not awarded)
        return False
    
    def _mark_milestone_awarded(self, user_id: int, milestone_type: str, value: int, db: Session):
        """Mark milestone as awarded"""
        # This would typically be stored in a rewards table
        pass
    
    def check_market_creation_reward(self, user_id: int, market: Market) -> Dict:
        """Check and award market creation rewards"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {"error": "User not found"}
            
            # Check if first market
            market_count = db.query(Market).filter(Market.creator_id == user_id).count()
            
            if market_count == 1:
                # First market reward
                tokens = 100
                xp = 200
                
                user.available_balance += tokens
                
                # Update governance tokens
                governance_tokens = db.query(GovernanceToken).filter(
                    GovernanceToken.user_id == user_id
                ).first()
                
                if governance_tokens:
                    governance_tokens.token_amount += tokens
                else:
                    governance_tokens = GovernanceToken(
                        user_id=user_id,
                        token_amount=tokens
                    )
                    db.add(governance_tokens)
                
                db.commit()
                
                return {
                    "type": RewardType.MARKET_CREATION,
                    "tokens": tokens,
                    "xp": xp,
                    "message": "First market created!"
                }
            
            # Regular market creation reward
            tokens = 25
            xp = 50
            
            user.available_balance += tokens
            
            # Update governance tokens
            governance_tokens = db.query(GovernanceToken).filter(
                GovernanceToken.user_id == user_id
            ).first()
            
            if governance_tokens:
                governance_tokens.token_amount += tokens
            
            db.commit()
            
            return {
                "type": RewardType.MARKET_CREATION,
                "tokens": tokens,
                "xp": xp,
                "message": "Market created successfully!"
            }
            
        finally:
            db.close()
    
    def check_governance_reward(self, user_id: int, proposal_id: int) -> Dict:
        """Check and award governance participation rewards"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {"error": "User not found"}
            
            # Check if first governance vote
            vote_count = db.query(func.count()).select_from(
                db.query(GovernanceToken).filter(
                    GovernanceToken.user_id == user_id
                ).subquery()
            ).scalar()
            
            if vote_count == 0:
                # First governance participation
                tokens = 75
                xp = 150
                
                user.available_balance += tokens
                
                # Update governance tokens
                governance_tokens = db.query(GovernanceToken).filter(
                    GovernanceToken.user_id == user_id
                ).first()
                
                if governance_tokens:
                    governance_tokens.token_amount += tokens
                else:
                    governance_tokens = GovernanceToken(
                        user_id=user_id,
                        token_amount=tokens
                    )
                    db.add(governance_tokens)
                
                db.commit()
                
                return {
                    "type": RewardType.GOVERNANCE_PARTICIPATION,
                    "tokens": tokens,
                    "xp": xp,
                    "message": "First governance participation!"
                }
            
            # Regular governance participation
            tokens = 25
            xp = 50
            
            user.available_balance += tokens
            
            # Update governance tokens
            governance_tokens = db.query(GovernanceToken).filter(
                GovernanceToken.user_id == user_id
            ).first()
            
            if governance_tokens:
                governance_tokens.token_amount += tokens
            
            db.commit()
            
            return {
                "type": RewardType.GOVERNANCE_PARTICIPATION,
                "tokens": tokens,
                "xp": xp,
                "message": "Governance participation rewarded!"
            }
            
        finally:
            db.close()
    
    def get_user_achievements(self, user_id: int) -> Dict:
        """Get user's achievements and progress"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {"error": "User not found"}
            
            achievements = []
            progress = {}
            
            # Check each achievement
            for achievement_id, achievement in Achievement.ACHIEVEMENTS.items():
                is_unlocked = self._check_achievement(user_id, achievement_id, db)
                progress_data = self._get_achievement_progress(user_id, achievement_id, db)
                
                achievements.append({
                    "id": achievement_id,
                    "name": achievement["name"],
                    "description": achievement["description"],
                    "unlocked": is_unlocked,
                    "progress": progress_data,
                    "reward": achievement["reward"]
                })
                
                progress[achievement_id] = progress_data
            
            return {
                "user_id": user_id,
                "achievements": achievements,
                "progress": progress,
                "total_unlocked": len([a for a in achievements if a["unlocked"]]),
                "total_achievements": len(achievements)
            }
            
        finally:
            db.close()
    
    def _check_achievement(self, user_id: int, achievement_id: str, db: Session) -> bool:
        """Check if achievement is unlocked"""
        achievement = Achievement.ACHIEVEMENTS.get(achievement_id)
        if not achievement:
            return False
        
        condition = achievement["condition"]
        
        if condition == "first_trade":
            trade_count = db.query(Trade).filter(Trade.user_id == user_id).count()
            return trade_count > 0
        
        elif condition == "first_market":
            market_count = db.query(Market).filter(Market.creator_id == user_id).count()
            return market_count > 0
        
        elif condition == "volume_milestone":
            user = db.query(User).filter(User.id == user_id).first()
            threshold = achievement.get("threshold", 10000)
            return user.total_volume >= threshold if user else False
        
        elif condition == "profit_milestone":
            user = db.query(User).filter(User.id == user_id).first()
            threshold = achievement.get("threshold", 1000)
            return user.total_profit >= threshold if user else False
        
        elif condition == "trade_streak":
            threshold = achievement.get("threshold", 7)
            recent_trades = db.query(Trade).filter(
                Trade.user_id == user_id,
                Trade.created_at >= datetime.utcnow() - timedelta(days=30)
            ).all()
            streak = self._calculate_trade_streak(recent_trades)
            return streak >= threshold
        
        elif condition == "first_governance_vote":
            # Check if user has participated in governance
            governance_tokens = db.query(GovernanceToken).filter(
                GovernanceToken.user_id == user_id
            ).first()
            return governance_tokens is not None
        
        elif condition == "markets_created":
            threshold = achievement.get("threshold", 10)
            market_count = db.query(Market).filter(Market.creator_id == user_id).count()
            return market_count >= threshold
        
        elif condition == "win_rate":
            threshold = achievement.get("threshold", 0.8)
            min_trades = achievement.get("min_trades", 50)
            trades = db.query(Trade).filter(Trade.user_id == user_id).all()
            
            if len(trades) < min_trades:
                return False
            
            winning_trades = len([t for t in trades if t.profit_loss > 0])
            win_rate = winning_trades / len(trades)
            return win_rate >= threshold
        
        return False
    
    def _get_achievement_progress(self, user_id: int, achievement_id: str, db: Session) -> Dict:
        """Get progress towards achievement"""
        achievement = Achievement.ACHIEVEMENTS.get(achievement_id)
        if not achievement:
            return {"current": 0, "target": 0, "percentage": 0}
        
        condition = achievement["condition"]
        
        if condition == "volume_milestone":
            user = db.query(User).filter(User.id == user_id).first()
            target = achievement.get("threshold", 10000)
            current = user.total_volume if user else 0
            percentage = min(100, (current / target) * 100) if target > 0 else 0
            
            return {
                "current": current,
                "target": target,
                "percentage": percentage
            }
        
        elif condition == "profit_milestone":
            user = db.query(User).filter(User.id == user_id).first()
            target = achievement.get("threshold", 1000)
            current = user.total_profit if user else 0
            percentage = min(100, (current / target) * 100) if target > 0 else 0
            
            return {
                "current": current,
                "target": target,
                "percentage": percentage
            }
        
        elif condition == "trade_streak":
            target = achievement.get("threshold", 7)
            recent_trades = db.query(Trade).filter(
                Trade.user_id == user_id,
                Trade.created_at >= datetime.utcnow() - timedelta(days=30)
            ).all()
            current = self._calculate_trade_streak(recent_trades)
            percentage = min(100, (current / target) * 100) if target > 0 else 0
            
            return {
                "current": current,
                "target": target,
                "percentage": percentage
            }
        
        elif condition == "markets_created":
            target = achievement.get("threshold", 10)
            current = db.query(Market).filter(Market.creator_id == user_id).count()
            percentage = min(100, (current / target) * 100) if target > 0 else 0
            
            return {
                "current": current,
                "target": target,
                "percentage": percentage
            }
        
        elif condition == "win_rate":
            target = achievement.get("threshold", 0.8)
            min_trades = achievement.get("min_trades", 50)
            trades = db.query(Trade).filter(Trade.user_id == user_id).all()
            
            if len(trades) < min_trades:
                current = 0
            else:
                winning_trades = len([t for t in trades if t.profit_loss > 0])
                current = winning_trades / len(trades)
            
            percentage = min(100, (current / target) * 100) if target > 0 else 0
            
            return {
                "current": current,
                "target": target,
                "percentage": percentage,
                "total_trades": len(trades),
                "min_trades_required": min_trades
            }
        
        return {"current": 0, "target": 0, "percentage": 0}

# Global rewards system instance
rewards_system = RewardsSystem()

def get_rewards_system() -> RewardsSystem:
    """Get the global rewards system instance"""
    return rewards_system
