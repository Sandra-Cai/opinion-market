from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.services.rewards_system import get_rewards_system
from app.models.trade import Trade

router = APIRouter()


@router.post("/daily-login")
def claim_daily_login_reward(current_user: User = Depends(get_current_user)):
    """Claim daily login reward"""
    rewards_system = get_rewards_system()
    result = rewards_system.check_daily_login_reward(current_user.id)

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=result["error"]
        )

    if result.get("claimed", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Daily reward already claimed",
        )

    return result


@router.get("/achievements")
def get_user_achievements(current_user: User = Depends(get_current_user)):
    """Get user's achievements and progress"""
    rewards_system = get_rewards_system()
    achievements = rewards_system.get_user_achievements(current_user.id)

    if "error" in achievements:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=achievements["error"]
        )

    return achievements


@router.get("/achievements/{achievement_id}")
def get_achievement_details(
    achievement_id: str, current_user: User = Depends(get_current_user)
):
    """Get details for a specific achievement"""
    rewards_system = get_rewards_system()
    achievements = rewards_system.get_user_achievements(current_user.id)

    if "error" in achievements:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=achievements["error"]
        )

    # Find the specific achievement
    achievement = next(
        (a for a in achievements["achievements"] if a["id"] == achievement_id), None
    )

    if not achievement:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Achievement not found"
        )

    return achievement


@router.get("/leaderboard/achievements")
def get_achievement_leaderboard(
    limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)
):
    """Get leaderboard of users by achievements unlocked"""
    rewards_system = get_rewards_system()

    # Get all users with their achievement counts
    users = db.query(User).filter(User.is_active == True).all()

    user_achievements = []
    for user in users:
        try:
            achievements = rewards_system.get_user_achievements(user.id)
            if "error" not in achievements:
                user_achievements.append(
                    {
                        "user_id": user.id,
                        "username": user.username,
                        "achievements_unlocked": achievements["total_unlocked"],
                        "total_achievements": achievements["total_achievements"],
                        "completion_percentage": (
                            (
                                achievements["total_unlocked"]
                                / achievements["total_achievements"]
                            )
                            * 100
                            if achievements["total_achievements"] > 0
                            else 0
                        ),
                    }
                )
        except:
            continue

    # Sort by achievements unlocked
    user_achievements.sort(key=lambda x: x["achievements_unlocked"], reverse=True)

    return {
        "leaderboard": user_achievements[:limit],
        "total_users": len(user_achievements),
    }


@router.get("/stats")
def get_rewards_stats(current_user: User = Depends(get_current_user)):
    """Get user's rewards statistics"""
    rewards_system = get_rewards_system()
    achievements = rewards_system.get_user_achievements(current_user.id)

    if "error" in achievements:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=achievements["error"]
        )

    # Calculate additional stats
    unlocked_achievements = [a for a in achievements["achievements"] if a["unlocked"]]
    locked_achievements = [a for a in achievements["achievements"] if not a["unlocked"]]

    # Calculate total rewards earned
    total_tokens_earned = sum(a["reward"]["tokens"] for a in unlocked_achievements)
    total_xp_earned = sum(a["reward"]["xp"] for a in unlocked_achievements)

    # Calculate potential rewards
    potential_tokens = sum(a["reward"]["tokens"] for a in locked_achievements)
    potential_xp = sum(a["reward"]["xp"] for a in locked_achievements)

    return {
        "user_id": current_user.id,
        "achievements_unlocked": len(unlocked_achievements),
        "achievements_locked": len(locked_achievements),
        "total_achievements": len(achievements["achievements"]),
        "completion_percentage": (
            (len(unlocked_achievements) / len(achievements["achievements"])) * 100
            if achievements["achievements"]
            else 0
        ),
        "rewards_earned": {"tokens": total_tokens_earned, "xp": total_xp_earned},
        "potential_rewards": {"tokens": potential_tokens, "xp": potential_xp},
        "recent_achievements": (
            unlocked_achievements[-5:]
            if len(unlocked_achievements) > 5
            else unlocked_achievements
        ),
        "next_achievements": locked_achievements[:5],
    }


@router.get("/milestones")
def get_user_milestones(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Get user's milestone progress"""
    # Define milestones
    volume_milestones = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    profit_milestones = [100, 500, 1000, 5000, 10000, 50000, 100000]
    trade_milestones = [10, 50, 100, 500, 1000, 5000, 10000]

    # Calculate progress
    volume_progress = []
    for milestone in volume_milestones:
        progress = (
            min(100, (current_user.total_volume / milestone) * 100)
            if milestone > 0
            else 0
        )
        volume_progress.append(
            {
                "milestone": milestone,
                "current": current_user.total_volume,
                "progress": progress,
                "achieved": current_user.total_volume >= milestone,
            }
        )

    profit_progress = []
    for milestone in profit_milestones:
        progress = (
            min(100, (current_user.total_profit / milestone) * 100)
            if milestone > 0
            else 0
        )
        profit_progress.append(
            {
                "milestone": milestone,
                "current": current_user.total_profit,
                "progress": progress,
                "achieved": current_user.total_profit >= milestone,
            }
        )

    trade_progress = []
    for milestone in trade_milestones:
        progress = (
            min(100, (current_user.total_trades / milestone) * 100)
            if milestone > 0
            else 0
        )
        trade_progress.append(
            {
                "milestone": milestone,
                "current": current_user.total_trades,
                "progress": progress,
                "achieved": current_user.total_trades >= milestone,
            }
        )

    return {
        "user_id": current_user.id,
        "volume_milestones": volume_progress,
        "profit_milestones": profit_progress,
        "trade_milestones": trade_progress,
        "next_milestones": {
            "volume": next((m for m in volume_progress if not m["achieved"]), None),
            "profit": next((m for m in profit_progress if not m["achieved"]), None),
            "trades": next((m for m in trade_progress if not m["achieved"]), None),
        },
    }


@router.get("/streaks")
def get_user_streaks(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Get user's current streaks"""
    # Calculate trading streak
    recent_trades = (
        db.query(Trade)
        .filter(
            Trade.user_id == current_user.id,
            Trade.created_at >= datetime.utcnow() - timedelta(days=30),
        )
        .order_by(Trade.created_at)
        .all()
    )

    # Calculate login streak (simplified)
    login_streak = (
        1
        if current_user.last_login
        and current_user.last_login.date() == datetime.utcnow().date()
        else 0
    )

    # Calculate winning streak
    winning_trades = [t for t in recent_trades if t.profit_loss > 0]
    winning_streak = 0
    if winning_trades:
        # Count consecutive winning trades from most recent
        for trade in reversed(recent_trades):
            if trade.profit_loss > 0:
                winning_streak += 1
            else:
                break

    return {
        "user_id": current_user.id,
        "trading_streak": {
            "current": len(set(t.created_at.date() for t in recent_trades)),
            "longest": len(
                set(t.created_at.date() for t in recent_trades)
            ),  # Simplified
            "last_trade": recent_trades[-1].created_at if recent_trades else None,
        },
        "login_streak": {
            "current": login_streak,
            "longest": login_streak,  # Simplified
            "last_login": current_user.last_login,
        },
        "winning_streak": {
            "current": winning_streak,
            "longest": winning_streak,  # Simplified
            "last_win": winning_trades[-1].created_at if winning_trades else None,
        },
    }


@router.get("/recent-rewards")
def get_recent_rewards(
    current_user: User = Depends(get_current_user), limit: int = Query(10, ge=1, le=50)
):
    """Get user's recent rewards and achievements"""
    rewards_system = get_rewards_system()
    achievements = rewards_system.get_user_achievements(current_user.id)

    if "error" in achievements:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=achievements["error"]
        )

    # Get recent achievements (simplified - in real implementation would track timestamps)
    recent_achievements = [a for a in achievements["achievements"] if a["unlocked"]][
        -limit:
    ]

    return {
        "user_id": current_user.id,
        "recent_achievements": recent_achievements,
        "total_rewards_earned": {
            "tokens": sum(a["reward"]["tokens"] for a in recent_achievements),
            "xp": sum(a["reward"]["xp"] for a in recent_achievements),
        },
    }


@router.get("/available-rewards")
def get_available_rewards(current_user: User = Depends(get_current_user)):
    """Get rewards that are close to being unlocked"""
    rewards_system = get_rewards_system()
    achievements = rewards_system.get_user_achievements(current_user.id)

    if "error" in achievements:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=achievements["error"]
        )

    # Find achievements that are close to being unlocked (>50% progress)
    close_achievements = []
    for achievement in achievements["achievements"]:
        if not achievement["unlocked"] and achievement["progress"]["percentage"] > 50:
            close_achievements.append(achievement)

    # Sort by progress percentage
    close_achievements.sort(key=lambda x: x["progress"]["percentage"], reverse=True)

    return {
        "user_id": current_user.id,
        "close_achievements": close_achievements,
        "total_potential_rewards": {
            "tokens": sum(a["reward"]["tokens"] for a in close_achievements),
            "xp": sum(a["reward"]["xp"] for a in close_achievements),
        },
    }
