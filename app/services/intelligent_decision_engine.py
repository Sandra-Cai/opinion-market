"""
Intelligent Decision Engine
Advanced decision-making capabilities with explainable AI and multi-criteria analysis
"""

import asyncio
import logging
import time
import json
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class DecisionContext(Enum):
    """Decision context types"""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    RISK_MANAGEMENT = "risk_management"
    COMPLIANCE = "compliance"
    CUSTOMER = "customer"
    MARKET = "market"

class DecisionComplexity(Enum):
    """Decision complexity levels"""
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    HIGHLY_COMPLEX = 4
    EXTREMELY_COMPLEX = 5

class DecisionOutcome(Enum):
    """Decision outcome types"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    UNCERTAIN = "uncertain"
    PENDING = "pending"

@dataclass
class DecisionCriteria:
    criteria_id: str
    name: str
    weight: float
    threshold: float
    importance: str
    description: str
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DecisionOption:
    option_id: str
    name: str
    description: str
    criteria_scores: Dict[str, float]
    risk_factors: List[str]
    benefits: List[str]
    costs: List[str]
    probability_of_success: float
    expected_value: float
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DecisionRequest:
    request_id: str
    context: DecisionContext
    complexity: DecisionComplexity
    description: str
    criteria: List[DecisionCriteria]
    options: List[DecisionOption]
    constraints: Dict[str, Any]
    deadline: Optional[datetime] = None
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"

@dataclass
class DecisionResult:
    result_id: str
    request_id: str
    recommended_option: str
    confidence_score: float
    reasoning: str
    alternative_options: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]
    implementation_plan: Dict[str, Any]
    monitoring_metrics: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DecisionHistory:
    history_id: str
    decision_id: str
    outcome: DecisionOutcome
    actual_results: Dict[str, Any]
    lessons_learned: List[str]
    feedback_score: float
    updated_at: datetime = field(default_factory=datetime.now)

class IntelligentDecisionEngine:
    def __init__(self):
        self.decision_requests: Dict[str, DecisionRequest] = {}
        self.decision_results: Dict[str, DecisionResult] = {}
        self.decision_history: Dict[str, DecisionHistory] = {}
        self.decision_templates: Dict[str, Dict[str, Any]] = {}
        self.decision_active = False
        self.decision_task: Optional[asyncio.Task] = None
        self.performance_metrics = {
            "decisions_made": 0,
            "successful_decisions": 0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0,
            "decision_accuracy": 0.0,
            "user_satisfaction": 0.0
        }

    async def start_intelligent_decision_engine(self):
        """Start the intelligent decision engine"""
        try:
            logger.info("Starting Intelligent Decision Engine...")
            
            # Initialize decision templates
            await self._initialize_decision_templates()
            
            # Start decision processing loop
            self.decision_active = True
            self.decision_task = asyncio.create_task(self._decision_processing_loop())
            
            logger.info("Intelligent Decision Engine started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Intelligent Decision Engine: {e}")
            return False

    async def stop_intelligent_decision_engine(self):
        """Stop the intelligent decision engine"""
        try:
            logger.info("Stopping Intelligent Decision Engine...")
            
            self.decision_active = False
            if self.decision_task:
                self.decision_task.cancel()
                try:
                    await self.decision_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Intelligent Decision Engine stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop Intelligent Decision Engine: {e}")
            return False

    async def _initialize_decision_templates(self):
        """Initialize decision templates"""
        try:
            # Financial decision template
            self.decision_templates["financial"] = {
                "criteria": [
                    {"name": "ROI", "weight": 0.3, "threshold": 0.15},
                    {"name": "Risk Level", "weight": 0.25, "threshold": 0.3},
                    {"name": "Liquidity", "weight": 0.2, "threshold": 0.7},
                    {"name": "Market Conditions", "weight": 0.15, "threshold": 0.6},
                    {"name": "Regulatory Compliance", "weight": 0.1, "threshold": 0.9}
                ],
                "constraints": {
                    "max_risk": 0.4,
                    "min_liquidity": 0.5,
                    "compliance_required": True
                }
            }
            
            # Operational decision template
            self.decision_templates["operational"] = {
                "criteria": [
                    {"name": "Efficiency", "weight": 0.3, "threshold": 0.8},
                    {"name": "Cost", "weight": 0.25, "threshold": 0.6},
                    {"name": "Quality", "weight": 0.2, "threshold": 0.85},
                    {"name": "Timeline", "weight": 0.15, "threshold": 0.7},
                    {"name": "Resource Availability", "weight": 0.1, "threshold": 0.8}
                ],
                "constraints": {
                    "max_cost": 100000,
                    "min_quality": 0.8,
                    "deadline_flexibility": 0.2
                }
            }
            
            # Strategic decision template
            self.decision_templates["strategic"] = {
                "criteria": [
                    {"name": "Strategic Alignment", "weight": 0.4, "threshold": 0.8},
                    {"name": "Market Opportunity", "weight": 0.25, "threshold": 0.7},
                    {"name": "Competitive Advantage", "weight": 0.2, "threshold": 0.6},
                    {"name": "Long-term Value", "weight": 0.15, "threshold": 0.75}
                ],
                "constraints": {
                    "strategic_fit": 0.8,
                    "market_size": 1000000,
                    "competitive_barrier": 0.6
                }
            }
            
            logger.info(f"Initialized {len(self.decision_templates)} decision templates")
            
        except Exception as e:
            logger.error(f"Failed to initialize decision templates: {e}")

    async def _decision_processing_loop(self):
        """Main decision processing loop"""
        while self.decision_active:
            try:
                # Process pending decision requests
                await self._process_pending_decisions()
                
                # Update decision history
                await self._update_decision_history()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                await asyncio.sleep(2)  # Process every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in decision processing loop: {e}")
                await asyncio.sleep(5)

    async def _process_pending_decisions(self):
        """Process pending decision requests"""
        try:
            pending_decisions = [req for req in self.decision_requests.values() if req.status == "pending"]
            
            for request in pending_decisions:
                # Check deadline
                if request.deadline and datetime.now() > request.deadline:
                    request.status = "overdue"
                    continue
                
                # Process decision
                await self._make_decision(request)
                
        except Exception as e:
            logger.error(f"Error processing pending decisions: {e}")

    async def _make_decision(self, request: DecisionRequest):
        """Make intelligent decision"""
        try:
            start_time = time.time()
            request.status = "processing"
            
            # Analyze decision complexity
            complexity_score = await self._analyze_decision_complexity(request)
            
            # Apply decision-making algorithm
            if complexity_score <= 2:
                result = await self._simple_decision_algorithm(request)
            elif complexity_score <= 4:
                result = await self._moderate_decision_algorithm(request)
            else:
                result = await self._complex_decision_algorithm(request)
            
            # Create decision result
            result_id = f"result_{secrets.token_hex(8)}"
            processing_time = time.time() - start_time
            
            decision_result = DecisionResult(
                result_id=result_id,
                request_id=request.request_id,
                recommended_option=result["recommended_option"],
                confidence_score=result["confidence_score"],
                reasoning=result["reasoning"],
                alternative_options=result["alternative_options"],
                risk_assessment=result["risk_assessment"],
                sensitivity_analysis=result["sensitivity_analysis"],
                implementation_plan=result["implementation_plan"],
                monitoring_metrics=result["monitoring_metrics"]
            )
            
            self.decision_results[result_id] = decision_result
            request.status = "completed"
            
            # Update metrics
            self.performance_metrics["decisions_made"] += 1
            self.performance_metrics["average_processing_time"] = (
                self.performance_metrics["average_processing_time"] + processing_time
            ) / 2
            self.performance_metrics["average_confidence"] = (
                self.performance_metrics["average_confidence"] + result["confidence_score"]
            ) / 2
            
            logger.info(f"Decision made for request {request.request_id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            request.status = "failed"

    async def _analyze_decision_complexity(self, request: DecisionRequest) -> float:
        """Analyze decision complexity"""
        try:
            complexity_factors = {
                "number_of_options": len(request.options),
                "number_of_criteria": len(request.criteria),
                "constraint_complexity": len(request.constraints),
                "context_complexity": {
                    DecisionContext.FINANCIAL: 3,
                    DecisionContext.OPERATIONAL: 2,
                    DecisionContext.STRATEGIC: 4,
                    DecisionContext.TACTICAL: 2,
                    DecisionContext.RISK_MANAGEMENT: 3,
                    DecisionContext.COMPLIANCE: 2,
                    DecisionContext.CUSTOMER: 2,
                    DecisionContext.MARKET: 3
                }.get(request.context, 2)
            }
            
            # Calculate complexity score
            complexity_score = (
                complexity_factors["number_of_options"] * 0.3 +
                complexity_factors["number_of_criteria"] * 0.2 +
                complexity_factors["constraint_complexity"] * 0.2 +
                complexity_factors["context_complexity"] * 0.3
            )
            
            return min(complexity_score, 5.0)  # Cap at 5
            
        except Exception as e:
            logger.error(f"Error analyzing decision complexity: {e}")
            return 2.0

    async def _simple_decision_algorithm(self, request: DecisionRequest) -> Dict[str, Any]:
        """Simple decision algorithm for low complexity decisions"""
        try:
            # Simple weighted scoring
            option_scores = {}
            
            for option in request.options:
                total_score = 0.0
                total_weight = 0.0
                
                for criteria in request.criteria:
                    if criteria.criteria_id in option.criteria_scores:
                        score = option.criteria_scores[criteria.criteria_id]
                        weight = criteria.weight
                        total_score += score * weight
                        total_weight += weight
                
                if total_weight > 0:
                    option_scores[option.option_id] = total_score / total_weight
                else:
                    option_scores[option.option_id] = 0.0
            
            # Find best option
            best_option_id = max(option_scores.keys(), key=lambda x: option_scores[x])
            best_option = next(opt for opt in request.options if opt.option_id == best_option_id)
            
            return {
                "recommended_option": best_option_id,
                "confidence_score": option_scores[best_option_id],
                "reasoning": f"Selected based on weighted criteria scoring. Score: {option_scores[best_option_id]:.2f}",
                "alternative_options": [
                    {"option_id": opt_id, "score": score} 
                    for opt_id, score in sorted(option_scores.items(), key=lambda x: x[1], reverse=True)[1:3]
                ],
                "risk_assessment": {"overall_risk": "low", "key_risks": best_option.risk_factors[:3]},
                "sensitivity_analysis": {"criteria_sensitivity": "low", "robustness": "high"},
                "implementation_plan": {"steps": ["plan", "execute", "monitor"], "timeline": "1-2 weeks"},
                "monitoring_metrics": ["success_rate", "cost_efficiency", "quality_score"]
            }
            
        except Exception as e:
            logger.error(f"Error in simple decision algorithm: {e}")
            return self._default_decision_result()

    async def _moderate_decision_algorithm(self, request: DecisionRequest) -> Dict[str, Any]:
        """Moderate decision algorithm for medium complexity decisions"""
        try:
            # Multi-criteria decision analysis with sensitivity analysis
            option_scores = {}
            sensitivity_data = {}
            
            for option in request.options:
                scores = []
                weights = []
                
                for criteria in request.criteria:
                    if criteria.criteria_id in option.criteria_scores:
                        score = option.criteria_scores[criteria.criteria_id]
                        weight = criteria.weight
                        scores.append(score)
                        weights.append(weight)
                
                # Calculate weighted score
                if weights:
                    weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                    option_scores[option.option_id] = weighted_score
                    
                    # Sensitivity analysis
                    sensitivity_data[option.option_id] = {
                        "score_variance": np.var(scores) if len(scores) > 1 else 0,
                        "weight_sensitivity": max(weights) - min(weights) if len(weights) > 1 else 0
                    }
            
            # Find best option
            best_option_id = max(option_scores.keys(), key=lambda x: option_scores[x])
            best_option = next(opt for opt in request.options if opt.option_id == best_option_id)
            
            return {
                "recommended_option": best_option_id,
                "confidence_score": option_scores[best_option_id] * 0.9,  # Slightly lower confidence for moderate complexity
                "reasoning": f"Selected using multi-criteria analysis with sensitivity analysis. Score: {option_scores[best_option_id]:.2f}",
                "alternative_options": [
                    {"option_id": opt_id, "score": score, "sensitivity": sensitivity_data.get(opt_id, {})} 
                    for opt_id, score in sorted(option_scores.items(), key=lambda x: x[1], reverse=True)[1:3]
                ],
                "risk_assessment": {
                    "overall_risk": "medium", 
                    "key_risks": best_option.risk_factors[:5],
                    "risk_score": best_option.probability_of_success
                },
                "sensitivity_analysis": {
                    "criteria_sensitivity": "medium",
                    "robustness": "medium",
                    "sensitivity_data": sensitivity_data.get(best_option_id, {})
                },
                "implementation_plan": {
                    "steps": ["analysis", "planning", "execution", "monitoring", "evaluation"],
                    "timeline": "2-4 weeks",
                    "resources": ["team", "budget", "tools"]
                },
                "monitoring_metrics": ["success_rate", "cost_efficiency", "quality_score", "timeline_adherence", "risk_mitigation"]
            }
            
        except Exception as e:
            logger.error(f"Error in moderate decision algorithm: {e}")
            return self._default_decision_result()

    async def _complex_decision_algorithm(self, request: DecisionRequest) -> Dict[str, Any]:
        """Complex decision algorithm for high complexity decisions"""
        try:
            # Advanced multi-criteria decision analysis with uncertainty handling
            option_scores = {}
            uncertainty_analysis = {}
            
            for option in request.options:
                # Monte Carlo simulation for uncertainty
                scores = []
                for _ in range(100):  # 100 iterations
                    iteration_score = 0.0
                    total_weight = 0.0
                    
                    for criteria in request.criteria:
                        if criteria.criteria_id in option.criteria_scores:
                            # Add uncertainty to scores
                            base_score = option.criteria_scores[criteria.criteria_id]
                            uncertainty = secrets.randbelow(20) / 100.0 - 0.1  # ±10% uncertainty
                            score = max(0, min(1, base_score + uncertainty))
                            weight = criteria.weight
                            
                            iteration_score += score * weight
                            total_weight += weight
                    
                    if total_weight > 0:
                        scores.append(iteration_score / total_weight)
                
                if scores:
                    option_scores[option.option_id] = np.mean(scores)
                    uncertainty_analysis[option.option_id] = {
                        "mean_score": np.mean(scores),
                        "std_deviation": np.std(scores),
                        "confidence_interval": [np.percentile(scores, 25), np.percentile(scores, 75)]
                    }
            
            # Find best option
            best_option_id = max(option_scores.keys(), key=lambda x: option_scores[x])
            best_option = next(opt for opt in request.options if opt.option_id == best_option_id)
            
            return {
                "recommended_option": best_option_id,
                "confidence_score": option_scores[best_option_id] * 0.8,  # Lower confidence for complex decisions
                "reasoning": f"Selected using advanced multi-criteria analysis with uncertainty handling. Score: {option_scores[best_option_id]:.2f} ± {uncertainty_analysis[best_option_id]['std_deviation']:.2f}",
                "alternative_options": [
                    {
                        "option_id": opt_id, 
                        "score": score, 
                        "uncertainty": uncertainty_analysis.get(opt_id, {})
                    } 
                    for opt_id, score in sorted(option_scores.items(), key=lambda x: x[1], reverse=True)[1:3]
                ],
                "risk_assessment": {
                    "overall_risk": "high", 
                    "key_risks": best_option.risk_factors,
                    "risk_score": best_option.probability_of_success,
                    "uncertainty_factors": ["market_volatility", "regulatory_changes", "technology_risks"]
                },
                "sensitivity_analysis": {
                    "criteria_sensitivity": "high",
                    "robustness": "medium",
                    "uncertainty_analysis": uncertainty_analysis.get(best_option_id, {}),
                    "scenario_analysis": ["best_case", "base_case", "worst_case"]
                },
                "implementation_plan": {
                    "steps": ["research", "analysis", "planning", "pilot", "execution", "monitoring", "evaluation", "optimization"],
                    "timeline": "1-3 months",
                    "resources": ["expert_team", "significant_budget", "advanced_tools", "external_consultants"],
                    "milestones": ["analysis_complete", "plan_approved", "pilot_launched", "full_implementation"]
                },
                "monitoring_metrics": [
                    "success_rate", "cost_efficiency", "quality_score", "timeline_adherence", 
                    "risk_mitigation", "stakeholder_satisfaction", "ROI", "learning_curve"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in complex decision algorithm: {e}")
            return self._default_decision_result()

    def _default_decision_result(self) -> Dict[str, Any]:
        """Default decision result for error cases"""
        return {
            "recommended_option": "default_option",
            "confidence_score": 0.5,
            "reasoning": "Default decision due to processing error",
            "alternative_options": [],
            "risk_assessment": {"overall_risk": "unknown", "key_risks": []},
            "sensitivity_analysis": {"criteria_sensitivity": "unknown", "robustness": "unknown"},
            "implementation_plan": {"steps": ["review", "decide"], "timeline": "1 week"},
            "monitoring_metrics": ["basic_metrics"]
        }

    async def _update_decision_history(self):
        """Update decision history with outcomes"""
        try:
            # Simulate decision outcome updates
            for result in self.decision_results.values():
                if result.result_id not in self.decision_history:
                    # Simulate outcome after some time
                    if (datetime.now() - result.created_at).days > 7:
                        outcome = secrets.choice(list(DecisionOutcome))
                        feedback_score = secrets.randbelow(100) / 100.0
                        
                        history = DecisionHistory(
                            history_id=f"history_{secrets.token_hex(8)}",
                            decision_id=result.result_id,
                            outcome=outcome,
                            actual_results={"performance": feedback_score, "satisfaction": feedback_score},
                            lessons_learned=[f"Lesson learned from {result.recommended_option}"],
                            feedback_score=feedback_score
                        )
                        
                        self.decision_history[result.result_id] = history
                        
        except Exception as e:
            logger.error(f"Error updating decision history: {e}")

    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate decision accuracy
            if self.decision_history:
                successful_decisions = len([h for h in self.decision_history.values() 
                                          if h.outcome in [DecisionOutcome.SUCCESS, DecisionOutcome.PARTIAL_SUCCESS]])
                self.performance_metrics["successful_decisions"] = successful_decisions
                self.performance_metrics["decision_accuracy"] = successful_decisions / len(self.decision_history)
                
                # Calculate user satisfaction
                if self.decision_history:
                    total_satisfaction = sum(h.feedback_score for h in self.decision_history.values())
                    self.performance_metrics["user_satisfaction"] = total_satisfaction / len(self.decision_history)
                    
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    async def submit_decision_request(self, context: DecisionContext, description: str, 
                                    options: List[Dict[str, Any]], criteria: List[Dict[str, Any]] = None,
                                    constraints: Dict[str, Any] = None, deadline: Optional[datetime] = None,
                                    priority: int = 1) -> str:
        """Submit decision request"""
        try:
            request_id = f"decision_{secrets.token_hex(8)}"
            
            # Create decision criteria
            decision_criteria = []
            if criteria:
                for i, crit in enumerate(criteria):
                    criteria_obj = DecisionCriteria(
                        criteria_id=f"criteria_{i}",
                        name=crit.get("name", f"Criteria {i}"),
                        weight=crit.get("weight", 1.0),
                        threshold=crit.get("threshold", 0.5),
                        importance=crit.get("importance", "medium"),
                        description=crit.get("description", "")
                    )
                    decision_criteria.append(criteria_obj)
            else:
                # Use default criteria from template
                template = self.decision_templates.get(context.value, {})
                for i, crit in enumerate(template.get("criteria", [])):
                    criteria_obj = DecisionCriteria(
                        criteria_id=f"criteria_{i}",
                        name=crit["name"],
                        weight=crit["weight"],
                        threshold=crit["threshold"],
                        importance="medium",
                        description=f"Default criteria for {crit['name']}"
                    )
                    decision_criteria.append(criteria_obj)
            
            # Create decision options
            decision_options = []
            for i, opt in enumerate(options):
                option_obj = DecisionOption(
                    option_id=f"option_{i}",
                    name=opt.get("name", f"Option {i}"),
                    description=opt.get("description", ""),
                    criteria_scores=opt.get("criteria_scores", {}),
                    risk_factors=opt.get("risk_factors", []),
                    benefits=opt.get("benefits", []),
                    costs=opt.get("costs", []),
                    probability_of_success=opt.get("probability_of_success", 0.5),
                    expected_value=opt.get("expected_value", 0.0)
                )
                decision_options.append(option_obj)
            
            # Determine complexity
            complexity = DecisionComplexity.SIMPLE
            if len(decision_options) > 5 or len(decision_criteria) > 7:
                complexity = DecisionComplexity.COMPLEX
            elif len(decision_options) > 3 or len(decision_criteria) > 5:
                complexity = DecisionComplexity.MODERATE
            
            request = DecisionRequest(
                request_id=request_id,
                context=context,
                complexity=complexity,
                description=description,
                criteria=decision_criteria,
                options=decision_options,
                constraints=constraints or {},
                deadline=deadline,
                priority=priority
            )
            
            self.decision_requests[request_id] = request
            
            logger.info(f"Decision request submitted: {request_id}")
            return request_id
            
        except Exception as e:
            logger.error(f"Error submitting decision request: {e}")
            return ""

    async def get_decision_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get decision result"""
        try:
            # Find result by request ID
            for result in self.decision_results.values():
                if result.request_id == request_id:
                    return {
                        "result_id": result.result_id,
                        "request_id": result.request_id,
                        "recommended_option": result.recommended_option,
                        "confidence_score": result.confidence_score,
                        "reasoning": result.reasoning,
                        "alternative_options": result.alternative_options,
                        "risk_assessment": result.risk_assessment,
                        "sensitivity_analysis": result.sensitivity_analysis,
                        "implementation_plan": result.implementation_plan,
                        "monitoring_metrics": result.monitoring_metrics,
                        "created_at": result.created_at.isoformat()
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting decision result: {e}")
            return None

    async def get_decision_performance_metrics(self) -> Dict[str, Any]:
        """Get decision engine performance metrics"""
        try:
            return {
                "performance_metrics": self.performance_metrics,
                "total_requests": len(self.decision_requests),
                "completed_requests": len([r for r in self.decision_requests.values() if r.status == "completed"]),
                "total_results": len(self.decision_results),
                "total_history": len(self.decision_history),
                "decision_templates": len(self.decision_templates)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

# Global instance
intelligent_decision_engine = IntelligentDecisionEngine()
