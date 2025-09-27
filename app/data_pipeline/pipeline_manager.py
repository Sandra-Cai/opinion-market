"""
Pipeline Manager for Data Processing
Orchestrates ETL pipelines with monitoring and error handling
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from app.core.enhanced_cache import enhanced_cache
from app.core.database import engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineStep:
    """Individual pipeline step configuration"""
    name: str
    step_type: str  # "extract", "transform", "load", "validate"
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout: int = 300  # 5 minutes
    enabled: bool = True


@dataclass
class PipelineExecution:
    """Pipeline execution tracking"""
    pipeline_id: str
    pipeline_name: str
    status: PipelineStatus
    start_time: float
    end_time: Optional[float] = None
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class PipelineManager:
    """Manages data processing pipelines"""
    
    def __init__(self):
        self.pipelines: Dict[str, List[PipelineStep]] = {}
        self.executions: Dict[str, PipelineExecution] = {}
        self.step_handlers: Dict[str, Callable] = {}
        self.running_pipelines: Dict[str, asyncio.Task] = {}
        
        # Pipeline configuration
        self.max_concurrent_pipelines = 5
        self.default_timeout = 3600  # 1 hour
        self.retry_delay = 60  # 1 minute
        
        # Register default step handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default step handlers"""
        self.step_handlers = {
            "extract": self._extract_step,
            "transform": self._transform_step,
            "load": self._load_step,
            "validate": self._validate_step,
            "custom": self._custom_step
        }
    
    def register_pipeline(self, pipeline_name: str, steps: List[PipelineStep]) -> bool:
        """Register a new pipeline"""
        try:
            # Validate pipeline
            if not self._validate_pipeline(steps):
                return False
            
            self.pipelines[pipeline_name] = steps
            logger.info(f"Registered pipeline: {pipeline_name} with {len(steps)} steps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register pipeline {pipeline_name}: {e}")
            return False
    
    def register_step_handler(self, step_type: str, handler: Callable):
        """Register a custom step handler"""
        self.step_handlers[step_type] = handler
        logger.info(f"Registered step handler for type: {step_type}")
    
    async def execute_pipeline(self, pipeline_name: str, 
                             config: Optional[Dict[str, Any]] = None) -> str:
        """Execute a pipeline"""
        try:
            if pipeline_name not in self.pipelines:
                raise ValueError(f"Pipeline {pipeline_name} not found")
            
            # Check concurrent pipeline limit
            if len(self.running_pipelines) >= self.max_concurrent_pipelines:
                raise Exception("Maximum concurrent pipelines reached")
            
            # Create execution tracking
            pipeline_id = f"{pipeline_name}_{int(time.time())}"
            execution = PipelineExecution(
                pipeline_id=pipeline_id,
                pipeline_name=pipeline_name,
                status=PipelineStatus.PENDING,
                start_time=time.time()
            )
            
            self.executions[pipeline_id] = execution
            
            # Start pipeline execution
            task = asyncio.create_task(
                self._execute_pipeline_async(pipeline_id, config)
            )
            self.running_pipelines[pipeline_id] = task
            
            logger.info(f"Started pipeline execution: {pipeline_id}")
            return pipeline_id
            
        except Exception as e:
            logger.error(f"Failed to execute pipeline {pipeline_name}: {e}")
            raise e
    
    async def _execute_pipeline_async(self, pipeline_id: str, 
                                    config: Optional[Dict[str, Any]] = None):
        """Execute pipeline asynchronously"""
        execution = self.executions[pipeline_id]
        pipeline_name = execution.pipeline_name
        steps = self.pipelines[pipeline_name]
        
        try:
            execution.status = PipelineStatus.RUNNING
            
            # Execute steps in order
            for step in steps:
                if not step.enabled:
                    continue
                
                # Check dependencies
                if not self._check_dependencies(step, execution.steps_completed):
                    execution.steps_failed.append(step.name)
                    execution.error_message = f"Dependency check failed for step: {step.name}"
                    execution.status = PipelineStatus.FAILED
                    return
                
                # Execute step with retry logic
                step_success = await self._execute_step_with_retry(step, config)
                
                if step_success:
                    execution.steps_completed.append(step.name)
                    logger.info(f"Completed step: {step.name} in pipeline {pipeline_id}")
                else:
                    execution.steps_failed.append(step.name)
                    execution.error_message = f"Step failed: {step.name}"
                    execution.status = PipelineStatus.FAILED
                    return
            
            # Pipeline completed successfully
            execution.status = PipelineStatus.COMPLETED
            execution.end_time = time.time()
            
            # Calculate metrics
            execution.metrics = {
                "total_duration": execution.end_time - execution.start_time,
                "steps_completed": len(execution.steps_completed),
                "steps_failed": len(execution.steps_failed),
                "success_rate": len(execution.steps_completed) / len(steps)
            }
            
            logger.info(f"Pipeline {pipeline_id} completed successfully")
            
        except Exception as e:
            execution.status = PipelineStatus.FAILED
            execution.error_message = str(e)
            execution.end_time = time.time()
            logger.error(f"Pipeline {pipeline_id} failed: {e}")
        
        finally:
            # Clean up
            if pipeline_id in self.running_pipelines:
                del self.running_pipelines[pipeline_id]
    
    async def _execute_step_with_retry(self, step: PipelineStep, 
                                     config: Optional[Dict[str, Any]] = None) -> bool:
        """Execute step with retry logic"""
        for attempt in range(step.retry_count):
            try:
                # Get step handler
                handler = self.step_handlers.get(step.step_type)
                if not handler:
                    logger.error(f"No handler found for step type: {step.step_type}")
                    return False
                
                # Execute step with timeout
                result = await asyncio.wait_for(
                    handler(step, config),
                    timeout=step.timeout
                )
                
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Step {step.name} timed out (attempt {attempt + 1})")
                if attempt < step.retry_count - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    return False
                    
            except Exception as e:
                logger.warning(f"Step {step.name} failed (attempt {attempt + 1}): {e}")
                if attempt < step.retry_count - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    return False
        
        return False
    
    def _check_dependencies(self, step: PipelineStep, completed_steps: List[str]) -> bool:
        """Check if step dependencies are satisfied"""
        for dependency in step.dependencies:
            if dependency not in completed_steps:
                return False
        return True
    
    def _validate_pipeline(self, steps: List[PipelineStep]) -> bool:
        """Validate pipeline configuration"""
        if not steps:
            return False
        
        # Check for duplicate step names
        step_names = [step.name for step in steps]
        if len(step_names) != len(set(step_names)):
            return False
        
        # Check dependencies
        for step in steps:
            for dependency in step.dependencies:
                if dependency not in step_names:
                    return False
        
        return True
    
    # Step handlers
    async def _extract_step(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Handle extract step"""
        try:
            # Extract data based on step configuration
            source_type = step.config.get("source_type", "database")
            
            if source_type == "database":
                return await self._extract_from_database(step, config)
            elif source_type == "api":
                return await self._extract_from_api(step, config)
            elif source_type == "file":
                return await self._extract_from_file(step, config)
            else:
                logger.error(f"Unknown source type: {source_type}")
                return False
                
        except Exception as e:
            logger.error(f"Extract step failed: {e}")
            return False
    
    async def _transform_step(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Handle transform step"""
        try:
            # Transform data based on step configuration
            transform_type = step.config.get("transform_type", "basic")
            
            if transform_type == "basic":
                return await self._basic_transform(step, config)
            elif transform_type == "aggregation":
                return await self._aggregation_transform(step, config)
            elif transform_type == "enrichment":
                return await self._enrichment_transform(step, config)
            else:
                logger.error(f"Unknown transform type: {transform_type}")
                return False
                
        except Exception as e:
            logger.error(f"Transform step failed: {e}")
            return False
    
    async def _load_step(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Handle load step"""
        try:
            # Load data based on step configuration
            target_type = step.config.get("target_type", "database")
            
            if target_type == "database":
                return await self._load_to_database(step, config)
            elif target_type == "cache":
                return await self._load_to_cache(step, config)
            elif target_type == "file":
                return await self._load_to_file(step, config)
            else:
                logger.error(f"Unknown target type: {target_type}")
                return False
                
        except Exception as e:
            logger.error(f"Load step failed: {e}")
            return False
    
    async def _validate_step(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Handle validate step"""
        try:
            # Validate data based on step configuration
            validation_type = step.config.get("validation_type", "basic")
            
            if validation_type == "basic":
                return await self._basic_validation(step, config)
            elif validation_type == "schema":
                return await self._schema_validation(step, config)
            elif validation_type == "business_rules":
                return await self._business_rules_validation(step, config)
            else:
                logger.error(f"Unknown validation type: {validation_type}")
                return False
                
        except Exception as e:
            logger.error(f"Validation step failed: {e}")
            return False
    
    async def _custom_step(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Handle custom step"""
        try:
            # Execute custom logic
            custom_function = step.config.get("function")
            if not custom_function:
                logger.error("No custom function specified")
                return False
            
            # Execute custom function
            result = await custom_function(step, config)
            return result
            
        except Exception as e:
            logger.error(f"Custom step failed: {e}")
            return False
    
    # Extract implementations
    async def _extract_from_database(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Extract data from database"""
        try:
            query = step.config.get("query")
            if not query:
                return False
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                data = [dict(row) for row in result]
            
            # Store extracted data
            await enhanced_cache.set(
                f"pipeline_data_{step.name}",
                data,
                ttl=3600,
                tags=["pipeline", "extracted"]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Database extraction failed: {e}")
            return False
    
    async def _extract_from_api(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Extract data from API"""
        try:
            url = step.config.get("url")
            if not url:
                return False
            
            # Mock API extraction (in real implementation, use HTTP client)
            data = {"extracted": "data", "source": "api", "timestamp": time.time()}
            
            # Store extracted data
            await enhanced_cache.set(
                f"pipeline_data_{step.name}",
                data,
                ttl=3600,
                tags=["pipeline", "extracted"]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"API extraction failed: {e}")
            return False
    
    async def _extract_from_file(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Extract data from file"""
        try:
            file_path = step.config.get("file_path")
            if not file_path:
                return False
            
            # Mock file extraction (in real implementation, read actual file)
            data = {"extracted": "data", "source": "file", "timestamp": time.time()}
            
            # Store extracted data
            await enhanced_cache.set(
                f"pipeline_data_{step.name}",
                data,
                ttl=3600,
                tags=["pipeline", "extracted"]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"File extraction failed: {e}")
            return False
    
    # Transform implementations
    async def _basic_transform(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Basic data transformation"""
        try:
            # Get input data
            input_data = await enhanced_cache.get(f"pipeline_data_{step.name}")
            if not input_data:
                return False
            
            # Apply basic transformations
            transformed_data = self._apply_basic_transforms(input_data, step.config)
            
            # Store transformed data
            await enhanced_cache.set(
                f"pipeline_data_{step.name}_transformed",
                transformed_data,
                ttl=3600,
                tags=["pipeline", "transformed"]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Basic transform failed: {e}")
            return False
    
    def _apply_basic_transforms(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply basic transformations to data"""
        # Apply field mappings
        field_mappings = config.get("field_mappings", {})
        if field_mappings and isinstance(data, list):
            for item in data:
                for old_field, new_field in field_mappings.items():
                    if old_field in item:
                        item[new_field] = item.pop(old_field)
        
        # Apply data type conversions
        type_conversions = config.get("type_conversions", {})
        if type_conversions and isinstance(data, list):
            for item in data:
                for field, target_type in type_conversions.items():
                    if field in item:
                        try:
                            if target_type == "int":
                                item[field] = int(item[field])
                            elif target_type == "float":
                                item[field] = float(item[field])
                            elif target_type == "str":
                                item[field] = str(item[field])
                        except (ValueError, TypeError):
                            pass
        
        return data
    
    async def _aggregation_transform(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Aggregation transformation"""
        try:
            # Get input data
            input_data = await enhanced_cache.get(f"pipeline_data_{step.name}")
            if not input_data:
                return False
            
            # Apply aggregations
            aggregation_config = config.get("aggregations", {})
            aggregated_data = self._apply_aggregations(input_data, aggregation_config)
            
            # Store aggregated data
            await enhanced_cache.set(
                f"pipeline_data_{step.name}_aggregated",
                aggregated_data,
                ttl=3600,
                tags=["pipeline", "aggregated"]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Aggregation transform failed: {e}")
            return False
    
    def _apply_aggregations(self, data: List[Dict[str, Any]], 
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply aggregations to data"""
        if not isinstance(data, list) or not data:
            return {}
        
        result = {}
        
        # Group by field
        group_by = config.get("group_by")
        if group_by:
            groups = {}
            for item in data:
                key = item.get(group_by, "unknown")
                if key not in groups:
                    groups[key] = []
                groups[key].append(item)
            
            # Apply aggregations to each group
            for group_key, group_data in groups.items():
                result[group_key] = self._aggregate_group(group_data, config)
        else:
            # Apply aggregations to entire dataset
            result = self._aggregate_group(data, config)
        
        return result
    
    def _aggregate_group(self, data: List[Dict[str, Any]], 
                        config: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate a group of data"""
        if not data:
            return {}
        
        result = {}
        
        # Count
        if config.get("count", False):
            result["count"] = len(data)
        
        # Sum
        sum_fields = config.get("sum", [])
        for field in sum_fields:
            values = [item.get(field, 0) for item in data if isinstance(item.get(field), (int, float))]
            result[f"sum_{field}"] = sum(values)
        
        # Average
        avg_fields = config.get("average", [])
        for field in avg_fields:
            values = [item.get(field, 0) for item in data if isinstance(item.get(field), (int, float))]
            if values:
                result[f"avg_{field}"] = sum(values) / len(values)
        
        # Min/Max
        min_max_fields = config.get("min_max", [])
        for field in min_max_fields:
            values = [item.get(field) for item in data if item.get(field) is not None]
            if values:
                result[f"min_{field}"] = min(values)
                result[f"max_{field}"] = max(values)
        
        return result
    
    async def _enrichment_transform(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Data enrichment transformation"""
        try:
            # Get input data
            input_data = await enhanced_cache.get(f"pipeline_data_{step.name}")
            if not input_data:
                return False
            
            # Apply enrichment
            enrichment_config = config.get("enrichment", {})
            enriched_data = self._apply_enrichment(input_data, enrichment_config)
            
            # Store enriched data
            await enhanced_cache.set(
                f"pipeline_data_{step.name}_enriched",
                enriched_data,
                ttl=3600,
                tags=["pipeline", "enriched"]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Enrichment transform failed: {e}")
            return False
    
    def _apply_enrichment(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply data enrichment"""
        if not isinstance(data, list):
            return data
        
        # Add computed fields
        computed_fields = config.get("computed_fields", {})
        for item in data:
            for field_name, field_config in computed_fields.items():
                if field_config.get("type") == "formula":
                    formula = field_config.get("formula", "")
                    # Simple formula evaluation (in real implementation, use proper expression evaluator)
                    try:
                        # Replace field references with actual values
                        eval_formula = formula
                        for field in item:
                            eval_formula = eval_formula.replace(f"{{{field}}}", str(item[field]))
                        
                        # Evaluate formula (simplified - in real implementation, use safe evaluator)
                        item[field_name] = eval(eval_formula)
                    except:
                        item[field_name] = None
        
        # Add lookup data
        lookups = config.get("lookups", {})
        for item in data:
            for lookup_field, lookup_config in lookups.items():
                lookup_key = item.get(lookup_config.get("key_field"))
                if lookup_key:
                    # Mock lookup (in real implementation, query lookup table)
                    item[lookup_field] = f"lookup_value_for_{lookup_key}"
        
        return data
    
    # Load implementations
    async def _load_to_database(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load data to database"""
        try:
            # Get transformed data
            data = await enhanced_cache.get(f"pipeline_data_{step.name}_transformed")
            if not data:
                return False
            
            table_name = step.config.get("table_name")
            if not table_name:
                return False
            
            # Load data to database (simplified - in real implementation, use proper ORM)
            with engine.connect() as conn:
                # Mock database load
                conn.execute(text(f"INSERT INTO {table_name} VALUES (:data)"), {"data": json.dumps(data)})
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Database load failed: {e}")
            return False
    
    async def _load_to_cache(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load data to cache"""
        try:
            # Get transformed data
            data = await enhanced_cache.get(f"pipeline_data_{step.name}_transformed")
            if not data:
                return False
            
            cache_key = step.config.get("cache_key", f"pipeline_result_{step.name}")
            ttl = step.config.get("ttl", 3600)
            
            # Load data to cache
            await enhanced_cache.set(
                cache_key,
                data,
                ttl=ttl,
                tags=["pipeline", "result"]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Cache load failed: {e}")
            return False
    
    async def _load_to_file(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load data to file"""
        try:
            # Get transformed data
            data = await enhanced_cache.get(f"pipeline_data_{step.name}_transformed")
            if not data:
                return False
            
            file_path = step.config.get("file_path")
            if not file_path:
                return False
            
            # Mock file load (in real implementation, write to actual file)
            logger.info(f"Loading data to file: {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"File load failed: {e}")
            return False
    
    # Validation implementations
    async def _basic_validation(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Basic data validation"""
        try:
            # Get data to validate
            data = await enhanced_cache.get(f"pipeline_data_{step.name}_transformed")
            if not data:
                return False
            
            # Apply basic validations
            validation_rules = config.get("validation_rules", {})
            is_valid = self._apply_basic_validations(data, validation_rules)
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Basic validation failed: {e}")
            return False
    
    def _apply_basic_validations(self, data: Any, rules: Dict[str, Any]) -> bool:
        """Apply basic validation rules"""
        if not isinstance(data, list):
            return True
        
        for item in data:
            # Required fields
            required_fields = rules.get("required_fields", [])
            for field in required_fields:
                if field not in item or item[field] is None:
                    return False
            
            # Data type validation
            type_validation = rules.get("type_validation", {})
            for field, expected_type in type_validation.items():
                if field in item:
                    if expected_type == "int" and not isinstance(item[field], int):
                        return False
                    elif expected_type == "float" and not isinstance(item[field], (int, float)):
                        return False
                    elif expected_type == "str" and not isinstance(item[field], str):
                        return False
            
            # Range validation
            range_validation = rules.get("range_validation", {})
            for field, range_config in range_validation.items():
                if field in item and isinstance(item[field], (int, float)):
                    min_val = range_config.get("min")
                    max_val = range_config.get("max")
                    if min_val is not None and item[field] < min_val:
                        return False
                    if max_val is not None and item[field] > max_val:
                        return False
        
        return True
    
    async def _schema_validation(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Schema validation"""
        try:
            # Get data to validate
            data = await enhanced_cache.get(f"pipeline_data_{step.name}_transformed")
            if not data:
                return False
            
            # Apply schema validation
            schema = config.get("schema", {})
            is_valid = self._apply_schema_validation(data, schema)
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
    
    def _apply_schema_validation(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Apply schema validation"""
        if not isinstance(data, list):
            return True
        
        for item in data:
            # Check required fields
            required_fields = schema.get("required", [])
            for field in required_fields:
                if field not in item:
                    return False
            
            # Check field types
            properties = schema.get("properties", {})
            for field, field_schema in properties.items():
                if field in item:
                    expected_type = field_schema.get("type")
                    if expected_type and not self._validate_field_type(item[field], expected_type):
                        return False
        
        return True
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field type"""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        else:
            return True
    
    async def _business_rules_validation(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Business rules validation"""
        try:
            # Get data to validate
            data = await enhanced_cache.get(f"pipeline_data_{step.name}_transformed")
            if not data:
                return False
            
            # Apply business rules
            business_rules = config.get("business_rules", [])
            is_valid = self._apply_business_rules(data, business_rules)
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Business rules validation failed: {e}")
            return False
    
    def _apply_business_rules(self, data: Any, rules: List[Dict[str, Any]]) -> bool:
        """Apply business rules validation"""
        if not isinstance(data, list):
            return True
        
        for rule in rules:
            rule_type = rule.get("type")
            
            if rule_type == "uniqueness":
                field = rule.get("field")
                if field:
                    values = [item.get(field) for item in data if field in item]
                    if len(values) != len(set(values)):
                        return False
            
            elif rule_type == "conditional":
                condition_field = rule.get("condition_field")
                condition_value = rule.get("condition_value")
                required_field = rule.get("required_field")
                
                for item in data:
                    if item.get(condition_field) == condition_value:
                        if required_field not in item or item[required_field] is None:
                            return False
            
            elif rule_type == "cross_field":
                field1 = rule.get("field1")
                field2 = rule.get("field2")
                operator = rule.get("operator", "==")
                
                for item in data:
                    if field1 in item and field2 in item:
                        val1 = item[field1]
                        val2 = item[field2]
                        
                        if operator == "==" and val1 != val2:
                            return False
                        elif operator == "!=" and val1 == val2:
                            return False
                        elif operator == ">" and val1 <= val2:
                            return False
                        elif operator == "<" and val1 >= val2:
                            return False
        
        return True
    
    # Pipeline management methods
    async def get_pipeline_status(self, pipeline_id: str) -> Optional[PipelineExecution]:
        """Get pipeline execution status"""
        return self.executions.get(pipeline_id)
    
    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel a running pipeline"""
        try:
            if pipeline_id in self.running_pipelines:
                task = self.running_pipelines[pipeline_id]
                task.cancel()
                
                # Update execution status
                if pipeline_id in self.executions:
                    self.executions[pipeline_id].status = PipelineStatus.CANCELLED
                    self.executions[pipeline_id].end_time = time.time()
                
                del self.running_pipelines[pipeline_id]
                logger.info(f"Cancelled pipeline: {pipeline_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel pipeline {pipeline_id}: {e}")
            return False
    
    async def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline execution metrics"""
        total_executions = len(self.executions)
        completed_executions = len([e for e in self.executions.values() if e.status == PipelineStatus.COMPLETED])
        failed_executions = len([e for e in self.executions.values() if e.status == PipelineStatus.FAILED])
        running_executions = len(self.running_pipelines)
        
        # Calculate average execution time
        completed_times = [
            e.end_time - e.start_time 
            for e in self.executions.values() 
            if e.status == PipelineStatus.COMPLETED and e.end_time
        ]
        avg_execution_time = sum(completed_times) / len(completed_times) if completed_times else 0
        
        return {
            "total_executions": total_executions,
            "completed_executions": completed_executions,
            "failed_executions": failed_executions,
            "running_executions": running_executions,
            "success_rate": completed_executions / total_executions if total_executions > 0 else 0,
            "average_execution_time": avg_execution_time,
            "registered_pipelines": len(self.pipelines)
        }
    
    async def cleanup_old_executions(self, max_age_hours: int = 24):
        """Clean up old pipeline executions"""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            old_executions = [
                pipeline_id for pipeline_id, execution in self.executions.items()
                if execution.start_time < cutoff_time
            ]
            
            for pipeline_id in old_executions:
                del self.executions[pipeline_id]
            
            logger.info(f"Cleaned up {len(old_executions)} old pipeline executions")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old executions: {e}")


# Global pipeline manager instance
pipeline_manager = PipelineManager()
