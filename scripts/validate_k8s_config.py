#!/usr/bin/env python3
"""
Kubernetes Configuration Validator
Validates Kubernetes manifests for best practices, security, and compliance
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ValidationResult:
    """Result of a validation check"""
    rule_name: str
    severity: str  # error, warning, info
    message: str
    file_path: str
    line_number: Optional[int] = None
    resource_name: Optional[str] = None
    resource_type: Optional[str] = None

class KubernetesValidator:
    """Validates Kubernetes configuration files"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.severity_colors = {
            'error': '\033[91m',    # Red
            'warning': '\033[93m',  # Yellow
            'info': '\033[94m',     # Blue
            'success': '\033[92m'   # Green
        }
        self.reset_color = '\033[0m'
    
    def validate_file(self, file_path: str) -> List[ValidationResult]:
        """Validate a single Kubernetes manifest file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse YAML
            documents = list(yaml.safe_load_all(content))
            
            for doc in documents:
                if doc and isinstance(doc, dict):
                    self._validate_resource(doc, file_path)
            
            return self.results
            
        except yaml.YAMLError as e:
            self.results.append(ValidationResult(
                rule_name="yaml_syntax",
                severity="error",
                message=f"YAML syntax error: {str(e)}",
                file_path=file_path
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                rule_name="file_parsing",
                severity="error",
                message=f"Error parsing file: {str(e)}",
                file_path=file_path
            ))
        
        return self.results
    
    def _validate_resource(self, resource: Dict[str, Any], file_path: str):
        """Validate a single Kubernetes resource"""
        if 'kind' not in resource or 'metadata' not in resource:
            return
        
        kind = resource['kind']
        metadata = resource['metadata']
        name = metadata.get('name', 'unknown')
        
        # Run all validation rules
        self._check_required_fields(resource, file_path, kind, name)
        self._check_security_context(resource, file_path, kind, name)
        self._check_resource_limits(resource, file_path, kind, name)
        self._check_health_checks(resource, file_path, kind, name)
        self._check_network_policies(resource, file_path, kind, name)
        self._check_image_security(resource, file_path, kind, name)
        self._check_secrets_management(resource, file_path, kind, name)
        self._check_pod_security(resource, file_path, kind, name)
        self._check_service_configuration(resource, file_path, kind, name)
        self._check_ingress_security(resource, file_path, kind, name)
        self._check_pod_disruption_budget(resource, file_path, kind, name)
        self._check_horizontal_pod_autoscaler(resource, file_path, kind, name)
        self._check_volume_security(resource, file_path, kind, name)
        self._check_environment_variables(resource, file_path, kind, name)
        self._check_image_pull_policy(resource, file_path, kind, name)
        self._check_service_account(resource, file_path, kind, name)
        self._check_affinity_rules(resource, file_path, kind, name)
        self._check_termination_grace_period(resource, file_path, kind, name)
    
    def _check_required_fields(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check for required fields"""
        required_fields = {
            'Deployment': ['spec.template.spec.containers'],
            'Service': ['spec.ports'],
            'ConfigMap': ['data'],
            'Secret': ['data', 'stringData'],
            'Ingress': ['spec.rules'],
            'NetworkPolicy': ['spec.podSelector']
        }
        
        if kind in required_fields:
            for field in required_fields[kind]:
                if not self._get_nested_value(resource, field):
                    self.results.append(ValidationResult(
                        rule_name="required_fields",
                        severity="error",
                        message=f"Missing required field: {field}",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
    
    def _check_security_context(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check security context configuration"""
        if kind in ['Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob']:
            containers = self._get_containers(resource)
            
            for i, container in enumerate(containers):
                # Check if running as root
                security_context = container.get('securityContext', {})
                run_as_user = security_context.get('runAsUser')
                
                if run_as_user is None or run_as_user == 0:
                    self.results.append(ValidationResult(
                        rule_name="run_as_root",
                        severity="warning",
                        message=f"Container '{container.get('name', f'container-{i}')}' may be running as root",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
                
                # Check privileged mode
                if security_context.get('privileged', False):
                    self.results.append(ValidationResult(
                        rule_name="privileged_container",
                        severity="error",
                        message=f"Container '{container.get('name', f'container-{i}')}' is running in privileged mode",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
                
                # Check read-only root filesystem
                if not security_context.get('readOnlyRootFilesystem', False):
                    self.results.append(ValidationResult(
                        rule_name="read_only_rootfs",
                        severity="warning",
                        message=f"Container '{container.get('name', f'container-{i}')}' should use read-only root filesystem",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
    
    def _check_resource_limits(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check resource limits and requests"""
        if kind in ['Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob']:
            containers = self._get_containers(resource)
            
            for i, container in enumerate(containers):
                resources = container.get('resources', {})
                limits = resources.get('limits', {})
                requests = resources.get('requests', {})
                
                # Check if limits are set
                if not limits:
                    self.results.append(ValidationResult(
                        rule_name="missing_limits",
                        severity="warning",
                        message=f"Container '{container.get('name', f'container-{i}')}' has no resource limits",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
                
                # Check if requests are set
                if not requests:
                    self.results.append(ValidationResult(
                        rule_name="missing_requests",
                        severity="warning",
                        message=f"Container '{container.get('name', f'container-{i}')}' has no resource requests",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
                
                # Check CPU limits
                cpu_limit = limits.get('cpu')
                if cpu_limit and self._parse_cpu_value(cpu_limit) > 2:
                    self.results.append(ValidationResult(
                        rule_name="high_cpu_limit",
                        severity="warning",
                        message=f"Container '{container.get('name', f'container-{i}')}' has high CPU limit: {cpu_limit}",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
                
                # Check memory limits
                memory_limit = limits.get('memory')
                if memory_limit and self._parse_memory_value(memory_limit) > 4 * 1024 * 1024 * 1024:  # 4GB
                    self.results.append(ValidationResult(
                        rule_name="high_memory_limit",
                        severity="warning",
                        message=f"Container '{container.get('name', f'container-{i}')}' has high memory limit: {memory_limit}",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
    
    def _check_health_checks(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check health check configuration"""
        if kind in ['Deployment', 'StatefulSet', 'DaemonSet']:
            containers = self._get_containers(resource)
            
            for i, container in enumerate(containers):
                # Check liveness probe
                if not container.get('livenessProbe'):
                    self.results.append(ValidationResult(
                        rule_name="missing_liveness_probe",
                        severity="warning",
                        message=f"Container '{container.get('name', f'container-{i}')}' has no liveness probe",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
                
                # Check readiness probe
                if not container.get('readinessProbe'):
                    self.results.append(ValidationResult(
                        rule_name="missing_readiness_probe",
                        severity="warning",
                        message=f"Container '{container.get('name', f'container-{i}')}' has no readiness probe",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
    
    def _check_network_policies(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check network policy configuration"""
        if kind == 'NetworkPolicy':
            spec = resource.get('spec', {})
            
            # Check if ingress rules are too permissive
            ingress_rules = spec.get('ingress', [])
            for rule in ingress_rules:
                if not rule.get('from') and not rule.get('ports'):
                    self.results.append(ValidationResult(
                        rule_name="permissive_ingress",
                        severity="warning",
                        message="NetworkPolicy has permissive ingress rule",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
    
    def _check_image_security(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check image security"""
        if kind in ['Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob']:
            containers = self._get_containers(resource)
            
            for i, container in enumerate(containers):
                image = container.get('image', '')
                
                # Check for latest tag
                if image.endswith(':latest') or ':' not in image:
                    self.results.append(ValidationResult(
                        rule_name="latest_tag",
                        severity="warning",
                        message=f"Container '{container.get('name', f'container-{i}')}' uses 'latest' tag",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
                
                # Check for public registry
                if image.startswith('docker.io/') or not '/' in image:
                    self.results.append(ValidationResult(
                        rule_name="public_registry",
                        severity="info",
                        message=f"Container '{container.get('name', f'container-{i}')}' uses public registry",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
    
    def _check_secrets_management(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check secrets management"""
        if kind == 'Secret':
            secret_type = resource.get('type', '')
            
            # Check for hardcoded secrets
            data = resource.get('data', {})
            string_data = resource.get('stringData', {})
            
            for key, value in {**data, **string_data}.items():
                if isinstance(value, str) and len(value) < 32:
                    self.results.append(ValidationResult(
                        rule_name="weak_secret",
                        severity="warning",
                        message=f"Secret '{key}' appears to be weak (too short)",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
        
        # Check for secret references in other resources
        if kind in ['Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob']:
            containers = self._get_containers(resource)
            
            for i, container in enumerate(containers):
                env_vars = container.get('env', [])
                for env_var in env_vars:
                    if env_var.get('valueFrom', {}).get('secretKeyRef'):
                        # This is good - using secret references
                        continue
                    elif env_var.get('value') and len(env_var.get('value', '')) > 20:
                        self.results.append(ValidationResult(
                            rule_name="hardcoded_secret",
                            severity="error",
                            message=f"Container '{container.get('name', f'container-{i}')}' has hardcoded secret in env var '{env_var.get('name')}'",
                            file_path=file_path,
                            resource_name=name,
                            resource_type=kind
                        ))
    
    def _check_pod_security(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check pod security configuration"""
        if kind in ['Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob']:
            pod_spec = self._get_pod_spec(resource)
            
            # Check host network
            if pod_spec.get('hostNetwork', False):
                self.results.append(ValidationResult(
                    rule_name="host_network",
                    severity="error",
                    message="Pod uses host network",
                    file_path=file_path,
                    resource_name=name,
                    resource_type=kind
                ))
            
            # Check host PID
            if pod_spec.get('hostPID', False):
                self.results.append(ValidationResult(
                    rule_name="host_pid",
                    severity="error",
                    message="Pod uses host PID namespace",
                    file_path=file_path,
                    resource_name=name,
                    resource_type=kind
                ))
            
            # Check host IPC
            if pod_spec.get('hostIPC', False):
                self.results.append(ValidationResult(
                    rule_name="host_ipc",
                    severity="error",
                    message="Pod uses host IPC namespace",
                    file_path=file_path,
                    resource_name=name,
                    resource_type=kind
                ))
    
    def _check_service_configuration(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check service configuration"""
        if kind == 'Service':
            spec = resource.get('spec', {})
            service_type = spec.get('type', 'ClusterIP')
            
            # Check for NodePort services
            if service_type == 'NodePort':
                self.results.append(ValidationResult(
                    rule_name="nodeport_service",
                    severity="warning",
                    message="Service uses NodePort type - consider using LoadBalancer or Ingress",
                    file_path=file_path,
                    resource_name=name,
                    resource_type=kind
                ))
            
            # Check for external IPs
            external_ips = spec.get('externalIPs', [])
            if external_ips:
                self.results.append(ValidationResult(
                    rule_name="external_ips",
                    severity="warning",
                    message=f"Service has external IPs: {external_ips}",
                    file_path=file_path,
                    resource_name=name,
                    resource_type=kind
                ))
    
    def _check_ingress_security(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check ingress security configuration"""
        if kind == 'Ingress':
            spec = resource.get('spec', {})
            
            # Check for TLS configuration
            tls = spec.get('tls', [])
            if not tls:
                self.results.append(ValidationResult(
                    rule_name="no_tls",
                    severity="warning",
                    message="Ingress has no TLS configuration",
                    file_path=file_path,
                    resource_name=name,
                    resource_type=kind
                ))
            
            # Check for security annotations
            annotations = resource.get('metadata', {}).get('annotations', {})
            if not annotations.get('nginx.ingress.kubernetes.io/ssl-redirect'):
                self.results.append(ValidationResult(
                    rule_name="no_ssl_redirect",
                    severity="info",
                    message="Ingress should redirect HTTP to HTTPS",
                    file_path=file_path,
                    resource_name=name,
                    resource_type=kind
                ))
            
            # Check for rate limiting
            if not annotations.get('nginx.ingress.kubernetes.io/rate-limit'):
                self.results.append(ValidationResult(
                    rule_name="no_rate_limit",
                    severity="warning",
                    message="Ingress should have rate limiting configured",
                    file_path=file_path,
                    resource_name=name,
                    resource_type=kind
                ))
            
            # Check for CORS configuration
            if not annotations.get('nginx.ingress.kubernetes.io/enable-cors'):
                self.results.append(ValidationResult(
                    rule_name="no_cors",
                    severity="info",
                    message="Consider enabling CORS for web applications",
                    file_path=file_path,
                    resource_name=name,
                    resource_type=kind
                ))
    
    def _check_pod_disruption_budget(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check for Pod Disruption Budget configuration"""
        if kind in ['Deployment', 'StatefulSet', 'DaemonSet']:
            # Check if there's a corresponding PDB
            # This is a simplified check - in practice, you'd need to check all files
            self.results.append(ValidationResult(
                rule_name="missing_pdb",
                severity="info",
                message=f"Consider creating a PodDisruptionBudget for {kind} '{name}'",
                file_path=file_path,
                resource_name=name,
                resource_type=kind
            ))
    
    def _check_horizontal_pod_autoscaler(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check for Horizontal Pod Autoscaler configuration"""
        if kind in ['Deployment', 'StatefulSet']:
            # Check if there's a corresponding HPA
            # This is a simplified check - in practice, you'd need to check all files
            self.results.append(ValidationResult(
                rule_name="missing_hpa",
                severity="info",
                message=f"Consider creating a HorizontalPodAutoscaler for {kind} '{name}'",
                file_path=file_path,
                resource_name=name,
                resource_type=kind
            ))
    
    def _check_volume_security(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check volume security configuration"""
        if kind in ['Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob']:
            pod_spec = self._get_pod_spec(resource)
            volumes = pod_spec.get('volumes', [])
            
            for volume in volumes:
                # Check for hostPath volumes
                if 'hostPath' in volume:
                    self.results.append(ValidationResult(
                        rule_name="hostpath_volume",
                        severity="warning",
                        message=f"Volume '{volume['name']}' uses hostPath - consider using persistent volumes",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
                
                # Check for emptyDir volumes
                if 'emptyDir' in volume:
                    empty_dir = volume['emptyDir']
                    if empty_dir.get('sizeLimit'):
                        size_limit = empty_dir['sizeLimit']
                        if self._parse_memory_value(size_limit) > 1024 * 1024 * 1024:  # 1GB
                            self.results.append(ValidationResult(
                                rule_name="large_emptydir",
                                severity="warning",
                                message=f"Volume '{volume['name']}' has large size limit: {size_limit}",
                                file_path=file_path,
                                resource_name=name,
                                resource_type=kind
                            ))
    
    def _check_environment_variables(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check environment variable configuration"""
        if kind in ['Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob']:
            containers = self._get_containers(resource)
            
            for i, container in enumerate(containers):
                env_vars = container.get('env', [])
                
                # Check for environment variables without valueFrom
                for env_var in env_vars:
                    if 'value' in env_var and not env_var.get('valueFrom'):
                        var_name = env_var.get('name', 'unknown')
                        var_value = env_var.get('value', '')
                        
                        # Check for potential secrets in environment variables
                        if any(keyword in var_value.lower() for keyword in ['password', 'secret', 'key', 'token']):
                            self.results.append(ValidationResult(
                                rule_name="potential_secret_env",
                                severity="warning",
                                message=f"Container '{container.get('name', f'container-{i}')}' has potential secret in env var '{var_name}'",
                                file_path=file_path,
                                resource_name=name,
                                resource_type=kind
                            ))
    
    def _check_image_pull_policy(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check image pull policy configuration"""
        if kind in ['Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob']:
            containers = self._get_containers(resource)
            
            for i, container in enumerate(containers):
                image_pull_policy = container.get('imagePullPolicy', 'IfNotPresent')
                
                # Check for Always pull policy
                if image_pull_policy == 'Always':
                    self.results.append(ValidationResult(
                        rule_name="always_pull_policy",
                        severity="warning",
                        message=f"Container '{container.get('name', f'container-{i}')}' uses 'Always' pull policy - may impact performance",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
                
                # Check for Never pull policy
                if image_pull_policy == 'Never':
                    self.results.append(ValidationResult(
                        rule_name="never_pull_policy",
                        severity="error",
                        message=f"Container '{container.get('name', f'container-{i}')}' uses 'Never' pull policy - may cause deployment issues",
                        file_path=file_path,
                        resource_name=name,
                        resource_type=kind
                    ))
    
    def _check_service_account(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check service account configuration"""
        if kind in ['Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob']:
            pod_spec = self._get_pod_spec(resource)
            service_account = pod_spec.get('serviceAccountName')
            
            # Check for default service account
            if not service_account or service_account == 'default':
                self.results.append(ValidationResult(
                    rule_name="default_service_account",
                    severity="warning",
                    message="Pod uses default service account - consider creating a dedicated service account",
                    file_path=file_path,
                    resource_name=name,
                    resource_type=kind
                ))
    
    def _check_affinity_rules(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check affinity and anti-affinity rules"""
        if kind in ['Deployment', 'StatefulSet', 'DaemonSet']:
            pod_spec = self._get_pod_spec(resource)
            affinity = pod_spec.get('affinity', {})
            
            # Check for pod anti-affinity
            pod_anti_affinity = affinity.get('podAntiAffinity', {})
            if not pod_anti_affinity:
                self.results.append(ValidationResult(
                    rule_name="no_pod_anti_affinity",
                    severity="info",
                    message="Consider adding pod anti-affinity rules for high availability",
                    file_path=file_path,
                    resource_name=name,
                    resource_type=kind
                ))
    
    def _check_termination_grace_period(self, resource: Dict[str, Any], file_path: str, kind: str, name: str):
        """Check termination grace period configuration"""
        if kind in ['Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob']:
            pod_spec = self._get_pod_spec(resource)
            termination_grace_period = pod_spec.get('terminationGracePeriodSeconds', 30)
            
            # Check for very short grace period
            if termination_grace_period < 10:
                self.results.append(ValidationResult(
                    rule_name="short_grace_period",
                    severity="warning",
                    message=f"Termination grace period is very short: {termination_grace_period}s",
                    file_path=file_path,
                    resource_name=name,
                    resource_type=kind
                ))
            
            # Check for very long grace period
            if termination_grace_period > 300:
                self.results.append(ValidationResult(
                    rule_name="long_grace_period",
                    severity="warning",
                    message=f"Termination grace period is very long: {termination_grace_period}s",
                    file_path=file_path,
                    resource_name=name,
                    resource_type=kind
                ))
    
    def _get_containers(self, resource: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract containers from a resource"""
        containers = []
        
        if 'spec' in resource:
            spec = resource['spec']
            
            # Direct containers
            if 'containers' in spec:
                containers.extend(spec['containers'])
            
            # Template containers (for Deployments, StatefulSets, etc.)
            if 'template' in spec and 'spec' in spec['template']:
                template_spec = spec['template']['spec']
                if 'containers' in template_spec:
                    containers.extend(template_spec['containers'])
        
        return containers
    
    def _get_pod_spec(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pod spec from a resource"""
        if 'spec' in resource:
            spec = resource['spec']
            
            # Direct pod spec
            if 'containers' in spec:
                return spec
            
            # Template pod spec
            if 'template' in spec and 'spec' in spec['template']:
                return spec['template']['spec']
        
        return {}
    
    def _get_nested_value(self, data: Dict[str, Any], key_path: str) -> Any:
        """Get nested value from dictionary using dot notation"""
        keys = key_path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _parse_cpu_value(self, cpu_str: str) -> float:
        """Parse CPU value to cores"""
        cpu_str = cpu_str.strip()
        
        if cpu_str.endswith('m'):
            return float(cpu_str[:-1]) / 1000
        elif cpu_str.endswith('n'):
            return float(cpu_str[:-1]) / 1000000000
        else:
            return float(cpu_str)
    
    def _parse_memory_value(self, memory_str: str) -> int:
        """Parse memory value to bytes"""
        memory_str = memory_str.strip().upper()
        
        multipliers = {
            'K': 1024,
            'M': 1024 * 1024,
            'G': 1024 * 1024 * 1024,
            'T': 1024 * 1024 * 1024 * 1024,
            'KI': 1024,
            'MI': 1024 * 1024,
            'GI': 1024 * 1024 * 1024,
            'TI': 1024 * 1024 * 1024 * 1024
        }
        
        for suffix, multiplier in multipliers.items():
            if memory_str.endswith(suffix):
                return int(float(memory_str[:-len(suffix)]) * multiplier)
        
        return int(memory_str)
    
    def print_results(self, results: List[ValidationResult]):
        """Print validation results"""
        if not results:
            print(f"{self.severity_colors['success']}✅ All validations passed!{self.reset_color}")
            return
        
        # Group results by severity
        errors = [r for r in results if r.severity == 'error']
        warnings = [r for r in results if r.severity == 'warning']
        infos = [r for r in results if r.severity == 'info']
        
        print(f"\n📊 Validation Results:")
        print(f"   {self.severity_colors['error']}❌ Errors: {len(errors)}{self.reset_color}")
        print(f"   {self.severity_colors['warning']}⚠️  Warnings: {len(warnings)}{self.reset_color}")
        print(f"   {self.severity_colors['info']}ℹ️  Info: {len(infos)}{self.reset_color}")
        
        # Print errors
        if errors:
            print(f"\n{self.severity_colors['error']}❌ ERRORS:{self.reset_color}")
            for result in errors:
                print(f"   {result.file_path}:{result.line_number or '?'} - {result.message}")
        
        # Print warnings
        if warnings:
            print(f"\n{self.severity_colors['warning']}⚠️  WARNINGS:{self.reset_color}")
            for result in warnings:
                print(f"   {result.file_path}:{result.line_number or '?'} - {result.message}")
        
        # Print info
        if infos:
            print(f"\n{self.severity_colors['info']}ℹ️  INFO:{self.reset_color}")
            for result in infos:
                print(f"   {result.file_path}:{result.line_number or '?'} - {result.message}")
    
    def export_results(self, results: List[ValidationResult], output_file: str):
        """Export results to JSON file"""
        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total": len(results),
                "errors": len([r for r in results if r.severity == 'error']),
                "warnings": len([r for r in results if r.severity == 'warning']),
                "info": len([r for r in results if r.severity == 'info'])
            },
            "results": [
                {
                    "rule_name": r.rule_name,
                    "severity": r.severity,
                    "message": r.message,
                    "file_path": r.file_path,
                    "line_number": r.line_number,
                    "resource_name": r.resource_name,
                    "resource_type": r.resource_type
                }
                for r in results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📄 Results exported to: {output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Kubernetes Configuration Validator")
    parser.add_argument("files", nargs="+", help="Kubernetes manifest files to validate")
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")
    parser.add_argument("--severity", choices=["error", "warning", "info"], default="info",
                       help="Minimum severity level to report")
    
    args = parser.parse_args()
    
    validator = KubernetesValidator()
    all_results = []
    
    print("🔍 Kubernetes Configuration Validator")
    print("=" * 50)
    
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        
        print(f"\n📁 Validating: {file_path}")
        results = validator.validate_file(file_path)
        all_results.extend(results)
    
    # Filter results by severity
    severity_levels = {"error": 0, "warning": 1, "info": 2}
    min_level = severity_levels[args.severity]
    filtered_results = [
        r for r in all_results 
        if severity_levels[r.severity] <= min_level
    ]
    
    # Print results
    validator.print_results(filtered_results)
    
    # Export results if requested
    if args.output:
        validator.export_results(filtered_results, args.output)
    
    # Exit with error code if there are errors
    error_count = len([r for r in filtered_results if r.severity == 'error'])
    if error_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
