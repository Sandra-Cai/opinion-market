#!/usr/bin/env python3
"""
Script to fix Redis type annotations in all service files
"""

import os
import re
from pathlib import Path

def fix_redis_types_in_file(file_path):
    """Fix Redis type annotations in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix import statements
        if 'import redis' in content and 'import redis as redis_sync' not in content:
            content = re.sub(r'import redis\n', 'import redis as redis_sync\n', content)
        
        # Fix type annotations
        content = re.sub(r'redis_client: redis\.Redis', 'redis_client: redis_sync.Redis', content)
        content = re.sub(r'redis_client: Optional\[redis\.Redis\]', 'redis_client: Optional[redis_sync.Redis]', content)
        content = re.sub(r'redis_client: redis\.Redis,', 'redis_client: redis_sync.Redis,', content)
        content = re.sub(r'redis_client: redis\.Redis\)', 'redis_client: redis_sync.Redis)', content)
        
        # Fix class attributes
        content = re.sub(r'self\.redis_client: Optional\[redis\.Redis\]', 'self.redis_client: Optional[redis_sync.Redis]', content)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed Redis types in {file_path}")
            return True
        else:
            print(f"⏭️  No changes needed in {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Fix Redis type annotations in all service files"""
    services_dir = Path("app/services")
    
    if not services_dir.exists():
        print("❌ Services directory not found")
        return
    
    fixed_count = 0
    total_count = 0
    
    for py_file in services_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        total_count += 1
        if fix_redis_types_in_file(py_file):
            fixed_count += 1
    
    print(f"\n🎉 Fixed Redis types in {fixed_count}/{total_count} service files")

if __name__ == "__main__":
    main()
