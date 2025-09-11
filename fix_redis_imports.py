#!/usr/bin/env python3
"""
Script to add missing redis_sync imports to service files
"""

import os
import re
from pathlib import Path

def fix_redis_imports_in_file(file_path):
    """Add missing redis_sync import to a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Check if file uses redis_sync but doesn't import it
        if 'redis_sync.Redis' in content and 'import redis as redis_sync' not in content:
            # Find the redis import line and add redis_sync import
            if 'import redis.asyncio as redis' in content:
                content = re.sub(
                    r'import redis\.asyncio as redis',
                    'import redis as redis_sync\nimport redis.asyncio as redis',
                    content
                )
            elif 'import redis' in content and 'import redis as redis_sync' not in content:
                content = re.sub(
                    r'import redis\n',
                    'import redis as redis_sync\n',
                    content
                )
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed Redis imports in {file_path}")
            return True
        else:
            print(f"⏭️  No changes needed in {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Fix Redis imports in all service files"""
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
        if fix_redis_imports_in_file(py_file):
            fixed_count += 1
    
    print(f"\n🎉 Fixed Redis imports in {fixed_count}/{total_count} service files")

if __name__ == "__main__":
    main()
