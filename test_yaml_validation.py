#!/usr/bin/env python3
import yaml
import sys
import glob

def validate_yaml_file(filepath):
    try:
        with open(filepath) as f:
            content = f.read()
            if '---' in content and content.count('---') > 1:
                list(yaml.safe_load_all(content))
            else:
                yaml.safe_load(content)
        print(f'✅ {filepath}: OK')
        return True
    except Exception as e:
        print(f'❌ {filepath}: {e}')
        return False

def main():
    yaml_files = glob.glob('**/*.yml', recursive=True) + glob.glob('**/*.yaml', recursive=True)
    failed_files = []
    
    for yaml_file in yaml_files:
        if not validate_yaml_file(yaml_file):
            failed_files.append(yaml_file)
    
    if failed_files:
        print(f"\n❌ {len(failed_files)} YAML files failed validation")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(yaml_files)} YAML files are valid")
        sys.exit(0)

if __name__ == "__main__":
    main()
