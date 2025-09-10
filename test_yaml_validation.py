#!/usr/bin/env python3
import yaml
import sys
import glob

def validate_yaml_file(filepath):
    try:
        with open(filepath) as f:
            content = f.read()
            # Try single document first
            try:
                yaml.safe_load(content)
                print(f'✅ {filepath}: OK (single document)')
                return True
            except yaml.composer.ComposerError as e:
                if "expected a single document" in str(e):
                    # Try multi-document
                    try:
                        list(yaml.safe_load_all(content))
                        print(f'✅ {filepath}: OK (multi-document)')
                        return True
                    except Exception as e2:
                        print(f'❌ {filepath}: Multi-document error: {e2}')
                        return False
                else:
                    print(f'❌ {filepath}: Single document error: {e}')
                    return False
    except Exception as e:
        print(f'❌ {filepath}: File error: {e}')
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
