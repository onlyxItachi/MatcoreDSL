#!/usr/bin/env python3
"""
MDSLC Windows Lowering Corpus — Strict JSON Schema Validator
Validates environment descriptors, case descriptors, and manifests against corpus/schema/*.schema.json.
"""

import json
import os
import sys
from pathlib import Path

def validate_dict_fields(instance, schema, path="root"):
    errors = []
    
    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in instance:
            errors.append(f"{path}: Missing required field '{field}'")
            
    # Check properties
    props = schema.get("properties", {})
    for k, v in instance.items():
        if k in props:
            prop_schema = props[k]
            sub_path = f"{path}.{k}"
            expected_type = prop_schema.get("type")
            
            if expected_type == "string" and not isinstance(v, str):
                errors.append(f"{sub_path}: Expected string, got {type(v).__name__}")
            elif expected_type == "integer" and not isinstance(v, int):
                errors.append(f"{sub_path}: Expected int, got {type(v).__name__}")
            elif expected_type == "boolean" and not isinstance(v, bool):
                errors.append(f"{sub_path}: Expected bool, got {type(v).__name__}")
            elif expected_type == "array" and not isinstance(v, list):
                errors.append(f"{sub_path}: Expected array, got {type(v).__name__}")
            elif expected_type == "object" and isinstance(v, dict):
                errors.extend(validate_dict_fields(v, prop_schema, sub_path))
                
            # Check const
            if "const" in prop_schema and v != prop_schema["const"]:
                errors.append(f"{sub_path}: Expected const '{prop_schema['const']}', got '{v}'")
                
            # Check enum
            if "enum" in prop_schema and v not in prop_schema["enum"]:
                errors.append(f"{sub_path}: Value '{v}' not in enum {prop_schema['enum']}")
                
    return errors

def main():
    script_dir = Path(__file__).resolve().parent
    schema_dir = script_dir.parent.parent / "schema"
    env_dir = script_dir.parent.parent / "environments" / "windows-x64"
    
    print("=== MDSLC Strict JSON Schema Validator ===")
    
    # 1. Load Schemas
    with open(schema_dir / "environment.schema.json", "r", encoding="utf-8") as f:
        env_schema = json.load(f)
    with open(schema_dir / "corpus-entry.schema.json", "r", encoding="utf-8") as f:
        case_schema = json.load(f)
        
    total_checked = 0
    total_errors = 0
    
    # 2. Validate Environment Descriptors
    if env_dir.exists():
        for env_file in env_dir.glob("*.json"):
            with open(env_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            errs = validate_dict_fields(data, env_schema, env_file.name)
            total_checked += 1
            if errs:
                print(f"[FAIL] {env_file.name}: {len(errs)} errors")
                for e in errs:
                    print(f"   - {e}")
                total_errors += len(errs)
            else:
                print(f"[PASS] {env_file.name} validated against environment.schema.json")

    # 3. Validate Case Descriptors in External Data Plane
    corpus_root = Path(r"C:\Users\hamza\MDSLC-Corpus\windows-x64")
    if corpus_root.exists():
        for desc_file in corpus_root.glob("**/descriptor.json"):
            with open(desc_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            rel_path = desc_file.relative_to(corpus_root)
            errs = validate_dict_fields(data, case_schema, str(rel_path))
            total_checked += 1
            if errs:
                print(f"[FAIL] {rel_path}: {len(errs)} errors")
                for e in errs:
                    print(f"   - {e}")
                total_errors += len(errs)
            else:
                print(f"[PASS] {rel_path} validated against corpus-entry.schema.json")
                
    print("\n=== Validation Summary ===")
    print(f"Total JSON Descriptors Validated: {total_checked}")
    print(f"Total Schema Errors: {total_errors}")
    
    if total_errors > 0:
        sys.exit(1)
    else:
        print("Schema Validation Status: 100% SUCCESSFUL.")
        sys.exit(0)

if __name__ == "__main__":
    main()
