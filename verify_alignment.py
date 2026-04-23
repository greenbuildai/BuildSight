import os
import re

ROOT_DIR = r"E:\Company\Green Build AI\Prototypes\BuildSight"
FORBIDDEN_PATTERN = re.compile(r"[Dd]:[\\/]Jovi", re.IGNORECASE)

def audit_workspace():
    print(f"--- Auditing Workspace: {ROOT_DIR} ---")
    stale_refs = []
    
    for root, dirs, files in os.walk(ROOT_DIR):
        # Skip heavy/binary directories
        if any(x in root for x in [".git", ".venv", "node_modules", "__pycache__", "weights", "runs"]):
            continue
            
        for file in files:
            if file.endswith(('.py', '.json', '.env', '.sh', '.md', '.html', '.ts', '.tsx')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if FORBIDDEN_PATTERN.search(content):
                            # Find line numbers
                            lines = content.splitlines()
                            for i, line in enumerate(lines):
                                if FORBIDDEN_PATTERN.search(line):
                                    stale_refs.append(f"{file_path}:{i+1} -> {line.strip()}")
                except Exception:
                    pass # Skip files that can't be read

    if stale_refs:
        print(f"\n[!] Found {len(stale_refs)} stale D: drive references:")
        for ref in stale_refs:
            print(f"  {ref}")
    else:
        print("\n[+] Success: No stale D: drive references found in scanned files!")

if __name__ == "__main__":
    audit_workspace()
