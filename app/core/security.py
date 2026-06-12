import os
import ast
from app.core.config import MCP_FS_ROOT

def is_path_sensitive(path_str: str) -> bool:
    path_str = str(path_str)
    if ".." in path_str:
        try:
            abs_path = os.path.abspath(path_str)
        except Exception:
            return True
    else:
        abs_path = os.path.abspath(path_str)
        
    base_name = os.path.basename(abs_path)
    
    # 1. Block hidden files/folders (starting with dot)
    if base_name.startswith('.'):
        return True
        
    # 2. Block python files and pycache
    if base_name.endswith('.py') or base_name.endswith('.pyc') or '__pycache__' in abs_path:
        return True
        
    # 3. Block database directories, scratch folders, and internal app files
    parts = abs_path.split(os.sep)
    if 'chroma_db' in parts or 'scratch' in parts or 'app' in parts:
        return True
        
    # 4. Block core configuration & server code files
    if base_name in (
        'backend.py', 'package.json', 'package-lock.json', 
        'production_readiness_report.md', 'whisper_integration_plan.md', 
        'playwright_persistence_report.md', 'security_hardening_plan.md'
    ):
        return True
        
    # 5. Block frontend code in static folder
    if 'static' in parts:
        if base_name.endswith(('.js', '.html', '.css')):
            return True
            
    return False


def is_code_safe(code_str: str) -> tuple[bool, str]:
    BANNED_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'socket', 'urllib', 'requests', 
        'importlib', 'ctypes', 'pty', 'platform', 'builtins'
    }
    BANNED_FUNCTIONS = {
        'eval', 'exec', 'compile', 'getattr', 'setattr', 'delattr', 'hasattr',
        'open', 'remove', 'unlink', 'rmdir', 'rmtree', 'system', 'popen', 'spawn', 'fork'
    }
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split('.')[0]
                if module_name in BANNED_MODULES:
                    return False, f"Import of dangerous module '{module_name}' is banned for security reasons."
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split('.')[0]
                if module_name in BANNED_MODULES:
                    return False, f"Importing from dangerous module '{module_name}' is banned for security reasons."
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in BANNED_FUNCTIONS:
                    return False, f"Calling function '{func_name}' is banned for security reasons."
            elif isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                if attr_name in BANNED_FUNCTIONS:
                    return False, f"Accessing attribute/method '{attr_name}' is banned for security reasons."
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value.lower()
            if any(sec in val for sec in ['.env', 'backend.py', 'chroma_db', 'scratch', '__pycache__', 'app.js', 'style.css', 'index.html', 'app/']):
                return False, f"Reference to restricted system file or path '{node.value}' is banned."
    return True, "Code is safe."


def sanitize_tool_output(output_str: str) -> str:
    if not isinstance(output_str, str):
        return output_str
    lines = output_str.splitlines()
    cleaned_lines = []
    for line in lines:
        lower_line = line.lower()
        # Skip lines containing sensitive filenames or folders
        if any(sec in lower_line for sec in [
            '.env', 'backend.py', 'chroma_db', 'scratch', '__pycache__', 
            'package.json', 'package-lock.json', 'production_readiness_report.md', 
            'whisper_integration_plan.md', 'playwright_persistence_report.md', 
            'security_hardening_plan.md', 'app'
        ]):
            continue
        # Skip lines referencing Python files or web code files (.js, .html, .css)
        if line.strip().endswith('.py') or '.py ' in line or '.pyc' in line:
            continue
        if any(ext in lower_line for ext in ['.js', '.html', '.css']):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def sanitize_value(val):
    if isinstance(val, str):
        return sanitize_tool_output(val)
    elif isinstance(val, tuple):
        return tuple(sanitize_value(item) for item in val)
    elif isinstance(val, list):
        return [sanitize_value(item) for item in val]
    elif isinstance(val, dict):
        return {k: sanitize_value(v) for k, v in val.items()}
    elif hasattr(val, 'text') and isinstance(val.text, str):
        val.text = sanitize_tool_output(val.text)
        return val
    elif hasattr(val, 'content') and isinstance(val.content, str):
        val.content = sanitize_tool_output(val.content)
        return val
    return val
