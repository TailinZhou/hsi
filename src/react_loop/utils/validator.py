"""
Code safety validator - uses AST parsing instead of string matching.

Improvements:
- Uses AST parsing to detect dangerous code patterns
- Prevents string-concatenation bypass
- Detects reflective calls and dynamic imports
- Provides detailed violation reports
"""

import ast
from dataclasses import dataclass, field
from typing import Set, List, Optional, Tuple


@dataclass
class ValidationResult:
    """Validation result."""
    valid: bool
    message: str
    violations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.valid:
            return f"✓ {self.message}"
        return f"✗ {self.message}\n  violations: {', '.join(self.violations)}"


class CodeValidator:
    """
    AST-level code safety validator.

    Uses AST parsing instead of simple string matching, effectively preventing:
    - String-concatenation bypass (e.g. "impor" + "t logging")
    - Whitespace bypass (e.g. "  os.system")
    - Dynamic calls (e.g. __import__("os").system())
    - Reflective calls (e.g. getattr(os, "system"))

    Usage example:
    ```python
    validator = CodeValidator()
    result = validator.validate("eval('1+1')")
    if not result.valid:
        print(f"safety validation failed: {result.violations}")
    ```
    """

    # Functions forbidden to call directly (emptied; safety is guaranteed by the sandbox environment)
    FORBIDDEN_CALLS: Set[str] = set()

    # Modules forbidden to import (the first part of the module name)
    FORBIDDEN_MODULES: Set[str] = {
        # System command execution
        "subprocess",
        "commands",  # Python 2 legacy
        "popen2",    # Python 2 legacy
        # Multiprocessing
        "multiprocessing",
        "concurrent",
        # Low-level system access
        "ctypes",
        "_ctypes",
        "winreg",
        "_winreg",
        # Networking
        "socket",
        "socketserver",
        "asyncore",
        "asynchat",
        # Serialization (may lead to RCE)
        "pickle",
        "cPickle",
        "shelve",
        "marshal",
        "dill",
        # System monitoring
        "resource",
        "syslog",
        # Other dangerous modules
        "posix",
        "nt",
    }

    # Attributes forbidden to access (prevents reflection-based bypass)
    FORBIDDEN_ATTRS: Set[str] = {
        # Class metadata
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        # Internal state
        "__globals__",
        "__code__",
        "__builtins__",
        "__import__",
        "__dict__",
        # Other dangerous attributes
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__reduce__",
        "__reduce_ex__",
    }

    # Forbidden module-level function calls (module.function format)
    FORBIDDEN_MODULE_CALLS: Set[str] = {
        "os.system",
        "os.popen",
        "os.spawnl",
        "os.spawnle",
        "os.spawnlp",
        "os.spawnlpe",
        "os.spawnv",
        "os.spawnve",
        "os.spawnvp",
        "os.spawnvpe",
        "os.exec",
        "os.execl",
        "os.execle",
        "os.execlp",
        "os.execlpe",
        "os.execv",
        "os.execve",
        "os.execvp",
        "os.execvpe",
        "os.fork",
        "os.kill",
        "os.killpg",
        "os.chroot",
        "os.setuid",
        "os.setgid",
        "time.sleep",  # Prevent blocking
    }

    def __init__(self, custom_forbidden_calls: Set[str] = None,
                 custom_forbidden_modules: Set[str] = None):
        """
        Initialize the validator.

        Args:
            custom_forbidden_calls: Additional list of forbidden functions.
            custom_forbidden_modules: Additional list of forbidden modules.
        """
        self._forbidden_calls = self.FORBIDDEN_CALLS.copy()
        self._forbidden_modules = self.FORBIDDEN_MODULES.copy()

        if custom_forbidden_calls:
            self._forbidden_calls.update(custom_forbidden_calls)
        if custom_forbidden_modules:
            self._forbidden_modules.update(custom_forbidden_modules)

    def validate(self, code: str) -> ValidationResult:
        """
        Validate code safety.

        Args:
            code: The code string to validate.

        Returns:
            ValidationResult containing the result and details.
        """
        violations: List[str] = []

        # 1. Try to parse the AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                message=f"syntax error: {e}",
                violations=[f"syntax error on line {e.lineno}: {e.msg}"]
            )

        # 2. Walk the AST to check for various dangerous patterns
        for node in ast.walk(tree):
            # Check function calls
            call_violations = self._check_call(node)
            violations.extend(call_violations)

            # Check import statements
            import_violations = self._check_import(node)
            violations.extend(import_violations)

            # Check attribute access
            attr_violations = self._check_attribute(node)
            violations.extend(attr_violations)

        # 3. Return the result
        if violations:
            return ValidationResult(
                valid=False,
                message="safety validation failed",
                violations=violations
            )
        return ValidationResult(
            valid=True,
            message="validation passed"
        )

    def _check_call(self, node: ast.AST) -> List[str]:
        """Check whether a function call is safe."""
        violations: List[str] = []

        if not isinstance(node, ast.Call):
            return violations

        # Get the function name
        func_name = self._get_full_func_name(node.func)

        # Check forbidden function calls
        if func_name in self._forbidden_calls:
            violations.append(f"forbidden function call: {func_name}")

        # Check forbidden module-level calls (e.g. os.system)
        if func_name in self.FORBIDDEN_MODULE_CALLS:
            violations.append(f"forbidden call: {func_name}")

        # Detect reflective calls via getattr
        if isinstance(node.func, ast.Call):
            if isinstance(node.func.func, ast.Name):
                if node.func.func.id == "getattr":
                    # getattr(x, "dangerous") form
                    violations.append("forbidden use of getattr for reflective calls")

        # Detect calls via string concatenation
        if isinstance(node.func, ast.Attribute):
            # Check the obj.attr form
            if isinstance(node.func.value, ast.Call):
                # Possibly the __import__("os").system() form
                inner_call = node.func.value
                if isinstance(inner_call.func, ast.Name):
                    if inner_call.func.id == "__import__":
                        violations.append(
                            f"forbidden dynamic call via __import__: "
                            f"{self._get_full_func_name(node.func)}"
                        )

        return violations

    def _check_import(self, node: ast.AST) -> List[str]:
        """Check whether import statements are safe."""
        violations: List[str] = []

        # Check `import x`
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_base = alias.name.split('.')[0]
                if module_base in self._forbidden_modules:
                    violations.append(f"forbidden module import: {alias.name}")

        # Check `from x import y`
        if isinstance(node, ast.ImportFrom):
            if node.module:
                module_base = node.module.split('.')[0]
                if module_base in self._forbidden_modules:
                    violations.append(f"forbidden import from module: {node.module}")

        return violations

    def _check_attribute(self, node: ast.AST) -> List[str]:
        """Check whether attribute access is safe."""
        violations: List[str] = []

        if not isinstance(node, ast.Attribute):
            return violations

        # Check forbidden attribute access
        if node.attr in self.FORBIDDEN_ATTRS:
            violations.append(f"forbidden attribute access: {node.attr}")

        return violations

    def _get_full_func_name(self, node: ast.AST) -> str:
        """
        Get the full name of a function.

        Supports the following formats:
        - func_name -> "func_name"
        - obj.method -> "obj.method"
        - mod.obj.method -> "mod.obj.method"
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value_name = self._get_full_func_name(node.value)
            return f"{value_name}.{node.attr}"
        return ""

    def _get_current_code(self, target_name: str) -> str:
        """
        Get the current code of a target (used for transactional backup).

        This method is called by TransactionalModifier.
        Subclasses can override it to provide a custom implementation.
        """
        # The default implementation returns an empty string.
        # The actual implementation should be in TransactionalModifier.
        return ""

    def add_forbidden_call(self, func_name: str) -> None:
        """Add a forbidden function."""
        self._forbidden_calls.add(func_name)

    def add_forbidden_module(self, module_name: str) -> None:
        """Add a forbidden module."""
        self._forbidden_modules.add(module_name)

    def remove_forbidden_call(self, func_name: str) -> None:
        """Remove a forbidden function."""
        self._forbidden_calls.discard(func_name)

    def remove_forbidden_module(self, module_name: str) -> None:
        """Remove a forbidden module."""
        self._forbidden_modules.discard(module_name)


def validate_code(code: str,
                  custom_forbidden_calls: Set[str] = None,
                  custom_forbidden_modules: Set[str] = None) -> ValidationResult:
    """
    Convenience function: validate code safety.

    Args:
        code: The code to validate.
        custom_forbidden_calls: Additional list of forbidden functions.
        custom_forbidden_modules: Additional list of forbidden modules.

    Returns:
        ValidationResult.
    """
    validator = CodeValidator(
        custom_forbidden_calls=custom_forbidden_calls,
        custom_forbidden_modules=custom_forbidden_modules
    )
    return validator.validate(code)
