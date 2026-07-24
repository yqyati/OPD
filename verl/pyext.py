"""Small Python 3.12-compatible subset of the unmaintained ``pyext`` package.

The local PRIME code reward only uses ``RuntimeModule.from_string`` to execute
a candidate solution in an isolated reward subprocess.  The PyPI ``pyext``
package relies on ``inspect.getargspec`` and cannot install on Python 3.12.
"""

from types import ModuleType


class RuntimeModule:
    """Create an in-memory Python module from source text."""

    @staticmethod
    def from_string(name: str, filename: str, source: str) -> ModuleType:
        module = ModuleType(name)
        module.__file__ = filename or f"<{name}>"
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        return module
