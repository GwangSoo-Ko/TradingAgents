"""alpha-pulse contract lock: freeze what the production consumer reads from this fork.

See README.md in this directory. Keep this module import-free: the harness
subprocess imports ``tests.alphapulse_contract._*`` modules before the fork is
imported, so nothing here may import ``tradingagents``.
"""
