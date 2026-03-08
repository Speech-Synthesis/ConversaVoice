"""Compatibility entrypoint.

Keeps older deployment commands working by launching the simulation app.
"""

try:
    from simulation_app import main
except ImportError:
    from frontend.simulation_app import main


if __name__ == "__main__":
    main()
