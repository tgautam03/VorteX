"""
Unit conversion utilities for LBM simulations.

This module provides conversion between SI units and lattice units.
"""

class UnitConverter:
    """
    Converts between SI units and lattice units for LBM simulations.
    
    The conversion is based on three characteristic scales:
    - dx: Spatial resolution (meters per lattice unit)
    - u_lattice: Reference velocity in lattice units
    - u_real: Physical velocity that u_lattice represents (m/s)
    
    From these, we derive the time scale:
        dt = (u_lattice / u_real) * dx
    """
    
    def __init__(self, dx: float, u_real: float, u_lattice: float = 0.15):
        """
        Initialize unit converter.
        
        Args:
            dx: Spatial resolution (meters per lattice unit)
            u_real: Physical characteristic velocity (m/s)
            u_lattice: Lattice characteristic velocity (dimensionless)
        """
        self.dx = dx
        self.u_real = u_real
        self.u_lattice = u_lattice
        
        # Derived time scale
        self.dt = (u_lattice / u_real) * dx
        
        # Velocity scale
        self.velocity_scale = u_real / u_lattice
    
    def gravity_to_lattice(self, g_SI: float) -> float:
        """
        Convert gravity from SI (m/s²) to lattice units.
        
        Formula: g_lattice = g_SI * dt² / dx
        
        Args:
            g_SI: Gravity in SI units (m/s²)
            
        Returns:
            Gravity in lattice units
        """
        return g_SI * self.dt**2 / self.dx
    
    def velocity_to_lattice(self, u_SI: float) -> float:
        """
        Convert velocity from SI (m/s) to lattice units.
        
        Args:
            u_SI: Velocity in SI units (m/s)
            
        Returns:
            Velocity in lattice units
        """
        return u_SI * self.dt / self.dx
    
    def time_to_lattice(self, t_SI: float) -> int:
        """
        Convert time from SI (s) to lattice timesteps.
        
        Args:
            t_SI: Time in SI units (s)
            
        Returns:
            Number of timesteps
        """
        return int(t_SI / self.dt)
    
    def force_to_lattice(self, f_SI: float) -> float:
        """
        Convert force from SI (N) to lattice units.
        
        Args:
            f_SI: Force in SI units (N)
            
        Returns:
            Force in lattice units
        """
        return f_SI * self.dt**2 / (self.dx**2)
    
    def __repr__(self) -> str:
        return (f"UnitConverter(dx={self.dx} m, "
                f"u_real={self.u_real} m/s, "
                f"u_lattice={self.u_lattice}, "
                f"dt={self.dt:.6e} s)")
