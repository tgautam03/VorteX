# Drone Geometry Constants Explained

The drone geometry is now defined dynamically in the `Drone2D` class in `vortex/drone2d.py`. You can customize these values by passing arguments to `Drone2D()`.

## 1. Central Body (The "Fuselage")
The main body is modeled as an **ellipse**.
*   **Parameter:** `body_width` (Default: 30.0)
    *   Maps to: `BODY_RADIUS_X`
    *   Meaning: Half the width of the central body. Total width is $2 \times$ `body_width`.
*   **Parameter:** `body_height` (Default: 20.0)
    *   Maps to: `BODY_RADIUS_Y`
    *   Meaning: Half the height of the central body. Total height is $2 \times$ `body_height`.

## 2. Arms (The Frame)
The arms hold the motors and connect them to the body.
*   **Parameter:** `arm_length` (Default: 180.0)
    *   Maps to: `ARM_LENGTH`
    *   Meaning: The total span from the tip of the left arm to the tip of the right arm.
*   **Parameter:** `arm_thickness` (Default: 10.0)
    *   Maps to: `ARM_THICKNESS`
    *   Meaning: How thick the structural bars are.

## 3. Motors (The Propulsion Units)
The motors are the square blocks at the ends of the arms.
*   **Parameter:** `motor_offset` (Default: 90.0)
    *   Maps to: `MOTOR_OFFSET`
    *   Meaning: The distance from the center of the drone to the center of each motor.
*   **Parameter:** `motor_size` (Default: 30.0)
    *   Maps to: `MOTOR_SIZE`
    *   Meaning: The side length of the square motor housing.

## 4. Propellers (The Rotors)
The propellers generate the thrust.
*   **Parameter:** `prop_width` (Default: 50.0)
    *   Maps to: `PROP_WIDTH`
    *   Meaning: The diameter of the rotor disk. This is the most important scale for aerodynamics.
*   **Parameter:** `prop_height_above_body_center` (Default: 10.0)
    *   Maps to: `PROP_OFFSET_Y`
    *   Meaning: The vertical distance from the center of the drone to the propeller line.

## Visual Layout
```text
       <-- prop_width -->
      [==================]      <-- Propeller (at y = +prop_height_above_body_center)
              |  |
          [MOTOR_BOX]           <-- Motor (at x = -motor_offset)
=============|  |=============  <-- Arm (Length = arm_length)
      (   Central Body   )      <-- Ellipse (Radius = body_width, body_height)
=============|  |=============
          [MOTOR_BOX]           <-- Motor (at x = +motor_offset)
              |  |
      [==================]
```
