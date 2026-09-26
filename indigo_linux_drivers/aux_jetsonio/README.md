# NVidia Jetson GPIO AUX Controller driver

https://www.nvidia.com/de-de/autonomous-machines/embedded-systems/jetson-nano/

## Supported devices

GPIO port of the NVidia Jetson Nano Developer Kit.

No hot plug support. One GPIO device is present at startup.

## Supported platforms

This driver is supported on Linux (ARM v8).

## License

INDIGO Astronomy open-source license.

## Use

indigo_server indigo_aux_jetsonio

## Status: More testing needed

Driver is developed and tested with:
* NVidia Jetson Nano Developer Kit 2GB
* NVidia Jetson Nano Developer Kit A02
* NVidia Jetson Nano Developer Kit B01

## GPIO Pin mapping:

### Inputs
* Not used

### Outputs
* Output 1 -> GPIO 38 (PWM2*)
* Output 2 -> GPIO 168 (PWM0*)
* Output 3 -> GPIO 76
* Output 4 -> GPIO 12
