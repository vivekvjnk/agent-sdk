# BMS IC Comparison for Stackable Multi-Purpose Platform

## 1. Texas Instruments bq79612-Q1




### 1.1 Key Features
- **Cell Count:** 6S to 12S (BQ79612-Q1), up to 16S within the family (BQ79616-Q1)
- **Stacking Capability:** Multiple devices can be daisy-chained. Isolated, bidirectional daisy chain ports with optional ring architecture.
- **Communication:** Isolated differential daisy chain, UART/SPI host interface.
- **Cell Voltage Accuracy:** +/- 1.5mV.
- **Cell Balancing:** Internal, passive balancing at 240 mA with thermal management.
- **Protection:** Integrated hardware OVUV and OTUT comparators.
- **Auxiliary Inputs:** 8 GPIOs configurable for thermistor measurements.

### 1.2 Specifications
- **Absolute Maximum Input Voltage (BAT, VC*, etc.):** -0.3V to 100V
- **Absolute Maximum Differential Cell Voltage (VCn to VCn-1):** -80V to 80V
- **Recommended Total Module Voltage (VBAT_RANGE):** 9V to 80V
- **Max Cell Balancing Current:** 240 mA (for 8 cells at 75°C ambient)
- **Operating Ambient Temperature:** -40°C to 130°C
- **Operating Junction Temperature:** -40°C to 150°C




## 2. Infineon TLE9012DQU

### 2.1 Key Features

- **Cell Count:** Up to 12 battery cells in series.
- **Stacking Capability:** Up to 38 devices via isolated daisy chain interface.
- **Communication:** Differential robust serial 2 Mbit/s, iso UART/UART with end-to-end CRC.
- **Cell Voltage Accuracy:** ±0.2 mV initial accuracy.
- **Cell Balancing:** Integrated balancing switch, up to 200 mA balancing current, with overcurrent/undercurrent detection.
- **Protection/Diagnostics:** Automatic open load/wire detection, NTC monitoring, internal round robin diagnostics.
- **Temperature Measurement:** 5 external NTC channels, 2 internal temperature sensors.

### 2.2 Specifications
- **Absolute Maximum Supply Voltage (VS):** -0.3V to 75V
- **Absolute Maximum Transient High Voltage:** 75V to 90V (max 60s)
- **Absolute Maximum Cell Sense Input Voltage:** -0.3V to 75V
- **Relative Cell Sense Input Voltage (differential):** Up to 9V
- **Operating Junction Temperature:** -40°C to 150°C



## 3. STMicroelectronics L9963E

### 3.1 Key Features
- **Cell Count:** 4 to 14 cells in series.
- **Stacking Capability:** Up to 31 devices for 434 cells total, via isolated serial communication with dual access ring.
- **Communication:** 2.66 Mbps isolated serial communication (SPI or isolated interface) with regenerative buffer.
- **Cell Voltage Accuracy:** ±2 mV max error in [0.5 – 4.3] V range.
- **Cell Balancing:** 200 mA passive internal balancing, with manual/timed and internal/external options.
- **Protection/Diagnostics:** Redundant cell measurement path (ADC Swap), intelligent diagnostic routines, robust hot-plugging.
- **Auxiliary Inputs/Temperature:** 9 GPIOs, up to 7 analog inputs for NTC sensing.
- **Additional Features:** Integrated Coulomb counter for pack overcurrent detection.

### 3.2 Specifications
- **Operating Global Supply Voltage (VBAT):** 9.6V to 64V (normal), 64V to 70V (transient)
- **Operating Cell Terminal Differential Voltage:** 0V to 4.7V
- **Absolute Maximum Supply Voltage (VBAT, C14):** -0.3V to 72V
- **Absolute Maximum Differential Cell Voltage (C(n)-C(n-1)):** -72V to 72V (no damage, potential leakage); -6V to 6V (guaranteed leakage)
- **Operating Ambient Temperature:** -40°C to 105°C
- **Operating Junction Temperature:** -40°C to 125°C


