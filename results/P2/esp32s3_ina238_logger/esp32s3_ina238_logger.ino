/*
 * esp32s3_ina238_logger.ino  --  P2 power logger (Reviewer 1, comment 3)
 * -----------------------------------------------------------------------
 * ESP32-S3 reads an INA238 over I2C and streams one CSV line per conversion
 * over USB serial. No external library needed (plain Wire register access).
 *
 * Wiring (INA238 breakout, 15 mOhm shunt):
 *   INA238 VS / VCC  -> ESP32-S3 3V3
 *   INA238 GND       -> ESP32-S3 GND  AND  Nucleo GND (common ground!)
 *   INA238 SDA       -> ESP32-S3 GPIO 8
 *   INA238 SCL       -> ESP32-S3 GPIO 9
 *   INA238 IN+       -> Nucleo IDD jumper pin on the SUPPLY side (reads 3.3 V with jumper removed)
 *   INA238 IN-       -> Nucleo IDD jumper pin on the MCU side
 *   INA238 VBUS      -> IN- (if the breakout has a separate VBUS pin)
 *   Address pins A0/A1 -> GND  (I2C address 0x40)
 *
 * Output (115200 baud, but native USB ignores the rate):
 *   # header lines start with '#'
 *   t_us,vbus_V,vshunt_uV,current_mA,power_mW
 *
 * ADC setup: continuous shunt+bus, VSHCT = 1052 us, VBUSCT = 150 us, AVG = 16
 *   -> one averaged sample every ~19.2 ms (~52 Hz). Each sample is the AVERAGE
 *      of the whole 19 ms window, so short current spikes are integrated, not missed.
 */
#include <Wire.h>

// ---------- user settings ----------
static const uint8_t  INA_ADDR   = 0x40;
static const int      PIN_SDA    = 8;
static const int      PIN_SCL    = 9;
static const float    R_SHUNT    = 0.015f;   // ohm  (15 mOhm)
static const bool     ADCRANGE_1 = true;     // true: +/-40.96 mV range, 1.25 uV LSB (best for < 2.7 A at 15 mOhm)
// ADC_CONFIG = MODE(0xB: cont. shunt+bus) | VBUSCT(2:150us) | VSHCT(5:1052us) | VTCT(0) | AVG(2:16)
static const uint16_t ADC_CONFIG = (0xB << 12) | (2 << 9) | (5 << 6) | (0 << 3) | 2;
// -----------------------------------

// INA238 registers
enum : uint8_t {
  REG_CONFIG = 0x00, REG_ADC_CONFIG = 0x01, REG_SHUNT_CAL = 0x02,
  REG_VSHUNT = 0x04, REG_VBUS = 0x05, REG_DIETEMP = 0x06,
  REG_DIAG_ALRT = 0x0B, REG_MANUF_ID = 0x3E, REG_DEVICE_ID = 0x3F
};

static bool wr16(uint8_t reg, uint16_t v) {
  Wire.beginTransmission(INA_ADDR);
  Wire.write(reg); Wire.write(v >> 8); Wire.write(v & 0xFF);
  return Wire.endTransmission() == 0;
}

static bool rd16(uint8_t reg, uint16_t &v) {
  Wire.beginTransmission(INA_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)INA_ADDR, 2) != 2) return false;
  v = ((uint16_t)Wire.read() << 8) | Wire.read();
  return true;
}

static const float VSH_LSB_V  = ADCRANGE_1 ? 1.25e-6f : 5.0e-6f;
static const float VBUS_LSB_V = 3.125e-3f;

void setup() {
  Serial.begin(115200);
  delay(1500);
  Wire.begin(PIN_SDA, PIN_SCL, 400000);

  uint16_t man = 0, dev = 0;
  if (!rd16(REG_MANUF_ID, man) || !rd16(REG_DEVICE_ID, dev)) {
    Serial.println("# ERROR: INA238 not found at 0x40 - check SDA/SCL/VS/GND wiring");
    while (true) delay(1000);
  }
  Serial.printf("# INA238 found: manufacturer 0x%04X (expect 0x5449), device 0x%04X (expect 0x238x)\n", man, dev);

  wr16(REG_CONFIG, 0x8000);                 // reset
  delay(5);
  wr16(REG_CONFIG, ADCRANGE_1 ? 0x0010 : 0x0000);
  wr16(REG_SHUNT_CAL, 0);                   // we compute current from VSHUNT ourselves
  wr16(REG_ADC_CONFIG, ADC_CONFIG);

  Serial.printf("# R_shunt=%.4f ohm, ADCRANGE=%d, ADC_CONFIG=0x%04X (VSHCT 1052us, VBUSCT 150us, AVG 16)\n",
                R_SHUNT, ADCRANGE_1 ? 1 : 0, ADC_CONFIG);
  Serial.println("t_us,vbus_V,vshunt_uV,current_mA,power_mW");
}

void loop() {
  // wait for "conversion ready" (CNVRF, bit 1 of DIAG_ALRT); reading the register clears it
  uint16_t diag = 0;
  if (!rd16(REG_DIAG_ALRT, diag) || !(diag & 0x0002)) { delayMicroseconds(500); return; }

  uint16_t rsh = 0, rbus = 0;
  if (!rd16(REG_VSHUNT, rsh) || !rd16(REG_VBUS, rbus)) { Serial.println("# i2c read error"); return; }

  uint32_t t   = micros();
  float vsh    = (int16_t)rsh * VSH_LSB_V;          // V (signed)
  float vbus   = rbus * VBUS_LSB_V;                  // V
  float i_mA   = vsh / R_SHUNT * 1000.0f;
  float p_mW   = vbus * i_mA;
  Serial.printf("%lu,%.5f,%.2f,%.4f,%.4f\n", (unsigned long)t, vbus, vsh * 1e6f, i_mA, p_mW);
}
