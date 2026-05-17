/*
 * =============================================================================
 * Water Quality CNN – ESP32-S3 TFLite Micro Inference Example
 * =============================================================================
 *
 * This sketch shows how to:
 *   1. Load the quantized TFLite model stored as a C byte array
 *   2. Read sensor values (pH, Turbidity, EC, Temperature)
 *   3. Normalise inputs using the saved MinMax scaler parameters
 *   4. Run inference with TFLite Micro
 *   5. Interpret the output risk score
 *
 * REQUIREMENTS:
 *   • Board        : ESP32-S3 (any variant with ≥4 MB flash)
 *   • Arduino libs : EloquentTinyML or TFLite_ESP32 (EloquentArduino)
 *   • Convert model: Run  python train.py  to generate .tflite files
 *   • Convert to C : xxd -i models/water_quality_cnn_quantized.tflite
 *                         > esp32/model_data.h
 *
 * MEMORY:
 *   Tensor arena   : 8 KB is usually sufficient for this small model
 *   Flash          : < 64 KB for quantized model
 * =============================================================================
 */

#include <Arduino.h>

// ── TFLite Micro includes ────────────────────────────────────────────────────
// Install "EloquentTinyML" from Arduino Library Manager
#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>

// ── Generated model byte array ───────────────────────────────────────────────
// Generate with: xxd -i models/water_quality_cnn_quantized.tflite > esp32/model_data.h
#include "model_data.h"   // defines: unsigned char model_data[]; unsigned int model_data_len;

// ── Configuration ────────────────────────────────────────────────────────────
#define N_INPUTS      4
#define N_OUTPUTS     1
#define TENSOR_ARENA  8 * 1024    // 8 KB – increase if inference fails

Eloquent::TinyML::TensorFlow::TensorFlow<N_INPUTS, N_OUTPUTS, TENSOR_ARENA> tf;

// ── MinMax Scaler parameters (copy from models/feature_scaler.npy output) ───
//    These values are printed during training; update them accordingly.
//    Order: pH, Turbidity, EC, Temperature
const float SCALER_MIN[4]   = { 3.50f,  0.10f,   50.0f,   5.0f };
const float SCALER_SCALE[4] = { 0.1429f, 0.0200f, 0.000513f, 0.0250f };
//                              1/(10.5-3.5), 1/50, 1/1950, 1/40

// ── Risk level labels ────────────────────────────────────────────────────────
const char* riskLabel(float score) {
  if (score < 0.20f) return "SAFE";
  if (score < 0.40f) return "LOW RISK";
  if (score < 0.60f) return "MODERATE";
  if (score < 0.80f) return "HIGH RISK";
  return "CRITICAL";
}

// ── Normalise a single feature value ─────────────────────────────────────────
float normalise(float value, int idx) {
  return (value - SCALER_MIN[idx]) * SCALER_SCALE[idx];
}


// =============================================================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== Water Quality CNN – ESP32-S3 ===");

  // Initialise TFLite interpreter
  if (!tf.begin(model_data).isOk()) {
    Serial.print("TFLite init error: ");
    Serial.println(tf.exception.toString());
    while (true) delay(1000);   // halt
  }

  Serial.println("Model loaded successfully.");
  Serial.printf("Input  tensor shape: [1, %d, 1]\n", N_INPUTS);
  Serial.printf("Tensor arena size  : %d bytes\n", TENSOR_ARENA);
}


// =============================================================================
void loop() {
  // ── Replace these with real ADC / I2C sensor readings ─────────────────────
  float pH          = 7.1f;    // pH sensor  (e.g. DFRobot SEN0161)
  float turbidity   = 3.2f;    // NTU        (e.g. DFRobot SEN0189)
  float ec          = 210.0f;  // µS/cm      (e.g. DFRobot SEN0451)
  float temperature = 27.0f;   // °C         (e.g. DS18B20 one-wire)

  // ── Normalise inputs ───────────────────────────────────────────────────────
  float input[N_INPUTS] = {
    normalise(pH,          0),
    normalise(turbidity,   1),
    normalise(ec,          2),
    normalise(temperature, 3),
  };

  // Clamp to [0, 1] (handles out-of-range sensor values)
  for (int i = 0; i < N_INPUTS; i++) {
    input[i] = constrain(input[i], 0.0f, 1.0f);
  }

  // ── Run inference ──────────────────────────────────────────────────────────
  float output[N_OUTPUTS];
  if (!tf.predict(input, output).isOk()) {
    Serial.print("Inference error: ");
    Serial.println(tf.exception.toString());
    delay(2000);
    return;
  }

  float riskScore = output[0];   // value ∈ [0, 1]

  // ── Print results ──────────────────────────────────────────────────────────
  Serial.println("\n--- Water Quality Reading ---");
  Serial.printf("pH          : %.2f\n", pH);
  Serial.printf("Turbidity   : %.2f NTU\n", turbidity);
  Serial.printf("EC          : %.1f µS/cm\n", ec);
  Serial.printf("Temperature : %.1f °C\n", temperature);
  Serial.printf("Risk Score  : %.4f\n", riskScore);
  Serial.printf("Risk Level  : %s\n", riskLabel(riskScore));

  // ── Optional: drive an LED or OLED display based on risk level ────────────
  // if (riskScore > 0.6) digitalWrite(LED_RED, HIGH);
  // else                 digitalWrite(LED_GREEN, HIGH);

  delay(5000);   // sample every 5 seconds
}

/*
 * =============================================================================
 * DEPLOYMENT NOTES
 * =============================================================================
 *
 * 1. FLASH THE MODEL
 *    xxd -i models/water_quality_cnn_quantized.tflite > esp32/model_data.h
 *    The header defines `model_data[]` and `model_data_len`.
 *
 * 2. MEMORY OPTIMISATION
 *    • Use INT8 quantized model (4× smaller than float32)
 *    • Reduce TENSOR_ARENA until inference fails, then add 1 KB margin
 *    • Avoid Serial.printf in production; use Serial.write for speed
 *
 * 3. SENSOR WIRING (example)
 *    pH sensor        → ADC GPIO 1 (10-bit ADC, 0–3.3V)
 *    Turbidity sensor → ADC GPIO 2
 *    EC sensor        → I2C SDA/SCL (GPIO 8/9)
 *    DS18B20 temp     → GPIO 4 (one-wire, with 4.7kΩ pull-up)
 *
 * 4. POWER OPTIMISATION
 *    Use esp_light_sleep_start() between readings if battery-powered.
 *    Duty cycle: 1 s active + 4 s sleep → ~80% power saving.
 *
 * 5. CALIBRATION
 *    Update SCALER_MIN and SCALER_SCALE after retraining the model.
 *    These exact values are printed to console when train.py finishes.
 * =============================================================================
 */
