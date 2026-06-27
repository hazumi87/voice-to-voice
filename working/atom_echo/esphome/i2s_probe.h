#pragma once
// Raw ESP-IDF I2S mic sweep. Owns I2S0 directly (no ESPHome i2s_audio component).
// Tries every relevant slot/bit-width config against the ICS43434 wired on
// BCLK=G19, WS=G33, DIN=G23 and logs the max sample amplitude for each.
// No speaking needed -- a live, powered mic shows maxabs well above 0 from ambient
// noise alone; a dead/unpowered/disconnected line reads flat 0 for every config.
#include "driver/i2s_std.h"
#include "esp_log.h"

static const char *const PROBE_TAG = "i2sprobe";

static void probe_one(int bits, i2s_std_slot_mask_t slot, const char *slotname) {
  i2s_chan_handle_t rx = nullptr;
  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  if (i2s_new_channel(&chan_cfg, nullptr, &rx) != ESP_OK) {
    ESP_LOGW(PROBE_TAG, "new_channel failed (bits=%d %s)", bits, slotname);
    return;
  }
  i2s_data_bit_width_t bw = (bits == 32) ? I2S_DATA_BIT_WIDTH_32BIT : I2S_DATA_BIT_WIDTH_16BIT;
  i2s_std_config_t cfg = {
      .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(bw, I2S_SLOT_MODE_MONO),
      .gpio_cfg = {
          .mclk = I2S_GPIO_UNUSED,
          .bclk = GPIO_NUM_19,
          .ws = GPIO_NUM_33,
          .dout = I2S_GPIO_UNUSED,
          .din = GPIO_NUM_23,
          .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
      },
  };
  cfg.slot_cfg.slot_mask = slot;
  if (i2s_channel_init_std_mode(rx, &cfg) != ESP_OK) {
    ESP_LOGW(PROBE_TAG, "init_std failed (bits=%d %s)", bits, slotname);
    i2s_del_channel(rx);
    return;
  }
  i2s_channel_enable(rx);

  static uint8_t buf[4096];
  const int bytes_per = (bits == 32) ? 4 : 2;
  long maxabs = 0, nonzero = 0, count = 0;
  size_t br = 0;
  for (int it = 0; it < 6; it++) {
    if (i2s_channel_read(rx, buf, sizeof(buf), &br, 200) != ESP_OK) continue;
    if (it == 0) continue;  // discard first read (settling)
    int n = br / bytes_per;
    for (int i = 0; i < n; i++) {
      long v;
      if (bits == 32) {
        v = ((int32_t *) buf)[i] >> 8;  // 24-bit datum sits in top of the 32-bit slot
      } else {
        v = ((int16_t *) buf)[i];
      }
      if (v < 0) v = -v;
      if (v > maxabs) maxabs = v;
      if (v != 0) nonzero++;
      count++;
    }
  }
  ESP_LOGI(PROBE_TAG, "bits=%2d slot=%-5s  maxabs=%-10ld nonzero=%ld/%ld", bits, slotname, maxabs, nonzero, count);

  i2s_channel_disable(rx);
  i2s_del_channel(rx);
}

static void run_i2s_probe() {
  ESP_LOGI(PROBE_TAG, "==== I2S mic sweep (no speech) — want maxabs >> 0 ====");
  probe_one(32, I2S_STD_SLOT_LEFT, "L32");
  probe_one(32, I2S_STD_SLOT_RIGHT, "R32");
  probe_one(16, I2S_STD_SLOT_LEFT, "L16");
  probe_one(16, I2S_STD_SLOT_RIGHT, "R16");
  ESP_LOGI(PROBE_TAG, "==== sweep done ====");
}
