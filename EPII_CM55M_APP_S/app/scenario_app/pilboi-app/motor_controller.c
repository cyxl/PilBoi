#include "motor_controller.h"
#include "hx_drv_gpio.h"
#include "hx_drv_timer.h"
#include "WE2_device_addr.h"
#include "hx_drv_scu_export.h"

#include <stdio.h>

uint8_t g_val;

int motor_controller_init()
{
  // /// The pin of GPIO0 is defined by the user application.
  // hx_drv_scu_set_PB6_pinmux(SCU_PB6_PINMUX_GPIO0_2, 1);

  // Configure PB2(D9) as GPIO13
  hx_drv_gpio_set_output(GPIO13, GPIO_OUT_LOW);
  hx_drv_scu_set_PB2_pinmux(SCU_PB2_PINMUX_GPIO13, 1);

  // Configure PB3(D10) as GPIO14
  hx_drv_gpio_set_output(GPIO14, GPIO_OUT_LOW);
  hx_drv_scu_set_PB3_pinmux(SCU_PB3_PINMUX_GPIO14, 1);

  /// Initialize GPIO_GROUP_0
  hx_drv_gpio_init(GPIO_GROUP_0, HX_GPIO_GROUP_0_BASE);
}

int set_output(int motor, int direction)
{
  printf("In set output motor = %d direction = %d", motor, direction);
  GPIO_INDEX_E gpioidxA;
  GPIO_INDEX_E gpioidxB;
  if (motor == MOTOR1)
  {
    gpioidxA = GPIO13;
    gpioidxB = GPIO14;
  }
  else
  {
    // TODO
  }
  if (!direction)
  {
    hx_drv_gpio_set_out_value(gpioidxA, GPIO_OUT_HIGH);
    hx_drv_gpio_set_out_value(gpioidxB, GPIO_OUT_LOW);
  }
  else
  {
    hx_drv_gpio_set_out_value(gpioidxA, GPIO_OUT_LOW);
    hx_drv_gpio_set_out_value(gpioidxB, GPIO_OUT_HIGH);
  }
}
