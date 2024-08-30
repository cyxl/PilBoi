
#include "uln2003.h"
#include "hx_drv_gpio.h"
#include "hx_drv_timer.h"
#include "WE2_device_addr.h"
#include "hx_drv_scu_export.h"
#include "pilboi_utils.h"

#include <math.h>
#include <stdio.h>
#define DEGS_TO_STEPS (4096. / 360.)
#define STEPPER_DELAY 350
//#define PIVOTORIGINX 128
#define PIVOTORIGINX 110
//#define PIVOTORIGINY 32
#define PIVOTORIGINY 40
#define CAMCENTERX 128
#define CAMCENTERY 128

float rads_to_degs(float X)
{
    return X * 180. / 3.1415926;
}

GPIO_INDEX_E table_motor_pins[NUM_MOTOR_PINS] = {GPIO13, GPIO14, GPIO15, GPIO0};
GPIO_INDEX_E wheel_motor_pins[NUM_MOTOR_PINS] = {GPIO1, GPIO2, SB_GPIO0, SB_GPIO1};

GPIO_INDEX_E *get_pins(int motor_id)
{
    if (motor_id == TABLE_MOTOR_ID)
        return table_motor_pins;
    if (motor_id == WHEEL_MOTOR_ID)
        return wheel_motor_pins;
    return NULL;
}

uint8_t step_sequence[STEP_SEQ_LEN][NUM_MOTOR_PINS] = {
    {GPIO_OUT_HIGH, GPIO_OUT_LOW, GPIO_OUT_LOW, GPIO_OUT_LOW},
    {GPIO_OUT_HIGH, GPIO_OUT_HIGH, GPIO_OUT_LOW, GPIO_OUT_LOW},
    {GPIO_OUT_LOW, GPIO_OUT_HIGH, GPIO_OUT_LOW, GPIO_OUT_LOW},
    {GPIO_OUT_LOW, GPIO_OUT_HIGH, GPIO_OUT_HIGH, GPIO_OUT_LOW},
    {GPIO_OUT_LOW, GPIO_OUT_LOW, GPIO_OUT_HIGH, GPIO_OUT_LOW},
    {GPIO_OUT_LOW, GPIO_OUT_LOW, GPIO_OUT_HIGH, GPIO_OUT_HIGH},
    {GPIO_OUT_LOW, GPIO_OUT_LOW, GPIO_OUT_LOW, GPIO_OUT_HIGH},
    {GPIO_OUT_HIGH, GPIO_OUT_LOW, GPIO_OUT_LOW, GPIO_OUT_HIGH},
};

uint8_t step_clockwise(uint8_t step_idx, int motor_id)
{
    GPIO_INDEX_E *pins = get_pins(motor_id);

    uint8_t l_step_idx = (step_idx + 1) % STEP_SEQ_LEN;
    for (int i = 0; i < NUM_MOTOR_PINS; i++)
    {
        hx_drv_timer_cm55x_delay_us(STEPPER_DELAY, TIMER_STATE_DC);
        hx_drv_gpio_set_out_value(pins[i], step_sequence[l_step_idx][i]);
    }
    return l_step_idx;
}

uint8_t step_anticlockwise(uint8_t step_idx, int motor_id)
{
    GPIO_INDEX_E *pins = get_pins(motor_id);

    uint8_t l_step_idx = (step_idx - 1) % STEP_SEQ_LEN;
    for (int i = 0; i < NUM_MOTOR_PINS; i++)
    {
        hx_drv_timer_cm55x_delay_us(STEPPER_DELAY, TIMER_STATE_DC);
        hx_drv_gpio_set_out_value(pins[i], step_sequence[l_step_idx][i]);
    }
    return l_step_idx;
}

int init_motors()
{
    int result;
    // hx_drv_gpio_init(GPIO_GROUP_0, HX_GPIO_GROUP_0_BASE);

    // Configure PB2(D9) as GPIO13
    hx_drv_gpio_set_output(GPIO13, GPIO_OUT_LOW);
    hx_drv_scu_set_PB2_pinmux(SCU_PB2_PINMUX_GPIO13, 1);

    // Configure PB3(D10) as GPIO14
    hx_drv_gpio_set_output(GPIO14, GPIO_OUT_LOW);
    hx_drv_scu_set_PB3_pinmux(SCU_PB3_PINMUX_GPIO14, 1);

    // Configure PB4(D8) as GPIO15
    hx_drv_gpio_set_output(GPIO15, GPIO_OUT_LOW);
    hx_drv_scu_set_PB4_pinmux(SCU_PB4_PINMUX_GPIO15, 1);

    // Configure PB6(D6) as GPIO0
    hx_drv_gpio_set_output(GPIO0, GPIO_OUT_LOW);
    hx_drv_scu_set_PB6_pinmux(SCU_PB6_PINMUX_GPIO0_1, 1);

    // Configure PB7(D7) as GPIO1
    hx_drv_gpio_set_output(GPIO1, GPIO_OUT_LOW);
    hx_drv_scu_set_PB7_pinmux(SCU_PB7_PINMUX_GPIO1_1, 1);

    // Configure PB8(D2) as GPIO2
    hx_drv_gpio_set_output(GPIO2, GPIO_OUT_LOW);
    hx_drv_scu_set_PB8_pinmux(SCU_PB8_PINMUX_GPIO2_1, 1);

    // Configure PA2(D5) as SB_GPIO0
    hx_drv_gpio_set_output(SB_GPIO0, GPIO_OUT_LOW);
    hx_drv_scu_set_PA2_pinmux(SCU_PA2_PINMUX_SB_GPIO0, 1);

    // Configure PA3(D4) as SB_GPIO1
    hx_drv_gpio_set_output(SB_GPIO1, GPIO_OUT_LOW);
    hx_drv_scu_set_PA3_pinmux(SCU_PA3_PINMUX_SB_GPIO1, 1);

    return result;
}

uint8_t step_some(uint8_t step_idx, int motor_id, uint8_t clockwise, int num)
{
    uint8_t idx = step_idx;
    for (int i = 0; i < num; i++)
    {
        if (clockwise)
            idx = step_clockwise(idx, motor_id);
        else
            idx = step_anticlockwise(idx, motor_id);
    }
    return idx;
}

uint8_t step_some_deg(uint8_t step_idx, int motor_id, uint8_t clockwise, float degrees)
{
    float degstosteps = degrees * DEGS_TO_STEPS;
    uint16_t intSteps = (degstosteps);
    return step_some(step_idx, motor_id, clockwise, intSteps);
}

float distanceformula(int x1, int y1, int x2, int y2)
{
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

float calc_deg(int16_t pillX, int16_t pillY)
{
    float pilltoorigin = distanceformula(pillX, pillY, PIVOTORIGINX, PIVOTORIGINY);
    float pilltocenter = distanceformula(pillX, pillY, CAMCENTERX, CAMCENTERY);
    float origintocenter = distanceformula(PIVOTORIGINX, PIVOTORIGINY, CAMCENTERX,CAMCENTERY);
    float f = (pow(origintocenter, 2) - pow(pilltocenter, 2) + pow(pilltoorigin, 2)) / (2. * origintocenter * pilltoorigin);
    float origindeg = acos(f);
    // float origindeg = acos(pow(origintocenter, 2) + pow(pilltocenter, 2) - pow(pilltoorigin, 2) / 2 * origintocenter * pilltocenter);
    xprintf("dist");
    printfloat(pilltoorigin);
    printfloat(pilltocenter);
    printfloat(origintocenter);
    printfloat(f);
    printfloat(origindeg);
    origindeg = rads_to_degs(origindeg);
    printfloat(origindeg);

    return origindeg;
}