#pragma once

#include <stdint.h>

#define MOTOR1 1
#define MOTOR2 2
#define MOTOR_FORWARD 1
#define MOTOR_BACKWARD 0
#ifdef __cplusplus
extern "C"
{
#endif

    /*
     * @brief Initialization of the Stepper Motors
     */
    int motor_controller_init();
    int set_output(int motor, int direction);

#ifdef __cplusplus
}
#endif