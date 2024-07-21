/*
 * cvapp.cpp
 *
 *  Created on: 2018�~12��4��
 *      Author: 902452
 */

#include <cstdio>
#include <forward_list>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "WE2_device.h"
#include "board.h"
#include "cvapp_yolov8n_ob.h"
#include "cisdp_sensor.h"
#include "uln2003.h"

#include "WE2_core.h"

#include "ethosu_driver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "img_proc_helium.h"
#include "yolo_postprocessing.h"

#include "xprintf.h"
#include "spi_master_protocol.h"
#include "cisdp_cfg.h"
#include "memory_manage.h"
extern "C"
{
  #include "pilboi_utils.h"
}
#include <send_result.h>

#define CHANGE_YOLOV8_OB_OUPUT_SHAPE 1

#define INPUT_IMAGE_CHANNELS 3

#if 1
#define YOLOV8_OB_INPUT_TENSOR_WIDTH 256
#define YOLOV8_OB_INPUT_TENSOR_HEIGHT 256
#define YOLOV8_OB_INPUT_TENSOR_CHANNEL INPUT_IMAGE_CHANNELS
#else
#define YOLOV8_OB_INPUT_TENSOR_WIDTH 224
#define YOLOV8_OB_INPUT_TENSOR_HEIGHT 224
#define YOLOV8_OB_INPUT_TENSOR_CHANNEL INPUT_IMAGE_CHANNELS
#endif

#define YOLOV8N_OB_DBG_APP_LOG 1

// #define EACH_STEP_TICK
#define TOTAL_STEP_TICK
#define YOLOV8_POST_EACH_STEP_TICK 0
uint32_t systick_1, systick_2;
uint32_t loop_cnt_1, loop_cnt_2;
#define CPU_CLK 0xffffff + 1
static uint32_t capture_image_tick = 0;
#ifdef TRUSTZONE_SEC
#define U55_BASE BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#ifndef TRUSTZONE
#define U55_BASE BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#define U55_BASE BASE_ADDR_APB_U55_CTRL
#endif
#endif


using namespace std;

namespace
{

	constexpr int tensor_arena_size = 1053 * 1024;

	static uint32_t tensor_arena = 0;

	struct ethosu_driver ethosu_drv; /* Default Ethos-U device driver */
	tflite::MicroInterpreter *yolov8n_ob_int_ptr = nullptr;
	TfLiteTensor *yolov8n_ob_input, *yolov8n_ob_output, *yolov8n_ob_output2;
};

static void _arm_npu_irq_handler(void)
{
	/* Call the default interrupt handler from the NPU driver */
	ethosu_irq_handler(&ethosu_drv);
}

/**
 * @brief  Initialises the NPU IRQ
 **/
static void _arm_npu_irq_init(void)
{
	const IRQn_Type ethosu_irqnum = (IRQn_Type)U55_IRQn;

	/* Register the EthosU IRQ handler in our vector table.
	 * Note, this handler comes from the EthosU driver */
	EPII_NVIC_SetVector(ethosu_irqnum, (uint32_t)_arm_npu_irq_handler);

	/* Enable the IRQ */
	NVIC_EnableIRQ(ethosu_irqnum);
}

static int _arm_npu_init(bool security_enable, bool privilege_enable)
{
	int err = 0;

	/* Initialise the IRQ */
	_arm_npu_irq_init();

	/* Initialise Ethos-U55 device */
#if TFLM2209_U55TAG2205
	const void *ethosu_base_address = (void *)(U55_BASE);
#else
	void *const ethosu_base_address = (void *)(U55_BASE);
#endif

	if (0 != (err = ethosu_init(
				  &ethosu_drv,		   /* Ethos-U driver device pointer */
				  ethosu_base_address, /* Ethos-U NPU's base address. */
				  NULL,				   /* Pointer to fast mem area - NULL for U55. */
				  0,				   /* Fast mem region size. */
				  security_enable,	   /* Security enable. */
				  privilege_enable)))
	{ /* Privilege enable. */
		xprintf("failed to initalise Ethos-U device\n");
		return err;
	}

	xprintf("Ethos-U55 device initialised\n");

	return 0;
}

int cv_yolov8n_ob_init(bool security_enable, bool privilege_enable, uint32_t model_addr)
{
	int ercode = 0;

	// set memory allocation to tensor_arena
	tensor_arena = mm_reserve_align(tensor_arena_size, 0x20); // 1mb
	xprintf("TA[%x]\r\n", tensor_arena);

	if (_arm_npu_init(security_enable, privilege_enable) != 0)
		return -1;

	if (model_addr != 0)
	{
		static const tflite::Model *yolov8n_ob_model = tflite::GetModel((const void *)model_addr);

		if (yolov8n_ob_model->version() != TFLITE_SCHEMA_VERSION)
		{
			xprintf(
				"[ERROR] yolov8n_ob_model's schema version %d is not equal "
				"to supported version %d\n",
				yolov8n_ob_model->version(), TFLITE_SCHEMA_VERSION);
			return -1;
		}
		else
		{
			xprintf("yolov8n_ob model's schema version %d\n", yolov8n_ob_model->version());
		}

		static tflite::MicroErrorReporter yolov8n_ob_micro_error_reporter;
		static tflite::MicroMutableOpResolver<13> yolov8n_ob_op_resolver;

		yolov8n_ob_op_resolver.AddQuantize();
		yolov8n_ob_op_resolver.AddGather();
		yolov8n_ob_op_resolver.AddTranspose();
		yolov8n_ob_op_resolver.AddConv2D();
		yolov8n_ob_op_resolver.AddDepthwiseConv2D();
		yolov8n_ob_op_resolver.AddAdd();
		yolov8n_ob_op_resolver.AddRelu6();
		yolov8n_ob_op_resolver.AddResizeNearestNeighbor();
		yolov8n_ob_op_resolver.AddReshape();
		yolov8n_ob_op_resolver.AddConcatenation();
		yolov8n_ob_op_resolver.AddLogistic();
		yolov8n_ob_op_resolver.AddPadV2();
		if (kTfLiteOk != yolov8n_ob_op_resolver.AddEthosU())
		{
			xprintf("Failed to add Arm NPU support to op resolver.");
			return false;
		}
#if TFLM2209_U55TAG2205
		static tflite::MicroInterpreter yolov8n_ob_static_interpreter(yolov8n_ob_model, yolov8n_ob_op_resolver,
																	  (uint8_t *)tensor_arena, tensor_arena_size, &yolov8n_ob_micro_error_reporter);
#else
		static tflite::MicroInterpreter yolov8n_ob_static_interpreter(yolov8n_ob_model, yolov8n_ob_op_resolver,
																	  (uint8_t *)tensor_arena, tensor_arena_size);
#endif

		if (yolov8n_ob_static_interpreter.AllocateTensors() != kTfLiteOk)
		{
			return false;
		}
		yolov8n_ob_int_ptr = &yolov8n_ob_static_interpreter;
		yolov8n_ob_input = yolov8n_ob_static_interpreter.input(0);
		yolov8n_ob_output = yolov8n_ob_static_interpreter.output(0);
#if CHANGE_YOLOV8_OB_OUPUT_SHAPE
		yolov8n_ob_output2 = yolov8n_ob_static_interpreter.output(1);
#endif
	}

	xprintf("initial done\n");
	return ercode;
}

#define INDEX_X 0
#define INDEX_Y 1
#define INDEX_W 2
#define INDEX_H 3
#define INDEX_S 4
#define INDEX_T 5
#define EL_CLIP(x, a, b)           ((x) < (a) ? (a) : ((x) > (b) ? (b) : (x)))

static void yolov8_ob_post_processing(tflite::MicroInterpreter *static_interpreter, float modelScoreThreshold, float modelNMSThreshold, struct_yolov8_ob_algoResult *alg,uint8_t* num)
{
	uint32_t img_w = app_get_raw_width();
	uint32_t img_h = app_get_raw_height();
	float scale_factor_w = (float)img_w / (float)YOLOV8_OB_INPUT_TENSOR_WIDTH;
	float scale_factor_h = (float)img_h / (float)YOLOV8_OB_INPUT_TENSOR_HEIGHT;
	std::forward_list<box> _results;
	// get output
	TfLiteTensor *output = static_interpreter->output(0);
	TfLiteTensor *input = static_interpreter->input(0);
	auto *data{static_cast<int8_t *>(output->data.int8)};

	auto width{input->dims->data[1]};
	auto height{input->dims->data[2]};

	float scale{((TfLiteAffineQuantization *)(output->quantization.params))->scale->data[0]};
	bool rescale{scale < 0.1f ? true : false};

	int32_t zero_point{((TfLiteAffineQuantization *)(output->quantization.params))->zero_point->data[0]};

	auto num_record{output->dims->data[1]};
	auto num_element{output->dims->data[2]};
	auto num_class{static_cast<uint8_t>(num_element - 5)};
	xprintf("Tensor width %d height %d num classes %d\n",width,height,num_class);

	float score_threshold{modelScoreThreshold};
	float iou_threshold{modelNMSThreshold};

	// parse output
	for (decltype(num_record) i{0}; i < num_record; ++i)
	{
		auto idx{i * num_element};
		auto score{static_cast<decltype(scale)>(data[idx + INDEX_S] - zero_point) * scale};
		score = rescale ? score * 100.f : score;
		if (score > score_threshold)
		{
			box box{
				.x = 0,
				.y = 0,
				.w = 0,
				.h = 0,
				.score = static_cast<decltype(box::score)>(score),
				.target = 0,
			};

			// get box target
			int8_t max{-128};
			for (decltype(num_class) t{0}; t < num_class; ++t)
			{
				if (max < data[idx + INDEX_T + t])
				{
					max = data[idx + INDEX_T + t];
					box.target = t;
				}
			}

			// get box position, int8_t - int32_t (narrowing)
			auto x{((data[idx + INDEX_X] - zero_point) * scale)};
			auto y{((data[idx + INDEX_Y] - zero_point) * scale)};
			auto w{((data[idx + INDEX_W] - zero_point) * scale)};
			auto h{((data[idx + INDEX_H] - zero_point) * scale)};

			if (rescale)
			{
				x = x * width;
				y = y * height;
				w = w * width;
				h = h * height;
			}

			box.x = EL_CLIP(x, 0, width) * scale_factor_w;
			box.y = EL_CLIP(y, 0, height) * scale_factor_h;
			box.w = EL_CLIP(w, 0, width) * scale_factor_w;
			box.h = EL_CLIP(h, 0, height) * scale_factor_h;

			_results.emplace_front(std::move(box));
		}
	}
	el_nms(_results, iou_threshold, score_threshold, false, true);

	_results.sort([](const box &a, const box &b)
				  { return a.x < b.x; });


	*num = 0;
	int lnum=0;
	for (auto& r : _results){
		if (lnum >= MAX_TRACKED_YOLOV8_ALGO_RES) {break;}
		printf("Target [%d] Box [%d,%d,%d,%d]\n",r.target,r.x,r.y,r.w,r.h);
		printfloat(r.score);
		alg->obr[lnum].bbox.x = r.x;
		alg->obr[lnum].bbox.y = r.y;
		alg->obr[lnum].bbox.width = r.w;
		alg->obr[lnum].bbox.height = r.h;
		alg->obr[lnum].class_idx = r.target;
		alg->obr[lnum].confidence = r.score/100.;
		printfloat(alg->obr[lnum].confidence);
		lnum++;
	}
	*num = lnum;
	printf ("num is %d \n",*num);

}

int cv_yolov8n_ob_run(struct_yolov8_ob_algoResult *algoresult_yolov8n_ob,uint8_t* num)
{
	int ercode = 0;
	float w_scale;
	float h_scale;
	uint32_t img_w = app_get_raw_width();
	uint32_t img_h = app_get_raw_height();
	uint32_t ch = app_get_raw_channels();
	uint32_t raw_addr = app_get_raw_addr();
	std::forward_list<el_box_t> el_algo;

#if YOLOV8N_OB_DBG_APP_LOG
	xprintf("raw info: w[%d] h[%d] ch[%d] addr[%x]\n", img_w, img_h, ch, raw_addr);
#endif

	if (yolov8n_ob_int_ptr != nullptr)
	{
#ifdef TOTAL_STEP_TICK
		SystemGetTick(&systick_1, &loop_cnt_1);
#endif
#ifdef EACH_STEP_TICK
		SystemGetTick(&systick_1, &loop_cnt_1);
#endif
		// get image from sensor and resize
		w_scale = (float)(img_w - 1) / (YOLOV8_OB_INPUT_TENSOR_WIDTH - 1);
		h_scale = (float)(img_h - 1) / (YOLOV8_OB_INPUT_TENSOR_HEIGHT - 1);

		hx_lib_image_resize_BGR8U3C_to_RGB24_helium((uint8_t *)raw_addr, (uint8_t *)yolov8n_ob_input->data.data,
													img_w, img_h, ch,
													YOLOV8_OB_INPUT_TENSOR_WIDTH, YOLOV8_OB_INPUT_TENSOR_HEIGHT, w_scale, h_scale);
#ifdef EACH_STEP_TICK
		SystemGetTick(&systick_2, &loop_cnt_2);
		dbg_printf(DBG_LESS_INFO, "Tick for resize image BGR8U3C_to_RGB24_helium for yolov8 OB:[%d]\r\n", (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2));
#endif

#ifdef EACH_STEP_TICK
		SystemGetTick(&systick_1, &loop_cnt_1);
#endif

		// //uint8 to int8
		for (size_t i = 0; i < yolov8n_ob_input->bytes; ++i)
		{
			*((int8_t *)yolov8n_ob_input->data.data + i) = *((int8_t *)yolov8n_ob_input->data.data + i) - 128;
		}

#ifdef EACH_STEP_TICK
		SystemGetTick(&systick_2, &loop_cnt_2);
		dbg_printf(DBG_LESS_INFO, "Tick for Invoke for uint8toint8 for YOLOV8_OB:[%d]\r\n\n", (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2));
#endif

#ifdef EACH_STEP_TICK
		SystemGetTick(&systick_1, &loop_cnt_1);
#endif
		TfLiteStatus invoke_status = yolov8n_ob_int_ptr->Invoke();

#ifdef EACH_STEP_TICK
		SystemGetTick(&systick_2, &loop_cnt_2);
#endif
		if (invoke_status != kTfLiteOk)
		{
			xprintf("yolov8 object detect invoke fail\n");
			return -1;
		}
		else
		{
#if YOLOV8N_OB_DBG_APP_LOG
			xprintf("yolov8 object detect  invoke pass\n");
#endif
		}
#ifdef EACH_STEP_TICK
		dbg_printf(DBG_LESS_INFO, "Tick for Invoke for YOLOV8_OB:[%d]\r\n\n", (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2));
#endif

#ifdef EACH_STEP_TICK
		SystemGetTick(&systick_1, &loop_cnt_1);
#endif
		// retrieve output data
		// bws yolov8_ob_post_processing(yolov8n_ob_int_ptr,0.25, 0.45, algoresult_yolov8n_ob,el_algo);
		// yolov8_ob_post_processing(yolov8n_ob_int_ptr,0.25, 0.45, algoresult_yolov8n_ob);
		yolov8_ob_post_processing(yolov8n_ob_int_ptr, .7, .5, algoresult_yolov8n_ob,num);
#ifdef EACH_STEP_TICK
		SystemGetTick(&systick_2, &loop_cnt_2);
		dbg_printf(DBG_LESS_INFO, "Tick for Invoke for YOLOV8_OB_post_processing:[%d]\r\n\n", (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2));
#endif
#if YOLOV8N_OB_DBG_APP_LOG
		xprintf("yolov8_ob_post_processing done\r\n");
#endif
#ifdef TOTAL_STEP_TICK
		SystemGetTick(&systick_2, &loop_cnt_2);
		// dbg_printf(DBG_LESS_INFO,"Tick for TOTAL YOLOV8 OB:[%d]\r\n",(loop_cnt_2-loop_cnt_1)*CPU_CLK+(systick_1-systick_2));
#endif
	}

#ifdef UART_SEND_ALOGO_RESEULT
	algoresult_yolov8n_ob->algo_tick = (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2) + capture_image_tick;
	uint32_t judge_case_data;
	uint32_t g_trans_type;
	hx_drv_swreg_aon_get_appused1(&judge_case_data);
	g_trans_type = (judge_case_data >> 16);
	if (g_trans_type == 0 || g_trans_type == 2) // transfer type is (UART) or (UART & SPI)
	{
		el_img_t temp_el_jpg_img = el_img_t{};
		temp_el_jpg_img.data = (uint8_t *)app_get_jpeg_addr();
		temp_el_jpg_img.size = app_get_jpeg_sz();
		temp_el_jpg_img.width = app_get_raw_width();
		temp_el_jpg_img.height = app_get_raw_height();
		temp_el_jpg_img.format = EL_PIXEL_FORMAT_JPEG;
		temp_el_jpg_img.rotate = EL_PIXEL_ROTATE_0;

		send_device_id();
		// event_reply(concat_strings(", ", box_results_2_json_str(el_algo), ", ", img_2_json_str(&temp_el_jpg_img)));
		event_reply(concat_strings(", ", algo_tick_2_json_str(algoresult_yolov8n_ob->algo_tick), ", ", box_results_2_json_str(el_algo), ", ", img_2_json_str(&temp_el_jpg_img)));
	}
	set_model_change_by_uart();
#endif

	SystemGetTick(&systick_1, &loop_cnt_1);

	SystemGetTick(&systick_2, &loop_cnt_2);
	capture_image_tick = (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2);
	return ercode;
}

int cv_yolov8n_ob_deinit()
{

	return 0;
}
