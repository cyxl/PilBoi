
#include "pilboi_utils.h"



void printfloat(float f)
{
	const char *tmpSign = (f < 0) ? "-" : "";
	float tmpVal = (f < 0) ? -f : f;

	int tmpInt1 = tmpVal;				  // Get the integer (678).
	float tmpFrac = tmpVal - tmpInt1;	  // Get fraction (0.0123).
	int tmpInt2 = trunc(tmpFrac * 10000); // Turn into integer (123).

	// Print as parts, note that you need 0-padding for fractional bit.
	xprintf("%s%d.%04d\n", tmpSign, tmpInt1, tmpInt2);
}