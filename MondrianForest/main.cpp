#include "test/MondrianTest.h"

int main(int argc, char** argv)
{
	MondrianTest test;
	test.trainAndTest("usps_train.txt", "usps_test.txt");

	return 0;
}
