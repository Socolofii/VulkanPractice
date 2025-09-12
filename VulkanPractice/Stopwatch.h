#pragma once
#include <chrono>

class Stopwatch
{
private:
	static std::chrono::steady_clock::time_point startTime;
	static bool running;
public:
	static float Start();
	static float Stop();
};

