#include "Stopwatch.h"

std::chrono::steady_clock::time_point Stopwatch::startTime = std::chrono::steady_clock::now();
bool Stopwatch::running = false;

float Stopwatch::Start()
{
	if (!running)
	{
		running = true;
		startTime = std::chrono::steady_clock::now();
		return 0;
	}
	else
	{
		printf("Stopwatch started while still running!\n");
		float returnVal = std::chrono::duration<float, std::chrono::seconds::period>(std::chrono::steady_clock::now() - startTime).count();
		startTime = std::chrono::steady_clock::now();
		return returnVal;
	}
}

/// <summary>
/// Return time since start.
/// </summary>
/// <returns></returns>
float Stopwatch::Stop()
{
	if (!running)
	{
		printf("Stopwatch stopped while not running!\n");
		return 0;
	}
	else
	{
		return std::chrono::duration<float, std::chrono::seconds::period>(std::chrono::steady_clock::now() - startTime).count();
	}
}