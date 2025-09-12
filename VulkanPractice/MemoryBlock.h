#pragma once
#define GLFW_INCLUDE_VULKAN
#include "glfw/glfw3.h"

struct BufferBinding
{
	uint32_t offset = 0;
	uint32_t size = 0;
	VkBuffer buffer = nullptr;
};

struct ImageBinding
{
	uint32_t offset = 0;
	uint32_t size = 0;
	VkImage image = nullptr;
};

struct BufferMemoryBlock
{
	constexpr static uint32_t MAX_BINDINGS = 64;
	VkBuffer bufferReturnArray[MAX_BINDINGS];
	BufferBinding bindings[MAX_BINDINGS];
	uint32_t numBindingsBound = 0;
	uint32_t memoryTypeIndex = 0;
	uint32_t allocatedSize = 0;
	VkDeviceMemory memory = nullptr;

	void allocate(VkDevice logicalDevice, uint32_t allocSize, uint32_t memoryTypeIndex);
	void deallocate(VkDevice logicalDevice);
	void bindBuffer(VkDevice logicalDevice, VkBuffer& buffer, uint32_t size, uint32_t alignment);
	BufferBinding& getNextEmptyBinding();
	void setBufferReturnArray();
};

struct ImageMemoryBlock
{
	constexpr static uint32_t MAX_BINDINGS = 64;
	VkImage bufferReturnArray[MAX_BINDINGS];
	ImageBinding bindings[MAX_BINDINGS];
	uint32_t numBindingsBound = 0;
	uint32_t memoryTypeIndex = 0;
	uint32_t allocatedSize = 0;
	VkDeviceMemory memory = nullptr;

	void allocate(VkDevice logicalDevice, uint32_t allocSize, uint32_t memoryTypeIndex);
	void deallocate(VkDevice logicalDevice);
	void bindImage(VkDevice logicalDevice, VkImage& image, uint32_t size, uint32_t alignment);
	ImageBinding& getNextEmptyBinding();
	void setBufferReturnArray();
};