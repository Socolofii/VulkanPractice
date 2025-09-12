#include "MemoryBlock.h"
#include <stdexcept>
#include <cstdio>


void BufferMemoryBlock::allocate(VkDevice logicalDevice, uint32_t allocSize, uint32_t memoryTypeIndex)
{
	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = allocSize;
	allocInfo.memoryTypeIndex = memoryTypeIndex;

	this->memoryTypeIndex = memoryTypeIndex;
	allocatedSize = allocSize;

	if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &memory) != VK_SUCCESS)
	{
		throw "Failed to allocate memory!\n";
	}
}

void BufferMemoryBlock::deallocate(VkDevice logicalDevice)
{
	for (int i = 0; i < MAX_BINDINGS; i++)
	{
		if (bindings[i].buffer != nullptr)
		{
			vkDestroyBuffer(logicalDevice, bindings[i].buffer, nullptr);
			bindings[i].buffer = nullptr;
			numBindingsBound--;
			printf("Destroyed buffer!\n");
		}
	}

	vkFreeMemory(logicalDevice, memory, nullptr);
}

void BufferMemoryBlock::bindBuffer(VkDevice logicalDevice, VkBuffer& buffer, uint32_t size, uint32_t alignment)
{
	uint32_t alignedSize = (size + alignment - 1) & ~(alignment - 1);
	uint32_t totalOffset = 0;
	int32_t targetIndex = -1;
	
	for (int i = 0; i < MAX_BINDINGS; i++)
	{
		if (bindings[i].buffer != nullptr)
		{
			totalOffset += bindings[i].size;
		}
		else if (targetIndex == -1)
		{
			targetIndex = i;
		}
	}

	if (allocatedSize - totalOffset < alignedSize)
		throw std::runtime_error("No more space in memory block!\n");
	
	printf("Bound buffer! alignment : %d,   size : %d,   offset : %d\n", alignment, alignedSize, totalOffset);
	vkBindBufferMemory(logicalDevice, buffer, memory, totalOffset);

	bindings[targetIndex] =
	{
		totalOffset,
		alignedSize,
		buffer
	};

	numBindingsBound++;
}

BufferBinding& BufferMemoryBlock::getNextEmptyBinding()
{
	for (int i = 0; i < MAX_BINDINGS; i++)
	{
		if (bindings[i].buffer == nullptr)
		{
			return bindings[i];
		}
	}
}

void BufferMemoryBlock::setBufferReturnArray()
{
	uint32_t bufferCount = 0;
	for (int i = 0; i < MAX_BINDINGS; i++)
	{
		if (bindings[i].buffer != nullptr)
		{
			bufferReturnArray[bufferCount] = bindings[i].buffer;
			bufferCount++;
			if (bufferCount == numBindingsBound)
			{
				break;
			}
		}
	}
}

// IMAGE IMAGE IMAGE IMAGE IMAGE IMAGE IMAGE IMAGE

void ImageMemoryBlock::allocate(VkDevice logicalDevice, uint32_t allocSize, uint32_t memoryTypeIndex)
{
	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = allocSize;
	allocInfo.memoryTypeIndex = memoryTypeIndex;

	this->memoryTypeIndex = memoryTypeIndex;
	allocatedSize = allocSize;

	if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &memory) != VK_SUCCESS)
	{
		throw "Failed to allocate memory!\n";
	}
}

void ImageMemoryBlock::deallocate(VkDevice logicalDevice)
{
	for (int i = 0; i < MAX_BINDINGS; i++)
	{
		if (bindings[i].image != nullptr)
		{
			vkDestroyImage(logicalDevice, bindings[i].image, nullptr);
			bindings[i].image = nullptr;
			numBindingsBound--;
			printf("Destroyed buffer!\n");
		}
	}

	vkFreeMemory(logicalDevice, memory, nullptr);
}

void ImageMemoryBlock::bindImage(VkDevice logicalDevice, VkImage& image, uint32_t size, uint32_t alignment)
{
	uint32_t alignedSize = (size + alignment - 1) & ~(alignment - 1);
	uint32_t totalOffset = 0;
	int32_t targetIndex = -1;

	for (int i = 0; i < MAX_BINDINGS; i++)
	{
		if (bindings[i].image != nullptr)
		{
			totalOffset += bindings[i].size;
		}
		else if (targetIndex == -1)
		{
			targetIndex = i;
		}
	}

	printf("Allocated size : %d\n", allocatedSize - totalOffset);
	printf("alignedSize : %d\n", alignedSize);
	if (allocatedSize - totalOffset < alignedSize)
		throw std::runtime_error("No more space in memory block!\n");

	printf("Bound image! alignment : %d,   size : %d,   offset : %d\n", alignment, alignedSize, totalOffset);
	vkBindImageMemory(logicalDevice, image, memory, totalOffset);

	bindings[targetIndex] =
	{
		totalOffset,
		alignedSize,
		image
	};

	numBindingsBound++;
}

ImageBinding& ImageMemoryBlock::getNextEmptyBinding()
{
	for (int i = 0; i < MAX_BINDINGS; i++)
	{
		if (bindings[i].image == nullptr)
		{
			return bindings[i];
		}
	}
}

void ImageMemoryBlock::setBufferReturnArray()
{
	uint32_t bufferCount = 0;
	for (int i = 0; i < MAX_BINDINGS; i++)
	{
		if (bindings[i].image != nullptr)
		{
			bufferReturnArray[bufferCount] = bindings[i].image;
			bufferCount++;
			if (bufferCount == numBindingsBound)
			{
				break;
			}
		}
	}
}