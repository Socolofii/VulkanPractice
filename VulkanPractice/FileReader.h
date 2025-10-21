#pragma once
#include <fstream>
#include <vector>

inline std::vector<char> readFile(const std::string& filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open file!\n");
	}


	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);


	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	return buffer;
}

inline unsigned char* loadTGA(const std::string& filename, uint16_t* texWidth, uint16_t* texHeight)
{
	std::vector<char> vec = readFile(filename);
	unsigned char* fileData = reinterpret_cast<unsigned char*>(vec.data());
	std::vector<unsigned char> pixelData;

	*texWidth = (static_cast<uint16_t>(fileData[13]) << 8) | fileData[12];
	*texHeight = (static_cast<uint16_t>(fileData[15]) << 8) | fileData[14];
	unsigned char* returnPixelData = nullptr;
	
	//printf("")
	printf("HI.\n");
	printf("ImageIdLength : %d, ColorMapType : %d, ImageTypeCode : %d, PixelSize : %d, BottomLeftX : %d, BottomLeftY : %d\n",
		fileData[0],
		fileData[1],
		fileData[2],
		fileData[16],
		(static_cast<uint16_t>(fileData[9]) << 8) | fileData[8],
		(static_cast<uint16_t>(fileData[11]) << 8) | fileData[10]);

	if (fileData[2] == static_cast<uint8_t>(2))
	{
		//unsigned char* startOffset = ;

		if (fileData[16] == static_cast<uint8_t>(32)) // 4 bytes per pixel
		{
			uint32_t numPixels = *texWidth * *texHeight;
			returnPixelData = new unsigned char[numPixels * 4];
			unsigned char* currentOffset = fileData + 18 + fileData[0];
			unsigned char* returnDataOffset = returnPixelData;
			for (uint32_t i = 0; i < numPixels; i++)
			{
				returnDataOffset[0] = currentOffset[2];
				returnDataOffset[1] = currentOffset[1];
				returnDataOffset[2] = currentOffset[0];
				returnDataOffset[3] = currentOffset[3];

				returnDataOffset += 4;
				currentOffset += 4;
			}
		}
	}

	return returnPixelData;
}