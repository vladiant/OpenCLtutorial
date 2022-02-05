// https://www.youtube.com/watch?v=fgg4YuA1Juk
#include <iostream>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 100
#include <CL/cl.hpp>

int main() {
  std::vector<cl::Platform> platforms;

  cl::Platform::get(&platforms);

  if (platforms.empty()) {
    std::cerr << "No OpenCL platform found!\n";
    return EXIT_FAILURE;
  }

  std::cout << "OpenCL platforms found: " << platforms.size() << '\n';

  auto platform = platforms.front();

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

  if (devices.empty()) {
    std::cerr << "No OpenCL device found!\n";
    return EXIT_FAILURE;
  }

  std::cout << "OpenCL devices found: " << devices.size() << '\n';

  auto device = devices.front();
  auto vendor = device.getInfo<CL_DEVICE_VENDOR>();
  auto version = device.getInfo<CL_DEVICE_VERSION>();

  std::cout << "First device vendor: " << vendor << '\n';
  std::cout << "First device version: " << version << '\n';

  return EXIT_SUCCESS;
}
