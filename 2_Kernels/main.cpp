// https://www.youtube.com/watch?v=f8jnAuFMnPY
#include <fstream>
#include <iostream>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 120
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
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

  if (devices.empty()) {
    std::cerr << "No OpenCL GPU device found!\n";
    return EXIT_FAILURE;
  }

  std::cout << "OpenCL devices found: " << devices.size() << '\n';

  auto device = devices.front();

  constexpr auto fileName = "HelloWorld.cl";
  std::ifstream helloWorldFile(fileName);

  std::string src{std::istreambuf_iterator<char>(helloWorldFile),
                  std::istreambuf_iterator<char>()};

  if (src.empty()) {
    std::cerr << " Error reading " << fileName << "\n";
    return EXIT_FAILURE;
  }

  cl::Program::Sources sources(1,
                               std::make_pair(src.c_str(), src.length() + 1));

  cl::Context context(device);

  cl::Program program(context, sources);
  auto err = program.build("-cl-std=CL1.2");
  if (err != CL_SUCCESS) {
    std::cerr << "Error building OpenCL program: " << err << "\n";
    return EXIT_FAILURE;
  }

  char buf[16] = {};
  cl::Buffer memBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                    sizeof(buf));  // 1.2 CL_MEM_HOST_READ_ONLY
  cl::Kernel kernel(program, "HelloWorld", &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Error creating OpenCL kernel: " << err << "\n";
    return EXIT_FAILURE;
  }
  kernel.setArg(0, memBuf);

  cl::CommandQueue queue(context, device);
  queue.enqueueTask(kernel);
  queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, sizeof(buf), buf);

  std::cout << buf;

  return EXIT_SUCCESS;
}
