// https://www.youtube.com/watch?v=N0H0NCoOTUA
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.hpp>

cl::Program createProgram(const std::string& fileName) {
  std::vector<cl::Platform> platforms;

  cl::Platform::get(&platforms);

  if (platforms.empty()) {
    std::cerr << "No OpenCL platform found!\n";
    exit(EXIT_FAILURE);
  }

  std::cout << "OpenCL platforms found: " << platforms.size() << '\n';

  auto platform = platforms.front();

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

  if (devices.empty()) {
    std::cerr << "No OpenCL GPU device found!\n";
    exit(EXIT_FAILURE);
  }

  std::cout << "OpenCL devices found: " << devices.size() << '\n';

  auto device = devices.front();

  std::ifstream helloWorldFile(fileName);

  std::string src{std::istreambuf_iterator<char>(helloWorldFile),
                  std::istreambuf_iterator<char>()};

  if (src.empty()) {
    std::cerr << " Error reading " << fileName << "\n";
    exit(EXIT_FAILURE);
  }

  cl::Program::Sources sources(1,
                               std::make_pair(src.c_str(), src.length() + 1));

  cl::Context context(device);

  cl::Program program(context, sources);
  auto err = program.build("-cl-std=CL1.2");
  if (err != CL_SUCCESS) {
    std::cerr << "Error building OpenCL program: " << err << "\n";
    exit(EXIT_FAILURE);
  }

  return program;
}

int main() {
  constexpr auto fileName = "ProcessArray.cl";
  auto program = createProgram(fileName);

  auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
  auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
  auto& device = devices.front();

  std::vector<int> vec(1024);
  std::fill(vec.begin(), vec.end(), 1);

  cl::Buffer inBuf(
      context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
      sizeof(int) * vec.size(), vec.data());  // 1.2 CL_MEM_HOST_NO_ACCESS
  cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                    sizeof(int) * vec.size());

  cl_int err;
  cl::Kernel kernel(program, "ProcessArray", &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Error creating OpenCL kernel: " << err << "\n";
    return EXIT_FAILURE;
  }
  kernel.setArg(0, inBuf);
  kernel.setArg(1, outBuf);

  cl::CommandQueue queue(context, device);

  // Another way to set the values of input buffer
  queue.enqueueFillBuffer(inBuf, 3, sizeof(int) * 10,
                          sizeof(int) * (vec.size() - 10));

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vec.size()));
  queue.enqueueReadBuffer(outBuf, CL_FALSE, 0, sizeof(int) * vec.size(),
                          vec.data());  // enqueueMapBuffer

  cl::finish();

  std::cout << vec[0] << " " << vec[20] << '\n';

  return EXIT_SUCCESS;
}
