// https://www.youtube.com/watch?v=aVGF8rF4p5o
#include <algorithm>
#include <array>
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
  constexpr auto fileName = "ProcessMultidimensionalArray.cl";
  auto program = createProgram(fileName);

  auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
  auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
  auto& device = devices.front();

  constexpr int numRows = 3;
  constexpr int numCols = 2;
  constexpr int count = numRows * numCols;
  std::array<std::array<int, numCols>, numRows> arr = {
      {{1, 1}, {2, 2}, {3, 3}}};

  cl::Buffer buf(
      context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(int) * count, arr.data());

  cl_int err;
  cl::Kernel kernel(program, "ProcessMultidimensionalArray", &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Error creating OpenCL kernel: " << err << "\n";
    return EXIT_FAILURE;
  }
  kernel.setArg(0, buf);

  cl::CommandQueue queue(context, device);

  queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                             cl::NDRange(numCols, numRows));
  queue.enqueueReadBuffer(buf, CL_TRUE, 0, sizeof(int) * count, arr.data());

  std::cout << arr[0][0] << " " << arr[0][1] << '\n';
  std::cout << arr[1][0] << " " << arr[1][1] << '\n';
  std::cout << arr[2][0] << " " << arr[2][1] << '\n';

  return EXIT_SUCCESS;
}
