// https://www.youtube.com/watch?v=2l2LXM1wWBg
#include <fstream>
#include <iostream>
#include <numeric>
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
  constexpr auto fileName = "NumericalReduction.cl";
  auto program = createProgram(fileName);

  auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
  auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
  auto& device = devices.front();

  std::vector<int> vec(1024);
  std::iota(vec.begin(), vec.end(), 0);
  auto expectedSum = std::accumulate(vec.begin(), vec.end(), 0);

  cl_int err;
  cl::Kernel kernel(program, "NumericalReduction", &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Error creating OpenCL kernel: " << err << "\n";
    return EXIT_FAILURE;
  }

  auto workGroupSize =
      kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
  auto numWorkGroups = vec.size() / workGroupSize;

  std::cout << "workGroupSize: " << workGroupSize << '\n';
  std::cout << "numWorkGroups: " << numWorkGroups << '\n';

  cl::Buffer buf(
      context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
      sizeof(int) * vec.size(), vec.data());

  cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                    sizeof(int) * numWorkGroups);

  kernel.setArg(0, buf);
  kernel.setArg(1, sizeof(int) * workGroupSize, nullptr);
  kernel.setArg(2, outBuf);

  std::vector<int> outVec(numWorkGroups);

  cl::CommandQueue queue(context, device);
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vec.size()),
                             cl::NDRange(workGroupSize));
  queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(int) * outVec.size(),
                          outVec.data());

  std::cout << "Expected sum: " << expectedSum << '\n';
  std::cout << "Actual sum: "
            << std::accumulate(outVec.begin(), outVec.end(), 0) << '\n';

  return EXIT_SUCCESS;
}
