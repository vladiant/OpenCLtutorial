__kernel void
ProcessArray(__global int* data, __global int* outData)
{
    size_t ind = get_global_id(0);
    outData[ind] = data[ind] * 2;
}