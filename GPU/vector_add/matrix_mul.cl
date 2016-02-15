__kernel void matrix_mul(__global const float *x, 
                        __global const float *y, 
                        __global float *restrict z,
  						 const unsigned size)
{
int idx = get_global_id(0);
int idy = get_global_id(1);
z[idx*( size) + idy] = 0;
for (int i = 0; i < size; i++) {

	z[idx*(size) + idy] += x[idx*(size) + i] * y[idy*(size) + i];

}
}






