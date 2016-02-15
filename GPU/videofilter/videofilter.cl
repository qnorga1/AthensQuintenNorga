
	
/////////////////////////////https://github.com/smistad/OpenCL-Gaussian-Blur/blob/master/gaussian_blur.cl
/////////////////////////////https://github.com/bamcis-io/SobelFilter/blob/main/SobelFilter/SobelFilter.cl

__kernel void gauss(__global const float *image, 
                        __global const float *filter, 
                        __global float *restrict output_image,
  			const unsigned filter_size,
			const unsigned IMAGE_W,
			const unsigned IMAGE_H)
{
	

	int fIndex = 0;
	float sum = (float) 0.0;
	int rowOffset = get_global_id(1) * 360;
	int my = get_global_id(0) + rowOffset;
		
	for (int r = -filter_size/2; r <= filter_size/2; r++)
	{
		int curRow = my + r * IMAGE_W;
		for (int c = -filter_size/2; c <= filter_size/2; c++)
		{				
			sum += image[ curRow + c ] * filter[ fIndex ];
			fIndex++;
		}
	}
	output_image[my] = sum;		
	

	//int idx = get_global_id(0);
	//int idy = get_global_id(1);
	//output_image[idx*(IMAGE_H) + idy] = image[idx*(IMAGE_H) + idy];

}

//https://stackoverflow.com/questions/45047672/sobel-filter-algorithm-c-no-libraries
__kernel void sobel(__global const float *image,  
                        __global float *restrict output_image,
  			const unsigned width,
			const unsigned height)
{

	int idx = get_global_id(0);
	int idy = get_global_id(1);
	float kernelx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	
	
	double magX = 0.0; // this is your magnitude

        for (int a = 0; a < 1.5; a++) {
            for (int b = 0; b < 1.5; b++) {
                magX += image[(idx - 1 + a) * 360 + idy - 1 + b] * kernelx[a][b];   
           }
        }
        output_image[idx * 360 + idy] = magX; 

	output_image[idx*(360) + idy] = image[idx*(360) + idy];
}

__kernel void add_weighted(__global const float *image1, 
			__global const float *image2, 
                        __global float *restrict output_image,
  			const float weight,
			const unsigned size)
{
	int idx = get_global_id(0);
	int idy = get_global_id(1);
	//for (int i = 0; i < size; i++) {
		output_image[idx*360 + idy] = image1[idx*360 + idy] * weight + image2[idx*360 + idy] * (1-weight);
	//}
}

__kernel void threshold(__global const float *input,
                        __global float *restrict output_image,
  			const float weight,
			const unsigned size)
{

	int idx = get_global_id(0);
	int idy = get_global_id(1);
	if (input[idx*360 + idy] > 80) {
		output_image[idx*360 + idy] = 0;
	}
	else {
		output_image[idx*360 + idy] = 255;
		}
	
}






