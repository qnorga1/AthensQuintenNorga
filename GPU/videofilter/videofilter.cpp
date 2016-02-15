#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <math.h>
#include <fstream>
#include <time.h>
#include "opencv2/opencv.hpp"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#define STRING_BUFFER_LEN 1024
#define BILLION 1000000000L

using namespace cv;
using namespace std;
#define SHOW

//////helper functions
//error handling
void print_clbuild_errors(cl_program program,cl_device_id device)
{
	cout<<"Program Build failed\n";
	size_t length;
	char buffer[2048];
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
	cout<<"--- Build log ---\n "<<buffer<<endl;
	exit(1);
}
//read file
unsigned char ** read_file(const char *name) {
  	size_t size;
  	unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
  	FILE* fp = fopen(name, "rb");
  	if (!fp) {
    		printf("no such file:%s",name);
    		exit(-1);
  	}

  	fseek(fp, 0, SEEK_END);
  	size = ftell(fp);
  	fseek(fp, 0, SEEK_SET);

  	*output = (unsigned char *)malloc(size);
  	unsigned char **outputstr=(unsigned char **)malloc(sizeof(unsigned char *));
  	*outputstr= (unsigned char *)malloc(size);
  	if (!*output) {
    		fclose(fp);
    		printf("mem allocate failure:%s",name);
    		exit(-1);
  	}

  	if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
  	fclose(fp);
  	printf("file size %d\n",size);
  	printf("-------------------------------------------\n");
  	snprintf((char *)*outputstr,size,"%s\n",*output);
  	printf("%s\n",*outputstr);
  	printf("-------------------------------------------\n");
  	return outputstr;
}
//callback
void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
        fwrite(buffer, 1, length, stdout);
}
//checkerror
void checkError(int status, const char *msg) {
	if(status!=CL_SUCCESS)	
		printf("%s\n",msg);
}





//////MAIN FUNCTION
int main(int, char**)
{
	char char_buffer[STRING_BUFFER_LEN];
     	cl_platform_id platform;
     	cl_device_id device;
     	cl_context context;
     	cl_context_properties context_properties[] =
     	{ 
          CL_CONTEXT_PLATFORM, 0,
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
          0
     	};
     	cl_command_queue queue;
     	cl_program program;
     	cl_kernel kernel_gauss, kernel_gauss2, kernel_gauss3, kernel_sabel, kernel_sabel2, kernel_add_weighted, kernel_threshold;
	cl_event write_event[1];	
	
	//////// CPU //////////
    	VideoCapture camera("./bourne.mp4");
    	if(!camera.isOpened())  // check if we succeeded
        	return -1;

    	const string NAME = "./output_cpu.avi";   // Form the new name with container
    	int ex = static_cast<int>(CV_FOURCC('M','J','P','G'));
    	Size S = Size((int) camera.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  	(int) camera.get(CV_CAP_PROP_FRAME_HEIGHT));
	//Size S =Size(1280,720);
	cout << "SIZE:" << S << endl;
	
    	VideoWriter outputVideo;                                        // Open the output
        outputVideo.open(NAME, ex, 25, S, true);

    	if (!outputVideo.isOpened())
    	{
        	cout  << "Could not open the output video for write: " << NAME << endl;
        	return -1;
    	}

	////CPU TIMING
	timespec start,end;
	double diff,tot,diff_gpu,tot_gpu;
	int count=0;
	tot = 0;
	tot_gpu = 0;
	const char *windowName = "filter";   // Name shown in the GUI window.
    	#ifdef SHOW
    	namedWindow(windowName); // Resizable window, might not work on Windows.
    	#endif
	while (true) {
        	Mat cameraFrame,displayframe;
		count=count+1;
		if(count > 299) break;
        	camera >> cameraFrame;
        	Mat filterframe = Mat(cameraFrame.size(), CV_8UC3);
        	Mat grayframe,edge_x,edge_y,edge,edge_inv;
    		cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);

		clock_gettime(CLOCK_REALTIME, &start);
    		GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
    		GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
    		GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
		Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT );
		Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT );
		addWeighted( edge_x, 0.5, edge_y, 0.5, 0, edge );
        	threshold(edge, edge, 80, 255, THRESH_BINARY_INV);
		clock_gettime(CLOCK_REALTIME, &end);

        	cvtColor(edge, edge_inv, CV_GRAY2BGR);
    		// Clear the output image to black, so that the cartoon line drawings will be black (ie: not drawn).
    		memset((char*)displayframe.data, 0, displayframe.step * displayframe.rows);
		grayframe.copyTo(displayframe,edge);
        	cvtColor(displayframe, displayframe, CV_GRAY2BGR);
		outputVideo << displayframe;
		#ifdef SHOW
        	imshow(windowName, displayframe);
		#endif
		diff = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/BILLION;
		tot+=diff;
	}

	outputVideo.release();
	camera.release();
  	printf ("CPU FPS %.2lf .\n", 299.0/tot );


	////////////// GPU /////////////////////
	///video stuff
    	VideoCapture camera_gpu("./bourne.mp4");
    	if(!camera_gpu.isOpened())  // check if we succeeded
        	return -1;

    	const string NAME_gpu = "./output_gpu.avi";   // Form the new name with container
    	int ex_gpu = static_cast<int>(CV_FOURCC('M','J','P','G'));
    	Size S_gpu = Size((int) camera_gpu.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  	(int) camera_gpu.get(CV_CAP_PROP_FRAME_HEIGHT));
	//Size S =Size(1280,720);
	cout << "SIZE:" << S_gpu << endl;
	
    	VideoWriter outputVideo_gpu;                                        // Open the output
        outputVideo_gpu.open(NAME_gpu, ex_gpu, 25, S_gpu, true);

    	if (!outputVideo_gpu.isOpened())
    	{
        	cout  << "Could not open the output video for write: " << NAME_gpu << endl;
        	return -1;
    	}

	///gpu stuff
	

	const unsigned N = 3;
	const unsigned X = S_gpu.width; //640
	const unsigned Y = S_gpu.height; //360
	const float weight = 0.5;
	float *input_image;//=(float *) malloc(sizeof(float)*N);
	float *input_image2;//=(float *) malloc(sizeof(float)*N);
	float *input_image3;//=(float *) malloc(sizeof(float)*N);
	float *input_filter;//=(float *) malloc(sizeof(float)*N);
	float *output;//=(float *) malloc(sizeof(float)*N);
	cl_mem input_image_buf; // num_devices elements
	cl_mem input_image2_buf; // num_devices elements
	cl_mem input_image3_buf; // num_devices elements
	cl_mem input_filter_buf; // num_devices elements
	cl_mem output_buf; // num_devices elements
	const size_t global_work_size[2] = {X,Y};
//TODO	const size_t local_work_size[2] = {16,16};
	int status, errcode;

	clGetPlatformIDs(1, &platform, NULL);
     	clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
     	printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
     	clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
     	printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
     	clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
     	printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

     	context_properties[1] = (cl_context_properties)platform;
     	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
     	context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
     	queue = clCreateCommandQueue(context, device, 0, NULL);

	unsigned char **opencl_program=read_file("videofilter.cl");
     	program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
     	if (program == NULL)
	{
         	printf("Program creation failed\n");
         	return 1;
	}	
     	int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
	kernel_gauss = clCreateKernel(program, "gauss", NULL);
	kernel_gauss2 = clCreateKernel(program, "gauss", NULL);
	kernel_gauss3 = clCreateKernel(program, "gauss", NULL);
	kernel_sabel = clCreateKernel(program, "sobel", NULL);
	kernel_sabel2 = clCreateKernel(program, "sobel", NULL);	
	kernel_add_weighted = clCreateKernel(program, "add_weighted", NULL);
	kernel_threshold = clCreateKernel(program, "threshold", NULL);

	// Input buffers.
  	input_image_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
       		X*Y* sizeof(float), NULL, &status);
    	checkError(status, "Failed to create buffer for input image");
	input_image2_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
       		X*Y* sizeof(float), NULL, &status);
    	checkError(status, "Failed to create buffer for input image");
	input_image3_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
       		X*Y* sizeof(float), NULL, &status);
    	checkError(status, "Failed to create buffer for input image");
  	input_filter_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        	N*N* sizeof(float), NULL, &status);
    	checkError(status, "Failed to create buffer for input filter");

    	// Output buffer.
  	output_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        	X*Y* sizeof(float), NULL, &status);
    	checkError(status, "Failed to create buffer for output");


	////GPU TIMING
	int count_gpu=0;
	const char *windowName_gpu = "filter_gpu";   // Name shown in the GUI window.
    	#ifdef SHOW
    	namedWindow(windowName_gpu); // Resizable window, might not work on Windows.
    	#endif
	while (true) {
        	Mat cameraFrame;
                Mat grayframe1=Mat::zeros(Y,X,CV_32FC1);
		count_gpu=count_gpu+1;
		if(count_gpu > 299) break;
        	camera_gpu >> cameraFrame;
        	Mat filterframe = Mat(cameraFrame.size(), CV_8UC3); //8 bit int -> CONVERT TO FP grayframe.ConvertTo(newframe, CV_32FC1)
        	Mat grayframe,displayframe,edge_x,edge_y,edge,edge_inv, newframe;
		edge = Mat::zeros(Y,X,CV_32FC1);
    		cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);
		grayframe.convertTo(newframe, CV_32FC1);
	
		// Map inputs
    		
    		cl_event kernel_event;//,finish_event;
		//printf("test %d\n", X);

  		input_image = (float *)clEnqueueMapBuffer(queue, input_image_buf, CL_TRUE, CL_MAP_WRITE,
        		0, X*Y* sizeof(float),0, NULL, &write_event[0], &errcode);
    		checkError(errcode, "Failed to map input image");
//  		input_filter = (float *)clEnqueueMapBuffer(queue, input_filter_buf, CL_TRUE, CL_MAP_WRITE, //DONT MAP ALL INTERMEDIATE BUFFERS BACK TO CPU 
//        		0, N*N* sizeof(float), 0, NULL, &write_event[1], &errcode);
//    		checkError(errcode, "Failed to map input filter");
//		input_image2 = (float *)clEnqueueMapBuffer(queue, input_image2_buf, CL_TRUE, CL_MAP_WRITE,
//        		0, X*Y* sizeof(float),0, NULL, &write_event[2], &errcode);
//    		checkError(errcode, "Failed to map input image 2");
//		input_image3 = (float *)clEnqueueMapBuffer(queue, input_image3_buf, CL_TRUE, CL_MAP_WRITE,
//        		0, X*Y* sizeof(float),0, NULL, &write_event[3], &errcode);
//    		checkError(errcode, "Failed to map input image 2");

		//fill input_a/... with data
		//copyMakeBorder(newframe, newframe, 1,1,1,1, BORDER_CONSTANT, (0,0,0)); TODO
		memcpy(input_image, newframe.data, (X)*(Y)*sizeof(float));
		float input_filter[] = {1/16,2/16,1/16,2/16,4/16,2/16,1/16,2/16,1/16};
		

		// Set kernel arguments.
    		unsigned argi = 0;
    		status = clSetKernelArg(kernel_gauss, argi++, sizeof(cl_mem), &input_image_buf);
    		checkError(status, "Failed to set argument 1 of kernel 1");
		status = clSetKernelArg(kernel_gauss, argi++, sizeof(cl_mem), &input_filter_buf);
    		checkError(status, "Failed to set argument 2 of kernel 1");
		status = clSetKernelArg(kernel_gauss, argi++, sizeof(cl_mem), &input_image2_buf);
    		checkError(status, "Failed to set argument 3 of kernel 1");
		status = clSetKernelArg(kernel_gauss, argi++, sizeof(unsigned), &N);
    		checkError(status, "Failed to set argument 4 of kernel 1");
		status = clSetKernelArg(kernel_gauss, argi++, sizeof(unsigned), &X); 
    		checkError(status, "Failed to set argument 5 of kernel 1");
		status = clSetKernelArg(kernel_gauss, argi++, sizeof(unsigned), &Y); 
    		checkError(status, "Failed to set argument 6 of kernel 1");
		/*
		argi = 0;

		status = clSetKernelArg(kernel_gauss2, argi++, sizeof(cl_mem), &input_image2_buf);
    		checkError(status, "Failed to set argument 1 of kernel 2");
		status = clSetKernelArg(kernel_gauss2, argi++, sizeof(cl_mem), &input_filter_buf);
    		checkError(status, "Failed to set argument 2 of kernel 2");
		status = clSetKernelArg(kernel_gauss2, argi++, sizeof(cl_mem), &input_image_buf);
    		checkError(status, "Failed to set argument 3 of kernel 2");
		status = clSetKernelArg(kernel_gauss2, argi++, sizeof(unsigned), &N);
    		checkError(status, "Failed to set argument 4 of kernel 2");
		status = clSetKernelArg(kernel_gauss2, argi++, sizeof(unsigned), &X); 
    		checkError(status, "Failed to set argument 5 of kernel 2");
		status = clSetKernelArg(kernel_gauss2, argi++, sizeof(unsigned), &Y); 
    		checkError(status, "Failed to set argument 6 of kernel 2");

		argi = 0;

		status = clSetKernelArg(kernel_gauss3, argi++, sizeof(cl_mem), &input_image_buf);
    		checkError(status, "Failed to set argument 1 of kernel 3");
		status = clSetKernelArg(kernel_gauss3, argi++, sizeof(cl_mem), &input_filter_buf);
    		checkError(status, "Failed to set argument 2 of kernel 3");
		status = clSetKernelArg(kernel_gauss3, argi++, sizeof(cl_mem), &input_image2_buf);
    		checkError(status, "Failed to set argument 3 of kernel 3");
		status = clSetKernelArg(kernel_gauss3, argi++, sizeof(unsigned), &N);
    		checkError(status, "Failed to set argument 4 of kernel 3");
		status = clSetKernelArg(kernel_gauss3, argi++, sizeof(unsigned), &X); 
    		checkError(status, "Failed to set argument 5 of kernel 3");
		status = clSetKernelArg(kernel_gauss3, argi++, sizeof(unsigned), &Y); 
    		checkError(status, "Failed to set argument 6 of kernel 3");

		argi = 0;

		status = clSetKernelArg(kernel_sabel, argi++, sizeof(cl_mem), &input_image2_buf);
    		checkError(status, "Failed to set argument 1 of kernel 4");
		status = clSetKernelArg(kernel_sabel, argi++, sizeof(cl_mem), &input_image_buf);
    		checkError(status, "Failed to set argument 2 of kernel 4");
		status = clSetKernelArg(kernel_sabel, argi++, sizeof(unsigned), &X);
    		checkError(status, "Failed to set argument 3 of kernel 4");
		status = clSetKernelArg(kernel_sabel, argi++, sizeof(unsigned), &Y);
    		checkError(status, "Failed to set argument 4 of kernel 4");

		argi = 0;

		status = clSetKernelArg(kernel_sabel2, argi++, sizeof(cl_mem), &input_image2_buf);
    		checkError(status, "Failed to set argument 1 of kernel 5");
		status = clSetKernelArg(kernel_sabel2, argi++, sizeof(cl_mem), &input_image3_buf); 
    		checkError(status, "Failed to set argument 2 of kernel 5");
		status = clSetKernelArg(kernel_sabel2, argi++, sizeof(unsigned), &X);
    		checkError(status, "Failed to set argument 3 of kernel 5");
		status = clSetKernelArg(kernel_sabel2, argi++, sizeof(unsigned), &Y);
    		checkError(status, "Failed to set argument 4 of kernel 5");

		argi = 0;

		status = clSetKernelArg(kernel_add_weighted, argi++, sizeof(cl_mem), &input_image_buf);
    		checkError(status, "Failed to set argument 1 of kernel 6");
		status = clSetKernelArg(kernel_add_weighted, argi++, sizeof(cl_mem), &input_image3_buf); 
    		checkError(status, "Failed to set argument 2 of kernel 6");
		status = clSetKernelArg(kernel_add_weighted, argi++, sizeof(cl_mem), &input_image2_buf); 
    		checkError(status, "Failed to set argument 3 of kernel 6");
		status = clSetKernelArg(kernel_add_weighted, argi++, sizeof(float), &weight);
    		checkError(status, "Failed to set argument 4 of kernel 6");
		status = clSetKernelArg(kernel_add_weighted, argi++, sizeof(unsigned), &X); //TODO SIZE
    		checkError(status, "Failed to set argument 5 of kernel 6");

		argi = 0;

		status = clSetKernelArg(kernel_threshold, argi++, sizeof(cl_mem), &input_image2_buf);
    		checkError(status, "Failed to set argument 1 of kernel 7");
		status = clSetKernelArg(kernel_threshold, argi++, sizeof(cl_mem), &output_buf);
    		checkError(status, "Failed to set argument 2 of kernel 7");
		status = clSetKernelArg(kernel_threshold, argi++, sizeof(float), &weight); //TODO
    		checkError(status, "Failed to set argument 3 of kernel 7"); 
		status = clSetKernelArg(kernel_threshold, argi++, sizeof(unsigned), &X); //TODO
    		checkError(status, "Failed to set argument 4 of kernel 7");
		
		*/
		
		

		// Unmap inputs
		status=clEnqueueUnmapMemObject(queue,input_image_buf,input_image,0,NULL,NULL);
        	checkError(status, "Failed to unmap buffer 1");
//        	status=clEnqueueUnmapMemObject(queue,input_filter_buf,input_filter,0,NULL,NULL);
//        	checkError(status, "Failed to unmap buffer filter");
//		status=clEnqueueUnmapMemObject(queue,input_image2_buf,input_image2,0,NULL,NULL);
//        	checkError(status, "Failed to unmap buffer 2");
//		status=clEnqueueUnmapMemObject(queue,input_image3_buf,input_image3,0,NULL,NULL);
//        	checkError(status, "Failed to unmap buffer 3");

		clock_gettime(CLOCK_REALTIME, &start);

    		///////////////GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
		status = clEnqueueNDRangeKernel(queue, kernel_gauss, 2, NULL,
        		global_work_size, NULL, 1, write_event, &kernel_event);
		checkError(status, "Failed to launch kernel");
    		status = clWaitForEvents(1,&kernel_event);
		checkError(status, "Failed to launch kernel11");		
		/*
    		//////////////GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
		status = clEnqueueNDRangeKernel(queue, kernel_gauss2, 2, NULL,
        		global_work_size, NULL, 1, write_event, &kernel_event);
		checkError(status, "Failed to launch kernel2");
    		status = clWaitForEvents(1,&kernel_event);
		checkError(status, "Failed to launch kernel22");
		
    		//////////////GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
		status = clEnqueueNDRangeKernel(queue, kernel_gauss3, 2, NULL,
        		global_work_size, NULL, 1, write_event, &kernel_event);
		checkError(status, "Failed to launch kernel3");
    		status = clWaitForEvents(1,&kernel_event);
		checkError(status, "Failed to launch kernel33");		

		//////////////Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT );
		status = clEnqueueNDRangeKernel(queue, kernel_sabel, 2, NULL,
        		global_work_size, NULL, 1, write_event, &kernel_event);
		checkError(status, "Failed to launch kernel4");
    		status = clWaitForEvents(1,&kernel_event);
		checkError(status, "Failed to launch kernel44");

		//////////////Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT );
		status = clEnqueueNDRangeKernel(queue, kernel_sabel2, 2, NULL,
        		global_work_size, NULL, 1, write_event, &kernel_event);
		checkError(status, "Failed to launch kernel5");
    		status = clWaitForEvents(1,&kernel_event);
		checkError(status, "Failed to launch kernelF55");

		//////////////addWeighted( edge_x, 0.5, edge_y, 0.5, 0, edge );
		status = clEnqueueNDRangeKernel(queue, kernel_add_weighted, 2, NULL,
        		global_work_size, NULL, 1, write_event, &kernel_event);
		checkError(status, "Failed to launch kernel6");
    		status = clWaitForEvents(1,&kernel_event);
		checkError(status, "Failed to launch kernel66");

        	//////////////threshold(edge, edge, 80, 255, THRESH_BINARY_INV);
		status = clEnqueueNDRangeKernel(queue, kernel_threshold, 2, NULL,
        		global_work_size, NULL, 1, write_event, &kernel_event);
		checkError(status, "Failed to launch kernel7");
    		status = clWaitForEvents(1,&kernel_event);
		checkError(status, "Failed to launch kernelFINAL");
		*/

    		clock_gettime(CLOCK_REALTIME, &end);

    		checkError(status, "Failed to launch kernel");

		//cl_int arrSize = clGetMemObjectInfo(output_buf, CL_MEM_SIZE, 64, NULL, NULL);
		//cout << "The size of array is :" << arrSize;
		//cl_int arrSize2 = clGetMemObjectInfo(input_image_buf, CL_MEM_SIZE, 64, NULL, NULL);
		//cout << "The size of array is :" << arrSize2;
    		

		// Map output
 		output = (float *)clEnqueueMapBuffer(queue, input_image2_buf, CL_TRUE, CL_MAP_READ, 
        		0, X*Y* sizeof(float), 0, NULL, NULL, &errcode);
    		checkError(errcode, "Failed to map output");
                
		//printf("test %d\n", errcode);
		memcpy(grayframe1.data,output,X*Y*sizeof(float));
                status=clEnqueueUnmapMemObject(queue,input_image2_buf,output,0,NULL,NULL);
                grayframe1.convertTo(grayframe1, CV_8UC3);
		//printf("test %d\n", X);
        	//cvtColor(edge, edge_inv, CV_GRAY2BGR);
                //edge.convertTo(edge, CV_8UC3);
    		// Clear the output image to black, so that the cartoon line drawings will be black (ie: not drawn).
    		memset((char*)displayframe.data, 0, displayframe.step * displayframe.rows);
		
		
		//printf("test %d\n", grayframe.size().width);
		//printf("test %d\n", grayframe.size().height);
		//printf("test %d\n", displayframe.size().width);
		//printf("test %d\n", displayframe.size().height);
		//printf("test %d\n", edge.size().width);
		//printf("test %d\n", edge.size().height);
		grayframe1.copyTo(displayframe);
        	cvtColor(displayframe, displayframe, CV_GRAY2BGR);
             
		outputVideo_gpu << displayframe;
		#ifdef SHOW
        	imshow(windowName_gpu, displayframe);
		#endif
		diff_gpu = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/BILLION;
		tot_gpu+=diff_gpu;
	}

	// Release local events.
    	clReleaseEvent(write_event[0]);
    	//clReleaseEvent(write_event[1]);
	clReleaseKernel(kernel_gauss);
	clReleaseKernel(kernel_gauss2);
	clReleaseKernel(kernel_gauss3);
	clReleaseKernel(kernel_sabel);
	clReleaseKernel(kernel_sabel2);
	clReleaseKernel(kernel_add_weighted);
	clReleaseKernel(kernel_threshold);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(input_image_buf);
	clReleaseMemObject(input_image2_buf);
	clReleaseMemObject(input_image3_buf);
	clReleaseMemObject(input_filter_buf);
	clReleaseMemObject(output_buf);
	clReleaseProgram(program);
	clReleaseContext(context);
	clFinish(queue);

	outputVideo_gpu.release();
	camera_gpu.release();
  	printf ("GPU FPS %.2lf .\n", 299.0/tot );

    	return EXIT_SUCCESS;

}
