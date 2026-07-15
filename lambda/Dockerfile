# Use the official AWS Lambda Python 3.10 base image
FROM public.ecr.aws/lambda/python:3.10

# Install C++ compilers and build tools required to compile llama.cpp natively
RUN yum update -y && \
    yum groupinstall -y "Development Tools" && \
    yum install -y gcc-c++ make

# Copy requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install Python dependencies
# CMAKE_ARGS allows llama-cpp-python to compile optimized for CPU
RUN CMAKE_ARGS="-DLLAMA_BLAS=OFF -DLLAMA_CUBLAS=OFF" pip install -r requirements.txt

# Copy our function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Set the CMD to point to our handler function
CMD [ "lambda_function.lambda_handler" ]