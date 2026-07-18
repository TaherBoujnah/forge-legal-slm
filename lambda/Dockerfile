# Use the official AWS Lambda Python 3.10 base image
FROM public.ecr.aws/lambda/python:3.10

# THE FIX: Install the missing GNU OpenMP system library required by llama.cpp

RUN dnf install -y libgomp && \
    dnf clean all && \
    rm -rf /var/cache/dnf

# Copy requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Download the pre-compiled CPU wheel instead of building from source
# This skips C++ compilation, preventing the "no space left on device" error.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Copy our function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Set the CMD to point to our handler function
CMD [ "lambda_function.lambda_handler" ]