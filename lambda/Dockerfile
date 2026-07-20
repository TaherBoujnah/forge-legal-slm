FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies needed to compile llama.cpp
RUN yum install -y gcc gcc-c++ make cmake

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Set the handler
CMD [ "lambda_function.lambda_handler" ]