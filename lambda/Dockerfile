FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies needed to compile llama.cpp.
# NOTE: this base image runs Amazon Linux 2023, which replaced `yum` with
# `dnf` -- `yum` is not present as a command at all in this image, hence
# the earlier "yum: command not found" build failure. `libgomp` is also
# required at runtime for llama.cpp's OpenMP-based threading.
RUN dnf install -y gcc gcc-c++ make cmake libgomp && \
    dnf clean all && \
    rm -rf /var/cache/dnf

# Install Python dependencies
COPY requirements.txt .

ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_BLAS=OFF -DGGML_CUDA=OFF"
ENV FORCE_CMAKE=1
RUN pip install --no-cache-dir --no-binary llama-cpp-python -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Set the handler
CMD [ "lambda_function.lambda_handler" ]