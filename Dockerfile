# TestPoint container images.
#
# Targets (docker build --target <name> .):
#   app        station app: API + database + station UI (small, no vision stack)
#   gpu        rep-counting engine on NVIDIA GPUs (default, used on the station PC)
#   cpu        same engine, CPU only (laptops, CI, machines without an NVIDIA GPU)
#   test       runs the unit tests (engine + app)
#   prototype  the original clickable UI mock-up served by nginx
#
# The test site has no internet, so the pose models are downloaded while the
# image is built and live inside it. Build on a machine with internet, then move
# the images with `docker save` / `docker load` (see README, "Docker").

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG NGINX_IMAGE=nginx:1.27-alpine

# ----------------------------------------------------------------------- pybase
FROM ${PYTHON_IMAGE} AS pybase
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/opt/testpoint/models
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app /data "$TORCH_HOME" \
 && chown app:app /app /data "$TORCH_HOME"
WORKDIR /app

# ------------------------------------------------------------------------- base
# Vision base: libgl1 + libglib2.0-0 are needed by the opencv-python wheels
# that rtmlib pulls in.
FROM pybase AS base
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------------- app
# Station app: FastAPI + SQLite + the station UI. Installs only the [app] extra.
FROM pybase AS app
COPY pyproject.toml ./
# Face check-in needs OpenCV; the headless wheel avoids the X11/GL libraries.
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']['app']))" > /tmp/app-req.txt \
 && pip install -r /tmp/app-req.txt "numpy>=1.26" "opencv-python-headless>=4.9"
COPY src ./src
RUN pip install --no-deps . && rm -rf build src/*.egg-info \
 && python -m testpoint.api.faces --download /opt/testpoint/models/face
COPY web/station ./web/station
ENV TESTPOINT_DB=sqlite:////data/testpoint.db \
    TESTPOINT_WEB=/app/web/station \
    TESTPOINT_FACE_MODELS=/opt/testpoint/models/face \
    TESTPOINT_PORT=8000
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
ENTRYPOINT ["testpoint-app"]

# ------------------------------------------------------------- dependencies (cpu)
# Dependencies are installed from pyproject.toml alone, before the source is
# copied, so editing code does not reinstall ~1 GB of wheels.
FROM base AS deps-cpu
COPY pyproject.toml ./
RUN mkdir -p src/testpoint && touch src/testpoint/__init__.py \
 && pip install . \
 && pip uninstall -y testpoint \
 && rm -rf src build

# ------------------------------------------------------------- dependencies (gpu)
# Same swap as on Windows (README, "GPU setup"): rtmlib depends on the CPU
# onnxruntime wheel, which shares its package directory with onnxruntime-gpu,
# so the CPU wheel is removed and the GPU wheel reinstalled on top. The CUDA 13
# and cuDNN runtime come from the nvidia-* wheels in the [gpu] extra; the GPU
# driver itself is provided by the host at run time (docker run --gpus all).
FROM base AS deps-gpu
COPY pyproject.toml ./
RUN mkdir -p src/testpoint && touch src/testpoint/__init__.py \
 && pip install ".[gpu]" \
 && ORT_GPU="$(pip show onnxruntime-gpu | sed -n 's/^Version: //p')" \
 && pip uninstall -y testpoint onnxruntime \
 && pip install --force-reinstall --no-deps "onnxruntime-gpu==${ORT_GPU}" \
 && python -c "import onnxruntime as o; p = o.get_available_providers(); print(p); assert 'CUDAExecutionProvider' in p" \
 && rm -rf src build

# ------------------------------------------------------------------------ models
# lightweight ~40 MB, balanced ~155 MB, performance ~590 MB.
FROM deps-cpu AS models
ARG MODEL_MODES="lightweight balanced"
COPY docker/fetch_models.py /tmp/fetch_models.py
USER app
RUN python /tmp/fetch_models.py ${MODEL_MODES}

# ----------------------------------------------------------------------- cpu app
FROM deps-cpu AS cpu
COPY --from=models --chown=app:app /opt/testpoint/models /opt/testpoint/models
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-deps . && rm -rf build src/*.egg-info
USER app
ENTRYPOINT ["testpoint-demo"]
CMD ["--help"]

# -------------------------------------------------------------------------- test
# Unit tests need no camera and no models, so this skips the models stage.
FROM deps-cpu AS test
RUN python -c "import tomllib; d = tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']; print('\n'.join(d['app'] + d['dev']))" > /tmp/test-req.txt \
 && pip install -r /tmp/test-req.txt
COPY src ./src
RUN pip install --no-deps . && rm -rf build src/*.egg-info
COPY tests ./tests
USER app
ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-q", "-p", "no:cacheprovider"]

# --------------------------------------------------------------------- prototype
FROM ${NGINX_IMAGE} AS prototype
COPY web/prototype/ /usr/share/nginx/html/

# ----------------------------------------------------------------------- gpu app
# Last stage, so a plain `docker build .` produces the station image.
FROM deps-gpu AS gpu
COPY --from=models --chown=app:app /opt/testpoint/models /opt/testpoint/models
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-deps . && rm -rf build src/*.egg-info
USER app
ENTRYPOINT ["testpoint-demo"]
CMD ["--help"]
