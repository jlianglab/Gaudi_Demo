#!/bin/bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration (can be overridden via env vars or CLI flags)
IMAGE_NAME="${IMAGE_NAME:-cuda-apptainer.sif}"
DEF_FILE="${DEF_FILE:-$WORKSPACE_DIR/cuda-apptainer.def}"

# Help menu
show_help() {
    echo "Usage: ./cuda-apptainer.sh [OPTIONS] COMMAND [ARGS]"
    echo ""
    echo "Commands:"
    echo "  build         Build the Apptainer image (.sif file)"
    echo "  run           Run an interactive shell inside the container"
    echo "  exec [cmd]    Execute a command inside the container (e.g., ./cuda-apptainer.sh exec ./run-xray14-resnet50-best.local.sh)"
    echo ""
    echo "Options:"
    echo "  -i, --image NAME   Image (.sif) filename, stored in the script directory (default: cuda-apptainer.sif)"
    echo "  -d, --def FILE     Definition file used by 'build' (default: cuda-apptainer.def)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./cuda-apptainer.sh build"
    echo "  ./cuda-apptainer.sh -i my-image.sif build"
    echo "  ./cuda-apptainer.sh -i my-image.sif run"
    echo "  ./cuda-apptainer.sh -i my-image.sif exec python src/train_v1_classic.py dataset=chestxray14"
    echo "  IMAGE_NAME=my-image.sif ./cuda-apptainer.sh run"
    echo ""
    echo "Environment variables:"
    echo "  IMAGE_NAME      Image (.sif) filename (overridden by -i/--image)"
    echo "  DEF_FILE        Definition file path (overridden by -d/--def)"
    echo "  HF_HOME         Path to Hugging Face cache (optional)"
    echo "  SSL_CERT_FILE   Path to SSL certificate bundle (optional)"
    echo "  EXTRA_MOUNTS    Space-separated list of extra bind mounts (optional)"
}

# ---- Parse options (must come before the command) ----
while [ $# -gt 0 ]; do
    case "$1" in
        -i|--image)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                echo "Error: $1 requires a value."
                exit 1
            fi
            IMAGE_NAME="$2"
            shift 2
            ;;
        --image=*)
            IMAGE_NAME="${1#*=}"
            shift
            ;;
        -d|--def)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                echo "Error: $1 requires a value."
                exit 1
            fi
            DEF_FILE="$2"
            shift 2
            ;;
        --def=*)
            DEF_FILE="${1#*=}"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# Image is always stored in the workspace directory; only a filename is allowed.
if [[ "$IMAGE_NAME" == */* ]]; then
    echo "Error: image name must be a filename only (no '/'); images are stored in $WORKSPACE_DIR."
    exit 1
fi
IMAGE_PATH="$WORKSPACE_DIR/$IMAGE_NAME"

# Def file: absolute paths are used as-is, relative ones resolve to the workspace
case "$DEF_FILE" in
    /*) ;;
    *)  DEF_FILE="$WORKSPACE_DIR/$DEF_FILE" ;;
esac

if ! command -v apptainer &> /dev/null; then
    echo "Error: apptainer could not be found."
    exit 1
fi

cmd="${1:-help}"
shift || true

# Always inject NVIDIA GPU support
GPU_FLAG="--nv"

# Prepare dynamic bind mounts
BINDS=()
for dir in "/scratch" "/data" "/mnt/local/dataset"; do
    if [ -d "$dir" ]; then
        # Explicitly request read-write mounts (some environments may default to ro)
        BINDS+=("--bind" "$dir:$dir:rw")
    fi
done

# Hugging Face cache handling.
# - If HF_HOME is provided by the user, mount it into the container's expected cache path.
# - Otherwise, point HF_HOME to a writable location inside the container.
ENV_FLAGS=()
if [ -n "${HF_HOME:-}" ]; then
    mkdir -p "$HF_HOME"
    BINDS+=("--bind" "${HF_HOME}:/root/.cache/huggingface:rw")
else
    # /root/.cache may be read-only in some environments, so force a writable cache.
    ENV_FLAGS+=("--env" "HF_HOME=/tmp/huggingface")
fi

# bind the ssl certificate file, since ASU host and container may be different
if [ -n "${SSL_CERT_FILE:-}" ]; then
    BINDS+=("--bind" "${SSL_CERT_FILE}:/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:ro")
fi

# Add user-provided EXTRA_MOUNTS
if [ -n "${EXTRA_MOUNTS:-}" ]; then
    for mount_pair in $EXTRA_MOUNTS; do
        BINDS+=("--bind" "$mount_pair")
    done
fi

case "$cmd" in
    build)
        if [ ! -f "$DEF_FILE" ]; then
            echo "Error: Definition file '$DEF_FILE' not found."
            exit 1
        fi
        echo "Building Apptainer image '${IMAGE_PATH}' from '${DEF_FILE}'..."
        export APPTAINER_TMPDIR="${SCRATCH:-$HOME}/.apptainer/tmp"
        export APPTAINER_CACHEDIR="${SCRATCH:-$HOME}/.apptainer/cache"
        mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

        # Requires fakeroot to install system packages as root within container during build
        apptainer build --fakeroot "$IMAGE_PATH" "$DEF_FILE"
        ;;
    run)
        if [ ! -f "$IMAGE_PATH" ]; then
            echo "Error: Image '$IMAGE_PATH' not found. Run './cuda-apptainer.sh -i $IMAGE_NAME build' first."
            exit 1
        fi
        echo "Starting an interactive shell in container '${IMAGE_PATH}'..."
        apptainer shell $GPU_FLAG "${ENV_FLAGS[@]}" "${BINDS[@]}" --bind "$WORKSPACE_DIR:/workspace" --pwd /workspace "$IMAGE_PATH"
        ;;
    exec)
        if [ $# -eq 0 ]; then
            echo "Error: No command provided to execute."
            exit 1
        fi
        if [ ! -f "$IMAGE_PATH" ]; then
            echo "Error: Image '$IMAGE_PATH' not found. Run './cuda-apptainer.sh -i $IMAGE_NAME build' first."
            exit 1
        fi
        echo "Executing command in container '${IMAGE_PATH}': $*"
        apptainer exec $GPU_FLAG "${ENV_FLAGS[@]}" "${BINDS[@]}" --bind "$WORKSPACE_DIR:/workspace" --pwd /workspace "$IMAGE_PATH" "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $cmd"
        show_help
        exit 1
        ;;
esac