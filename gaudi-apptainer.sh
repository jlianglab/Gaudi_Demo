#!/bin/bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (can be overridden via env vars or CLI flags)
IMAGE_NAME="${IMAGE_NAME:-gaudi-apptainer.sif}"
DEF_FILE="${DEF_FILE:-$WORKSPACE_DIR/gaudi-apptainer.def}"

show_help() {
    echo "Usage: ./gaudi-apptainer.sh [OPTIONS] COMMAND [ARGS]"
    echo ""
    echo "Commands:"
    echo "  build         Build the Apptainer image (.sif file)"
    echo "  run           Run an interactive shell inside the container"
    echo "  exec [cmd]    Execute a command inside the container"
    echo ""
    echo "Options:"
    echo "  -i, --image NAME   Image (.sif) filename, stored in the script directory (default: gaudi-apptainer.sif)"
    echo "  -d, --def FILE     Definition file used by 'build' (default: gaudi-apptainer.def)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./gaudi-apptainer.sh build"
    echo "  ./gaudi-apptainer.sh -i my-image.sif build"
    echo "  ./gaudi-apptainer.sh -i my-image.sif run"
    echo "  ./gaudi-apptainer.sh -i my-image.sif exec python src/train.py"
    echo "  IMAGE_NAME=my-image.sif ./gaudi-apptainer.sh run"
    echo ""
    echo "Environment variables:"
    echo "  IMAGE_NAME      Image (.sif) filename (overridden by -i/--image)"
    echo "  DEF_FILE        Definition file path (overridden by -d/--def)"
    echo "  HF_HOME         Path to Hugging Face cache (optional)"
    echo "  SSL_CERT_FILE   Path to SSL certificate bundle (optional)"
    echo "  EXTRA_MOUNTS    Space-separated list of extra bind mounts (optional)"
    echo "  PT_HPU_LAZY_MODE  Set to 1 to enable lazy mode (default: 0, eager mode)"
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

SHM_BIND="--bind /dev/shm:/dev/shm"

BINDS=()
for dir in "/scratch" "/data" "/mnt/local/dataset"; do
    if [ -d "$dir" ]; then
        BINDS+=("--bind" "$dir:$dir:rw")
    fi
done

ENV_FLAGS=()

HABANA_LOG_DIR="${SCRATCH:-$HOME}/.habana_logs"
mkdir -p "$HABANA_LOG_DIR"
BINDS+=("--bind" "$HABANA_LOG_DIR:/var/log/habana_logs")

if [ -n "${HF_HOME:-}" ]; then
    mkdir -p "$HF_HOME"
    BINDS+=("--bind" "${HF_HOME}:/root/.cache/huggingface:rw")
else
    ENV_FLAGS+=("--env" "HF_HOME=/tmp/huggingface")
fi

if [ -n "${SSL_CERT_FILE:-}" ]; then
    BINDS+=("--bind" "${SSL_CERT_FILE}:/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:ro")
fi

# User-provided extra mounts
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
        echo "Building Gaudi Apptainer image '${IMAGE_PATH}' from '${DEF_FILE}'..."
        export APPTAINER_TMPDIR="${SCRATCH:-$HOME}/.apptainer/tmp"
        export APPTAINER_CACHEDIR="${SCRATCH:-$HOME}/.apptainer/cache"
        mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

        apptainer build --fakeroot "$IMAGE_PATH" "$DEF_FILE"
        ;;
    run)
        if [ ! -f "$IMAGE_PATH" ]; then
            echo "Error: Image '$IMAGE_PATH' not found. Run './gaudi-apptainer.sh -i $IMAGE_NAME build' first."
            exit 1
        fi
        echo "Starting an interactive shell in container '${IMAGE_PATH}'..."
        apptainer shell \
            $SHM_BIND \
            "${ENV_FLAGS[@]}" \
            "${BINDS[@]}" \
            --bind "$WORKSPACE_DIR:/workspace" \
            --pwd /workspace \
            "$IMAGE_PATH"
        ;;
    exec)
        if [ $# -eq 0 ]; then
            echo "Error: No command provided to execute."
            exit 1
        fi
        if [ ! -f "$IMAGE_PATH" ]; then
            echo "Error: Image '$IMAGE_PATH' not found. Run './gaudi-apptainer.sh -i $IMAGE_NAME build' first."
            exit 1
        fi
        echo "Executing command in container '${IMAGE_PATH}': $*"
        apptainer exec \
            $SHM_BIND \
            "${ENV_FLAGS[@]}" \
            "${BINDS[@]}" \
            --bind "$WORKSPACE_DIR:/workspace" \
            --pwd /workspace \
            "$IMAGE_PATH" \
            "$@"
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