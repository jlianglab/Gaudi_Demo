#!/bin/bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
IMAGE_NAME="cuda-apptainer.sif"
DEF_FILE="$WORKSPACE_DIR/cuda-apptainer.def"

# Help menu
show_help() {
    echo "Usage: ./apptainer.sh [COMMAND] [ARGS]"
    echo ""
    echo "Commands:"
    echo "  build         Build the Apptainer image (.sif file)"
    echo "  run           Run an interactive shell inside the container"
    echo "  exec [cmd]    Execute a command inside the container (e.g., ./apptainer.sh exec ./run-xray14-resnet50-best.local.sh)"
    echo ""
    echo "Examples:"
    echo "  ./apptainer.sh build"
    echo "  ./apptainer.sh run"
    echo "  ./apptainer.sh exec python src/train_v1_classic.py dataset=chestxray14"
}

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
        echo "Building Apptainer image '${IMAGE_NAME}'..."
        export APPTAINER_TMPDIR="${SCRATCH:-$HOME}/.apptainer/tmp"
        export APPTAINER_CACHEDIR="${SCRATCH:-$HOME}/.apptainer/cache"
        mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
        
        # Requires fakeroot to install system packages as root within container during build
        apptainer build --fakeroot "$WORKSPACE_DIR/$IMAGE_NAME" $DEF_FILE
        ;;
    run)
        if [ ! -f "$WORKSPACE_DIR/$IMAGE_NAME" ]; then
            echo "Error: Image '$IMAGE_NAME' not found. Run './apptainer-for-cuda.sh build' first."
            exit 1
        fi
        echo "Starting an interactive shell in the Apptainer container..."
        apptainer shell $GPU_FLAG "${ENV_FLAGS[@]}" "${BINDS[@]}" --bind "$WORKSPACE_DIR:/workspace" --pwd /workspace "$WORKSPACE_DIR/$IMAGE_NAME"
        ;;
    exec)
        if [ $# -eq 0 ]; then
            echo "Error: No command provided to execute."
            exit 1
        fi
        if [ ! -f "$WORKSPACE_DIR/$IMAGE_NAME" ]; then
            echo "Error: Image '$IMAGE_NAME' not found. Run './apptainer-for-cuda.sh build' first."
            exit 1
        fi
        echo "Executing command in container: $@"
        apptainer exec $GPU_FLAG "${ENV_FLAGS[@]}" "${BINDS[@]}" --bind "$WORKSPACE_DIR:/workspace" --pwd /workspace "$WORKSPACE_DIR/$IMAGE_NAME" "$@"
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