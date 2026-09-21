#!/bin/bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="gaudi-apptainer.sif"
DEF_FILE="$WORKSPACE_DIR/gaudi-apptainer.def"

show_help() {
    echo "Usage: ./gaudi-apptainer.sh [COMMAND] [ARGS]"
    echo ""
    echo "Commands:"
    echo "  build         Build the Apptainer image (.sif file)"
    echo "  run           Run an interactive shell inside the container"
    echo "  exec [cmd]    Execute a command inside the container"
    echo ""
    echo "Examples:"
    echo "  ./gaudi-apptainer.sh build"
    echo "  ./gaudi-apptainer.sh run"
    echo "  ./gaudi-apptainer.sh exec python src/train.py"
    echo ""
    echo "Environment variables:"
    echo "  HF_HOME         Path to Hugging Face cache (optional)"
    echo "  SSL_CERT_FILE   Path to SSL certificate bundle (optional)"
    echo "  EXTRA_MOUNTS    Space-separated list of extra bind mounts (optional)"
    echo "  PT_HPU_LAZY_MODE  Set to 1 to enable lazy mode (default: 0, eager mode)"
}

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
        echo "Building Gaudi Apptainer image '${IMAGE_NAME}'..."
        export APPTAINER_TMPDIR="${SCRATCH:-$HOME}/.apptainer/tmp"
        export APPTAINER_CACHEDIR="${SCRATCH:-$HOME}/.apptainer/cache"
        mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

        apptainer build --fakeroot "$WORKSPACE_DIR/$IMAGE_NAME" "$DEF_FILE"
        ;;
    run)
        if [ ! -f "$WORKSPACE_DIR/$IMAGE_NAME" ]; then
            echo "Error: Image '$IMAGE_NAME' not found. Run './gaudi-apptainer.sh build' first."
            exit 1
        fi
        echo "Starting an interactive shell in the Gaudi Apptainer container..."
        apptainer shell \
            $SHM_BIND \
            "${ENV_FLAGS[@]}" \
            "${BINDS[@]}" \
            --bind "$WORKSPACE_DIR:/workspace" \
            --pwd /workspace \
            "$WORKSPACE_DIR/$IMAGE_NAME"
        ;;
    exec)
        if [ $# -eq 0 ]; then
            echo "Error: No command provided to execute."
            exit 1
        fi
        if [ ! -f "$WORKSPACE_DIR/$IMAGE_NAME" ]; then
            echo "Error: Image '$IMAGE_NAME' not found. Run './gaudi-apptainer.sh build' first."
            exit 1
        fi
        echo "Executing command in container: $*"
        apptainer exec \
            $SHM_BIND \
            "${ENV_FLAGS[@]}" \
            "${BINDS[@]}" \
            --bind "$WORKSPACE_DIR:/workspace" \
            --pwd /workspace \
            "$WORKSPACE_DIR/$IMAGE_NAME" \
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
