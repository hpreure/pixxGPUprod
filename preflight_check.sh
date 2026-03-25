#!/bin/bash
#
# VPS Worker Pre-Flight Check
# ============================
# Verifies all dependencies and configurations before starting worker
#

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          pixxEngine VPS Worker Pre-Flight Check                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

ERRORS=0
WARNINGS=0

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
PROJECT_ROOT="$(pwd)"

# 1. Check virtual environment
echo "1. Checking virtual environment..."
if [ -f "pixxEngine_venv/bin/activate" ]; then
    source pixxEngine_venv/bin/activate
    echo "   ✓ Virtual environment found"
else
    echo "   ✗ ERROR: Virtual environment not found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 2. Check .env file
echo "2. Checking configuration..."
if [ -f ".env" ]; then
    echo "   ✓ .env file found"
    
    # Check critical variables
    if grep -q "VPS_RABBITMQ_HOST" .env; then
        echo "   ✓ VPS_RABBITMQ_HOST configured"
    else
        echo "   ✗ ERROR: VPS_RABBITMQ_HOST not in .env"
        ERRORS=$((ERRORS + 1))
    fi
    
    if grep -q "POSTGRES_HOST" .env; then
        echo "   ✓ POSTGRES_HOST configured"
    else
        echo "   ✗ ERROR: POSTGRES_HOST not in .env"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "   ✗ ERROR: .env file not found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 3. Check GPU
echo "3. Checking GPU..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'   ✓ GPU available: {torch.cuda.get_device_name(0)}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   ✗ ERROR: GPU not available"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 4. Check Python packages
echo "4. Checking Python packages..."
REQUIRED_PACKAGES=("torch" "ultralytics" "pika" "psycopg2" "numpy" "cv2" "timm")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    python -c "import $pkg" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "   ✓ $pkg installed"
    else
        echo "   ✗ ERROR: $pkg not installed"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# 5. Check model weights
echo "5. Checking model weights..."
if [ -f "weights/yolo26l.engine" ] || [ -f "weights/yolo26l.pt" ]; then
    echo "   ✓ YOLO Person model found"
else
    echo "   ⚠ WARNING: YOLO Person model not found"
    WARNINGS=$((WARNINGS + 1))
fi

if [ -f "weights/YOLO26lBib.engine" ] || [ -f "weights/YOLO26lBib.pt" ]; then
    echo "   ✓ YOLO Bib model found"
else
    echo "   ⚠ WARNING: YOLO Bib model not found"
    WARNINGS=$((WARNINGS + 1))
fi

if [ -f "weights/YOLO26nBibText.engine" ] || [ -f "weights/YOLO26nBibText.pt" ]; then
    echo "   ✓ YOLO Text model found"
else
    echo "   ⚠ WARNING: YOLO Text model not found"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# 6. Check worker files
echo "6. Checking worker files..."
WORKER_FILES=(
    "src/workers/inference_engine.py"
    "src/workers/probe_calibration.py"
    "src/workers/asymmetric_gpu_worker.py"
    "src/workers/cpu_worker.py"
    "src/workers/db_scribe.py"
    "src/workers/image_feeder.py"
)
for file in "${WORKER_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file exists"
    else
        echo "   ✗ ERROR: $file not found"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# 7. Check database connectivity
echo "7. Checking database connectivity..."
source .env
PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Database connection successful"
    
    # Check participant_info table
    COUNT=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM participant_info;" 2>/dev/null | tr -d ' ')
    if [ -n "$COUNT" ]; then
        echo "   ✓ participant_info table accessible ($COUNT rows)"
    else
        echo "   ⚠ WARNING: participant_info table empty or not accessible"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "   ✗ ERROR: Database connection failed"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 8. Check VPS RabbitMQ connectivity
echo "8. Checking VPS RabbitMQ connectivity..."
timeout 3 bash -c "cat < /dev/null > /dev/tcp/$VPS_RABBITMQ_HOST/$VPS_RABBITMQ_PORT" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✓ VPS RabbitMQ reachable at $VPS_RABBITMQ_HOST:$VPS_RABBITMQ_PORT"
else
    echo "   ⚠ WARNING: Cannot reach VPS RabbitMQ at $VPS_RABBITMQ_HOST:$VPS_RABBITMQ_PORT"
    echo "   (This may be normal if Tailscale is not connected)"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# 9. Check logs directory
echo "9. Checking logs directory..."
if [ -d "logs" ]; then
    echo "   ✓ logs directory exists"
    if [ -w "logs" ]; then
        echo "   ✓ logs directory writable"
    else
        echo "   ✗ ERROR: logs directory not writable"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "   ⚠ WARNING: logs directory not found, creating..."
    mkdir -p logs
    if [ $? -eq 0 ]; then
        echo "   ✓ logs directory created"
    else
        echo "   ✗ ERROR: Failed to create logs directory"
        ERRORS=$((ERRORS + 1))
    fi
fi
echo ""

# 10. Check media sync
echo "10. Checking media directory..."
if [ -d "media/proxies" ]; then
    echo "   ✓ media/proxies directory exists"
else
    echo "   ⚠ WARNING: media/proxies directory not found"
    echo "   (Files from VPS may not be accessible)"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                        Pre-Flight Summary                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "   ✓✓✓ ALL CHECKS PASSED ✓✓✓"
    echo ""
    echo "   System is ready to start the VPS worker."
    echo ""
    echo "   To start:"
    echo "     ./src/workers/start_worker.sh           # Foreground"
    echo "     ./src/workers/start_worker.sh --daemon  # Background"
    echo "     sudo systemctl start pixxengine-vps-worker  # Systemd"
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo "   ✓ PASSED WITH WARNINGS"
    echo ""
    echo "   Errors: $ERRORS"
    echo "   Warnings: $WARNINGS"
    echo ""
    echo "   The system should work, but some features may be limited."
    echo "   Review warnings above."
    echo ""
    exit 0
else
    echo "   ✗ FAILED"
    echo ""
    echo "   Errors: $ERRORS"
    echo "   Warnings: $WARNINGS"
    echo ""
    echo "   Please fix errors before starting the worker."
    echo ""
    exit 1
fi
