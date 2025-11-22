# Troubleshooting Guide

## Error: "connect ECONNREFUSED 0.0.0.0:7860"

This error means the application cannot connect to the API server. Here's how to fix it:

### Step 1: Check if Container is Running

```powershell
docker ps
```

You should see a container named `face-detection-api` or similar. If not, the container isn't running.

### Step 2: Check Container Logs

```powershell
docker ps -a
docker logs <container_id>
```

Look for:
- Any error messages during startup
- "Application startup complete" message
- Port binding information

### Step 3: Verify Port Binding

Make sure you're running the container with port mapping:
```powershell
docker run -p 7860:7860 face-detection-api
```

### Step 4: Test from Browser

Open your browser and go to:
- http://localhost:7860/health
- http://localhost:7860/

If these work, the API is running correctly.

### Step 5: Check for Startup Errors

Common issues:
1. **Keras/TensorFlow errors** - Check logs for import errors
2. **Port already in use** - Another service might be using port 7860
3. **Container crashed** - Check `docker ps -a` for exited containers

### Step 6: Restart Container

If the container is not running:
```powershell
# Remove old container
docker ps -a
docker rm <container_id>

# Run again
docker run -p 7860:7860 face-detection-api
```

### Step 7: Check Port Availability

If port 7860 is in use:
```powershell
netstat -ano | findstr :7860
```

Use a different port:
```powershell
docker run -p 8080:7860 face-detection-api
# Then access at http://localhost:8080
```

