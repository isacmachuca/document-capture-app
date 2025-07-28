// Enhanced DocumentCapture class with OpenCV.js integration
class DocumentCapture {
    constructor(options = {}) {
        this.options = {
            container: options.container || '#cameraVideo',
            onCapture: options.onCapture || null,
            onError: options.onError || null,
            onStatusChange: options.onStatusChange || null,
            width: options.width || 1280,
            height: options.height || 720,
            enableDocumentDetection: options.enableDocumentDetection !== false,
            enhanceImage: options.enhanceImage !== true
        };

        this.stream = null;
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.isCapturing = false;
        
        this.init();
    }

    init() {
        this.video = document.querySelector(this.options.container);
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
        
        if (!this.video) {
            this.handleError('Video element not found');
            return;
        }
    }

    async startCamera() {
        try {
            this.updateStatus('Initializing camera...', 'info');
            
            const constraints = {
                video: {
                    width: { ideal: this.options.width },
                    height: { ideal: this.options.height },
                    facingMode: 'environment',
                    focusMode: 'continuous',
                    exposureMode: 'continuous',
                    whiteBalanceMode: 'continuous'
                }
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            this.isCapturing = true;
            
            return true;
        } catch (error) {
            this.handleError('Failed to access camera: ' + error.message);
            return false;
        }
    }

    async captureDocument() {
        if (!this.isCapturing) {
            this.handleError('Camera not started');
            return null;
        }

        try {
            this.updateStatus('Capturing document...', 'info');
            
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.ctx.drawImage(this.video, 0, 0);
            
            const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
            const originalImage = this.canvas.toDataURL('image/jpeg', 0.9);
            
            let processedImage = originalImage;
            let documentBounds = null;
            let perspectiveTransform = null;
            
            if (this.options.enableDocumentDetection) {
                const detectionResult = await this.detectDocument(imageData);
                if (detectionResult.bounds) {
                    documentBounds = detectionResult.bounds;
                    perspectiveTransform = detectionResult.perspectiveTransform;
                    processedImage = await this.cropAndEnhance(originalImage, documentBounds, perspectiveTransform);
                }
            }
            
            const result = {
                originalImage,
                processedImage,
                documentBounds,
                perspectiveTransform,
                timestamp: new Date().toISOString(),
                metadata: {
                    width: this.canvas.width,
                    height: this.canvas.height,
                    hasDocumentDetection: !!documentBounds,
                    hasPerspectiveCorrection: !!perspectiveTransform,
                    detectionMethod: (typeof cv !== 'undefined' && cv.Mat) ? 'opencv' : 'simple'
                }
            };
            
            this.updateStatus('Document captured successfully!', 'success');
            
            if (this.options.onCapture) {
                this.options.onCapture(result);
            }
            
            return result;
            
        } catch (error) {
            this.handleError('Failed to capture document: ' + error.message);
            return null;
        }
    }

    // Updated detectDocument method
    async detectDocument(imageData) {
        if (typeof cv !== 'undefined' && cv.Mat) {
            return this.detectDocumentWithOpenCV(imageData);
        }
        
        return this.detectDocumentSimple(imageData);
    }

    // Enhanced detectDocumentWithOpenCV method
    async detectDocumentWithOpenCV(imageData) {
        try {
            const { data, width, height } = imageData;
            
            // Convert ImageData to cv.Mat
            const src = new cv.Mat(height, width, cv.CV_8UC4);
            src.data.set(data);
            
            // Convert RGBA to RGB
            const srcRGB = new cv.Mat();
            cv.cvtColor(src, srcRGB, cv.COLOR_RGBA2RGB);
            
            // Convert to grayscale
            const gray = new cv.Mat();
            cv.cvtColor(srcRGB, gray, cv.COLOR_RGB2GRAY);
            // --- Preprocessing Enhancements ---
            // 1. Adaptive Thresholding (less aggressive for mobile)
            const adaptive = new cv.Mat();
            cv.adaptiveThreshold(gray, adaptive, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 2);
            // 2. Smaller Gaussian Blur (3x3 for mobile)
            const blurred = new cv.Mat();
            const ksize = new cv.Size(3, 3);
            cv.GaussianBlur(adaptive, blurred, ksize, 0);
            // Edge detection using Canny (adjusted for mobile)
            const edges = new cv.Mat();
            cv.Canny(blurred, edges, 30, 100);
            
            // Morphological operations (gentler for mobile)
            const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2, 2));
            const closedFinal = new cv.Mat();
            cv.morphologyEx(edges, closedFinal, cv.MORPH_CLOSE, kernel);
            
            // Find contours
            const contours = new cv.MatVector();
            const hierarchy = new cv.Mat();
            cv.findContours(closedFinal, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            
            // Find the largest quadrilateral contour
            const documentContour = this.findLargestQuadrilateral(contours, width, height);
            
            let bounds = null;
            let perspectiveTransform = null;
            
            if (documentContour) {
                const rect = cv.boundingRect(documentContour);
                bounds = {
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: rect.height
                };
                
                perspectiveTransform = this.calculatePerspectiveTransform(documentContour, width, height);
            }
            
            // Clean up
            src.delete();
            srcRGB.delete();
            gray.delete();
            adaptive.delete();
            blurred.delete();
            edges.delete();
            kernel.delete();
            closedFinal.delete();
            contours.delete();
            hierarchy.delete();
            if (documentContour) {
                documentContour.delete();
            }
            
            return {
                bounds,
                perspectiveTransform,
                hasDocument: !!documentContour
            };
            
        } catch (error) {
            console.error('OpenCV document detection error:', error);
            return this.detectDocumentSimple(imageData);
        }
    }

    // Helper method to find the largest quadrilateral contour
    findLargestQuadrilateral(contours, imageWidth, imageHeight) {
        const minArea = (imageWidth * imageHeight) * 0.1;
        let largestArea = 0;
        let largestQuad = null;
        const borderMargin = 10;
        for (let i = 0; i < contours.size(); i++) {
            const contour = contours.get(i);
            const rect = cv.boundingRect(contour);
            // Skip contours too close to the image border
            if (
                rect.x < borderMargin ||
                rect.y < borderMargin ||
                rect.x + rect.width > imageWidth - borderMargin ||
                rect.y + rect.height > imageHeight - borderMargin
            ) {
                contour.delete();
                continue;
            }
            const area = cv.contourArea(contour);
            if (area < minArea) {
                contour.delete();
                continue;
            }
            const epsilon = 0.02 * cv.arcLength(contour, true);
            const approx = new cv.Mat();
            cv.approxPolyDP(contour, approx, epsilon, true);
            if (approx.rows === 4 && area > largestArea) {
                largestArea = area;
                if (largestQuad) {
                    largestQuad.delete();
                }
                largestQuad = approx.clone();
            }
            approx.delete();
            contour.delete();
        }
        return largestQuad;
    }

    // Helper method to calculate perspective transformation
    calculatePerspectiveTransform(quadContour, imageWidth, imageHeight) {
        try {
            const points = [];
            for (let i = 0; i < 4; i++) {
                const point = quadContour.data32S.slice(i * 2, i * 2 + 2);
                points.push({ x: point[0], y: point[1] });
            }
            
            const orderedPoints = this.orderPoints(points);
            
            const widthA = Math.sqrt(
                Math.pow(orderedPoints[2].x - orderedPoints[3].x, 2) +
                Math.pow(orderedPoints[2].y - orderedPoints[3].y, 2)
            );
            const widthB = Math.sqrt(
                Math.pow(orderedPoints[1].x - orderedPoints[0].x, 2) +
                Math.pow(orderedPoints[1].y - orderedPoints[0].y, 2)
            );
            const maxWidth = Math.max(widthA, widthB);
            
            const heightA = Math.sqrt(
                Math.pow(orderedPoints[1].x - orderedPoints[2].x, 2) +
                Math.pow(orderedPoints[1].y - orderedPoints[2].y, 2)
            );
            const heightB = Math.sqrt(
                Math.pow(orderedPoints[0].x - orderedPoints[3].x, 2) +
                Math.pow(orderedPoints[0].y - orderedPoints[3].y, 2)
            );
            const maxHeight = Math.max(heightA, heightB);
            
            const srcPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
                orderedPoints[0].x, orderedPoints[0].y,
                orderedPoints[1].x, orderedPoints[1].y,
                orderedPoints[2].x, orderedPoints[2].y,
                orderedPoints[3].x, orderedPoints[3].y
            ]);
            
            const dstPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
                0, 0,
                maxWidth - 1, 0,
                maxWidth - 1, maxHeight - 1,
                0, maxHeight - 1
            ]);
            
            const transformMatrix = cv.getPerspectiveTransform(srcPoints, dstPoints);
            
            const transformArray = [];
            for (let i = 0; i < 9; i++) {
                transformArray.push(transformMatrix.data64F[i]);
            }
            
            srcPoints.delete();
            dstPoints.delete();
            transformMatrix.delete();
            
            return {
                matrix: transformArray,
                outputWidth: Math.round(maxWidth),
                outputHeight: Math.round(maxHeight),
                sourcePoints: orderedPoints
            };
            
        } catch (error) {
            console.error('Error calculating perspective transform:', error);
            return null;
        }
    }

    // Helper method to order points
    orderPoints(points) {
        const cx = points.reduce((sum, p) => sum + p.x, 0) / points.length;
        const cy = points.reduce((sum, p) => sum + p.y, 0) / points.length;
        
        const sortedPoints = points.map(point => ({
            ...point,
            angle: Math.atan2(point.y - cy, point.x - cx)
        })).sort((a, b) => a.angle - b.angle);
        
        let topLeftIndex = 0;
        let minSum = sortedPoints[0].x + sortedPoints[0].y;
        
        for (let i = 1; i < sortedPoints.length; i++) {
            const sum = sortedPoints[i].x + sortedPoints[i].y;
            if (sum < minSum) {
                minSum = sum;
                topLeftIndex = i;
            }
        }
        
        const orderedPoints = [];
        for (let i = 0; i < 4; i++) {
            const index = (topLeftIndex + i) % 4;
            orderedPoints.push({
                x: sortedPoints[index].x,
                y: sortedPoints[index].y
            });
        }
        
        return orderedPoints;
    }

    // Simple document detection fallback
    async detectDocumentSimple(imageData) {
        const { data, width, height } = imageData;
        const gray = new Uint8Array(width * height);
        
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            gray[i / 4] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
        }
        
        const edges = this.detectEdges(gray, width, height);
        const bounds = this.findDocumentBounds(edges, width, height);
        
        return { bounds, edges };
    }

    detectEdges(gray, width, height) {
        const edges = new Uint8Array(width * height);
        
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = y * width + x;
                
                const gx = 
                    -1 * gray[(y-1) * width + (x-1)] +
                    -2 * gray[y * width + (x-1)] +
                    -1 * gray[(y+1) * width + (x-1)] +
                    1 * gray[(y-1) * width + (x+1)] +
                    2 * gray[y * width + (x+1)] +
                    1 * gray[(y+1) * width + (x+1)];
                
                const gy = 
                    -1 * gray[(y-1) * width + (x-1)] +
                    -2 * gray[(y-1) * width + x] +
                    -1 * gray[(y-1) * width + (x+1)] +
                    1 * gray[(y+1) * width + (x-1)] +
                    2 * gray[(y+1) * width + x] +
                    1 * gray[(y+1) * width + (x+1)];
                
                edges[idx] = Math.min(255, Math.sqrt(gx * gx + gy * gy));
            }
        }
        
        return edges;
    }

    // Helper method to find document bounds
    findDocumentBounds(edges, width, height) {
        const threshold = 50;
        let minX = width, maxX = 0, minY = height, maxY = 0;
        let edgeCount = 0;
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                if (edges[y * width + x] > threshold) {
                    edgeCount++;
                    minX = Math.min(minX, x);
                    maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                }
            }
        }
        
        if (edgeCount > width * height * 0.01) {
            return {
                x: Math.max(0, minX - 10),
                y: Math.max(0, minY - 10),
                width: Math.min(width, maxX - minX + 20),
                height: Math.min(height, maxY - minY + 20)
            };
        }
        
        return null;
    }

    // Enhanced cropAndEnhance method
    async cropAndEnhance(imageDataUrl, bounds, perspectiveTransform = null) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                let canvas, ctx;
                
                if (perspectiveTransform && typeof cv !== 'undefined') {
                    canvas = document.createElement('canvas');
                    ctx = canvas.getContext('2d');
                    
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                    
                    const correctedCanvas = this.applyPerspectiveTransform(canvas, perspectiveTransform);
                    if (correctedCanvas) {
                        canvas = correctedCanvas;
                        ctx = canvas.getContext('2d');
                    } else {
                        // If perspective transform fails, fall back to simple cropping
                        const tempCanvas = document.createElement('canvas');
                        const tempCtx = tempCanvas.getContext('2d');
                        
                        tempCanvas.width = bounds.width;
                        tempCanvas.height = bounds.height;
                        
                        tempCtx.drawImage(
                            img,
                            bounds.x, bounds.y, bounds.width, bounds.height,
                            0, 0, bounds.width, bounds.height
                        );
                        
                        canvas = tempCanvas;
                        ctx = tempCtx;
                    }
                } else {
                    // Simple cropping without perspective correction
                    canvas = document.createElement('canvas');
                    ctx = canvas.getContext('2d');
                    
                    canvas.width = bounds.width;
                    canvas.height = bounds.height;
                    
                    ctx.drawImage(
                        img,
                        bounds.x, bounds.y, bounds.width, bounds.height,
                        0, 0, bounds.width, bounds.height
                    );
                }
                
                if (this.options.enhanceImage) {
                    this.enhanceImage(ctx, canvas.width, canvas.height);
                }
                
                resolve(canvas.toDataURL('image/jpeg', 0.9));
            };
            img.src = imageDataUrl;
        });
    }

    // Apply perspective transformation using OpenCV
    applyPerspectiveTransform(canvas, perspectiveTransform) {
        try {
            const ctx = canvas.getContext('2d');
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            
            const src = new cv.Mat(canvas.height, canvas.width, cv.CV_8UC4);
            src.data.set(imageData.data);
            
            const dst = new cv.Mat();
            const M = cv.matFromArray(3, 3, cv.CV_64FC1, perspectiveTransform.matrix);
            
            const dsize = new cv.Size(perspectiveTransform.outputWidth, perspectiveTransform.outputHeight);
            cv.warpPerspective(src, dst, M, dsize);
            
            const outputCanvas = document.createElement('canvas');
            outputCanvas.width = perspectiveTransform.outputWidth;
            outputCanvas.height = perspectiveTransform.outputHeight;
            
            cv.imshow(outputCanvas, dst);
            
            src.delete();
            dst.delete();
            M.delete();
            
            return outputCanvas;
            
        } catch (error) {
            console.error('Error applying perspective transform:', error);
            return null;
        }
    }

    // Helper method to enhance image
    enhanceImage(ctx, width, height) {
        const imageData = ctx.getImageData(0, 0, width, height);
        const data = imageData.data;
        
        const contrast = 1.2;
        const brightness = 10;
        
        for (let i = 0; i < data.length; i += 4) {
            data[i] = Math.min(255, Math.max(0, contrast * data[i] + brightness));
            data[i + 1] = Math.min(255, Math.max(0, contrast * data[i + 1] + brightness));
            data[i + 2] = Math.min(255, Math.max(0, contrast * data[i + 2] + brightness));
        }
        
        ctx.putImageData(imageData, 0, 0);
    }

    updateStatus(message, type) {
        if (this.options.onStatusChange) {
            this.options.onStatusChange(message, type);
        }
    }

    handleError(message) {
        console.error('[DocumentCapture]', message);
        if (this.options.onError) {
            this.options.onError(new Error(message));
        }
    }

    destroy() {
        this.video = null;
        this.canvas = null;
        this.ctx = null;
    }
}
