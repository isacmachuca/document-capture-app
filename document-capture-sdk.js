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
                    // Enhanced mobile focus constraints
                    focusMode: 'continuous',
                    exposureMode: 'continuous',
                    whiteBalanceMode: 'continuous'
                }
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            this.isCapturing = true;
            
            // Add tap-to-focus for mobile devices
            this.setupTapToFocus();
            
            return true;
        } catch (error) {
            this.handleError('Failed to access camera: ' + error.message);
            return false;
        }
    }

    // Setup tap-to-focus functionality for mobile devices
    setupTapToFocus() {
        if (!this.video) return;
        
        this.video.addEventListener('click', async (event) => {
            const rect = this.video.getBoundingClientRect();
            const x = (event.clientX - rect.left) / rect.width;
            const y = (event.clientY - rect.top) / rect.height;
            
            try {
                const track = this.stream.getVideoTracks()[0];
                const capabilities = track.getCapabilities();
                
                if (capabilities.focusMode && capabilities.focusMode.includes('single-shot')) {
                    await track.applyConstraints({
                        advanced: [{
                            focusMode: 'single-shot',
                            pointsOfInterest: [{x: x, y: y}]
                        }]
                    });
                    
                    // Visual feedback
                    this.showFocusIndicator(event.clientX, event.clientY);
                }
                            } catch (error) {
                    // Tap-to-focus not supported on this device
                }
        });
    }

    // Show visual focus indicator
    showFocusIndicator(x, y) {
        const indicator = document.createElement('div');
        indicator.style.cssText = `
            position: absolute;
            left: ${x - 25}px;
            top: ${y - 25}px;
            width: 50px;
            height: 50px;
            border: 2px solid #fff;
            border-radius: 50%;
            pointer-events: none;
            animation: focusPulse 0.6s ease-out;
            z-index: 1000;
        `;
        
        // Add animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes focusPulse {
                0% { transform: scale(1.5); opacity: 1; }
                100% { transform: scale(1); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(indicator);
        setTimeout(() => {
            document.body.removeChild(indicator);
            document.head.removeChild(style);
        }, 600);
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

    // Robust document detection with OpenCV
    async detectDocumentWithOpenCV(imageData) {
        try {
            const { data, width, height } = imageData;
            
            // Convert ImageData to cv.Mat
            const src = new cv.Mat(height, width, cv.CV_8UC4);
            src.data.set(data);
            
            // Convert RGBA to RGB
            const srcRGB = new cv.Mat();
            cv.cvtColor(src, srcRGB, cv.COLOR_RGBA2RGB);

            // Multi-method detection approach
            const edgeResult = this.detectByEnhancedEdges(srcRGB);
            const colorResult = this.detectByColorSegmentation(srcRGB);
            const gradientResult = this.detectByGradients(srcRGB);
            
            // Combine results using confidence scoring
            const finalResult = this.combineDetectionResults([edgeResult, colorResult, gradientResult], width, height);
            
            let bounds = null;
            let perspectiveTransform = null;
            
            if (finalResult.documentContour) {
                const rect = cv.boundingRect(finalResult.documentContour);
                bounds = {
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: rect.height
                };
                
                perspectiveTransform = this.calculatePerspectiveTransform(finalResult.documentContour, width, height);
            }
            
            // Clean up
            src.delete();
            srcRGB.delete();
            this.cleanupDetectionResults([edgeResult, colorResult, gradientResult, finalResult]);
            
            return {
                bounds,
                perspectiveTransform,
                hasDocument: !!finalResult.documentContour,
                confidence: finalResult.confidence
            };
            
        } catch (error) {
            return this.detectDocumentSimple(imageData);
        }
    }
  
    // Method 1: Enhanced Edge Detection with Advanced Noise Suppression
    detectByEnhancedEdges(srcRGB) {
        const gray = new cv.Mat();
        cv.cvtColor(srcRGB, gray, cv.COLOR_RGB2GRAY);
        
        // Step 1: Adaptive Contrast Enhancement (CLAHE or fallback)
        const enhanced = new cv.Mat();
        try {
            const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
            clahe.apply(gray, enhanced);
            clahe.delete();
        } catch (error) {
            cv.equalizeHist(gray, enhanced);
        }
        
        // Step 2: Advanced Noise Reduction - Bilateral Filter
        const denoised = new cv.Mat();
        try {
            cv.bilateralFilter(enhanced, denoised, 9, 75, 75);
        } catch (error) {
            cv.GaussianBlur(enhanced, denoised, new cv.Size(5, 5), 0);
        }
        
        // Step 3: Multi-scale Edge Detection
        const edges1 = new cv.Mat();
        const edges2 = new cv.Mat();
        const edges3 = new cv.Mat();
        
        // Different scales to catch various edge types
        cv.Canny(denoised, edges1, 50, 150);  // Fine edges
        cv.Canny(denoised, edges2, 30, 100);  // Medium edges  
        cv.Canny(denoised, edges3, 80, 200);  // Strong edges only
        
        // Combine edge maps
        const combinedEdges = new cv.Mat();
        cv.bitwise_or(edges1, edges2, combinedEdges);
        cv.bitwise_or(combinedEdges, edges3, combinedEdges);
        
        // Step 4: Morphological noise suppression
        const kernel1 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
        const opened = new cv.Mat();
        cv.morphologyEx(combinedEdges, opened, cv.MORPH_OPEN, kernel1); // Remove small noise
        
        const kernel2 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
        const closed = new cv.Mat();
        cv.morphologyEx(opened, closed, cv.MORPH_CLOSE, kernel2); // Connect broken lines
        
        // Find contours
        const contours = new cv.MatVector();
        const hierarchy = new cv.Mat();
        cv.findContours(closed, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        
        // Clean up intermediate matrices
        gray.delete();
        enhanced.delete();
        denoised.delete();
        edges1.delete();
        edges2.delete();
        edges3.delete();
        combinedEdges.delete();
        opened.delete();
        closed.delete();
        kernel1.delete();
        kernel2.delete();
        hierarchy.delete();
        
        return {
            contours,
            confidence: 0.7
        };
    }

    // Method 2: Color-based Document Segmentation
    detectByColorSegmentation(srcRGB) {
        // Convert to different color spaces for better separation
        const hsv = new cv.Mat();
        const lab = new cv.Mat();
        cv.cvtColor(srcRGB, hsv, cv.COLOR_RGB2HSV);
        cv.cvtColor(srcRGB, lab, cv.COLOR_RGB2Lab);
        
        // Create mask for document-like colors (typically lighter)
        const hsvChannels = new cv.MatVector();
        const labChannels = new cv.MatVector();
        cv.split(hsv, hsvChannels);
        cv.split(lab, labChannels);
        
        // Use L channel from LAB (lightness) and V channel from HSV (brightness)
        const lChannel = labChannels.get(0);  // Lightness
        const vChannel = hsvChannels.get(2);  // Value/Brightness
        
        // Adaptive thresholding on lightness
        const lightMask = new cv.Mat();
        cv.adaptiveThreshold(lChannel, lightMask, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 5);
        
        // Create brightness mask
        const brightMask = new cv.Mat();
        cv.threshold(vChannel, brightMask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
        
        // Combine masks
        const combinedMask = new cv.Mat();
        cv.bitwise_and(lightMask, brightMask, combinedMask);
        
        // Morphological operations to clean up the mask
        const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(7, 7));
        const cleaned = new cv.Mat();
        cv.morphologyEx(combinedMask, cleaned, cv.MORPH_CLOSE, kernel);
        cv.morphologyEx(cleaned, cleaned, cv.MORPH_OPEN, kernel);
        
        // Find contours on the cleaned mask
        const contours = new cv.MatVector();
        const hierarchy = new cv.Mat();
        cv.findContours(cleaned, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        
        // Clean up
        hsv.delete();
        lab.delete();
        hsvChannels.delete();
        labChannels.delete();
        lChannel.delete();
        vChannel.delete();
        lightMask.delete();
        brightMask.delete();
        combinedMask.delete();
        cleaned.delete();
        kernel.delete();
        hierarchy.delete();
        
        return {
            contours,
            confidence: 0.6
        };
    }

    // Method 3: Gradient-based Detection
    detectByGradients(srcRGB) {
        const gray = new cv.Mat();
        cv.cvtColor(srcRGB, gray, cv.COLOR_RGB2GRAY);
        
        // Apply Gaussian blur to reduce noise
        const blurred = new cv.Mat();
        cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
        
        // Calculate gradients using Sobel operators
        const gradX = new cv.Mat();
        const gradY = new cv.Mat();
        cv.Sobel(blurred, gradX, cv.CV_16S, 1, 0, 3);
        cv.Sobel(blurred, gradY, cv.CV_16S, 0, 1, 3);
        
        // Convert to absolute values
        const absGradX = new cv.Mat();
        const absGradY = new cv.Mat();
        cv.convertScaleAbs(gradX, absGradX);
        cv.convertScaleAbs(gradY, absGradY);
        
        // Combine gradients
        const gradMagnitude = new cv.Mat();
        cv.addWeighted(absGradX, 0.5, absGradY, 0.5, 0, gradMagnitude);
        
        // Threshold gradient magnitude
        const gradThresh = new cv.Mat();
        cv.threshold(gradMagnitude, gradThresh, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
        
        // Morphological operations
        const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
        const gradClosed = new cv.Mat();
        cv.morphologyEx(gradThresh, gradClosed, cv.MORPH_CLOSE, kernel);
        
        // Find contours
        const contours = new cv.MatVector();
        const hierarchy = new cv.Mat();
        cv.findContours(gradClosed, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        
        // Clean up
        gray.delete();
        blurred.delete();
        gradX.delete();
        gradY.delete();
        absGradX.delete();
        absGradY.delete();
        gradMagnitude.delete();
        gradThresh.delete();
        gradClosed.delete();
        kernel.delete();
        hierarchy.delete();
        
        return {
            contours,
            confidence: 0.5
        };
    }

    // Combine detection results using confidence scoring
    combineDetectionResults(results, width, height) {
        let bestContour = null;
        let bestScore = 0;
        let allContours = new cv.MatVector();
        
        // Collect all contours from all methods
        for (const result of results) {
            for (let i = 0; i < result.contours.size(); i++) {
                const contour = result.contours.get(i);
                const score = this.scoreContour(contour, width, height) * result.confidence;
                allContours.push_back(contour.clone());
                
                if (score > bestScore) {
                    bestScore = score;
                    if (bestContour) bestContour.delete();
                    bestContour = contour.clone();
                }
            }
        }
        
        // Clean up allContours
        allContours.delete();
        
        return {
            documentContour: bestContour,
            confidence: bestScore
        };
    }

    // Score contour based on document-like properties
    scoreContour(contour, imageWidth, imageHeight) {
        const area = cv.contourArea(contour);
        const imageArea = imageWidth * imageHeight;
        const rect = cv.boundingRect(contour);
        
        // Must be reasonable size (10% to 90% of image)
        const areaRatio = area / imageArea;
        if (areaRatio < 0.1 || areaRatio > 0.9) return 0;
        
        // Check if near borders (documents shouldn't touch edges)
        const margin = Math.min(imageWidth, imageHeight) * 0.05;
        if (rect.x < margin || rect.y < margin || 
            rect.x + rect.width > imageWidth - margin ||
            rect.y + rect.height > imageHeight - margin) {
            return 0;
        }
        
        // Prefer rectangular shapes
        const epsilon = 0.02 * cv.arcLength(contour, true);
        const approx = new cv.Mat();
        cv.approxPolyDP(contour, approx, epsilon, true);
        
        let shapeScore = 0;
        if (approx.rows === 4) {
            shapeScore = 1.0; // Perfect quadrilateral
        } else if (approx.rows >= 4 && approx.rows <= 6) {
            shapeScore = 0.8; // Close to quadrilateral
        } else {
            shapeScore = 0.3; // Not very rectangular
        }
        
        // Aspect ratio score (documents are typically rectangular)
        const aspectRatio = rect.width / rect.height;
        const aspectScore = (aspectRatio > 0.5 && aspectRatio < 2.0) ? 1.0 : 0.5;
        
        approx.delete();
        
        return areaRatio * shapeScore * aspectScore;
    }

    // Clean up detection results
    cleanupDetectionResults(results) {
        for (const result of results) {
            if (result.contours) result.contours.delete();
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
                continue;
            }
            const area = cv.contourArea(contour);
            if (area < minArea) {
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
        }
        return largestQuad;
    }

    // Enhanced perspective transformation calculation for irregular contours
    calculatePerspectiveTransform(contour, imageWidth, imageHeight) {
        try {
            // First, try to get the best quadrilateral approximation
            let points = this.extractBestQuadrilateral(contour);
            
            // If we don't have exactly 4 points, create them from bounding rectangle
            if (!points || points.length !== 4) {
                const rect = cv.boundingRect(contour);
                points = [
                    { x: rect.x, y: rect.y },
                    { x: rect.x + rect.width, y: rect.y },
                    { x: rect.x + rect.width, y: rect.y + rect.height },
                    { x: rect.x, y: rect.y + rect.height }
                ];
            }
            
            // Validate points are reasonable
            if (!this.validateQuadrilateral(points, imageWidth, imageHeight)) {
                return null;
            }
            
            const orderedPoints = this.orderPoints(points);
            
            // Calculate target dimensions with aspect ratio preservation
            const dimensions = this.calculateTargetDimensions(orderedPoints);
            
            const srcPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
                orderedPoints[0].x, orderedPoints[0].y,
                orderedPoints[1].x, orderedPoints[1].y,
                orderedPoints[2].x, orderedPoints[2].y,
                orderedPoints[3].x, orderedPoints[3].y
            ]);
            
            const dstPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
                0, 0,
                dimensions.width - 1, 0,
                dimensions.width - 1, dimensions.height - 1,
                0, dimensions.height - 1
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
                outputWidth: Math.round(dimensions.width),
                outputHeight: Math.round(dimensions.height),
                sourcePoints: orderedPoints
            };
            
        } catch (error) {
            return null;
        }
    }

    // Extract best quadrilateral from irregular contour
    extractBestQuadrilateral(contour) {
        try {
            // Try different epsilon values to find the best quadrilateral approximation
            const epsilons = [0.01, 0.02, 0.03, 0.05, 0.08];
            
            for (const epsilon of epsilons) {
                const arcLength = cv.arcLength(contour, true);
                const approx = new cv.Mat();
                cv.approxPolyDP(contour, approx, epsilon * arcLength, true);
                
                if (approx.rows === 4) {
                    const points = [];
                    for (let i = 0; i < 4; i++) {
                        const point = approx.data32S.slice(i * 2, i * 2 + 2);
                        points.push({ x: point[0], y: point[1] });
                    }
                    approx.delete();
                    return points;
                }
                approx.delete();
            }
            
            // If no good quadrilateral found, use convex hull approach
            const hull = new cv.Mat();
            cv.convexHull(contour, hull);
            
            if (hull.rows >= 4) {
                // Take the 4 most extreme points from convex hull
                const hullPoints = [];
                for (let i = 0; i < hull.rows; i++) {
                    const point = hull.data32S.slice(i * 2, i * 2 + 2);
                    hullPoints.push({ x: point[0], y: point[1] });
                }
                hull.delete();
                
                // Select 4 corner points using distance-based selection
                return this.selectCornerPoints(hullPoints);
            }
            
            hull.delete();
            return null;
            
        } catch (error) {
            return null;
        }
    }

    // Select 4 corner points from a set of points
    selectCornerPoints(points) {
        if (points.length <= 4) return points;
        
        // Find centroid
        const cx = points.reduce((sum, p) => sum + p.x, 0) / points.length;
        const cy = points.reduce((sum, p) => sum + p.y, 0) / points.length;
        
        // Find points in 4 quadrants
        const quadrants = [[], [], [], []]; // TL, TR, BR, BL
        
        for (const point of points) {
            if (point.x <= cx && point.y <= cy) quadrants[0].push(point); // Top-left
            else if (point.x > cx && point.y <= cy) quadrants[1].push(point); // Top-right
            else if (point.x > cx && point.y > cy) quadrants[2].push(point); // Bottom-right
            else quadrants[3].push(point); // Bottom-left
        }
        
        const corners = [];
        for (let i = 0; i < 4; i++) {
            if (quadrants[i].length > 0) {
                // Find the most extreme point in each quadrant
                let extremePoint = quadrants[i][0];
                let maxDistance = this.distanceFromCenter(extremePoint, cx, cy);
                
                for (const point of quadrants[i]) {
                    const distance = this.distanceFromCenter(point, cx, cy);
                    if (distance > maxDistance) {
                        maxDistance = distance;
                        extremePoint = point;
                    }
                }
                corners.push(extremePoint);
            } else {
                // If no point in quadrant, use a reasonable default
                const defaultPoints = [
                    { x: cx - 50, y: cy - 50 }, // TL
                    { x: cx + 50, y: cy - 50 }, // TR
                    { x: cx + 50, y: cy + 50 }, // BR
                    { x: cx - 50, y: cy + 50 }  // BL
                ];
                corners.push(defaultPoints[i]);
            }
        }
        
        return corners;
    }

    // Validate quadrilateral points
    validateQuadrilateral(points, imageWidth, imageHeight) {
        if (!points || points.length !== 4) return false;
        
        // Check if all points are within image bounds
        for (const point of points) {
            if (point.x < 0 || point.y < 0 || point.x >= imageWidth || point.y >= imageHeight) {
                return false;
            }
        }
        
        // Check if the quadrilateral has reasonable area
        const area = this.calculatePolygonArea(points);
        const imageArea = imageWidth * imageHeight;
        
        return area > imageArea * 0.05 && area < imageArea * 0.95;
    }

    // Calculate polygon area using shoelace formula
    calculatePolygonArea(points) {
        let area = 0;
        const n = points.length;
        
        for (let i = 0; i < n; i++) {
            const j = (i + 1) % n;
            area += points[i].x * points[j].y;
            area -= points[j].x * points[i].y;
        }
        
        return Math.abs(area) / 2;
    }

    // Calculate target dimensions for perspective transform
    calculateTargetDimensions(orderedPoints) {
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
        
        // Ensure reasonable minimum dimensions
        const minDimension = 200;
        return {
            width: Math.max(maxWidth, minDimension),
            height: Math.max(maxHeight, minDimension)
        };
    }

    // Helper function for distance calculation
    distanceFromCenter(point, cx, cy) {
        return Math.sqrt(Math.pow(point.x - cx, 2) + Math.pow(point.y - cy, 2));
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

    // Enhanced cropAndEnhance method with robust irregular contour handling
    async cropAndEnhance(imageDataUrl, bounds, perspectiveTransform = null) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                let canvas, ctx;
                let success = false;
                
                // Method 1: Try perspective transformation if available and valid
                if (perspectiveTransform && typeof cv !== 'undefined' && perspectiveTransform.sourcePoints && perspectiveTransform.sourcePoints.length === 4) {
                    canvas = document.createElement('canvas');
                    ctx = canvas.getContext('2d');
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                    
                    const correctedCanvas = this.applyPerspectiveTransform(canvas, perspectiveTransform);
                    if (correctedCanvas && correctedCanvas.width > 0 && correctedCanvas.height > 0) {
                        canvas = correctedCanvas;
                        ctx = canvas.getContext('2d');
                        success = true;
                    }
                }
                
                // Method 2: Smart bounding box cropping with padding
                if (!success && bounds) {
                    canvas = this.performSmartCrop(img, bounds);
                    ctx = canvas.getContext('2d');
                    success = true;
                }
                
                // Method 3: Fallback - return original with enhancement
                if (!success) {
                    canvas = document.createElement('canvas');
                    ctx = canvas.getContext('2d');
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                }
                
                // Apply image enhancement
                if (this.options.enhanceImage) {
                    this.enhanceImage(ctx, canvas.width, canvas.height);
                }
                
                resolve(canvas.toDataURL('image/jpeg', 0.9));
            };
            
            img.onerror = () => {
                resolve(imageDataUrl); // Return original if loading fails
            };
            
            img.src = imageDataUrl;
        });
    }

    // Smart cropping method for irregular bounds
    performSmartCrop(img, bounds) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Add intelligent padding based on detected bounds
        const padding = Math.min(bounds.width, bounds.height) * 0.05; // 5% padding
        
        // Ensure bounds don't exceed image dimensions
        const cropX = Math.max(0, bounds.x - padding);
        const cropY = Math.max(0, bounds.y - padding);
        const cropWidth = Math.min(img.width - cropX, bounds.width + padding * 2);
        const cropHeight = Math.min(img.height - cropY, bounds.height + padding * 2);
        
        // Set canvas size to cropped dimensions
        canvas.width = cropWidth;
        canvas.height = cropHeight;
        
        // Draw the cropped region
        ctx.drawImage(
            img,
            cropX, cropY, cropWidth, cropHeight,  // Source rectangle
            0, 0, cropWidth, cropHeight           // Destination rectangle
        );
        
        return canvas;
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
