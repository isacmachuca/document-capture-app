// Application logic
let documentCapture = null;

// State for multi-step capture
let currentStep = 'front'; // or 'back'
let frontResult = null;
let backResult = null;

// Cache DOM elements for better performance
const captureBtn = document.getElementById('captureBtn');
const statusMessage = document.getElementById('statusMessage');
const backBtn = document.getElementById('backBtn');
const closeBtn = document.getElementById('closeBtn');
const frontTab = document.getElementById('frontTab');
const backTab = document.getElementById('backTab');
const videoContainer = document.querySelector('.video-container');
const cameraVideo = document.getElementById('cameraVideo');
const documentFrame = document.querySelector('.document-frame');

// Modal elements
const confirmModal = document.getElementById('confirmModal');
const modalMessage = document.getElementById('modalMessage');
const modalOkBtn = document.getElementById('modalOkBtn');
const modalRetakeBtn = document.getElementById('modalRetakeBtn');
const detectionWarning = document.getElementById('detectionWarning');

// Confirm image element (created dynamically)
let confirmImage = null;

// Cleanup function for confirmImage
function cleanupConfirmImage() {
    if (confirmImage && confirmImage.parentNode) {
        confirmImage.parentNode.removeChild(confirmImage);
        confirmImage = null;
    }
}

// OpenCV.js loading
function onOpenCvReady() {
    initializeCapture();
}

// Initialize document capture
function initializeCapture() {
    documentCapture = new DocumentCapture({
        container: '#cameraVideo',
        onCapture: handleCapture,
        onError: handleError,
        onStatusChange: updateStatus,
        enableDocumentDetection: true,
        enhanceImage: true
    });
    // Automatically start the camera
    documentCapture.startCamera().then(success => {
        if (success) {
            captureBtn.disabled = false;
        }
    });
}

// Event handlers
function updateStatus(message, type) {
    statusMessage.textContent = message;
}

function handleError(error) {
    updateStatus(`Erro: ${error.message}`);
}

function updateStepUI() {
    if (currentStep === 'front') {
        updateStatus('Posicione a frente do seu documento dentro da marcação em uma superfície plana e fotografe.');
        frontTab.classList.add('active');
        backTab.classList.remove('active');
    } else {
        updateStatus('Posicione o verso do seu documento dentro da marcação em uma superfície plana e fotografe.');
        frontTab.classList.remove('active');
        backTab.classList.add('active');
    }
}

function showModal(message, onOk, onRetake, imageUrl, detectionSuccessful = true) {
    modalMessage.textContent = message;
    confirmModal.style.display = 'flex';
    
    // Set status for confirmation
    updateStatus('Confirme se os dados ficaram nítidos e os textos legíveis');
    
    // Hide camera video and show detected image in video container
    cameraVideo.style.display = 'none';
    if (!confirmImage) {
        confirmImage = document.createElement('div');
        confirmImage.id = 'confirmImage';
        confirmImage.style.width = '100%';
        confirmImage.style.height = '100%';
        confirmImage.style.backgroundSize = 'cover';
        confirmImage.style.backgroundPosition = 'center';
        confirmImage.style.backgroundRepeat = 'no-repeat';
        confirmImage.style.borderRadius = '10px';
        documentFrame.appendChild(confirmImage);
    }
    confirmImage.style.backgroundImage = `url(${imageUrl})`;
    confirmImage.style.display = 'block';
    
    // Show/hide detection warning based on detection success
    if (detectionSuccessful) {
        detectionWarning.style.display = 'none';
    } else {
        detectionWarning.style.display = 'flex';
    }
    
    // Remove previous listeners
    modalOkBtn.onclick = null;
    modalRetakeBtn.onclick = null;
    
    // Set new listeners
    modalOkBtn.onclick = () => {
        confirmModal.style.display = 'none';
        // Restore camera video, hide detected image
        cameraVideo.style.display = '';
        if (confirmImage) confirmImage.style.display = 'none';
        onOk();
    };
    modalRetakeBtn.onclick = () => {
        confirmModal.style.display = 'none';
        // Restore camera video, hide detected image
        cameraVideo.style.display = '';
        if (confirmImage) confirmImage.style.display = 'none';
        onRetake();
    };
}

function handleCapture(result) {
    // Hide capture button during confirmation
    captureBtn.style.display = 'none';
    
    // Check if document detection was successful
    const detectionSuccessful = result.metadata.hasDocumentDetection;
    
    // Show modal for confirmation, pass the PROCESSED image (not original)
    showModal(
        `A foto do documento ficou boa?`,
        () => { /* OK handler */
            if (currentStep === 'front') {
                frontResult = result;
                currentStep = 'back';
                updateStepUI();
                captureBtn.style.display = '';
            } else {
                backResult = result;
                updateStatus('Ambos os lados foram capturados com sucesso!');
                setTimeout(() => {
                    currentStep = 'front';
                    frontResult = null;
                    backResult = null;
                    updateStepUI();
                    captureBtn.style.display = '';
                }, 3000);
            }
        },
        () => { /* Retake handler */
            captureBtn.style.display = '';
            updateStepUI();
        },
        result.processedImage, // This is the detected/cropped document image
        detectionSuccessful // Pass detection status
    );
}

// Navigation event listeners
backBtn.addEventListener('click', () => {
    if (currentStep === 'back') {
        currentStep = 'front';
        updateStepUI();
    }
});

closeBtn.addEventListener('click', () => {
    // Handle close action - could reset or navigate away
    if (confirm('Deseja sair do processo de captura?')) {
        // Reset or close
        currentStep = 'front';
        frontResult = null;
        backResult = null;
        cleanupConfirmImage();
        updateStepUI();
    }
});

// Step tab event listeners (optional - for manual navigation)
frontTab.addEventListener('click', () => {
    if (currentStep !== 'front') {
        currentStep = 'front';
        updateStepUI();
    }
});

backTab.addEventListener('click', () => {
    if (currentStep !== 'back' && frontResult) {
        currentStep = 'back';
        updateStepUI();
    }
});

// Capture button event listener
captureBtn.addEventListener('click', async () => {
    captureBtn.disabled = true;
    await documentCapture.captureDocument();
    captureBtn.disabled = false;
});

// Initialize when OpenCV is ready
if (typeof cv !== 'undefined') {
    onOpenCvReady();
} else {
    let checkInterval = setInterval(() => {
        if (typeof cv !== 'undefined' && cv.Mat) {
            clearInterval(checkInterval);
            onOpenCvReady();
        }
    }, 100);
    
    setTimeout(() => {
        if (typeof cv === 'undefined') {
            clearInterval(checkInterval);
            updateStatus('⚠️ OpenCV.js falhou ao carregar - Usando detecção alternativa');
            initializeCapture();
        }
    }, 10000);
}

// On load, show step UI
updateStepUI();