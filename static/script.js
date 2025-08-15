const form = document.getElementById('upload-form');
const results = document.getElementById('results');
const description = document.getElementById('description');
const processedImage = document.getElementById('processed-image');
const rawOutput = document.getElementById('raw-output');
const downloadLink = document.getElementById('download-link');
const status = document.getElementById('status');

// Imagen
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    results.classList.remove('hidden');
    description.textContent = 'Processing image, please wait...';
    processedImage.classList.add('hidden');
    downloadLink.classList.add('hidden');
    rawOutput.textContent = '';
    status.textContent = '';
    status.className = 'status-message';

    const formData = new FormData(form);

    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            description.textContent = data.description || 'No description was generated.';
            if (data.output_image) {
                const imageUrl = `/${data.output_image}?t=${new Date().getTime()}`;
                processedImage.src = imageUrl;
                processedImage.classList.remove('hidden');
                downloadLink.href = imageUrl;
                downloadLink.classList.remove('hidden');
            } else {
                processedImage.classList.add('hidden');
                downloadLink.classList.add('hidden');
            }
            rawOutput.textContent = JSON.stringify({
                detections: data.detections,
                segmentation: data.segmentation
            }, null, 2);
            status.textContent = 'Processing completed successfully.';
            status.className = 'status-message success';
        } else {
            description.textContent = `Error: ${data.error}`;
            processedImage.classList.add('hidden');
            downloadLink.classList.add('hidden');
            rawOutput.textContent = '';
            status.textContent = `Error: ${data.error}`;
            status.className = 'status-message error';
        }
    } catch (error) {
        description.textContent = `Network error: ${error.message}`;
        processedImage.classList.add('hidden');
        downloadLink.classList.add('hidden');
        rawOutput.textContent = '';
        status.textContent = `Network error: ${error.message}`;
        status.className = 'status-message error';
    }
});

// Chat IA
document.getElementById('chat-send').onclick = async function() {
    const inputEl = document.getElementById('chat-input');
    const log = document.getElementById('chat-log');
    const userMsg = inputEl.value.trim();
    if (!userMsg) return;
    log.innerHTML += `<div><b>Tú:</b> ${userMsg}</div>`;
    inputEl.value = '';
    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: userMsg})
        });
        const data = await res.json();
        if (data.reply) {
            log.innerHTML += `<div><b>Asistente:</b> ${data.reply}</div>`;
        } else {
            log.innerHTML += `<div style="color:red"><b>Error:</b> ${data.error}</div>`;
        }
    } catch (error) {
        log.innerHTML += `<div style="color:red"><b>Error de red:</b> ${error.message}</div>`;
    }
};