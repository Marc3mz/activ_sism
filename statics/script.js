document.getElementById('predictButton').addEventListener('click', async () => {
    // Obtener los valores de los campos de entrada
    const mag = parseFloat(document.getElementById('mag').value);
    const depth = parseFloat(document.getElementById('depth').value);
    const latitude = parseFloat(document.getElementById('latitude').value);
    const longitude = parseFloat(document.getElementById('longitude').value);

    // Validar que los valores sean números válidos
    if (isNaN(mag) || isNaN(depth) || isNaN(latitude) || isNaN(longitude)) {
        alert("Por favor, ingresa valores válidos.");
        return;
    }

    // Enviar los datos al backend
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                mag: mag,
                depth: depth,
                latitude: latitude,
                longitude: longitude,
            }),
        });

        // Verificar si la respuesta es exitosa
        if (!response.ok) {
            throw new Error('Error en la solicitud.');
        }

        // Obtener la respuesta del backend
        const result = await response.json();

        // Mostrar el resultado en la página
        document.getElementById('result').innerText = `
            Predicción: ${result.prediction}, 
            Probabilidad: ${result.probability.toFixed(2)}, 
            Densidad Sísmica: ${result.densidad_sismica.toFixed(2)}
        `;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'Error al realizar la predicción.';
    }
});