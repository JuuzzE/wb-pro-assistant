// Замените на URL вашего сервера на Render, когда будете готовы
// const API_URL = 'http://127.0.0.1:8000'; // Старая строка
const API_URL = 'https://wb-pro-assistant.onrender.com'; // Новая строка

const reviewInput = document.getElementById('reviewInput');
const generateBtn = document.getElementById('generateBtn');
const loader = document.getElementById('loader');
const resultArea = document.getElementById('resultArea');
const analysisOutput = document.getElementById('analysisOutput');
const responseOutput = document.getElementById('responseOutput');
const copyBtn = document.getElementById('copyBtn');

generateBtn.addEventListener('click', async () => {
    const reviewText = reviewInput.value.trim();
    if (!reviewText) {
        alert('Пожалуйста, вставьте текст отзыва.');
        return;
    }

    loader.style.display = 'block';
    generateBtn.style.display = 'none';
    resultArea.style.display = 'none';

    try {
        const response = await fetch(`${API_URL}/generate-response`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ review_text: reviewText })
        });
        if (!response.ok) throw new Error(`Ошибка сервера: ${response.statusText}`);

        const data = await response.json();

        // <<< ИЗМЕНЕНИЕ: Получаем и отображаем психотип >>>
        const sentiment = data.analysis?.sentiment || 'N/A';
        const psychotype = data.analysis?.psychotype || 'N/A';
        const request = data.analysis?.main_problem || 'Нет';
        
        analysisOutput.innerHTML = `<b>Тональность:</b> ${sentiment}<br><b>Психотип клиента:</b> ${psychotype}<br><b>Ключевая проблема:</b> ${request}`;
        // <<< КОНЕЦ ИЗМЕНЕНИЯ >>>

        responseOutput.value = data.response;
        
        resultArea.style.display = 'block';
    } catch (error) {
        console.error('Ошибка:', error);
        alert(`Произошла ошибка: ${error.message}`);
    } finally {
        loader.style.display = 'none';
        generateBtn.style.display = 'block';
    }
});

copyBtn.addEventListener('click', () => {
    responseOutput.select();
    document.execCommand('copy');
    alert('Ответ скопирован в буфер обмена!');
});