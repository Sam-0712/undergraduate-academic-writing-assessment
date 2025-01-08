document.getElementById('resetButton').addEventListener('click', function() {
    document.getElementById('title').value = '';
    document.getElementById('abstract').value = '';
    document.getElementById('keywords').value = '';
    document.getElementById('content').value = '';
    document.getElementById('date').value = '';
    document.getElementById('result').innerText = ''; // 清空结果框
});
